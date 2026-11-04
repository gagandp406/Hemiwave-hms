"""Shared data loading, evaluation and saving for the CoRaL screening scripts."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from coral_lib import (CLASSES, HQ_MIN_VOTES, VOTE_COLS, kld_rows, paired_patient_bootstrap,
                       votes_to_share)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def recover_votes(eeg_ids: np.ndarray, yt: np.ndarray, wt: Optional[np.ndarray],
                  train: pd.DataFrame) -> (np.ndarray, np.ndarray, Dict[str, int]):
    """Recover raw vote counts and patient ids for OOF rows from train.csv.

    A row is matched by eeg_id + normalised vote shares equal to yt. If several vote vectors
    of one eeg_id share the same proportions (e.g. 1 vs 3 seizure votes), the row weight wt
    is used when it looks like n/20; otherwise the candidate with the most votes is taken
    and the row is counted as ambiguous.
    """
    tv = train[["eeg_id", "patient_id"] + VOTE_COLS].copy()
    tv = tv.drop_duplicates(subset=["eeg_id"] + VOTE_COLS)
    by_eeg = {k: g for k, g in tv.groupby("eeg_id")}
    votes = np.zeros((len(eeg_ids), 6), dtype=np.int64)
    patient = np.zeros(len(eeg_ids), dtype=np.int64)
    stats = {"unique": 0, "by_weight": 0, "ambiguous": 0, "missing": 0}
    wt_like_n = wt is not None and np.allclose(np.round(np.asarray(wt) * 20), np.asarray(wt) * 20, atol=1e-3)
    for i, (e, y) in enumerate(zip(eeg_ids, yt)):
        g = by_eeg.get(int(e))
        if g is None:
            stats["missing"] += 1
            continue
        V = g[VOTE_COLS].to_numpy()
        S = V / V.sum(1, keepdims=True)
        ok = np.where(np.abs(S - y[None]).max(1) < 1e-4)[0]
        if len(ok) == 0:
            stats["missing"] += 1
            continue
        if len(ok) == 1:
            j = ok[0]; stats["unique"] += 1
        else:
            ns = V[ok].sum(1)
            if wt_like_n:
                j = ok[int(np.argmin(np.abs(ns - wt[i] * 20)))]; stats["by_weight"] += 1
            else:
                j = ok[int(np.argmax(ns))]; stats["ambiguous"] += 1
        votes[i] = V[j]
        patient[i] = int(g["patient_id"].iloc[0])
    return votes, patient, stats


def evaluate(name: str, P: np.ndarray, votes: np.ndarray, patient: np.ndarray, fold: np.ndarray,
             ref: Optional[np.ndarray] = None, ref_name: str = "VIPEEGNet") -> Dict:
    n = votes.sum(1)
    hq = n >= HQ_MIN_VOTES
    Y = votes_to_share(votes)
    L = kld_rows(Y, P)
    res = {
        "model": name,
        "kld_hq_pooled": float(L[hq].mean()),
        "kld_hq_fold_mean": float(np.mean([L[hq & (fold == k)].mean() for k in np.unique(fold)])),
        "kld_hq_per_fold": [round(float(L[hq & (fold == k)].mean()), 4) for k in np.unique(fold)],
        "kld_all_rows": float(L.mean()),
        "n_hq": int(hq.sum()),
    }
    if ref is not None:
        Lr = kld_rows(Y, ref)
        bs = paired_patient_bootstrap(Lr[hq], L[hq], patient[hq])
        res.update({f"delta_vs_{ref_name}": bs["delta"], "ci_lo": bs["lo"], "ci_hi": bs["hi"],
                    "p_better": bs["p_better"],
                    "folds_better": int(sum(L[hq & (fold == k)].mean() < Lr[hq & (fold == k)].mean()
                                            for k in np.unique(fold)))})
    return res


def save_oof_like_archive(out_dir: str, tag: str, P: np.ndarray, fold: np.ndarray, votes: np.ndarray,
                          eeg_ids: np.ndarray, archive_oof_dir: Optional[str] = None):
    """Write fold{k}_oof.npz (>=10 votes) and fold{k}_oof_all.npz in the VIPEEGNet schema.

    Non-prediction arrays (yt, wt, ...) are copied from the archive file of the same fold when
    available and when the row order matches, so load_oof / compare_models work unchanged.
    """
    d = os.path.join(out_dir, tag, "oof")
    os.makedirs(d, exist_ok=True)
    n = votes.sum(1)
    for k in np.unique(fold):
        for suffix, mask in (("_all", fold == k), ("", (fold == k) & (n >= HQ_MIN_VOTES))):
            arrs = {}
            src = os.path.join(archive_oof_dir, f"fold{k}_oof{suffix}.npz") if archive_oof_dir else None
            if src and os.path.exists(src):
                with np.load(src, allow_pickle=True) as z:
                    if len(z["eeg_ids"]) == mask.sum() and np.array_equal(np.asarray(z["eeg_ids"]),
                                                                          eeg_ids[mask]):
                        arrs = {key: z[key] for key in z.files}
            if not arrs:
                arrs = {"yt": votes_to_share(votes[mask]).astype(np.float32),
                        "eeg_ids": eeg_ids[mask], "votes": votes[mask]}
            arrs["yp"] = P[mask].astype(np.float32)
            arrs["yp_raw_logits"] = np.log(np.clip(P[mask], 1e-12, 1)).astype(np.float32)
            np.savez(os.path.join(d, f"fold{k}_oof{suffix}.npz"), **arrs)


def add_control_deltas(rows: List[Dict], preds: Dict[str, np.ndarray], control: str,
                       votes: np.ndarray, patient: np.ndarray):
    """Paired patient-bootstrap delta of every model vs a same-protocol control (e.g. rung A)."""
    if control not in preds:
        return
    hq = votes.sum(1) >= HQ_MIN_VOTES
    Y = votes_to_share(votes)
    Lc = kld_rows(Y, preds[control])[hq]
    for r in rows:
        P = preds.get(r["model"])
        if P is None or r["model"] == control:
            continue
        bs = paired_patient_bootstrap(Lc, kld_rows(Y, P)[hq], patient[hq], seed=1)
        r["delta_vs_control"] = bs["delta"]
        r["ctl_ci"] = f"[{bs['lo']:+.4f}, {bs['hi']:+.4f}]"


def print_table(rows: List[Dict], path: Optional[str] = None):
    df = pd.DataFrame(rows)
    cols = [c for c in ["model", "kld_hq_pooled", "kld_hq_fold_mean", "delta_vs_VIPEEGNet", "ci_lo",
                        "ci_hi", "p_better", "folds_better", "delta_vs_control", "ctl_ci",
                        "kld_all_rows", "kld_hq_per_fold",
                        "best_epochs"] if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(df[cols].to_string(index=False))
    if path:
        df.to_csv(path, index=False)
        with open(path.replace(".csv", ".json"), "w") as f:
            json.dump(rows, f, indent=1, default=float)
