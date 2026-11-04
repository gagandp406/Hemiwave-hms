"""CoRaL screen, track A — committee heads on VIPEEGNet's archived out-of-fold logits.

No retraining and no feature extraction: runs on the archive you already have.
For fold k, heads are fitted on the OOF logits of the other four folds (all 20,183-row
VIPEEGNet rows, both rater pools) and predict fold k. Every input row is out-of-fold for the
model that produced it, so there is no leak. Capacity is low (6-d input), so this track tests
the committee *output layer + likelihood* as a structured recalibration, against the
standard control (affine recalibration fitted on >=10-vote rows).

Usage:
  python screen_logits.py --archive ~/hms/results/vipeegnet_groupkfold_v1 \
      --train-csv /path/to/train.csv --out ~/hms/fair_comparison/coral_screen
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from coral_eval import (add_control_deltas, evaluate, print_table, recover_votes,
                        save_oof_like_archive, softmax)
from coral_lib import HQ_MIN_VOTES, HeadConfig, Stage, fit_head, predict, standard_configs


def load_archive(archive: str, train_csv: str):
    oof_dir = os.path.join(archive, "oof")
    train = pd.read_csv(train_csv)
    parts = []
    for k in range(5):
        p = os.path.join(oof_dir, f"fold{k}_oof_all.npz")
        with np.load(p, allow_pickle=True) as z:
            if k == 0:
                print("npz keys:", z.files)
            eeg = np.asarray(z["eeg_ids"]).astype(np.int64)
            yt = np.asarray(z["yt"], dtype=np.float64)
            yp = np.asarray(z["yp"], dtype=np.float64)
            wt = np.asarray(z["wt"], dtype=np.float64) if "wt" in z.files else None
            logits = (np.asarray(z["yp_raw_logits"], dtype=np.float64) if "yp_raw_logits" in z.files
                      else np.log(np.clip(yp, 1e-12, 1)))
            votes = None
            for key in ("votes", "y_votes", "vote_counts"):
                if key in z.files:
                    votes = np.asarray(z[key]).astype(np.int64)
        if votes is not None and "patient_id" in train:
            pmap = train.drop_duplicates("eeg_id").set_index("eeg_id")["patient_id"]
            patient = pmap.loc[eeg].to_numpy()
            stats = {"from_npz": len(eeg)}
        else:
            votes, patient, stats = recover_votes(eeg, yt, wt, train)
        print(f"fold {k}: rows={len(eeg)} hq={(votes.sum(1) >= HQ_MIN_VOTES).sum()} match={stats} "
              f"max|softmax(logits)-yp|={np.abs(softmax(logits) - yp).max():.2e}")
        parts.append(dict(eeg=eeg, votes=votes, patient=patient, logits=logits, yp=yp,
                          fold=np.full(len(eeg), k)))
    cat = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    ok = cat["votes"].sum(1) > 0
    if (~ok).any():
        print(f"WARNING: {int((~ok).sum())} rows without recovered votes are dropped")
        cat = {key: v[ok] for key, v in cat.items()}
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of config-name prefixes to run")
    ap.add_argument("--seeds", type=int, default=3, help="seeds averaged per config (prob. mean)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    D = load_archive(a.archive, a.train_csv)
    X, V, fold, pat, ref = D["logits"], D["votes"], D["fold"], D["patient"], D["yp"]
    rows = [evaluate("VIPEEGNet (archive)", ref, V, pat, fold, ref=ref)]

    cfgs = [HeadConfig("A0_affine_hq_kl", n_raters=1, n_pools=1,
                       stages=[Stage("hq", "kl", 40, 3e-3)])] + standard_configs()
    if a.only:
        keep = tuple(s.strip() for s in a.only.split(","))
        cfgs = [c for c in cfgs if c.name.startswith(keep)]

    eye, zero = np.eye(6), np.zeros(6)
    preds = {}
    for cfg in cfgs:
        t0 = time.time()
        P = np.zeros_like(ref)
        best = []
        for k in range(5):
            tr, te = fold != k, fold == k
            acc = 0
            for s in range(a.seeds):
                cfg.seed = s
                head, be = fit_head(cfg, X[tr], V[tr], pat[tr], init_W=eye, init_b=zero, device=a.device)
                acc = acc + predict(head, X[te], device=a.device)
                best.append(be)
            P[te] = acc / a.seeds
        r = evaluate(cfg.name, P, V, pat, fold, ref=ref)
        r["best_epochs"] = str(best[:5])
        rows.append(r)
        preds[cfg.name] = P
        save_oof_like_archive(a.out, f"logits_{cfg.name}", P, fold, V, D["eeg"],
                              archive_oof_dir=os.path.join(a.archive, "oof"))
        print(f"{cfg.name}: HQ KLD {r['kld_hq_pooled']:.4f}  delta {r['delta_vs_VIPEEGNet']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  ({time.time() - t0:.0f}s)")
    print()
    add_control_deltas(rows, preds, "A0_affine_hq_kl", V, pat)
    print("delta_vs_control: paired vs A0_affine_hq_kl (standard recalibration on >=10-vote rows)")
    print_table(rows, os.path.join(a.out, "screen_logits_results.csv"))


if __name__ == "__main__":
    main()
