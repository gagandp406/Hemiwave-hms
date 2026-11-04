"""CoRaL screen, track B — committee heads + vote-memory retrieval on frozen VIPEEGNet features.

Input: feats/fold{k}.npz from extract_vip_features.py. For fold k, the fold-k model's features
are used; heads are fitted on that model's training rows (fold != k) and predict fold k.
Heads start exactly at VIPEEGNet's own final layer (W, b) and are anchored to it.

Caveat: training-row features come from the model that was trained on those rows, so they
are cleaner than validation features. That biases *every* rung of the ladder the same way;
compare rungs with each other (and with A0/A) rather than reading absolute gains literally.
Track A (screen_logits.py) has no such bias.

Retrieval: for each validation row, the k nearest training rows (cosine, same fold model)
contribute their raw vote counts; the blend weight and k/tau/key-set are chosen
leave-one-fold-out, so no fold picks its own hyperparameters.

Usage:
  python screen_features.py --feats ~/hms/fair_comparison/coral_feats/feats \
      --archive ~/hms/results/vipeegnet_groupkfold_v1 --out ~/hms/fair_comparison/coral_screen
"""
from __future__ import annotations

import argparse
import itertools
import os
import time

import numpy as np
import torch

from coral_eval import add_control_deltas, evaluate, print_table, save_oof_like_archive, softmax
from coral_lib import (HQ_MIN_VOTES, HeadConfig, Stage, blend, fit_head, kld_rows, knn_topk,
                       knn_vote_share, predict, standard_configs, votes_to_share)

K_GRID = [8, 16, 32, 64, 128, 256]
TAU_GRID = [0.01, 0.03, 0.1, 0.3]
KEYS_GRID = ["all", "hq"]
LAM_GRID = [round(float(x), 2) for x in np.arange(0.05, 0.55, 0.05)]
MODE_GRID = ["arith", "geo"]


def load_folds(feat_dir):
    F = []
    for k in range(5):
        with np.load(os.path.join(feat_dir, f"fold{k}.npz"), allow_pickle=True) as z:
            F.append({key: z[key] for key in z.files})
    base = F[0]
    for k in range(1, 5):
        assert np.array_equal(F[k]["eeg_id"], base["eeg_id"]), "row order differs between fold files"
    return F


def archived_ref(archive, eeg, fold, fallback):
    ref = fallback.copy()
    if not archive:
        return ref, "VIP head (recomputed)"
    for k in range(5):
        p = os.path.join(archive, "oof", f"fold{k}_oof_all.npz")
        m = fold == k
        with np.load(p, allow_pickle=True) as z:
            if np.array_equal(np.asarray(z["eeg_ids"]).astype(np.int64), eeg[m].astype(np.int64)):
                ref[m] = z["yp"]
            else:
                print(f"fold {k}: archive order differs, using recomputed VIP head as reference")
    return ref, "VIPEEGNet (archive)"


def retrieval_lofo(P_model, Y, hq, fold, topk_cache, V):
    """Leave-one-fold-out choice of (k, tau, keys, lam, mode); returns blended P and choices."""
    combos = list(itertools.product(K_GRID, TAU_GRID, KEYS_GRID, LAM_GRID, MODE_GRID))
    S = np.full((5, len(combos)), np.inf)
    cand = {}
    for f in range(5):
        m = fold == f
        mh = hq[m]
        for (kk, tau, keys) in itertools.product(K_GRID, TAU_GRID, KEYS_GRID):
            sims, idx, Vk = topk_cache[(f, keys)]
            pk = knn_vote_share(sims, idx, Vk, kk, tau)
            cand[(f, kk, tau, keys)] = pk
        for ci, (kk, tau, keys, lam, mode) in enumerate(combos):
            Pb = blend(P_model[m][mh], cand[(f, kk, tau, keys)][mh], lam, mode)
            S[f, ci] = kld_rows(Y[m][mh], Pb).mean()
    P = P_model.copy()
    chosen = []
    for f in range(5):
        others = [j for j in range(5) if j != f]
        ci = int(np.argmin(S[others].mean(0)))
        kk, tau, keys, lam, mode = combos[ci]
        m = fold == f
        P[m] = blend(P_model[m], cand[(f, kk, tau, keys)], lam, mode)
        chosen.append(combos[ci])
    return P, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--archive", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Fd = load_folds(a.feats)
    eeg = Fd[0]["eeg_id"].astype(np.int64)
    V = Fd[0]["votes"].astype(np.int64)
    pat = Fd[0]["patient_id"].astype(np.int64)
    fold = Fd[0]["fold"].astype(np.int64)
    n = V.sum(1); hq = n >= HQ_MIN_VOTES
    Y = votes_to_share(V)
    print(f"rows {len(eeg)}, HQ {hq.sum()}, feat dim {Fd[0]['feat'].shape[1]}")

    P_vip = np.zeros((len(eeg), 6))
    for k in range(5):
        m = fold == k
        P_vip[m] = softmax(Fd[k]["logits"][m].astype(np.float64))
    ref, ref_name = archived_ref(a.archive, eeg, fold, P_vip)
    rows = [evaluate("VIPEEGNet (archive)" if ref_name.startswith("VIPEEGNet") else "VIP head", ref,
                     V, pat, fold, ref=ref),
            evaluate("E0_vip_head_recomputed", P_vip, V, pat, fold, ref=ref)]

    topk_cache = {}
    if not a.no_retrieval:
        kmax = max(K_GRID)
        for k in range(5):
            X = Fd[k]["feat"].astype(np.float32)
            te, tr = fold == k, fold != k
            for keys in KEYS_GRID:
                kmask = tr & (hq if keys == "hq" else np.ones_like(hq))
                sims, idx = knn_topk(X[te], X[kmask], kmax, device=a.device)
                topk_cache[(k, keys)] = (sims, idx, V[kmask])
        P, ch = retrieval_lofo(P_vip, Y, hq, fold, topk_cache, V)
        r = evaluate("R_vip_head+knn_votes", P, V, pat, fold, ref=ref); r["best_epochs"] = str(ch)
        rows.append(r)
        save_oof_like_archive(a.out, "feat_R_vip_head_knn", P, fold, V, eeg,
                              os.path.join(a.archive, "oof") if a.archive else None)
        print(f"retrieval on VIP head: {r['kld_hq_pooled']:.4f} ({r['delta_vs_VIPEEGNet']:+.4f}) {ch}")

    cfgs = [HeadConfig("A0_refit_hq_kl", n_raters=1, n_pools=1,
                       stages=[Stage("hq", "kl", 30, 1e-3)])] + standard_configs(lr=1e-3)
    if a.only:
        keep = tuple(s.strip() for s in a.only.split(","))
        cfgs = [c for c in cfgs if c.name.startswith(keep)]

    preds = {}
    for cfg in cfgs:
        t0 = time.time()
        P = np.zeros((len(eeg), 6)); best = []
        for k in range(5):
            X = Fd[k]["feat"].astype(np.float32)
            tr, te = fold != k, fold == k
            acc = 0
            for s in range(a.seeds):
                cfg.seed = s
                head, be = fit_head(cfg, X[tr], V[tr], pat[tr], init_W=Fd[k]["W"], init_b=Fd[k]["b"],
                                    device=a.device)
                acc = acc + predict(head, X[te], device=a.device)
                best.append(be)
            P[te] = acc / a.seeds
        r = evaluate(cfg.name, P, V, pat, fold, ref=ref); r["best_epochs"] = str(best[:5])
        rows.append(r); preds[cfg.name] = P
        save_oof_like_archive(a.out, f"feat_{cfg.name}", P, fold, V, eeg,
                              os.path.join(a.archive, "oof") if a.archive else None)
        msg = f"{cfg.name}: {r['kld_hq_pooled']:.4f} ({r['delta_vs_VIPEEGNet']:+.4f})"
        if topk_cache:
            Pr, ch = retrieval_lofo(P, Y, hq, fold, topk_cache, V)
            rr = evaluate(cfg.name + "+knn", Pr, V, pat, fold, ref=ref); rr["best_epochs"] = str(ch)
            rows.append(rr); preds[cfg.name + "+knn"] = Pr
            save_oof_like_archive(a.out, f"feat_{cfg.name}_knn", Pr, fold, V, eeg,
                                  os.path.join(a.archive, "oof") if a.archive else None)
            msg += f" | +knn {rr['kld_hq_pooled']:.4f} ({rr['delta_vs_VIPEEGNet']:+.4f})"
        print(msg + f"  ({time.time() - t0:.0f}s)")
    print()
    add_control_deltas(rows, preds, "A_softmax_kl_2stage", V, pat)
    print("delta_vs_control: paired vs A_softmax_kl_2stage (VIPEEGNet label handling, same frozen features)")
    print_table(rows, os.path.join(a.out, "screen_features_results.csv"))


if __name__ == "__main__":
    main()
