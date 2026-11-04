"""Dump VIPEEGNet penultimate features for every one of the 20,183 rows, per fold model.

Run in the VIPEEGNet environment (~/hms/.venv, TF 2.19 / Keras 3). For fold k the fold-k
model (final stage-2 weights) embeds all rows: rows of fold k are out-of-fold, the rest are
that model's own training rows. Output per fold: feats/fold{k}.npz with
  feat (N, D) float16   penultimate features (input of the final Dense layer)
  W (D, 6), b (6)       the final Dense layer
  logits (N, 6)         feat @ W + b
  eeg_id, patient_id, fold, votes (N, 6)

The script only needs three functions from your pipeline, supplied by an adapter module
(see adapter_template.py): load_rows(), build_model(fold), make_batches(df).

Sanity check printed per fold: softmax(logits) on the fold's own rows vs the archived OOF
predictions (max abs diff should be ~1e-3 or less; mixed precision adds a little noise).

Usage:
  python extract_vip_features.py --adapter my_adapter.py --out ~/hms/fair_comparison/coral_feats \
      --archive ~/hms/results/vipeegnet_groupkfold_v1 [--folds 0,1,2,3,4]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

VOTE_COLS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]


def load_adapter(path: str):
    spec = importlib.util.spec_from_file_location("coral_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["coral_adapter"] = mod
    spec.loader.exec_module(mod)
    for fn in ("load_rows", "build_model", "make_batches"):
        if not hasattr(mod, fn):
            raise SystemExit(f"adapter is missing {fn}()")
    return mod


def feature_model(model):
    import keras
    dense = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
    if not dense:
        raise SystemExit("no Dense layer found at the top level of the model; pass the inner "
                         "functional model from build_model() instead")
    head = dense[-1]
    W, b = [np.asarray(w, dtype=np.float32) for w in head.get_weights()]
    if W.shape[1] != 6:
        raise SystemExit(f"last Dense has output {W.shape[1]}, expected 6")
    try:
        fm = keras.Model(model.inputs, head.input)
    except Exception as e:  # subclassed models have no symbolic graph
        raise SystemExit(f"could not build a feature model ({e}). Build VIPEEGNet with the "
                         "functional API in build_model(), or expose the pooled features.")
    print(f"head layer '{head.name}': W {W.shape}, activation={getattr(head, 'activation', None)}")
    return fm, W, b


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive", default="")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    ad = load_adapter(a.adapter)
    os.makedirs(os.path.join(a.out, "feats"), exist_ok=True)

    df = ad.load_rows().reset_index(drop=True)
    need = ["eeg_id", "patient_id", "fold"] + VOTE_COLS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"load_rows() is missing columns {miss}")
    print(f"rows: {len(df)} (expected 20,183), HQ rows: {(df[VOTE_COLS].sum(1) >= 10).sum()} (expected 6,350)")

    for k in [int(x) for x in a.folds.split(",")]:
        t0 = time.time()
        model = ad.build_model(k)
        fm, W, b = feature_model(model)
        feats = []
        for xb in ad.make_batches(df):
            f = fm.predict_on_batch(xb) if hasattr(fm, "predict_on_batch") else fm(xb, training=False)
            feats.append(np.asarray(f, dtype=np.float32).reshape(len(f), -1))
        F = np.concatenate(feats)
        assert len(F) == len(df), f"make_batches yielded {len(F)} rows, expected {len(df)}"
        logits = F @ W + b
        out = os.path.join(a.out, "feats", f"fold{k}.npz")
        np.savez(out, feat=F.astype(np.float16), W=W, b=b, logits=logits.astype(np.float32),
                 eeg_id=df["eeg_id"].to_numpy(), patient_id=df["patient_id"].to_numpy(),
                 fold=df["fold"].to_numpy(), votes=df[VOTE_COLS].to_numpy().astype(np.int16))
        msg = ""
        arch = os.path.join(a.archive, "oof", f"fold{k}_oof_all.npz") if a.archive else ""
        if arch and os.path.exists(arch):
            with np.load(arch, allow_pickle=True) as z:
                m = df["fold"].to_numpy() == k
                if np.array_equal(np.asarray(z["eeg_ids"]).astype(np.int64),
                                  df.loc[m, "eeg_id"].to_numpy().astype(np.int64)):
                    msg = f" | max|p - archived yp| = {np.abs(softmax(logits[m]) - z['yp']).max():.2e}"
                else:
                    msg = " | archive row order differs; matching skipped"
        print(f"fold {k}: feat {F.shape} in {time.time() - t0:.0f}s -> {out}{msg}")
        del model, fm
        try:
            import keras
            keras.backend.clear_session()
        except Exception:
            pass


if __name__ == "__main__":
    main()
