"""Adapter for extract_vip_features.py — written against VIPEEGNet_local.ipynb (archived run).

Reproduces the notebook's *validation* path exactly:
  * rows: train.csv deduplicated on (eeg_id + 6 vote columns) -> 20,183 rows, notebook order
  * folds: the archive's patient->fold mapping (VIPEEGNet's own GroupKFold(5))
  * window: for each row, the middle segment by eeg_sub_id among train.csv rows with the same
    eeg_id + exact votes (mode='valid'), 50 s from eeg_label_offset_seconds, no jitter
  * signal: eeg_cache_v2/{eeg_id}.npy (18-ch bipolar, clip, 0.5-45 Hz filtfilt) -> first 8 +
    last 8 channels -> the notebook's channel reorder -> pad -> clip -> (x+1024)/2048*255
  * model: the notebook's build_model (EfficientNetV2B3 + SumToOne Conv1D), mixed_float16,
    weights/fold{k}_stage2.weights.h5

Paths can be overridden with env vars HMS, TRAIN_CSV, VIP_ARCHIVE, EEG_CACHE.
"""
import os

import numpy as np
import pandas as pd

HMS = os.path.expanduser(os.environ.get("HMS", "~/hms"))
TRAIN_CSV = os.environ.get("TRAIN_CSV") or f"{HMS}/hms-data/train.csv"
ARCHIVE = os.environ.get("VIP_ARCHIVE") or f"{HMS}/results/vipeegnet_groupkfold_v1"
CACHE_DIR = os.environ.get("EEG_CACHE") or f"{HMS}/eeg_cache_v2"

TARGET_COLS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
RSFREQ = 200
EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
N_T = round(EEG_LENGTH_USED * RSFREQ)          # 10,000 samples
BATCH = 64

_X_CACHE = {}                                  # built once, reused by all 5 folds


# --------------------------------------------------------------------------- rows
def _fold_map() -> pd.DataFrame:
    for p in (f"{ARCHIVE}/folds/patient_folds_groupkfold.csv",
              f"{HMS}/fair_comparison/patient_folds_groupkfold.csv"):
        if os.path.isfile(p):
            print(f"folds from {p}")
            return pd.read_csv(p)[["patient_id", "fold"]]
    raise SystemExit("patient_folds_groupkfold.csv not found in the archive or fair_comparison/")


def load_rows() -> pd.DataFrame:
    df_full = pd.read_csv(TRAIN_CSV)
    train = df_full.drop_duplicates(["eeg_id"] + TARGET_COLS).reset_index(drop=True)
    pf = _fold_map()
    train = train.merge(pf, on="patient_id", how="left")           # left merge keeps row order
    assert train["fold"].notna().all(), "some patients have no fold"
    train["fold"] = train["fold"].astype(int)

    # validation window per row: middle segment by eeg_sub_id (notebook _resolve_fast, mode='valid')
    cols = ["eeg_id"] + TARGET_COLS
    lookup = {}
    for key, g in df_full[cols + ["eeg_sub_id", "eeg_label_offset_seconds"]].groupby(cols, sort=False):
        sub, off = g["eeg_sub_id"].to_numpy(), g["eeg_label_offset_seconds"].to_numpy()
        lookup[tuple(int(k) for k in key)] = off[np.argsort(sub, kind="stable")[len(off) // 2]]
    keys = zip(train["eeg_id"].astype(int), *[train[c].astype(int) for c in TARGET_COLS])
    train["val_offset"] = [lookup[tuple(k)] for k in keys]
    print(f"rows {len(train):,} | HQ {(train[TARGET_COLS].sum(1) >= 10).sum():,} | "
          f"folds {train['fold'].value_counts().sort_index().tolist()}")
    return train


# --------------------------------------------------------------------------- inputs
def _one(eeg_full: np.ndarray, r_eeg: float) -> np.ndarray:
    eeg = eeg_full[:, round(r_eeg * RSFREQ):round((r_eeg + 50) * RSFREQ)]
    eeg = np.concatenate((eeg[0:EEG_CHANNEL_USED // 2, :], eeg[-EEG_CHANNEL_USED // 2:, :]), axis=0)
    eeg = eeg[:, round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2):
                 round((EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2)]
    if eeg.shape[1] < N_T:
        eeg = np.pad(eeg, ((0, 0), (0, N_T - eeg.shape[1])))
    eeg2 = eeg.copy()
    eeg[4:8, :] = eeg2[12:16, :]
    eeg[8:12, :] = eeg2[4:8, :]
    eeg[12:16, :] = eeg2[8:12, :]
    eeg = np.clip(eeg, -1024, 1024)
    return ((eeg + 1024) / 2048 * 255).astype(np.float32)


def _build_inputs(df: pd.DataFrame) -> np.ndarray:
    key = (len(df), int(df["eeg_id"].iloc[0]), int(df["eeg_id"].iloc[-1]))
    if key in _X_CACHE:
        return _X_CACHE[key]
    X = np.zeros((len(df), EEG_CHANNEL_USED, N_T), dtype=np.float32)   # ~12.9 GB for 20,183 rows
    cur_id, cur = None, None
    order = np.argsort(df["eeg_id"].to_numpy(), kind="stable")          # read each .npy once
    eids, offs = df["eeg_id"].to_numpy(), df["val_offset"].to_numpy()
    for n, i in enumerate(order):
        if eids[i] != cur_id:
            cur_id = eids[i]
            cur = np.load(os.path.join(CACHE_DIR, f"{int(cur_id)}.npy"))
        X[i] = _one(cur, float(offs[i]))
        if n % 2000 == 0:
            print(f"  inputs {n:,}/{len(df):,}", flush=True)
    _X_CACHE[key] = X
    return X


def make_batches(df: pd.DataFrame):
    X = _build_inputs(df)
    for i in range(0, len(X), BATCH):
        yield X[i:i + BATCH]


# --------------------------------------------------------------------------- model
def build_model(fold: int):
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.keras.backend.clear_session()             # fresh layer names, as in the notebook's eval cell

    class IniToOne(tf.keras.initializers.Initializer):
        def __call__(self, shape, dtype=None):
            filter_length, input_channel, filter_count = shape
            kernel = np.zeros(shape, dtype=np.float32)
            for i in range(filter_count):
                kernel[i % filter_length, 0, i] = 1.0
            return tf.convert_to_tensor(kernel, dtype=dtype)

        def get_config(self):
            return {}

    class SumToOne(tf.keras.constraints.Constraint):
        def __call__(self, w):
            w = tf.abs(w)
            return w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)

        def get_config(self):
            return {}

    strides = 10
    inp_eeg = tf.keras.Input(shape=(EEG_CHANNEL_USED, N_T), name="eeg")
    x_raw = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(inp_eeg)
    embed = tf.keras.layers.Conv1D(filters=strides * 3, kernel_size=strides, strides=strides,
                                   padding="same", use_bias=False, activation=None,
                                   kernel_initializer=IniToOne(), kernel_constraint=SumToOne(),
                                   input_shape=(None, 1))
    x = tf.keras.layers.TimeDistributed(embed)(x_raw)
    x = tf.keras.layers.Concatenate(axis=-1)([
        tf.keras.layers.Reshape((x.shape[1], x.shape[2], -1, 1))(x[:, :, :, 0 * strides:1 * strides]),
        tf.keras.layers.Reshape((x.shape[1], x.shape[2], -1, 1))(x[:, :, :, 1 * strides:2 * strides]),
        tf.keras.layers.Reshape((x.shape[1], x.shape[2], -1, 1))(x[:, :, :, 2 * strides:3 * strides]),
    ])
    x = tf.keras.layers.Permute([4, 2, 1, 3])(x)
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[2], -1))(x)
    x = tf.keras.layers.Permute((3, 2, 1))(x)
    base = tf.keras.applications.EfficientNetV2B3(include_top=False, weights=None,
                                                  include_preprocessing=True)
    base._name = "eeg_extractor"
    x = base(x)
    W = x.shape[2]
    cs, ce = ((W - 1) // 2, W // 2 + 1) if W is not None else (15, 17)
    x = x[:, :, cs:ce, :]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    y = tf.keras.layers.Dense(len(TARGET_COLS), activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(inputs=inp_eeg, outputs=y)

    wpath = f"{ARCHIVE}/weights/fold{fold}_stage2.weights.h5"
    model.load_weights(wpath)
    print(f"fold {fold}: loaded {wpath} ({model.count_params():,} params, expected 12,940,144)")
    return model
