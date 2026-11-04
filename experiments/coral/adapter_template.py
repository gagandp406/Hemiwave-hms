"""Adapter between extract_vip_features.py and your archived VIPEEGNet code.

Copy to my_adapter.py and fill in the three functions using the code in
~/hms/results/vipeegnet_groupkfold_v1/code/ (the notebook that produced the archive).
Everything must match the archived run exactly: same 20,183-row table and order, same
preprocessing, no augmentation, no shuffling, center label window per row.
"""
import pandas as pd

ARCHIVE = "/home/nfs/jhb-dehzangi/ws04/gs1147/hms/results/vipeegnet_groupkfold_v1"


def load_rows() -> pd.DataFrame:
    """The deduplicated 20,183-row VIPEEGNet table, in the archive's row order.

    Required columns: eeg_id, patient_id, fold (VIPEEGNet's own GroupKFold, see
    folds/patient_folds_groupkfold.csv), seizure_vote, lpd_vote, gpd_vote, lrda_vote,
    grda_vote, other_vote, plus whatever make_batches() needs (e.g. label offset).
    Easiest: call the same function the notebook uses to build its train dataframe, then
    merge the fold column. Order rows so that, within each fold, they match the order of
    oof/fold{k}_oof_all.npz['eeg_ids'] (the extractor checks this).
    """
    raise NotImplementedError


def build_model(fold: int):
    """Return the Keras (functional) VIPEEGNet with weights/fold{fold}_stage2.weights.h5 loaded.

    e.g.  model = build_vipeegnet()          # the notebook's model builder
          model.load_weights(f"{ARCHIVE}/weights/fold{fold}_stage2.weights.h5")
          return model
    """
    raise NotImplementedError


def make_batches(df: pd.DataFrame):
    """Yield model-ready input batches for df's rows, in df order (validation preprocessing).

    e.g.  gen = DataGenerator(df, mode="valid", batch_size=64, shuffle=False)
          for i in range(len(gen)):
              x, _ = gen[i]
              yield x
    """
    raise NotImplementedError
