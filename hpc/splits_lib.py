"""Shared by every Vanda script: class names, LoRA prompts, and which photos go where.

Splits (made once by make_splits.py, saved in hpc/splits/te_splits.csv):
  * 5 folds, photos grouped by 10-minute capture session (from the file name's time stamp),
    so near-identical photos shot seconds apart never sit on both sides of a split;
  * each fold k:  test = fold k,  validation = fold k+1 (for picking the best epoch),
                  train = the other three folds;
  * few-shot mode (the AITK situation): from the training folds, keep at most N photos per
    defect class and at most OK_CAP good photos, always the same ones (by a fixed random rank),
    so the detector and the generator see exactly the same real photos.
"""
import os

import pandas as pd

CLASSES = ["OK", "Misalign", "NoSolder", "LowSolder", "Single"]
DEFECTS = [c for c in CLASSES if c != "OK"]

# One prompt per class for the per-class LoRAs (the previous team's v6 used one LoRA per class
# with a descriptive prompt). "tejoint" is a rare word the LoRA learns to tie to TE's photos.
PROMPTS = {
    "OK": "a close-up photo of a tejoint welded pin joint, good weld, pins aligned and fully soldered",
    "Misalign": "a close-up photo of a tejoint welded pin joint, pins misaligned, all pins soldered",
    "NoSolder": "a close-up photo of a tejoint welded pin joint, no solder on the pins",
    "LowSolder": "a close-up photo of a tejoint welded pin joint, not enough solder on the pins",
    "Single": "a close-up photo of a tejoint welded pin joint, single electrode defect, one weld missing",
}

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPLITS = os.path.join(HERE, "splits", "te_splits.csv")


def load_splits(path=DEFAULT_SPLITS):
    df = pd.read_csv(path)
    missing = df[~df.path.map(os.path.exists)]
    if len(missing):
        raise SystemExit(f"{len(missing)} photos in {path} are not on disk (first: {missing.path.iloc[0]}). "
                         "Re-run make_splits.py on this machine.")
    return df


def fold_parts(df, k, n_folds=5):
    """train / validation / test rows for fold k."""
    val_k = (k + 1) % n_folds
    test = df[df.fold == k]
    val = df[df.fold == val_k]
    train = df[~df.fold.isin([k, val_k])]
    return train, val, test


def fewshot(train, n_defect=10, ok_cap=100):
    """The same small set every time: lowest fixed rank first. n_defect=0 means keep everything."""
    if not n_defect:
        return train
    parts = []
    for c, g in train.groupby("label"):
        parts.append(g.sort_values("rank").head(ok_cap if c == "OK" else n_defect))
    return pd.concat(parts)
