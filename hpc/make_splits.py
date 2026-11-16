"""Make the 5-fold, session-grouped splits of TE's photos (run once, on the machine that will train).

  python hpc/make_splits.py            (reads te/te_outputs/te_pool.csv from te_build_pool.py)

Why grouped: TE's photos were taken in bursts, and each class was mostly shot in its own
sessions (e.g. ~300 of the 359 'Single' photos on one afternoon). A random split puts
near-identical burst shots in both train and test and inflates accuracy. Grouping by
10-minute capture session keeps each burst on one side while still giving every fold
some photos of every class (hourly or daily groups leave whole classes out of some folds).
Of 1000 shuffles it keeps the one where the worst-off fold still gets a fair share of the rare
classes (OK, Misalign, NoSolder, LowSolder). The result depends on the scikit-learn version
(1.5.2 on Vanda gives shuffle seed 687), so make the splits once and keep the CSV.

Left out: the 7 photos with two conflicting labels (until TE decides) and the 3 'Too much
solder' photos (not one of the 5 classes).
Writes hpc/splits/te_splits.csv with columns path, label, session, fold, rank.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from splits_lib import CLASSES, DEFAULT_SPLITS  # noqa: E402

POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "te", "te_outputs", "te_pool.csv")
N_FOLDS, SESSION = 5, "10min"

pool = pd.read_csv(POOL, parse_dates=["capture_time"])
print(f"{len(pool)} unique photos in {os.path.abspath(POOL)}")
df = pool[(~pool.conflict) & pool.label.isin(CLASSES)].reset_index(drop=True)
print(f"left out: {int(pool.conflict.sum())} conflicting labels, {int((pool.label == 'TooMuch').sum())} 'Too much solder'")
df["session"] = df.capture_time.dt.floor(SESSION).dt.strftime("%Y%m%d_%H%M")

FAIR = ["OK", "Misalign", "NoSolder", "LowSolder"]  # 'Single' sits in 2 huge sessions (180 and 123 photos), so it cannot be even
N_SEEDS = 1000  # ~30 s


def unevenness(ct):
    """Lower is better. 1st: every fold has >= 5 'Single'. 2nd: the fold that gets the SMALLEST share
    of OK / Misalign / NoSolder / LowSolder gets as much as possible (a perfect split gives each fold 20%).
    3rd: total distance from 20% over those four classes."""
    share = ct / ct.sum()
    return (-int(ct["Single"].min() >= 5), -round(float(share[FAIR].min().min()), 3),
            float(np.abs(share[FAIR] - 1 / N_FOLDS).to_numpy().sum()))


best = None
for seed in range(N_SEEDS):  # keep the fairest of 1000 shuffles
    folds = np.full(len(df), -1)
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgk.split(df, df.label, df.session)):
        folds[te] = k
    ct = pd.crosstab(folds, df.label).reindex(columns=CLASSES, fill_value=0)
    score = unevenness(ct)
    if best is None or score < best[0]:
        best = (score, seed, folds, ct)
_, seed, df["fold"], ct = best
ct.index.name, ct.columns.name = "fold", None
df["rank"] = np.random.default_rng(0).permutation(len(df))

os.makedirs(os.path.dirname(DEFAULT_SPLITS), exist_ok=True)
df[["path", "label", "session", "fold", "rank"]].to_csv(DEFAULT_SPLITS, index=False)
print(f"{df.session.nunique()} capture sessions -> {N_FOLDS} folds (shuffle seed {seed}). Photos per fold:")
print(ct.to_string())
print(f"smallest share of a rare class in any fold: {ct[FAIR].div(ct[FAIR].sum()).min().min():.0%} (perfect = 20%)")
if (ct == 0).to_numpy().any():
    print("!! some fold has no photos of some class")
# each test fold k trains on the 3 folds that are neither k (test) nor k+1 (validation)
train_min = min(ct.drop(index=[k, (k + 1) % N_FOLDS])[c].sum() for k in range(N_FOLDS) for c in CLASSES[1:])
print(f"fewest real training photos of a defect class in any fold: {train_min} (the 10-photo test needs >= 10)")
print("wrote", DEFAULT_SPLITS)
