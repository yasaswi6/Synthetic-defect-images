"""B4 - diversity: mean pairwise cosine similarity (ResNet-50) within each class.
Real photos vs each SDXL version; higher similarity = less variety (mode collapse)."""
import os
import sys
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402
WORK = config.TE_WORK          # big files (features, generated images) live outside OneDrive
os.makedirs(WORK, exist_ok=True)
OUT = os.environ.get("TE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_outputs"))
z = np.load(os.path.join(WORK, "te_features.npz"), allow_pickle=True)
pool = pd.read_csv(os.path.join(OUT, "te_pool.csv"))
F = normalize(z["resnet_d4"][:, 0]); lab, ver = z["label"], z["version"]
real = np.zeros(len(lab), bool); real[:len(pool)] = (~pool.conflict.values)
rng = np.random.default_rng(0)
rows = []
for c in ["OK", "Misalign", "NoSolder", "LowSolder", "Single"]:
    row = dict(cls=c)
    for v in ["real", "v4", "v5", "v6", "v7"]:
        idx = np.flatnonzero((real if v == "real" else (ver == v)) & (lab == c))
        vals = []
        for _ in range(100):                      # 15 images per set, so every version is compared fairly
            s = rng.choice(idx, 15, replace=False)
            S = F[s] @ F[s].T
            vals.append(S[np.triu_indices(15, 1)].mean())
        row[v] = np.mean(vals)
    rows.append(row)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "te_diversity.csv"), index=False)
print("mean pairwise cosine similarity within class (15-image sets; higher = less varied)")
print(d.round(3).to_string(index=False))
