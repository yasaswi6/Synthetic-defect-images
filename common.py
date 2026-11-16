"""Shared helpers for the UROP public-dataset experiments.

Datasets used (both freely available on GitHub):
  * RIAWELC  - radiographic weld-defect patches (github.com/stefyste/RIAWELC)
  * MTD      - Magnetic Tile Defect dataset (github.com/abin24/Magnetic-tile-defect-datasets.)

setup_data.py downloads them; paths come from config.py.
"""
import os
import re

import numpy as np

import config

DATA_ROOTS = {
    # folder that contains NoDifetto / Difetto1 / Difetto2 / Difetto4
    "riawelc": os.environ.get("RIAWELC_DIR", config.RIAWELC_DIR),
    # folder that contains MT_Free / MT_Crack / ...
    "mtd": os.environ.get("MTD_DIR", config.MTD_DIR),
}

# Map folder names to readable class names. "OK" is the majority / no-defect class.
# RIAWELC's folder-to-class mapping is not documented; from looking at samples:
#   D1 (Difetto1) = cracks (thin lines), D2 (Difetto2) = porosity (round voids),
#   D4 (Difetto4) = lack of penetration (dark band along the weld root).
RIAWELC_CLASSES = {"NoDifetto": "OK", "Difetto1": "D1", "Difetto2": "D2", "Difetto4": "D4"}
MTD_CLASSES = {"MT_Free": "OK", "MT_Blowhole": "Blowhole", "MT_Break": "Break",
               "MT_Crack": "Crack", "MT_Fray": "Fray", "MT_Uneven": "Uneven"}

OUT_DIR = os.environ.get("UROP_OUT", os.path.join(config.CODE_DIR, "outputs"))
FEAT_DIR = os.path.join(config.DATA_DIR, "features")          # big cached features stay outside OneDrive
FIG_DIR = os.path.join(OUT_DIR, "figures")
RES_DIR = os.path.join(OUT_DIR, "results")
for _d in (FEAT_DIR, FIG_DIR, RES_DIR):
    os.makedirs(_d, exist_ok=True)


def list_images(name):
    """Return (paths, labels, groups). `groups` = source part/radiograph id (for leakage-free splits)."""
    paths, labels, groups = [], [], []
    if name == "riawelc":
        root = DATA_ROOTS["riawelc"]
        for folder, cls in RIAWELC_CLASSES.items():
            for f in sorted(os.listdir(os.path.join(root, folder))):
                if f.lower().endswith(".png"):
                    paths.append(os.path.join(root, folder, f))
                    labels.append(cls)
                    # e.g. RRT-40R_Img1_A80_S4_[8][49].png -> film "RRT-40R"
                    groups.append(f.split("_")[0])
    elif name == "mtd":
        root = DATA_ROOTS["mtd"]
        for folder, cls in MTD_CLASSES.items():
            d = os.path.join(root, folder, "Imgs")
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".jpg"):
                    paths.append(os.path.join(d, f))
                    labels.append(cls)
                    groups.append(re.sub(r"^exp\d+_", "", f).split(".")[0])
    else:
        raise ValueError(name)
    return np.array(paths), np.array(labels), np.array(groups)


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half
