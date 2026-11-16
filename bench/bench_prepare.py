"""Step 4a - unpack the public benchmark downloads and index them.

Put these (unopened) in  <UROP_data>/downloads :
    transistor.tar.xz, cable.tar.xz      (MVTec AD, mvtec.com/company/research/datasets/mvtec-ad/downloads)
    the SolDef_AI .zip from Kaggle       (indexed once we know its layout)

Writes one index per dataset to bench/bench_outputs/<dataset>_index.csv with columns
    path, label ("OK" = defect-free, otherwise the defect type), role, mask
role: "train_good" (MVTec defect-free training images) or "test" (everything else).
"""
import glob
import os
import sys
import tarfile
import zipfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

DL = os.path.join(config.DATA_DIR, "downloads")
MVTEC = os.path.join(config.DATA_DIR, "mvtec")
SOLDEF = os.path.join(config.DATA_DIR, "soldef")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_outputs")
os.makedirs(OUT, exist_ok=True)


def extract():
    os.makedirs(DL, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(DL, "*.tar.xz"))):
        cat = os.path.basename(f).split(".")[0]
        if os.path.isdir(os.path.join(MVTEC, cat, "test")):
            print("  already unpacked", cat)
            continue
        print("  unpacking", os.path.basename(f), "(a minute or two)")
        os.makedirs(MVTEC, exist_ok=True)
        with tarfile.open(f, "r:xz") as t:
            try:
                t.extractall(MVTEC, filter="data")
            except TypeError:                     # Python < 3.12
                t.extractall(MVTEC)
    for f in sorted(glob.glob(os.path.join(DL, "*.zip"))):
        if os.path.isdir(SOLDEF) and os.listdir(SOLDEF):
            print("  already unpacked", os.path.basename(f))
            continue
        print("  unpacking", os.path.basename(f))
        with zipfile.ZipFile(f) as z:
            z.extractall(SOLDEF)


def index_mvtec(cat):
    root = os.path.join(MVTEC, cat)
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "train", "good", "*.png"))):
        rows.append(dict(path=p, label="OK", role="train_good", mask=""))
    for d in sorted(os.listdir(os.path.join(root, "test"))):
        for p in sorted(glob.glob(os.path.join(root, "test", d, "*.png"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            m = os.path.join(root, "ground_truth", d, stem + "_mask.png")
            rows.append(dict(path=p, label="OK" if d == "good" else d, role="test",
                             mask=m if os.path.exists(m) else ""))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"mvtec_{cat}_index.csv"), index=False)
    print(f"  mvtec_{cat}:", df.groupby(["role", "label"]).size().to_dict())


if __name__ == "__main__":
    print("Downloads folder:", DL)
    extract()
    cats = [c for c in ("transistor", "cable") if os.path.isdir(os.path.join(MVTEC, c, "test"))]
    if not cats:
        print("\n!! No MVTec categories found. Put transistor.tar.xz / cable.tar.xz in", DL)
    for c in cats:
        index_mvtec(c)
    if os.path.isdir(SOLDEF):
        print("  SolDef_AI unpacked to", SOLDEF, "- its loader comes next")
