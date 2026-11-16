"""Step 1a - build one clean pool of the real TE images from the previous team's repo
(github.com/TPH777/NUSxTE_UROP) and list every generated image we want to score.

The repo stores the same photos in several dataset folders (v1, v1_cut, training_set,
extended_set, new_set_with_vae). We de-duplicate by file content (MD5), record which
folders/splits each photo appears in, and flag:
  * label conflicts  - same photo filed under different classes
  * split leaks      - same photo in train and test (or valid and test) of one dataset
VAE outputs (generated_*.png inside new_set_with_vae) are not real and are dropped.

Usage: python te_build_pool.py [path to NUSxTE_UROP]   (default: config.TE_REPO)   -> writes te_pool.csv, te_generated.csv
"""
import hashlib
import os
import sys

import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402
WORK = config.TE_WORK          # big files (features, generated images) live outside OneDrive
os.makedirs(WORK, exist_ok=True)

REPO = sys.argv[1] if len(sys.argv) > 1 else config.TE_REPO
OUT = os.environ.get("TE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_outputs"))
os.makedirs(OUT, exist_ok=True)

SHORT = {"NG - Misalign": "Misalign", "NG - No solder": "NoSolder", "NG - Not enough solder": "LowSolder",
         "NG - Single": "Single", "OK": "OK", "NG - Too much solder": "TooMuch"}
# label precedence: the curated set used for the reported results first, raw folders last
PRECEDENCE = ["extended_set", "training_set", "new_set_with_vae", "v1_cut", "v1"]

rows = []
root = os.path.join(REPO, "datasets")
for dp, _, fn in os.walk(root):
    for f in fn:
        if not f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            continue
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, root).split(os.sep)
        rows.append(dict(path=p, set=rel[0], split=rel[1] if len(rel) == 4 else "-", cls=SHORT[rel[-2]],
                         name=f, md5=hashlib.md5(open(p, "rb").read()).hexdigest()))
inv = pd.DataFrame(rows)
inv = inv[~inv.name.str.startswith("generated_")]          # VAE outputs, not real

pool = []
for h, g in inv.groupby("md5"):
    labels = set(g.cls)
    best = min(g.itertuples(), key=lambda r: PRECEDENCE.index(r.set))
    ext = g[g.set == "extended_set"]
    pool.append(dict(md5=h, path=best.path, name=best.name, label=best.cls,
                     all_labels="|".join(sorted(labels)), conflict=len(labels) > 1,
                     lora_train=bool(((ext.split == "train")).any()),       # SDXL LoRAs were trained on extended_set/train
                     ext_split="|".join(sorted(set(ext.split))) if len(ext) else "",
                     in_sets="|".join(sorted(set(g.set + ":" + g.split)))))
pool = pd.DataFrame(pool)
pool["capture_time"] = pd.to_datetime(pool.name.str.extract(r"_(\d{8}_\d{6})_")[0], format="%Y%m%d_%H%M%S",
                                      errors="coerce")
pool.to_csv(os.path.join(OUT, "te_pool.csv"), index=False)

# ---- generated images (SDXL, previous team) ----
GEN_CLASS = {"Low_Solder": "LowSolder", "Misaligned": "Misalign", "No_Solder": "NoSolder",
             "Single": "Single", "CLS_OK": "OK", "NG - Misalign": "Misalign", "NG - No solder": "NoSolder",
             "NG - Not enough solder": "LowSolder", "NG - Single": "Single", "OK": "OK"}
gen = []
for v in ["v4", "v5", "v6", "v7"]:
    vd = os.path.join(REPO, "sdxl", "inferred", v)
    for d in sorted(os.listdir(vd)):
        if not os.path.isdir(os.path.join(vd, d)):
            continue
        cls = next((c for k, c in GEN_CLASS.items() if k in d), None)
        for f in sorted(os.listdir(os.path.join(vd, d))):
            if f.lower().endswith((".png", ".jpg")):
                gen.append(dict(version=v, label=cls, path=os.path.join(vd, d, f)))
gen = pd.DataFrame(gen)
gen.to_csv(os.path.join(OUT, "te_generated.csv"), index=False)

# ---- audit printout ----
print("real photos (unique):", len(pool))
print(pool.groupby("label").size().to_string())
print("\nlabel conflicts:", int(pool.conflict.sum()))
print(pool[pool.conflict][["name", "all_labels", "in_sets"]].to_string(index=False))
for s in ["extended_set", "training_set", "new_set_with_vae"]:
    g = inv[inv.set == s]
    sp = {k: set(v.md5) for k, v in g.groupby("split")}
    print(f"\n{s}: " + ", ".join(f"{k}={len(v)}" for k, v in sp.items()) + " | overlaps: " +
          ", ".join(f"{a}&{b}={len(sp[a] & sp[b])}" for i, a in enumerate(sp) for b in list(sp)[i + 1:]))
print("\ngenerated:", gen.groupby(["version", "label"]).size().unstack().to_string())
