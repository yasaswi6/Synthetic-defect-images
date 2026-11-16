"""Make the training folders for the per-class SDXL LoRAs of one fold.

  python hpc/lora_data.py --fold 0 --fewshot 10 --out $UROP_WORK/lora_data/f0

Each class gets out/<class>/ with its photos plus a metadata.jsonl ({"file_name", "text"}),
the format diffusers' train_text_to_image_lora_sdxl.py reads (--train_data_dir, --caption_column text).
Only the fold's TRAINING photos are used, with the same few-shot picks as vit_detector.py,
so the generator never sees a validation or test photo.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from splits_lib import CLASSES, DEFAULT_SPLITS, PROMPTS, fewshot, fold_parts, load_splits  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--fewshot", type=int, default=10, help="photos per defect class (0 = all training photos)")
ap.add_argument("--ok_cap", type=int, default=100)
ap.add_argument("--classes", nargs="+", default=CLASSES)
ap.add_argument("--splits", default=DEFAULT_SPLITS)
ap.add_argument("--out", required=True)
args = ap.parse_args()

train, _, _ = fold_parts(load_splits(args.splits), args.fold)
train = fewshot(train, args.fewshot, args.ok_cap)
for c in args.classes:
    d = os.path.join(args.out, c)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    rows = train[train.label == c]
    with open(os.path.join(d, "metadata.jsonl"), "w") as f:
        for i, p in enumerate(rows.path):
            name = f"{i:03d}{os.path.splitext(p)[1].lower()}"
            shutil.copy2(p, os.path.join(d, name))
            f.write(json.dumps({"file_name": name, "text": PROMPTS[c]}) + "\n")
    print(f"fold {args.fold}  {c:<10} {len(rows):3d} photos -> {d}")
