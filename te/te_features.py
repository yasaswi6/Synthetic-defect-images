"""Step 1b - features for every real and generated TE image (CPU, ~10 min).

  inception : FID Inception-v3 pool features (for FID / KID)
  resnet    : ResNet-50 features of the original image (classifier, near-duplicate search)
  resnet_d4 : the 7 extra flip/rotate views, only for the LoRA training photos
              (flip/rotate baseline, and flip-aware memorisation check)

Usage: python te_features.py <inception|resnet|aug|pack>
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402
WORK = config.TE_WORK          # big files (features, generated images) live outside OneDrive
os.makedirs(WORK, exist_ok=True)
import extract_features as E  # noqa: E402  (shared loaders / models)

OUT = os.environ.get("TE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_outputs"))
torch.set_num_threads(os.cpu_count())


def load_rgb(path, size):
    """TE images are colour (copper vs solder), so keep RGB, unlike the grey public datasets."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def table():
    pool = pd.read_csv(os.path.join(OUT, "te_pool.csv"))
    gen = pd.read_csv(os.path.join(OUT, "te_generated.csv"))
    pool["kind"], gen["kind"] = "real", "gen"
    pool["version"] = "real"
    t = pd.concat([pool[["path", "label", "kind", "version", "lora_train"]],
                   gen[["path", "label", "kind", "version"]]], ignore_index=True)
    t["lora_train"] = t.lora_train.fillna(False).astype(bool)
    return t


@torch.inference_mode()
def run(stage, batch=16):
    t = table()
    out = os.path.join(WORK, f"feat_{stage}.npy")
    if stage == "inception":
        model, size, norm, todo = E.inception_model(), 299, False, [(i, 0) for i in range(len(t))]
    elif stage == "resnet":
        model, size, norm, todo = E.resnet_model(), 224, True, [(i, 0) for i in range(len(t))]
    else:
        model, size, norm = E.resnet_model(), 224, True
        todo = [(i, v) for i in np.flatnonzero(t.lora_train.values) for v in range(1, 8)]
    feats = {}
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        xs = []
        for i, v in chunk:
            x = E.d4(load_rgb(t.path[i], size))[v]
            xs.append((x - E.MEAN) / E.STD if norm else x)
        y = model(torch.stack(xs))
        y = (y[0] if isinstance(y, list) else y).reshape(len(chunk), -1).numpy()
        for (i, v), row in zip(chunk, y):
            feats[(i, v)] = row
        if (k // batch) % 20 == 0:
            print(f"  {stage} {k + len(chunk)}/{len(todo)}", flush=True)
    if stage == "aug":
        arr = np.full((len(t), 8, 2048), np.nan, np.float32)
        for (i, v), row in feats.items():
            arr[i, v] = row
    else:
        arr = np.stack([feats[(i, 0)] for i in range(len(t))]).astype(np.float32)
    np.save(out, arr)
    print("saved", out, arr.shape)


def pack():
    t = table()
    inc = np.load(os.path.join(WORK, "feat_inception.npy"))
    res = np.load(os.path.join(WORK, "feat_resnet.npy"))
    aug = np.load(os.path.join(WORK, "feat_aug.npy"))
    aug[:, 0] = res
    np.savez_compressed(os.path.join(WORK, "te_features.npz"), inception=inc, resnet_d4=aug,
                        path=t.path.values, label=t.label.values, kind=t.kind.values,
                        version=t.version.values, lora_train=t.lora_train.values)
    print("packed", len(t))


if __name__ == "__main__":
    s = sys.argv[1]
    pack() if s == "pack" else run(s)
