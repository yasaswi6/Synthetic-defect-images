"""Extract and cache image features for the experiments (CPU is fine, ~20 min in total).

  * Inception-v3 pool3 (2048-d) with the official FID weights  -> used for FID / KID
  * ResNet-50 (timm a1_0, ImageNet) penultimate (2048-d)      -> used for the classifier,
    plus the 7 flip/rotate (D4) versions of a subset of defect images, so classical
    augmentation can be simulated without re-running the network.

Usage:  python extract_features.py <riawelc|mtd> <index|inception|resnet|aug|pack>
Run the stages in that order; each stage caches its output so it can be resumed.
Weights: setup_data.py downloads both (GitHub releases of mseitzer/pytorch-fid and
huggingface/pytorch-image-models) to the locations in config.py.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

import config
from common import FEAT_DIR, list_images

WEIGHTS_DIR = config.WEIGHTS_DIR
torch.set_num_threads(os.cpu_count())

MAX_PER_CLASS = 300      # cap per class (keeps CPU time reasonable)
MAX_AUG_PER_CLASS = 80   # defect images per class that also get the 7 extra flip/rotate views
BATCH = 16


def load_gray(path):
    return Image.open(path).convert("L")


def to_tensor(img, size):
    img = img.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return x[None].repeat(3, 1, 1)  # grey -> 3 channels


def d4(x):
    """The 8 flip/rotate versions of a CHW tensor (index 0 = original)."""
    outs = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(1, 2))
        outs += [r, torch.flip(r, dims=(2,))]
    return outs


def inception_model():
    from pytorch_fid.inception import InceptionV3
    # pytorch-fid looks for its weights in <TORCH_HOME>/hub/checkpoints; setup_data.py puts them there
    os.environ["TORCH_HOME"] = WEIGHTS_DIR
    m = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], use_fid_inception=True)
    return m.eval()


def resnet_model():
    import timm
    m = timm.create_model("resnet50", pretrained=False, num_classes=0)
    sd = torch.load(config.RESNET_FILE, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    m.load_state_dict(sd, strict=True)
    return m.eval()


MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def f(name, stage):
    return os.path.join(FEAT_DIR, f"{name}_{stage}.npz")


def stage_index(name):
    paths, labels, groups = list_images(name)
    rng = np.random.default_rng(0)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        keep += list(rng.permutation(idx)[:MAX_PER_CLASS])
    keep = np.sort(np.array(keep))
    paths, labels, groups = paths[keep], labels[keep], groups[keep]
    has_aug = np.zeros(len(paths), bool)
    for c in np.unique(labels):
        if c != "OK":
            idx = np.flatnonzero(labels == c)
            has_aug[rng.permutation(idx)[:MAX_AUG_PER_CLASS]] = True
    np.savez(f(name, "index"), paths=paths, labels=labels, groups=groups, has_aug=has_aug)
    print(name, len(paths), "images;", has_aug.sum(), "get flip/rotate views;",
          dict(zip(*np.unique(labels, return_counts=True))))


@torch.inference_mode()
def stage_net(name, which):
    z = np.load(f(name, "index"), allow_pickle=True)
    paths, has_aug = z["paths"], z["has_aug"]
    out = f(name, which)
    if which == "inception":
        model, size, norm = inception_model(), 299, False   # pytorch-fid normalises internally
        todo = [(i, 0) for i in range(len(paths))]
    elif which == "resnet":
        model, size, norm = resnet_model(), 224, True
        todo = [(i, 0) for i in range(len(paths))]
    else:  # aug
        model, size, norm = resnet_model(), 224, True
        todo = [(i, v) for i in np.flatnonzero(has_aug) for v in range(1, 8)]
    # resume support
    done = {}
    part = out + ".part.npz"
    if os.path.exists(part):
        p = np.load(part)
        done = {(int(i), int(v)): row for i, v, row in zip(p["i"], p["v"], p["x"])}
    todo = [t for t in todo if t not in done]
    print(name, which, len(todo), "views to do,", len(done), "cached")
    cache = {}
    for k in range(0, len(todo), BATCH):
        chunk = todo[k:k + BATCH]
        xs = []
        for i, v in chunk:
            if i not in cache:
                cache = {i: to_tensor(load_gray(paths[i]), size)}
            x = d4(cache[i])[v]
            xs.append((x - MEAN) / STD if norm else x)
        y = model(torch.stack(xs))
        y = (y[0] if isinstance(y, list) else y).reshape(len(chunk), -1).numpy()
        for (i, v), row in zip(chunk, y):
            done[(i, v)] = row
        if (k // BATCH) % 10 == 0 or k + BATCH >= len(todo):
            keys = list(done)
            np.savez(part, i=[a for a, _ in keys], v=[b for _, b in keys], x=np.stack([done[t] for t in keys]))
            print(f"  {which} {k + len(chunk)}/{len(todo)}", flush=True)
    keys = list(done)
    np.savez(out, i=[a for a, _ in keys], v=[b for _, b in keys], x=np.stack([done[t] for t in keys]))
    os.remove(part)
    print("saved", out)


def stage_pack(name):
    z = np.load(f(name, "index"), allow_pickle=True)
    n = len(z["paths"])
    inc = np.load(f(name, "inception"))
    X_inc = np.zeros((n, 2048), np.float32)
    X_inc[inc["i"]] = inc["x"]
    feats = np.full((n, 8, 2048), np.nan, np.float32)
    for stage in ("resnet", "aug"):
        r = np.load(f(name, stage))
        feats[r["i"], r["v"]] = r["x"]
    np.savez_compressed(os.path.join(FEAT_DIR, f"{name}.npz"), paths=z["paths"], labels=z["labels"],
                        groups=z["groups"], has_aug=z["has_aug"], inception=X_inc, resnet_d4=feats)
    print("packed", name, n)


if __name__ == "__main__":
    name, stage = sys.argv[1], sys.argv[2]
    if stage == "index":
        stage_index(name)
    elif stage in ("inception", "resnet", "aug"):
        stage_net(name, stage)
    elif stage == "pack":
        stage_pack(name)
