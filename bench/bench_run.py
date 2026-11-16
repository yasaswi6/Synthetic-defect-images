"""Step 4b - the same CPU test as Result 4, on a public benchmark (MVTec AD category, later SolDef_AI).

Protocol (mirrors last semester's TE setup and the usual MVTec few-shot protocol):
  * training set: 100 defect-free images + 1/3 of the images of each defect type
    (the split DFMGAN / AnomalyDiffusion use, so SDXL runs later can be compared with them)
  * test set: every other image (MVTec's defect-free test images + the other 2/3 of defects)
  * methods: baseline, class weights, +15 flip/rotate, +15 stronger augmentation, +15 cut-and-mix,
    +15 mask paste (defect region cut out with MVTec's ground-truth mask and blended onto a
    defect-free image), and +real defects (ceiling, half of the test defects lent for training,
    halves swapped)
  * 5 random splits; frozen ResNet-50 + PCA + logistic regression; 95% bootstrap CIs

Usage: python bench/bench_run.py mvtec_transistor     (or mvtec_cable)
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import config  # noqa: E402
import extract_features as E  # noqa: E402

OUT = os.path.join(HERE, "bench_outputs")
WORK = os.path.join(config.DATA_DIR, "bench_work")
os.makedirs(WORK, exist_ok=True)
torch.set_num_threads(os.cpu_count())

SPLITS, N_OK_TRAIN, N_ADD, BOOT, SIZE = 5, 100, 15, 2000, 512
CHEAP = ["stronger augmentation", "cut-and-mix", "mask paste"]


# ---------------- images ----------------
def load(p, size=SIZE):
    return Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)


def load_mask(p, size=SIZE):
    return np.asarray(Image.open(p).convert("L").resize((size, size), Image.NEAREST)) > 127


def rich_aug(img, rng):
    w, h = img.size
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    M[:, 2] += rng.uniform(-0.05, 0.05, 2) * w
    img = Image.fromarray(cv2.warpAffine(np.asarray(img), M, (w, h), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REFLECT_101))
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enh(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def cutmix(a, b, rng):
    a, b = np.asarray(a).copy(), np.asarray(b)
    h, w = a.shape[:2]
    area, ratio = rng.uniform(0.25, 0.5) * h * w, rng.uniform(0.5, 2)
    rh, rw = int(min(h, np.sqrt(area * ratio))), int(min(w, np.sqrt(area / ratio)))
    y, x = rng.integers(0, h - rh + 1), rng.integers(0, w - rw + 1)
    a[y:y + rh, x:x + rw] = b[y:y + rh, x:x + rw]
    return Image.fromarray(a)


def mask_paste(src, mask, dst):
    """Blend the masked defect region of `src` onto defect-free `dst` at the same place."""
    m = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
    m[:3], m[-3:], m[:, :3], m[:, -3:] = 0, 0, 0, 0
    if m.sum() == 0:
        return None
    ys, xs = np.nonzero(m)
    centre = (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2))
    s = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
    d = cv2.cvtColor(np.asarray(dst), cv2.COLOR_RGB2BGR)
    try:
        out = cv2.seamlessClone(s, d, m, centre, cv2.NORMAL_CLONE)
    except cv2.error:
        return None
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


# ---------------- features ----------------
@torch.inference_mode()
def resnet_feats(model, imgs, batch=16):
    out = []
    for k in range(0, len(imgs), batch):
        x = torch.stack([(torch.from_numpy(np.asarray(i.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255)
                          .permute(2, 0, 1) - E.MEAN) / E.STD for i in imgs[k:k + batch]])
        out.append(model(x).numpy())
    return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)


def real_features(name, df, model):
    cache = os.path.join(WORK, f"{name}_real.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z["f"]) == len(df):
            return z["f"], z["d4"]
    print("  features of", len(df), "real images (+ flips/rotations of defects)", flush=True)
    f = resnet_feats(model, [load(p) for p in df.path])
    d4 = np.full((len(df), 8, 2048), np.nan, np.float32)
    for i in np.flatnonzero(df.label.values != "OK"):
        x = torch.from_numpy(np.asarray(load(df.path.iloc[i], 224), dtype=np.float32) / 255).permute(2, 0, 1)
        views = [(v - E.MEAN) / E.STD for v in E.d4(x)]
        with torch.inference_mode():
            d4[i] = model(torch.stack(views)).numpy()
    np.savez_compressed(cache, f=f, d4=d4)
    return f, d4


# ---------------- protocol ----------------
def fit(X, y, weighted=False):
    k = min(128, len(X) - 1)
    return make_pipeline(StandardScaler(), PCA(k, random_state=0),
                         LogisticRegression(C=0.5, max_iter=5000,
                                            class_weight="balanced" if weighted else None)).fit(X, y)


def run(name):
    df = pd.read_csv(os.path.join(OUT, f"{name}_index.csv")).fillna("")
    model = E.resnet_model()
    F, D4 = real_features(name, df, model)
    y = df.label.values
    defects = sorted(set(y) - {"OK"})
    classes = ["OK"] + defects
    methods = ["A baseline", "B class weights", "C +15 flip/rotate"] + [f"+15 {c}" for c in CHEAP] + \
              ["+real defects (ceiling)"]
    per_split = []
    for s in range(SPLITS):
        rng = np.random.default_rng(s)
        ok_tr = rng.permutation(np.flatnonzero((y == "OK") & (df.role.values == "train_good")))[:N_OK_TRAIN]
        ok_te = np.flatnonzero((y == "OK") & (df.role.values == "test"))
        tr, te = list(ok_tr), list(ok_te)
        for d in defects:
            ii = rng.permutation(np.flatnonzero(y == d))
            k = int(np.ceil(len(ii) / 3))
            tr += list(ii[:k])
            te += list(ii[k:])
        tr, te = np.array(tr), np.array(te)
        # cheap fakes from this split's training defects only
        fakes = {c: [] for c in CHEAP}
        good = [load(p) for p in df.path.values[ok_tr[:30]]]
        for d in defects:
            src = tr[y[tr] == d]
            imgs = {i: load(df.path.values[i]) for i in src}
            for _ in range(N_ADD):
                i = rng.choice(src)
                fakes["stronger augmentation"].append((rich_aug(imgs[i], rng), d))
                fakes["cut-and-mix"].append((cutmix(imgs[i], imgs[rng.choice(src)], rng), d))
                for _try in range(10):
                    j = rng.choice(src)
                    m = df["mask"].values[j]
                    out = mask_paste(imgs[j], load_mask(m), good[rng.integers(len(good))]) if m else None
                    if out is not None:
                        fakes["mask paste"].append((out, d))
                        break
        fake_X = {c: (resnet_feats(model, [a for a, _ in v]), np.array([b for _, b in v])) for c, v in fakes.items()}
        if s == 0:
            prev = [a for c in CHEAP for a, b in fakes[c][:: max(1, N_ADD)]][:12]
            sheet = Image.new("RGB", (6 * 160, 2 * 160), "white")
            for k_, im in enumerate(prev):
                sheet.paste(im.resize((156, 156)), ((k_ % 6) * 160, (k_ // 6) * 160))
            sheet.save(os.path.join(OUT, f"{name}_cheap_preview.png"))
        # two folds over the test set so the "+real" ceiling can borrow half of it
        correct = {m: np.zeros(len(te)) for m in methods}
        pred_ok = {m: np.zeros(len(te)) for m in methods}
        half = np.zeros(len(te), int)
        for c in classes:
            jj = np.flatnonzero(y[te] == c)
            half[rng.permutation(jj)[: len(jj) // 2]] = 1
        Xtr, ytr = F[tr], y[tr]
        for fold in (0, 1):
            tt, ex = np.flatnonzero(half == fold), np.flatnonzero(half != fold)
            aug = []
            for d in defects:
                pairs = [(i, v) for i in tr[ytr == d] for v in range(1, 8)]
                aug += [(D4[pairs[k][0], pairs[k][1]], d) for k in rng.choice(len(pairs), min(N_ADD, len(pairs)),
                                                                               replace=False)]
            extra = [(F[te[i]], y[te[i]]) for i in ex if y[te[i]] != "OK"]
            for m in methods:
                if m == "A baseline":
                    clf = fit(Xtr, ytr)
                elif m == "B class weights":
                    clf = fit(Xtr, ytr, weighted=True)
                else:
                    if m == "C +15 flip/rotate":
                        add = aug
                        Xa, ya = np.array([a for a, _ in add]), np.array([b for _, b in add])
                    elif m == "+real defects (ceiling)":
                        Xa, ya = np.array([a for a, _ in extra]), np.array([b for _, b in extra])
                    else:
                        Xa, ya = fake_X[m.replace("+15 ", "")]
                    clf = fit(np.vstack([Xtr, Xa]), np.concatenate([ytr, ya]))
                p = clf.predict(F[te[tt]])
                correct[m][tt] = p == y[te[tt]]
                pred_ok[m][tt] = p == "OK"
        per_split.append(dict(te=te, correct=correct, pred_ok=pred_ok))
        print(f"  split {s + 1}/{SPLITS} done (train {len(tr)}, test {len(te)})", flush=True)

    # ---- summarise: average over splits, bootstrap test images within each split ----
    rng = np.random.default_rng(7)

    def macro(sp, m, sel=None):
        yt = y[sp["te"]]
        return np.mean([sp["correct"][m][(np.flatnonzero(yt == c) if sel is None else sel[c])].mean()
                        for c in classes])

    boots = [[{c: rng.choice(np.flatnonzero(y[sp["te"]] == c), (y[sp["te"]] == c).sum()) for c in classes}
              for sp in per_split] for _ in range(BOOT)]
    rows = []
    for m in methods:
        est = np.mean([macro(sp, m) for sp in per_split])
        base = np.mean([macro(sp, "A baseline") for sp in per_split])
        b = np.array([np.mean([macro(sp, m, bs[i]) for i, sp in enumerate(per_split)]) for bs in boots])
        db = np.array([np.mean([macro(sp, m, bs[i]) - macro(sp, "A baseline", bs[i]) for i, sp in enumerate(per_split)])
                       for bs in boots])
        ng = [np.flatnonzero(y[sp["te"]] != "OK") for sp in per_split]
        okk = [np.flatnonzero(y[sp["te"]] == "OK") for sp in per_split]
        rows.append(dict(method=m, macro_acc=est, ci_lo=np.quantile(b, .025), ci_hi=np.quantile(b, .975),
                         gain_vs_A=est - base, gain_ci_lo=np.quantile(db, .025), gain_ci_hi=np.quantile(db, .975),
                         escape_rate=np.mean([sp["pred_ok"][m][g].mean() for sp, g in zip(per_split, ng)]),
                         false_alarm=np.mean([1 - sp["pred_ok"][m][o].mean() for sp, o in zip(per_split, okk)]),
                         **{f"acc_{c}": np.mean([sp["correct"][m][y[sp["te"]] == c].mean() for sp in per_split])
                            for c in classes}))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, f"{name}_downstream.csv"), index=False)
    counts = {c: int((y == c).sum()) for c in classes}
    pd.set_option("display.width", 250)
    print(f"\n=== {name}: images per class {counts}; {SPLITS} random splits, 95% bootstrap CIs")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    run(sys.argv[1])
