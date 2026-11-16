"""Step 1c - re-score last semester's SDXL outputs on TE data.

A. near-duplicates among the real photos
B. quality of generated images (v4 old prompt, v5 new prompt, v6 one LoRA per class,
   v7 = v6 + LPIPS filter):
     B1 class fidelity - does a classifier trained on held-out REAL photos recognise the
        intended defect in the generated images?
     B2 KID - are generated images as close to held-out real photos as the real LoRA
        training photos are?
     B3 memorisation - is a generated image closer to a LoRA training photo than any
        genuinely new photo is?
C. downstream - same fixed training set the LoRAs were trained on (100 OK + 10 per NG),
   tested on EVERY held-out real photo (not 15), comparing:
   baseline / class weights / +15 flip-rotate / +15 SDXL (v4..v7) / +50 SDXL v6 /
   +~10 extra real photos (ceiling). Paired bootstrap CIs over test photos.

Usage: python te_analysis.py   (after te_build_pool.py and te_features.py)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402
WORK = config.TE_WORK          # big files (features, generated images) live outside OneDrive
os.makedirs(WORK, exist_ok=True)
from exp1_fid_reliability import fid, kid  # noqa: E402

OUT = os.environ.get("TE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_outputs"))
CLASSES = ["OK", "Misalign", "NoSolder", "LowSolder", "Single"]
NG = CLASSES[1:]
VERSIONS = ["v4", "v5", "v6", "v7"]
VNAME = {"v4": "v4 one LoRA, label prompt", "v5": "v5 one LoRA, descriptive prompt",
         "v6": "v6 one LoRA per class", "v7": "v7 per class + LPIPS filter"}

z = np.load(os.path.join(WORK, "te_features.npz"), allow_pickle=True)
pool = pd.read_csv(os.path.join(OUT, "te_pool.csv"))
INC, RES = z["inception"].astype(np.float64), z["resnet_d4"]
label, kind, version = z["label"], z["kind"], z["version"]
n_real = len(pool)
real_ok = np.zeros(len(label), bool)
real_ok[:n_real] = (~pool.conflict.values) & (pool.label.values != "TooMuch")
train = real_ok & z["lora_train"].astype(bool)          # photos the LoRAs (and our classifier) train on
held = real_ok & ~train                                   # every other usable real photo
gen = kind == "gen"
RN = normalize(RES[:, 0])


def report(title, df):
    print(f"\n=== {title}")
    print(df.to_string(index=False) if isinstance(df, pd.DataFrame) else df)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)


# ---------------- A. near duplicates among real photos ----------------
idx = np.flatnonzero(real_ok)
S = RN[idx] @ RN[idx].T
np.fill_diagonal(S, -1)
nn = S.max(1)
pairs = [(idx[i], idx[j], S[i, j]) for i in range(len(idx)) for j in range(i + 1, len(idx)) if S[i, j] > 0.95]
cross = [p for p in pairs if train[p[0]] != train[p[1]]]
report("A. real photos: nearest-neighbour cosine similarity (ResNet-50)",
       f"median {np.median(nn):.3f}, 99th pct {np.quantile(nn, .99):.3f}; pairs > 0.95: {len(pairs)} "
       f"(of which train/held-out: {len(cross)})")
counts = pd.DataFrame([dict(cls=c, train=int((train & (label == c)).sum()), held_out=int((held & (label == c)).sum()))
                       for c in CLASSES])
report("usable real photos per class", counts)
save(counts, "te_counts.csv")

# ---------------- B1. class fidelity ----------------
Xh, yh = RES[held, 0], label[held]
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))
cv_pred = np.empty(len(yh), dtype=object)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xh, yh):
    cv_pred[te] = clf.fit(Xh[tr], yh[tr]).predict(Xh[te])
clf.fit(Xh, yh)
rows = [dict(images="real held-out (5-fold CV)", **{c: (cv_pred[yh == c] == c).mean() for c in CLASSES})]
for v in VERSIONS:
    m = gen & (version == v)
    p = clf.predict(RES[m, 0])
    rows.append(dict(images=VNAME[v], **{c: (p[label[m] == c] == c).mean() if (label[m] == c).any() else np.nan
                                          for c in CLASSES}))
fid_tab = pd.DataFrame(rows)
report("B1. share of images recognised as their intended class (classifier trained on held-out real photos)",
       fid_tab.round(2))
save(fid_tab, "te_class_fidelity.csv")
# where do generated images of each class land?
conf = []
for v in VERSIONS:
    for c in CLASSES:
        m = gen & (version == v) & (label == c)
        if m.any():
            p = clf.predict(RES[m, 0])
            conf.append(dict(version=v, intended=c, **{f"pred_{k}": (p == k).mean() for k in CLASSES}))
save(pd.DataFrame(conf), "te_class_fidelity_confusion.csv")

# ---------------- B2. KID / FID versus real ----------------
rng = np.random.default_rng(0)
rows = []
for c in CLASSES:
    ref = INC[held & (label == c)]
    tr_idx = np.flatnonzero(train & (label == c))
    n = min(10, len(tr_idx))
    # real comparator: n LoRA-training photos vs held-out photos (resampled when there are >10)
    real_k = [kid(INC[rng.choice(tr_idx, n, replace=False)], ref) for _ in range(200)]
    real_f = [fid(INC[rng.choice(tr_idx, n, replace=False)], ref) for _ in range(200)]
    row = dict(cls=c, n_ref=len(ref), kid_real=np.median(real_k), fid_real=np.median(real_f))
    for v in VERSIONS:
        g_idx = np.flatnonzero(gen & (version == v) & (label == c))
        ks = [kid(INC[rng.choice(g_idx, n, replace=False)], ref) for _ in range(200)]
        fs = [fid(INC[rng.choice(g_idx, n, replace=False)], ref) for _ in range(200)]
        row[f"kid_{v}"] = np.median(ks)
        row[f"fid_{v}"] = np.median(fs)
    rows.append(row)
kid_tab = pd.DataFrame(rows)
report("B2. median KID and FID to held-out real photos, 10 images per set, 200 draws "
       "(kid_real/fid_real = real LoRA-training photos)", kid_tab.round(3))
save(kid_tab, "te_kid_fid.csv")

# ---------------- B3. memorisation ----------------
rows, examples = [], []
for c in CLASSES:
    tr_idx = np.flatnonzero(train & (label == c))
    TV = normalize(RES[tr_idx].reshape(-1, 2048)).reshape(len(tr_idx), 8, 2048)   # all flips of training photos
    TV = np.nan_to_num(TV)

    def nn_sim(F):
        s = np.einsum("nd,tvd->ntv", normalize(F), TV)
        flat = s.reshape(len(F), -1)
        return flat.max(1), tr_idx[flat.argmax(1) // 8]

    h_sim, _ = nn_sim(RES[held & (label == c), 0])
    thr = h_sim.max()
    row = dict(cls=c, heldout_median=np.median(h_sim), heldout_max=thr)
    for v in VERSIONS:
        g_idx = np.flatnonzero(gen & (version == v) & (label == c))
        g_sim, g_nn = nn_sim(RES[g_idx, 0])
        row[f"{v}_median"] = np.median(g_sim)
        row[f"{v}_closer_than_any_real"] = (g_sim > thr).mean()
        for k in np.argsort(-g_sim)[:2]:
            examples.append(dict(version=v, cls=c, gen=z["path"][g_idx[k]], train=z["path"][g_nn[k]], sim=g_sim[k]))
    rows.append(row)
mem = pd.DataFrame(rows)
report("B3. nearest-training-photo similarity (cosine, flip-aware); share of generated images "
       "closer to a training photo than ANY held-out real photo is", mem.round(3))
save(mem, "te_memorisation.csv")
save(pd.DataFrame(examples), "te_memorisation_examples.csv")

# ---------------- C. downstream ----------------
REPEATS, BOOT = 50, 2000
tr = np.flatnonzero(train)
ho = np.flatnonzero(held)
Xtr, ytr = RES[tr, 0], label[tr]
methods = ["A baseline", "B class weights", "C +15 flip/rotate"] + [f"+15 SDXL {v}" for v in VERSIONS] + \
          ["+50 SDXL v6", "+~10 real (ceiling)"]
correct = {m: np.zeros(len(ho)) for m in methods}           # how often each held-out photo is right
pred_ok = {m: np.zeros(len(ho)) for m in methods}            # how often predicted OK (for escapes)


def fit(X, y, weighted=False):
    # PCA to 128 dims (fit on that model's own training data) keeps 900 fits fast on a CPU
    m = make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(C=0.5, max_iter=5000,
                                                           class_weight="balanced" if weighted else None))
    return m.fit(X, y)


for r in range(REPEATS):
    rng = np.random.default_rng(1000 + r)
    half = np.zeros(len(ho), int)
    for c in CLASSES:
        ii = np.flatnonzero(label[ho] == c)
        half[rng.permutation(ii)[: len(ii) // 2]] = 1
    for fold in (0, 1):
        te, ex = np.flatnonzero(half == fold), np.flatnonzero(half != fold)
        add = {}
        aug = []
        for c in NG:
            ids = tr[ytr == c]
            pairs_ = [(i, v) for i in ids for v in range(1, 8)]
            aug += [(RES[pairs_[k][0], pairs_[k][1]], c) for k in rng.choice(len(pairs_), 15, replace=False)]
        add["C +15 flip/rotate"] = aug
        for v in VERSIONS:
            add[f"+15 SDXL {v}"] = [(RES[i, 0], label[i]) for c in NG
                                     for i in rng.choice(np.flatnonzero(gen & (version == v) & (label == c)), 15,
                                                         replace=False)]
        add["+50 SDXL v6"] = [(RES[i, 0], label[i]) for c in NG for i in np.flatnonzero(gen & (version == "v6") & (label == c))]
        add["+~10 real (ceiling)"] = [(RES[ho[i], 0], label[ho[i]]) for c in NG
                                      for i in rng.permutation(ex[label[ho[ex]] == c])[:15]]
        Xte = RES[ho[te], 0]
        for m in methods:
            if m == "A baseline":
                model = fit(Xtr, ytr)
            elif m == "B class weights":
                model = fit(Xtr, ytr, weighted=True)
            else:
                Xa = np.array([a for a, _ in add[m]])
                ya = np.array([b for _, b in add[m]])
                model = fit(np.vstack([Xtr, Xa]), np.concatenate([ytr, ya]))
            p = model.predict(Xte)
            correct[m][te] += (p == label[ho[te]])
            pred_ok[m][te] += (p == "OK")
for m in methods:
    correct[m] /= REPEATS
    pred_ok[m] /= REPEATS

y_ho = label[ho]
cls_idx = {c: np.flatnonzero(y_ho == c) for c in CLASSES}


def macro(v, sel=None):
    return np.mean([v[cls_idx[c] if sel is None else sel[c]].mean() for c in CLASSES])


rng = np.random.default_rng(7)
boot_sel = [{c: rng.choice(cls_idx[c], len(cls_idx[c])) for c in CLASSES} for _ in range(BOOT)]
rows = []
for m in methods:
    est = macro(correct[m])
    b = np.array([macro(correct[m], s) for s in boot_sel])
    d = np.array([macro(correct[m], s) - macro(correct["A baseline"], s) for s in boot_sel])
    ng_idx = np.concatenate([cls_idx[c] for c in NG])
    rows.append(dict(method=m, macro_acc=est, ci_lo=np.quantile(b, .025), ci_hi=np.quantile(b, .975),
                     gain_vs_A=est - macro(correct["A baseline"]), gain_ci_lo=np.quantile(d, .025),
                     gain_ci_hi=np.quantile(d, .975),
                     escape_rate=pred_ok[m][ng_idx].mean(), false_alarm=1 - pred_ok[m][cls_idx["OK"]].mean(),
                     **{f"acc_{c}": correct[m][cls_idx[c]].mean() for c in CLASSES}))
down = pd.DataFrame(rows)
report(f"C. downstream on {len(ho)} held-out real photos "
       f"({', '.join(f'{c} {len(cls_idx[c])}' for c in CLASSES)}); {REPEATS} repeats, 95% bootstrap CIs",
       down.round(3))
save(down, "te_downstream.csv")
