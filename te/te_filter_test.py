"""Approach A - keep only the fakes that look right, then re-run the detector test.

A "judge" (logistic regression on ResNet-50 features) is trained on the SAME training photos
the detector uses (100 OK + ~10 per defect, plus their flipped/rotated views). It never sees a
test photo, so it is a judge AITK could really have. Each SDXL fake is kept only if the judge
calls it the class it was meant to be. The detector test is then exactly Section C of
te_analysis.py: 436 held-out real photos, 50 repeats, half-swap, 95% bootstrap ranges.

Usage: python te/te_filter_test.py   (after te_build_pool.py and te_features.py)
Writes te_outputs/te_filter_survivors.csv and te_outputs/te_filter_downstream.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

WORK = config.TE_WORK
OUT = os.environ.get("TE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_outputs"))
CLASSES = ["OK", "Misalign", "NoSolder", "LowSolder", "Single"]
NG = CLASSES[1:]

z = np.load(os.path.join(WORK, "te_features.npz"), allow_pickle=True)
pool = pd.read_csv(os.path.join(OUT, "te_pool.csv"))
RES, label, kind, version = z["resnet_d4"], z["label"], z["kind"], z["version"]
names = np.array([os.path.basename(str(p)) for p in z["path"]])
bad = set(pool.name[pool.conflict | (pool.label == "TooMuch")])
real_ok = (kind == "real") & ~np.isin(names, list(bad))
train = real_ok & z["lora_train"].astype(bool)
held = real_ok & ~train
gen = kind == "gen"
tr, ho = np.flatnonzero(train), np.flatnonzero(held)

# ---- the judge: trained on training photos only (8 views each: original + flips/rotations) ----
Xj = RES[tr].reshape(-1, RES.shape[2])
yj = np.repeat(label[tr], RES.shape[1])
judge = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")).fit(Xj, yj)
gi = np.flatnonzero(gen & np.isin(label, NG))
proba = judge.predict_proba(RES[gi, 0])
cls_pos = {c: list(judge.classes_).index(c) for c in CLASSES}
p_intended = np.array([proba[k, cls_pos[label[i]]] for k, i in enumerate(gi)])
pred = judge.classes_[proba.argmax(1)]
keep = pred == label[gi]

rows = []
for v in ["v4", "v5", "v6", "v7", "all"]:
    for c in NG:
        m = (label[gi] == c) & ((version[gi] == v) if v != "all" else True)
        rows.append(dict(version=v, cls=c, made=int(m.sum()), kept=int((m & keep).sum())))
surv = pd.DataFrame(rows)
surv["kept_share"] = (surv.kept / surv.made).round(2)
print("=== fakes the judge accepts (judge trained on training photos only)")
print(surv.pivot(index="version", columns="cls", values="kept").to_string())
surv.to_csv(os.path.join(OUT, "te_filter_survivors.csv"), index=False)


def pick(rng, sel, c, n=15, top=False):
    """n fakes of class c from index set sel (into gi); top=True takes the most confident."""
    ids = sel[label[gi[sel]] == c]
    if top:
        ids = ids[np.argsort(-p_intended[ids])][:n]
    elif len(ids) > n:
        ids = rng.choice(ids, n, replace=False)
    return [(RES[gi[k], 0], c) for k in ids]


all_idx = np.arange(len(gi))
v6 = all_idx[version[gi] == "v6"]
v6_keep = all_idx[(version[gi] == "v6") & keep]
any_keep = all_idx[keep]
METHODS = ["A baseline", "C +15 flip/rotate", "+15 SDXL v6 (unfiltered)", "+15 SDXL v6, judge-filtered",
           "+15 SDXL v4-v7, judge-filtered", "+15 SDXL v4-v7, most confident", "+~10 real (ceiling)"]
REPEATS, BOOT = 50, 2000
Xtr, ytr = RES[tr, 0], label[tr]
correct = {m: np.zeros(len(ho)) for m in METHODS}
pred_ok = {m: np.zeros(len(ho)) for m in METHODS}
n_added = {m: [] for m in METHODS}


def fit(X, y):
    return make_pipeline(StandardScaler(), PCA(128, random_state=0),
                         LogisticRegression(C=0.5, max_iter=5000)).fit(X, y)


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
        add["+15 SDXL v6 (unfiltered)"] = sum((pick(rng, v6, c) for c in NG), [])
        add["+15 SDXL v6, judge-filtered"] = sum((pick(rng, v6_keep, c) for c in NG), [])
        add["+15 SDXL v4-v7, judge-filtered"] = sum((pick(rng, any_keep, c) for c in NG), [])
        add["+15 SDXL v4-v7, most confident"] = sum((pick(rng, all_idx, c, top=True) for c in NG), [])
        add["+~10 real (ceiling)"] = [(RES[ho[i], 0], label[ho[i]]) for c in NG
                                      for i in rng.permutation(ex[label[ho[ex]] == c])[:15]]
        Xte = RES[ho[te], 0]
        for m in METHODS:
            if m == "A baseline":
                model = fit(Xtr, ytr)
            else:
                Xa = np.array([a for a, _ in add[m]])
                ya = np.array([b for _, b in add[m]])
                n_added[m].append(len(ya))
                model = fit(np.vstack([Xtr, Xa]), np.concatenate([ytr, ya]))
            p = model.predict(Xte)
            correct[m][te] += (p == label[ho[te]])
            pred_ok[m][te] += (p == "OK")
    if (r + 1) % 10 == 0:
        print(f"  {r + 1}/{REPEATS} repeats", flush=True)
for m in METHODS:
    correct[m] /= REPEATS
    pred_ok[m] /= REPEATS

y_ho = label[ho]
cls_idx = {c: np.flatnonzero(y_ho == c) for c in CLASSES}
ng_idx = np.concatenate([cls_idx[c] for c in NG])


def macro(v, sel=None):
    return np.mean([v[cls_idx[c] if sel is None else sel[c]].mean() for c in CLASSES])


rng = np.random.default_rng(7)
boot_sel = [{c: rng.choice(cls_idx[c], len(cls_idx[c])) for c in CLASSES} for _ in range(BOOT)]
rows = []
for m in METHODS:
    est = macro(correct[m])
    d = np.array([macro(correct[m], s) - macro(correct["A baseline"], s) for s in boot_sel])
    rows.append(dict(method=m, images_added=int(np.mean(n_added[m])) if n_added[m] else 0,
                     macro_acc=est, gain_vs_A=est - macro(correct["A baseline"]),
                     gain_ci_lo=np.quantile(d, .025), gain_ci_hi=np.quantile(d, .975),
                     escape_rate=pred_ok[m][ng_idx].mean(), false_alarm=1 - pred_ok[m][cls_idx["OK"]].mean(),
                     **{f"acc_{c}": correct[m][cls_idx[c]].mean() for c in CLASSES}))
down = pd.DataFrame(rows)
print(f"\n=== detector test on {len(ho)} held-out real photos, {REPEATS} repeats, 95% bootstrap ranges")
print(down.round(3).to_string(index=False))
down.to_csv(os.path.join(OUT, "te_filter_downstream.csv"), index=False)
