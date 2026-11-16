"""Step 2b - score the cheap synthetic sets exactly like the SDXL sets, and rerun the
downstream comparison with everything side by side.

Usage: python te_cheap_eval.py features   (~3 min, caches te_outputs/feat_cheap.npz)
       python te_cheap_eval.py eval
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import config  # noqa: E402
import extract_features as E  # noqa: E402
from exp1_fid_reliability import kid  # noqa: E402
from te_features import load_rgb  # noqa: E402

OUT = os.environ.get("TE_OUT", os.path.join(HERE, "te_outputs"))
WORK = config.TE_WORK
os.makedirs(WORK, exist_ok=True)
CLASSES = ["OK", "Misalign", "NoSolder", "LowSolder", "Single"]
NG = CLASSES[1:]
CHEAP = ["rich_aug", "cutmix", "transplant"]
SDXL = ["v4", "v5", "v6", "v7"]


@torch.inference_mode()
def features():
    t = pd.read_csv(os.path.join(OUT, "te_cheap.csv"))
    inc, res = E.inception_model(), E.resnet_model()
    fi, fr = [], []
    for k in range(0, len(t), 16):
        ps = t.path[k:k + 16]
        fi.append(inc(torch.stack([load_rgb(p, 299) for p in ps]))[0].flatten(1).numpy())
        fr.append(res(torch.stack([(load_rgb(p, 224) - E.MEAN) / E.STD for p in ps])).numpy())
        if k % 160 == 0:
            print(" ", k, "/", len(t), flush=True)
    np.savez_compressed(os.path.join(WORK, "feat_cheap.npz"), inception=np.concatenate(fi),
                        resnet=np.concatenate(fr), label=t.label.values, version=t.version.values)
    print("saved")


def evaluate(repeats=50, boot=2000):
    z = np.load(os.path.join(WORK, "te_features.npz"), allow_pickle=True)
    c = np.load(os.path.join(WORK, "feat_cheap.npz"), allow_pickle=True)
    pool = pd.read_csv(os.path.join(OUT, "te_pool.csv"))
    # one table: real + SDXL (from Step 1) + cheap sets
    INC = np.vstack([z["inception"], c["inception"]]).astype(np.float64)
    RES0 = np.vstack([z["resnet_d4"][:, 0], c["resnet"]])
    RESD4 = z["resnet_d4"]                                   # flip/rotate views of the training photos
    label = np.concatenate([z["label"], c["label"]])
    version = np.concatenate([z["version"], c["version"]])
    n_real = len(pool)
    real_ok = np.zeros(len(label), bool)
    real_ok[:n_real] = (~pool.conflict.values) & (pool.label.values != "TooMuch")
    lora = np.zeros(len(label), bool)
    lora[:len(z["lora_train"])] = z["lora_train"].astype(bool)
    train, held = real_ok & lora, real_ok & ~lora
    sets = SDXL + CHEAP

    # ---- quality ----
    Xh, yh = RES0[held], label[held]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))
    cvp = np.empty(len(yh), dtype=object)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xh, yh):
        cvp[b] = clf.fit(Xh[a], yh[a]).predict(Xh[b])
    clf.fit(Xh, yh)
    rng = np.random.default_rng(0)
    RN = normalize(RES0)
    q = []
    for v in ["real"] + sets:
        for cl in NG:
            row = dict(set=v, cls=cl)
            ref = INC[held & (label == cl)]
            tr_idx = np.flatnonzero(train & (label == cl))
            n = min(10, len(tr_idx))
            if v == "real":
                row["recognised"] = (cvp[yh == cl] == cl).mean()
                src = tr_idx
                div_idx = np.flatnonzero(real_ok & (label == cl))
            else:
                src = np.flatnonzero((version == v) & (label == cl))
                row["recognised"] = (clf.predict(RES0[src]) == cl).mean()
                div_idx = src
            row["kid"] = np.median([kid(INC[rng.choice(src, n, replace=False)], ref) for _ in range(200)])
            sims = []
            for _ in range(100):
                s = rng.choice(div_idx, 15, replace=False)
                S = RN[s] @ RN[s].T
                sims.append(S[np.triu_indices(15, 1)].mean())
            row["within_class_similarity"] = np.mean(sims)
            q.append(row)
    q = pd.DataFrame(q)
    q.to_csv(os.path.join(OUT, "te_quality_all.csv"), index=False)
    print("=== quality (defect classes)")
    print(q.pivot(index="set", columns="cls", values="recognised").round(2).to_string())
    print(q.groupby("set")[["recognised", "kid", "within_class_similarity"]].mean().round(3).to_string())

    # ---- downstream (same protocol as te_analysis.py, section C) ----
    tr, ho = np.flatnonzero(train), np.flatnonzero(held)
    Xtr, ytr = RES0[tr], label[tr]
    methods = ["A baseline", "B class weights", "C +15 flip/rotate"] + [f"+15 {s}" for s in CHEAP] + \
              [f"+15 SDXL {v}" for v in SDXL] + ["+~10 real (ceiling)"]
    correct = {m: np.zeros(len(ho)) for m in methods}
    pred_ok = {m: np.zeros(len(ho)) for m in methods}

    def fit(X, y, weighted=False):
        return make_pipeline(StandardScaler(), PCA(128, random_state=0),
                             LogisticRegression(C=0.5, max_iter=5000,
                                                class_weight="balanced" if weighted else None)).fit(X, y)

    for r in range(repeats):
        rng = np.random.default_rng(1000 + r)
        half = np.zeros(len(ho), int)
        for cl in CLASSES:
            ii = np.flatnonzero(label[ho] == cl)
            half[rng.permutation(ii)[: len(ii) // 2]] = 1
        for fold in (0, 1):
            te, ex = np.flatnonzero(half == fold), np.flatnonzero(half != fold)
            add = {}
            aug = []
            for cl in NG:
                pairs = [(i, v) for i in tr[ytr == cl] for v in range(1, 8)]
                aug += [(RESD4[pairs[k][0], pairs[k][1]], cl) for k in rng.choice(len(pairs), 15, replace=False)]
            add["C +15 flip/rotate"] = aug
            for s in CHEAP:
                add[f"+15 {s}"] = [(RES0[i], cl) for cl in NG for i in
                                   rng.choice(np.flatnonzero((version == s) & (label == cl)), 15, replace=False)]
            for v in SDXL:
                add[f"+15 SDXL {v}"] = [(RES0[i], cl) for cl in NG for i in
                                        rng.choice(np.flatnonzero((version == v) & (label == cl)), 15, replace=False)]
            add["+~10 real (ceiling)"] = [(RES0[ho[i]], label[ho[i]]) for cl in NG
                                          for i in rng.permutation(ex[label[ho[ex]] == cl])[:15]]
            Xte = RES0[ho[te]]
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
                correct[m][te] += p == label[ho[te]]
                pred_ok[m][te] += p == "OK"
        if r % 10 == 9:
            print(f"  repeat {r + 1}/{repeats}", flush=True)
    for m in methods:
        correct[m] /= repeats
        pred_ok[m] /= repeats
    y_ho = label[ho]
    cls_idx = {cl: np.flatnonzero(y_ho == cl) for cl in CLASSES}
    macro = lambda v, sel=None: np.mean([v[cls_idx[cl] if sel is None else sel[cl]].mean() for cl in CLASSES])
    rng = np.random.default_rng(7)
    boots = [{cl: rng.choice(cls_idx[cl], len(cls_idx[cl])) for cl in CLASSES} for _ in range(boot)]
    ng_idx = np.concatenate([cls_idx[cl] for cl in NG])
    rows = []
    for m in methods:
        b = np.array([macro(correct[m], s) for s in boots])
        d = np.array([macro(correct[m], s) - macro(correct["A baseline"], s) for s in boots])
        dc = np.array([macro(correct[m], s) - macro(correct["C +15 flip/rotate"], s) for s in boots])
        rows.append(dict(method=m, macro_acc=macro(correct[m]), ci_lo=np.quantile(b, .025), ci_hi=np.quantile(b, .975),
                         gain_vs_A=macro(correct[m]) - macro(correct["A baseline"]),
                         gain_ci_lo=np.quantile(d, .025), gain_ci_hi=np.quantile(d, .975),
                         vs_flip=macro(correct[m]) - macro(correct["C +15 flip/rotate"]),
                         vs_flip_ci_lo=np.quantile(dc, .025), vs_flip_ci_hi=np.quantile(dc, .975),
                         escape_rate=pred_ok[m][ng_idx].mean(), false_alarm=1 - pred_ok[m][cls_idx["OK"]].mean(),
                         **{f"acc_{cl}": correct[m][cls_idx[cl]].mean() for cl in CLASSES}))
    down = pd.DataFrame(rows)
    down.to_csv(os.path.join(OUT, "te_downstream_all.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"=== downstream on {len(ho)} held-out photos, {repeats} repeats")
    print(down.round(3).to_string(index=False))


if __name__ == "__main__":
    features() if sys.argv[1] == "features" else evaluate()
