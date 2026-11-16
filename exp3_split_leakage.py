"""Experiment 3 - Does splitting by image instead of by source part inflate accuracy?

RIAWELC patches are crops of a few dozen weld radiographs ("films", e.g. RRT-40R).
We compare two ways of making a 50/50 train/test split of the same images:

    random  - patches split at random (what a naive split does)
    grouped - whole films go to either train or test (no film in both)

Same features (frozen ResNet-50) and classifiers (logistic regression, 1-nearest-neighbour).
A big gap means the random split lets the model "recognise the film" instead of the defect,
which is exactly the risk with several photos of the same TE part, or with synthetic
images generated from a LoRA that saw the test images.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from common import FEAT_DIR, RES_DIR

REPEATS = 20


def grouped_split(y, g, rng, min_per_class=20):
    films = np.unique(g)
    for _ in range(10000):
        test_films = rng.choice(films, len(films) // 2, replace=False)
        te = np.isin(g, test_films)
        ok = all(((y == c) & te).sum() >= min_per_class and ((y == c) & ~te).sum() >= min_per_class
                 for c in np.unique(y))
        if ok:
            return ~te, te
    raise RuntimeError("no valid grouped split")


def evaluate(Xtr, ytr, Xte, yte):
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    nn = KNeighborsClassifier(1, metric="cosine").fit(Xtr, ytr)
    return balanced_accuracy_score(yte, lr.predict(Xte)), balanced_accuracy_score(yte, nn.predict(Xte))


def main():
    z = np.load(os.path.join(FEAT_DIR, "riawelc.npz"), allow_pickle=True)
    X, y, g = z["resnet_d4"][:, 0], z["labels"], z["groups"]
    print("films:", len(np.unique(g)), "images:", len(y))
    rows = []
    for r in range(REPEATS):
        rng = np.random.default_rng(100 + r)
        tr_g, te_g = grouped_split(y, g, rng)
        # random split with the same per-class train/test sizes as the grouped one
        tr_r = np.zeros(len(y), bool)
        for c in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == c))
            tr_r[idx[:(tr_g & (y == c)).sum()]] = True
        te_r = ~tr_r
        for split, tr, te in [("random", tr_r, te_r), ("grouped", tr_g, te_g)]:
            lr, nn = evaluate(X[tr], y[tr], X[te], y[te])
            # near-duplicate check: cosine similarity of each test image to its nearest train image
            Xn = normalize(X)
            sim = (Xn[te] @ Xn[tr].T).max(1)
            rows.append(dict(rep=r, split=split, logreg_bal_acc=lr, nn1_bal_acc=nn,
                             nn_sim_median=np.median(sim), n_train=tr.sum(), n_test=te.sum()))
        print(f"rep {r}: random LR {rows[-2]['logreg_bal_acc']:.3f}  grouped LR {rows[-1]['logreg_bal_acc']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES_DIR, "exp3_leakage_riawelc.csv"), index=False)
    summ = df.groupby("split")[["logreg_bal_acc", "nn1_bal_acc", "nn_sim_median"]].agg(["mean", "std"]).round(3)
    print(summ.to_string())
    summ.to_csv(os.path.join(RES_DIR, "exp3_summary.csv"))


if __name__ == "__main__":
    main()
