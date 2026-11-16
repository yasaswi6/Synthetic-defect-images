"""Experiment 2 - Re-create the AITK / previous-UROP evaluation protocol on public data.

Protocol (same as the previous UROP report, section 3):
    train: 100 OK + 10 real images per defect (NG) class
    test : 15 images per class
and compare ways of "balancing" the 10-image defect classes up to 25:

    A  baseline           - train on the imbalanced set as-is
    B  class-weighted     - same data, loss re-weighted by class frequency (free)
    C  flip/rotate x15    - +15 classical-augmentation copies per NG class
                            (stand-in for the 15 synthetic images per class)
    D  +15 real (ceiling) - +15 extra REAL images per NG class; best case for what
                            15 perfect synthetic images could ever add

Classifier: frozen ImageNet ResNet-50 features + logistic regression (CPU, seconds).
Every configuration is repeated over many random draws. Each draw is also scored on a
LARGE held-out set (all unused images) so we can see how far the 15-image test
estimate is from the "true" accuracy, and how often a 15-image test picks the wrong
winner between two methods.
"""
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FEAT_DIR, FIG_DIR, RES_DIR, wilson_ci

N_OK_TRAIN, N_NG_TRAIN, N_TEST, N_EXTRA = 100, 10, 15, 15
SEEDS = 50
METHODS = ["A baseline", "B class-weighted", "C flip/rotate x15", "D +15 real (ceiling)"]


def fit_predict(Xtr, ytr, Xte, weighted=False):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.5, max_iter=3000,
                                           class_weight="balanced" if weighted else None))
    clf.fit(Xtr, ytr)
    return [clf.predict(x) if len(x) else np.array([]) for x in Xte]


def scores(y, p, classes):
    out = {}
    for c in classes:
        m = y == c
        out[f"acc_{c}"] = (p[m] == c).mean() if m.any() else np.nan
    ng = y != "OK"
    out["macro_acc"] = np.nanmean([out[f"acc_{c}"] for c in classes])
    out["escape_rate"] = (p[ng] == "OK").mean()        # defect passed as OK
    out["false_alarm"] = (p[~ng] != "OK").mean()       # OK flagged as defect
    return out


def run(name):
    z = np.load(os.path.join(FEAT_DIR, f"{name}.npz"), allow_pickle=True)
    F, y, has_aug = z["resnet_d4"], z["labels"], z["has_aug"]
    classes = sorted(np.unique(y), key=lambda c: (c != "OK", c))
    ng_classes = [c for c in classes if c != "OK"]
    rows = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        used = np.zeros(len(y), bool)
        tr, te, extra = [], [], []
        ok = rng.permutation(np.flatnonzero(y == "OK"))
        tr += list(ok[:N_OK_TRAIN]); te += list(ok[N_OK_TRAIN:N_OK_TRAIN + N_TEST])
        for c in ng_classes:
            cand = rng.permutation(np.flatnonzero((y == c) & has_aug))
            tr_c = cand[:N_NG_TRAIN]
            rest = rng.permutation(np.setdiff1d(np.flatnonzero(y == c), tr_c))
            tr += list(tr_c); te += list(rest[:N_TEST]); extra += list(rest[N_TEST:N_TEST + N_EXTRA])
        tr, te, extra = map(np.array, (tr, te, extra))
        used[tr] = used[te] = True
        used[extra] = True
        big = np.flatnonzero(~used)                     # large held-out set

        Xtr, ytr = F[tr, 0], y[tr]
        # classical augmentation: 15 random (image, view) pairs per NG class, views 1..7
        Xaug, yaug = [], []
        for c in ng_classes:
            ids = tr[y[tr] == c]
            pairs = [(i, v) for i in ids for v in range(1, 8)]
            for k in rng.choice(len(pairs), N_EXTRA, replace=False):
                i, v = pairs[k]
                Xaug.append(F[i, v]); yaug.append(c)
        Xaug, yaug = np.array(Xaug), np.array(yaug)

        sets = [F[te, 0], F[big, 0]]
        preds = {
            METHODS[0]: fit_predict(Xtr, ytr, sets),
            METHODS[1]: fit_predict(Xtr, ytr, sets, weighted=True),
            METHODS[2]: fit_predict(np.vstack([Xtr, Xaug]), np.concatenate([ytr, yaug]), sets),
            METHODS[3]: fit_predict(np.vstack([Xtr, F[extra, 0]]), np.concatenate([ytr, y[extra]]), sets),
        }
        for m, (p_small, p_big) in preds.items():
            r = dict(dataset=name, seed=seed, method=m, n_big=len(big))
            r.update({f"small_{k}": v for k, v in scores(y[te], p_small, classes).items()})
            r.update({f"big_{k}": v for k, v in scores(y[big], p_big, classes).items()})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES_DIR, f"exp2_protocol_{name}.csv"), index=False)
    return df, classes


def report(df, classes, name):
    g = df.groupby("method")
    cols = ["small_macro_acc", "big_macro_acc", "big_escape_rate", "big_false_alarm"]
    summ = g[cols].agg(["mean", "std"]).round(3)
    print(f"\n=== {name}: mean (sd) over {SEEDS} random draws ===")
    print(summ.to_string())
    summ.to_csv(os.path.join(RES_DIR, f"exp2_summary_{name}.csv"))

    # (1) how far is the 15-image estimate from the large-test estimate, per class?
    errs = []
    for c in classes:
        e = (df[f"small_acc_{c}"] - df[f"big_acc_{c}"]).abs()
        errs.append(dict(dataset=name, cls=c, mean_abs_err=e.mean(), p90_abs_err=e.quantile(.9),
                         max_abs_err=e.max()))
    errs = pd.DataFrame(errs)
    print("\n|15-image per-class accuracy - large-test accuracy|")
    print(errs.round(3).to_string(index=False))
    errs.to_csv(os.path.join(RES_DIR, f"exp2_test_noise_{name}.csv"), index=False)

    # (2) how often does the 15-image test pick a different winner than the big test?
    flips = []
    piv_s = df.pivot(index="seed", columns="method", values="small_macro_acc")
    piv_b = df.pivot(index="seed", columns="method", values="big_macro_acc")
    for m in METHODS[1:]:
        ds, db = piv_s[m] - piv_s[METHODS[0]], piv_b[m] - piv_b[METHODS[0]]
        flips.append(dict(dataset=name, method=m, big_gain_mean=db.mean(), small_gain_mean=ds.mean(),
                          small_gain_sd=ds.std(),
                          p_wrong_sign=((np.sign(ds) != np.sign(db)) & (db.abs() > 0)).mean(),
                          p_small_says_no_gain=(ds <= 0).mean()))
    flips = pd.DataFrame(flips)
    print("\nGain over baseline: large test vs 15-image test")
    print(flips.round(3).to_string(index=False))
    flips.to_csv(os.path.join(RES_DIR, f"exp2_winner_flips_{name}.csv"), index=False)

    # figure: per-class accuracy, 15-image test vs large test, for baseline and ceiling
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, m in zip(axes, [METHODS[0], METHODS[3]]):
        d = df[df.method == m]
        pos = np.arange(len(classes))
        ax.boxplot([d[f"small_acc_{c}"] for c in classes], positions=pos - .18, widths=.3,
                   patch_artist=True, boxprops=dict(facecolor="#9db9ea"), medianprops=dict(color="k"))
        ax.boxplot([d[f"big_acc_{c}"].dropna() for c in classes], positions=pos + .18, widths=.3,
                   patch_artist=True, boxprops=dict(facecolor="#f2b48f"), medianprops=dict(color="k"))
        ax.set_xticks(pos)
        ax.set_xticklabels(classes)
        ax.set_title(m)
        ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("per-class accuracy")
    axes[0].plot([], [], "s", color="#9db9ea", label="15-image test")
    axes[0].plot([], [], "s", color="#f2b48f", label="large held-out test")
    fig.legend(frameon=False, loc="upper right", ncol=2)
    fig.suptitle(f"{name.upper()}: same classifiers, two test sets ({SEEDS} random draws)", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG_DIR, f"exp2_protocol_{name}.png"), dpi=150)
    return summ, errs, flips


if __name__ == "__main__":
    lo, hi = wilson_ci(12, 15)
    print(f"Wilson 95% CI for 12/15 = 80%: [{lo:.2f}, {hi:.2f}]")
    for name in ["riawelc", "mtd"]:
        df, classes = run(name)
        report(df, classes, name)
