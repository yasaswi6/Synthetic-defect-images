"""Experiment 1 - How trustworthy is per-class FID at the sample sizes we use (10-30 images)?

For each class we draw two *disjoint* random sets of n REAL images of the SAME class and
compute FID / KID between them. Since both sets are real and come from the same class, a
perfect metric would give ~0. Whatever FID we get is the "noise floor" at that n: a
generator cannot be judged better than this floor, and differences smaller than the
spread across draws are noise.

We also compute FID between sets from two DIFFERENT classes, and ask how often the
same-class pair scores lower than the different-class pair (i.e. does FID even tell
defect classes apart at this n?).

FID is computed exactly with a low-rank formula (n << 2048), so no sqrtm is needed:
    tr sqrt(S1 S2) = || X1c X2c^T ||_*  / sqrt((n1-1)(n2-1))
KID uses the unbiased MMD^2 estimator with the cubic polynomial kernel (Binkowski 2018).
"""
import itertools
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FEAT_DIR, FIG_DIR, RES_DIR

NS = [10, 15, 25, 50, 100, 150]
REPEATS = 30


def fid(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    x1, x2 = a - mu1, b - mu2
    n1, n2 = len(a), len(b)
    tr1 = (x1 ** 2).sum() / (n1 - 1)
    tr2 = (x2 ** 2).sum() / (n2 - 1)
    nuc = np.linalg.svd(x1 @ x2.T, compute_uv=False).sum() / np.sqrt((n1 - 1) * (n2 - 1))
    return float(((mu1 - mu2) ** 2).sum() + tr1 + tr2 - 2 * nuc)


def kid(a, b):
    d = a.shape[1]
    k = lambda x, y: (x @ y.T / d + 1) ** 3
    kxx, kyy, kxy = k(a, a), k(b, b), k(a, b)
    m, n = len(a), len(b)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                 + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())


def run(name):
    z = np.load(os.path.join(FEAT_DIR, f"{name}.npz"), allow_pickle=True)
    X, y = z["inception"].astype(np.float64), z["labels"]
    classes = list(np.unique(y))
    rng = np.random.default_rng(1)
    rows = []
    for n in NS:
        for c in classes:
            idx = np.flatnonzero(y == c)
            if len(idx) < 2 * n:
                continue
            for r in range(REPEATS):
                p = rng.permutation(idx)
                a, b = X[p[:n]], X[p[n:2 * n]]
                rows.append(dict(dataset=name, n=n, kind="same", c1=c, c2=c, rep=r, fid=fid(a, b), kid=kid(a, b)))
        for c1, c2 in itertools.combinations(classes, 2):
            i1, i2 = np.flatnonzero(y == c1), np.flatnonzero(y == c2)
            if len(i1) < 2 * n or len(i2) < 2 * n:
                continue
            for r in range(REPEATS):
                a, b = X[rng.choice(i1, n, replace=False)], X[rng.choice(i2, n, replace=False)]
                rows.append(dict(dataset=name, n=n, kind="diff", c1=c1, c2=c2, rep=r, fid=fid(a, b), kid=kid(a, b)))
        print(name, "n =", n, "done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES_DIR, f"exp1_fid_{name}.csv"), index=False)
    return df


def summarise(df):
    out = []
    for (ds, n), g in df.groupby(["dataset", "n"]):
        s, d = g[g.kind == "same"], g[g.kind == "diff"]
        # probability that a random same-class pair scores LOWER (better) than a random diff-class pair
        if len(d):
            ps = (s.fid.values[:, None] < d.fid.values[None, :]).mean()
            pk = (s.kid.values[:, None] < d.kid.values[None, :]).mean()
        else:
            ps = pk = np.nan
        out.append(dict(dataset=ds, n=n,
                        fid_same_mean=s.fid.mean(), fid_same_sd=s.fid.std(),
                        fid_same_p5=s.fid.quantile(.05), fid_same_p95=s.fid.quantile(.95),
                        fid_diff_mean=d.fid.mean() if len(d) else np.nan,
                        kid_same_mean=s.kid.mean(), kid_same_sd=s.kid.std(),
                        kid_diff_mean=d.kid.mean() if len(d) else np.nan,
                        p_fid_separates=ps, p_kid_separates=pk))
    return pd.DataFrame(out)


def plot(df, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, lab in [(axes[0], "fid", "FID"), (axes[1], "kid", "KID")]:
        for kind, colour, txt in [("same", "#2a6fdb", "same class (real vs real)"),
                                  ("diff", "#e0662d", "different classes")]:
            g = df[(df.dataset == name) & (df.kind == kind)].groupby("n")[metric]
            m, lo, hi = g.mean(), g.quantile(.05), g.quantile(.95)
            ax.plot(m.index, m.values, "o-", color=colour, label=txt)
            ax.fill_between(m.index, lo.values, hi.values, color=colour, alpha=.18)
        ax.set_xscale("log")
        ax.set_xticks(NS)
        ax.set_xticklabels(NS)
        ax.set_xlabel("images per set (n)")
        ax.set_ylabel(lab)
        ax.grid(alpha=.3)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{name.upper()}: metric between two sets of real images (mean, 5-95% band)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"exp1_fid_{name}.png"), dpi=150)


def null_difference(name, repeats=200):
    """Two 'generators' that are equally good (both = fresh real images of the class) are scored
    against the SAME real reference set, as one would compare two AITK runs. The spread of
    FID(G1, R) - FID(G2, R) is how big a difference can appear by chance alone."""
    z = np.load(os.path.join(FEAT_DIR, f"{name}.npz"), allow_pickle=True)
    X, y = z["inception"].astype(np.float64), z["labels"]
    rng = np.random.default_rng(2)
    rows = []
    for n in NS:
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            if len(idx) < 3 * n:
                continue
            for r in range(repeats):
                p = rng.permutation(idx)
                R, g1, g2 = X[p[:n]], X[p[n:2 * n]], X[p[2 * n:3 * n]]
                rows.append(dict(dataset=name, n=n, cls=c, rep=r,
                                 fid_diff=fid(g1, R) - fid(g2, R), kid_diff=kid(g1, R) - kid(g2, R)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES_DIR, f"exp1_null_diff_{name}.csv"), index=False)
    s = df.groupby("n").agg(fid_abs_diff_median=("fid_diff", lambda v: np.median(np.abs(v))),
                            fid_abs_diff_p95=("fid_diff", lambda v: np.quantile(np.abs(v), .95)),
                            p_fid_diff_ge_27=("fid_diff", lambda v: (np.abs(v) >= 27.4).mean()),
                            kid_abs_diff_p95=("kid_diff", lambda v: np.quantile(np.abs(v), .95)),
                            classes=("cls", "nunique")).reset_index()
    s.insert(0, "dataset", name)
    return s


if __name__ == "__main__":
    frames = []
    for name in ["riawelc", "mtd"]:
        df = run(name)
        plot(df, name)
        frames.append(df)
    summ = summarise(pd.concat(frames))
    summ.to_csv(os.path.join(RES_DIR, "exp1_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    per_class = pd.concat(frames).query("kind == 'same'").groupby(["dataset", "n", "c1"]).fid.agg(["mean", "std"])
    per_class.round(1).to_csv(os.path.join(RES_DIR, "exp1_same_class_per_class.csv"))
    print(per_class.round(1).unstack("c1").to_string())
    nd = pd.concat([null_difference(n) for n in ["riawelc", "mtd"]])
    nd.to_csv(os.path.join(RES_DIR, "exp1_null_difference.csv"), index=False)
    print("\nFID difference between two equally good generators (same reference set):")
    print(nd.round(3).to_string(index=False))
