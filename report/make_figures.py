"""Make the report's charts from the result tables in this repository.

  python report/make_figures.py        (needs pandas + matplotlib; writes report/figures/*.png)

Every chart is drawn from a CSV that the experiment scripts wrote, so the figures can be
re-made after re-running the experiments. No photos are used.
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

# colours: one per kind of method, kept the same in every chart
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
KIND = {"Free tricks": "#2a78d6", "SDXL fakes": "#eb6834", "More real photos": "#1baf7a", "Mask paste": "#eda100"}
BLUES = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Inter"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "semibold",
})


def pts(x):
    return 100 * x


def dot_ci(ax, rows, xlim):
    """rows: list of (label, kind, gain, lo, hi), drawn top to bottom."""
    for i, (label, kind, g, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        c = KIND[kind]
        ax.plot([pts(lo), pts(hi)], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(pts(g), y, "o", ms=8, color=c, mec=SURFACE, mew=2, zorder=3)
        ax.text(pts(hi) + 0.6, y, f"{pts(g):+.1f}", va="center", ha="left", color=INK, fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def legend(fig, kinds, y=0.005):
    handles = [plt.Line2D([], [], marker="o", ls="-", lw=2, ms=7, color=KIND[k], mec=SURFACE, mew=1.5) for k in kinds]
    fig.legend(handles, kinds, loc="lower center", ncol=len(kinds), frameon=False, bbox_to_anchor=(0.5, y))


# ---------- Figure 1: the main TE result ----------
d = pd.read_csv(os.path.join(ROOT, "te", "te_outputs", "te_downstream_all.csv")).set_index("method")
names = {
    "+~10 real (ceiling)": ("~10 more real photos per class", "More real photos"),
    "+15 rich_aug": ("Stronger augmentation", "Free tricks"),
    "B class weights": ("Class weights", "Free tricks"),
    "C +15 flip/rotate": ("Flip / rotate copies", "Free tricks"),
    "+15 transplant": ("Joint transplant", "Free tricks"),
    "+15 cutmix": ("Cut-and-mix", "Free tricks"),
    "+15 SDXL v4": ("SDXL v4", "SDXL fakes"),
    "+15 SDXL v5": ("SDXL v5", "SDXL fakes"),
    "+15 SDXL v6 ": None,
    "+15 SDXL v6": ("SDXL v6 (adopted last semester)", "SDXL fakes"),
    "+15 SDXL v7": ("SDXL v7", "SDXL fakes"),
}
rows = [(names[m][0], names[m][1], d.loc[m, "gain_vs_A"], d.loc[m, "gain_ci_lo"], d.loc[m, "gain_ci_hi"])
        for m in names if names[m] and m in d.index]
fig, ax = plt.subplots(figsize=(9, 5.2))
dot_ci(ax, rows, (-6, 20))
ax.set_xlabel("gain in macro accuracy over the baseline (percentage points)")
base = pts(d.loc["A baseline", "macro_acc"])
fig.suptitle(f"TE photos: what helps a detector trained on ~10 real photos per defect (baseline {base:.1f}%)",
             x=0.02, ha="left", fontsize=12, fontweight="semibold")
ax.set_title("436 held-out real photos, 50 repeats; dot = mean gain, line = 95% bootstrap interval",
             loc="left", fontsize=9.5, fontweight="normal", color=INK2)
legend(fig, ["More real photos", "Free tricks", "SDXL fakes"])
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(OUT, "fig1_te_main_result.png"), dpi=200)
plt.close(fig)

# ---------- Figure 2: are the fakes recognised as their defect? ----------
f = pd.read_csv(os.path.join(ROOT, "te", "te_outputs", "te_class_fidelity.csv")).set_index("images")
f.index = ["Real photos (held out)", "SDXL v4", "SDXL v5", "SDXL v6", "SDXL v7"]
cols = ["Misalign", "NoSolder", "LowSolder", "Single", "OK"]
f = f[cols]
fig, ax = plt.subplots(figsize=(8, 3.8))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUES)
ax.imshow(f.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(f.shape[0]):
    for j in range(f.shape[1]):
        v = f.values[i, j]
        ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=10.5,
                color="white" if v > 0.55 else INK)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(["Misalign", "No solder", "Low solder", "Single", "OK"])
ax.set_yticks(range(f.shape[0]))
ax.set_yticklabels(f.index)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, f.shape[0])], minor=True)
ax.grid(which="minor", color=SURFACE, lw=3)
ax.tick_params(which="minor", length=0)
fig.suptitle("Share of images a real-photo classifier recognises as the intended class",
             x=0.02, ha="left", fontsize=12, fontweight="semibold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_te_recognised.png"), dpi=200)
plt.close(fig)

# ---------- Figure 3: how much FID moves by chance ----------
n = pd.read_csv(os.path.join(ROOT, "outputs", "results", "exp1_null_difference.csv"))
n = n[n.n <= 50]
fig, ax = plt.subplots(figsize=(8, 4.2))
series = [("riawelc", "RIAWELC (weld X-rays)", KIND["Free tricks"]), ("mtd", "Magnetic Tile", KIND["SDXL fakes"])]
for key, label, c in series:
    s = n[n.dataset == key].sort_values("n")
    ax.plot(s.n, s.fid_abs_diff_p95, "-o", color=c, lw=2, ms=8, mec=SURFACE, mew=2, label=label)
    last = s.iloc[0]
    ax.text(last.n - 1.2, last.fid_abs_diff_p95, f"{last.fid_abs_diff_p95:.0f}", ha="right", va="center", fontsize=10)
ax.axhline(27.5, color=INK2, lw=1, ls="--")
ax.text(50, 28.8, "last semester's FID improvement for Single (168.6 → 141.1)", ha="right", va="bottom",
        fontsize=9.5, color=INK2)
ax.set_xticks([10, 15, 25, 50])
ax.set_xlim(5, 52)
ax.set_ylim(0, 64)
ax.set_xlabel("images per set")
ax.set_ylabel("FID gap, 95th percentile")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(frameon=False, loc="upper right")
fig.suptitle("FID gap between two equally good image sets, by chance alone",
             x=0.02, ha="left", fontsize=12, fontweight="semibold")
ax.set_title("5% of the time two sets of the same real class differ by more than this",
             loc="left", fontsize=9.5, fontweight="normal", color=INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_fid_noise.png"), dpi=200)
plt.close(fig)

# ---------- Figure 4: the same test on MVTec AD ----------
mv_names = {
    "+real defects (ceiling)": ("More real defect photos", "More real photos"),
    "+15 mask paste": ("Mask paste (uses defect outline)", "Mask paste"),
    "+15 stronger augmentation": ("Stronger augmentation", "Free tricks"),
    "+15 cut-and-mix": ("Cut-and-mix", "Free tricks"),
    "C +15 flip/rotate": ("Flip / rotate copies", "Free tricks"),
    "B class weights": ("Class weights", "Free tricks"),
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, cat in zip(axes, ["transistor", "cable"]):
    m = pd.read_csv(os.path.join(ROOT, "bench", "bench_outputs", f"mvtec_{cat}_downstream.csv")).set_index("method")
    rows = [(mv_names[k][0], mv_names[k][1], m.loc[k, "gain_vs_A"], m.loc[k, "gain_ci_lo"], m.loc[k, "gain_ci_hi"])
            for k in mv_names]
    dot_ci(ax, rows, (-2, 31))
    ax.set_title(f"{cat} (baseline {pts(m.loc['A baseline', 'macro_acc']):.1f}%)", loc="left", fontsize=11)
    ax.set_xlabel("gain over baseline (percentage points)")
fig.suptitle("MVTec AD: the same test on a public benchmark (5 splits, 95% bootstrap intervals)",
             x=0.02, ha="left", fontsize=12, fontweight="semibold")
legend(fig, ["More real photos", "Mask paste", "Free tricks"])
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(OUT, "fig4_mvtec.png"), dpi=200)
plt.close(fig)

print("wrote", sorted(os.listdir(OUT)))
