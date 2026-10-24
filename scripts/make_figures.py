"""Figures for the README, written to assets/ (tracked in git, unlike runs/).

  assets/val_dice_curves.png       validation Dice per epoch, mean over the 5 folds, one line per model
  assets/representative_cases.png  images at the 10th-90th percentiles of mean Dice: what typical output looks like
  assets/worst_cases.png           the 8 images with the lowest mean Dice

Usage: python scripts/make_figures.py   (after scripts/evaluate_runs.py)
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_results import MODELS, cases_montage, worst_cases  # noqa: E402

RUNS, ROOT, ASSETS = Path("runs/kvasir"), Path("Kvasir-SEG"), Path("assets")
# First three slots of the validated reference categorical palette (fixed order: blue, orange, aqua)
COLORS = {"unet": "#2a78d6", "attention_unet": "#eb6834", "unetpp": "#1baf7a"}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def val_dice_curves(out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for m, label in MODELS.items():
        h = pd.concat(pd.read_csv(f).assign(fold=f.parent.name) for f in (RUNS / f"{m}_double_conv").glob("fold*/history.csv"))
        mean = h.groupby("epoch").val_dice.mean()
        ax.plot(mean.index, mean.values, color=COLORS[m], lw=2, label=label)
        ax.annotate(f"{label}  {mean.iloc[-10:].mean():.3f}", (mean.index[-1], mean.values[-1]),
                    xytext=(6, {"unet": 7, "attention_unet": -9, "unetpp": -1}[m]), textcoords="offset points",
                    color=TEXT, fontsize=8, va="center")
    ax.set_xlim(1, 100)
    ax.set_ylim(0.6, 0.9)
    ax.set_xlabel("epoch", color=TEXT_2, fontsize=9)
    ax.set_ylabel("validation Dice (mean of 5 folds)", color=TEXT_2, fontsize=9)
    ax.set_title("All three variants converge to the same validation Dice", color=TEXT, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT)
    fig.subplots_adjust(right=0.8)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    d = {m: pd.read_csv(RUNS / f"{m}_double_conv" / "per_image.csv").set_index("image").sort_index() for m in MODELS}
    val_dice_curves(ASSETS / "val_dice_curves.png")

    mean_dice = (sum(d[m].dice for m in MODELS) / len(MODELS)).sort_values()
    picks = [mean_dice.index[int(q * (len(mean_dice) - 1))] for q in np.linspace(0.1, 0.9, 6)]
    cases_montage(RUNS, d, ROOT, picks, ASSETS / "representative_cases.png", tile=130)
    cases_montage(RUNS, d, ROOT, worst_cases(d), ASSETS / "worst_cases.png", tile=130)
    print("wrote", *sorted(str(p) for p in ASSETS.glob("*.png")))


if __name__ == "__main__":
    main()
