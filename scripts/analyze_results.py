"""Compare the three variants over all 5 folds, from runs/kvasir/<run>/per_image.csv (scripts/evaluate_runs.py).

Every image is a test image exactly once, so each model has 1000 paired per-image scores.
Confidence intervals use a cluster bootstrap over the near-duplicate groups in splits/kvasir_5fold.json
(images from the same group are resampled together), so look-alike images are not counted as independent.

Writes runs/kvasir/analysis/{summary.csv, per_fold.csv, pairwise.csv, by_size.csv, worst_cases.png}.
Usage: python scripts/analyze_results.py [--runs runs/kvasir] [--boot 5000]
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.stats import wilcoxon

MODELS = {"unet": "U-Net", "attention_unet": "Attention U-Net", "unetpp": "UNet++"}
METRICS = {"dice": "Dice", "iou": "IoU", "nsd": "NSD", "hd95_pct": "HD95 (% diag)"}


def cluster_bootstrap(values, groups, n_boot, rng, stat=np.mean):
    """95% CI of stat(values), resampling whole groups with replacement."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        stats.append(stat(values[idx]))
    return np.percentile(stats, [2.5, 97.5])


def cases_montage(runs, d, root, names, out, tile=150):
    """One row per image: image | ground truth | each model's prediction (Dice in red, polyp area in yellow)."""
    cols = 2 + len(MODELS)
    sheet = Image.new("RGB", (tile * cols, tile * len(names) + 20), "white")
    draw = ImageDraw.Draw(sheet)
    for c, title in enumerate(["image", "ground truth", *MODELS.values()]):
        draw.text((c * tile + 4, 4), title, fill="black")
    for r, name in enumerate(names):
        y = 20 + r * tile
        tiles = [Image.open(root / "images" / name).convert("RGB"), Image.open(root / "masks" / name).convert("RGB")]
        tiles += [Image.open(runs / f"{m}_double_conv" / "preds" / name.replace(".jpg", ".png")).convert("RGB") for m in MODELS]
        for c, im in enumerate(tiles):
            sheet.paste(im.resize((tile, tile)), (c * tile, y))
        for c, m in enumerate(MODELS):
            draw.text(((2 + c) * tile + 4, y + 4), f"{d[m].dice[name]:.2f}", fill="red")
        draw.text((4, y + 4), f"{d['unet'].polyp_fraction[name]:.1%}", fill="yellow")
    sheet.save(out)


def worst_cases(d, n=8):
    """Images with the lowest Dice averaged over the three models."""
    return list((sum(d[m].dice for m in MODELS) / len(MODELS)).sort_values().index[:n])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/kvasir")
    ap.add_argument("--splits", default="splits/kvasir_5fold.json")
    ap.add_argument("--data-root", default="Kvasir-SEG")
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.runs) / "analysis"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    d = {m: pd.read_csv(runs / f"{m}_double_conv" / "per_image.csv").set_index("image").sort_index() for m in MODELS}
    names = d["unet"].index
    assert all((d[m].index == names).all() for m in MODELS) and len(names) == 1000
    groups = np.array([json.loads(Path(args.splits).read_text())["groups"][n] for n in names])

    # Pooled summary with cluster-bootstrap CIs
    rows = []
    for m, label in MODELS.items():
        row = {"model": label}
        for k, name in METRICS.items():
            v = d[m][k].values
            centre = np.median(v) if k == "hd95_pct" else v.mean()
            lo, hi = cluster_bootstrap(v, groups, args.boot, rng, np.median if k == "hd95_pct" else np.mean)
            row[name] = f"{centre:.3f} [{lo:.3f}, {hi:.3f}]"
        row["median Dice"] = round(np.median(d[m].dice), 3)
        row["Dice < 0.5"] = int((d[m].dice < 0.5).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    print("Pooled over 1000 test images (mean, HD95 median; 95% cluster-bootstrap CI):")
    print(summary.to_string(index=False))

    # Per fold: does any model win consistently?
    per_fold = pd.DataFrame({MODELS[m]: d[m].groupby("fold").dice.mean() for m in MODELS})
    per_fold.loc["mean"], per_fold.loc["sd"] = per_fold.mean(), per_fold.iloc[:5].std()
    per_fold.to_csv(out / "per_fold.csv")
    print("\nMean test Dice per fold:")
    print(per_fold.round(4).to_string())
    print("winner per fold:", per_fold.iloc[:5].idxmax(axis=1).tolist())

    # Paired comparisons
    rows = []
    for a, b in combinations(MODELS, 2):
        for k in ("dice", "nsd"):
            diff = (d[a][k] - d[b][k]).values
            lo, hi = cluster_bootstrap(diff, groups, args.boot, rng)
            rows.append({"comparison": f"{MODELS[a]} - {MODELS[b]}", "metric": METRICS[k],
                         "mean diff": round(diff.mean(), 4), "95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
                         "Wilcoxon p": round(wilcoxon(d[a][k], d[b][k]).pvalue, 4),
                         "folds a > b": int((d[a].groupby("fold")[k].mean() > d[b].groupby("fold")[k].mean()).sum())})
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(out / "pairwise.csv", index=False)
    print(f"\nPaired differences (3 comparisons x 2 metrics; Bonferroni threshold p < {0.05 / 6:.4f}):")
    print(pairwise.to_string(index=False))

    # By polyp size
    size = pd.qcut(d["unet"].polyp_fraction, 3, labels=["small", "medium", "large"])
    by_size = pd.DataFrame({MODELS[m]: d[m].dice.groupby(size.values, observed=True).mean() for m in MODELS})
    by_size["n"] = size.value_counts()
    by_size["polyp area range"] = d["unet"].polyp_fraction.groupby(size.values, observed=True).agg(
        lambda s: f"{s.min():.1%}-{s.max():.1%}")
    by_size.to_csv(out / "by_size.csv")
    print("\nMean Dice by polyp-size tercile:")
    print(by_size.round(3).to_string())

    # Failures: shared by all models (hard images / label issues) vs model-specific
    fail = pd.DataFrame({m: d[m].dice < 0.5 for m in MODELS})
    print(f"\nDice < 0.5: all three models fail on {fail.all(axis=1).sum()} images; "
          f"at least one fails on {fail.any(axis=1).sum()}")
    empty = pd.DataFrame({m: d[m].pred_fraction == 0 for m in MODELS})
    print("empty predictions:", empty.sum().to_dict())
    worst = worst_cases(d)
    cases_montage(runs, d, Path(args.data_root), worst, out / "worst_cases.png")
    print(f"worst-case montage: {out / 'worst_cases.png'}  ({', '.join(w[:12] for w in worst)})")


if __name__ == "__main__":
    main()
