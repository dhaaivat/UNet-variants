"""Re-evaluate every trained fold on its test images at original resolution, with boundary metrics.

For each run in runs/kvasir/<run>/fold<k>/ this loads best.pth, predicts that fold's test images,
saves predicted masks to runs/kvasir/<run>/preds/<image>.png and writes runs/kvasir/<run>/per_image.csv with
Dice, IoU, HD95 and NSD for all 1000 images (each image is tested exactly once, by its fold's model).

Images differ in size (332-1920 px wide) and have no physical pixel spacing, so boundary metrics are scale-free:
  hd95_pct  95th-percentile symmetric boundary distance, as % of the image diagonal
            (empty prediction on a non-empty mask -> 100%, the worst possible)
  nsd       normalised surface Dice at tolerance = 1% of the image diagonal (~8 px on a 620x530 image)

Usage: python scripts/evaluate_runs.py [--runs runs/kvasir]
"""
import argparse
import json
import math
import sys
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.metrics import compute_hausdorff_distance, compute_surface_dice
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import build_model  # noqa: E402
from utils.data import eval_transform, load_split, read_pair  # noqa: E402
from utils.losses import dice_iou  # noqa: E402
from utils.utils import get_device, merge_logits  # noqa: E402

warnings.filterwarnings("ignore", module="monai")


def boundary_metrics(pred, target):
    """pred, target: bool tensors [H, W]. Returns (hd95 as % of diagonal, NSD at 1%-of-diagonal tolerance)."""
    diag = math.hypot(*target.shape)
    p, t = pred[None, None].float(), target[None, None].float()
    if not t.any():
        return (0.0, 1.0) if not p.any() else (100.0, 0.0)
    if not p.any():
        return 100.0, 0.0
    hd95 = compute_hausdorff_distance(p, t, include_background=True, percentile=95).item()
    nsd = compute_surface_dice(p, t, [0.01 * diag], include_background=True).item()
    return 100 * hd95 / diag, nsd


@torch.no_grad()
def evaluate_run(run_dir, device):
    rows = []
    pred_dir = run_dir / "preds"
    pred_dir.mkdir(exist_ok=True)
    for fold_dir in sorted(run_dir.glob("fold*")):
        args = Namespace(**json.loads((fold_dir / "config.json").read_text()))
        model = build_model(args).to(device).eval()
        model.load_state_dict(torch.load(fold_dir / "best.pth", map_location=device))
        tf = eval_transform(args.size)
        for name in tqdm(load_split(args.splits, args.fold)["test"], desc=f"{run_dir.name}/{fold_dir.name}", leave=False):
            image, mask = read_pair(args.data_root, name)
            x = tf(image=image)["image"][None].to(device)
            logits = F.interpolate(merge_logits(model(x)), size=mask.shape, mode="bilinear", align_corners=False)
            pred = (logits[0, 0] > 0).cpu()
            target = torch.from_numpy(mask).bool()
            dice, iou = dice_iou(pred[None], target[None])
            hd95, nsd = boundary_metrics(pred, target)
            Image.fromarray(pred.numpy().astype(np.uint8) * 255).save(pred_dir / name.replace(".jpg", ".png"))
            rows.append({"image": name, "fold": args.fold, "dice": dice.item(), "iou": iou.item(),
                         "hd95_pct": hd95, "nsd": nsd, "polyp_fraction": mask.mean(),
                         "pred_fraction": pred.float().mean().item()})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "per_image.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs/kvasir")
    args = ap.parse_args()
    device = get_device()
    for run_dir in sorted(p for p in Path(args.runs).iterdir() if list(p.glob("fold*/best.pth"))):
        df = evaluate_run(run_dir, device)
        # The training-time test Dice was computed the same way; they should agree (up to device numerics)
        orig = pd.concat(pd.read_csv(f) for f in run_dir.glob("fold*/test_metrics.csv")).set_index("image").dice
        gap = (df.set_index("image").dice - orig).abs()
        print(f"{run_dir.name:28s} n={len(df)} Dice {df.dice.mean():.4f}  HD95 {df.hd95_pct.median():.2f}% (median)  "
              f"NSD {df.nsd.mean():.4f}  | max |Dice - training-time Dice| {gap.max():.4f}, >0.01 on {(gap > 0.01).sum()} images")


if __name__ == "__main__":
    main()
