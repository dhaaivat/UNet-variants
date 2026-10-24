"""Train one U-Net variant on one Kvasir-SEG fold and evaluate it on that fold's test set.

Examples:
  python train.py --model unet --fold 0
  python train.py --model unetpp --fold 0 --deep-supervision
  python train.py --model attention_unet --fold 0 --benchmark 10   # time 10 training batches, no training
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.UNet_models import AttentionUNet, DoubleConv, FlexibleUNet, FlexibleUNetPP, ResidualBlock
from utils.data import KvasirDataset, eval_transform, load_split, train_transform
from utils.losses import DiceBCELoss
from utils.utils import get_device, set_seed, test_original_resolution, train_model

MODELS = {"unet": FlexibleUNet, "attention_unet": AttentionUNet, "unetpp": FlexibleUNetPP}
BLOCKS = {"double_conv": DoubleConv, "residual": ResidualBlock}


def build_model(args):
    kwargs = dict(num_classes=1, in_channels=3, base_channels=args.base_channels, block=BLOCKS[args.block])
    if args.model == "unetpp":
        kwargs["deep_supervision"] = args.deep_supervision
    return MODELS[args.model](**kwargs)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(model, loader, criterion, optimizer, device, n_batches, scaler):
    """Time n_batches training batches (data loading included), after 2 warm-up batches."""
    model.to(device).train()
    for i, (images, masks) in enumerate(loader):
        if i == 2:
            synchronize(device)
            start = time.time()
        if i == n_batches + 2:
            break
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler is not None):
            outputs = model(images)
        loss = criterion(outputs, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    synchronize(device)
    per_batch = (time.time() - start) / n_batches
    if device.type == "mps":
        memory = f", {torch.mps.driver_allocated_memory() / 2**30:.1f} GB MPS memory"
    elif device.type == "cuda":
        memory = f", {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak CUDA memory"
    else:
        memory = ""
    print(f"{args.model}/{args.block} base {args.base_channels} size {args.size}: {per_batch:.3f}s per batch of "
          f"{loader.batch_size} -> ~{per_batch * len(loader):.0f}s of training compute per epoch "
          f"({len(loader)} batches){memory}", flush=True)


def main(args):
    set_seed(args.seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input size: let cuDNN pick the fastest kernels
    if args.amp and device.type != "cuda":
        raise SystemExit("--amp is only supported on CUDA (it was slower than fp32 on Apple MPS)")
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    split = load_split(args.splits, args.fold)
    print(f"device {device}; fold {args.fold}: {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test")

    loader_kwargs = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    preload = args.size if args.preload else None
    train_loader = DataLoader(KvasirDataset(args.data_root, split["train"], train_transform(args.size), preload_size=preload),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(KvasirDataset(args.data_root, split["val"], eval_transform(args.size), preload_size=preload),
                            batch_size=args.batch_size, **loader_kwargs)

    model = build_model(args)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.benchmark:
        benchmark(model, train_loader, criterion, optimizer, device, args.benchmark, scaler)
        return

    run_name = args.model + ("_ds" if args.deep_supervision else "") + f"_{args.block}"
    out = Path(args.out) / run_name / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"device": str(device), "n_params": sum(p.numel() for p in model.parameters()),
                           "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"])}
    (out / "config.json").write_text(json.dumps(config, indent=1))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    writer = SummaryWriter(out / "tb")
    history, best_val_dice, best_epoch = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs,
        out / "best.pth", writer=writer, early_stop_patience=args.early_stop_patience, scaler=scaler)
    writer.close()
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # Test the best-on-validation checkpoint at each image's original resolution
    model.load_state_dict(torch.load(out / "best.pth", map_location=device))
    test_loader = DataLoader(KvasirDataset(args.data_root, split["test"], eval_transform(args.size), return_original=True),
                             batch_size=1, num_workers=args.workers)
    rows = pd.DataFrame(test_original_resolution(model, test_loader, device))
    rows.to_csv(out / "test_metrics.csv", index=False)

    summary = {"run": run_name, "fold": args.fold, "best_epoch": best_epoch, "best_val_dice": best_val_dice,
               "test_dice": rows.dice.mean(), "test_iou": rows.iou.mean(),
               "train_minutes": sum(h["seconds"] for h in history) / 60}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--block", choices=BLOCKS, default="double_conv")
    ap.add_argument("--deep-supervision", action="store_true", help="UNet++ only")
    ap.add_argument("--base-channels", type=int, default=32, help="32 as in the UNet++ paper; 64 as in U-Net (needs >16 GB for UNet++)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--size", type=int, default=256, help="training resolution; must be divisible by 16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--early-stop-patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-preload", dest="preload", action="store_false",
                    help="decode JPEGs every epoch instead of resizing all train/val images once up front")
    ap.add_argument("--amp", action="store_true", help="fp16 mixed precision (CUDA only); fold 0 was trained in fp32")
    ap.add_argument("--data-root", default="Kvasir-SEG")
    ap.add_argument("--splits", default="splits/kvasir_5fold.json")
    ap.add_argument("--out", default="runs/kvasir")
    ap.add_argument("--benchmark", type=int, default=0, metavar="N", help="time N training batches and exit")
    args = ap.parse_args()
    main(args)
