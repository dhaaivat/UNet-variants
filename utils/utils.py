import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.losses import dice_iou


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def merge_logits(outputs):
    # UNet++ with deep supervision returns one logit map per head; average them ("accurate mode")
    return torch.stack(outputs).mean(0) if isinstance(outputs, (list, tuple)) else outputs


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """scaler: a torch.amp.GradScaler to train in fp16 mixed precision (CUDA only), or None for fp32."""
    model.train()
    total = 0.0
    bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in bar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler is not None):
            outputs = model(images)
        if scaler is not None:
            outputs = [o.float() for o in outputs] if isinstance(outputs, (list, tuple)) else outputs.float()
        loss = criterion(outputs, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * images.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Mean loss and per-image mean Dice at the training resolution (used for model selection)."""
    model.eval()
    total, dices = 0.0, []
    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        total += criterion(outputs, masks).item() * images.size(0)
        dice, _ = dice_iou(merge_logits(outputs) > 0, masks > 0.5)
        dices.append(dice.cpu())
    return total / len(loader.dataset), torch.cat(dices).mean().item()


@torch.no_grad()
def test_original_resolution(model, loader, device):
    """Per-image Dice/IoU after upsampling logits back to each image's original size (batch_size=1)."""
    model.eval()
    rows = []
    for images, _, original_mask, name in tqdm(loader, desc="Test", leave=False):
        logits = merge_logits(model(images.to(device)))
        logits = F.interpolate(logits, size=original_mask.shape[-2:], mode="bilinear", align_corners=False)
        dice, iou = dice_iou(logits.cpu() > 0, original_mask.unsqueeze(1) > 0)
        rows.append({"image": name[0], "dice": dice.item(), "iou": iou.item(),
                     "polyp_fraction": original_mask.float().mean().item()})
    return rows


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,
                checkpoint_path, writer=None, early_stop_patience=None, scaler=None):
    """Train for num_epochs, keeping the checkpoint with the best validation Dice.

    Early stopping is off by default so every model gets the same epoch budget.
    """
    best_val_dice, best_epoch, epochs_without_improvement = -1.0, 0, 0
    history = []
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        seconds = time.time() - start

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_dice": val_dice, "lr": lr, "seconds": seconds})
        if writer is not None:
            for k in ("train_loss", "val_loss", "val_dice", "lr"):
                writer.add_scalar(k, history[-1][k], epoch)

        improved = val_dice > best_val_dice
        if improved:
            best_val_dice, best_epoch, epochs_without_improvement = val_dice, epoch, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:3d}/{num_epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"val Dice {val_dice:.4f}{' *' if improved else ''}  lr {lr:.2e}  {seconds:.0f}s", flush=True)

        if early_stop_patience and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping: no val Dice improvement for {early_stop_patience} epochs.")
            break

    return history, best_val_dice, best_epoch
