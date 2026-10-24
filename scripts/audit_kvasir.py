"""Audit Kvasir-SEG before training: pairing, sizes, mask values, polyp sizes, bbox agreement, near-duplicates.

Usage: python scripts/audit_kvasir.py [--root Kvasir-SEG] [--montage runs/audit/near_duplicates.png]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def dhash(img, size=16):
    """Difference hash: compare horizontally adjacent pixels of a size x (size+1) grayscale thumbnail."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def near_duplicate_pairs(hashes, max_dist):
    h = np.stack(hashes).astype(np.uint8)
    dist = (h[:, None, :] != h[None, :, :]).sum(-1)
    i, j = np.where(np.triu(dist <= max_dist, k=1))
    return [(int(a), int(b), int(dist[a, b])) for a, b in zip(i, j)], dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Kvasir-SEG")
    ap.add_argument("--montage", default="runs/audit/near_duplicates.png")
    args = ap.parse_args()
    root = Path(args.root)

    images = sorted(p.name for p in (root / "images").glob("*.jpg"))
    masks = sorted(p.name for p in (root / "masks").glob("*.jpg"))
    print(f"images: {len(images)}  masks: {len(masks)}")
    print(f"unpaired images: {sorted(set(images) - set(masks))}  unpaired masks: {sorted(set(masks) - set(images))}")
    names = sorted(set(images) & set(masks))
    bboxes = json.loads((root / "kavsir_bboxes.json").read_text())

    sizes, size_mismatch, mask_rgb_differs = Counter(), [], []
    gray_frac, area_frac, n_components, bbox_count_mismatch, empty = [], [], [], [], []
    hashes = []
    for n in names:
        img = Image.open(root / "images" / n)
        m = np.asarray(Image.open(root / "masks" / n))
        sizes[img.size] += 1
        if img.size != (m.shape[1], m.shape[0]):
            size_mismatch.append(n)
        if m.ndim == 3:
            if np.abs(m.astype(int) - m[..., :1]).max() > 10:
                mask_rgb_differs.append(n)
            m = m[..., 0]
        # JPEG masks: anything not near 0 or 255 is a compression artefact
        gray_frac.append(((m > 10) & (m < 245)).mean())
        binary = m >= 128
        if not binary.any():
            empty.append(n)
        area_frac.append(binary.mean())
        _, k = ndimage.label(binary)
        n_components.append(k)
        nb = len(bboxes.get(Path(n).stem, {}).get("bbox", []))
        if nb != k:
            bbox_count_mismatch.append((n, nb, k))
        hashes.append(dhash(img))

    print(f"\nimage sizes (W x H): {len(sizes)} distinct; most common: {sizes.most_common(5)}")
    ws, hs = zip(*[s for s, c in sizes.items() for _ in range(c)])
    print(f"  width  min/median/max: {min(ws)}/{int(np.median(ws))}/{max(ws)}")
    print(f"  height min/median/max: {min(hs)}/{int(np.median(hs))}/{max(hs)}")
    print(f"image/mask size mismatches: {len(size_mismatch)}")
    print(f"masks whose RGB channels differ: {len(mask_rgb_differs)}")

    gray_frac, area_frac = np.array(gray_frac), np.array(area_frac)
    print(f"\nmask pixels neither ~0 nor ~255 (JPEG artefacts): mean {gray_frac.mean():.4%}, max {gray_frac.max():.4%}")
    print(f"empty masks after thresholding at 128: {len(empty)}")
    q = np.percentile(area_frac, [0, 5, 25, 50, 75, 95, 100])
    print("polyp area as % of image (min, p5, p25, median, p75, p95, max): " + ", ".join(f"{v:.1%}" for v in q))
    print(f"polyp components per mask: {sorted(Counter(n_components).items())}")
    print(f"masks whose component count != bbox count in kavsir_bboxes.json: {len(bbox_count_mismatch)}")

    print("\nnear-duplicates (16x16 dHash, 256 bits; Hamming distance <= d):")
    for d in (0, 10, 20, 30, 40):
        pairs, dist = near_duplicate_pairs(hashes, d)
        print(f"  d <= {d:2d}: {len(pairs)} pairs")
    pairs, dist = near_duplicate_pairs(hashes, 40)

    if args.montage and pairs:
        out = Path(args.montage)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Sample pairs across the distance range so the threshold can be chosen by eye
        pairs = sorted(pairs, key=lambda p: p[2])
        pick = [pairs[int(i)] for i in np.linspace(0, len(pairs) - 1, min(12, len(pairs)))]
        tile = 160
        sheet = Image.new("RGB", (tile * 4, tile * ((len(pick) + 1) // 2)), "white")
        for k, (a, b, d) in enumerate(pick):
            x0, y0 = (k % 2) * tile * 2, (k // 2) * tile
            sheet.paste(Image.open(root / "images" / names[a]).convert("RGB").resize((tile, tile)), (x0, y0))
            sheet.paste(Image.open(root / "images" / names[b]).convert("RGB").resize((tile, tile)), (x0 + tile, y0))
            print(f"  montage row {k // 2} {'left' if k % 2 == 0 else 'right'}: d={d}  {names[a]}  {names[b]}")
        sheet.save(out)
        print(f"montage: {out}")


if __name__ == "__main__":
    main()
