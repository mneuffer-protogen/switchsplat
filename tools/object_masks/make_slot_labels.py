# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn one slot's comparison residuals into red/blue pseudo-labels.

Photo and comparison render (render_pairs.py) are resized to 512 px and compared with grayscale
1-SSIM. Thresholds are quantiles of that residual over the slot's control frames (outside every
window): pixels of a labelled frame above the high quantile become positive (red, object), pixels
below the low quantile become negative (blue, background), and everything in between is ignored.
The valid region (lens circle and camera-holder mask) is eroded first, so the vignetted lens rim
is not treated as image.

Writes SCENE_DIR/slotNN/labels/{labels.npz, labels_meta.json, gt_512/} and, unless --no-previews,
preview_frames/ and seed_preview.mp4.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import (
    RESOLUTION,
    annotate,
    encode_frames,
    load_rgb,
    load_valid_mask,
    raw_ssim_error,
    savez_compressed_atomic,
    slot_dir,
    to_u8,
    write_text_atomic,
)
from .windows import frame_role, load_windows_record

FPS = 5


def magma_rgb(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [0.00, 0.001462, 0.000466, 0.013866],
            [0.25, 0.316654, 0.071243, 0.485380],
            [0.50, 0.716387, 0.214982, 0.329526],
            [0.75, 0.973895, 0.554948, 0.150780],
            [1.00, 0.987053, 0.991438, 0.749504],
        ],
        dtype=np.float32,
    )
    x = np.clip(values, 0.0, 1.0)
    out = np.stack([np.interp(x, stops[:, 0], stops[:, c + 1]) for c in range(3)], axis=-1)
    return (out * 255.0).astype(np.uint8)


def overlay_labels(gt: np.ndarray, high: np.ndarray, low: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = gt.copy()
    result[~valid] = 0.0
    result[low] = 0.35 * result[low] + 0.65 * np.array([0.05, 0.35, 1.0], dtype=np.float32)
    result[high] = 0.2 * result[high] + 0.8 * np.array([1.0, 0.05, 0.05], dtype=np.float32)
    return result.clip(0.0, 1.0)


def resize_rgb(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(to_u8(rgb), mode="RGB").resize((RESOLUTION, RESOLUTION))
    return np.asarray(image, dtype=np.float32) / 255.0


def erode_valid(valid: np.ndarray, pixels: int) -> np.ndarray:
    """Shrink the valid region so vignetted border pixels are not treated as image."""
    if pixels <= 0:
        return valid
    from scipy.ndimage import binary_erosion

    size = 2 * pixels + 1
    y, x = np.ogrid[-pixels : pixels + 1, -pixels : pixels + 1]
    disk = (x * x + y * y) <= pixels * pixels
    return binary_erosion(valid, structure=disk.reshape(size, size), border_value=0)


def load_frame(row: dict, pairs_dir: Path, cache_dir: Path, device: torch.device) -> tuple[np.ndarray, ...]:
    """Return 512px (gt, residual, valid), computing and caching the residual on first use."""
    cache_file = cache_dir / f"time_{int(row['time_index']):04d}.npz"
    if cache_file.is_file():
        with np.load(cache_file) as cached:
            return cached["gt"].astype(np.float32) / 255.0, cached["residual"], cached["valid"]

    gt = resize_rgb(load_rgb(pairs_dir / row["gt_path"]))
    render = resize_rgb(load_rgb(pairs_dir / row["render_path"]))
    valid_native = load_valid_mask(pairs_dir / row["valid_mask_path"]).astype(np.uint8) * 255
    valid_image = Image.fromarray(valid_native, mode="L").resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
    valid = np.asarray(valid_image) > 127
    residual = raw_ssim_error(render, gt, device).copy()
    residual[~valid] = 0.0
    # The temporary name is unique per process so the replace stays atomic.
    tmp_file = cache_file.parent / f".{cache_file.stem}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp_file, gt=to_u8(gt), residual=residual.astype(np.float32), valid=valid)
    tmp_file.replace(cache_file)
    return gt, residual, valid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", type=Path, required=True, help="pipeline output folder")
    parser.add_argument("--slot", type=int, required=True, help="1-based object slot")
    parser.add_argument("--high-quantile", type=float, default=0.995)
    parser.add_argument("--low-quantile", type=float, default=0.5)
    parser.add_argument(
        "--valid-erode-pixels",
        type=int,
        default=8,
        help="shrink the valid region by this many 512px pixels before labeling (fisheye rim)",
    )
    parser.add_argument("--no-previews", action="store_true", help="skip the preview frames and video")
    args = parser.parse_args(argv)

    record = load_windows_record(args.scene_dir)
    windows = np.asarray(record["windows"], dtype=np.float32)
    if not 1 <= args.slot <= len(windows):
        raise SystemExit(f"slot {args.slot} out of range for {len(windows)} windows")
    directory = slot_dir(args.scene_dir, args.slot)
    pairs_dir = directory / "pairs"
    cache_dir = directory / "residual_cache"
    out_dir = directory / "labels"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (pairs_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    active = []
    inactive_values = []
    for row_index, row in enumerate(rows):
        time_index = int(row["time_index"])
        # Labelled frames lie in the slot's window, also where other windows overlap it; frames
        # that lie only in other windows are neither labels nor clean controls.
        role = frame_role(windows, args.slot, time_index)
        if role is None:
            continue
        gt, residual, valid = load_frame(row, pairs_dir, cache_dir, device)
        valid = erode_valid(valid, args.valid_erode_pixels)
        residual = np.where(valid, residual, 0.0)
        if role == "control":
            inactive_values.append(residual[valid])
        else:
            active.append(
                {
                    "time_index": time_index,
                    "image_name": row["image_name"],
                    "gt": gt,
                    "residual": residual,
                    "valid": valid,
                }
            )
        if row_index % 100 == 0:
            print(f"residual {row_index + 1}/{len(rows)}", flush=True)

    if not inactive_values:
        raise SystemExit("need control frames to calibrate thresholds")
    if not active:
        raise SystemExit(f"no labelled frames for slot {args.slot}")
    active.sort(key=lambda item: item["time_index"])
    inactive_cat = np.concatenate(inactive_values)
    high_threshold = float(np.quantile(inactive_cat, args.high_quantile))
    low_threshold = float(np.quantile(inactive_cat, args.low_quantile))
    print(
        f"slot{args.slot:02d} control pixels={inactive_cat.size} "
        f"high q{args.high_quantile}={high_threshold:.6f} low q{args.low_quantile}={low_threshold:.6f}",
        flush=True,
    )

    gt_dir = out_dir / "gt_512"
    frames_dir = out_dir / "preview_frames"
    gt_dir.mkdir(exist_ok=True)
    if not args.no_previews:
        frames_dir.mkdir(exist_ok=True)
    count = len(active)
    masks_high = np.zeros((count, RESOLUTION, RESOLUTION), dtype=np.float32)
    masks_low = np.zeros((count, RESOLUTION, RESOLUTION), dtype=np.float32)
    valids = np.zeros((count, RESOLUTION, RESOLUTION), dtype=np.uint8)
    trusted = []
    frames = []
    for i, item in enumerate(active):
        valid = item["valid"]
        high = valid & (item["residual"] >= high_threshold)
        low = valid & (item["residual"] <= low_threshold) & ~high
        masks_high[i] = high
        masks_low[i] = low
        valids[i] = valid
        if high.any() or low.any():
            trusted.append(i)

        stem = f"time_{item['time_index']:04d}"
        Image.fromarray(to_u8(item["gt"])).save(gt_dir / f"{stem}.png")
        if not args.no_previews:
            gt_visible = item["gt"] * valid[..., None]  # previews show only the valid region
            panel = np.concatenate(
                [
                    annotate(to_u8(gt_visible), [f"GT time={item['time_index']}", f"slot{args.slot} {i + 1}/{count}"]),
                    annotate(
                        magma_rgb(item["residual"]),
                        ["1-SSIM @512", f"hi={high_threshold:.4f}", f"lo={low_threshold:.4f}"],
                    ),
                    annotate(
                        to_u8(overlay_labels(gt_visible, high, low, valid)),
                        [f"pos(red)={int(high.sum())}", f"neg(blue)={int(low.sum())}", "ignore=clear"],
                    ),
                ],
                axis=1,
            )
            Image.fromarray(panel).save(frames_dir / f"{i:04d}_{stem}.jpg", quality=90)
        frames.append({"time_index": item["time_index"], "pos_pixels": int(high.sum()), "neg_pixels": int(low.sum())})

    if not args.no_previews:
        encode_frames(frames_dir, out_dir / "seed_preview.mp4", FPS)
    start, end = (float(value) for value in windows[args.slot - 1])
    meta = {
        "slot": args.slot,
        "window": [start, end],
        "windows_source": record["source"],
        "n_active_frames": count,
        "n_control_frames": len(inactive_values),
        "n_trusted": len(trusted),
        "resolution": RESOLUTION,
        "high_quantile": args.high_quantile,
        "low_quantile": args.low_quantile,
        "valid_erode_pixels": args.valid_erode_pixels,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "total_pos_pixels": int(masks_high.sum()),
        "total_neg_pixels": int(masks_low.sum()),
        "frames": frames,
    }
    write_text_atomic(out_dir / "labels_meta.json", json.dumps(meta, indent=2) + "\n")
    # labels.npz is written last: it marks the stage as done.
    savez_compressed_atomic(
        out_dir / "labels.npz",
        masks_high=masks_high,
        masks_low=masks_low,
        valid=valids,
        trusted=np.asarray(trusted, dtype=np.int32),
        time_indices=np.asarray([item["time_index"] for item in active], dtype=np.int32),
        image_names=np.asarray([item["image_name"] for item in active]),
        high_threshold=np.float32(high_threshold),
        low_threshold=np.float32(low_threshold),
    )
    print(f"done slot{args.slot:02d} frames={count} pos={int(masks_high.sum())} neg={int(masks_low.sum())}", flush=True)


if __name__ == "__main__":
    main()
