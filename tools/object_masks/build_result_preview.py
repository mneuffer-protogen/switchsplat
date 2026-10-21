# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a slot's 4-panel inspection video: photo | red/blue labels | classifier mask | SAM-refined mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import annotate, encode_frames, to_u8

PINK = np.array([1.0, 0.2, 0.85], dtype=np.float32)
RED = np.array([1.0, 0.05, 0.05], dtype=np.float32)
BLUE = np.array([0.05, 0.35, 1.0], dtype=np.float32)


def overlay_mask(gt: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = gt.copy()
    selected = mask > 0.5
    out[selected] = (1 - alpha) * out[selected] + alpha * PINK
    return out


def overlay_seeds(gt: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    out = gt.copy()
    out[neg > 0.5] = 0.35 * out[neg > 0.5] + 0.65 * BLUE
    out[pos > 0.5] = 0.2 * out[pos > 0.5] + 0.8 * RED
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot-dir", type=Path, required=True, help="SCENE_DIR/slotNN")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args(argv)

    labels_dir = args.slot_dir / "labels"
    mask_dir = args.slot_dir / "mask"
    frames_dir = mask_dir / "result_preview_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((labels_dir / "labels_meta.json").read_text(encoding="utf-8"))
    slot = meta["slot"]

    with np.load(mask_dir / "masks.npz") as archive:
        pred, sam = archive["pred_masks"], archive["pred_masks_sam"]
        high, low = archive["masks_high"], archive["masks_low"]
        times = [int(t) for t in archive["time_indices"]]
    with np.load(labels_dir / "labels.npz") as archive:
        valid = archive["valid"].astype(bool)

    for i, time_index in enumerate(times):
        gt_path = labels_dir / "gt_512" / f"time_{time_index:04d}.png"
        gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0
        gt = gt * valid[i][..., None]  # show only what the valid mask keeps
        panels = [
            annotate(to_u8(gt), [f"GT t={time_index}", f"slot{slot} {i + 1}/{len(times)}"]),
            annotate(
                to_u8(overlay_seeds(gt, high[i], low[i])),
                [f"seeds q{meta['high_quantile']:g}/{meta['low_quantile']:g}", "red=pos blue=neg"],
            ),
            annotate(to_u8(overlay_mask(gt, pred[i])), ["MLP > 0.5", f"frac={float((pred[i] >= 0.5).mean()):.3f}"]),
            annotate(to_u8(overlay_mask(gt, sam[i])), ["SAM refine", f"frac={float((sam[i] >= 0.5).mean()):.3f}"]),
        ]
        frame_path = frames_dir / f"{i:04d}_time_{time_index:04d}.jpg"
        Image.fromarray(np.concatenate(panels, axis=1)).save(frame_path, quality=92)

    if encode_frames(frames_dir, mask_dir / "result_preview.mp4", args.fps):
        print(f"wrote {mask_dir / 'result_preview.mp4'} ({len(times)} frames)", flush=True)


if __name__ == "__main__":
    main()
