# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write per-slot mask coverage metrics and join every slot's preview into one scene video.

Writes SCENE_DIR/metrics.csv and, when every slot has a preview and ffmpeg is installed,
SCENE_DIR/all_slots.mp4.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .common import slot_dir
from .windows import load_windows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", type=Path, required=True, help="pipeline output folder")
    args = parser.parse_args(argv)

    slot_dirs = [slot_dir(args.scene_dir, slot) for slot in range(1, len(load_windows(args.scene_dir)) + 1)]
    incomplete = [path.name for path in slot_dirs if not (path / "mask" / "masks.npz").is_file()]
    if incomplete:
        raise SystemExit(f"slots without masks: {incomplete}")

    metrics_path = args.scene_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slot", "frames", "mlp_fraction", "sam_fraction", "mlp_zero_frames", "sam_zero_frames"])
        for slot, path in enumerate(slot_dirs, 1):
            with np.load(path / "mask" / "masks.npz") as archive:
                mlp = (archive["pred_masks"] >= 0.5).reshape(len(archive["pred_masks"]), -1)
                sam = (archive["pred_masks_sam"] >= 0.5).reshape(len(mlp), -1)
            writer.writerow(
                [
                    slot,
                    len(mlp),
                    f"{mlp.mean():.6f}",
                    f"{sam.mean():.6f}",
                    int((mlp.sum(1) == 0).sum()),
                    int((sam.sum(1) == 0).sum()),
                ]
            )
    print(f"metrics={metrics_path}", flush=True)

    previews = [path / "mask" / "result_preview.mp4" for path in slot_dirs]
    if shutil.which("ffmpeg") is None or not all(preview.is_file() for preview in previews):
        return
    video_path = args.scene_dir / "all_slots.mp4"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        list_path = Path(handle.name)
        for preview in previews:
            handle.write(f"file '{preview.resolve()}'\n")
    try:
        command = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_path)]
        command += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(video_path)]
        subprocess.run(command, check=True)
    finally:
        list_path.unlink(missing_ok=True)
    print(f"video={video_path}", flush=True)


if __name__ == "__main__":
    main()
