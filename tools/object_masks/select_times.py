# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the frames to render for every object slot.

For slot k: every training frame of the camera inside window k (the frames to label), plus evenly
spaced training frames outside all windows (the control frames that set the label thresholds).
Training frames follow the dataset loader: images of the camera that are not listed in
test_frames.txt and are not novel-view or novel-combination test images.

Writes SCENE_DIR/slotNN/render_times.txt (comma-separated time indices).
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import numpy as np

from .common import (
    IMAGE_SUFFIXES,
    NOVEL_EVAL_PATTERN,
    frame_time_index,
    slot_dir,
    write_text_atomic,
)
from .windows import load_windows_record, select_slot_frames


def training_frames(data: Path, camera_id: str, image_folder: str) -> list[int]:
    """Sorted time indices of one camera's training frames, split as the dataset loader splits them."""
    test_file = data / "test_frames.txt"
    test = set()
    if test_file.is_file():
        lines = test_file.read_text(encoding="utf-8").splitlines()
        test = {str(PurePosixPath(line.strip())) for line in lines if line.strip()}
    image_dir = data / image_folder / camera_id
    if not image_dir.is_dir():
        raise SystemExit(f"missing image folder {image_dir}")
    times = set()
    for path in image_dir.iterdir():
        if path.suffix.lower() not in IMAGE_SUFFIXES or NOVEL_EVAL_PATTERN.search(path.name):
            continue
        if f"{camera_id}/{path.name}" in test:
            continue
        time_index = frame_time_index(path.name)
        if time_index is not None:
            times.add(time_index)
    return sorted(times)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="scene folder (images_2/, sparse/0, ...)")
    parser.add_argument("--scene-dir", type=Path, required=True, help="pipeline output folder with windows.json")
    parser.add_argument("--camera-id", default="camera1")
    parser.add_argument("--image-folder", help="image folder inside --data (default: from windows.json)")
    parser.add_argument("--n-controls", type=int, default=40, help="control frames per slot")
    parser.add_argument("--slot", type=int, action="append", help="1-based slot (repeatable); default: all")
    args = parser.parse_args(argv)

    record = load_windows_record(args.scene_dir)
    windows = np.asarray(record["windows"], dtype=np.float32)
    image_folder = args.image_folder or record["images_folder"]
    train = training_frames(args.data, args.camera_id, image_folder)
    for slot in args.slot or range(1, len(windows) + 1):
        if not 1 <= slot <= len(windows):
            raise SystemExit(f"slot {slot} is outside 1..{len(windows)}")
        labelled, controls = select_slot_frames(train, windows, slot, args.n_controls)
        start, end = windows[slot - 1]
        if not labelled:
            raise SystemExit(f"slot{slot:02d}: no {args.camera_id} training frame lies in [{start:.2f}, {end:.2f}]")
        if not controls:
            raise SystemExit("no training frame lies outside all windows to serve as a control frame")
        selected = sorted(set(labelled) | set(controls))
        out = slot_dir(args.scene_dir, slot) / "render_times.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(out, ",".join(str(t) for t in selected) + "\n")
        print(
            f"wrote {out}: n={len(selected)} labelled={len(labelled)} controls={len(controls)} "
            f"window=[{start:.2f}, {end:.2f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
