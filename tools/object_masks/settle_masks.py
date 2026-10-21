# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep only the masks of frames in which no object is moving ("settle").

A slot keeps a mask frame only if the frame lies inside the slot's window and at least --margin
frames (default 10) from the start and the end of every window. Near a window edge the photo may
show an object part-way through its motion, while the model has one fully switched state, so the
mask cannot cover the model's object and the fine-tune would fade it.

The copy holds every array of masks.npz and labels.npz minus the dropped rows (``trusted`` row
indices are renumbered), in the layout the fine-tuning loss reads (loss.slot_mask.mask_run).
DROP_TRANSITIONS.json records the windows, the margin and every dropped frame with its reason.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import savez_compressed_atomic, slot_dir, write_text_atomic
from .windows import load_windows_record, settle_keep, settle_reason

RECORD_FILE = "DROP_TRANSITIONS.json"


def drop_rows(bundle, keep: np.ndarray, num_rows: int) -> dict[str, np.ndarray]:
    """All arrays of an npz bundle without the dropped rows; ``trusted`` row indices are renumbered."""
    new_index = np.cumsum(keep) - 1
    arrays = {}
    for key in bundle.files:
        value = bundle[key]
        if key == "trusted":  # row indices into the kept frames
            value = new_index[value[keep[value]]].astype(value.dtype)
        elif value.ndim and value.shape[0] == num_rows:
            value = value[keep]
        arrays[key] = value
    return arrays


def settle_slot(src_slot: Path, dst_slot: Path, windows: np.ndarray, slot: int, margin: int) -> dict:
    """Write one slot's settled masks.npz and labels.npz; return its record."""
    with np.load(src_slot / "mask" / "masks.npz") as masks, np.load(src_slot / "labels" / "labels.npz") as labels:
        times = masks["time_indices"]
        if not np.array_equal(times, labels["time_indices"]):
            raise SystemExit(f"{src_slot}: masks and labels cover different frames")
        keep = settle_keep(times, windows, slot, margin)
        for name, bundle in (("mask/masks.npz", masks), ("labels/labels.npz", labels)):
            (dst_slot / name).parent.mkdir(parents=True, exist_ok=True)
            savez_compressed_atomic(dst_slot / name, **drop_rows(bundle, keep, len(times)))
    dropped = [int(t) for t in times[~keep]]
    return {
        "frames": int(len(times)),
        "kept": int(keep.sum()),
        "dropped": dropped,
        "reasons": [settle_reason(t, windows, slot, margin) for t in dropped],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", type=Path, required=True, help="pipeline output folder")
    parser.add_argument("--out", type=Path, help="settled copy (default: SCENE_DIR/settled)")
    parser.add_argument("--margin", type=int, default=10, help="frames to keep away from every window start and end")
    args = parser.parse_args(argv)
    out = args.out or args.scene_dir / "settled"

    record = load_windows_record(args.scene_dir)
    windows = np.asarray(record["windows"], dtype=np.float32)
    summary = {
        "source": str(args.scene_dir),
        "windows_source": record["source"],
        "windows_file": record.get("windows_file"),
        "windows": record["windows"],
        "settle_margin": args.margin,
        "rule": "a slot keeps a frame only inside its own window and at least settle_margin frames "
        "from the start and the end of every window",
        "slots": {},
    }
    out.mkdir(parents=True, exist_ok=True)
    for slot in range(1, len(windows) + 1):
        name = slot_dir(args.scene_dir, slot).name
        result = settle_slot(slot_dir(args.scene_dir, slot), out / name, windows, slot, args.margin)
        summary["slots"][name] = result
        print(f"{name}: kept {result['kept']}/{result['frames']}", flush=True)
        if result["kept"] == 0:
            print(f"warning: {name} keeps no frame; its object gets no mask supervision", flush=True)
    write_text_atomic(out / RECORD_FILE, json.dumps(summary, indent=1) + "\n")  # written last: marks the copy done
    print(f"settled masks: {out}", flush=True)


if __name__ == "__main__":
    main()
