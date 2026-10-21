# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interaction windows and every frame rule that depends on them.

All timing decisions of the pipeline come from one set of windows, one ``[start, end]`` pair per
object slot, in frames:

* by default the windows the SwitchSplat model learned, read from the checkpoint being refined:
  ``endpoints = (num_frames - 1) * sigmoid(endpoints_raw)`` and ``window = [min, max]`` of each
  pair, exactly as ``SwitchSplatController.window_bounds`` computes them;
* or windows from a text file (``--windows-file``) with one "start end" pair per line, such as the
  dataset's ``vlm_prediction.txt`` or ``interaction_times.txt``. No annotation file is read unless
  it is passed here.

A frame t lies in a window when ``start <= t <= end``. For object slot k:

* its labelled frames are the training frames inside window k, also where other windows overlap it;
* its control frames, which set the label thresholds, are evenly spaced training frames outside
  every window;
* frames that lie only inside other windows are not used for slot k;
* its comparison renders switch only slot k to its base state ("flip-one");
* its masks are kept ("settled") only at least ``margin`` frames inside window k and away from the
  start and end of every window, so frames in which an object may still be moving are dropped.

Usage:
    python -m tools.object_masks.windows --checkpoint ckpt_last.pt --out SCENE_DIR/windows.json \\
        [--windows-file vlm_prediction.txt]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import write_text_atomic

WINDOWS_FILE = "windows.json"


def learned_windows(checkpoint: dict) -> np.ndarray:
    """The [K, 2] float32 windows a SwitchSplat checkpoint learned (``SwitchSplatController.window_bounds``)."""
    endpoints_raw = checkpoint["switch_splat"]["endpoints_raw"].detach().cpu()
    num_frames = int(checkpoint["config"]["switch_splat"]["num_frames"])
    endpoints = float(num_frames - 1) * torch.sigmoid(endpoints_raw)
    starts = torch.minimum(endpoints[:, 0], endpoints[:, 1])
    ends = torch.maximum(endpoints[:, 0], endpoints[:, 1])
    return torch.stack([starts, ends], dim=1).numpy().astype(np.float32)


def read_windows_file(path: Path) -> np.ndarray:
    """[K, 2] float32 windows from a text file with one "start end" pair per line (the trainer's format)."""
    windows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if windows.size == 0 or windows.ndim != 2 or windows.shape[1] != 2 or not np.all(np.isfinite(windows)):
        raise ValueError(f"{path}: expected one 'start end' pair per line")
    if np.any(windows[:, 0] > windows[:, 1]):
        raise ValueError(f"{path}: a window starts after it ends")
    return windows.astype(np.float32)


def in_window(time_index: int, window) -> bool:
    return bool(window[0] <= time_index <= window[1])


def active_slots(windows: np.ndarray, time_index: int) -> list[int]:
    """1-based slots whose window contains ``time_index``."""
    return [slot for slot, window in enumerate(windows, 1) if in_window(time_index, window)]


def frame_role(windows: np.ndarray, slot: int, time_index: int) -> str | None:
    """Role of a frame for one slot: label (inside its window), control (outside every window) or None."""
    if in_window(time_index, windows[slot - 1]):
        return "label"
    if active_slots(windows, time_index):
        return None
    return "control"


def select_slot_frames(
    train_times: list[int], windows: np.ndarray, slot: int, n_controls: int
) -> tuple[list[int], list[int]]:
    """Labelled frames of one slot, and its control frames: ``n_controls`` evenly spaced training
    frames outside every window (all of them if there are fewer)."""
    if n_controls < 2:
        raise ValueError("need at least two control frames")
    times = sorted({int(t) for t in train_times})
    labelled = [t for t in times if in_window(t, windows[slot - 1])]
    outside = [t for t in times if not active_slots(windows, t)]
    if len(outside) <= n_controls:
        controls = outside
    else:
        idxs = {round(i * (len(outside) - 1) / (n_controls - 1)) for i in range(n_controls)}
        controls = [outside[i] for i in sorted(idxs)]
    return labelled, controls


def flip_one_state(controller, time_index: int, slot: int) -> torch.Tensor:
    """The model's state activations at ``time_index`` with only ``slot`` (1-based) in its base state.

    Slot k owns state 2k-1 (base) and 2k (switched); every other object keeps the state the model
    gives it at that time.
    """
    time = torch.as_tensor([time_index], device=controller.endpoints_raw.device)
    state = controller.state_activations(time).clone()
    state[:, 2 * slot - 1] = 1.0
    state[:, 2 * slot] = 0.0
    return state


def settle_keep(times, windows: np.ndarray, slot: int, margin: int) -> np.ndarray:
    """Rows of a slot's masks to keep: inside the slot's window and at least ``margin`` frames from
    the start and the end of every window. Window bounds are float32, the precision of the learned
    windows, also for windows from a file."""
    windows = np.asarray(windows, dtype=np.float32)
    times = np.asarray(times).astype(np.int64)
    start, end = windows[slot - 1]
    kept = (times >= start + np.float32(margin)) & (times <= end - np.float32(margin))
    edges = windows.reshape(-1)
    kept &= np.abs(times[:, None] - edges[None, :]).min(axis=1) >= margin
    return kept


def settle_reason(time_index: int, windows: np.ndarray, slot: int, margin: int) -> str:
    """Why ``settle_keep`` drops a frame, for the record."""
    start, end = (float(value) for value in windows[slot - 1])
    if not start <= time_index <= end:
        return f"outside slot{slot:02d} window [{start:.2f}, {end:.2f}]"
    edges = [
        (abs(time_index - float(value)), other, name, float(value))
        for other, window in enumerate(windows, 1)
        for name, value in (("start", window[0]), ("end", window[1]))
    ]
    distance, other, name, value = min(edges)
    if distance < margin:
        return f"{distance:.2f} frames from slot{other:02d} {name} {value:.2f}"
    return f"less than {margin} frames inside slot{slot:02d} window [{start:.2f}, {end:.2f}]"


def load_windows_record(scene_dir: Path) -> dict:
    path = Path(scene_dir) / WINDOWS_FILE
    if not path.is_file():
        raise SystemExit(f"missing {path}; run tools.object_masks.windows first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_windows(scene_dir: Path) -> np.ndarray:
    """The [K, 2] float32 windows chosen for a scene (``SCENE_DIR/windows.json``)."""
    return np.asarray(load_windows_record(scene_dir)["windows"], dtype=np.float32)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the interaction windows of a scene (windows.json).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SwitchSplat checkpoint that will be refined")
    parser.add_argument(
        "--windows-file", type=Path, help='use these windows ("start end" per line) instead of the learned ones'
    )
    parser.add_argument("--out", type=Path, required=True, help="output path, normally SCENE_DIR/windows.json")
    args = parser.parse_args(argv)

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    if "switch_splat" not in checkpoint:
        raise SystemExit(f"{args.checkpoint} has no SwitchSplat slots")
    config = checkpoint["config"]
    learned = learned_windows(checkpoint)
    num_frames = int(config["switch_splat"]["num_frames"])
    downsample = int(config["dataset"].get("downsample_factor", 1))
    del checkpoint

    if args.windows_file is None:
        windows, source = learned, "learned"
    else:
        windows, source = read_windows_file(args.windows_file), "file"
        if len(windows) != len(learned):
            raise SystemExit(f"{args.windows_file} has {len(windows)} windows but the model has {len(learned)} slots")
        for slot, ((start, end), (lstart, lend)) in enumerate(zip(windows, learned), 1):
            if end < lstart or start > lend:
                print(
                    f"warning: slot{slot:02d} window [{start:g}, {end:g}] does not overlap the learned window "
                    f"[{lstart:.1f}, {lend:.1f}]; are the lines in slot order?",
                    flush=True,
                )

    record = {
        "source": source,
        "checkpoint": str(args.checkpoint),
        "windows_file": None if args.windows_file is None else str(args.windows_file),
        "num_frames": num_frames,
        "num_slots": len(windows),
        "images_folder": "images" if downsample == 1 else f"images_{downsample}",
        "learned_windows": [[float(start), float(end)] for start, end in learned],
        "windows": [[float(start), float(end)] for start, end in windows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.out, json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out} ({source} windows)", flush=True)
    for slot, ((start, end), (lstart, lend)) in enumerate(zip(windows, learned), 1):
        print(f"  slot{slot:02d} window [{start:8.2f}, {end:8.2f}]  learned [{lstart:8.2f}, {lend:8.2f}]", flush=True)


if __name__ == "__main__":
    main()
