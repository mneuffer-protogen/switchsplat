# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the object masks of one scene: every stage in order, resuming where a previous run stopped.

Stages (each one is also a command of its own, python -m tools.object_masks.<stage>):
  windows               windows.json: the checkpoint's learned windows, or --windows-file
  select_times          per slot: the frames inside its window, plus control frames outside all windows
  render_pairs          per slot: flip-one comparison renders next to the photos (GPU)
  make_slot_labels      per slot: 1-SSIM residuals -> red/blue pseudo-labels
  extract_dinov3_cache  DINOv3 features of the labelled photos (GPU)
  train_slot_mask       per slot: DINOv3+XY MLP, then SAM 2.1 odd/even propagation (GPU)
  build_result_preview  per slot: inspection video (skipped with --no-previews)
  settle_masks          settled/: masks without the frames near any window start or end
  finalize_scene        metrics.csv (and all_slots.mp4)

Pass OUT/settled as loss.slot_mask.mask_run of the fine-tuning. Rerunning with the same arguments
skips finished work; other settings need a new --out.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a plain script: python tools/object_masks/make_masks.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "tools.object_masks"

from .common import REPO_ROOT, slot_dir, write_text_atomic
from .extract_dinov3_cache import DEFAULT_MODEL
from .settle_masks import RECORD_FILE
from .windows import WINDOWS_FILE, load_windows_record


def run(stage: str, *args) -> None:
    """Run one stage as its own process (from the repository root; all paths are absolute), so each
    GPU stage starts with free memory."""
    command = [sys.executable, "-m", f"tools.object_masks.{stage}", *(str(arg) for arg in args)]
    print("+ " + " ".join(command), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(command, env=env, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"stage {stage} failed with exit code {result.returncode}")


def check_run_config(out: Path, config: dict) -> None:
    """Record the settings of a new output folder, or make sure a resumed run uses the same ones."""
    path = out / "run_config.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        changed = sorted(key for key in set(previous) | set(config) if previous.get(key) != config.get(key))
        if changed:
            raise SystemExit(f"{out} was made with other settings ({', '.join(changed)}); use a new --out")
        return
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(config, indent=2) + "\n")


def trained_scene(checkpoint: Path) -> Path:
    """The scene folder stored in the checkpoint's config (relative paths are from the repository root)."""
    import torch

    path = Path(str(torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]["path"]))
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="SwitchSplat checkpoint that will be refined")
    parser.add_argument("--data", type=Path, help="scene folder (default: the one the checkpoint was trained on)")
    parser.add_argument("--out", type=Path, help="output folder (default: <checkpoint folder>/object_masks)")
    parser.add_argument(
        "--windows-file", type=Path, help='use these windows ("start end" per line) instead of the learned ones'
    )
    parser.add_argument(
        "--valid-mask", type=Path, help="lens valid-region mask (default: the one the checkpoint was trained with)"
    )
    parser.add_argument(
        "--sam-checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt", help="SAM 2.1 Hiera-L"
    )
    parser.add_argument("--dinov3-model", default=DEFAULT_MODEL, help="Hugging Face model id or local folder")
    parser.add_argument("--camera-id", default="camera1", help="camera to mask (the fine-tune uses camera1)")
    parser.add_argument("--n-controls", type=int, default=40, help="control frames per slot")
    parser.add_argument("--high-quantile", type=float, default=0.995, help="object label threshold quantile")
    parser.add_argument("--low-quantile", type=float, default=0.5, help="background label threshold quantile")
    parser.add_argument("--valid-erode-pixels", type=int, default=8, help="valid-region erosion at 512 px")
    parser.add_argument("--seed", type=int, default=0, help="classifier seed")
    parser.add_argument("--settle-margin", type=int, default=10, help="frames to keep from every window start and end")
    parser.add_argument("--no-previews", action="store_true", help="skip the inspection images and videos")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.resolve()
    out = (args.out or checkpoint.parent / "object_masks").resolve()
    data = (args.data or trained_scene(checkpoint)).resolve()
    sam_checkpoint = args.sam_checkpoint.resolve()
    windows_file = None if args.windows_file is None else args.windows_file.resolve()
    valid_mask = None if args.valid_mask is None else args.valid_mask.resolve()
    for path in (checkpoint, data, sam_checkpoint, windows_file, valid_mask):
        if path is not None and not path.exists():
            raise SystemExit(f"not found: {path}")
    local_model = Path(args.dinov3_model).expanduser()
    dinov3_model = str(local_model.resolve()) if local_model.exists() else args.dinov3_model
    check_run_config(
        out,
        {
            "checkpoint": str(checkpoint),
            "data": str(data),
            "windows_file": None if windows_file is None else str(windows_file),
            "valid_mask": None if valid_mask is None else str(valid_mask),
            "sam_checkpoint": str(sam_checkpoint),
            "dinov3_model": dinov3_model,
            "camera_id": args.camera_id,
            "n_controls": args.n_controls,
            "high_quantile": args.high_quantile,
            "low_quantile": args.low_quantile,
            "valid_erode_pixels": args.valid_erode_pixels,
            "seed": args.seed,
            "settle_margin": args.settle_margin,
            "previews": not args.no_previews,
        },
    )
    label_flags = ["--no-previews"] if args.no_previews else []

    if not (out / WINDOWS_FILE).is_file():
        extra = [] if windows_file is None else ["--windows-file", windows_file]
        run("windows", "--checkpoint", checkpoint, "--out", out / WINDOWS_FILE, *extra)
    record = load_windows_record(out)
    slots = list(range(1, len(record["windows"]) + 1))

    unselected = [slot for slot in slots if not (slot_dir(out, slot) / "render_times.txt").is_file()]
    if unselected:
        options = ["--data", data, "--camera-id", args.camera_id, "--n-controls", args.n_controls]
        run("select_times", "--scene-dir", out, *options, *[arg for slot in unselected for arg in ("--slot", slot)])

    unrendered = [slot for slot in slots if not (slot_dir(out, slot) / "pairs" / "manifest.csv").is_file()]
    if unrendered:
        options = ["--checkpoint", checkpoint, "--data", data, "--camera-id", args.camera_id]
        options += [] if valid_mask is None else ["--valid-mask", valid_mask]
        run("render_pairs", "--scene-dir", out, *options, *[arg for slot in unrendered for arg in ("--slot", slot)])

    for slot in slots:
        if not (slot_dir(out, slot) / "labels" / "labels.npz").is_file():
            options = ["--high-quantile", args.high_quantile, "--low-quantile", args.low_quantile]
            options += ["--valid-erode-pixels", args.valid_erode_pixels, *label_flags]
            run("make_slot_labels", "--scene-dir", out, "--slot", slot, *options)

    pending = [slot for slot in slots if not (slot_dir(out, slot) / "mask" / ".complete").is_file()]
    cache = out / f"dinov3_cache_{args.camera_id}"
    if pending:
        image_dir = data / record["images_folder"] / args.camera_id
        labels = [slot_dir(out, slot) / "labels" / "labels.npz" for slot in pending]
        options = ["--image-dir", image_dir, "--cache-dir", cache, "--model", dinov3_model]
        run("extract_dinov3_cache", *options, "--labels", *labels)
    for slot in pending:
        mask_dir = slot_dir(out, slot) / "mask"
        if not (mask_dir / "masks.npz").is_file():
            options = ["--feature-cache", cache, "--sam-checkpoint", sam_checkpoint, "--seed", args.seed]
            run("train_slot_mask", "--labels-dir", slot_dir(out, slot) / "labels", "--out-dir", mask_dir, *options)
        if not args.no_previews:
            run("build_result_preview", "--slot-dir", slot_dir(out, slot))
        (mask_dir / ".complete").touch()

    if not (out / "settled" / RECORD_FILE).is_file():
        run("settle_masks", "--scene-dir", out, "--out", out / "settled", "--margin", args.settle_margin)
    run("finalize_scene", "--scene-dir", out)
    print(f"done; refine with loss.slot_mask.mask_run={out / 'settled'}", flush=True)


if __name__ == "__main__":
    main()
