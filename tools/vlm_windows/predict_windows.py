# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Predict the interaction windows of one scene with a vision-language model, in one command.

Builds the model input (every 8th front-lens frame, stamped with its frame number), calls the model
once through the Codex CLI with the paper's prompt and JSON schema, filters the answer (windows
shorter than 20 frames are dropped, near-duplicates are merged) and writes vlm_prediction.txt: one
"start end" line per object interaction, in frame numbers. Training initialises one object slot
per line (switch_splat.init_windows_file).

    python tools/vlm_windows/predict_windows.py data/my_scene

If the scene has no vlm_prediction.txt yet, the result is also copied there, where training
looks for it by default. An existing file (such as the ones shipped with the released scenes) is
never overwritten; train with switch_splat.init_windows_file=<printed path> instead.

Requirements: the Codex CLI, logged in (`codex login`) with access to the model (default
gpt-5.6-sol, the paper's), and the DejaVu Sans Bold font for the frame stamps. --dry-run only
builds the model input; --replay reuses a recorded answer. See README.md in this folder for the method and the paper's analysis.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect_windows  # noqa: E402
import postprocess  # noqa: E402

WINDOWS_FILE = "vlm_prediction.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_root", type=Path, help="Scene folder with images_2/camera1 (or images/camera1)")
    parser.add_argument("--out-dir", type=Path, default=None, help="New folder; default runs/vlm/<scene>_<time>")
    parser.add_argument("--model", default=detect_windows.DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=detect_windows.DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument("--camera", default=detect_windows.DEFAULT_CAMERA)
    parser.add_argument(
        "--image-root", type=Path, default=None, help="Default: the first of images_2 and images in the scene"
    )
    parser.add_argument("--font", default=None, help="Path to DejaVuSans-Bold.ttf if Pillow cannot find it")
    parser.add_argument("--codex-executable", default="codex")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Only build the model input; call no model")
    mode.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Reuse a recorded answer instead of calling the model, e.g. "
        "tools/vlm_windows/results/gpt-5.6-sol/max/trial1/kitchen2 (the paper's windows)",
    )
    args = parser.parse_args()

    scene_root = args.scene_root.resolve()
    if not scene_root.is_dir():
        raise SystemExit(f"Scene folder not found: {scene_root}")
    out_dir = args.out_dir or Path("runs/vlm") / f"{scene_root.name}_{datetime.now():%Y%m%d_%H%M%S}"

    detect_args = [str(scene_root), "--out-dir", str(out_dir), "--model", args.model]
    detect_args += ["--reasoning-effort", args.reasoning_effort, "--camera", args.camera]
    detect_args += ["--codex-executable", args.codex_executable]
    if args.image_root is not None:
        detect_args += ["--image-root", str(args.image_root)]
    if args.font is not None:
        detect_args += ["--font", args.font]
    if args.replay is not None:
        detect_args += ["--replay", str(args.replay)]
    elif not args.dry_run:
        detect_args.append("--run")
    detect_windows.run(detect_windows.build_parser().parse_args(detect_args))
    if args.dry_run:
        print(f"dry run: model input in {out_dir}")
        return

    events = json.loads((out_dir / "raw_observation.json").read_text(encoding="utf-8"))["events"]
    kept, audit = postprocess.process_events(events)
    windows_path = out_dir / WINDOWS_FILE
    windows_path.write_text(postprocess.format_windows(kept), encoding="utf-8")
    (out_dir / "postprocess_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"{len(kept)} interaction windows (of {len(events)} raw answers):")
    for event in kept:
        print(f"  {int(event['on_frame'])}-{int(event['off_frame'])}  {event['object_description']}")
    print(f"wrote {windows_path.resolve()}")

    scene_file = scene_root / WINDOWS_FILE
    if scene_file.exists():
        print(f"{scene_file} already exists and was left unchanged; to train with the new windows, add")
        print(f"  switch_splat.init_windows_file={windows_path.resolve()}")
    else:
        shutil.copyfile(windows_path, scene_file)
        print(f"copied to {scene_file}, where training reads it by default")


if __name__ == "__main__":
    main()
