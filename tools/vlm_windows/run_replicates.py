# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run detect_windows.py on several scenes and trials, as for the paper's 5 x 10 study.

Each (trial, scene) gets its own folder OUT_ROOT/<model>/<effort>/trial<N>/<scene>,
laid out like results/gpt-5.6-sol/, and the rollouts run concurrently. Like
detect_windows.py this is a dry run unless --run is given; --replay-root reuses
recorded answers instead (for example results/gpt-5.6-sol). After a --run or
--replay batch, OUT_ROOT/<model>/<effort>/trial<N>/index.json summarises each
trial (prompt and input hashes must equal trial 1's), and the trial can be scored:

  python evaluate.py OUT_ROOT/gpt-5.6-sol/max/trial1 --postprocess --gt-root <dataset>/pinhole

Example (all 12 scenes, 10 trials at max effort, as in the paper):
  python run_replicates.py <dataset>/pinhole --out-root runs/vlm --trials 1 2 3 4 5 6 7 8 9 10 --run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
DETECT = TOOL_DIR / "detect_windows.py"
sys.path.insert(0, str(TOOL_DIR))

from detect_windows import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    write_json,
)
from evaluate import paper_scenes  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_dir(args: argparse.Namespace, trial: int, scene: str) -> Path:
    return args.out_root / args.model / args.reasoning_effort / f"trial{trial}" / scene


def command(args: argparse.Namespace, trial: int, scene: str) -> list[str]:
    cmd = [
        sys.executable,
        str(DETECT),
        str(args.dataset_root / scene),
        "--out-dir",
        str(output_dir(args, trial, scene)),
        "--model",
        args.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--codex-executable",
        args.codex_executable,
    ]
    if args.font:
        cmd.extend(["--font", args.font])
    if args.run:
        cmd.append("--run")
    if args.replay_root:
        cmd.extend(["--replay", str(args.replay_root / args.reasoning_effort / f"trial{trial}" / scene)])
    return cmd


def run_one(args: argparse.Namespace, trial: int, scene: str) -> int:
    log_dir = args.out_root / args.model / args.reasoning_effort / f"trial{trial}" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[start] trial{trial} {scene}", flush=True)
    result = subprocess.run(command(args, trial, scene), check=False, capture_output=True, text=True)
    (log_dir / f"{scene}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{scene}.stderr.log").write_text(result.stderr, encoding="utf-8")
    print(f"[done]  trial{trial} {scene} rc={result.returncode}", flush=True)
    return result.returncode


def build_index(args: argparse.Namespace, trial: int, baseline_trial: int) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for scene in args.scenes:
        directory = output_dir(args, trial, scene)
        manifest = load_json(directory / "manifest.json")
        baseline = load_json(output_dir(args, baseline_trial, scene) / "manifest.json")
        index[scene] = {
            "status": manifest["status"],
            "camera": manifest["camera"],
            "event_count": manifest["event_count"],
            "prompt_profile": manifest["prompt_profile"],
            "prompt_sha256": manifest["prompt_sha256"],
            f"prompt_matches_trial{baseline_trial}": manifest["prompt_sha256"] == baseline["prompt_sha256"],
            f"inputs_match_trial{baseline_trial}": (manifest["input_sha256_digest"] == baseline["input_sha256_digest"]),
            "input_sha256_digest": manifest["input_sha256_digest"],
            "usage": manifest.get("usage"),
            "command_executions": manifest.get("command_executions", 0),
            "raw_observation": f"{scene}/raw_observation.json",
        }
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="Folder with one subfolder per scene")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=None, help="Default: the 12 paper scenes")
    parser.add_argument("--trials", type=int, nargs="+", default=[1])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--font", default=None, help="Path to DejaVuSans-Bold.ttf")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent rollouts (default: all)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Call the model (one Codex rollout per task)")
    mode.add_argument("--replay-root", type=Path, default=None, help="e.g. results/gpt-5.6-sol")
    args = parser.parse_args()
    args.scenes = args.scenes or paper_scenes()
    args.dataset_root = args.dataset_root.resolve()
    args.out_root = args.out_root.resolve()
    if args.replay_root:
        args.replay_root = args.replay_root.resolve()

    trials = tuple(args.trials)
    if not trials or len(set(trials)) != len(trials):
        raise SystemExit("--trials must be nonempty and contain no duplicates")
    tasks = [(trial, scene) for trial in trials for scene in args.scenes]
    workers = args.workers if args.workers is not None else len(tasks)
    if workers < 1:
        raise SystemExit("--workers must be positive")
    for trial, scene in tasks:
        if not (args.dataset_root / scene).is_dir():
            raise SystemExit(f"Missing scene folder: {args.dataset_root / scene}")
        if output_dir(args, trial, scene).exists():
            raise SystemExit(f"Refusing to overwrite: {output_dir(args, trial, scene)}")

    failures: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, args, trial, scene): (trial, scene) for trial, scene in tasks}
        for future in as_completed(futures):
            if future.result() != 0:
                failures.append(futures[future])
    if failures:
        raise SystemExit(f"Failed rollouts (see the logs folders): {sorted(failures)}")
    if not (args.run or args.replay_root):
        print("dry run complete; add --run to call the model")
        return
    for trial in trials:
        index_path = args.out_root / args.model / args.reasoning_effort / f"trial{trial}" / "index.json"
        index = build_index(args, trial, trials[0])
        write_json(index_path, index)
        mismatched = [
            scene
            for scene, row in index.items()
            if not (row[f"prompt_matches_trial{trials[0]}"] and row[f"inputs_match_trial{trials[0]}"])
        ]
        print(f"[index] {index_path}" + (f"  prompt/input mismatch: {mismatched}" if mismatched else ""))


if __name__ == "__main__":
    main()
