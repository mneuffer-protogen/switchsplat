# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a SwitchSplat checkpoint on a (possibly different) dataset of the same scene.

Loads the checkpoint, changes only the dataset path and the fisheye lens-circle masks in its
in-memory config, and delegates rendering and every metric to the project's normal
``Renderer`` (same splits, camera-holder masks, renderer and sort order as training-time
evaluation).  Used to score fisheye-trained models on the pinhole test views: the two
datasets share camera poses, frame lists and object timings; only the lens model differs.
The checkpoint on disk is never modified.

usage: python tools/eval_on_dataset.py --checkpoint CKPT --out-dir NEW_DIR [--path DATASET] [--fisheye-mask none|MASK]
                          [--global-z-order true|false]
--global-z-order overrides the checkpoint's sort order at render time (a diagnostic: the model is then
drawn with an ordering it was not trained with).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from threedgrut.render import Renderer

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l], replace=True)

# Frame counts of the supplement's pinhole/fisheye table (Kitchen 2), a guard against a mismatched split.
_KITCHEN2_FRAMES = {"test": 198, "base_state": 88, "novel_view": 308, "novel_combo": 290}
EXPECTED_FRAMES = {"kitchen2": _KITCHEN2_FRAMES}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--path", default="", help="dataset to evaluate on (default: the checkpoint's own)")
    parser.add_argument("--fisheye-mask", default=None,
                        help="'none' clears the lens-circle masks (pinhole data); a path replaces them")
    parser.add_argument("--global-z-order", choices=("true", "false"), default=None,
                        help="render-time sort-order override (diagnostic)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve(strict=True)
    if args.out_dir.exists():
        raise FileExistsError(f"evaluation output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = checkpoint["config"]
    mutations = {}
    if args.path:
        mutations["path"] = str(Path(args.path).resolve(strict=True))
    if args.fisheye_mask is not None:
        mask = "" if args.fisheye_mask == "none" else str(Path(args.fisheye_mask).resolve(strict=True))
        mutations["dataset.fullcircle.fisheye_valid_mask"] = mask
        mutations["dataset.fullcircle.fisheye_valid_mask_tripod"] = mask
    if args.global_z_order is not None:
        mutations["render.splat.global_z_order"] = args.global_z_order == "true"
    for key, value in mutations.items():
        OmegaConf.update(conf, key, value)
    scene = str(conf.path).rstrip("/").split("/")[-1]

    contract = {
        "global_step": int(checkpoint["global_step"]),
        "scene": scene,
        "renderer": str(conf.render.method),
        "global_z_order": bool(conf.render.splat.global_z_order),
        "downsample_factor": int(conf.dataset.downsample_factor),
        "distractor_mask_dilation": int(conf.dataset.fullcircle.distractor_mask_dilation),
        "distractor_mask_white_is_valid": bool(conf.dataset.fullcircle.get("distractor_mask_white_is_valid", False)),
        "normalize_world_space": bool(conf.dataset.normalize_world_space),
        "eval_splits": list(conf.eval.splits),
        "fisheye_valid_mask": str(conf.dataset.fullcircle.fisheye_valid_mask),
        "path": str(conf.path),
    }
    print("CONTRACT " + json.dumps(contract), flush=True)

    original_torch_load = torch.load

    def return_mutated_checkpoint(path, *load_args, **load_kwargs):
        if Path(path).resolve() == checkpoint_path:
            return checkpoint
        return original_torch_load(path, *load_args, **load_kwargs)

    with patch.object(torch, "load", side_effect=return_mutated_checkpoint):
        renderer = Renderer.from_checkpoint(
            checkpoint_path=str(checkpoint_path), out_dir=str(args.out_dir), save_gt=False, computes_extra_metrics=True
        )
    frames = {split: len(dataset) for split, (dataset, _) in renderer.eval_dataloaders.items()}
    expected = EXPECTED_FRAMES.get(scene)
    if expected is not None and frames != expected:
        raise RuntimeError(f"split contract mismatch: frames={frames} expected={expected}")
    print(f"EVAL_HEALTHY frames={frames}", flush=True)

    renderer.render_all()

    step_dir = Path(renderer.out_dir) / f"ours_{int(checkpoint['global_step'])}"
    metrics = {}
    for split in frames:
        m = json.loads((step_dir / split / "metrics.json").read_text())
        metrics[split] = {k: v for k, v in m.items() if not isinstance(v, (list, dict))}
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "in_memory_config_mutations": mutations,
        "contract": contract,
        "frames": frames,
        "metrics": metrics,
        "disk_checkpoint_modified": False,
    }
    (args.out_dir / "evaluation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("EVAL_COMPLETED " + json.dumps({s: round(m["mean_psnr"], 4) for s, m in metrics.items()}), flush=True)


if __name__ == "__main__":
    main()
