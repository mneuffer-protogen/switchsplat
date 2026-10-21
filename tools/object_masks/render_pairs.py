# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render each slot's comparison images next to the photos ("flip-one").

For every frame chosen for object slot k (select_times.py), the model being refined renders the
photo's view at the photo's time with only object k switched to its base state; every other object
keeps the state the model gives it at that time. Where object k is switched in the photo, render and
photo then differ mainly where object k changed. Control frames, outside all windows, follow the same
rule. Rendering uses the checkpoint's own model, renderer and dataset settings (for the fisheye
models: gsplat 3DGUT with distance sorting, the lens circle and the camera-holder masks).

Writes SCENE_DIR/slotNN/pairs/{renders,gt,valid_masks}/time_NNNN.png at full resolution, plus
manifest.csv and metadata.json. Needs a CUDA GPU.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from omegaconf import open_dict

from .common import REPO_ROOT, slot_dir, write_text_atomic
from .windows import flip_one_state, load_windows_record

MANIFEST_FIELDS = ["time_index", "image_name", "render_path", "gt_path", "valid_mask_path", "mean_l1"]


def save_hwc(image: torch.Tensor, path: Path) -> None:
    torchvision.utils.save_image(image.squeeze(0).permute(2, 0, 1), str(path))


def _resolve_config_path(value: str) -> str:
    """A relative path from a training config (relative to the repository root), made usable from any folder."""
    path = Path(str(value))
    if path.is_absolute() or path.exists() or not (REPO_ROOT / path).exists():
        return str(path)
    return str(REPO_ROOT / path)


def prepare_config(conf, data: Path, valid_mask: Path | None = None):
    """The checkpoint's own config, pointed at ``data`` and, if given, at another lens valid-region mask."""
    with open_dict(conf):
        conf.path = str(data)
        fullcircle = conf.dataset.get("fullcircle", None)
        if valid_mask is not None:
            if not fullcircle or fullcircle.get("distractor_mask_dilation", None) is None:
                raise SystemExit("--valid-mask needs a checkpoint trained with dataset.fullcircle masks (fisheye)")
            fullcircle.fisheye_valid_mask = str(valid_mask)
            fullcircle.fisheye_valid_mask_tripod = str(valid_mask)
        elif fullcircle:
            for key in ("fisheye_valid_mask", "fisheye_valid_mask_tripod"):
                if fullcircle.get(key, None):
                    fullcircle[key] = _resolve_config_path(fullcircle[key])
    return conf


def read_render_times(path: Path) -> list[int]:
    return sorted({int(value) for value in path.read_text(encoding="utf-8").replace("\n", "").split(",") if value})


@torch.no_grad()
def render_slot(model, controller, dataset, index_of_time: dict, slot: int, times: list[int], pairs_dir: Path):
    """Render one slot's frames; returns the manifest rows and the frames that are not training frames."""
    from threedgrut.utils.render import apply_background

    for sub in ("renders", "gt", "valid_masks"):
        (pairs_dir / sub).mkdir(parents=True, exist_ok=True)
    present = [t for t in times if t in index_of_time]
    missing = [t for t in times if t not in index_of_time]
    rows = []
    for count, time_index in enumerate(present):
        idx = index_of_time[time_index]
        batch = dataset.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([dataset[idx]]))
        state = flip_one_state(controller, time_index, slot)
        outputs = apply_background(model.background, model(batch, state_activations=state), batch)
        pred = outputs["pred_features"].clamp(0.0, 1.0)
        gt = batch.rgb_gt.clamp(0.0, 1.0)
        valid = batch.mask if batch.mask is not None else torch.ones_like(gt[..., :1])

        stem = f"time_{time_index:04d}.png"
        save_hwc(pred, pairs_dir / "renders" / stem)
        save_hwc(gt, pairs_dir / "gt" / stem)
        save_hwc(valid.float(), pairs_dir / "valid_masks" / stem)
        valid_rgb = valid.expand_as(gt) > 0.5
        rows.append(
            {
                "time_index": time_index,
                "image_name": Path(str(dataset.image_paths[idx])).stem,
                "render_path": f"renders/{stem}",
                "gt_path": f"gt/{stem}",
                "valid_mask_path": f"valid_masks/{stem}",
                "mean_l1": f"{torch.abs(pred - gt)[valid_rgb].mean().item():.8f}",
            }
        )
        if (count + 1) % 50 == 0 or count + 1 == len(present):
            print(f"slot{slot:02d}: rendered {count + 1}/{len(present)}", flush=True)
    return rows, missing


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="SwitchSplat checkpoint that will be refined")
    parser.add_argument("--data", type=Path, required=True, help="scene folder the checkpoint was trained on")
    parser.add_argument("--scene-dir", type=Path, required=True, help="pipeline output folder")
    parser.add_argument("--camera-id", default="camera1")
    parser.add_argument(
        "--valid-mask",
        type=Path,
        help="lens valid-region mask (default: the checkpoint's dataset.fullcircle.fisheye_valid_mask)",
    )
    parser.add_argument("--slot", type=int, action="append", help="1-based slot (repeatable); default: all")
    args = parser.parse_args(argv)

    record = load_windows_record(args.scene_dir)
    num_slots = len(record["windows"])
    todo = []
    for slot in args.slot or range(1, num_slots + 1):
        if not 1 <= slot <= num_slots:
            raise SystemExit(f"slot {slot} is outside 1..{num_slots}")
        if (slot_dir(args.scene_dir, slot) / "pairs" / "manifest.csv").is_file():
            print(f"slot{slot:02d}: already rendered", flush=True)
        elif not (slot_dir(args.scene_dir, slot) / "render_times.txt").is_file():
            raise SystemExit(f"slot{slot:02d}: missing render_times.txt; run tools.object_masks.select_times first")
        else:
            todo.append(slot)
    if not todo:
        return

    import threedgrut.datasets as datasets
    from threedgrut.model.model import MixtureOfGaussians

    checkpoint = torch.load(str(args.checkpoint), weights_only=False)
    if "post_processing" in checkpoint or "feature_decoder" in checkpoint:
        raise SystemExit("checkpoints with post-processing or a feature decoder are not supported")
    conf = prepare_config(checkpoint["config"], args.data, args.valid_mask)
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()
    controller = model.switch_splat_controller
    if controller is None:
        raise SystemExit("checkpoint has no SwitchSplat slots")
    if controller.num_slots != num_slots:
        raise SystemExit(f"windows.json has {num_slots} windows but the model has {controller.num_slots} slots")
    bounds = torch.stack(controller.window_bounds(), dim=1).detach().cpu().numpy()
    if not np.allclose(bounds, record["learned_windows"], atol=1e-3):
        raise SystemExit(f"windows.json was written for another checkpoint than {args.checkpoint}")
    global_step = int(checkpoint["global_step"])
    del checkpoint

    dataset = datasets.make_test(name=conf.dataset.type, config=conf, eval_split="train")
    if dataset.time_indices is None:
        raise SystemExit("the training frames carry no time index (expected frame_<time> image names)")
    index_of_time = {}
    for idx in range(len(dataset)):
        if Path(str(dataset.image_paths[idx])).parent.name != args.camera_id:
            continue
        time_index = int(dataset.time_indices[idx])
        if time_index in index_of_time:
            raise SystemExit(f"two {args.camera_id} training frames share time index {time_index}")
        index_of_time[time_index] = idx
    fullcircle = conf.dataset.get("fullcircle", None)

    for slot in todo:
        directory = slot_dir(args.scene_dir, slot)
        pairs_dir = directory / "pairs"
        times = read_render_times(directory / "render_times.txt")
        rows, missing = render_slot(model, controller, dataset, index_of_time, slot, times, pairs_dir)
        if missing:
            print(f"warning: slot{slot:02d}: {len(missing)} frames are not {args.camera_id} training frames: ", end="")
            print(missing[:20], flush=True)
        metadata = {
            "checkpoint": str(args.checkpoint),
            "global_step": global_step,
            "dataset": str(args.data),
            "split": "train",
            "camera_id": args.camera_id,
            "num_slots": num_slots,
            "slot": slot,
            "comparison": "flip-one: the model's state at the photo's time with only this slot in its base state",
            "windows_source": record["source"],
            "window": record["windows"][slot - 1],
            "valid_mask": None if not fullcircle else fullcircle.get("fisheye_valid_mask", None),
            "requested_frames": len(times),
            "rendered_frames": len(rows),
            "missing_frames": missing,
        }
        write_text_atomic(pairs_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        write_text_atomic(pairs_dir / "manifest.csv", buffer.getvalue())  # written last: marks the slot done


if __name__ == "__main__":
    main()
