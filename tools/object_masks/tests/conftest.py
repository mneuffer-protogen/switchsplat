# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers; also makes ``tools.object_masks`` and ``threedgrut`` importable from any folder."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_mask_run(root: Path, times_of_slot: dict[int, list[int]], size: int = 8, seed: int = 0) -> Path:
    """A small mask folder in the pipeline's layout: slotNN/mask/masks.npz and slotNN/labels/labels.npz."""
    rng = np.random.default_rng(seed)
    for slot, times in times_of_slot.items():
        count = len(times)
        shape = (count, size, size)
        high = (rng.random(shape) > 0.8).astype(np.float32)
        low = ((rng.random(shape) > 0.5) & (high == 0)).astype(np.float32)
        time_indices = np.asarray(times, dtype=np.int32)
        image_names = np.asarray([f"frame_{t:04d}" for t in times])
        (root / f"slot{slot:02d}" / "mask").mkdir(parents=True)
        (root / f"slot{slot:02d}" / "labels").mkdir(parents=True)
        np.savez_compressed(
            root / f"slot{slot:02d}" / "mask" / "masks.npz",
            pred_masks=rng.random(shape).astype(np.float32),
            pred_masks_sam=(rng.random(shape) > 0.6).astype(np.float32),
            masks_high=high,
            masks_low=low,
            time_indices=time_indices,
            image_names=image_names,
        )
        np.savez_compressed(
            root / f"slot{slot:02d}" / "labels" / "labels.npz",
            masks_high=high,
            masks_low=low,
            valid=(rng.random(shape) > 0.2).astype(np.uint8),
            trusted=np.asarray([i for i in range(count) if i % 3 != 1], dtype=np.int32),
            time_indices=time_indices,
            image_names=image_names,
            high_threshold=np.float32(0.9),
            low_threshold=np.float32(0.02),
        )
    return root


def fine_tune_reader(run_dir: Path, num_slots: int):
    """The fine-tuning loss, set up as configs/apps/colmap_switchsplat_fisheye_finetune.yaml sets it up."""
    from threedgrut.utils.slot_mask_supervision import SlotMaskSupervision

    settings = SimpleNamespace(
        mask_run=str(run_dir),
        mask_key="pred_masks_sam",
        camera_id="camera1",
        other_cameras_empty=False,
        soft_gating=False,
        use_label_valid=True,
        mask_dilate_pixels=6,
    )
    conf = SimpleNamespace(loss=SimpleNamespace(slot_mask=settings), render=SimpleNamespace(method="gsplat_3dgut"))
    num_states = 1 + 2 * num_slots
    model = SimpleNamespace(
        switch_splat_controller=SimpleNamespace(num_slots=num_slots, num_states=num_states),
        switch_logits=torch.eye(num_states)[1:],
    )
    return SlotMaskSupervision(conf, model, SimpleNamespace(get_camera_names=lambda: ["camera1", "camera2"]))


@pytest.fixture
def mask_run():
    return write_mask_run


@pytest.fixture
def reader():
    return fine_tune_reader
