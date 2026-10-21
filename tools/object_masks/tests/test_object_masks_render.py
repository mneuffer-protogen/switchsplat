# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The render stage on CPU with a stand-in renderer: flip-one states, frame lookup and outputs."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.model.switch_splat_controller import SwitchSplatController
from tools.object_masks import render_pairs, windows
from tools.object_masks.common import REPO_ROOT


def test_render_stage_renders_flip_one_states(tmp_path: Path, monkeypatch):
    controller = SwitchSplatController(
        OmegaConf.create({"num_slots": 2, "num_frames": 60, "init_windows": [[10, 30], [20, 40]], "temperature": 1})
    )
    learned = windows.learned_windows(
        {"switch_splat": controller.state_dict(), "config": {"switch_splat": {"num_frames": 60}}}
    ).tolist()
    scene = tmp_path / "scene"
    for slot, times in ((1, "5,12,25,50"), (2, "25,33,99")):  # frame 99 is no training frame
        (scene / f"slot{slot:02d}").mkdir(parents=True)
        (scene / f"slot{slot:02d}" / "render_times.txt").write_text(times + "\n")
    record = {"source": "learned", "windows": learned, "learned_windows": learned, "images_folder": "images_2"}
    (scene / "windows.json").write_text(json.dumps(record))

    rendered = []

    class Model:
        def __init__(self, conf):
            self.conf, self.switch_splat_controller, self.background = conf, controller, None

        def init_from_checkpoint(self, checkpoint, setup_optimizer=True):
            assert not setup_optimizer

        def build_acc(self):
            pass

        def __call__(self, batch, state_activations=None):
            rendered.append((int(batch.time_idx[0]), state_activations.clone()))
            value = float(state_activations[0, 1:].mean())
            return {"pred_features": torch.full((1, 4, 4, 3), value), "pred_opacity": torch.ones((1, 4, 4, 1))}

    class Dataset:
        time_indices = np.array([5, 12, 25, 33, 50, 12])
        image_paths = np.array(
            [f"/data/images_2/camera1/frame_{t:04d}.png" for t in (5, 12, 25, 33, 50)]
            + ["/data/images_2/camera2/frame_0012.png"]
        )

        def __len__(self):
            return len(self.time_indices)

        def __getitem__(self, idx):
            return {"idx": idx}

        def get_gpu_batch_with_intrinsics(self, batch):
            idx = int(batch["idx"][0])
            mask = torch.ones((1, 4, 4, 1))
            mask[:, :, 0] = 0.0
            return SimpleNamespace(
                rgb_gt=torch.full((1, 4, 4, 3), 0.25), mask=mask, time_idx=torch.tensor([self.time_indices[idx]])
            )

    conf = OmegaConf.create(
        {
            "path": "elsewhere",
            "dataset": {
                "type": "colmap",
                "fullcircle": {
                    "distractor_mask_dilation": 20,
                    "fisheye_valid_mask": "masking/fisheye_valid_mask_circle_r1260.png",
                    "fisheye_valid_mask_tripod": "masking/fisheye_valid_mask_circle_r1260.png",
                },
            },
        }
    )
    made = {}

    def make_test(name, config, eval_split):
        made.update(name=name, path=config.path, split=eval_split, mask=config.dataset.fullcircle.fisheye_valid_mask)
        return Dataset()

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"config": conf, "global_step": 7})
    monkeypatch.setattr("threedgrut.model.model.MixtureOfGaussians", Model)
    monkeypatch.setattr("threedgrut.datasets.make_test", make_test)
    monkeypatch.chdir(tmp_path)  # a relative lens mask path of the config resolves against the repository
    argv = ["--checkpoint", str(tmp_path / "ckpt.pt"), "--data", str(tmp_path / "data"), "--scene-dir", str(scene)]
    render_pairs.main(argv)

    assert made == {
        "name": "colmap",
        "path": str(tmp_path / "data"),
        "split": "train",
        "mask": str(REPO_ROOT / "masking/fisheye_valid_mask_circle_r1260.png"),
    }
    expected_frames = {1: [5, 12, 25, 50], 2: [25, 33]}
    assert [t for t, _ in rendered] == expected_frames[1] + expected_frames[2]
    for (time_index, state), slot in zip(rendered, [1] * 4 + [2] * 2):
        assert torch.equal(state, windows.flip_one_state(controller, time_index, slot))
    for slot, frames in expected_frames.items():
        pairs = scene / f"slot{slot:02d}" / "pairs"
        with (pairs / "manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [int(row["time_index"]) for row in rows] == frames
        assert [row["image_name"] for row in rows] == [f"frame_{t:04d}" for t in frames]
        for row in rows:
            for key in ("render_path", "gt_path", "valid_mask_path"):
                assert (pairs / row[key]).is_file()
        metadata = json.loads((pairs / "metadata.json").read_text())
        assert metadata["slot"] == slot and metadata["global_step"] == 7
        assert metadata["missing_frames"] == ([99] if slot == 2 else [])

    rendered.clear()  # finished slots are not rendered again
    render_pairs.main(argv)
    assert rendered == []

    (scene / "slot01" / "pairs" / "manifest.csv").unlink()
    record["learned_windows"] = [[0.0, 5.0], [6.0, 9.0]]
    (scene / "windows.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit):  # windows.json of another checkpoint
        render_pairs.main(argv)
