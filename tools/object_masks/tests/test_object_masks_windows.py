# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the window rules: learned windows, frame selection, flip-one states and settling."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.model.switch_splat_controller import SwitchSplatController
from tools.object_masks import settle_masks, windows
from tools.object_masks.select_times import training_frames


def _trained_controller(num_frames: int = 200) -> SwitchSplatController:
    config = OmegaConf.create(
        {"num_slots": 2, "num_frames": num_frames, "init_windows": [[30.0, 60.0], [90.0, 120.0]], "temperature": 2.0}
    )
    controller = SwitchSplatController(config)
    with torch.no_grad():
        # As after training: the second pair has its endpoints in reverse order.
        controller.endpoints_raw.copy_(torch.tensor([[-1.2, 0.6335], [0.9, 0.2]]))
    return controller


def _checkpoint(controller: SwitchSplatController) -> dict:
    config = OmegaConf.create(
        {
            "switch_splat": {"num_slots": controller.num_slots, "num_frames": controller.num_frames},
            "dataset": {"downsample_factor": 2},
        }
    )
    return {"switch_splat": controller.state_dict(), "config": config}


def test_learned_windows_match_the_controller():
    controller = _trained_controller()
    starts, ends = controller.window_bounds()
    expected = torch.stack([starts, ends], dim=1).detach().numpy()
    learned = windows.learned_windows(_checkpoint(controller))
    assert learned.dtype == np.float32
    assert np.array_equal(learned, expected)
    assert np.all(learned[:, 0] <= learned[:, 1])


def test_windows_cli_uses_the_learned_windows_unless_a_file_is_given(tmp_path: Path):
    controller = _trained_controller()
    ckpt = str(tmp_path / "ckpt.pt")
    torch.save(_checkpoint(controller), ckpt)
    windows.main(["--checkpoint", ckpt, "--out", str(tmp_path / "learned" / "windows.json")])
    record = json.loads((tmp_path / "learned" / "windows.json").read_text())
    assert record["source"] == "learned" and record["images_folder"] == "images_2"
    assert np.array_equal(windows.load_windows(tmp_path / "learned"), windows.learned_windows(_checkpoint(controller)))

    (tmp_path / "vlm.txt").write_text("40 70\n100 131.5\n")
    out = tmp_path / "file" / "windows.json"
    windows.main(["--checkpoint", ckpt, "--windows-file", str(tmp_path / "vlm.txt"), "--out", str(out)])
    assert json.loads(out.read_text())["source"] == "file"
    assert windows.load_windows(out.parent).tolist() == [[40.0, 70.0], [100.0, 131.5]]

    (tmp_path / "three.txt").write_text("1 2\n3 4\n5 6\n")
    with pytest.raises(SystemExit):  # one window per slot of the checkpoint
        windows.main(["--checkpoint", ckpt, "--windows-file", str(tmp_path / "three.txt"), "--out", str(out)])


def test_windows_file_must_hold_ordered_pairs(tmp_path: Path):
    (tmp_path / "one.txt").write_text("153 299\n")
    assert windows.read_windows_file(tmp_path / "one.txt").tolist() == [[153.0, 299.0]]
    for text in ("10 5\n", "1 2 3\n"):
        (tmp_path / "bad.txt").write_text(text)
        with pytest.raises(ValueError):
            windows.read_windows_file(tmp_path / "bad.txt")


def test_flip_one_switches_only_the_target_slot():
    class Controller:
        endpoints_raw = torch.zeros(1)

        def state_activations(self, times):
            return torch.tensor([[1.0, 0.1, 0.9, 0.2, 0.8]])

    # Always flip-one, also where the target is the only switched object.
    assert torch.equal(windows.flip_one_state(Controller(), 17, 1), torch.tensor([[1.0, 1.0, 0.0, 0.2, 0.8]]))
    assert torch.equal(windows.flip_one_state(Controller(), 17, 2), torch.tensor([[1.0, 0.1, 0.9, 1.0, 0.0]]))


def test_flip_one_keeps_the_model_state_of_the_other_objects():
    controller = _trained_controller()
    (start1, start2), (end1, end2) = (value.tolist() for value in controller.window_bounds())
    overlap = int(round((max(start1, start2) + min(end1, end2)) / 2))
    assert max(start1, start2) < overlap < min(end1, end2), "the test windows must overlap"
    both = controller.state_activations(torch.tensor([overlap]))
    assert both[0, 2] > 0.99 and both[0, 4] > 0.99  # both objects are switched at that time
    for time_index in (overlap, 10, 199):
        own = controller.state_activations(torch.tensor([time_index]))
        for slot in (1, 2):
            state = windows.flip_one_state(controller, time_index, slot)
            other = 3 - slot
            assert state[0, 2 * slot - 1] == 1.0 and state[0, 2 * slot] == 0.0
            assert torch.equal(state[0, 2 * other - 1 : 2 * other + 1], own[0, 2 * other - 1 : 2 * other + 1])
            assert state[0, 0] == own[0, 0]


def test_frame_roles_with_overlapping_windows():
    w = np.array([[10, 20], [15, 25]], dtype=np.float32)
    assert windows.frame_role(w, 1, 12) == "label"
    assert windows.frame_role(w, 1, 17) == "label"  # both objects switched: still a frame of slot 1
    assert windows.frame_role(w, 1, 22) is None  # only the other object: neither label nor control
    assert windows.frame_role(w, 1, 30) == "control"
    assert windows.frame_role(w, 2, 12) is None
    assert windows.frame_role(w, 2, 17) == "label"
    assert windows.frame_role(w, 2, 5) == "control"
    assert windows.active_slots(w, 20) == [1, 2] and windows.active_slots(w, 26) == []


def test_slot_frames_and_evenly_spaced_controls():
    w = np.array([[10, 20], [15, 25.5]], dtype=np.float32)
    train = [t for t in range(40) if t not in (3, 17)]
    labelled, controls = windows.select_slot_frames(train, w, 1, n_controls=5)
    assert labelled == [10, 11, 12, 13, 14, 15, 16, 18, 19, 20]
    # 23 frames lie outside both windows; indices round(i * 22 / 4) = 0, 6, 11, 16, 22
    assert controls == [0, 7, 28, 33, 39]
    labelled, controls2 = windows.select_slot_frames(train, w, 2, n_controls=5)
    assert labelled == [15, 16, 18, 19, 20, 21, 22, 23, 24, 25]
    assert controls2 == controls  # controls do not depend on the slot
    _, few = windows.select_slot_frames([0, 1, 12, 30], w, 1, n_controls=5)
    assert few == [0, 1, 30]


def test_training_frames_follow_the_loader_split(tmp_path: Path):
    for name in (
        "frame_0001.png",
        "frame_0002.png",
        "frame_0003.png",
        "frame_0004.png",
        "frame_test_novel_view_000_0003.png",
        "frame_test_novel_combo_011_0001.png",
        "notes.txt",
    ):
        (tmp_path / "images_2" / "camera1").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images_2" / "camera1" / name).touch()
    (tmp_path / "test_frames.txt").write_text(
        "camera1/frame_0002.png\ncamera2/frame_0004.png\ncamera1/frame_test_novel_view_000_0003.png\n"
    )
    # Frame 2 is a test frame of camera1; frame 4 is a test frame of camera2 only; novel-view images never train.
    assert training_frames(tmp_path, "camera1", "images_2") == [1, 3, 4]


def test_settle_margin_uses_learned_windows():
    times = np.arange(90, 131)
    kept = windows.settle_keep(times, np.array([[100.0, 120.0]], dtype=np.float32), 1, 5)
    assert times[kept].tolist() == list(range(105, 116))  # inside [100, 120], 5 frames from both ends


def test_settle_margin_respects_every_window():
    w = np.array([[100.0, 140.0], [118.5, 160.0]], dtype=np.float32)
    times = np.arange(95, 146)
    kept = windows.settle_keep(times, w, 1, 5)
    assert times[kept].tolist() == list(range(105, 114)) + list(range(124, 136))
    assert windows.settle_reason(97, w, 1, 5) == "outside slot01 window [100.00, 140.00]"
    assert windows.settle_reason(102, w, 1, 5) == "2.00 frames from slot01 start 100.00"
    assert windows.settle_reason(114, w, 1, 5) == "4.50 frames from slot02 start 118.50"
    assert windows.settle_reason(138, w, 1, 5) == "2.00 frames from slot01 end 140.00"


def test_settled_copy_drops_rows_and_loads_in_the_fine_tune(tmp_path: Path, mask_run, reader):
    scene = tmp_path / "scene"
    times = {1: list(range(10, 31)), 2: list(range(25, 51))}
    mask_run(scene, times)
    (scene / "windows.json").write_text(json.dumps({"source": "file", "windows": [[10, 30], [25, 50]]}))
    settle_masks.main(["--scene-dir", str(scene), "--margin", "3"])

    expected = {1: list(range(13, 23)), 2: list(range(33, 48))}  # 3 frames from 10, 25, 30 and 50
    record = json.loads((scene / "settled" / "DROP_TRANSITIONS.json").read_text())
    assert record["settle_margin"] == 3 and record["windows"] == [[10, 30], [25, 50]]
    for slot, kept_times in expected.items():
        name = f"slot{slot:02d}"
        keep = np.isin(times[slot], kept_times)
        assert record["slots"][name]["kept"] == len(kept_times)
        assert record["slots"][name]["dropped"] == [t for t in times[slot] if t not in kept_times]
        assert len(record["slots"][name]["reasons"]) == len(record["slots"][name]["dropped"])
        for part in ("mask/masks.npz", "labels/labels.npz"):
            with np.load(scene / name / part) as source, np.load(scene / "settled" / name / part) as settled:
                assert sorted(source.files) == sorted(settled.files)
                assert settled["time_indices"].tolist() == kept_times
                for key in source.files:
                    if key == "trusted":
                        new_row = np.cumsum(keep) - 1
                        assert settled[key].tolist() == [int(new_row[i]) for i in source[key] if keep[i]]
                    elif source[key].ndim:
                        assert np.array_equal(settled[key], source[key][keep])
                    else:
                        assert settled[key] == source[key]

    loaded = reader(scene / "settled", 2)
    assert sorted(loaded.slots_of_time) == expected[1] + expected[2]
    with np.load(scene / "slot02" / "mask" / "masks.npz") as source:
        row = loaded.row_of_time[(2, 40)]
        assert np.array_equal(loaded.masks[row].numpy(), source["pred_masks_sam"][times[2].index(40)] > 0.5)
