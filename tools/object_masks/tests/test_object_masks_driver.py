# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The driver on CPU: real windows, frame selection, settling and metrics; GPU stages parse their
arguments with their own command line and leave stand-in outputs."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.model.switch_splat_controller import SwitchSplatController
from tools.object_masks import make_masks as run_scene

CPU_STAGES = {"windows", "select_times", "settle_masks", "finalize_scene"}


class Parsed(Exception):
    pass


def _fake_scene(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = OmegaConf.create(
        {"num_slots": 2, "num_frames": 120, "init_windows": [[20, 45], [60, 90]], "temperature": 1}
    )
    controller = SwitchSplatController(config)
    checkpoint = {
        "switch_splat": controller.state_dict(),
        "config": OmegaConf.create(
            {"switch_splat": {"num_slots": 2, "num_frames": 120}, "dataset": {"downsample_factor": 2}}
        ),
    }
    torch.save(checkpoint, tmp_path / "ckpt.pt")
    data = tmp_path / "data"
    (data / "images_2" / "camera1").mkdir(parents=True)
    for time_index in range(120):
        (data / "images_2" / "camera1" / f"frame_{time_index:04d}.png").touch()
    (data / "test_frames.txt").write_text("".join(f"camera1/frame_{t:04d}.png\n" for t in range(0, 120, 8)))
    (tmp_path / "sam.pt").touch()
    return tmp_path / "ckpt.pt", data, tmp_path / "sam.pt"


def _stand_in_outputs(stage: str, args: argparse.Namespace) -> None:
    """What a GPU stage would leave behind, so the next stages can run."""
    rng = np.random.default_rng(0)
    if stage == "render_pairs":
        for slot in args.slot:
            (args.scene_dir / f"slot{slot:02d}" / "pairs").mkdir(parents=True)
            (args.scene_dir / f"slot{slot:02d}" / "pairs" / "manifest.csv").write_text("time_index\n")
    elif stage == "make_slot_labels":
        windows = json.loads((args.scene_dir / "windows.json").read_text())["windows"]
        start, end = windows[args.slot - 1]
        directory = args.scene_dir / f"slot{args.slot:02d}"
        times = [int(t) for t in (directory / "render_times.txt").read_text().split(",") if start <= int(t) <= end]
        (directory / "labels").mkdir()
        shape = (len(times), 4, 4)
        np.savez_compressed(
            directory / "labels" / "labels.npz",
            valid=np.ones(shape, np.uint8),
            time_indices=np.asarray(times, np.int32),
            image_names=np.asarray([f"frame_{t:04d}" for t in times]),
        )
    elif stage == "train_slot_mask":
        with np.load(args.labels_dir / "labels.npz") as labels:
            times = labels["time_indices"]
        args.out_dir.mkdir(parents=True)
        shape = (len(times), 4, 4)
        np.savez_compressed(
            args.out_dir / "masks.npz",
            pred_masks=rng.random(shape).astype(np.float32),
            pred_masks_sam=(rng.random(shape) > 0.5).astype(np.float32),
            time_indices=times,
        )


def test_driver_runs_every_stage_and_resumes(tmp_path: Path, monkeypatch, reader):
    checkpoint, data, sam = _fake_scene(tmp_path)
    out = tmp_path / "out"
    calls = []
    parse_args = argparse.ArgumentParser.parse_args

    def fake_run(stage, *args):
        argv = [str(arg) for arg in args]
        module = importlib.import_module(f"tools.object_masks.{stage}")
        if stage in CPU_STAGES:
            calls.append((stage, None))
            module.main(argv)
            return

        def parse_then_stop(self, args=None, namespace=None):
            raise Parsed(parse_args(self, args, namespace))

        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
        try:
            module.main(argv)
        except Parsed as parsed:
            calls.append((stage, parsed.args[0]))
        finally:
            monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
        _stand_in_outputs(stage, calls[-1][1])

    monkeypatch.setattr(run_scene, "run", fake_run)
    argv = ["--checkpoint", str(checkpoint), "--data", str(data), "--out", str(out), "--sam-checkpoint", str(sam)]
    run_scene.main(argv + ["--no-previews"])

    stages = [stage for stage, _ in calls]
    assert stages == [
        "windows",
        "select_times",
        "render_pairs",
        "make_slot_labels",
        "make_slot_labels",
        "extract_dinov3_cache",
        "train_slot_mask",
        "train_slot_mask",
        "settle_masks",
        "finalize_scene",
    ]
    parsed = {stage: args for stage, args in calls if args is not None}
    assert parsed["render_pairs"].slot == [1, 2] and parsed["render_pairs"].checkpoint == checkpoint.resolve()
    assert parsed["make_slot_labels"].high_quantile == 0.995 and parsed["make_slot_labels"].no_previews
    assert parsed["extract_dinov3_cache"].image_dir == data.resolve() / "images_2" / "camera1"
    assert len(parsed["extract_dinov3_cache"].labels) == 2
    assert parsed["train_slot_mask"].seed == 0 and parsed["train_slot_mask"].sam_checkpoint == sam.resolve()

    record = json.loads((out / "settled" / "DROP_TRANSITIONS.json").read_text())
    assert record["windows_source"] == "learned" and record["settle_margin"] == 10
    assert (out / "metrics.csv").is_file() and (out / "slot02" / "mask" / ".complete").is_file()
    settled = reader(out / "settled", 2)
    for slot, time_index in settled.row_of_time:
        start, end = record["windows"][slot - 1]
        assert start + 10 <= time_index <= end - 10

    calls.clear()  # a rerun only refreshes the metrics
    run_scene.main(argv + ["--no-previews"])
    assert [stage for stage, _ in calls] == ["finalize_scene"]
    with pytest.raises(SystemExit):  # other settings need a new output folder
        run_scene.main(argv + ["--no-previews", "--high-quantile", "0.99"])
