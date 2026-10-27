# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-part pipeline options (the car's hood): reveal_above, fog_alpha, and their CLI / scene.json plumbing."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from tools.articulation import estimate_joint, make_synthetic_scene, recruit
from tools.articulation.ply_io import Splats, read_ply, write_ply


@pytest.fixture(scope="module")
def door_with_air(tmp_path_factory):
    """The synthetic cabinet, with 40 opaque blue floaters far above the door's closed surface in the switched set."""
    out = tmp_path_factory.mktemp("opt")
    make_synthetic_scene.build_scene(out, seed=3)
    switched = read_ply(out / "slot01_switched.ply")
    rng = np.random.default_rng(1)
    # the door lies in z ~ 0.41; "above" is +z (scene_up below). The open door has swung away, so put the floaters where
    # the closed door's volume is nowhere near: x, y over the door, z a metre up.
    air = np.column_stack([rng.uniform(-0.3, 0.3, 40), rng.uniform(0.4, 0.9, 40), rng.uniform(1.3, 1.6, 40)])
    floaters = Splats(air, np.tile((np.array([0.1, 0.2, 0.9]) - 0.5) / 0.2820948, (40, 1)), np.zeros((40, 0)),
                      np.full((40, 1), 3.0), np.full((40, 3), np.log(0.03)), np.tile([1.0, 0, 0, 0], (40, 1)))
    write_ply(out / "slot01_switched.ply", Splats.concat([switched, floaters]))
    return out, len(read_ply(out / "slot01_switched.ply")) - 40


def run(scene_dir, **kwargs):
    scene = json.loads((scene_dir / "scene.json").read_text())
    part = scene["parts"][0]
    estimate_joint.process_part(scene_dir, part, None, up=scene["up"], scene_up=[0.0, 0.0, 1.0], **kwargs)
    return part


def test_reveal_above_drops_air_above_the_closed_surface_only(door_with_air, tmp_path):
    src, _ = door_with_air
    work = tmp_path / "w"
    shutil.copytree(src, work)
    plain = run(work)
    n_plain = len(read_ply(work / plain["files"]["reveal"]))
    pos_plain = read_ply(work / plain["files"]["reveal"]).positions
    cut = run(work, options={"reveal_above": 0.12})
    reveal = read_ply(work / cut["files"]["reveal"])
    assert n_plain - len(reveal) >= 35  # the floaters are gone
    assert (pos_plain[:, 2] > 1.2).sum() >= 35 and (reveal.positions[:, 2] > 1.2).sum() == 0
    assert len(reveal) > 0.9 * (n_plain - 40)  # the content behind the door (inside the cabinet) stays
    assert cut["joint"]["open"] == pytest.approx(plain["joint"]["open"], abs=1e-6)  # the fit is untouched


def test_height_above_signs_and_units():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 1, 800), rng.uniform(0, 1, 800), rng.normal(scale=0.002, size=800)])
    skin = Splats(pts, np.zeros((800, 3)), np.zeros((800, 0)), np.zeros((800, 1)), np.zeros((800, 3)), np.tile([1.0, 0, 0, 0], (800, 1)))
    probe = np.array([[0.5, 0.5, 0.3], [0.5, 0.5, -0.2], [0.2, 0.8, 0.0]])
    dh = recruit.height_above(skin, probe, metres_per_unit=0.5, up=[0, 0, 1])
    assert dh[0] == pytest.approx(0.15, abs=0.01) and dh[1] == pytest.approx(-0.10, abs=0.01) and abs(dh[2]) < 0.01
    assert recruit.height_above(skin, probe, 0.5, up=[0, 0, -1])[0] == pytest.approx(-0.15, abs=0.01)


def test_fog_option_cleans_the_open_pose_copy_of_a_translucent_part(door_with_air, tmp_path):
    src, _ = door_with_air
    work = tmp_path / "f"
    shutil.copytree(src, work)
    for name in ("slot01_base.ply", "slot01_switched.ply"):  # glass-like: opacity 0.3 -> inner mode attach
        s = read_ply(work / name)
        write_ply(work / name, Splats(s.positions, s.f_dc, s.f_rest, np.full_like(s.opacity, -0.85), s.scale, s.rot))
    base_run = run(work)
    assert base_run["stats"]["innerMode"] == "attach"
    n_attached = len(read_ply(work / "slot01_partReveal.ply"))
    # make a third of the copy large and faint: fog
    s = read_ply(work / "slot01_switched.ply")
    big = np.arange(len(s)) % 3 == 0
    scale = s.scale.copy()
    scale[big] += np.log(6.0)
    opacity = s.opacity.copy()
    opacity[big] = -2.5
    write_ply(work / "slot01_switched.ply", Splats(s.positions, s.f_dc, s.f_rest, opacity, scale, s.rot))
    foggy = run(work)
    n_foggy, reveal_foggy = len(read_ply(work / "slot01_partReveal.ply")), len(read_ply(work / foggy["files"]["reveal"]))
    clean = run(work, options={"fog_alpha": 0.2})
    n_clean, reveal_clean = len(read_ply(work / "slot01_partReveal.ply")), len(read_ply(work / clean["files"]["reveal"]))
    assert reveal_clean < 0.85 * reveal_foggy  # a third of the reveal was large and faint
    assert n_foggy >= n_clean >= 0.4 * n_attached  # the open-pose copy keeps its body
    assert clean["stats"]["attached"] == n_clean and clean["stats"]["reveal"] == reveal_clean


def test_part_option_cli_stores_the_setting_and_removes_it(door_with_air, tmp_path, monkeypatch):
    src, _ = door_with_air
    work = tmp_path / "c"
    shutil.copytree(src, work)
    seen = []
    monkeypatch.setattr(estimate_joint, "process_part", lambda *a, **k: seen.append(a[1].get("pipeline")) or "stub")
    monkeypatch.setattr(estimate_joint, "recruit_scene", lambda *a, **k: [])
    estimate_joint.main([str(work), "--slots", "1", "--no-merge", "--part-option", "slot01:reveal_above=0.12",
                         "--part-option", "slot01:fog_alpha=0.25"])
    assert seen[-1] == {"reveal_above": 0.12, "fog_alpha": 0.25}
    assert json.loads((work / "scene.json").read_text())["parts"][0]["pipeline"] == {"reveal_above": 0.12, "fog_alpha": 0.25}
    estimate_joint.main([str(work), "--slots", "1", "--no-merge"])  # later runs keep it
    assert seen[-1] == {"reveal_above": 0.12, "fog_alpha": 0.25}
    estimate_joint.main([str(work), "--slots", "1", "--no-merge", "--part-option", "slot01:reveal_above=none",
                         "--part-option", "slot01:fog_alpha=none"])
    assert "pipeline" not in json.loads((work / "scene.json").read_text())["parts"][0]
    with pytest.raises(SystemExit):
        estimate_joint.main([str(work), "--part-option", "slot99:reveal_above=1"])
