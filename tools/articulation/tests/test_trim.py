# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the part body trim: floaters that register by chance leave the part and become vanish."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from tools.articulation import estimate_joint, make_synthetic_scene
from tools.articulation.geometry import apply, hinge_transform
from tools.articulation.ply_io import Splats, read_ply, write_ply


def splats_at(points: np.ndarray, alpha: float = 0.9, scale: float = 0.01, seed: int = 0) -> Splats:
    n = len(points)
    rng = np.random.default_rng(seed)
    logit = np.log(alpha / (1.0 - alpha))
    return Splats(points, rng.normal(size=(n, 3)) * 0.1, np.zeros((n, 0)), np.full((n, 1), logit),
                  np.full((n, 3), np.log(scale)), np.tile([1.0, 0, 0, 0], (n, 1)))


def test_trim_body_removes_floaters_and_keeps_the_panel():
    rng = np.random.default_rng(0)
    panel = np.column_stack([rng.uniform(0, 1.4, 1500), rng.uniform(0, 1.0, 1500), rng.normal(scale=0.004, size=1500)])
    far = rng.uniform([-3, -3, -3], [4, 4, 4], size=(40, 3))
    far = far[np.linalg.norm(far - [0.7, 0.5, 0], axis=1) > 1.5]  # genuinely outside the panel
    wisp = np.column_stack([np.linspace(1.6, 2.6, 6), np.full(6, 0.5), np.zeros(6)])  # a thin chain off the edge
    points = np.concatenate([panel, far, wisp])
    splats = splats_at(points)
    keep = estimate_joint.trim_body(splats, np.ones(len(points), dtype=bool))
    assert keep[:1500].mean() > 0.98
    assert not keep[1500:].any()


def test_trim_body_is_scale_aware_and_ignores_faint_large_splats():
    rng = np.random.default_rng(1)
    # a sparse "car panel": 400 gaussians over 2 m x 1.5 m (spacing ~ 0.1), far looser than a cabinet door
    panel = np.column_stack([rng.uniform(0, 2.0, 400), rng.uniform(0, 1.5, 400), rng.normal(scale=0.01, size=400)])
    solid = splats_at(panel, alpha=0.9, scale=0.03)
    haze = splats_at(rng.uniform([-1, -1, -1], [3, 2.5, 1], size=(60, 3)), alpha=0.04, scale=0.5, seed=2)
    near_haze = splats_at(panel[:20] + [0.0, 0.0, 0.02], alpha=0.04, scale=0.5, seed=3)  # faint splats lying on the panel
    allsplats = Splats.concat([solid, haze, near_haze])
    keep = estimate_joint.trim_body(allsplats, np.ones(len(allsplats), dtype=bool))
    assert keep[:400].mean() > 0.95  # the sparse panel survives: the link radius follows its own spacing
    assert keep[460:].mean() > 0.8  # faint ones on the body stay (the closed look needs them)
    assert keep[400:460].mean() < 0.35  # faint, oversized haze away from it goes
    # tiny inputs are returned untouched
    tiny = splats_at(panel[:5])
    assert estimate_joint.trim_body(tiny, np.ones(5, dtype=bool)).all()
    # only candidates can be kept
    cand = np.zeros(len(allsplats), dtype=bool)
    cand[:200] = True
    assert not estimate_joint.trim_body(allsplats, cand)[200:].any()


@pytest.fixture(scope="module")
def door_with_floaters(tmp_path_factory):
    """The synthetic scene plus floaters that exist in both the closed and the open pose (so they register)."""
    clean = tmp_path_factory.mktemp("trim_clean")
    make_synthetic_scene.build_scene(clean, seed=3)
    dirty = tmp_path_factory.mktemp("trim_dirty")
    shutil.copytree(clean, dirty, dirs_exist_ok=True)
    truth = json.loads((clean / "ground_truth.json").read_text())["slot01"]

    base, switched = read_ply(clean / "slot01_base.ply"), read_ply(clean / "slot01_switched.ply")
    centre = base.positions.mean(axis=0)
    rng = np.random.default_rng(9)
    spots = rng.uniform(-1.5, 1.5, size=(60, 3)) + centre
    spots = spots[np.linalg.norm(spots - centre, axis=1) > 0.9]  # well off the door (about 0.7 m across)
    rot, trans = hinge_transform(np.array(truth["axis"]), np.array(truth["pivot"]), np.deg2rad(truth["open"]))
    floaters_base = splats_at(spots, alpha=0.5, scale=0.05, seed=4)
    floaters_open = splats_at(apply(rot, trans, spots) + rng.normal(scale=0.002, size=spots.shape), alpha=0.5,
                              scale=0.05, seed=5)
    write_ply(dirty / "slot01_base.ply", Splats.concat([base, floaters_base]))
    write_ply(dirty / "slot01_switched.ply", Splats.concat([switched, floaters_open]))
    return clean, dirty, spots, len(base)


def run_slot01(scene_dir, **kwargs):
    scene = json.loads((scene_dir / "scene.json").read_text())
    part = scene["parts"][0]
    estimate_joint.process_part(scene_dir, part, None, up=scene["up"], **kwargs)
    return part


def test_floaters_end_up_in_vanish_and_the_joint_is_unchanged(door_with_floaters, tmp_path):
    clean, dirty, spots, n_door = door_with_floaters
    clean_work, dirty_work = tmp_path / "clean", tmp_path / "dirty"
    shutil.copytree(clean, clean_work)
    shutil.copytree(dirty, dirty_work)
    reference = run_slot01(clean_work)
    part = run_slot01(dirty_work)

    kept = read_ply(dirty_work / part["files"]["part"]).positions.astype(np.float64)
    vanish = read_ply(dirty_work / part["files"]["vanish"]).positions.astype(np.float64)
    assert len(kept) >= 0.9 * len(read_ply(clean_work / reference["files"]["part"]))  # the door stays whole

    def counted(points):  # how many of the planted floaters appear in ``points``
        d = np.linalg.norm(points[:, None, :] - spots[None], axis=2).min(axis=0)
        return int((d < 1e-4).sum())

    assert counted(kept) <= 0.05 * len(spots)
    assert counted(vanish) >= 0.9 * len(spots)  # still in the file: visible when closed, fading as the door opens

    j, r = part["joint"], reference["joint"]
    assert j["kind"] == r["kind"] == "hinge"
    cos = abs(np.dot(j["axis"], r["axis"]))
    assert np.degrees(np.arccos(min(cos, 1.0))) < 1.0
    assert abs(j["open"] - r["open"]) < 1.5
    assert j["fit"]["inlierRatio"] > 0.8


def test_trim_is_optional_and_reprocessing_is_idempotent(door_with_floaters, tmp_path):
    _, dirty, spots, _ = door_with_floaters
    work = tmp_path / "again"
    shutil.copytree(dirty, work)
    first = run_slot01(work)
    sizes = {k: len(read_ply(work / first["files"][k])) for k in ("part", "vanish", "reveal")}
    second = run_slot01(work)  # re-derived from base/switched, nothing compounds
    assert {k: len(read_ply(work / second["files"][k])) for k in sizes} == sizes
    assert second["joint"]["open"] == pytest.approx(first["joint"]["open"], abs=0.5)
    untrimmed = run_slot01(work, trim=False)
    assert len(read_ply(work / untrimmed["files"]["part"])) > sizes["part"]  # floaters stay in the part without trim
