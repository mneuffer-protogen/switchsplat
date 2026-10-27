# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of slot merging and of the fit priors added for real scenes (cluster filter, axis / pivot snaps)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.articulation import estimate_joint, make_synthetic_scene
from tools.articulation.geometry import hinge_transform, point_to_line_distance
from tools.articulation.ply_io import Splats, read_ply


def make_splats(points: np.ndarray, seed: int = 0) -> Splats:
    n = len(points)
    rng = np.random.default_rng(seed)
    return Splats(points, rng.normal(size=(n, 3)) * 0.1, np.zeros((n, 0)), np.zeros((n, 1)), np.full((n, 3), -4.0),
                  np.tile([1.0, 0, 0, 0], (n, 1)))


def sheet(rng, x0, x1, y0, y1, count, z=0.0):
    return np.column_stack([rng.uniform(x0, x1, count), rng.uniform(y0, y1, count), np.full(count, z)])


# --------------------------------------------------------------------------------------------
# Compatibility rule
# --------------------------------------------------------------------------------------------

def hinge(axis=(0, 1, 0), pivot=(0, 0, 0), open_deg=90.0, inlier=0.9):
    return {"kind": "hinge", "axis": list(axis), "pivot": list(pivot), "min": 0, "max": open_deg, "open": open_deg,
            "fit": {"rmse": 0.004, "inlierRatio": inlier, "screw": 0.001}}


def test_joints_compatible_rules():
    rng = np.random.default_rng(0)
    base = sheet(rng, 0, 1, 0, 1, 800)
    other = sheet(rng, 0, 1, 0, 1, 800)  # same closed pose, resampled
    far = sheet(rng, 3, 4, 0, 1, 800)
    ok = estimate_joint.joints_compatible
    assert ok(hinge(), hinge(), base, other)
    tilted = np.array([np.sin(np.radians(5)), np.cos(np.radians(5)), 0])
    assert ok(hinge(), hinge(axis=tilted), base, other)  # 5 deg < 8 deg
    tilted = np.array([np.sin(np.radians(12)), np.cos(np.radians(12)), 0])
    assert not ok(hinge(), hinge(axis=tilted), base, other)
    assert not ok(hinge(), hinge(axis=(0, -1, 0)), base, other)  # opposite sign: not the same open direction
    assert not ok(hinge(), hinge(pivot=(0.2, 0, 0)), base, other)  # lines 20 cm apart
    assert ok(hinge(), hinge(pivot=(0.05, 0.3, 0.0)), base, other)  # 5 cm apart; sliding along the line is free
    assert not ok(hinge(), hinge(), base, far)  # same joint on paper, but a different part elsewhere
    slider = {"kind": "slider", "axis": [0, 0, 1], "pivot": [0, 0, 0], "min": 0, "max": 0.3, "open": 0.3,
              "fit": {"rmse": 0.004, "inlierRatio": 0.9, "screw": 0}}
    assert ok(slider, dict(slider, open=0.35), base, other)
    assert not ok(slider, dict(slider), base, far)
    assert not ok(slider, hinge(), base, other)
    assert not ok(None, hinge(), base, other)


def test_union_dedupes_by_position():
    rng = np.random.default_rng(1)
    pts = sheet(rng, 0, 1, 0, 1, 300)
    first = make_splats(pts)
    second = make_splats(np.concatenate([pts[:100] + 0.001, sheet(rng, 2, 3, 0, 1, 50)]), seed=1)  # 100 duplicates, 50 new
    merged = estimate_joint.union_splats(first, second, tol=0.005)
    assert len(merged) == 350
    assert estimate_joint.union_splats(None, second) is second and estimate_joint.union_splats(first, None) is first


# --------------------------------------------------------------------------------------------
# Merging the synthetic door and a resampled duplicate of it
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def duplicated_door(tmp_path_factory):
    """slot01 door, slot02 drawer and slot03 = the same door sampled again (a second opening episode)."""
    scene_dir = tmp_path_factory.mktemp("merge_scene")
    make_synthetic_scene.build_scene(scene_dir, seed=3)
    other = tmp_path_factory.mktemp("merge_other")
    make_synthetic_scene.build_scene(other, seed=11)
    shutil.copy(other / "slot01_base.ply", scene_dir / "slot03_base.ply")
    shutil.copy(other / "slot01_switched.ply", scene_dir / "slot03_switched.ply")
    scene = json.loads((scene_dir / "scene.json").read_text())
    scene["parts"].append({"id": "slot03", "name": "cabinet door, second episode",
                           "files": {"base": "slot03_base.ply", "switched": "slot03_switched.ply", "part": None,
                                     "reveal": None, "vanish": None}, "joint": None})
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=2))
    for part in scene["parts"]:
        estimate_joint.process_part(scene_dir, part, None, up=scene["up"])
    (scene_dir / "scene.json").write_text(json.dumps(scene, indent=2))
    return scene_dir


def test_duplicate_door_is_merged(duplicated_door, tmp_path):
    work = tmp_path / "scene"
    shutil.copytree(duplicated_door, work)
    scene = json.loads((work / "scene.json").read_text())
    before = {p["id"]: p for p in scene["parts"]}
    assert len(before) == 3 and before["slot03"]["joint"]["kind"] == "hinge"
    n1 = len(read_ply(work / "slot01_part.ply"))
    n3 = len(read_ply(work / "slot03_part.ply"))

    lines = estimate_joint.merge_parts(work, scene)
    assert len(lines) == 1 and "slot03" in lines[0]
    ids = [p["id"] for p in scene["parts"]]
    assert ids == ["slot01", "slot02"]  # the drawer is a different joint and stays
    door = scene["parts"][0]
    assert door["merged"] == ["slot03"] and "merged" not in scene["parts"][1]
    assert door["joint"]["kind"] == "hinge" and abs(door["joint"]["open"] - 95.0) < 3.0
    assert door["joint"]["max"] == door["joint"]["open"]
    assert np.dot(door["joint"]["axis"], [0, -1, 0]) > 0.99

    merged_part = read_ply(work / door["files"]["part"])
    assert max(n1, n3) < len(merged_part) <= n1 + n3  # union, not either set alone
    assert not (work / "slot03_part.ply").exists() and not (work / "slot03_base.ply").exists()
    assert not (work / "slot03_reveal.ply").exists()
    # the union is deduplicated: no two gaussians of one slot's set closer than the tolerance are added twice
    first = read_ply(duplicated_door / "slot01_part.ply")
    added = len(merged_part) - len(first)
    assert 0 < added <= n3
    assert len(read_ply(work / door["files"]["base"])) > len(read_ply(duplicated_door / "slot01_base.ply"))
    assert door["files"]["reveal"] is not None  # reveal sets are united as well
    assert door["joint"]["fit"]["inlierRatio"] > 0.9


def test_merge_flag_on_the_command_line(monkeypatch, tmp_path):
    scene_dir = tmp_path / "s"
    scene_dir.mkdir()
    scene = {"version": 1, "up": [0, 1, 0], "static": "static.ply",
             "parts": [{"id": "slot01", "files": {}, "joint": None}]}
    (scene_dir / "scene.json").write_text(json.dumps(scene))
    calls = []
    monkeypatch.setattr(estimate_joint, "process_part", lambda *a, **k: "stub")
    monkeypatch.setattr(estimate_joint, "merge_parts", lambda *a, **k: calls.append(a) or [])
    estimate_joint.main([str(scene_dir)])
    assert len(calls) == 1  # default on
    estimate_joint.main([str(scene_dir), "--no-merge"])
    assert len(calls) == 1
    estimate_joint.main([str(scene_dir), "--merge"])
    assert len(calls) == 2


# --------------------------------------------------------------------------------------------
# Fit priors
# --------------------------------------------------------------------------------------------

def test_largest_cluster_drops_stray_floaters():
    rng = np.random.default_rng(0)
    body = sheet(rng, 0, 1, 0, 1, 1500)
    strays = np.array([[3.0, 3.0, 0.5], [3.02, 3.0, 0.5], [-2.0, 1.0, 1.0]])
    mask = estimate_joint.largest_cluster(np.concatenate([body, strays]))
    assert mask[:1500].all() and not mask[1500:].any()
    assert estimate_joint.largest_cluster(body[:5]).all()  # too small to cluster: everything is kept


def test_gravity_snap_makes_the_door_axis_exactly_vertical():
    with tempfile.TemporaryDirectory() as tmp:
        make_synthetic_scene.build_scene(Path(tmp), seed=5)
        base, switched = read_ply(Path(tmp) / "slot01_base.ply"), read_ply(Path(tmp) / "slot01_switched.ply")
    fit = estimate_joint.fit_part(base, switched, 0, up=[0.0, 1.0, 0.0])
    assert fit.kind == "hinge"
    np.testing.assert_allclose(fit.axis, [0, -1, 0], atol=1e-6)  # snapped: exactly (anti)parallel to up, open positive
    assert 90 < fit.open < 100
    assert point_to_line_distance(fit.pivot, make_synthetic_scene.DOOR_HINGE_PIVOT, [0, 1, 0]) < 0.015


def test_pivot_snap_puts_the_hinge_in_the_plane_of_a_thin_sheet():
    """A narrow strip registered on a wide open sheet leaves the pivot free along the bisector; the snap fixes it."""
    rng = np.random.default_rng(0)
    axis, pivot, angle = np.array([0.0, 1.0, 0.0]), np.zeros(3), np.radians(40.0)
    strip = sheet(rng, 0.0, 1.5, 0.0, 0.3, 1200)  # closed: the plane z = 0, hinge line = the y axis at x = 0, z = 0
    wide = sheet(rng, 0.0, 1.5, -0.6, 1.0, 5000)
    rot, trans = hinge_transform(axis, pivot, angle)
    open_sheet = wide @ rot.T + trans
    matcher = estimate_joint.Matcher(make_splats(strip), make_splats(open_sheet, 1), np.random.default_rng(0))

    # slide the pivot along the family of equally good centres (the angle bisector of the two planes)
    n1 = rot @ np.array([0.0, 0.0, 1.0])
    v = (np.eye(3) - rot).T @ n1  # centres c with v . c = 0 keep the closed sheet on the open one
    direction = np.array([-v[2], 0.0, v[0]]) / np.hypot(v[0], v[2])
    wrong = pivot + 0.5 * direction
    r, t = hinge_transform(axis, wrong, angle)
    wrong_fit = estimate_joint.PartFit("hinge", axis, wrong, float(np.degrees(angle)), r, t, 0.0, matcher.score(r, t))
    assert matcher.score(r, t) < 0.02  # the wrong pivot registers the strip just as well (random colours floor the score)

    snapped = estimate_joint.snap_pivot_to_plane(matcher, wrong_fit)
    # the hinge line passes (almost) through the true one: distance of the snapped pivot to the y axis
    assert np.hypot(snapped.pivot[0], snapped.pivot[2]) < 0.03
    assert abs(snapped.open - 40.0) < 2.0
    # a pivot that is already in the plane is left alone
    good_fit = estimate_joint.PartFit("hinge", axis, pivot, 40.0, rot, trans, 0.0, matcher.score(rot, trans))
    assert estimate_joint.snap_pivot_to_plane(matcher, good_fit) is good_fit
    # non-hinges and parts whose axis stands on the sheet are not touched
    slider = estimate_joint.PartFit("slider", axis, pivot, 0.3, np.eye(3), np.zeros(3), 0.0, 0.0)
    assert estimate_joint.snap_pivot_to_plane(matcher, slider) is slider
    upright = estimate_joint.PartFit("hinge", np.array([0.0, 0.0, 1.0]), wrong, 40.0, r, t, 0.0, 0.0)
    assert estimate_joint.snap_pivot_to_plane(matcher, upright) is upright
