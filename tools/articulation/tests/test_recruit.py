# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skin recruitment: static gaussians on a moving panel's closed skin join the part (and leave static.ply)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from tools.articulation import estimate_joint, export_parts, make_synthetic_scene, quality_report, recruit
from tools.articulation.ply_io import Splats, read_ply, write_ply


def splats(points, rgb, scale=0.01, alpha=0.9, normal_axis=2):
    n = len(points)
    quat = np.tile([1.0, 0, 0, 0], (n, 1))
    log_scale = np.tile(np.log([scale, scale, scale * 0.15])[None], (n, 1))  # flat gaussians, thin axis = local z
    f_dc = np.tile((np.asarray(rgb, dtype=np.float64) - 0.5) / 0.2820948, (n, 1))
    return Splats(points, f_dc, np.zeros((n, 0)), np.full((n, 1), np.log(alpha / (1 - alpha))), log_scale, quat)


def panel_points(rng, n, x=(0.0, 1.0), y=(0.0, 0.8), z=0.0):
    return np.column_stack([rng.uniform(*x, n), rng.uniform(*y, n), z + rng.normal(scale=0.003, size=n)])


@pytest.fixture()
def scene():
    """A 1 x 0.8 panel in z=0. The part has the left and right thirds, the middle band is labelled static; around it:
    a fender next to the panel (z=0 too, beyond its edge), the cabin 15 cm behind it, rubber on it, a floor."""
    rng = np.random.default_rng(0)
    left = panel_points(rng, 500, x=(0.0, 0.33))
    right = panel_points(rng, 500, x=(0.67, 1.0))
    part = splats(np.concatenate([left, right]), (0.2, 0.3, 0.8))
    band = splats(panel_points(rng, 300, x=(0.35, 0.65)), (0.22, 0.32, 0.78))
    fender = splats(panel_points(rng, 200, x=(1.15, 1.6)), (0.2, 0.3, 0.8))  # same plane and colour, 15 cm past the edge
    cabin = splats(panel_points(rng, 300, z=-0.15), (0.2, 0.3, 0.8))
    rubber = splats(panel_points(rng, 60, x=(0.4, 0.6)), (0.02, 0.02, 0.02))
    floor = splats(np.column_stack([rng.uniform(-1, 3, 300), np.full(300, -0.3), rng.uniform(-1, 1, 300)]), (0.5, 0.5, 0.5))
    static = Splats.concat([band, fender, cabin, rubber, floor])
    labels = np.concatenate([np.full(300, "band"), np.full(200, "fender"), np.full(300, "cabin"), np.full(60, "rubber"),
                             np.full(300, "floor")])
    return part, static, labels


def test_select_skin_takes_the_band_and_nothing_around_it(scene):
    part, static, labels = scene
    skin = recruit.select_skin(part, static, recruit.RecruitParams(metres_per_unit=1.0))
    taken = labels[skin.mask]
    assert (taken == "band").sum() >= 0.9 * 300  # the hole between the two thirds is filled
    assert (taken == "fender").sum() <= 0.1 * 200  # outside the footprint (3 cm margin only)
    assert (taken == "cabin").sum() == 0  # 15 cm behind the skin
    assert (taken == "floor").sum() == 0
    assert (taken == "rubber").sum() == 0  # black on blue paint: not the same surface
    assert skin.thickness < 0.02 and skin.reason is not None and (skin.reason == 4).sum() == skin.mask.sum()


def test_edge_margin_is_a_few_centimetres_only(scene):
    part, static, labels = scene
    rng = np.random.default_rng(3)
    edge = splats(np.column_stack([rng.uniform(1.0, 1.1, 200), rng.uniform(0.0, 0.8, 200), rng.normal(scale=0.003, size=200)]),
                  (0.2, 0.3, 0.8))
    both = Splats.concat([static, edge])
    x_edge = both.positions[len(static):, 0]
    mask = recruit.select_skin(part, both, recruit.RecruitParams(metres_per_unit=1.0)).mask[len(static):]
    assert mask[x_edge < 1.03].mean() > 0.8 and mask[x_edge > 1.06].mean() < 0.1
    wide = recruit.select_skin(part, both, recruit.RecruitParams(metres_per_unit=1.0, edge=0.12)).mask[len(static):]
    assert wide.mean() > mask.mean()


def test_orientation_gate_rejects_perpendicular_gaussians(scene):
    part, static, labels = scene
    rng = np.random.default_rng(4)
    standing = splats(panel_points(rng, 100, x=(0.4, 0.6)), (0.2, 0.3, 0.8))
    standing.rot = np.tile([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0], (100, 1)).astype(np.float32)  # flat axis along y
    both = Splats.concat([static, standing])
    assert recruit.select_skin(part, both, recruit.RecruitParams(metres_per_unit=1.0)).mask[len(static):].sum() < 10
    assert recruit.select_skin(part, both, recruit.RecruitParams(metres_per_unit=1.0, angle=90.0)).mask[len(static):].sum() > 80


def test_metres_per_unit_scales_the_lengths(scene):
    part, static, labels = scene
    big = lambda s: Splats(s.positions * 10, s.f_dc, s.f_rest, s.opacity, s.scale + np.log(10), s.rot)
    a = recruit.select_skin(part, static, recruit.RecruitParams(metres_per_unit=1.0)).mask
    b = recruit.select_skin(big(part), big(static), recruit.RecruitParams(metres_per_unit=0.1)).mask
    assert (a == b).mean() > 0.97


def test_ghost_mass_counts_static_alpha_on_the_surface(scene):
    part, static, labels = scene
    ghost = recruit.ghost_mass(part, static, 0.04)
    alpha = 1 / (1 + np.exp(-static.opacity[:, 0]))
    assert ghost == pytest.approx(alpha[labels == "band"].sum() + alpha[labels == "rubber"].sum(), rel=0.1)


# --------------------------------------------------------------------------------------------
# the pipeline on the synthetic scene
# --------------------------------------------------------------------------------------------

def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def run(scene_dir, recruit_on=True, only=("slot01",)):
    scene_path = scene_dir / "scene.json"
    scene = json.loads(scene_path.read_text())
    for part in scene["parts"]:
        if part["id"] in only:
            estimate_joint.process_part(scene_dir, part, None, up=scene["up"])
    if recruit_on:
        recruit.recruit_scene(scene_dir, scene, set(only), verbose=False)
    else:
        recruit.restore_static(scene_dir, scene, set(only))
    scene_path.write_text(json.dumps(scene, indent=2))
    return scene


@pytest.fixture(scope="module")
def static_skin_scene(tmp_path_factory):
    out = tmp_path_factory.mktemp("skin")
    truth = make_synthetic_scene.build_scene(out, seed=3, static_skin=True)
    return out, truth["slot01"]["static_skin"], len(read_ply(out / "static.ply"))


def copy(src, dst):
    import shutil

    shutil.copytree(src, dst)
    return dst


def test_synthetic_door_skin_moves_from_static_into_the_part(static_skin_scene, tmp_path):
    src, patch, n_static = static_skin_scene
    work = copy(src, tmp_path / "w")
    original = read_ply(work / "static.ply")
    scene = run(work)
    part_entry = scene["parts"][0]
    moved = part_entry["stats"]["skin"]
    assert patch > 100 and moved >= 0.7 * patch  # the colour gate keeps some of the differently coloured back face out
    after = read_ply(work / "static.ply")
    assert len(after) == n_static - moved  # rows leave static.ply, they are not duplicated
    assert len(read_ply(work / part_entry["files"]["part"])) >= part_entry["stats"]["partBeforeSkin"] + moved
    # what was taken is the band of the door, not the cabinet: nearly every removed row lies at the door's z
    gone = original.positions[~np.isin(np.round(original.positions, 5).view([("", np.float32)] * 3).ravel(),
                                       np.round(after.positions, 5).view([("", np.float32)] * 3).ravel())]
    assert len(gone) == moved and np.abs(gone[:, 2] - 0.41).max() < 0.03 and (gone[:, 0] > -0.3).all() and (gone[:, 0] < 0.3).all()
    # the ghost left behind in static when the part moves away has dropped
    base_part = read_ply(work / part_entry["files"]["part"])
    ghost_after = recruit.ghost_mass(base_part, after, 0.04)
    ghost_before = recruit.ghost_mass(base_part, original, 0.04)
    assert ghost_after < 0.5 * ghost_before  # (the colour gate leaves the differently coloured back face, and the cabinet is near)


def test_recruitment_is_idempotent_and_reversible(static_skin_scene, tmp_path):
    src, patch, n_static = static_skin_scene
    work = copy(src, tmp_path / "w")
    original_positions = np.sort(read_ply(work / "static.ply").positions.view(np.float32).reshape(-1))
    run(work)
    files = ["static.ply", "slot01_part.ply", "slot01_skin.ply", "slot01_vanish.ply"]
    first = {f: md5(work / f) for f in files if (work / f).exists()}
    scene_first = json.loads((work / "scene.json").read_text())
    run(work)  # the whole pipeline again, from the same base / switched / static state
    assert {f: md5(work / f) for f in first} == first
    assert json.loads((work / "scene.json").read_text()) == scene_first
    run(work, recruit_on=False)  # --no-recruit puts every row back
    assert not (work / "slot01_skin.ply").exists()
    restored = np.sort(read_ply(work / "static.ply").positions.view(np.float32).reshape(-1))
    assert np.array_equal(restored, original_positions)


def test_reimport_discards_stale_skin_files(tmp_path):
    (tmp_path / "slot01_skin.ply").write_bytes(b"stale")
    rng = np.random.default_rng(0)
    n = 60
    sp = splats(rng.normal(size=(n, 3)) * 0.1, (0.5, 0.5, 0.5))
    logits = np.zeros((n, 3))
    logits[:20, 0], logits[20:40, 1], logits[40:, 2] = 9, 9, 9
    export_parts.export_parts(sp, logits, tmp_path)
    assert not (tmp_path / "slot01_skin.ply").exists()


def test_quality_report_lists_the_skin_columns(static_skin_scene, tmp_path):
    src, patch, _ = static_skin_scene
    work = copy(src, tmp_path / "w")
    run(work)
    rows = quality_report.analyse(work, None)
    row = next(r for r in rows if r["id"] == "slot01")
    assert row["new"]["ghost"] is not None and row["stats"]["skin"] > 0
    assert "ghost" in quality_report.format_table(rows)
