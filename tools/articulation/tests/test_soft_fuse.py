# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Soft state membership of the exporter, the open-pose fusion of the estimator and the quality gate."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from tools.articulation import estimate_joint, export_parts, fuse, make_synthetic_scene, quality_report
from tools.articulation.geometry import hinge_transform
from tools.articulation.ply_io import Splats, read_ply, write_ply


def alpha_of(splats: Splats) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-splats.opacity[:, 0].astype(np.float64)))


def random_splats(n: int, seed: int, rest: int = 0) -> Splats:
    rng = np.random.default_rng(seed)
    return Splats(rng.normal(size=(n, 3)), rng.normal(size=(n, 3)) * 0.2, rng.normal(size=(n, rest)) * 0.05,
                  rng.normal(size=(n, 1)) + 2.0, rng.normal(size=(n, 3)) * 0.1 - 4.0,
                  rng.normal(size=(n, 4)) + [2.0, 0, 0, 0])


# --------------------------------------------------------------------------------------------
# soft split
# --------------------------------------------------------------------------------------------

def test_soft_split_clear_owner_keeps_full_opacity():
    probs = np.array([[0.9996, 0.0002, 0.0002], [0.0003, 0.9995, 0.0002], [0.0002, 0.0003, 0.9995]])
    soft = export_parts.soft_split(probs, 0.1)
    assert soft.static.tolist() == [1.0, 0.0, 0.0]
    assert soft.base[1].tolist() == [0.0, 1.0, 0.0]
    assert soft.switched[1].tolist() == [0.0, 0.0, 1.0]


def test_soft_split_emits_two_copies_with_their_shares():
    soft = export_parts.soft_split(np.array([[0.4, 0.6, 0.0]]), 0.1)
    assert soft.static[0] == pytest.approx(0.4) and soft.base[1][0] == pytest.approx(0.6)
    assert soft.switched[1][0] == 0.0
    # the closed render (static + base) shows 0.4 + 0.6 of the gaussian, as the hosted viewer does
    assert soft.static[0] + soft.base[1][0] == pytest.approx(1.0)


def test_soft_split_gives_dropped_closed_mass_to_the_most_probable_closed_copy_only():
    # K = 2: states static, b1, s1, b2, s2. b2 = 0.05 is below the threshold, b1 and static are emitted.
    probs = np.array([[0.30, 0.55, 0.0, 0.05, 0.10]])
    soft = export_parts.soft_split(probs, 0.1)
    assert soft.base[1][0] == pytest.approx(0.55 + 0.05)  # the most probable closed copy carries the dropped 0.05
    assert soft.static[0] == pytest.approx(0.30)
    assert soft.base[2][0] == 0.0
    assert soft.switched[2][0] == pytest.approx(0.10)  # open render of slot 2: static and base 1 are drawn with it
    closed = soft.static[0] + soft.base[1][0] + soft.base[2][0]
    assert closed == pytest.approx(0.30 + 0.55 + 0.05)  # exactly what the closed render shows, nothing counted twice


def test_soft_split_switched_copy_takes_mass_nobody_else_carries():
    # almost all mass is in switched 1; a sliver of static and base 2 is below the threshold
    probs = np.array([[0.04, 0.0, 0.9, 0.06, 0.0]])
    soft = export_parts.soft_split(probs, 0.1)
    assert soft.static[0] == 0.0 and soft.base[1][0] == 0.0 and soft.base[2][0] == 0.0
    assert soft.switched[1][0] == pytest.approx(1.0)  # 0.9 + 0.04 + 0.06 are all drawn in the open render of slot 1


def test_soft_split_argmax_is_always_emitted_and_closed_mass_is_conserved():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(9) * 0.4, size=500)
    soft = export_parts.soft_split(probs, 0.3)
    assert ((soft.static > 0) | np.any([soft.base[k] > 0 for k in soft.base], axis=0)
            | np.any([soft.switched[k] > 0 for k in soft.switched], axis=0)).all()
    closed_total = soft.static + sum(soft.base[k] for k in soft.base)
    target = probs[:, 0] + probs[:, [1, 3, 5, 7]].sum(axis=1)
    has_closed_copy = closed_total > 0
    assert np.allclose(closed_total[has_closed_copy], np.minimum(target[has_closed_copy], 1.0), atol=0.04)


def test_soft_split_rejects_bad_shapes():
    with pytest.raises(ValueError):
        export_parts.soft_split(np.zeros((3, 4)))


def test_scale_opacity_scales_alpha_in_logit_units():
    splats = random_splats(5, 0)
    alpha = alpha_of(splats)
    out = export_parts.scale_opacity(splats, np.array([1.0, 0.5, 0.25, 1.0, 0.1]))
    assert np.allclose(alpha_of(out), alpha * [1.0, 0.5, 0.25, 1.0, 0.1], rtol=1e-3)
    assert np.array_equal(out.opacity[[0, 3]], splats.opacity[[0, 3]])  # multiplier 1 leaves the logit alone


def test_export_parts_soft_keeps_confident_sets_for_the_fit_and_adds_candidates(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    splats = random_splats(n, 2)
    splats.positions = rng.normal(scale=0.1, size=(n, 3)).astype(np.float32)
    probs = np.zeros((n, 3))
    probs[:200, 0] = 1.0
    probs[200:300, 1] = 1.0  # confident base
    probs[300:340, 2] = 1.0  # confident switched
    probs[340:370] = [0.0, 0.7, 0.3]  # base candidates, ambiguous
    probs[370:385] = [0.45, 0.0, 0.55]  # nearly half static, half switched
    probs[385:] = [1.0, 0.0, 0.0]
    probs += 1e-9
    logits = np.log(probs / probs.sum(1, keepdims=True))
    hard_dir, soft_dir = tmp_path / "hard", tmp_path / "soft"
    export_parts.export_parts(splats, logits, hard_dir, min_confidence=0.6)
    export_parts.export_parts(splats, logits, soft_dir, min_confidence=0.6, soft_threshold=0.1)

    base_hard, base_soft = read_ply(hard_dir / "slot01_base.ply"), read_ply(soft_dir / "slot01_base.ply")
    member = np.load(soft_dir / "slot01_membership.npz")
    assert len(base_soft) > len(base_hard)  # the ambiguous gaussians are candidates now
    assert int(member["base_fit"].sum()) == len(base_hard)  # ... but the joint is fitted on the confident ones
    assert len(member["base_fit"]) == len(member["base_mult"]) == len(base_soft)
    assert member["base_mult"].min() == pytest.approx(0.7, abs=1e-3)  # soft members carry their share of the opacity
    assert (member["base_mult"] < 1.0).sum() >= 25
    assert len(read_ply(soft_dir / "static.ply")) > len(read_ply(hard_dir / "static.ply"))  # half-static ones
    assert not (hard_dir / "slot01_membership.npz").exists()


# --------------------------------------------------------------------------------------------
# rotating gaussians
# --------------------------------------------------------------------------------------------

def random_rotation(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_sh_rotation_is_equivariant(degree):
    """f'(d) = f(R^T d): rotating the coefficients rotates the function the way the object turns."""
    rot = random_rotation(3)
    m = (degree + 1) ** 2 - 1
    rng = np.random.default_rng(4)
    coeffs = rng.normal(size=(1, 3 * m))
    rotated = fuse.rotate_f_rest(coeffs, rot)
    dirs = rng.normal(size=(20, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def evaluate(c, d):  # band 1..degree part of the SH function of channel 0
        return fuse.sh_basis(degree, d)[:, 1:] @ c.reshape(3, m)[0]

    assert np.allclose(evaluate(rotated, dirs), evaluate(coeffs, dirs @ rot), atol=1e-9)


def test_sh_rotation_matches_the_training_code_when_torch_is_available():
    torch = pytest.importorskip("torch")
    from threedgrut.export.sh_rotation import rotate_specular

    rot = random_rotation(5)
    rng = np.random.default_rng(6)
    f_rest = rng.normal(size=(7, 45))  # channel-major
    ours = fuse.rotate_f_rest(f_rest, rot)
    spec = torch.from_numpy(f_rest.reshape(7, 3, 15).transpose(0, 2, 1).reshape(7, 45).copy())  # feature-major
    theirs = rotate_specular(spec, torch.from_numpy(rot), 3).numpy().reshape(7, 15, 3).transpose(0, 2, 1).reshape(7, 45)
    assert np.allclose(ours, theirs, atol=1e-9)


def test_transform_splats_round_trip_and_orientation():
    splats = random_splats(30, 7, rest=45)
    rot, trans = hinge_transform(np.array([0.0, 1.0, 0.0]), np.array([0.3, 0.0, 0.1]), np.deg2rad(70.0))
    moved = fuse.transform_splats(splats, rot, trans)
    assert np.allclose(moved.positions, splats.positions @ rot.T + trans, atol=1e-5)
    inv_rot, inv_trans = fuse.invert_rigid(rot, trans)
    back = fuse.transform_splats(moved, inv_rot, inv_trans)
    assert np.allclose(back.positions, splats.positions, atol=1e-5)
    assert np.allclose(back.f_rest, splats.f_rest, atol=1e-5)
    mats = fuse.quaternions_to_matrices(splats.rot)
    assert np.allclose(fuse.quaternions_to_matrices(moved.rot), rot[None] @ mats, atol=1e-5)
    assert np.allclose(moved.scale, splats.scale) and np.allclose(moved.opacity, splats.opacity)
    assert np.allclose(moved.f_dc, splats.f_dc)  # DC colour does not rotate


# --------------------------------------------------------------------------------------------
# fusion on synthetic data
# --------------------------------------------------------------------------------------------

SPACING = 0.02


def sheet(rng, z: float, n: int, rgb, w: float = 1.0, h: float = 0.8) -> Splats:
    pts = np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h, n), np.full(n, z)])
    return Splats(pts, np.tile((np.asarray(rgb) - 0.5) / 0.2820948, (n, 1)), np.zeros((n, 0)),
                  np.full((n, 1), 2.0), np.full((n, 3), np.log(0.02)), np.tile([1.0, 0, 0, 0], (n, 1)))


def door_scene(seed: int = 0):
    """A door sampled in the closed pose (outer skin) and, independently, in the open pose (skin + inner face +
    an unrelated interior). Returns closed part, switched set, the transform and the row groups of ``switched``."""
    rng = np.random.default_rng(seed)
    n = int(1.0 * 0.8 / SPACING**2)
    skin = sheet(rng, 0.0, n, (0.2, 0.3, 0.8))
    rot, trans = hinge_transform(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.deg2rad(70.0))
    skin_open = fuse.transform_splats(sheet(rng, 0.0, n, (0.2, 0.3, 0.8)), rot, trans)
    inner_open = fuse.transform_splats(sheet(rng, -0.03, n // 2, (0.1, 0.1, 0.1)), rot, trans)  # inner face, darker
    interior = random_splats(300, 3)
    interior.positions = (rng.uniform([-2.0, -1.0, 1.5], [-1.0, 2.0, 2.5], size=(300, 3))).astype(np.float32)
    switched = Splats.concat([skin_open, inner_open, interior])
    groups = {"skin": np.arange(n), "inner": np.arange(n, n + n // 2), "interior": np.arange(n + n // 2, len(switched))}
    return skin, switched, rot, trans, groups


def test_fusion_adds_the_inner_face_and_leaves_the_interior_alone():
    skin, switched, rot, trans, groups = door_scene()
    spacing = estimate_joint.median_spacing(skin.positions.astype(np.float64))
    result = fuse.fuse_open_pose(skin, switched, rot, trans, spacing)
    assert not result.envelope[groups["interior"]].any()  # the interior stays reveal
    inner_share = result.envelope[groups["inner"]].mean()
    assert inner_share > 0.95  # the inner face lies within the door's thickness
    fused = Splats.concat([result.near, result.inner])
    # fused gaussians are in the CLOSED pose: inner-face ones sit at z = -0.08 again, with their colour
    inner_rows = fused.subset(np.abs(fused.positions[:, 2] + 0.03) < 0.005)
    assert len(inner_rows) > 0.4 * len(groups["inner"])
    assert np.allclose(inner_rows.rgb.mean(axis=0), 0.1, atol=0.05)
    # their orientation was turned back too: identity quaternion up to sign
    assert np.abs(inner_rows.rot[:, 0]).min() > 0.999


def test_fusion_does_not_double_the_density_where_the_samples_overlap():
    skin, switched, rot, trans, groups = door_scene(seed=1)
    spacing = estimate_joint.median_spacing(skin.positions.astype(np.float64))
    result = fuse.fuse_open_pose(skin, switched, rot, trans, spacing, dup=1.5)
    on_skin = Splats.concat([result.near, result.inner]).subset(
        np.abs(Splats.concat([result.near, result.inner]).positions[:, 2]) < 0.01)
    density_before = len(skin)
    density_after = len(skin) + len(on_skin)
    assert density_after < 1.5 * density_before  # an independent second sample would give 2x; duplicates are dropped
    assert len(on_skin) > 0.05 * density_before  # ... but the second sample is not discarded wholesale
    # no fused gaussian repeats a closed one: none is within 0.4 spacings of a part gaussian with a similar look
    from scipy.spatial import cKDTree

    d, _ = cKDTree(skin.positions.astype(np.float64)).query(on_skin.positions.astype(np.float64))
    assert (d < 0.3 * spacing).mean() < 0.02
    assert result.duplicates > 0.2 * len(groups["skin"])


def test_duplicate_mask_needs_similar_colour_and_scale():
    rng = np.random.default_rng(2)
    base = sheet(rng, 0.0, 200, (0.5, 0.5, 0.5))
    same = Splats(base.positions + 1e-4, base.f_dc, base.f_rest, base.opacity, base.scale, base.rot)
    assert fuse.duplicate_mask(same, base, 0.01).all()
    other_colour = Splats(same.positions, same.f_dc + 3.0, same.f_rest, same.opacity, same.scale, same.rot)
    assert not fuse.duplicate_mask(other_colour, base, 0.01).any()  # a different look at the same place is kept
    bigger = Splats(same.positions, same.f_dc, same.f_rest, same.opacity, same.scale + 2.0, same.rot)
    assert not fuse.duplicate_mask(bigger, base, 0.01).any()


def test_thin_self_duplicates_collapses_clusters_only():
    rng = np.random.default_rng(3)
    a = sheet(rng, 0.0, 100, (0.5, 0.5, 0.5))
    pair = Splats.concat([a, Splats(a.positions + 1e-4, a.f_dc, a.f_rest, a.opacity, a.scale, a.rot)])
    keep = fuse.thin_self_duplicates(pair, 0.01)
    assert 95 <= keep.sum() <= 130


# --------------------------------------------------------------------------------------------
# the estimator end to end
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def door_with_inner_face(tmp_path_factory):
    """The synthetic cabinet, plus a darker inner face of the door that only the open pose sampled."""
    clean = tmp_path_factory.mktemp("fuse_clean")
    make_synthetic_scene.build_scene(clean, seed=3)
    truth = json.loads((clean / "ground_truth.json").read_text())["slot01"]
    base = read_ply(clean / "slot01_base.ply")
    switched = read_ply(clean / "slot01_switched.ply")
    rng = np.random.default_rng(11)
    lo, hi = base.positions.min(axis=0), base.positions.max(axis=0)
    extent = hi - lo
    thin = int(np.argmin(extent))  # the door is a thin slab: its normal axis
    inner_closed = base.subset(rng.random(len(base)) < 0.6)
    offset = np.zeros(3)
    offset[thin] = 0.02
    inner_closed = Splats(inner_closed.positions + offset, np.full_like(inner_closed.f_dc, -1.7), inner_closed.f_rest,
                          inner_closed.opacity, inner_closed.scale, inner_closed.rot)
    rot, trans = hinge_transform(np.array(truth["axis"]), np.array(truth["pivot"]), np.deg2rad(truth["open"]))
    inner_open = fuse.transform_splats(inner_closed, rot, trans)
    dirty = tmp_path_factory.mktemp("fuse_dirty")
    shutil.copytree(clean, dirty, dirs_exist_ok=True)
    write_ply(dirty / "slot01_switched.ply", Splats.concat([switched, inner_open]))
    return clean, dirty, inner_closed, len(switched), len(inner_open)


def run_slot01(scene_dir, **kwargs):
    scene = json.loads((scene_dir / "scene.json").read_text())
    part = scene["parts"][0]
    estimate_joint.process_part(scene_dir, part, None, up=scene["up"], **kwargs)
    return part


def test_estimator_fuses_the_inner_face_into_an_opaque_part(door_with_inner_face, tmp_path):
    _, dirty, inner_closed, n_switched, n_inner = door_with_inner_face
    work = tmp_path / "w"
    shutil.copytree(dirty, work)
    plain = run_slot01(work, fuse=False)
    n_plain = len(read_ply(work / plain["files"]["part"]))
    reveal_plain = len(read_ply(work / plain["files"]["reveal"]))
    fused = run_slot01(work)
    stats = fused["stats"]
    part = read_ply(work / fused["files"]["part"])
    assert fused["joint"]["open"] == pytest.approx(plain["joint"]["open"], abs=1e-6)  # the fit does not change
    assert stats["innerMode"] == "part"  # the synthetic door is opaque
    assert len(part) > n_plain + 0.4 * n_inner  # the inner face moved into the part
    assert len(read_ply(work / fused["files"]["reveal"])) < reveal_plain - 0.2 * n_inner  # ... and left the reveal
    dark = part.subset(part.rgb.mean(axis=1) < 0.1)
    assert len(dark) > 0.4 * len(inner_closed)  # in the closed pose, with its own colour
    assert stats["registered"] <= stats["base"] and stats["part"] <= stats["registered"]


def test_inner_modes(door_with_inner_face, tmp_path):
    _, dirty, _, _, n_inner = door_with_inner_face
    work = tmp_path / "m"
    shutil.copytree(dirty, work)
    revealed = run_slot01(work, inner="reveal")
    assert "partReveal" not in revealed["files"]
    n_reveal = len(read_ply(work / revealed["files"]["reveal"]))
    attached = run_slot01(work, inner="attach")
    assert attached["files"]["partReveal"] == "slot01_partReveal.ply"
    carried = read_ply(work / "slot01_partReveal.ply")
    assert len(carried) > 0.3 * n_inner
    assert len(read_ply(work / attached["files"]["reveal"])) < n_reveal  # what the part carries is not revealed twice
    back = run_slot01(work, inner="reveal")  # switching modes leaves no stale file behind
    assert "partReveal" not in back["files"] and not (work / "slot01_partReveal.ply").exists()
    with pytest.raises(ValueError):
        run_slot01(work, inner="nonsense")


def test_translucent_part_keeps_its_inner_face_out_of_the_part(door_with_inner_face, tmp_path):
    _, dirty, _, _, _ = door_with_inner_face
    work = tmp_path / "t"
    shutil.copytree(dirty, work)
    for name in ("slot01_base.ply", "slot01_switched.ply"):  # make the door glass-like: opacity 0.3
        splats = read_ply(work / name)
        write_ply(work / name, Splats(splats.positions, splats.f_dc, splats.f_rest, np.full_like(splats.opacity, -0.85),
                                      splats.scale, splats.rot))
    part = run_slot01(work)
    assert part["stats"]["innerMode"] == "reveal"


def test_soft_export_then_estimate_is_idempotent_and_uses_the_sidecar(tmp_path):
    scene_dir = tmp_path / "scene"
    make_synthetic_scene.build_scene(scene_dir, seed=3)
    base = read_ply(scene_dir / "slot01_base.ply")
    # a soft sidecar: the last 40 base rows are soft candidates that must not enter the fit
    n = len(base)
    fit = np.ones(n, dtype=bool)
    fit[-40:] = False
    switched = read_ply(scene_dir / "slot01_switched.ply")
    np.savez(export_parts.membership_path(scene_dir, "slot01"), base_fit=fit, switched_fit=np.ones(len(switched), bool),
             base_mult=np.where(fit, 1.0, 0.5).astype(np.float32), switched_mult=np.ones(len(switched), np.float32))
    first = run_slot01(scene_dir)
    sizes = {k: len(read_ply(scene_dir / first["files"][k])) if first["files"][k] else 0 for k in ("part", "vanish", "reveal")}
    second = run_slot01(scene_dir)
    assert {k: len(read_ply(scene_dir / second["files"][k])) if second["files"][k] else 0 for k in sizes} == sizes
    assert first["joint"] == second["joint"] and first["stats"] == second["stats"]
    assert first["stats"]["baseConfident"] == n - 40
    # a stale sidecar (row count changed) is ignored instead of misaligning the flags
    np.savez(export_parts.membership_path(scene_dir, "slot01"), base_fit=fit[:-5], switched_fit=np.ones(len(switched), bool),
             base_mult=np.ones(n - 5, np.float32), switched_mult=np.ones(len(switched), np.float32))
    assert run_slot01(scene_dir)["stats"]["baseConfident"] == n


def test_trim_keeps_sparse_second_lobes_of_a_registered_panel():
    """A panel with a second, sparser region (the hood's flanks) is no floater: it stays in the body."""
    rng = np.random.default_rng(5)
    centre = np.column_stack([rng.uniform(0, 1.0, 900), rng.uniform(-0.1, 0.1, 900), rng.normal(scale=0.004, size=900)])
    flank = np.column_stack([rng.uniform(0, 1.0, 150), rng.uniform(0.7, 0.9, 150), rng.normal(scale=0.004, size=150)])
    far = np.array([[3.0, 3.0, 3.0], [-2.0, 0.5, 1.0], [0.5, -2.5, 0.2]])
    pts = np.concatenate([centre, flank, far])
    splats = Splats(pts, np.zeros((len(pts), 3)), np.zeros((len(pts), 0)), np.full((len(pts), 1), 2.0),
                    np.full((len(pts), 3), np.log(0.02)), np.tile([1.0, 0, 0, 0], (len(pts), 1)))
    keep = estimate_joint.trim_body(splats, np.ones(len(pts), dtype=bool))
    assert keep[900:1050].mean() > 0.9  # the flank survives
    assert not keep[1050:].any()  # the isolated floaters do not


# --------------------------------------------------------------------------------------------
# the quality gate
# --------------------------------------------------------------------------------------------

def test_quality_report_prefers_the_fused_part(door_with_inner_face, tmp_path):
    _, dirty, _, _, _ = door_with_inner_face
    old, new = tmp_path / "old", tmp_path / "new"
    shutil.copytree(dirty, old)
    shutil.copytree(dirty, new)
    for target, fuse_on in ((old, False), (new, True)):
        scene = json.loads((target / "scene.json").read_text())
        part = scene["parts"][0]
        estimate_joint.process_part(target, part, None, up=scene["up"], fuse=fuse_on)
        (target / "scene.json").write_text(json.dumps(scene))  # the report reads the part entries
    rows = quality_report.analyse(new, old, cell_metres=0.03)
    row = next(r for r in rows if r["id"] == "slot01")
    assert row["new"]["counts"]["part"] > row["old"]["counts"]["part"]
    assert row["new"]["occupancy"] >= row["old"]["occupancy"]
    assert quality_report.verdict(row)
    assert "slot01" in quality_report.format_table(rows) and "PASS" in quality_report.format_table(rows)


def test_hole_stats_and_occupancy_on_a_known_grid():
    grid = np.ones((10, 10))
    grid[2:5, 2:5] = 0.0
    grid[8, 8] = 0.0
    share, largest = quality_report.hole_stats(grid)
    assert share == pytest.approx(0.10) and largest == 9
    pts = np.column_stack([np.linspace(0, 0.99, 100), np.zeros(100), np.zeros(100)])
    frame = quality_report.Frame(np.zeros(3), np.eye(3), np.array([0.0, -0.05]), np.array([1.0, 0.05]), (-1, 1))
    assert quality_report.occupancy(frame, pts, 0.1) == pytest.approx(1.0)
    assert quality_report.occupancy(frame, pts[:50], 0.1) == pytest.approx(0.5)
