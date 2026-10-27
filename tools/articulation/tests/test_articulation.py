# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the articulation tools: PLY I/O, state split, synthetic scene and joint estimation."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from tools.articulation import estimate_joint, export_parts, make_synthetic_scene
from tools.articulation.geometry import point_to_line_distance
from tools.articulation.ply_io import Splats, read_ply, write_ply


def random_splats(n: int, num_rest: int, seed: int = 0) -> Splats:
    rng = np.random.default_rng(seed)
    return Splats(
        rng.normal(size=(n, 3)), rng.normal(size=(n, 3)), rng.normal(size=(n, num_rest)), rng.normal(size=(n, 1)),
        rng.normal(size=(n, 3)), rng.normal(size=(n, 4)),
    )


@pytest.mark.parametrize("num_rest", [0, 9, 45])
def test_ply_round_trip(tmp_path, num_rest):
    splats = random_splats(37, num_rest)
    write_ply(tmp_path / "a.ply", splats)
    back = read_ply(tmp_path / "a.ply")
    for name in ("positions", "f_dc", "f_rest", "opacity", "scale", "rot"):
        np.testing.assert_array_equal(getattr(back, name), getattr(splats, name))


def test_ply_header_property_order(tmp_path):
    write_ply(tmp_path / "a.ply", random_splats(3, 6))
    header = (tmp_path / "a.ply").read_bytes().split(b"end_header")[0].decode().splitlines()
    props = [line.split()[-1] for line in header if line.startswith("property")]
    assert props == (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(6)]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    assert "binary_little_endian" in "\n".join(header)


def test_subset_and_concat():
    splats = random_splats(10, 3)
    mask = np.arange(10) % 2 == 0
    even, odd = splats.subset(mask), splats.subset(~mask)
    assert len(even) == 5 and len(odd) == 5
    joined = Splats.concat([even, odd])
    assert len(joined) == 10
    np.testing.assert_array_equal(joined.positions[:5], splats.positions[mask])
    assert len(Splats.concat([Splats.empty(3), odd])) == 5
    with pytest.raises(ValueError):
        Splats.concat([random_splats(2, 3), random_splats(2, 6)])


def test_split_by_state():
    # K = 2 slots -> 5 states: static, slot1 base/switched, slot2 base/switched.
    big = 10.0
    logits = np.zeros((8, 5))
    logits[0, 0] = big  # static
    logits[1, 1] = big  # slot 1 base
    logits[2, 2] = big  # slot 1 switched
    logits[3, 3] = big  # slot 2 base
    logits[4, 4] = big  # slot 2 switched
    logits[5] = [0.2, 0.3, 0.1, 0.0, 0.0]  # flat: argmax 1 but far below 0.6 -> dropped
    logits[6] = [0.5, 0.0, 0.0, 0.0, 0.0]  # flat but argmax 0 -> still static
    logits[7] = [0.0, 3.0, 0.0, 0.0, 0.0]  # argmax 1 with p ~ 0.84 -> slot 1 base
    split = export_parts.split_by_state(logits, min_confidence=0.6)
    assert list(np.flatnonzero(split.static)) == [0, 6]
    assert list(np.flatnonzero(split.slots[1][0])) == [1, 7]
    assert list(np.flatnonzero(split.slots[1][1])) == [2]
    assert list(np.flatnonzero(split.slots[2][0])) == [3]
    assert list(np.flatnonzero(split.slots[2][1])) == [4]
    kept = split.static | split.slots[1][0] | split.slots[1][1] | split.slots[2][0] | split.slots[2][1]
    assert not kept[5]
    with pytest.raises(ValueError):
        export_parts.split_by_state(np.zeros((4, 4)))


def test_outlier_mask():
    rng = np.random.default_rng(0)
    points = np.concatenate([rng.normal(scale=0.05, size=(500, 3)), [[5.0, 5.0, 5.0], [-4.0, 3.0, 2.0]]])
    keep = export_parts.outlier_mask(points)
    assert not keep[-1] and not keep[-2]
    assert keep[:500].mean() > 0.9


class FakeTensor:
    """The bits of a torch tensor that splats_from_checkpoint uses."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return FakeTensor(self.array.astype(np.float32))

    def numpy(self):
        return self.array


def test_checkpoint_layout_matches_ply_exporter():
    n, degree = 4, 2
    num_speculars = (degree + 1) ** 2 - 1
    rng = np.random.default_rng(1)
    specular = rng.normal(size=(n, num_speculars * 3)).astype(np.float32)
    checkpoint = {
        "positions": FakeTensor(rng.normal(size=(n, 3))),
        "rotation": FakeTensor(rng.normal(size=(n, 4))),
        "scale": FakeTensor(rng.normal(size=(n, 3))),
        "density": FakeTensor(rng.normal(size=(n, 1))),
        "features_albedo": FakeTensor(rng.normal(size=(n, 3))),
        "features_specular": FakeTensor(specular),
        "max_n_features": degree,
    }
    splats = export_parts.splats_from_checkpoint(checkpoint)
    # ply_exporter.py: reshape (N, num_speculars, 3) -> transpose(0, 2, 1) -> (N, 3 * num_speculars)
    expected = specular.reshape(n, num_speculars, 3).transpose(0, 2, 1).reshape(n, num_speculars * 3)
    np.testing.assert_array_equal(splats.f_rest, expected)
    np.testing.assert_array_equal(splats.opacity, checkpoint["density"].array.astype(np.float32))
    np.testing.assert_array_equal(splats.rot, checkpoint["rotation"].array.astype(np.float32))
    checkpoint["max_n_features"] = 3
    with pytest.raises(SystemExit):
        export_parts.splats_from_checkpoint(checkpoint)


def test_export_parts_directory(tmp_path):
    rng = np.random.default_rng(0)
    splats = random_splats(300, 0)
    splats.positions = rng.normal(scale=0.1, size=(300, 3)).astype(np.float32)
    logits = np.zeros((300, 3))
    logits[:100, 0] = 8.0
    logits[100:200, 1] = 8.0
    logits[200:, 2] = 8.0
    scene = export_parts.export_parts(splats, logits, tmp_path, names=["cabinet door"], name="t")
    assert scene["up"] == [0.0, -1.0, 0.0] and scene["parts"][0]["name"] == "cabinet door"
    assert len(read_ply(tmp_path / "static.ply")) == 100
    assert 0 < len(read_ply(tmp_path / "slot01_base.ply")) <= 100
    assert json.loads((tmp_path / "scene.json").read_text())["parts"][0]["joint"] is None


@pytest.fixture(scope="module")
def estimated(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    make_synthetic_scene.build_scene(out, seed=3)
    scene_path = out / "scene.json"
    scene = json.loads(scene_path.read_text())
    timings = {}
    for part in scene["parts"]:
        start = time.time()
        estimate_joint.process_part(out, part, None)
        timings[part["id"]] = time.time() - start
    scene_path.write_text(json.dumps(scene))
    truth = json.loads((out / "ground_truth.json").read_text())
    return out, {p["id"]: p for p in scene["parts"]}, truth, timings


def angle_between_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    cos = abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def test_hinge_recovered(estimated):
    _, parts, truth, _ = estimated
    joint, true = parts["slot01"]["joint"], truth["slot01"]
    assert joint["kind"] == "hinge"
    assert angle_between_deg(joint["axis"], true["axis"]) < 1.0
    # the open direction is positive: the axis must point the same way as the true one, not just be parallel
    assert np.dot(joint["axis"], true["axis"]) > 0
    assert point_to_line_distance(joint["pivot"], true["pivot"], true["axis"]) < 0.01
    assert abs(joint["open"] - true["open"]) < 2.0
    assert joint["min"] == 0 and joint["max"] == joint["open"]
    assert joint["mass"] == 2 and joint["frictionTorque"] == 0
    assert joint["fit"]["inlierRatio"] > 0.9


def test_slider_recovered(estimated):
    _, parts, truth, _ = estimated
    joint, true = parts["slot02"]["joint"], truth["slot02"]
    assert joint["kind"] == "slider"
    assert angle_between_deg(joint["axis"], true["axis"]) < 2.0
    assert np.dot(joint["axis"], true["axis"]) > 0
    assert abs(joint["open"] - true["open"]) < 0.01


def test_reveal_and_part_files(estimated):
    out, parts, _, _ = estimated
    files = parts["slot01"]["files"]
    assert files["reveal"] == "slot01_reveal.ply"
    reveal = read_ply(out / files["reveal"])
    # the four interior objects are the only switched-only content
    assert len(reveal) > 2000
    assert (reveal.positions[:, 2] < 0.4).mean() > 0.99  # inside the cabinet (a few door-edge strays allowed)
    part = read_ply(out / files["part"])
    base = read_ply(out / files["base"])
    assert len(part) >= 0.95 * len(base)
    assert parts["slot02"]["files"]["reveal"] is None
    assert not (out / "slot02_reveal.ply").exists()
    for part_info in parts.values():
        assert part_info["files"]["part"] is not None


def test_scene_json_preserved(estimated):
    out, parts, _, _ = estimated
    assert parts["slot01"]["name"] == "cabinet door"
    scene = json.loads((out / "scene.json").read_text())
    assert scene["up"] == [0, 1, 0] and scene["static"] == "static.ply"


def test_estimator_speed(estimated):
    assert max(estimated[3].values()) < 20.0


def test_split_inliers_vanish_set():
    """A base region with no counterpart in the switched set becomes vanish."""
    rng = np.random.default_rng(0)
    plane = np.column_stack([rng.random(4000), rng.random(4000), np.zeros(4000)])
    base = Splats(plane, np.zeros((4000, 3)), np.zeros((4000, 0)), np.zeros((4000, 1)), np.zeros((4000, 3)),
                  np.tile([1.0, 0, 0, 0], (4000, 1)))
    switched = base.subset(plane[:, 0] < 0.6)  # the switched set only has the left 60 %
    switched.positions = switched.positions + np.array([0.0, 0.0, 0.5], dtype=np.float32)
    split = estimate_joint.split_inliers(base, switched, np.eye(3), np.array([0.0, 0.0, 0.5]), 0.03)
    part, reveal = estimate_joint.clean_split(split)
    assert abs(part.mean() - 0.6) < 0.05  # includes the inlier-distance band at the cut
    assert not reveal.any()
