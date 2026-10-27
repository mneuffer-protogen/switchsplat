# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the .gutb importer: layout round trips, manifest parsing, floor estimation and the full import."""

from __future__ import annotations

import gzip
import json
import struct

import numpy as np
import pytest

from tools.articulation import export_parts, import_gutb
from tools.articulation.ply_io import read_ply

SH_C0 = 0.28209479177387814


# --------------------------------------------------------------------------------------------
# A tiny .gutb encoder, written from the layout the hosted viewer's WGSL / loader defines
# --------------------------------------------------------------------------------------------

def encode_gutb(path, pos, alpha, quat, scale, dc, rest, probs, *, encoding="dc-f16/rest-i8", gzipped=True, gap=0):
    """rest is coefficient-major [N, 15, 3]; the sections are optionally separated by ``gap`` padding bytes."""
    n = len(pos)
    geometry = np.concatenate([pos, alpha[:, None], quat, scale, np.zeros((n, 1))], axis=1).astype("<f4")  # + 1 pad float
    assert geometry.shape == (n, 12)
    if encoding == "dc-f16/rest-i8":
        peak = np.abs(rest.reshape(n, 45)).max(axis=1)
        rest_scale = np.maximum(peak / 127.0, 1e-8).astype("<f2")
        q = np.clip(np.round(rest.reshape(n, 45) / rest_scale.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
        sh = np.zeros((n, 56), dtype=np.uint8)
        sh[:, :6] = dc.astype("<f2").view(np.uint8).reshape(n, 6)
        sh[:, 6:8] = rest_scale.view(np.uint8).reshape(n, 2)
        sh[:, 8:53] = q.view(np.uint8)
    elif encoding == "f16":
        sh = np.concatenate([dc, rest.reshape(n, 45)], axis=1).astype("<f2").view(np.uint8)
    else:
        sh = np.concatenate([dc, rest.reshape(n, 45)], axis=1).astype("<f4").view(np.uint8)
    switches = probs.astype("<f2")
    blobs = [("geometry", geometry.tobytes()), ("sh", sh.tobytes()), ("switches", switches.tobytes())]
    sections, body, offset = [], b"", 0
    for name, blob in blobs:
        body += b"\x00" * gap
        offset += gap
        sections.append({"name": name, "offset": offset, "bytes": len(blob)})
        body += blob
        offset += len(blob)
    header = json.dumps({
        "version": 1, "count": n, "shDegree": 3, "shEncoding": encoding, "switchCount": probs.shape[1],
        "kernelDegree": 2, "renderer": "gsplat_3dgut", "sections": sections,
    }).encode()
    data = b"GUTB0001" + struct.pack("<I", len(header)) + header + body
    path.write_bytes(gzip.compress(data) if gzipped else data)


def random_scene(n=40, k=9, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)).astype(np.float32)
    alpha = rng.uniform(0.02, 0.98, size=n).astype(np.float32)
    quat = rng.normal(size=(n, 4)).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    scale = rng.uniform(0.01, 0.5, size=(n, 3)).astype(np.float32)
    dc = rng.normal(scale=0.6, size=(n, 3)).astype(np.float32)
    rest = rng.normal(scale=0.1, size=(n, 15, 3)).astype(np.float32)
    probs = rng.dirichlet(np.ones(k) * 0.2, size=n).astype(np.float32)
    return pos, alpha, quat, scale, dc, rest, probs


@pytest.mark.parametrize("encoding,gzipped,gap", [("dc-f16/rest-i8", True, 0), ("dc-f16/rest-i8", False, 7),
                                                  ("f16", True, 0), ("f32", False, 3)])
def test_gutb_round_trip(tmp_path, encoding, gzipped, gap):
    pos, alpha, quat, scale, dc, rest, probs = random_scene()
    path = tmp_path / "scene.gutb"
    encode_gutb(path, pos, alpha, quat, scale, dc, rest, probs, encoding=encoding, gzipped=gzipped, gap=gap)
    gutb = import_gutb.read_gutb(path)
    assert gutb.count == len(pos) and gutb.header["switchCount"] == 9

    p, a, q, s = import_gutb.decode_geometry(gutb.geometry)
    np.testing.assert_array_equal(p, pos)
    np.testing.assert_array_equal(a, alpha)
    np.testing.assert_array_equal(q, quat)
    np.testing.assert_array_equal(s, scale)

    f_dc, f_rest = import_gutb.decode_sh(gutb.sh, encoding, 3)
    tol = {"dc-f16/rest-i8": 2e-3, "f16": 1e-3, "f32": 1e-7}[encoding]
    np.testing.assert_allclose(f_dc, dc, atol=2e-3 if encoding != "f32" else 1e-7)
    # coefficient-major [N, 15, 3] on disk -> channel-major [N, 45] (all R coefficients, then G, then B)
    expected = rest.transpose(0, 2, 1).reshape(len(pos), 45)
    np.testing.assert_allclose(f_rest, expected, atol=tol)
    np.testing.assert_allclose(f_rest[:, 15 * 2 + 4], rest[:, 4, 2], atol=tol)  # blue channel, coefficient 5

    np.testing.assert_allclose(gutb.switches, probs.astype(np.float16).astype(np.float32))


def test_sh_degree_truncation(tmp_path):
    pos, alpha, quat, scale, dc, rest, probs = random_scene(10)
    path = tmp_path / "scene.gutb.gz"
    encode_gutb(path, pos, alpha, quat, scale, dc, rest, probs)
    gutb = import_gutb.read_gutb(path)
    _, f_rest1 = import_gutb.decode_sh(gutb.sh, "dc-f16/rest-i8", 1)
    _, f_rest3 = import_gutb.decode_sh(gutb.sh, "dc-f16/rest-i8", 3)
    assert f_rest1.shape == (10, 9) and f_rest3.shape == (10, 45)
    # degree 1 keeps the first 3 coefficients of every channel
    np.testing.assert_array_equal(f_rest1[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]], f_rest3[:, [0, 1, 2, 15, 16, 17, 30, 31, 32]])
    splats = import_gutb.to_splats(gutb, sh_degree=0)
    assert splats.f_rest.shape == (10, 0)


def test_conversion_to_ply_units(tmp_path):
    pos, alpha, quat, scale, dc, rest, probs = random_scene()
    alpha[0], alpha[1] = 1.0, 0.0  # saturated values must stay finite after the logit
    path = tmp_path / "scene.gutb"
    encode_gutb(path, pos, alpha, quat, scale, dc, rest, probs, encoding="f32", gzipped=False)
    splats = import_gutb.to_splats(import_gutb.read_gutb(path))
    assert np.isfinite(splats.opacity).all()
    np.testing.assert_allclose(1 / (1 + np.exp(-splats.opacity[2:, 0])), alpha[2:], atol=1e-5)
    np.testing.assert_allclose(np.exp(splats.scale), scale, rtol=1e-5)
    np.testing.assert_array_equal(splats.rot, quat)  # already (w, x, y, z)
    np.testing.assert_allclose(splats.rgb, dc * SH_C0 + 0.5, atol=1e-6)


def test_bad_files(tmp_path):
    (tmp_path / "x.gutb").write_bytes(b"NOTAGUTB" + b"\0" * 20)
    with pytest.raises(ValueError):
        import_gutb.read_gutb(tmp_path / "x.gutb")
    pos, alpha, quat, scale, dc, rest, probs = random_scene(5)
    good = tmp_path / "g.gutb"
    encode_gutb(good, pos, alpha, quat, scale, dc, rest, probs, gzipped=False)
    (tmp_path / "short.gutb").write_bytes(good.read_bytes()[:-10])
    with pytest.raises(ValueError):
        import_gutb.read_gutb(tmp_path / "short.gutb")


def test_probabilities_split_matches_state_order():
    # K = 2 slots: states 0 static, 1/2 slot 1 base/switched, 3/4 slot 2 base/switched
    probs = np.array([[1, 0, 0, 0, 0], [0, 0.9, 0.1, 0, 0], [0, 0.05, 0.95, 0, 0], [0, 0, 0, 0.7, 0.3],
                      [0, 0, 0, 0.1, 0.9], [0.3, 0.35, 0.35, 0, 0]], dtype=np.float64)
    split = export_parts.split_by_state(import_gutb.probabilities_to_logits(probs), 0.6)
    assert list(np.flatnonzero(split.static)) == [0]
    assert list(np.flatnonzero(split.slots[1][0])) == [1]
    assert list(np.flatnonzero(split.slots[1][1])) == [2]
    assert list(np.flatnonzero(split.slots[2][0])) == [3]
    assert list(np.flatnonzero(split.slots[2][1])) == [4]
    assert split.assignment[5] == 1 and split.confidence[5] < 0.6  # ambiguous: dropped from the slot sets


# --------------------------------------------------------------------------------------------
# Manifest parsing
# --------------------------------------------------------------------------------------------

MANIFEST = (
    "var x={id:`other`,name:`Other`,camera:{up:[0,-1,0]}},"
    "y=[{id:`best-car`,name:`Car`,renderer:`gsplat_3dgut`,flag:!0,off:!1,"
    "camera:{position:[1.5,-.25,5e-1],target:[.04,2.58,-0.51],up:[.119,-.9876,-.1022],fov:123,source:`datasetCamera`},"
    "base:{url:`a/b/scene.gutb.gz`,bytes:46511633},source:{note:`step 33e3 \\` quoted`,v:void 0,fn:Q(t,e)[0].x},"
    "switches:[{id:`b`,label:`Switch B`,semantic:{label:`Door`,confidence:.98}},"
    "{id:`c`,label:`Switch C`,semantic:{label:`Hood, front`,confidence:.99}},{id:`d`,label:`Switch D`}],"
    "globalZOrder:!1},{id:`best-chair`,name:`Chair`}];export{y};"
)


def test_manifest_parser():
    entry = import_gutb.find_manifest_entry(MANIFEST, "best-car")
    assert entry["name"] == "Car" and entry["flag"] is True and entry["off"] is False
    assert entry["camera"]["position"] == [1.5, -0.25, 0.5]
    assert entry["camera"]["target"] == [0.04, 2.58, -0.51]
    assert entry["source"]["note"] == "step 33e3 ` quoted" and entry["source"]["v"] is None and entry["source"]["fn"] is None
    assert import_gutb.names_from_manifest(entry) == ["Door", "Hood, front", "Switch D"]  # falls back to the label
    assert import_gutb.find_manifest_entry(MANIFEST, "best-chair")["name"] == "Chair"
    with pytest.raises(KeyError):
        import_gutb.find_manifest_entry(MANIFEST, "nope")
    assert import_gutb.parse_js_literal("[1,{a:`x`,b:[!0]}]") == [1.0, {"a": "x", "b": [True]}]
    np.testing.assert_allclose(import_gutb.unit([0, -2, 0]), [0, -1, 0])
    np.testing.assert_allclose(import_gutb.unit(None), [0, -1, 0])


# --------------------------------------------------------------------------------------------
# Floor and ground
# --------------------------------------------------------------------------------------------

def test_estimate_floor_finds_dominant_low_plane():
    rng = np.random.default_rng(0)
    up = np.array([0.0, -1.0, 0.0])  # y down, as in COLMAP scenes: height = -y
    ground = np.column_stack([rng.uniform(-6, 6, 8000), 2.0 + rng.normal(scale=0.01, size=8000), rng.uniform(-6, 6, 8000)])
    clutter = rng.uniform([-6, -3, -6], [6, 2.0, 6], size=(3000, 3))  # sparse content above the ground
    parts = np.column_stack([rng.uniform(1.5, 2.5, 600), rng.uniform(0.0, 1.6, 600), rng.uniform(0.0, 0.1, 600)])
    # the ground leaking into a part label is part of the input too
    points = np.concatenate([ground, clutter, parts])
    floor, thickness = import_gutb.estimate_floor(points, parts, up)
    assert abs(float(floor @ up) - (-2.0)) < 0.03  # height of the plane along up (up = -y)
    assert abs(floor[0] - np.median(parts[:, 0])) < 1e-6 and abs(floor[2] - np.median(parts[:, 2])) < 1e-6
    assert 0 < thickness < 0.05
    assert import_gutb.estimate_floor(np.zeros((10, 3)), parts, up) is None


def test_demote_ground_makes_low_slot_gaussians_static():
    up = np.array([0.0, 0.0, 1.0])
    positions = np.array([[0, 0, 0.02], [0, 0, 0.5], [0, 0, 0.03], [0, 0, 0.9]], dtype=np.float64)
    logits = np.log(np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8]]))
    slot_mask = np.array([True, True, False, True])
    out, demoted = import_gutb.demote_ground(logits, positions, slot_mask, np.zeros(3), up, 0.1)
    assert list(demoted) == [True, False, False, False]
    assert list(export_parts.split_by_state(out, 0.6).static) == [True, False, True, False]


# --------------------------------------------------------------------------------------------
# Full import
# --------------------------------------------------------------------------------------------

def test_import_gutb_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    n_static, n_base, n_switched = 1500, 400, 500
    ground = np.column_stack([rng.uniform(-3, 3, n_static), rng.normal(scale=0.005, size=n_static), rng.uniform(-3, 3, n_static)])
    base = rng.uniform([0.5, -1.2, -0.02], [1.0, -0.4, 0.02], size=(n_base, 3))  # y down: above the ground at y = 0
    switched = rng.uniform([0.5, -1.2, 0.5], [1.0, -0.4, 0.54], size=(n_switched, 3))
    pos = np.concatenate([ground, base, switched]).astype(np.float32)
    n = len(pos)
    probs = np.zeros((n, 3), dtype=np.float32)
    probs[:n_static, 0] = 1.0
    probs[n_static:n_static + n_base, 1] = 0.97
    probs[n_static:n_static + n_base, 0] = 0.03
    probs[n_static + n_base:, 2] = 0.9
    probs[n_static + n_base:, 0] = 0.1
    alpha = np.full(n, 0.9, dtype=np.float32)
    quat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    scale = np.full((n, 3), 0.02, dtype=np.float32)
    dc = rng.normal(scale=0.3, size=(n, 3)).astype(np.float32)
    rest = rng.normal(scale=0.05, size=(n, 15, 3)).astype(np.float32)
    gutb = tmp_path / "scene.gutb.gz"
    encode_gutb(gutb, pos, alpha, quat, scale, dc, rest, probs)
    js = tmp_path / "index.js"
    js.write_text(
        "x=[{id:`t`,name:`Toy`,camera:{position:[0,-2,5],target:[0.7,-0.8,0],up:[0,-1,0]},"
        "switches:[{id:`b`,label:`Switch B`,semantic:{label:`Toy door`,confidence:.9}}]}]", encoding="utf-8")

    out = tmp_path / "out"
    scene = import_gutb.import_gutb(gutb, out, manifest_js=js, scene_id="t", verbose=False)
    assert scene["name"] == "Toy" and scene["up"] == [0.0, -1.0, 0.0]
    assert scene["view"] == {"position": [0.0, -2.0, 5.0], "target": [0.7, -0.8, 0.0]}
    assert scene["parts"][0]["name"] == "Toy door" and scene["parts"][0]["joint"] is None
    assert abs(scene["floor"][1]) < 0.05  # ground at y = 0
    assert len(read_ply(out / "static.ply")) == n_static
    assert len(read_ply(out / "slot01_base.ply")) > 0.9 * n_base
    sw = read_ply(out / "slot01_switched.ply")
    assert len(sw) > 0.9 * n_switched and sw.f_rest.shape[1] == 45
    on_disk = json.loads((out / "scene.json").read_text())
    assert on_disk["source"]["format"] == "gutb"
