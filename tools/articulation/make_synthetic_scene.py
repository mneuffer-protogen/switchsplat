# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a small procedural articulated scene (Y-up, metres) in the export layout.

Static: floor, back wall, an open-front cabinet with shelves.
slot01: a hinged cabinet door (hinge on the left front edge, opens 95 deg) with objects behind it
        that exist only in the switched set (the reveal).
slot02: a drawer below the door that slides out 0.35 m.

The base set is the part in its closed pose, the switched set the part in its open pose, each
sampled independently (fresh random surface samples plus a small position jitter), so the two
sets are not the same points transformed. Writes static.ply, slotNN_{base,switched}.ply,
scene.json (joints null) and ground_truth.json (the true joints) into ``--out``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import hinge_transform, quaternions_from_matrices
from .ply_io import Splats, read_ply, rgb_to_f_dc, write_ply

STATIC_SPACING = 0.035
PART_SPACING = 0.015
REVEAL_SPACING = 0.01
JITTER = 0.002  # std of the position jitter added to the switched samples, metres

DOOR_HINGE_AXIS = np.array([0.0, -1.0, 0.0])
DOOR_HINGE_PIVOT = np.array([-0.40, 0.65, 0.42])
DOOR_OPEN_DEG = 95.0
DRAWER_AXIS = np.array([0.0, 0.0, 1.0])
DRAWER_OPEN = 0.35

Points = tuple  # (positions [N,3], normals [N,3], rgb [N,3])


def _stripes(coord: np.ndarray, period: float, duty: float = 0.5) -> np.ndarray:
    return ((coord / period) % 1.0) < duty


def sample_rect(rng, origin, u, v, spacing, colour_fn) -> Points:
    """Uniform random samples of the parallelogram origin + a*u + b*v; colour_fn(a, b, pos) -> rgb."""
    origin, u, v = (np.asarray(x, dtype=np.float64) for x in (origin, u, v))
    cross = np.cross(u, v)
    area = np.linalg.norm(cross)
    count = max(int(round(area / spacing**2)), 1)
    ab = rng.random((count, 2))
    pos = origin + ab[:, :1] * u + ab[:, 1:] * v
    normals = np.tile(cross / area, (count, 1))
    return pos, normals, np.clip(colour_fn(ab[:, 0], ab[:, 1], pos), 0.0, 1.0)


def solid(rgb):
    return lambda a, b, pos: np.tile(np.asarray(rgb, dtype=np.float64), (len(a), 1))


def sample_box(rng, lo, hi, spacing, colours: dict) -> Points:
    """Six faces of an axis-aligned box. ``colours`` maps '+x', '-x', ... (or '*') to a colour function or rgb."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    size = hi - lo
    out = []
    for face in ("+x", "-x", "+y", "-y", "+z", "-z"):
        axis = "xyz".index(face[1])
        a1, a2 = [i for i in range(3) if i != axis]
        origin = lo.copy()
        if face[0] == "+":
            origin[axis] = hi[axis]
        u, v = np.zeros(3), np.zeros(3)
        u[a1], v[a2] = size[a1], size[a2]
        fn = colours.get(face, colours.get("*"))
        out.append(sample_rect(rng, origin, u, v, spacing, fn if callable(fn) else solid(fn)))
    return concat_points(out)


def sample_sphere(rng, centre, radius, spacing, rgb) -> Points:
    count = max(int(round(4.0 * np.pi * radius**2 / spacing**2)), 1)
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return np.asarray(centre) + radius * normals, normals, np.tile(np.asarray(rgb, dtype=np.float64), (count, 1))


def concat_points(parts: list) -> Points:
    return tuple(np.concatenate(x) for x in zip(*parts))


def make_gaussians(rng, pos: np.ndarray, normals: np.ndarray, rgb: np.ndarray, spacing: float) -> Splats:
    """Flat discs lying in the surface: thin along the normal, ~sample-spacing wide, SH degree 0."""
    n = len(pos)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    helper = np.where(np.abs(normals[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    t1 = np.cross(normals, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)
    spin = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    a = np.cos(spin) * t1 + np.sin(spin) * t2
    b = np.cross(normals, a)
    rot = quaternions_from_matrices(np.stack([a, b, normals], axis=2))  # columns = local axes, z = normal
    wide = np.log(spacing * 0.6) + rng.normal(0.0, 0.05, size=(n, 1))
    scale = np.concatenate([wide, wide, np.full((n, 1), np.log(spacing * 0.08))], axis=1)
    return Splats(
        positions=pos,
        f_dc=rgb_to_f_dc(rgb),
        f_rest=np.zeros((n, 0)),
        opacity=3.0 + rng.normal(0.0, 0.2, size=(n, 1)),
        scale=scale,
        rot=rot,
    )


# --------------------------------------------------------------------------------------------
# Scene content. The part generators return samples in the CLOSED pose.
# --------------------------------------------------------------------------------------------

def static_points(rng) -> Points:
    s = STATIC_SPACING
    wood = np.array([0.55, 0.36, 0.20])

    def floor_colour(a, b, pos):
        tile = (np.floor(pos[:, 0] / 0.25) + np.floor(pos[:, 2] / 0.25)) % 2
        return np.where(tile[:, None] == 0, [0.55, 0.55, 0.52], [0.33, 0.33, 0.34])

    def wall_colour(a, b, pos):
        return np.tile([0.85, 0.82, 0.72], (len(a), 1)) * (0.9 + 0.1 * b[:, None])

    def wood_colour(a, b, pos):
        return wood * (0.85 + 0.15 * np.sin(pos[:, 1] * 40.0)[:, None])

    def shelf(a, b, pos):
        return np.tile([0.75, 0.6, 0.4], (len(a), 1)) * (0.9 + 0.1 * a[:, None])

    parts = [
        sample_rect(rng, [-1.6, 0, -0.6], [3.2, 0, 0], [0, 0, 2.6], s, floor_colour),
        sample_rect(rng, [-1.6, 0, -0.05], [3.2, 0, 0], [0, 2.0, 0], s, wall_colour),
        # Cabinet shell, open at the front (z = 0.40): back, two sides, top, shelves.
        sample_rect(rng, [-0.4, 0, 0], [0.8, 0, 0], [0, 1.0, 0], s, solid([0.80, 0.78, 0.70])),
        sample_rect(rng, [-0.4, 0, 0], [0, 0, 0.4], [0, 1.0, 0], s, wood_colour),
        sample_rect(rng, [0.4, 0, 0], [0, 0, 0.4], [0, 1.0, 0], s, wood_colour),
        sample_rect(rng, [-0.4, 1.0, 0], [0.8, 0, 0], [0, 0, 0.4], s, solid([0.45, 0.30, 0.18])),
        sample_rect(rng, [-0.4, 0.30, 0], [0.8, 0, 0], [0, 0, 0.4], s, shelf),
        sample_rect(rng, [-0.4, 0.65, 0], [0.8, 0, 0], [0, 0, 0.4], s, shelf),
        sample_rect(rng, [-0.4, 0.02, 0], [0.8, 0, 0], [0, 0, 0.4], s, solid([0.35, 0.28, 0.2])),
    ]
    return concat_points(parts)


def door_points(rng) -> Points:
    """Cabinet door: x in [-0.40, 0.38], y in [0.31, 0.99], z in [0.40, 0.42]; handle toward the free edge."""
    s = PART_SPACING
    lo, hi = np.array([-0.40, 0.31, 0.40]), np.array([0.38, 0.99, 0.42])

    def front(a, b, pos):
        base = np.where(_stripes(pos[:, 0], 0.10)[:, None], [0.62, 0.42, 0.25], [0.50, 0.32, 0.18])
        base = base * (0.8 + 0.4 * b[:, None])
        logo = (pos[:, 0] > -0.32) & (pos[:, 0] < -0.20) & (pos[:, 1] > 0.84) & (pos[:, 1] < 0.94)
        return np.where(logo[:, None], [0.95, 0.80, 0.10], base)

    def back(a, b, pos):
        base = np.where(_stripes(pos[:, 0] + pos[:, 1], 0.12)[:, None], [0.18, 0.55, 0.62], [0.85, 0.92, 0.90])
        mark = (pos[:, 0] > 0.20) & (pos[:, 1] < 0.45)
        return np.where(mark[:, None], [0.7, 0.15, 0.55], base)

    door = sample_box(rng, lo, hi, s, {"+z": front, "-z": back, "*": [0.9, 0.9, 0.88]})
    handle = sample_box(rng, [0.27, 0.58, 0.42], [0.31, 0.72, 0.45], s, {"*": [0.78, 0.78, 0.85]})
    return concat_points([door, handle])


def drawer_points(rng) -> Points:
    """Drawer: front panel plus an open-top tray behind it, closed pose, slides along +z."""
    s = PART_SPACING

    def front(a, b, pos):
        base = np.where(_stripes(pos[:, 1], 0.05)[:, None], [0.30, 0.38, 0.55], [0.22, 0.28, 0.42])
        logo = (pos[:, 0] < -0.22) & (pos[:, 0] > -0.34) & (pos[:, 1] > 0.18) & (pos[:, 1] < 0.26)
        return np.where(logo[:, None], [0.95, 0.5, 0.1], base)

    def side(a, b, pos):
        checker = _stripes(pos[:, 2], 0.06) ^ _stripes(pos[:, 1], 0.06)
        return np.where(checker[:, None], [0.9, 0.85, 0.7], [0.6, 0.5, 0.35])

    def bottom(a, b, pos):
        return np.tile([0.3, 0.55, 0.3], (len(a), 1)) * (0.7 + 0.5 * a[:, None] * b[:, None] + 0.1 * b[:, None])

    panel = sample_box(rng, [-0.39, 0.04, 0.40], [0.39, 0.29, 0.42], s, {"+z": front, "*": [0.4, 0.4, 0.45]})
    tray = [
        sample_rect(rng, [-0.37, 0.04, 0.10], [0, 0.20, 0], [0, 0, 0.30], s, side),
        sample_rect(rng, [0.37, 0.04, 0.10], [0, 0.20, 0], [0, 0, 0.30], s, side),
        sample_rect(rng, [-0.37, 0.04, 0.10], [0.74, 0, 0], [0, 0, 0.30], s, bottom),
        sample_rect(rng, [-0.37, 0.04, 0.10], [0.74, 0, 0], [0, 0.20, 0], s, solid([0.85, 0.55, 0.50])),
        sample_rect(rng, [-0.37, 0.04, 0.40], [0.74, 0, 0], [0, 0.20, 0], s, solid([0.85, 0.85, 0.6])),
    ]
    handle = sample_box(rng, [-0.10, 0.14, 0.42], [0.10, 0.18, 0.45], s, {"*": [0.8, 0.8, 0.82]})
    return concat_points([panel, *tray, handle])


def reveal_points(rng) -> Points:
    """Objects on the cabinet's middle shelf that only exist once the door is open."""
    s = REVEAL_SPACING
    return concat_points([
        sample_box(rng, [-0.32, 0.65, 0.12], [-0.20, 0.79, 0.24], s, {"*": [0.85, 0.12, 0.12]}),
        sample_sphere(rng, [0.02, 0.72, 0.22], 0.07, s, [0.15, 0.7, 0.2]),
        sample_box(rng, [0.16, 0.65, 0.14], [0.26, 0.85, 0.24], s, {"*": [0.15, 0.25, 0.85]}),
        sample_sphere(rng, [-0.05, 0.85, 0.12], 0.05, s, [0.95, 0.85, 0.1]),
    ])


def _to_splats(rng, points: Points, spacing: float, rot=None, trans=None, jitter: float = 0.0) -> Splats:
    pos, normals, rgb = points
    if rot is not None:
        pos, normals = pos @ rot.T + trans, normals @ rot.T
    if jitter > 0.0:
        pos = pos + rng.normal(0.0, jitter, size=pos.shape)
    return make_gaussians(rng, pos, normals, rgb, spacing)


STATIC_SKIN_PATCH = (0.35, 0.7)  # fraction of the door's width (along its longer in-plane axis) that is labelled static


def build_scene(out: Path, seed: int = 0, static_skin: bool = False) -> dict:
    """Write the synthetic scene into ``out``; returns the ground-truth joints keyed by slot id.

    ``static_skin``: a vertical band of the door's closed skin is labelled *static* instead of base (like the patchy
    labels of a real scan); the closed door gaussians of that band are appended to static.ply, and their count is
    recorded as ``static_skin`` in the ground truth of slot01.
    """
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    static_splats = _to_splats(rng, static_points(rng), STATIC_SPACING)

    door_rot, door_trans = hinge_transform(DOOR_HINGE_AXIS, DOOR_HINGE_PIVOT, np.deg2rad(DOOR_OPEN_DEG))
    door_base = _to_splats(rng, door_points(rng), PART_SPACING)
    patch_rows = 0
    if static_skin:
        x = door_base.positions[:, 0]
        lo, hi = x.min(), x.max()
        band = (x >= lo + STATIC_SKIN_PATCH[0] * (hi - lo)) & (x <= lo + STATIC_SKIN_PATCH[1] * (hi - lo))
        static_splats = Splats.concat([static_splats, door_base.subset(band)])
        door_base = door_base.subset(~band)
        patch_rows = int(band.sum())
    write_ply(out / "static.ply", static_splats)
    write_ply(out / "slot01_base.ply", door_base)
    switched_door = _to_splats(rng, door_points(rng), PART_SPACING, door_rot, door_trans, JITTER)
    reveal = _to_splats(rng, reveal_points(rng), REVEAL_SPACING)
    write_ply(out / "slot01_switched.ply", Splats.concat([switched_door, reveal]))

    write_ply(out / "slot02_base.ply", _to_splats(rng, drawer_points(rng), PART_SPACING))
    write_ply(
        out / "slot02_switched.ply",
        _to_splats(rng, drawer_points(rng), PART_SPACING, np.eye(3), DRAWER_OPEN * DRAWER_AXIS, JITTER),
    )

    def files(slot: str) -> dict:
        return {"base": f"{slot}_base.ply", "switched": f"{slot}_switched.ply", "part": None, "reveal": None,
                "vanish": None}

    scene = {
        "version": 1,
        "name": "synthetic",
        "up": [0, 1, 0],
        "static": "static.ply",
        "parts": [
            {"id": "slot01", "name": "cabinet door", "files": files("slot01"), "joint": None},
            {"id": "slot02", "name": "drawer", "files": files("slot02"), "joint": None},
        ],
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")

    truth = {
        "slot01": {"kind": "hinge", "axis": DOOR_HINGE_AXIS.tolist(), "pivot": DOOR_HINGE_PIVOT.tolist(),
                   "min": 0, "max": DOOR_OPEN_DEG, "open": DOOR_OPEN_DEG, "static_skin": patch_rows},
        "slot02": {"kind": "slider", "axis": DRAWER_AXIS.tolist(), "pivot": [0.0, 0.0, 0.0],
                   "min": 0, "max": DRAWER_OPEN, "open": DRAWER_OPEN},
    }
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    return truth


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--static-skin", action="store_true",
                        help="label a vertical band of the door's closed skin static (tests skin recruitment)")
    args = parser.parse_args(argv)
    build_scene(args.out, args.seed, args.static_skin)
    counts = {p.name: len(read_ply(p)) for p in sorted(args.out.glob("*.ply"))}
    print(f"wrote {args.out}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
