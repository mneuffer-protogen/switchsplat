# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimate the 1-DOF joint of every part from its base (closed) and switched (open) gaussians.

For each slot the rigid transform base -> switched is found with a hypothesis sweep that exploits
the one-degree-of-freedom prior (hinge about / slide along the part's principal or world axes),
refined by trimmed ICP and re-projected onto an exact hinge or slider. The base gaussians that land
on the switched set form ``slotNN_part.ply`` (stored in the base pose), the unmatched base ones
``slotNN_vanish.ply`` and the unmatched switched ones ``slotNN_reveal.ply``. ``scene.json`` gets the
joint (other fields are preserved).

Uses numpy and scipy only. Run as ``python -m tools.articulation.estimate_joint <dir>``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geometry import apply, hinge_transform, rodrigues, rotation_to_axis_angle
from .ply_io import Splats, read_ply, write_ply

COLOUR_WEIGHT = 0.05  # metres of feature distance per unit RGB difference
TRIM = 0.7  # fraction of best correspondences that enter a score
HINGE_MIN_DEG = 5.0  # below this rotation the motion is a slider
MAX_BASE_SAMPLES = 3000
MAX_SWITCHED_SAMPLES = 6000
SWEEP_SAMPLES = 1500  # base points used for the hypothesis sweep (cheaper than the ICP set)
MIN_SET = 50  # vanish / reveal sets smaller than this (or 1% of their parent) are noise
NUM_REFINED = 6
SMALL_PART_DIAGONAL = 1.2  # metres; below this a part gets the "cabinet door" mass


@dataclass
class PartFit:
    kind: str  # "hinge" | "slider"
    axis: np.ndarray
    pivot: np.ndarray
    open: float  # degrees (hinge) or metres (slider)
    rot: np.ndarray  # final rigid transform base -> switched, p' = rot p + trans
    trans: np.ndarray
    screw: float  # translation along the hinge axis left over by the 6-DOF fit, metres
    score: float


# --------------------------------------------------------------------------------------------
# Features and scoring
# --------------------------------------------------------------------------------------------

def _features(points: np.ndarray, rgb: np.ndarray, weight: float = COLOUR_WEIGHT) -> np.ndarray:
    return np.concatenate([points, weight * rgb], axis=1)


def _subsample(count: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.sort(rng.choice(count, size=limit, replace=False))


def trimmed_mean(distances: np.ndarray, trim: float = TRIM) -> np.ndarray:
    """Mean of the smallest ``trim`` fraction along the last axis."""
    k = max(int(np.ceil(trim * distances.shape[-1])), 1)
    return np.partition(distances, k - 1, axis=-1)[..., :k].mean(axis=-1)


class Matcher:
    """One-directional base -> switched nearest-neighbour scoring (switched also holds extra reveal content)."""

    def __init__(self, base: Splats, switched: Splats, rng: np.random.Generator):
        sw_idx = _subsample(len(switched), MAX_SWITCHED_SAMPLES, rng)
        self.sw_pos = switched.positions[sw_idx].astype(np.float64)
        self.sw_rgb = np.clip(switched.rgb[sw_idx], 0.0, 1.0).astype(np.float64)
        self.tree = cKDTree(_features(self.sw_pos, self.sw_rgb))
        b_idx = _subsample(len(base), MAX_BASE_SAMPLES, rng)
        self.base_pos = base.positions[b_idx].astype(np.float64)
        self.base_rgb = np.clip(base.rgb[b_idx], 0.0, 1.0).astype(np.float64)
        sweep = _subsample(len(self.base_pos), SWEEP_SAMPLES, rng)
        self.sweep_pos, self.sweep_rgb = self.base_pos[sweep], self.base_rgb[sweep]
        extent = self.base_pos.max(axis=0) - self.base_pos.min(axis=0)
        self.diagonal = float(np.linalg.norm(extent))

    def query(self, points: np.ndarray, rgb: np.ndarray):
        return self.tree.query(_features(points, rgb), k=1, workers=-1)

    def score(self, rot: np.ndarray, trans: np.ndarray, trim: float = TRIM) -> float:
        dist, _ = self.query(apply(rot, trans, self.base_pos), self.base_rgb)
        return float(trimmed_mean(dist, trim))

    def score_batch(self, points: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """points [A, N, 3] (A hypotheses of the sweep subsample) -> trimmed score [A]."""
        a, n, _ = points.shape
        feats = np.concatenate([points, np.broadcast_to(COLOUR_WEIGHT * rgb, (a, n, 3))], axis=2)
        dist, _ = self.tree.query(feats.reshape(a * n, 6), k=1, workers=-1)
        return trimmed_mean(dist.reshape(a, n))


# --------------------------------------------------------------------------------------------
# Hypothesis sweep
# --------------------------------------------------------------------------------------------

def principal_frame(points: np.ndarray) -> np.ndarray:
    """Rows are the principal axes (largest variance first), right-handed."""
    centred = points - points.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(centred.T))
    axes = vecs[:, ::-1].T
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]
    return axes


def candidate_hypotheses(points: np.ndarray):
    """Yield (kind, axis, pivot-or-None, group-key) for the 1-DOF sweep.

    Hinge axes: every axis of the principal frame and of the world frame, with each of the four
    box edges parallel to that axis as pivot line. Slider axes: the same axes, both signs.
    """
    centre = points.mean(axis=0)
    frames = [("pca", principal_frame(points)), ("world", np.eye(3))]
    for fname, axes in frames:
        proj = (points - centre) @ axes.T
        lo, hi = proj.min(axis=0), proj.max(axis=0)
        for i in range(3):
            j, k = [d for d in range(3) if d != i]
            mid_i = 0.5 * (lo[i] + hi[i])
            for cj in (lo[j], hi[j]):
                for ck in (lo[k], hi[k]):
                    local = np.zeros(3)
                    local[i], local[j], local[k] = mid_i, cj, ck
                    yield "hinge", axes[i], centre + local @ axes, (fname, i, cj == lo[j], ck == lo[k])
            for sign in (1.0, -1.0):
                yield "slider", sign * axes[i], None, (fname, i, sign)


def sweep(matcher: Matcher):
    """Score all hypotheses; returns the best (score, kind, axis, pivot, parameter) per group, sorted."""
    pos, rgb = matcher.sweep_pos, matcher.sweep_rgb
    box = pos.max(axis=0) - pos.min(axis=0)
    largest = float(box.max())
    angles = np.deg2rad(np.concatenate([np.arange(-170, 0, 5), np.arange(5, 171, 5)]).astype(float))
    distances = np.arange(0.02 * largest, 2.5 * largest + 1e-9, 0.02 * largest)
    seen_slider: list[np.ndarray] = []
    results = []
    for kind, axis, pivot, key in candidate_hypotheses(matcher.base_pos):
        if kind == "hinge":
            rots = rodrigues(axis, angles)  # [A,3,3]
            moved = np.einsum("aij,nj->ani", rots, pos - pivot) + pivot
            scores = matcher.score_batch(moved, rgb)
            best = int(np.argmin(scores))
            results.append((float(scores[best]), kind, axis, pivot, float(angles[best])))
        else:
            if any(np.dot(axis, other) > 0.9995 for other in seen_slider):
                continue
            seen_slider.append(axis)
            moved = pos[None] + distances[:, None, None] * axis[None, None]
            scores = matcher.score_batch(moved, rgb)
            best = int(np.argmin(scores))
            results.append((float(scores[best]), kind, axis, None, float(distances[best])))
    results.sort(key=lambda r: r[0])
    return results


# --------------------------------------------------------------------------------------------
# Trimmed ICP
# --------------------------------------------------------------------------------------------

def kabsch(source: np.ndarray, target: np.ndarray):
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    h = (source - mu_s).T @ (target - mu_t)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, mu_t - rot @ mu_s


def trimmed_icp(matcher: Matcher, rot: np.ndarray, trans: np.ndarray, iters: int = 30,
                trim_start: float = 0.85, trim_end: float = 0.6):
    """6-DOF trimmed point-to-point ICP; the kept fraction shrinks linearly from trim_start to trim_end."""
    for it in range(iters):
        frac = trim_start + (trim_end - trim_start) * it / max(iters - 1, 1)
        moved = apply(rot, trans, matcher.base_pos)
        dist, idx = matcher.query(moved, matcher.base_rgb)
        keep = np.argpartition(dist, max(int(frac * len(dist)) - 1, 0))[: max(int(frac * len(dist)), 3)]
        d_rot, d_trans = kabsch(moved[keep], matcher.sw_pos[idx[keep]])
        rot, trans = d_rot @ rot, d_rot @ trans + d_trans
        if np.linalg.norm(d_trans) < 1e-6 and abs(np.trace(d_rot) - 3.0) < 1e-10:
            break
    return rot, trans


# --------------------------------------------------------------------------------------------
# Joint conversion and 1-DOF refinement
# --------------------------------------------------------------------------------------------

def golden_minimize(fn, lo: float, hi: float, tol: float = 1e-4) -> float:
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = fn(c), fn(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = fn(d)
    return 0.5 * (a + b)


def rigid_to_joint(matcher: Matcher, rot: np.ndarray, trans: np.ndarray) -> PartFit:
    """Convert a 6-DOF fit to an exact hinge or slider and refine its single parameter."""
    centroid = matcher.base_pos.mean(axis=0)
    axis, angle = rotation_to_axis_angle(rot)
    trans_c = trans - (np.eye(3) - rot) @ centroid  # translation of the centred problem
    if np.degrees(angle) > HINGE_MIN_DEG:
        # Closest axis point to the centroid. I - R has one (numerically tiny) singular value along the
        # axis, which must be cut: the others are 2 sin(angle / 2) >= 0.087 for angles above 5 deg.
        pivot = centroid + np.linalg.pinv(np.eye(3) - rot, rcond=1e-2) @ trans_c
        screw = float(abs(trans_c @ axis))

        def cost(theta: float) -> float:
            r, t = hinge_transform(axis, pivot, theta)
            return matcher.score(r, t)

        theta = golden_minimize(cost, angle - np.deg2rad(4.0), angle + np.deg2rad(4.0))
        r, t = hinge_transform(axis, pivot, theta)
        return PartFit("hinge", axis, pivot, float(np.degrees(theta)), r, t, screw, matcher.score(r, t))

    motion = rot @ centroid + trans - centroid  # displacement of the centroid
    length = float(np.linalg.norm(motion))
    axis = motion / length if length > 1e-12 else np.array([1.0, 0.0, 0.0])

    def cost_slide(dist: float) -> float:
        return matcher.score(np.eye(3), dist * axis)

    dist = golden_minimize(cost_slide, max(length - 0.02 * matcher.diagonal, 0.0), length + 0.02 * matcher.diagonal)
    return PartFit("slider", axis, centroid, dist, np.eye(3), dist * axis, 0.0, cost_slide(dist))


def fit_part(base: Splats, switched: Splats, seed: int = 0) -> PartFit:
    rng = np.random.default_rng(seed)
    matcher = Matcher(base, switched, rng)
    hypotheses = sweep(matcher)[:NUM_REFINED]
    best: PartFit | None = None
    for score, kind, axis, pivot, param in hypotheses:
        if kind == "hinge":
            rot, trans = hinge_transform(axis, pivot, param)
        else:
            rot, trans = np.eye(3), param * axis
        rot, trans = trimmed_icp(matcher, rot, trans)
        fit = rigid_to_joint(matcher, rot, trans)
        if best is None or fit.score < best.score:
            best = fit
    assert best is not None
    best.score = float(best.score)
    return best


# --------------------------------------------------------------------------------------------
# Inlier split
# --------------------------------------------------------------------------------------------

def median_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    dist, _ = cKDTree(points).query(points, k=2, workers=-1)
    return float(np.median(dist[:, 1]))


def smooth_labels(points: np.ndarray, labels: np.ndarray, k: int = 9, rounds: int = 2) -> np.ndarray:
    """Majority vote among the k nearest neighbours (self included).

    Independent random sampling leaves isolated points without a partner inside the inlier
    distance; the vote removes such salt-and-pepper labels and keeps coherent regions.
    """
    if len(points) <= k or labels.all() or not labels.any():
        return labels.copy()
    _, nbr = cKDTree(points).query(points, k=k, workers=-1)
    labels = labels.copy()
    for _ in range(rounds):
        labels = labels[nbr].mean(axis=1) > 0.5
    return labels


@dataclass
class SplitResult:
    part: np.ndarray  # bool [Nb]
    reveal: np.ndarray  # bool [Ns]
    threshold: float
    rmse: float
    inlier_ratio: float


def split_inliers(base: Splats, switched: Splats, rot: np.ndarray, trans: np.ndarray,
                  inlier_dist: float | None = None) -> SplitResult:
    """Classify every base gaussian (part / vanish) and switched gaussian (registered / reveal).

    A base gaussian is part when its transformed centre has a switched centre within the threshold;
    a switched gaussian is reveal when no transformed base centre is that close. The default
    threshold is 2.5x the larger of the two median nearest-neighbour spacings.
    """
    if inlier_dist is None:
        inlier_dist = 2.5 * max(median_spacing(base.positions), median_spacing(switched.positions))
    moved = apply(rot, trans, base.positions.astype(np.float64))
    sw = switched.positions.astype(np.float64)
    d_base, _ = cKDTree(sw).query(moved, k=1, workers=-1)
    d_sw, _ = cKDTree(moved).query(sw, k=1, workers=-1)
    raw_part = d_base <= inlier_dist
    rmse = float(np.sqrt(np.mean(d_base[raw_part] ** 2))) if raw_part.any() else float("nan")
    part = smooth_labels(base.positions, raw_part)
    reveal = smooth_labels(switched.positions, d_sw > inlier_dist)
    return SplitResult(part, reveal, float(inlier_dist), rmse, float(part.mean()))


def clean_split(split: SplitResult) -> tuple[np.ndarray, np.ndarray]:
    """Fold a negligible vanish set back into the part and drop a negligible reveal set.

    "Negligible" means fewer than max(50 gaussians, 1% of the parent set): such leftovers are
    sampling noise, and as separate fade-in/out layers they would only flicker.
    """
    part, reveal = split.part.copy(), split.reveal.copy()
    vanish_count = int((~part).sum())
    if vanish_count < max(MIN_SET, 0.01 * len(part)):
        part[:] = True
    if int(reveal.sum()) < max(MIN_SET, 0.01 * len(reveal)):
        reveal[:] = False
    return part, reveal


# --------------------------------------------------------------------------------------------
# Scene I/O
# --------------------------------------------------------------------------------------------

def joint_defaults(kind: str, base_positions: np.ndarray) -> dict:
    diagonal = float(np.linalg.norm(base_positions.max(axis=0) - base_positions.min(axis=0)))
    small = diagonal < SMALL_PART_DIAGONAL
    return {
        "friction": 1.5,
        "frictionTorque": 0 if small else 5,
        "latch": 3 if kind == "hinge" else 0.02,  # degrees for a hinge, metres for a slider
        "mass": 2 if small else 25,
    }


def _round(values, digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in values]


def process_part(scene_dir: Path, part: dict, inlier_dist: float | None, seed: int = 0) -> str:
    """Estimate one part, write its PLYs and update ``part`` in place. Returns a summary line."""
    files = part["files"]
    base = read_ply(scene_dir / files["base"])
    switched = read_ply(scene_dir / files["switched"])
    fit = fit_part(base, switched, seed)
    split = split_inliers(base, switched, fit.rot, fit.trans, inlier_dist)
    part_mask, reveal_mask = clean_split(split)

    slot = part["id"]
    outputs = {
        "part": base.subset(part_mask),
        "vanish": base.subset(~part_mask) if (~part_mask).any() else None,
        "reveal": switched.subset(reveal_mask) if reveal_mask.any() else None,
    }
    for key, splats in outputs.items():
        name = f"{slot}_{key}.ply"
        if splats is None:
            files[key] = None
            (scene_dir / name).unlink(missing_ok=True)
        else:
            write_ply(scene_dir / name, splats)
            files[key] = name

    joint = {
        "kind": fit.kind,
        "axis": _round(fit.axis),
        "pivot": _round(fit.pivot),
        "min": 0,
        "max": round(fit.open, 4),
        "open": round(fit.open, 4),
        **joint_defaults(fit.kind, base.positions),
        "fit": {"rmse": round(split.rmse, 5), "inlierRatio": round(split.inlier_ratio, 4), "screw": round(fit.screw, 5)},
    }
    part["joint"] = joint
    unit = "deg" if fit.kind == "hinge" else "m"
    return (
        f"{slot} ({part.get('name', '')}): {fit.kind} axis={joint['axis']} pivot={joint['pivot']} "
        f"open={joint['open']}{unit} rmse={split.rmse * 1000:.1f}mm inliers={split.inlier_ratio:.2f} "
        f"part={int(part_mask.sum())} vanish={int((~part_mask).sum())} reveal={int(reveal_mask.sum())}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir", type=Path, help="export directory holding scene.json")
    parser.add_argument("--slots", type=int, nargs="*", default=None, help="1-based slot numbers (default: all)")
    parser.add_argument("--inlier-dist", default="auto", help="'auto' (2.5x median spacing) or a distance in metres")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    inlier = None if args.inlier_dist == "auto" else float(args.inlier_dist)

    scene_path = args.scene_dir / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    wanted = None if args.slots is None else {f"slot{s:02d}" for s in args.slots}
    for part in scene["parts"]:
        if wanted is not None and part["id"] not in wanted:
            continue
        print(process_part(args.scene_dir, part, inlier, args.seed), flush=True)
    scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
