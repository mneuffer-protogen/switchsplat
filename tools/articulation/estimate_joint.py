# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Estimate the 1-DOF joint of every part from its base (closed) and switched (open) gaussians.

For each slot the rigid transform base -> switched is found with a hypothesis sweep that exploits
the one-degree-of-freedom prior (hinge about / slide along the part's principal or world axes),
refined by trimmed ICP and re-projected onto an exact hinge or slider. The base gaussians that land
on the switched set form ``slotNN_part.ply`` (stored in the base pose), the unmatched base ones
``slotNN_vanish.ply`` and the unmatched switched ones ``slotNN_reveal.ply``. ``scene.json`` gets the
joint (other fields are preserved).

Soft part, fused door: the joint is fitted on the confident base / switched gaussians only (the rows flagged
``base_fit`` / ``switched_fit`` in ``slotNN_membership.npz`` when the importer wrote soft sets, otherwise every
row), but the part is built from every candidate: all registered base gaussians (the stray-removal trim only
drops isolated floaters and sparse wisps; a sparse flank of a hood is no floater), plus the open-pose door.
Mapped back into the closed pose by the inverse joint transform (``fuse.py``: positions, orientations and the
view-dependent colour are rotated with it) the switched gaussians that lie within ``--fuse-reach`` median spacings
(and the part's thickness) of the part are a second, independent sample of the same rigid body, which the
non-moving reveal set used to hold: a ghost door fading in at the open pose while the real one swings. Those that
repeat a closed-pose gaussian (within ``--fuse-dup`` spacings, similar colour and size) are dropped, the rest are

* ``near`` (the same surface and colour as the closed part): always added to the part;
* ``inner`` (the inner face, the edge, the window frame; invisible when closed unless the skin is translucent),
  by ``--inner``: ``part`` adds them to the part too (``auto`` does so for an opaque part, whose median opacity is
  at least 0.75), ``reveal`` leaves them waiting at the open pose (the old behaviour), ``attach`` writes
  ``slotNN_partReveal.ply`` (closed-pose coordinates, ``files.partReveal``) for a viewer that moves it with the
  part and fades it in with q: closed look untouched, no ghost, the whole door open.

What does not belong to the door stays reveal (open-pose only, fades in) and vanish (closed-pose only, fades
out). ``--no-fuse`` gives the old behaviour (switched gaussians never join the part).

Uses numpy and scipy only. Run as ``python -m tools.articulation.estimate_joint <dir>``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .export_parts import membership_path
from .fuse import fuse_open_pose
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
AXIS_SNAP_DEG = 15.0  # a hinge axis this close to vertical / horizontal is snapped to it
PLANE_OFFSET_SPACINGS = 4.0  # ... and only when the pivot is further than this many spacings off the plane
THIN_PLANE_RATIO = 0.35  # smallest / middle singular value below which a base set counts as a thin sheet
SNAP_TOLERANCE_SPACINGS = 0.25  # a pivot snap may worsen the score by this many median spacings
MERGE_AXIS_DEG = 8.0  # parts whose axes differ by less than this (same sign) may be the same joint
MERGE_LINE_DIST = 0.08  # ... and whose hinge lines pass this close to each other near the part (metres)
MERGE_DEDUP = 0.005  # gaussians of the second slot closer than this to one of the first are dropped (metres)
MERGE_MIN_OVERLAP = 0.2  # the closed poses must overlap: fraction of the smaller base set near the other
TRIM_CORE_SPACINGS = 8.0  # body trim: a gaussian needs TRIM_MIN_DEGREE neighbours within this many median spacings
TRIM_MIN_DEGREE = 2
TRIM_LINK_SPACINGS = 10.0  # ... and the body is what links up within this many spacings (panel lines of a sparse part)
TRIM_MAD_K = 12.0  # ... and lie within median +- k x 1.4826 MAD along the body's principal axes (a far-out guard only)
TRIM_SOLID_ALPHA = 0.1  # gaussians fainter than this (or half the part's median) do not define the body
TRIM_SOLID_SCALE = 4.0  # ... nor those whose footprint exceeds this multiple of the median
TRIM_KEEP_FRACTION = 0.05  # a further component joins the body when it carries this share of the main one's weight
INNER_MODES = ("auto", "part", "reveal", "attach")
OPAQUE_ALPHA = 0.75  # parts whose median opacity is at least this hide what is behind their skin ("auto" inner mode)
FUSE_REACH_SPACINGS = 10.0  # open-pose gaussians this close (median spacings) to the closed part belong to the door
FUSE_DUP_SPACINGS = 1.5  # ... and are dropped when a similar closed-pose gaussian is this close (the median nearest-neighbour
#                          distance is about half the mean point spacing, so 0.5 would only catch ~15 % of a second sample)
CLUSTER_LINK_SPACINGS = 5.0  # base points closer than this many median spacings belong to one cluster


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


def snap_axis_to_gravity(matcher: Matcher, fit: PartFit, up: np.ndarray | None) -> PartFit:
    """Prefer a vertical or exactly horizontal hinge axis when the data cannot tell it from the fitted one.

    Doors turn about the up axis, hoods, hatches and lids about a horizontal one; a free 6-DOF fit of noisy
    gaussians tilts the axis by a few degrees. When the fitted axis is within AXIS_SNAP_DEG of the up vector
    (or of its horizontal projection) the hinge is re-fitted with that axis fixed (pivot and angle by
    Nelder-Mead on the registration score) and kept if the score is not worse by more than a fraction of the
    point spacing.
    """
    if fit.kind != "hinge" or up is None:
        return fit
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    axis = fit.axis
    candidates = [up * np.sign(axis @ up)]
    flat = axis - (axis @ up) * up
    if np.linalg.norm(flat) > 1e-6:
        candidates.append(flat / np.linalg.norm(flat))
    centroid = matcher.base_pos.mean(axis=0)
    spacing = median_spacing(matcher.base_pos)
    best = fit
    for target in candidates:
        tilt = np.degrees(np.arccos(np.clip(axis @ target, -1.0, 1.0)))
        if tilt < 1e-3 or tilt > AXIS_SNAP_DEG:
            continue
        e1 = np.cross(target, [1.0, 0.0, 0.0] if abs(target[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(target, e1)
        start = fit.pivot + ((centroid - fit.pivot) @ axis) * axis  # point of the old axis nearest the centroid

        def cost(params: np.ndarray) -> float:
            r, t = hinge_transform(target, start + params[1] * e1 + params[2] * e2, params[0])
            return matcher.score(r, t)

        x0 = np.array([np.deg2rad(fit.open), 0.0, 0.0])
        steps = np.array([np.deg2rad(3.0), 0.02 * matcher.diagonal, 0.02 * matcher.diagonal])
        simplex = np.vstack([x0] + [x0 + np.diag(steps)[i] for i in range(3)])
        result = minimize(cost, x0, method="Nelder-Mead",
                          options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
        if result.fun > fit.score + SNAP_TOLERANCE_SPACINGS * spacing:
            continue
        pivot = start + result.x[1] * e1 + result.x[2] * e2
        r, t = hinge_transform(target, pivot, result.x[0])
        new_axis, new_angle = (target, result.x[0]) if result.x[0] >= 0 else (-target, -result.x[0])  # open is positive
        best = PartFit("hinge", new_axis, pivot, float(np.degrees(new_angle)), r, t, fit.screw, float(result.fun))
        break  # the vertical axis (first candidate) wins over the horizontal projection
    return best


def snap_pivot_to_plane(matcher: Matcher, fit: PartFit) -> PartFit:
    """Resolve the pivot ambiguity of a thin part (hood, hatch, door skin) by putting the hinge in its plane.

    A thin sheet that is registered onto its open pose constrains the rotation angle and the axis direction,
    but not where along the sheet the pivot sits: every centre on the angle bisector of the closed and open
    planes maps the closed sheet onto the same open sheet, and a partial or narrow sample of the sheet cannot tell
    them apart. Physically the hinge lies at the part's edge, in the plane of the closed sheet, and that is the
    single member of the family that does. The snap is applied only when the axis runs along the sheet, the
    fitted pivot is clearly off the sheet's plane, and the registration score stays as good as before.
    """
    if fit.kind != "hinge":
        return fit
    pts = matcher.base_pos
    centroid = pts.mean(axis=0)
    _, sing, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2]
    if sing[2] > THIN_PLANE_RATIO * sing[1] or abs(fit.axis @ normal) > 0.5:
        return fit
    theta = np.deg2rad(fit.open)
    rot = rodrigues(fit.axis, theta)
    along = np.cross(normal, fit.axis)
    along /= np.linalg.norm(along)
    w = np.linalg.pinv(np.eye(3) - rot, rcond=1e-2) @ (rot @ along)  # pivot shift that slides the open sheet in-plane
    denom = float(normal @ w)
    if abs(denom) < 1e-6:
        return fit
    offset = float(normal @ (fit.pivot - centroid))
    if abs(offset) < max(PLANE_OFFSET_SPACINGS * median_spacing(pts), 0.03 * matcher.diagonal):
        return fit  # already (nearly) in the plane: the data pinned it down
    shift = -offset / denom
    pivot = fit.pivot + shift * w
    if np.linalg.norm(pivot - fit.pivot) > 1.5 * matcher.diagonal:
        return fit

    def cost(angle: float) -> float:
        r, t = hinge_transform(fit.axis, pivot, angle)
        return matcher.score(r, t)

    angle = golden_minimize(cost, theta - np.deg2rad(4.0), theta + np.deg2rad(4.0))
    new_score = cost(angle)
    if new_score > fit.score + SNAP_TOLERANCE_SPACINGS * median_spacing(pts):
        return fit
    r, t = hinge_transform(fit.axis, pivot, angle)
    return PartFit("hinge", fit.axis, pivot, float(np.degrees(angle)), r, t, fit.screw, new_score)


def fit_part(base: Splats, switched: Splats, seed: int = 0, up=None) -> PartFit:
    rng = np.random.default_rng(seed)
    matcher = Matcher(base.subset(largest_cluster(base.positions.astype(np.float64))), switched, rng)
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
    return snap_pivot_to_plane(matcher, snap_axis_to_gravity(matcher, best, up))


# --------------------------------------------------------------------------------------------
# Inlier split
# --------------------------------------------------------------------------------------------

def median_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    dist, _ = cKDTree(points).query(points, k=2, workers=-1)
    return float(np.median(dist[:, 1]))


def largest_cluster(points: np.ndarray, link_factor: float = CLUSTER_LINK_SPACINGS) -> np.ndarray:
    """Boolean mask of the biggest connected cluster (points closer than ``link_factor`` x the median spacing).

    Real part sets carry stray floaters far from the body; they stretch the bounding box the hinge
    hypotheses are built from and drag the fit, so the fit only sees the main cluster.
    """
    n = len(points)
    if n < 10:
        return np.ones(n, dtype=bool)
    radius = link_factor * median_spacing(points)
    if radius <= 0.0:
        return np.ones(n, dtype=bool)
    tree = cKDTree(points)
    graph = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    _, labels = connected_components(graph, directed=False)
    return labels == np.argmax(np.bincount(labels))


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
# Trimming the part to its coherent body
# --------------------------------------------------------------------------------------------

def _components(points: np.ndarray, radius: float) -> np.ndarray:
    tree = cKDTree(points)
    graph = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    return connected_components(graph, directed=False)[1]


def trim_body(base: Splats, candidate: np.ndarray, link_factor: float = TRIM_LINK_SPACINGS,
              core_factor: float = TRIM_CORE_SPACINGS,
              min_degree: int = TRIM_MIN_DEGREE, mad_k: float = TRIM_MAD_K,
              weights: np.ndarray | None = None) -> np.ndarray:
    """Mask (over ``base``) of the coherent body of the part among the ``candidate`` registered gaussians.

    Registered gaussians of a real part still include floaters and bits of neighbouring panels that happen to
    land on switched gaussians. The body is found at a scale given by the part itself (the link radius is
    ``link_factor`` x the median nearest-neighbour spacing, so a sparse car door and a dense cabinet door are
    treated alike):

    1. only "solid" gaussians define it: not near-transparent (alpha < TRIM_SOLID_ALPHA) and not oversized
       (footprint above TRIM_SOLID_SCALE x the median); glass is faint, so a part that is faint throughout falls
       back to its own median alpha;
    2. a k-core filter (every kept gaussian has ``min_degree`` neighbours within the link radius) removes chains
       and wisps that dangle off the body;
    3. of the connected components the heaviest (by summed alpha) is the body, together with every component
       that carries at least TRIM_KEEP_FRACTION of its weight (a second panel of the same part);
    4. a robust extent clamp along the body's principal axes (median +- ``mad_k`` x 1.4826 MAD) drops what is
       still far out;
    5. non-solid gaussians (faint or large) are kept only when they lie within the link radius of that body.

    ``weights`` [len(base)] are the opacity multipliers of soft members (see ``export_parts``): a soft member is
    faint because it only carries part of a gaussian, so its solidity is judged on opacity / weight.
    """
    keep = np.zeros(len(base), dtype=bool)
    idx = np.flatnonzero(candidate)
    if len(idx) < 4 * min_degree + 4:
        keep[idx] = True
        return keep
    pts = base.positions[idx].astype(np.float64)
    alpha = 1.0 / (1.0 + np.exp(-base.opacity[idx, 0].astype(np.float64)))
    full = alpha if weights is None else np.minimum(alpha / np.maximum(np.asarray(weights)[idx], 1.0e-3), 1.0)
    size = np.exp(base.scale[idx]).mean(axis=1).astype(np.float64)
    spacing = median_spacing(pts)
    radius, core_radius = link_factor * spacing, core_factor * spacing
    if radius <= 0.0:
        keep[idx] = True
        return keep

    min_alpha = min(TRIM_SOLID_ALPHA, 0.5 * float(np.median(full)))
    solid = (full >= min_alpha) & (size <= TRIM_SOLID_SCALE * float(np.median(size)))
    if solid.sum() < 4 * min_degree + 4:
        solid[:] = True
    body = solid.copy()
    for _ in range(10):  # k-core: peel gaussians with too few neighbours until stable
        sub = np.flatnonzero(body)
        if len(sub) <= min_degree:
            break
        degree = cKDTree(pts[sub]).query_ball_point(pts[sub], core_radius, return_length=True) - 1
        weak = degree < min_degree
        if not weak.any():
            break
        body[sub[weak]] = False
    if body.sum() < min_degree + 1:
        body = solid.copy()

    sub = np.flatnonzero(body)
    labels = _components(pts[sub], radius)
    weight = np.bincount(labels, weights=full[sub])
    good = weight >= TRIM_KEEP_FRACTION * weight.max()
    body[sub[~good[labels]]] = False

    sub = np.flatnonzero(body)
    axes = principal_frame(pts[sub])
    centre = np.median(pts[sub], axis=0)
    ref = (pts[sub] - centre) @ axes.T
    mid = np.median(ref, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(ref - mid), axis=0), 0.25 * radius)  # thin sheets must not clamp to nothing
    inside = (np.abs((pts - centre) @ axes.T - mid) <= mad_k * spread).all(axis=1)
    body &= inside
    sub = np.flatnonzero(body)
    near_body = np.zeros(len(pts), dtype=bool)
    if len(sub):
        dist, _ = cKDTree(pts[sub]).query(pts, k=1, workers=-1)
        near_body = dist <= radius
    final = (body | (~solid & near_body)) & inside
    keep[idx[final]] = True
    return keep


# --------------------------------------------------------------------------------------------
# Merging slots that are the same physical part
# --------------------------------------------------------------------------------------------

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def axis_line_gap(pivot_a, axis_a, pivot_b, axis_b, near) -> float:
    """Distance between two hinge lines measured around the point ``near`` (the closed part)."""
    near = np.asarray(near, dtype=np.float64)
    points = []
    for pivot, axis in ((pivot_a, axis_a), (pivot_b, axis_b)):
        pivot, axis = np.asarray(pivot, dtype=np.float64), _unit(axis)
        points.append(pivot + ((near - pivot) @ axis) * axis)
    return float(np.linalg.norm(points[0] - points[1]))


def base_overlap(base_a: np.ndarray, base_b: np.ndarray, radius: float) -> float:
    """Fraction of the smaller base set that has a neighbour of the other set within ``radius``."""
    small, large = (base_a, base_b) if len(base_a) <= len(base_b) else (base_b, base_a)
    if len(small) == 0:
        return 0.0
    dist, _ = cKDTree(large).query(small, k=1, workers=-1)
    return float((dist <= radius).mean())


def joints_compatible(joint_a: dict | None, joint_b: dict | None, base_a: np.ndarray, base_b: np.ndarray,
                      axis_deg: float = MERGE_AXIS_DEG, line_dist: float = MERGE_LINE_DIST,
                      min_overlap: float = MERGE_MIN_OVERLAP) -> bool:
    """Are two fitted joints the same physical joint of the same part?

    Same kind, axes within ``axis_deg`` with the same sign (so the open direction agrees), for hinges the
    axis lines within ``line_dist`` of each other around the closed part, and the closed poses overlapping.
    """
    if joint_a is None or joint_b is None or joint_a["kind"] != joint_b["kind"]:
        return False
    axis_a, axis_b = _unit(joint_a["axis"]), _unit(joint_b["axis"])
    if np.degrees(np.arccos(np.clip(axis_a @ axis_b, -1.0, 1.0))) > axis_deg:
        return False
    if joint_a["kind"] == "hinge":
        near = np.median(np.concatenate([base_a, base_b]), axis=0)
        if axis_line_gap(joint_a["pivot"], axis_a, joint_b["pivot"], axis_b, near) > line_dist:
            return False
    spacing = max(median_spacing(base_a), median_spacing(base_b))
    return base_overlap(base_a, base_b, 4.0 * spacing) >= min_overlap


def dedupe_against(new: Splats, existing: Splats, tol: float = MERGE_DEDUP) -> np.ndarray:
    """Mask of the ``new`` gaussians that have no ``existing`` gaussian closer than ``tol``."""
    if len(new) == 0 or len(existing) == 0:
        return np.ones(len(new), dtype=bool)
    dist, _ = cKDTree(existing.positions.astype(np.float64)).query(new.positions.astype(np.float64), k=1, workers=-1)
    return dist >= tol


def union_splats(first: Splats | None, second: Splats | None, tol: float = MERGE_DEDUP) -> Splats | None:
    """``first`` plus the gaussians of ``second`` that duplicate none of them (either may be None)."""
    if first is None or len(first) == 0:
        return second
    if second is None or len(second) == 0:
        return first
    return Splats.concat([first, second.subset(dedupe_against(second, first, tol))])


PART_FILE_KEYS = ("base", "switched", "part", "reveal", "vanish", "partReveal")


def merge_two(a: dict, b: dict, scene_dir: Path) -> None:
    """Fold part ``b`` into part ``a`` (both estimated): write the unions under a's file names, delete b's."""
    sets_a = {k: read_ply(scene_dir / a["files"][k]) if a["files"].get(k) else None for k in PART_FILE_KEYS}
    sets_b = {k: read_ply(scene_dir / b["files"][k]) if b["files"].get(k) else None for k in PART_FILE_KEYS}
    merged = {k: union_splats(sets_a[k], sets_b[k]) for k in PART_FILE_KEYS}
    if merged["vanish"] is not None and merged["part"] is not None:  # part in one slot means not vanishing
        merged["vanish"] = merged["vanish"].subset(dedupe_against(merged["vanish"], merged["part"]))
        if len(merged["vanish"]) == 0:
            merged["vanish"] = None
    joint_a, joint_b = a["joint"], b["joint"]
    best = joint_a if joint_a["fit"]["inlierRatio"] >= joint_b["fit"]["inlierRatio"] else joint_b
    count_a, count_b = len(sets_a["part"]), len(sets_b["part"])
    joint = dict(best)
    joint["open"] = joint["max"] = max(joint_a["open"], joint_b["open"])
    joint["min"] = min(joint_a.get("min", 0), joint_b.get("min", 0))
    rmse = (joint_a["fit"]["rmse"] * count_a + joint_b["fit"]["rmse"] * count_b) / max(count_a + count_b, 1)
    joint["fit"] = {
        "rmse": round(rmse, 5),
        "inlierRatio": round(min(len(merged["part"]) / max(len(merged["base"]), 1), 1.0), 4),
        "screw": round(max(joint_a["fit"]["screw"], joint_b["fit"]["screw"]), 5),
    }
    for key, splats in merged.items():
        name = a["files"].get(key) or f"{a['id']}_{key}.ply"
        if splats is None:
            if key == "partReveal":
                a["files"].pop(key, None)
            else:
                a["files"][key] = None
            (scene_dir / name).unlink(missing_ok=True)
        else:
            write_ply(scene_dir / name, splats)
            a["files"][key] = name
    for key in PART_FILE_KEYS:
        if b["files"].get(key) and b["files"][key] != a["files"].get(key):
            (scene_dir / b["files"][key]).unlink(missing_ok=True)
    for part in (a, b):  # the union replaces a's base / switched: the row flags of the soft export no longer line up
        membership_path(scene_dir, part["id"]).unlink(missing_ok=True)
    a["joint"] = joint
    a["merged"] = list(a.get("merged", [])) + [b["id"]] + list(b.get("merged", []))


def merge_parts(scene_dir: Path, scene: dict, only: set[str] | None = None) -> list[str]:
    """Merge compatible parts of ``scene`` in place (and on disk); returns one summary line per merge.

    Slots may be the same physical part opened in separate episodes (a door opened twice): they share
    the hinge but are separate slots. ``only`` restricts the candidates to those part ids.
    """
    lines = []
    parts = scene["parts"]
    merged_one = True
    while merged_one:
        merged_one = False
        for i, a in enumerate(parts):
            for b in parts[i + 1:]:
                if a.get("joint") is None or b.get("joint") is None:
                    continue
                if only is not None and (a["id"] not in only or b["id"] not in only):
                    continue
                base_a = read_ply(scene_dir / a["files"]["base"]).positions.astype(np.float64)
                base_b = read_ply(scene_dir / b["files"]["base"]).positions.astype(np.float64)
                if not joints_compatible(a["joint"], b["joint"], base_a, base_b):
                    continue
                merge_two(a, b, scene_dir)
                parts.remove(b)
                lines.append(f"merged {b['id']} into {a['id']} ({a.get('name', '')}): open={a['joint']['open']}")
                merged_one = True
                break
            if merged_one:
                break
    return lines


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


def load_membership(scene_dir: Path, slot_id: str, n_base: int, n_switched: int) -> dict | None:
    """The soft-export sidecar of a slot (row flags / multipliers), or None when absent or out of step with the PLYs."""
    path = membership_path(scene_dir, slot_id)
    if not path.exists():
        return None
    with np.load(path) as data:
        member = {key: data[key] for key in data.files}
    if len(member.get("base_fit", ())) != n_base or len(member.get("switched_fit", ())) != n_switched:
        return None
    return member


def process_part(scene_dir: Path, part: dict, inlier_dist: float | None, seed: int = 0, up=None,
                 trim: bool = True, fuse: bool = True, fuse_reach: float = FUSE_REACH_SPACINGS,
                 fuse_dup: float = FUSE_DUP_SPACINGS, fuse_sh: bool = True, inner: str = "auto") -> str:
    """Estimate one part, write its PLYs and update ``part`` in place. Returns a summary line."""
    files = part["files"]
    base = read_ply(scene_dir / files["base"])
    switched = read_ply(scene_dir / files["switched"])
    slot = part["id"]
    member = load_membership(scene_dir, slot, len(base), len(switched))
    fit_b = member["base_fit"].astype(bool) if member else np.ones(len(base), dtype=bool)
    fit_s = member["switched_fit"].astype(bool) if member else np.ones(len(switched), dtype=bool)
    weights = member["base_mult"] if member else None

    # the joint comes from the confident gaussians only; the part is built from all candidates
    base_fit, switched_fit = base.subset(fit_b), switched.subset(fit_s)
    fit = fit_part(base_fit, switched_fit, seed, up)
    if inlier_dist is None:
        inlier_dist = 2.5 * max(median_spacing(base_fit.positions), median_spacing(switched_fit.positions))
    split = split_inliers(base, switched, fit.rot, fit.trans, inlier_dist)
    part_mask, reveal_mask = clean_split(split)
    registered = int(part_mask.sum())
    if trim:  # floaters that registered by chance leave the part and fade out as it opens
        part_mask = trim_body(base, part_mask, weights=weights)
    reported = split if member is None else split_inliers(base_fit, switched_fit, fit.rot, fit.trans, inlier_dist)

    if inner not in INNER_MODES:
        raise ValueError(f"inner must be one of {INNER_MODES}, got {inner!r}")
    part_base = base.subset(part_mask)
    near = inner_splats = Splats.empty(part_base.f_rest.shape[1])
    envelope = np.zeros(len(switched), dtype=bool)
    if fuse and len(part_base) > 3:
        fusion = fuse_open_pose(part_base, switched, fit.rot, fit.trans, median_spacing(part_base.positions),
                                reach=fuse_reach, dup=fuse_dup, sh=fuse_sh)
        near, inner_splats, envelope = fusion.near, fusion.inner, fusion.envelope
        if inner == "auto":  # an opaque skin hides the inner face when closed; a translucent one (hood, glass) does not
            alpha = 1.0 / (1.0 + np.exp(-base.opacity[:, 0].astype(np.float64)))  # of the whole closed sample
            alpha = alpha if weights is None else alpha / np.maximum(np.asarray(weights), 1.0e-3)
            inner = "part" if float(np.median(alpha)) >= OPAQUE_ALPHA else "reveal"
        reveal_mask = ~envelope  # what is not the door is revealed when it opens
        if inner == "reveal":  # the door's inner face waits at the open pose until the viewer can carry it
            reveal_mask[fusion.inner_index] = True
        if int(reveal_mask.sum()) < max(MIN_SET, 0.01 * len(reveal_mask)):
            reveal_mask[:] = False
    attached = inner_splats if inner == "attach" and len(inner_splats) else None
    body = [part_base, near] + ([inner_splats] if inner == "part" else [])

    outputs = {
        "part": Splats.concat([s for s in body if len(s)]),
        "vanish": base.subset(~part_mask) if (~part_mask).any() else None,
        "reveal": switched.subset(reveal_mask) if reveal_mask.any() else None,
        "partReveal": attached,
    }
    for key, splats in outputs.items():
        name = f"{slot}_{key}.ply"
        if splats is None:
            if key == "partReveal":
                files.pop(key, None)  # optional file: only present when the part carries open-only content
            else:
                files[key] = None
            (scene_dir / name).unlink(missing_ok=True)
        else:
            write_ply(scene_dir / name, splats)
            files[key] = name

    part_centre = base.positions[part_mask].astype(np.float64).mean(axis=0)
    pivot = fit.pivot
    if fit.kind == "hinge":  # any point of the hinge line will do: report the one nearest the part
        pivot = pivot + ((part_centre - pivot) @ fit.axis) * fit.axis
    joint = {
        "kind": fit.kind,
        "axis": _round(fit.axis),
        "pivot": _round(pivot),
        "min": 0,
        "max": round(fit.open, 4),
        "open": round(fit.open, 4),
        **joint_defaults(fit.kind, base.positions),
        "fit": {"rmse": round(reported.rmse, 5), "inlierRatio": round(reported.inlier_ratio, 4),
                "screw": round(fit.screw, 5)},
    }
    part["joint"] = joint
    part["stats"] = {
        "base": len(base), "baseConfident": int(fit_b.sum()), "switched": len(switched),
        "registered": registered, "part": len(part_base), "fused": len(near) + (len(inner_splats) if inner == "part" else 0),
        "inner": len(inner_splats), "innerMode": inner,
        "doorEnvelope": int(envelope.sum()), "reveal": int(reveal_mask.sum()), "vanish": int((~part_mask).sum()),
    }
    unit = "deg" if fit.kind == "hinge" else "m"
    return (
        f"{slot} ({part.get('name', '')}): {fit.kind} axis={joint['axis']} pivot={joint['pivot']} "
        f"open={joint['open']}{unit} rmse={reported.rmse * 1000:.1f}mm inliers={reported.inlier_ratio:.2f} "
        f"base={len(base)} registered={registered} part={len(part_base)} (trimmed {registered - len(part_base)}) "
        f"fused={len(near)} inner={len(inner_splats)} ({inner}) vanish={int((~part_mask).sum())} reveal={int(reveal_mask.sum())}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir", type=Path, help="export directory holding scene.json")
    parser.add_argument("--slots", type=int, nargs="*", default=None, help="1-based slot numbers (default: all)")
    parser.add_argument("--inlier-dist", default="auto", help="'auto' (2.5x median spacing) or a distance in metres")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trim", action=argparse.BooleanOptionalAction, default=True,
                        help="trim each part to its coherent body, the rest fades out as vanish (default on)")
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True,
                        help="merge slots that turn out to be the same hinge / slider of the same part (default on)")
    parser.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True,
                        help="add the open-pose door (switched gaussians mapped back to the closed pose) to the part "
                             "(default on)")
    parser.add_argument("--fuse-reach", type=float, default=FUSE_REACH_SPACINGS,
                        help="switched gaussians within this many median spacings of the closed part belong to the "
                             "door (default %(default)s, about the door thickness)")
    parser.add_argument("--fuse-dup", type=float, default=FUSE_DUP_SPACINGS,
                        help="fused gaussians this many spacings from a similar closed-pose gaussian are duplicates "
                             "(default %(default)s)")
    parser.add_argument("--inner", choices=INNER_MODES, default="auto",
                        help="where the door's open-pose-only content (inner face, edge, window frame) goes: 'part' = "
                             "into the moving part (always drawn: fine behind an opaque skin, shows through a translucent "
                             "one), 'reveal' = waits at the open pose and fades in (ghosts while the part moves), "
                             "'attach' = slotNN_partReveal.ply, moves with the part and fades in with it (files.partReveal; "
                             "needs viewer support), 'auto' (default) = 'part' for opaque parts, 'reveal' for translucent ones")
    parser.add_argument("--fuse-sh", action=argparse.BooleanOptionalAction, default=True,
                        help="rotate the view-dependent colour of fused gaussians (off: they keep only the DC colour)")
    args = parser.parse_args(argv)
    inlier = None if args.inlier_dist == "auto" else float(args.inlier_dist)

    scene_path = args.scene_dir / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    wanted = None if args.slots is None else {f"slot{s:02d}" for s in args.slots}
    done = set()
    for part in scene["parts"]:
        if wanted is not None and part["id"] not in wanted:
            continue
        print(process_part(args.scene_dir, part, inlier, args.seed, scene.get("up"), args.trim,
                           fuse=args.fuse, fuse_reach=args.fuse_reach, fuse_dup=args.fuse_dup, fuse_sh=args.fuse_sh,
                           inner=args.inner),
              flush=True)
        done.add(part["id"])
    if args.merge:
        for line in merge_parts(args.scene_dir, scene, done):
            print(line, flush=True)
    scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
