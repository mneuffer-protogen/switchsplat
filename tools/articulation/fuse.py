# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fusing the open-pose sample of a moving part into its closed-pose gaussians (numpy only).

A scanned door exists twice: as the base-state gaussians (closed) and as the switched-state gaussians (open).
The joint registers the two samplings onto each other, so after the inverse rigid transform the open-pose
gaussians are a second, independent sample of the same rigid body. They cover what the closed scan could not
see (the inner face, the door edge and the window frame, visible only when the door is open) and fill holes of
the closed sampling. Without this the open-pose door (never registered one-to-one, the two samples differ) stays
in the non-moving "reveal" set and fades in at the open pose while the real door is still swinging: a ghost.

``fuse_open_pose`` selects the switched gaussians that belong to the door (they lie within the part's thickness of
a closed-pose part gaussian once mapped back), moves them (position, orientation and the view-dependent colour,
``transform_splats``), drops the ones that merely repeat a closed-pose gaussian and splits the rest into ``near``
(the same surface, always safe to draw) and ``inner`` (hidden when the door is closed, unless the skin is
translucent: a hood shows its dark underside through it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import apply, quaternions_from_matrices
from .ply_io import Splats

# Real spherical-harmonics constants of the 3DGS renderer (same as threedgrut/export/sh_rotation.py, kept here
# so that the tools stay numpy-only).
_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
       1.445305721320277, -0.5900435899266435)


# --------------------------------------------------------------------------------------------
# Rotating gaussians
# --------------------------------------------------------------------------------------------

def sh_basis(degree: int, dirs: np.ndarray) -> np.ndarray:
    """The 3DGS real-SH basis [P, (degree+1)^2] at the direction vectors ``dirs`` [P, 3]."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    cols = [np.full_like(x, _C0)]
    if degree >= 1:
        cols += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        cols += [_C2[0] * xy, _C2[1] * yz, _C2[2] * (2.0 * zz - xx - yy), _C2[3] * xz, _C2[4] * (xx - yy)]
    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        cols += [_C3[0] * y * (3.0 * xx - yy), _C3[1] * x * y * z, _C3[2] * y * (4.0 * zz - xx - yy),
                 _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy), _C3[4] * x * (4.0 * zz - xx - yy),
                 _C3[5] * z * (xx - yy), _C3[6] * x * (xx - 3.0 * yy)]
    return np.stack(cols, axis=-1)


def sh_band_matrices(rot: np.ndarray, max_degree: int) -> dict[int, np.ndarray]:
    """Per-band matrices D_l with ``c_l' = D_l c_l`` for the SH of an object turned by ``rot``: f'(d) = f(R^T d).

    Derived numerically from the basis above (sample 2l+1 directions, solve), like the training code does.
    """
    rot = np.asarray(rot, dtype=np.float64)
    out = {}
    for degree in range(1, max_degree + 1):
        n = 2 * degree + 1
        rng = np.random.default_rng(20260620 + n)
        dirs = rng.standard_normal((n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        lo = degree * degree
        basis = sh_basis(degree, dirs)[:, lo:lo + n]
        rotated = sh_basis(degree, dirs @ rot)[:, lo:lo + n]  # Y(R^T d): the row d @ R is R^T d
        out[degree] = np.linalg.solve(basis, rotated)
    return out


def rotate_f_rest(f_rest: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Rotate channel-major SH coefficients [N, 3 * M] (M = 3, 8 or 15; bands 1..3) by ``rot``."""
    f_rest = np.asarray(f_rest)
    n, width = f_rest.shape
    if width == 0:
        return f_rest.copy()
    m = width // 3
    degree = int(round(np.sqrt(m + 1))) - 1
    if 3 * ((degree + 1) ** 2 - 1) != width:
        raise ValueError(f"f_rest has {width} columns, which is no SH degree 1..3 layout")
    coeffs = f_rest.reshape(n, 3, m).astype(np.float64)
    out = np.empty_like(coeffs)
    blocks = sh_band_matrices(rot, degree)
    offset = 0
    for band in range(1, degree + 1):
        w = 2 * band + 1
        out[:, :, offset:offset + w] = np.einsum("ij,ncj->nci", blocks[band], coeffs[:, :, offset:offset + w])
        offset += w
    return out.reshape(n, width).astype(f_rest.dtype)


def quaternions_to_matrices(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) quaternions [N, 4] (need not be unit) -> rotation matrices [N, 3, 3]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def transform_splats(splats: Splats, rot: np.ndarray, trans: np.ndarray) -> Splats:
    """The gaussians moved rigidly by ``p' = rot p + trans``: centres, orientations and view-dependent colour.

    Scales, opacity and the DC colour are rotation invariant. The SH bands 1..3 are rotated with the object.
    """
    rot = np.asarray(rot, dtype=np.float64)
    if len(splats) == 0 or (np.allclose(rot, np.eye(3)) and not np.any(trans)):
        return splats
    positions = apply(rot, np.asarray(trans, dtype=np.float64), splats.positions.astype(np.float64))
    if np.allclose(rot, np.eye(3)):
        return Splats(positions, splats.f_dc, splats.f_rest, splats.opacity, splats.scale, splats.rot)
    mats = rot[None] @ quaternions_to_matrices(splats.rot)
    return Splats(positions, splats.f_dc, rotate_f_rest(splats.f_rest, rot), splats.opacity, splats.scale,
                  quaternions_from_matrices(mats))


def invert_rigid(rot: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(rot, dtype=np.float64)
    return rot.T, -rot.T @ np.asarray(trans, dtype=np.float64)


# --------------------------------------------------------------------------------------------
# Duplicate suppression
# --------------------------------------------------------------------------------------------

DUP_COLOUR = 0.35  # rgb distance up to which two gaussians at the same place count as the same sample
DUP_LOG_SCALE = float(np.log(2.5))  # ... and mean |log scale ratio|


def similar(a: Splats, b: Splats) -> np.ndarray:
    """Elementwise (row i of ``a`` against row i of ``b``): same colour and footprint, so duplicates."""
    colour = np.linalg.norm(np.clip(a.rgb, 0, 1) - np.clip(b.rgb, 0, 1), axis=1)
    size = np.abs(a.scale - b.scale).mean(axis=1)
    return (colour <= DUP_COLOUR) & (size <= DUP_LOG_SCALE)


def duplicate_mask(candidates: Splats, existing: Splats, radius: float) -> np.ndarray:
    """True for ``candidates`` rows that repeat an ``existing`` gaussian: within ``radius``, similar colour and scale."""
    dup = np.zeros(len(candidates), dtype=bool)
    if len(candidates) == 0 or len(existing) == 0 or radius <= 0.0:
        return dup
    tree = cKDTree(existing.positions.astype(np.float64))
    neighbours = tree.query_ball_point(candidates.positions.astype(np.float64), radius)
    for i, near in enumerate(neighbours):
        if not near:
            continue
        probe = candidates.subset(np.full(len(near), i))
        dup[i] = bool(similar(probe, existing.subset(np.asarray(near))).any())
    return dup


def thin_self_duplicates(splats: Splats, radius: float) -> np.ndarray:
    """Mask of rows to keep when near-identical rows of ``splats`` are collapsed (most opaque first).

    Greedy: a row is dropped when an already kept row lies within ``radius`` and is similar. Guards against
    the open-pose sample doubling the density where it overlaps itself.
    """
    n = len(splats)
    keep = np.ones(n, dtype=bool)
    if n < 2 or radius <= 0.0:
        return keep
    order = np.argsort(-splats.opacity[:, 0], kind="stable")
    pairs = cKDTree(splats.positions.astype(np.float64)).query_ball_point(splats.positions.astype(np.float64), radius)
    accepted = np.zeros(n, dtype=bool)
    for i in order:
        near = [j for j in pairs[i] if j != i and accepted[j]]
        if near and similar(splats.subset(np.full(len(near), i)), splats.subset(np.asarray(near))).any():
            keep[i] = False
        else:
            accepted[i] = True
    return keep


# --------------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------------

@dataclass
class Fusion:
    """Result of ``fuse_open_pose``; every Splats is in the CLOSED pose, ``*_index`` rows refer to ``switched``."""

    near: Splats  # same surface as the closed part and looks like it: safe to draw whenever the part is drawn
    inner: Splats  # the door's other open-pose content (inner face, edge, window frame): hidden when closed
    inner_index: np.ndarray  # switched rows of ``inner``
    envelope: np.ndarray  # bool [Ns]: switched rows in the door's volume (near, inner or duplicate)
    duplicates: int  # envelope rows dropped as duplicates of a closed-pose gaussian (or of each other)
    extra: Splats | None = None  # open-pose gaussians of the part's footprint beyond ``reach`` (glass inside a frame, ...)
    extra_index: np.ndarray | None = None  # switched rows of ``extra``
    panel: Splats | None = None  # the whole open panel (every envelope row, duplicates included) in the CLOSED pose
    components: tuple[int, int] = (0, 0)  # (rows kept, rows dropped) by the connectivity pruning of the open panel


def part_thickness(points: np.ndarray) -> float:
    """5-95 % extent of the part along its thinnest principal axis (a flat sheet: its thickness incl. curvature)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        return 0.0
    _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    w = (points - points.mean(axis=0)) @ vt[2]
    return float(np.percentile(w, 95) - np.percentile(w, 5))


MIN_REACH_SPACINGS = 4.0
NEAR_SPACINGS = 3.0  # an open-pose gaussian this close to the closed part is on the same surface (joint fit error!)
NEAR_COLOUR = 0.4  # ... and a clearly different colour than its closed neighbours would change the closed look
NEAR_NEIGHBOURS = 4  # the colour of the closed surface there: mean of this many nearest part gaussians


def door_reach(points: np.ndarray, spacing: float, reach_spacings: float) -> float:
    """How far from a closed-pose part gaussian an open-pose one may be to count as the door's.

    ``reach_spacings`` x the median spacing, but never more than the part's own thickness (a thin hood must not
    recruit the engine bay that lies a few spacings below it) and never less than MIN_REACH_SPACINGS spacings.
    """
    return float(min(reach_spacings * spacing, max(part_thickness(points), MIN_REACH_SPACINGS * spacing)))


LINK_SPACINGS = 6.0  # open-pose panel gaussians this close (median spacings) belong to one connected piece
KEEP_PIECE = 0.05  # a further piece must carry this share of the main piece's rows


def _connected_panel(open_positions: np.ndarray, envelope: np.ndarray, spacing: float):
    """Keep the connected pieces of the open panel candidates (in the OPEN pose, where the panel hangs in the air).

    At the open pose the panel has swung away from the body, so what truly belongs to it is one (or a few) connected
    sheets; candidates that only fall into the closed-pose volume by accident (interior far behind, scattered bits)
    form small separate pieces and are dropped. Returns (new envelope, number of rows dropped).
    """
    from scipy.sparse.csgraph import connected_components

    idx = np.flatnonzero(envelope)
    if len(idx) < 8 or spacing <= 0.0:
        return envelope, 0
    pts = open_positions[idx].astype(np.float64)
    tree = cKDTree(pts)
    labels = connected_components(tree.sparse_distance_matrix(tree, LINK_SPACINGS * spacing, output_type="coo_matrix"),
                                  directed=False)[1]
    size = np.bincount(labels)
    good = size >= KEEP_PIECE * size.max()
    out = np.zeros_like(envelope)
    out[idx[good[labels]]] = True
    return out, int(len(idx) - good[labels].sum())


def fuse_open_pose(part: Splats, switched: Splats, rot: np.ndarray, trans: np.ndarray, spacing: float,
                   reach: float = 10.0, dup: float = 0.5, sh: bool = True, near: float = NEAR_SPACINGS,
                   footprint=None) -> Fusion:
    """Move the door's open-pose gaussians into the closed pose and sort them into near / inner.

    ``rot``/``trans`` is the base -> switched transform. A switched gaussian belongs to the door when, mapped
    back into the closed pose, it lies within ``door_reach`` of a part gaussian (``reach`` x ``spacing``, at most
    the part's thickness): interior content that the open door reveals maps to where the door is not and stays
    reveal. Of the door's gaussians the ones within ``dup`` x ``spacing`` of a closed-pose part gaussian of
    similar colour and scale are dropped (the same sample twice) and the rest are collapsed against each other.
    The survivors split by what they do to the closed look:

    * ``near``: within ``near`` x ``spacing`` of a part gaussian and of the colour of the closed surface there
      (the mean of its nearest part gaussians). A second sample of the surface that is already drawn: it fills the
      gaps between the closed gaussians and moves with them, and cannot change how the closed part looks.
    * ``inner``: everything else of the door (the inner face, the edge, the window frame). Under a translucent
      skin (a hood) it shows through, so it must not be drawn when the door is closed.

    ``sh=False`` zeroes the view-dependent colour of the fused gaussians instead of rotating it.
    """
    empty = Splats.empty(part.f_rest.shape[1])
    none = np.zeros(0, dtype=np.int64)
    inv_rot, inv_trans = invert_rigid(rot, trans)
    moved = transform_splats(switched, inv_rot, inv_trans)
    if len(part) == 0 or len(moved) == 0 or spacing <= 0.0:
        return Fusion(empty, empty, none, np.zeros(len(switched), dtype=bool), 0)
    tree = cKDTree(part.positions.astype(np.float64))
    dist, _ = tree.query(moved.positions.astype(np.float64), k=1, workers=-1)
    envelope = dist <= door_reach(part.positions, spacing, reach)
    footprint_only = np.zeros(len(moved), dtype=bool)
    if footprint is not None:  # the whole footprint of the part's surface (``recruit.RecruitParams``), glass inside a frame included
        from .recruit import select_skin

        footprint_only = select_skin(part, moved, footprint).mask & ~envelope
        envelope = envelope | footprint_only
    envelope, dropped = _connected_panel(switched.positions, envelope, spacing)
    footprint_only &= envelope
    env_idx = np.flatnonzero(envelope)
    inside = moved.subset(env_idx)
    kept = ~duplicate_mask(inside, part, dup * spacing)
    kept_idx = env_idx[kept]
    survivors = inside.subset(kept)
    thin = thin_self_duplicates(survivors, dup * spacing)
    final_idx = kept_idx[thin]
    fused = survivors.subset(thin)
    if not sh and fused.f_rest.shape[1]:
        fused = Splats(fused.positions, fused.f_dc, np.zeros_like(fused.f_rest), fused.opacity, fused.scale, fused.rot)
    k = min(NEAR_NEIGHBOURS, len(part))
    d_near, nearest = tree.query(fused.positions.astype(np.float64), k=k, workers=-1)
    d_near, nearest = d_near.reshape(len(fused), k), nearest.reshape(len(fused), k)
    surface = np.clip(part.rgb, 0, 1)[nearest].mean(axis=1)  # the closed surface colour around the gaussian
    colour = np.linalg.norm(np.clip(fused.rgb, 0, 1) - surface, axis=1)
    is_near = (d_near[:, 0] <= near * spacing) & (colour <= NEAR_COLOUR)
    only = footprint_only[final_idx]  # beyond the reach: the part's own open-pose copy, never drawn when closed
    return Fusion(fused.subset(is_near & ~only), fused.subset(~is_near & ~only), final_idx[~is_near & ~only], envelope,
                  len(env_idx) - len(final_idx), fused.subset(only), final_idx[only], moved.subset(envelope),
                  (int(envelope.sum()), dropped))
