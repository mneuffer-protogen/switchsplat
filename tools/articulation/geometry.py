# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small rigid-motion helpers shared by the articulation tools (numpy only)."""

from __future__ import annotations

import numpy as np


def rodrigues(axis: np.ndarray, angle: float | np.ndarray) -> np.ndarray:
    """Rotation matrix about a unit axis by ``angle`` radians (right-hand rule).

    ``angle`` may be an array of shape [A], giving [A, 3, 3].
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    angle = np.asarray(angle, dtype=np.float64)
    sin, cos = np.sin(angle)[..., None, None], np.cos(angle)[..., None, None]
    return np.eye(3) + sin * k + (1.0 - cos) * (k @ k)


def rotation_to_axis_angle(rot: np.ndarray) -> tuple[np.ndarray, float]:
    """Axis (unit) and angle in [0, pi] of a rotation matrix. The axis is arbitrary for the identity."""
    cos = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1.0e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if np.pi - angle < 1.0e-4:
        # Near 180 deg the antisymmetric part vanishes: take the axis from the symmetric part.
        sym = (rot + np.eye(3)) / 2.0
        axis = sym[:, np.argmax(np.diag(sym))]
        axis = axis / np.linalg.norm(axis)
        return axis, angle
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2.0 * np.sin(angle))
    return axis / np.linalg.norm(axis), angle


def hinge_transform(axis: np.ndarray, pivot: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of ``p' = R(axis, angle) (p - pivot) + pivot`` written as ``p' = R p + t``."""
    rot = rodrigues(axis, angle)
    pivot = np.asarray(pivot, dtype=np.float64)
    return rot, pivot - rot @ pivot


def apply(rot: np.ndarray, trans: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ rot.T + trans


def point_to_line_distance(point: np.ndarray, line_point: np.ndarray, line_axis: np.ndarray) -> float:
    axis = np.asarray(line_axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    delta = np.asarray(point, dtype=np.float64) - np.asarray(line_point, dtype=np.float64)
    return float(np.linalg.norm(delta - (delta @ axis) * axis))


def quaternions_from_matrices(rot: np.ndarray) -> np.ndarray:
    """(w, x, y, z) unit quaternions of rotation matrices [N, 3, 3] (vectorized, branch-free per row)."""
    m = rot
    n = len(m)
    q = np.empty((n, 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    # Four candidate formulas; pick the numerically best-conditioned one per row.
    cand = np.stack([tr, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], axis=1)
    best = np.argmax(cand, axis=1)
    for case in range(4):
        sel = best == case
        if not sel.any():
            continue
        r = m[sel]
        if case == 0:
            s = np.sqrt(1.0 + tr[sel]) * 2.0
            q[sel] = np.stack([s / 4, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                               (r[:, 1, 0] - r[:, 0, 1]) / s], axis=1)
        elif case == 1:
            s = np.sqrt(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]) * 2.0
            q[sel] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, s / 4, (r[:, 0, 1] + r[:, 1, 0]) / s,
                               (r[:, 0, 2] + r[:, 2, 0]) / s], axis=1)
        elif case == 2:
            s = np.sqrt(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]) * 2.0
            q[sel] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, s / 4,
                               (r[:, 1, 2] + r[:, 2, 1]) / s], axis=1)
        else:
            s = np.sqrt(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]) * 2.0
            q[sel] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                               (r[:, 1, 2] + r[:, 2, 1]) / s, s / 4], axis=1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)
