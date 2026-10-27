# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recruit the static gaussians that lie on a moving panel's closed skin into the part (numpy / scipy only).

The state labels of a real scan are patchy: part of a door's, hood's or hatch's paint is labelled *static* (it looks
the same in the closed and the open state of the scan, e.g. because the same patch was visible all the time), so
when the panel swings away those gaussians stay behind as a flat, translucent ghost of the closed panel. The
viewer draws static + part with fixed roles, so the only cure is to give those gaussians to the part.

``select_skin`` decides which static gaussians are the panel's skin. The part it is given is a *template*: the part's
closed skin plus the panel as the scan saw it in the OPEN pose, mapped back (``slotNN_panel.ply``, made by
estimate_joint). At the open pose the panel hangs in the air, away from the bodywork, so its extent (outline, both
faces) is far cleaner there than in the closed pose, where it is interleaved with the car; the mapped-back panel
gives the footprint and the shell. (``recruit_scene(use_panel=True)``, off by default: on the car it over-recruited
the cowl and the windscreen base at the hinge, which then swung up with the hood as a dark smear.) A candidate must

1. lie in the part's closed-pose *footprint*: the part's gaussians projected onto its thin PCA plane are rasterised,
   holes smaller than ``close`` are filled (the part's own sampling is sparse) and the result grows by ``edge``
   (a few cm: the labelled part never reaches exactly to the edge of the panel); bodywork around the panel is
   outside of it;
2. sit on the surface: its distance along the plane normal to the part's local surface (a height field fitted to the
   nearest part gaussians) is at most ``height`` x the part's own thickness (robust spread of the part about that
   surface), but never less than ``min_height``;
3. face the same way (the flat axis of the gaussian within ``angle`` degrees of the local surface normal; round
   gaussians pass) and have a colour that continues the part's (not black rubber, not glass against paint).

``recruit_scene`` applies it to every part of an export directory: the recruited rows are removed from
``static.ply`` and appended to the part's ``slotNN_part.ply``, and kept in ``slotNN_skin.ply`` so that a re-run of the
whole pipeline (which re-derives the part from base / switched) restores the original static set first and gets the
same result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .ply_io import Splats, read_ply, write_ply

# lengths are metres of the real scene (scaled by metresPerUnit); the rest are factors / degrees
EDGE_M = 0.03  # the footprint grows this far beyond the part's own gaussians
CLOSE_M = 0.25  # gaps in the part's sampling up to twice this wide are filled along both plane axes, enclosed holes of any size too
HEIGHT_FACTOR = 2.5  # max distance to the surface in thicknesses of the part
POLY_DEGREE = 3  # the surface is a polynomial height field over the part's plane (a hood curves; gaps are bridged)
MIN_HEIGHT_M = 0.015
MAX_HEIGHT_M = 0.08
ANGLE_DEG = 55.0
COLOUR = 0.55
CELL_M = 0.01
NEIGHBOURS = 8


@dataclass
class RecruitParams:
    edge: float = EDGE_M
    close: float = CLOSE_M
    height: float = HEIGHT_FACTOR
    min_height: float = MIN_HEIGHT_M
    max_height: float = MAX_HEIGHT_M
    angle: float = ANGLE_DEG
    colour: float = COLOUR
    cell: float = CELL_M
    degree: int = POLY_DEGREE
    metres_per_unit: float = 1.0

    def units(self, metres: float) -> float:
        return metres / self.metres_per_unit


@dataclass
class Skin:
    mask: np.ndarray  # bool over the static candidates
    score: np.ndarray  # |height above the surface| / allowed height (lower = more clearly on the skin), inf outside
    thickness: float
    footprint_cells: int
    reason: np.ndarray | None = None  # per candidate: 0 outside footprint, 1 too high/low, 2 colour, 3 facing, 4 recruited


def _frame(points: np.ndarray):
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    return centre, vt


def _poly_terms(uv: np.ndarray, degree: int) -> np.ndarray:
    u, v = uv[:, 0], uv[:, 1]
    return np.stack([u**i * v**j for i in range(degree + 1) for j in range(degree + 1 - i)], axis=1)


def fit_surface(uv: np.ndarray, w: np.ndarray, degree: int = POLY_DEGREE, rounds: int = 3):
    """Robust polynomial height field w(u, v) of the part: least squares, refitted on the best 80 % a few times.

    Returns (function of (u, v) -> w, residual spread (1.4826 x MAD) of the part about it).
    """
    scale = max(float(np.ptp(uv[:, :2], axis=0).max()), 1.0e-9)
    norm = lambda q: (q - uv.mean(axis=0)) / scale
    terms = _poly_terms(norm(uv), degree)
    use = np.ones(len(w), dtype=bool)
    coef = np.zeros(terms.shape[1])
    for _ in range(rounds):
        if use.sum() < terms.shape[1] + 2:
            break
        coef = np.linalg.lstsq(terms[use], w[use], rcond=None)[0]
        resid = np.abs(w - terms @ coef)
        use = resid <= np.quantile(resid, 0.8)
    resid = w - terms @ coef
    spread = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    return (lambda q: _poly_terms(norm(q), degree) @ coef), spread


def _height_field(uv: np.ndarray, w: np.ndarray, query_uv: np.ndarray, k: int, skip_self: bool = False) -> np.ndarray:
    """Surface height at ``query_uv``: inverse-distance mean of the heights of the k nearest part gaussians (in the plane)."""
    k_eff = min(k + (1 if skip_self else 0), len(uv))
    dist, idx = cKDTree(uv).query(query_uv, k=k_eff)
    dist, idx = dist.reshape(len(query_uv), k_eff), idx.reshape(len(query_uv), k_eff)
    if skip_self:
        dist, idx = dist[:, 1:], idx[:, 1:]
    weight = 1.0 / (dist + 1.0e-6)
    return (weight * w[idx]).sum(axis=1) / weight.sum(axis=1)


def _normals_of(rot_quats: np.ndarray, log_scale: np.ndarray):
    """(flat axis of every gaussian, roundness = smallest / largest scale)."""
    from .fuse import quaternions_to_matrices

    mats = quaternions_to_matrices(rot_quats)
    smallest = np.argmin(log_scale, axis=1)
    axis = mats[np.arange(len(mats)), :, smallest]
    scale = np.exp(log_scale.astype(np.float64))
    return axis, scale.min(axis=1) / np.maximum(scale.max(axis=1), 1.0e-12)


def select_skin(part: Splats, candidates: Splats, params: RecruitParams | None = None,
                skin_rows: int | None = None) -> Skin:
    """Which of the ``candidates`` (static gaussians) are the skin of ``part`` (closed pose)? See the module docstring.

    ``skin_rows``: only the first rows of ``part`` are the outer skin (the rest, e.g. a fused inner face, would
    thicken the surface fit); default all.
    """
    if skin_rows is not None:
        part = part.subset(np.arange(min(skin_rows, len(part))))
    params = params or RecruitParams()
    n = len(candidates)
    nothing = Skin(np.zeros(n, dtype=bool), np.full(n, np.inf), 0.0, 0)
    if n == 0 or len(part) < 12:
        return nothing
    pts = part.positions.astype(np.float64)
    cand = candidates.positions.astype(np.float64)
    centre, vt = _frame(pts)
    puv = ((pts - centre) @ vt.T)
    cuv = ((cand - centre) @ vt.T)
    k = min(NEIGHBOURS, len(pts) - 1)

    # the surface of the part and its thickness (the spread of the part about it)
    surface, thickness = fit_surface(puv[:, :2], puv[:, 2], params.degree)
    allowed = float(np.clip(params.height * thickness, params.units(params.min_height), params.units(params.max_height)))

    # footprint raster in the plane
    cell = params.units(params.cell)
    lo = puv[:, :2].min(axis=0) - params.units(params.close + params.edge) - 2 * cell
    shape = np.ceil((puv[:, :2].max(axis=0) + params.units(params.close + params.edge) + 2 * cell - lo) / cell).astype(int) + 1
    grid = np.zeros(shape, dtype=bool)
    ij = np.floor((puv[:, :2] - lo) / cell).astype(int)
    grid[ij[:, 0], ij[:, 1]] = True

    def disk(radius_cells: int) -> np.ndarray:
        r = max(int(round(radius_cells)), 1)
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return x * x + y * y <= r * r

    # fill gaps of the part's sampling along both plane axes (line closings: no rounded corners at the panel's free edges)
    run = int(round(2 * params.units(params.close) / cell)) + 1
    pad = run + 2
    padded = np.pad(grid, pad)
    grid = (ndimage.binary_closing(padded, structure=np.ones((run, 1), dtype=bool))
            | ndimage.binary_closing(padded, structure=np.ones((1, run), dtype=bool)))[pad:-pad, pad:-pad]
    core = ndimage.binary_fill_holes(grid)
    grid = ndimage.binary_dilation(core, structure=disk(params.units(params.edge) / cell))

    cij = np.floor((cuv[:, :2] - lo) / cell).astype(int)
    inside = (cij >= 0).all(axis=1) & (cij[:, 0] < shape[0]) & (cij[:, 1] < shape[1])
    in_core = np.zeros(n, dtype=bool)
    in_core[inside] = core[cij[inside, 0], cij[inside, 1]]
    inside[inside] = grid[cij[inside, 0], cij[inside, 1]]
    # cheap bound before the neighbour searches
    inside &= np.abs(cuv[:, 2] - np.median(puv[:, 2])) < 10 * allowed + 4 * np.abs(puv[:, 2] - np.median(puv[:, 2])).max()
    sel = np.flatnonzero(inside)
    if len(sel) == 0:
        return Skin(nothing.mask, nothing.score, thickness, int(grid.sum()))

    height = surface(cuv[sel, :2])
    above = np.abs(cuv[sel, 2] - height)
    ok = above <= allowed
    reason = np.zeros(n, dtype=np.int8)
    reason[sel] = 1

    # orientation and colour continuity against the nearest part gaussians
    kk = min(6, len(pts))
    dist3, idx3 = cKDTree(pts).query(cand[sel], k=kk)
    dist3, idx3 = dist3.reshape(len(sel), kk), idx3.reshape(len(sel), kk)
    # the closest colour among the nearest template gaussians (a two-faced panel template mixes paint and underside)
    colour = np.linalg.norm(np.clip(candidates.rgb[sel], 0, 1)[:, None, :] - np.clip(part.rgb, 0, 1)[idx3], axis=2).min(axis=1)
    good_h = ok.copy()
    # (Waiving the colour / orientation gates well inside the footprint, where a stripe or decal differs in colour,
    # was tried on the car: it also pulled the dark cowl and windscreen base at the hood's hinge into the part.)
    deep = np.zeros(len(sel), dtype=bool)
    ok &= colour <= params.colour
    reason[sel[good_h & ~ok]] = 2
    c_axis, roundness = _normals_of(candidates.rot[sel], candidates.scale[sel])
    # the local surface normal: from the gradient of the fitted height field (nearest-neighbour planes are far too noisy)
    eps = 0.02 * max(float(np.ptp(puv[:, :2], axis=0).max()), 1.0e-9)
    q = cuv[sel, :2]
    gu = (surface(q + [eps, 0.0]) - surface(q - [eps, 0.0])) / (2 * eps)
    gv = (surface(q + [0.0, eps]) - surface(q - [0.0, eps])) / (2 * eps)
    local = np.stack([-gu, -gv, np.ones_like(gu)], axis=1)
    local /= np.linalg.norm(local, axis=1, keepdims=True)
    normals = local @ vt
    facing = np.abs(np.einsum("ij,ij->i", c_axis, normals)) >= np.cos(np.deg2rad(params.angle))
    good_c = ok.copy()
    ok &= facing | (roundness > 0.6) | deep
    reason[sel[good_c & ~ok]] = 3
    reason[sel[ok]] = 4

    mask = np.zeros(n, dtype=bool)
    mask[sel[ok]] = True
    score = np.full(n, np.inf)
    score[sel] = above / allowed
    score[~mask] = np.inf
    return Skin(mask, score, thickness, int(grid.sum()), reason)


# --------------------------------------------------------------------------------------------
# Whole exports
# --------------------------------------------------------------------------------------------

def skin_path(scene_dir: Path, slot_id: str) -> Path:
    return Path(scene_dir) / f"{slot_id}_skin.ply"


def restore_static(scene_dir: Path, scene: dict, slot_ids: set[str] | None = None) -> int:
    """Put the recruited rows of ``slot_ids`` (default: all) back into static.ply and delete their skin files.

    Returns the number of rows restored. After this static.ply is the importer's original static set again
    (up to row order, which does not matter), so recruitment can be re-derived.
    """
    scene_dir = Path(scene_dir)
    files = sorted(scene_dir.glob("slot*_skin.ply"))
    files = [f for f in files if slot_ids is None or f.name.split("_")[0] in slot_ids]
    if not files:
        return 0
    static_path = scene_dir / scene["static"]
    back = [read_ply(f) for f in files]
    write_ply(static_path, Splats.concat([read_ply(static_path)] + back))
    for f in files:
        f.unlink()
    return sum(len(b) for b in back)


def recruit_scene(scene_dir: Path, scene: dict, slot_ids: set[str] | None = None, params: RecruitParams | None = None,
                  verbose: bool = True, use_panel: bool = False) -> list[str]:
    """Recruit skin for every estimated part (or ``slot_ids``): rewrites static.ply, the part files and stats."""
    scene_dir = Path(scene_dir)
    params = params or RecruitParams(metres_per_unit=float(scene.get("metresPerUnit", 1.0)))
    parts = [p for p in scene["parts"] if p.get("joint") and p["files"].get("part") and (slot_ids is None or p["id"] in slot_ids)]
    for p in parts:  # a part file that still carries the skin of an earlier run: strip it before re-deriving
        stats, sidecar = p.get("stats", {}), skin_path(scene_dir, p["id"])
        if "skin" in stats and sidecar.exists():
            body = read_ply(scene_dir / p["files"]["part"])
            keep = len(body) - len(read_ply(sidecar))
            write_ply(scene_dir / p["files"]["part"], body.subset(np.arange(max(keep, 0))))
    restore_static(scene_dir, scene, slot_ids)
    if not parts:
        return []
    static_path = scene_dir / scene["static"]
    static = read_ply(static_path)
    bodies = {p["id"]: read_ply(scene_dir / p["files"]["part"]) for p in parts}
    def template_skin(p):
        body = bodies[p["id"]]
        return body.subset(np.arange(min(p.get("stats", {}).get("skinRows", len(body)), len(body))))

    def template(p):  # the closed skin of the part plus the open panel mapped back: its exact outline and both faces
        body = bodies[p["id"]].subset(np.arange(min(p.get("stats", {}).get("skinRows", len(bodies[p["id"]])), len(bodies[p["id"]]))))
        panel = scene_dir / f"{p['id']}_panel.ply"
        return Splats.concat([body, read_ply(panel)]) if panel.exists() else body

    results = {p["id"]: select_skin(template(p) if use_panel else template_skin(p), static, params) for p in parts}
    # a gaussian goes to the part it sits most clearly on
    scores = np.stack([results[p["id"]].score for p in parts])
    owner = scores.argmin(axis=0)
    lines, taken = [], np.zeros(len(static), dtype=bool)
    for i, p in enumerate(parts):
        res = results[p["id"]]
        mine = res.mask & (owner == i) & np.isfinite(scores[i])
        taken |= mine
        skin = static.subset(mine)
        stats = p.setdefault("stats", {})
        stats["skin"] = int(mine.sum())
        stats["skinThickness"] = round(res.thickness, 5)
        stats["partBeforeSkin"] = len(bodies[p["id"]])
        if len(skin):
            write_ply(skin_path(scene_dir, p["id"]), skin)
            write_ply(scene_dir / p["files"]["part"], Splats.concat([bodies[p["id"]], skin]))
        lines.append(f"{p['id']}: recruited {int(mine.sum())} static gaussians into the part "
                     f"(thickness {res.thickness:.4f}, {res.footprint_cells} footprint cells)")
    if taken.any():
        write_ply(static_path, static.subset(~taken))
    if verbose:
        for line in lines:
            print(line, flush=True)
    return lines


def ghost_mass(part: Splats, static: Splats, metres: float = 0.04, metres_per_unit: float = 1.0,
               skin_rows: int | None = None) -> float:
    """Summed opacity of the static gaussians within ``metres`` of the part's closed surface (inside its footprint).

    What is left of the closed panel when the part has moved away: a ghost. Colour and orientation are not tested.
    """
    probe = RecruitParams(min_height=metres, max_height=metres, colour=1.0e9, angle=90.0, metres_per_unit=metres_per_unit)
    skin = select_skin(part, static, probe, skin_rows)
    alpha = 1.0 / (1.0 + np.exp(-static.opacity[:, 0].astype(np.float64)))
    return float(alpha[skin.mask].sum())
