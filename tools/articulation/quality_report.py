# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quality gate for the moving parts of an export: how much of each part does the scan actually carry?

    python -m tools.articulation.quality_report DIR [--reference OLD_DIR]

For every part of ``DIR/scene.json`` it prints the gaussian counts (and, with ``--reference``, the same for an
earlier export of the scene) and two coverage metrics measured in the part's closed pose:

* occupancy: the part's gaussian centres are projected onto its thin PCA plane; the share of the 3 cm grid cells
  inside the 5-95 % hull box (of the new and the reference part together, a common yardstick) that hold at least
  one centre. This is what is left to see when the door is mid-swing and only the moving gaussians are drawn.
* closed coverage: the closed render is static + part + vanish; each cell of the same box accumulates the opacity
  of the gaussians of that render whose centre lies in the cell and within the part's slab, clipped to 1. The
  mean over the box, the share of "hole" cells (coverage < 0.05) and the largest connected hole are reported.

* ghost: the summed opacity of the *static* gaussians within 4 cm of the part's closed surface (inside its footprint):
  what a moving panel leaves behind as a flat translucent copy of itself. Measured against the new part's own skin
  for both exports, so recruitment (which moves those gaussians into the part) shows as a drop.

The verdict per part is PASS when the new closed coverage is not lower than the reference and no bigger hole
appeared. The grid cell is 3 cm of real size (``metresPerUnit`` of the scene).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .ply_io import Splats, read_ply
from .recruit import ghost_mass

HOLE_COVERAGE = 0.05
SLAB_MARGIN = 0.05  # scene units beyond the part's 5-95 % thickness


@dataclass
class Frame:
    """The closed part's thin plane: centre, rows u v w, 5-95 % hull box in (u, v) and the slab in w."""

    centre: np.ndarray
    axes: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    slab: tuple[float, float]


def part_frame(points: np.ndarray) -> Frame:
    points = np.asarray(points, dtype=np.float64)
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    coords = (points - centre) @ vt.T
    lo, hi = np.percentile(coords[:, :2], 5, axis=0), np.percentile(coords[:, :2], 95, axis=0)
    w_lo, w_hi = np.percentile(coords[:, 2], [5, 95])
    return Frame(centre, vt, lo, hi, (float(w_lo) - SLAB_MARGIN, float(w_hi) + SLAB_MARGIN))


def project(frame: Frame, points: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - frame.centre) @ frame.axes.T


def grid_shape(frame: Frame, cell: float) -> tuple[int, int]:
    return tuple(int(max(np.ceil((frame.hi[i] - frame.lo[i]) / cell), 1)) for i in range(2))


def cells_of(frame: Frame, coords: np.ndarray, cell: float):
    """Integer (i, j) cell of every projected point and the mask of those inside the hull box."""
    ij = np.floor((coords[:, :2] - frame.lo) / cell).astype(int)
    nu, nv = grid_shape(frame, cell)
    inside = (ij >= 0).all(axis=1) & (ij[:, 0] < nu) & (ij[:, 1] < nv)
    return ij, inside


def occupancy(frame: Frame, points: np.ndarray, cell: float) -> float:
    """Fraction of hull-box cells holding at least one projected centre."""
    if len(points) == 0:
        return 0.0
    ij, inside = cells_of(frame, project(frame, points), cell)
    grid = np.zeros(grid_shape(frame, cell), dtype=bool)
    grid[ij[inside, 0], ij[inside, 1]] = True
    return float(grid.mean())


def coverage_grid(frame: Frame, splats: Splats, cell: float) -> np.ndarray:
    """Opacity accumulated per cell (clipped to 1) by the gaussians within the part's slab."""
    grid = np.zeros(grid_shape(frame, cell))
    if len(splats) == 0:
        return grid
    coords = project(frame, splats.positions)
    ij, inside = cells_of(frame, coords, cell)
    inside &= (coords[:, 2] >= frame.slab[0]) & (coords[:, 2] <= frame.slab[1])
    alpha = 1.0 / (1.0 + np.exp(-splats.opacity[:, 0].astype(np.float64)))
    np.add.at(grid, (ij[inside, 0], ij[inside, 1]), alpha[inside])
    return np.minimum(grid, 1.0)


def hole_stats(grid: np.ndarray) -> tuple[float, int]:
    """(share of hole cells, size in cells of the largest connected hole)."""
    holes = grid < HOLE_COVERAGE
    labels, count = ndimage.label(holes)
    largest = int(np.bincount(labels.ravel())[1:].max()) if count else 0
    return float(holes.mean()), largest


def _read(scene_dir: Path, name: str | None) -> Splats | None:
    return read_ply(scene_dir / name) if name and (scene_dir / name).exists() else None


def closed_render(scene_dir: Path, scene: dict, part: dict, static: Splats, region: np.ndarray) -> Splats:
    """Static + part + vanish (the closed state) restricted to the bounding sphere of ``region``."""
    centre, radius = region.mean(axis=0), 1.1 * float(np.linalg.norm(region - region.mean(axis=0), axis=1).max())
    pieces = []
    for splats in (static, _read(scene_dir, part["files"].get("part")), _read(scene_dir, part["files"].get("vanish"))):
        if splats is not None and len(splats):
            near = np.linalg.norm(splats.positions - centre, axis=1) <= radius
            pieces.append(splats.subset(near))
    return Splats.concat(pieces) if pieces else Splats.empty()


def analyse(scene_dir: Path, reference: Path | None, cell_metres: float = 0.03) -> list[dict]:
    scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    ref_scene = json.loads((reference / "scene.json").read_text(encoding="utf-8")) if reference else None
    cell = cell_metres / float(scene.get("metresPerUnit", 1.0))
    static = read_ply(scene_dir / scene["static"])
    ref_static = read_ply(reference / ref_scene["static"]) if reference else None
    rows = []
    for part in scene["parts"]:
        new_part = _read(scene_dir, part["files"].get("part"))
        if new_part is None:
            continue
        ref_part_entry = next((p for p in ref_scene["parts"] if p["id"] == part["id"]), None) if ref_scene else None
        old_part = _read(reference, ref_part_entry["files"].get("part")) if ref_part_entry else None
        both = new_part.positions if old_part is None else np.concatenate([old_part.positions, new_part.positions])
        frame = part_frame(both)
        row = {"id": part["id"], "name": part.get("name", ""), "stats": part.get("stats", {})}
        for tag, scene_path, entry, sc, stat in (("new", scene_dir, part, scene, static),
                                                 ("old", reference, ref_part_entry, ref_scene, ref_static)):
            if entry is None or scene_path is None:
                continue
            counts = {key: (len(s) if (s := _read(scene_path, entry["files"].get(key))) is not None else 0)
                      for key in ("base", "switched", "part", "reveal", "vanish")}
            p = _read(scene_path, entry["files"].get("part"))
            closed = closed_render(scene_path, sc, entry, stat, p.positions)
            grid = coverage_grid(frame, closed, cell)
            hole_share, hole_max = hole_stats(grid)
            row[tag] = {"counts": counts, "occupancy": occupancy(frame, p.positions, cell),
                        "closed_mean": float(grid.mean()), "hole_share": hole_share, "hole_max": hole_max,
                        "ghost": ghost_mass(new_part, stat, 0.04, float(sc.get("metresPerUnit", 1.0)),
                                            part.get("stats", {}).get("skinRows"))}
        rows.append(row)
    return rows


def verdict(row: dict) -> bool:
    if "old" not in row:
        return True
    new, old = row["new"], row["old"]
    return new["closed_mean"] >= old["closed_mean"] - 1.0e-3 and new["hole_max"] <= old["hole_max"]


def format_table(rows: list[dict]) -> str:
    head = ("slot", "base", "conf", "part", "fused", "inner", "reveal", "vanish", "occupancy", "closed cov", "holes",
            "max hole", "ghost", "gate")
    lines = ["  ".join(f"{h:>10}" for h in head)]

    def pair(new, old, fmt):
        return f"{fmt.format(old)}>{fmt.format(new)}" if old is not None else fmt.format(new)

    for row in rows:
        new, old = row["new"], row.get("old")
        stats = row["stats"]
        o = old["counts"] if old else {}
        cells = (
            row["id"],
            str(new["counts"]["base"]),
            str(stats.get("baseConfident", "")),
            pair(new["counts"]["part"], o.get("part"), "{}"),
            str(stats.get("fused", "")),
            f'{stats.get("inner", "")}/{stats.get("innerMode", "")}' if stats else "",
            pair(new["counts"]["reveal"], o.get("reveal"), "{}"),
            pair(new["counts"]["vanish"], o.get("vanish"), "{}"),
            pair(new["occupancy"], old["occupancy"] if old else None, "{:.2f}"),
            pair(new["closed_mean"], old["closed_mean"] if old else None, "{:.3f}"),
            pair(new["hole_share"], old["hole_share"] if old else None, "{:.3f}"),
            pair(new["hole_max"], old["hole_max"] if old else None, "{}"),
            pair(new["ghost"], old["ghost"] if old else None, "{:.0f}"),
            "PASS" if verdict(row) else "FAIL",
        )
        lines.append("  ".join(f"{c:>10}" for c in cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--reference", type=Path, default=None, help="an earlier export of the same scene to compare with")
    parser.add_argument("--cell", type=float, default=0.03, help="grid cell in metres (default 0.03)")
    args = parser.parse_args(argv)
    rows = analyse(args.scene_dir, args.reference, args.cell)
    print(format_table(rows))
    return 0 if all(verdict(r) for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
