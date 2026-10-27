# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and write 3DGS PLY files (the layout of ``threedgrut/export/ply_exporter.py``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


def rgb_to_f_dc(rgb: np.ndarray) -> np.ndarray:
    """Degree-0 SH coefficients of a plain RGB colour in [0, 1]."""
    return ((np.asarray(rgb, dtype=np.float32) - 0.5) / SH_C0).astype(np.float32)


def f_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    return np.asarray(f_dc, dtype=np.float32) * SH_C0 + 0.5


@dataclass
class Splats:
    """A set of gaussians as numpy arrays in PLY (pre-activation) units.

    ``opacity`` is the sigmoid logit, ``scale`` the log scale, ``rot`` a (w, x, y, z) quaternion.
    ``f_rest`` is channel-major ([all R coeffs, all G, all B]) and may have zero columns.
    """

    positions: np.ndarray  # [N, 3]
    f_dc: np.ndarray  # [N, 3]
    f_rest: np.ndarray  # [N, 3 * M]
    opacity: np.ndarray  # [N, 1]
    scale: np.ndarray  # [N, 3]
    rot: np.ndarray  # [N, 4]

    def __post_init__(self) -> None:
        n = len(self.positions)
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float32).reshape(n, 3)
        self.f_dc = np.ascontiguousarray(self.f_dc, dtype=np.float32).reshape(n, 3)
        f_rest = np.asarray(self.f_rest, dtype=np.float32)
        self.f_rest = np.ascontiguousarray(f_rest if f_rest.ndim == 2 else f_rest.reshape(n, -1 if n else 0))
        self.opacity = np.ascontiguousarray(self.opacity, dtype=np.float32).reshape(n, 1)
        self.scale = np.ascontiguousarray(self.scale, dtype=np.float32).reshape(n, 3)
        self.rot = np.ascontiguousarray(self.rot, dtype=np.float32).reshape(n, 4)

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def rgb(self) -> np.ndarray:
        return f_dc_to_rgb(self.f_dc)

    def subset(self, mask_or_index: np.ndarray) -> "Splats":
        """Rows picked by a boolean mask or an integer index array."""
        idx = np.asarray(mask_or_index)
        return Splats(
            self.positions[idx], self.f_dc[idx], self.f_rest[idx], self.opacity[idx], self.scale[idx], self.rot[idx]
        )

    @staticmethod
    def empty(num_rest: int = 0) -> "Splats":
        return Splats(
            np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, num_rest)), np.zeros((0, 1)), np.zeros((0, 3)),
            np.zeros((0, 4)),
        )

    @staticmethod
    def concat(parts: list["Splats"]) -> "Splats":
        parts = [p for p in parts if len(p) > 0] or parts
        if not parts:
            return Splats.empty()
        widths = {p.f_rest.shape[1] for p in parts}
        if len(widths) != 1:
            raise ValueError(f"cannot concatenate splats with different f_rest widths: {sorted(widths)}")
        return Splats(
            *(np.concatenate([getattr(p, name) for p in parts], axis=0) for name in (
                "positions", "f_dc", "f_rest", "opacity", "scale", "rot"))
        )


def ply_property_names(num_rest: int) -> list[str]:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(num_rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    return names


def write_ply(path: str | Path, splats: Splats) -> None:
    """Write a binary little-endian 3DGS PLY (normals are written as (0, 0, 1), like the exporter)."""
    n = len(splats)
    normals = np.zeros((n, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    table = np.concatenate(
        [splats.positions, normals, splats.f_dc, splats.f_rest, splats.opacity, splats.scale, splats.rot],
        axis=1,
    ).astype("<f4")
    names = ply_property_names(splats.f_rest.shape[1])
    dtype = np.dtype([(name, "<f4") for name in names])
    elements = np.ascontiguousarray(table).view(dtype).reshape(n)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")], text=False, byte_order="<").write(str(path))


def read_ply(path: str | Path) -> Splats:
    """Read a 3DGS PLY; ``f_rest`` columns are sorted by their numeric suffix, normals are ignored."""
    vertex = PlyData.read(str(path))["vertex"].data
    names = vertex.dtype.names

    def column(name: str) -> np.ndarray:
        return np.asarray(vertex[name], dtype=np.float32)

    def stack(prefix: str, count: int) -> np.ndarray:
        return np.stack([column(f"{prefix}{i}") for i in range(count)], axis=1)

    num_rest = sum(1 for name in names if name.startswith("f_rest_"))
    n = len(vertex)
    return Splats(
        positions=np.stack([column("x"), column("y"), column("z")], axis=1),
        f_dc=stack("f_dc_", 3),
        f_rest=stack("f_rest_", num_rest) if num_rest else np.zeros((n, 0), dtype=np.float32),
        opacity=column("opacity")[:, None],
        scale=stack("scale_", 3),
        rot=stack("rot_", 4),
    )
