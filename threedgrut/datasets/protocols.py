# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import torch


@dataclass
class Batch:
    rays_ori: torch.Tensor  # [B, H, W, 3] ray origins in arbitrary space
    rays_dir: torch.Tensor  # [B, H, W, 3] ray directions in arbitrary space
    T_to_world: (
        torch.Tensor
    )  # [B, 4, 4] transformation matrix from the ray space to the world space
    T_to_world_end: Optional[torch.Tensor] = None  # [B, 4, 4] END pose for rolling shutter
    rays_in_world_space: bool = False  # True if rays are already in world space
    rgb_gt: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    intrinsics: Optional[list] = None
    intrinsics_OpenCVPinholeCameraModelParameters: Optional[dict] = None
    intrinsics_OpenCVFisheyeCameraModelParameters: Optional[dict] = None
    intrinsics_FThetaCameraModelParameters: Optional[dict] = None
    # Camera/frame indices for post-processing
    camera_idx: int | torch.Tensor | list[int] = -1  # 0-based camera index
    frame_idx: int | torch.Tensor | list[int] = -1  # 0-based frame index (global across split)
    # Optional raw numeric ID parsed from a frame_XXXX image filename.
    time_idx: Optional[torch.Tensor] = None  # [B]
    # Optional explicit SwitchSplat object-state code for non-temporal evaluations.
    state_code: Optional[torch.Tensor] = None  # [B, K]
    # Pixel coordinates for post-processing
    pixel_coords: Optional[torch.Tensor] = None  # [B, H, W, 2] (x, y) with +0.5 center offset
    # Exposure prior from EXIF metadata (mean-normalized log2 exposure [1], None if unavailable)
    exposure: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.T_to_world.ndim != 3 or self.T_to_world.shape[1:] != (4, 4):
            raise ValueError(f"T_to_world must have shape [B,4,4], got {tuple(self.T_to_world.shape)}")
        batch_size = self.T_to_world.shape[0]
        if batch_size < 1:
            raise ValueError("Batch must contain at least one camera")
        if self.rays_ori.ndim != 4 or self.rays_ori.shape[0] != batch_size or self.rays_ori.shape[-1] != 3:
            raise ValueError(f"rays_ori must have shape [B,H,W,3], got {tuple(self.rays_ori.shape)}")
        if self.rays_dir.ndim != 4 or self.rays_dir.shape[0] != batch_size or self.rays_dir.shape[-1] != 3:
            raise ValueError(f"rays_dir must have shape [B,H,W,3], got {tuple(self.rays_dir.shape)}")
        if self.rays_ori.shape[1:3] != self.rays_dir.shape[1:3]:
            raise ValueError(
                f"All rays in a logical batch must share one H x W resolution; "
                f"origins are {tuple(self.rays_ori.shape[1:3])}, directions are {tuple(self.rays_dir.shape[1:3])}"
            )
        height, width = self.rays_dir.shape[1:3]
        if self.T_to_world_end is not None and self.T_to_world_end.shape != self.T_to_world.shape:
            raise ValueError(
                f"T_to_world_end must have shape {tuple(self.T_to_world.shape)}, "
                f"got {tuple(self.T_to_world_end.shape)}"
            )
        if self.rgb_gt is not None:
            if self.rgb_gt.shape != (batch_size, height, width, 3):
                raise ValueError(
                    f"rgb_gt must have shape [B,H,W,3] matching the rays, got {tuple(self.rgb_gt.shape)}"
                )
        if self.mask is not None:
            if self.mask.shape != (batch_size, height, width, 1):
                raise ValueError(
                    f"mask must have shape [B,H,W,1] matching the rays, got {tuple(self.mask.shape)}"
                )
        if self.pixel_coords is not None and self.pixel_coords.shape != (batch_size, height, width, 2):
            raise ValueError(
                f"pixel_coords must have shape [B,H,W,2] matching the rays, got {tuple(self.pixel_coords.shape)}"
            )
        if self.time_idx is not None and self.time_idx.shape != (batch_size,):
            raise ValueError(f"time_idx must have shape [B], got {tuple(self.time_idx.shape)}")
        if self.state_code is not None and (self.state_code.ndim != 2 or self.state_code.shape[0] != batch_size):
            raise ValueError(f"state_code must have shape [B,K], got {tuple(self.state_code.shape)}")
        if self.intrinsics:
            if not isinstance(self.intrinsics, list) or len(self.intrinsics) != 4:
                raise ValueError("intrinsics must be a list [fx, fy, cx, cy]")

    @property
    def batch_size(self) -> int:
        return int(self.T_to_world.shape[0])

    def masked(self, image: torch.Tensor) -> torch.Tensor:
        return image if self.mask is None else image * self.mask


_CAMERA_VECTOR_KEYS = {
    "resolution",
    "principal_point",
    "focal_length",
    "radial_coeffs",
    "tangential_coeffs",
    "thin_prism_coeffs",
    "pixeldist_to_angle_poly",
    "angle_to_pixeldist_poly",
    "linear_cde",
}
_CAMERA_SCALAR_KEYS = {"shutter_type", "reference_poly", "max_angle"}


def _slice_intrinsic_component(value: Any, index: int, batch_size: int) -> Any:
    """Slice a possibly per-camera scalar while leaving shared scalars untouched."""
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def _slice_camera_parameter(key: str, value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, dict):
        return {
            nested_key: _slice_camera_parameter(nested_key, nested, index, batch_size)
            for nested_key, nested in value.items()
        }

    if isinstance(value, torch.Tensor):
        if key in _CAMERA_SCALAR_KEYS:
            return value[index] if value.ndim > 0 and value.shape[0] == batch_size else value
        if key in _CAMERA_VECTOR_KEYS:
            return value[index] if value.ndim > 1 and value.shape[0] == batch_size else value
        return value[index] if value.ndim > 0 and value.shape[0] == batch_size else value

    # Dataset conversion represents per-camera strings/enums as a list. Numeric
    # vectors that are shared by all cameras remain ordinary literal lists.
    if key in _CAMERA_SCALAR_KEYS and isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def _slice_index(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        selected = value[index]
        return selected.item() if selected.numel() == 1 else selected
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def slice_batch(batch: Batch, index: int) -> Batch:
    """Return one camera from a logical batch while preserving image batch axes.

    Image/ray/pose tensors retain a leading dimension of one. Per-camera
    intrinsics and nested camera parameters are reduced to their original
    single-camera representation; shared scalar metadata is reused unchanged.
    """
    batch_size = batch.batch_size
    if not 0 <= index < batch_size:
        raise IndexError(f"Batch index {index} is outside [0, {batch_size})")

    def slice_leading(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index : index + 1]
        return value

    intrinsics = None
    if batch.intrinsics is not None:
        intrinsics = [_slice_intrinsic_component(value, index, batch_size) for value in batch.intrinsics]

    def slice_parameters(parameters: Optional[dict]) -> Optional[dict]:
        if parameters is None:
            return None
        return {
            key: _slice_camera_parameter(key, value, index, batch_size)
            for key, value in parameters.items()
        }

    return Batch(
        rays_ori=slice_leading(batch.rays_ori),
        rays_dir=slice_leading(batch.rays_dir),
        T_to_world=slice_leading(batch.T_to_world),
        T_to_world_end=slice_leading(batch.T_to_world_end),
        rays_in_world_space=batch.rays_in_world_space,
        rgb_gt=slice_leading(batch.rgb_gt),
        mask=slice_leading(batch.mask),
        intrinsics=intrinsics,
        intrinsics_OpenCVPinholeCameraModelParameters=slice_parameters(
            batch.intrinsics_OpenCVPinholeCameraModelParameters
        ),
        intrinsics_OpenCVFisheyeCameraModelParameters=slice_parameters(
            batch.intrinsics_OpenCVFisheyeCameraModelParameters
        ),
        intrinsics_FThetaCameraModelParameters=slice_parameters(batch.intrinsics_FThetaCameraModelParameters),
        camera_idx=_slice_index(batch.camera_idx, index, batch_size),
        frame_idx=_slice_index(batch.frame_idx, index, batch_size),
        time_idx=slice_leading(batch.time_idx),
        state_code=slice_leading(batch.state_code),
        pixel_coords=slice_leading(batch.pixel_coords),
        exposure=slice_leading(batch.exposure),
    )


class BoundedMultiViewDataset(Protocol):
    """Defines the basic functionality required from all datasets that can be used with the 3dgrut Trainer."""

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the bounding box of the scene as a tuple of vec3 (min,max)"""
        ...

    def get_scene_extent(self) -> float:
        """TODO"""
        ...

    def get_observer_points(self) -> np.ndarray:
        """TODO"""
        ...

    def get_poses(self) -> np.ndarray:
        """Get camera poses as 4x4 transformation matrices.

        Returns camera-to-world (C2W) transformation matrices using the 3DGRUT
        "right down front" camera coordinate system convention.

        Camera Coordinate System:
            - Right: +X axis points to the camera's right
            - Down: +Y axis points downward
            - Front: +Z axis points forward (viewing direction)

        Returns:
            np.ndarray: Camera poses with shape (N, 4, 4)
        """
        ...

    def get_gpu_batch_with_intrinsics(self, batch: dict) -> Batch:
        """Add the intrinsics to the batch and move data to GPU."""
        ...

    def get_camera_idx(self, frame_idx: int) -> int:
        """Return 0-based camera index for a given train split frame index."""
        ...

    def get_frames_per_camera(self) -> list[int]:
        """Return list of frame counts per camera.

        Returns a list where index i contains the number of frames captured
        by camera i. Derived values:
        - num_cameras = len(frames_per_camera)
        - num_frames = sum(frames_per_camera)
        """
        ...

    def __getitem__(self, index: int) -> dict: ...

    def __len__(self) -> int: ...


@runtime_checkable
class WorldTransformProvider(Protocol):
    """Optional dataset capability for mapping source geometry into dataset world space."""

    def get_world_normalization_transform(self) -> np.ndarray:
        """Return the 4x4 transform from source coordinates to dataset world coordinates."""
        ...


def get_dataset_world_transform(dataset: object) -> np.ndarray | None:
    """Return a validated, non-identity source-to-dataset transform when available."""
    if not isinstance(dataset, WorldTransformProvider):
        return None

    transform = np.asarray(dataset.get_world_normalization_transform())
    if transform.shape != (4, 4):
        raise ValueError(f"Dataset world transform must have shape (4, 4), got {transform.shape}.")
    if not np.issubdtype(transform.dtype, np.number) or np.iscomplexobj(transform):
        raise ValueError(f"Dataset world transform must be real-valued, got dtype {transform.dtype}.")
    if not np.all(np.isfinite(transform)):
        raise ValueError("Dataset world transform must contain only finite values.")
    if np.array_equal(transform, np.eye(4, dtype=transform.dtype)):
        return None
    return transform.copy()


@runtime_checkable
class DatasetVisualization(Protocol):
    """Defines the basic functionality required from all datasets that can be visualized in the GUI app."""

    def create_dataset_camera_visualization(self): ...
