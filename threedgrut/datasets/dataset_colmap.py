# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import copy
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Optional

import ncore.sensors
import numpy as np
import torch
from ncore.data import (
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from PIL import Image
from torch.utils.data import Dataset

from threedgrut.utils.logger import logger

from .colmap_gsplat import normalize_world_space, scene_scale
from .protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from .utils import (
    compute_inscribed_radius,
    create_camera_visualization,
    create_pixel_coords,
    get_center_and_diag,
    get_worker_id,
    parse_frame_time_indices,
    pinhole_camera_rays,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_extrinsics_text,
    read_colmap_intrinsics_binary,
    read_colmap_intrinsics_text,
    read_colmap_points3D_binary,
    read_colmap_points3D_text,
    stack_camera_parameter_dicts,
)

_NOVEL_EVAL_RE = re.compile(r"novel_(view|combo|combination)_([01]+)_\d+")


def _novel_eval_metadata(path: str) -> tuple[str, str] | None:
    match = _NOVEL_EVAL_RE.search(os.path.basename(path))
    if match is None:
        return None
    kind, state_code = match.groups()
    return ("novel_combo" if kind != "view" else "novel_view", state_code)


def derive_state_code_mapping(
    slot_windows: list[list[float]] | np.ndarray,
    canonical_windows: list[list[float]] | np.ndarray,
) -> list[Optional[int]]:
    """Map ablated slots to canonical novel-evaluation state bits.

    Exact timeline matches inherit the corresponding canonical semantic bit.
    An added slot with no canonical interaction match maps to ``None``; novel
    evaluation treats that slot as state 0, its outside-window/neutral state.
    """

    def validated(windows, label: str) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float64)
        if values.size == 0:
            return values.reshape(0, 2)
        if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"{label} must be a finite array with shape [K,2]")
        if np.any(values[:, 0] > values[:, 1]):
            raise ValueError(f"{label} starts must not exceed their ends")
        return values

    slots = validated(slot_windows, "slot windows")
    canonical = validated(canonical_windows, "canonical windows")
    canonical_lookup: dict[tuple[float, float], int] = {}
    for index, window in enumerate(canonical):
        key = (float(window[0]), float(window[1]))
        if key in canonical_lookup:
            raise ValueError(f"canonical windows contain a duplicate interval: {key}")
        canonical_lookup[key] = index

    return [canonical_lookup.get((float(window[0]), float(window[1]))) for window in slots]


def remap_novel_state_code(state_code: str, mapping: Optional[list[Optional[int]]]) -> list[int]:
    """Apply a semantic slot mapping to one canonical filename state code."""

    source = [int(bit) for bit in state_code]
    if mapping is None:
        return source

    remapped = []
    for source_index in mapping:
        if source_index is None:
            remapped.append(0)
            continue
        source_index = int(source_index)
        if source_index < 0 or source_index >= len(source):
            raise ValueError(
                f"state-code mapping index {source_index} is outside canonical code '{state_code}'"
            )
        remapped.append(source[source_index])
    return remapped


def _get_relative_paths(path_dir: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(path_dir):
        for filename in filenames:
            paths.append(os.path.relpath(os.path.join(dirpath, filename), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    logger.info(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    for image_file in _get_relative_paths(image_dir):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(resized_dir, os.path.splitext(image_file)[0] + ".png")
        if os.path.isfile(resized_path):
            continue

        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            resized_size = (
                int(round(image.width / factor)),
                int(round(image.height / factor)),
            )
            image.resize(resized_size, Image.Resampling.BICUBIC).save(resized_path)

    return resized_dir


def _opencv_pinhole_intrinsics_from_colmap(intr_model: str, intr_params: np.ndarray, scaling_factor: float):
    """Convert COLMAP distorted pinhole parameters to OpenCVPinholeCameraModelParameters fields."""
    intr_params = np.asarray(intr_params, dtype=np.float32)
    radial_coeffs = np.zeros((6,), dtype=np.float32)
    tangential_coeffs = np.zeros((2,), dtype=np.float32)
    thin_prism_coeffs = np.zeros((4,), dtype=np.float32)

    if intr_model == "SIMPLE_RADIAL":
        focal_length = np.array([intr_params[0], intr_params[0]], dtype=np.float32) / scaling_factor
        principal_point = intr_params[1:3] / scaling_factor
        radial_coeffs[0] = intr_params[3]
    elif intr_model == "RADIAL":
        focal_length = np.array([intr_params[0], intr_params[0]], dtype=np.float32) / scaling_factor
        principal_point = intr_params[1:3] / scaling_factor
        radial_coeffs[:2] = intr_params[3:5]
    elif intr_model == "OPENCV":
        focal_length = intr_params[:2] / scaling_factor
        principal_point = intr_params[2:4] / scaling_factor
        radial_coeffs[:2] = intr_params[4:6]
        tangential_coeffs[:] = intr_params[6:8]
    elif intr_model == "FULL_OPENCV":
        focal_length = intr_params[:2] / scaling_factor
        principal_point = intr_params[2:4] / scaling_factor
        radial_coeffs[:] = intr_params[[4, 5, 8, 9, 10, 11]]
        tangential_coeffs[:] = intr_params[6:8]
    else:
        raise ValueError(f"Unsupported distorted pinhole camera model: {intr_model}")

    return focal_length, principal_point, radial_coeffs, tangential_coeffs, thin_prism_coeffs


def _test_frame_mask_from_file(
    dataset_path: str,
    test_frames_file: str,
    image_names: list[str],
    eval_split: Optional[str] = None,
) -> np.ndarray:
    test_frames_path = Path(test_frames_file)
    if not test_frames_path.is_absolute():
        test_frames_path = Path(dataset_path) / test_frames_path

    entries = [line.strip() for line in test_frames_path.read_text().splitlines() if line.strip()]
    normalized_entries = [str(PurePosixPath(entry)) for entry in entries]
    if not normalized_entries:
        raise ValueError(f"Test-frame file is empty: {test_frames_path}")
    if len(normalized_entries) != len(set(normalized_entries)):
        raise ValueError(f"Test-frame file contains duplicate paths: {test_frames_path}")

    normalized_names = [str(PurePosixPath(name)) for name in image_names]
    missing = sorted(set(normalized_entries) - set(normalized_names))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Test-frame file contains paths absent from COLMAP: {preview}")

    test_frames = {
        entry
        for entry in normalized_entries
        if eval_split is None or ((_novel_eval_metadata(entry) or ("test", ""))[0] == eval_split)
    }
    return np.asarray([name in test_frames for name in normalized_names], dtype=bool)


class ColmapDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
    def __init__(
        self,
        path,
        device="cuda",
        split="train",
        downsample_factor=1,
        test_split_interval=8,
        test_frame_suffix: Optional[str] = None,
        test_frames_file: Optional[str] = None,
        ray_jitter=None,
        exif_exposures: Optional[list[Optional[float]]] = None,
        camera_names: Optional[list[str]] = None,
        camera_ids: Optional[list[int]] = None,
        normalize_world_space: bool = False,
        gsplat_image_downscale: bool = False,
        preload_data: bool = False,
        state_code_mapping: Optional[list[Optional[int]]] = None,
        fullcircle=None,
    ):
        self.path = path
        self.device = device
        self.split = split
        self.downsample_factor = downsample_factor
        self.ray_jitter = ray_jitter
        self.test_split_interval = test_split_interval
        self.test_frame_suffix = test_frame_suffix
        self.test_frames_file = test_frames_file
        self._all_exif_exposures = exif_exposures  # Exposure values for all frames (pre-split)
        self.camera_names = list(camera_names) if camera_names is not None else None
        self.camera_ids = [int(camera_id) for camera_id in camera_ids] if camera_ids is not None else None
        self.normalize_world_space = bool(normalize_world_space)
        self.gsplat_image_downscale = gsplat_image_downscale
        self.preload_data = bool(preload_data)
        self._preloaded_data: Optional[list[tuple[torch.Tensor, Optional[torch.Tensor]]]] = None
        self.state_code_mapping = None if state_code_mapping is None else list(state_code_mapping)
        self.valid_mask_path = ""
        self.fullcircle_distractor_mask_root = ""
        self.distractor_mask_white_is_valid = False
        if fullcircle and (dilation := fullcircle.get("distractor_mask_dilation", None)) is not None:
            self.valid_mask_path = (
                fullcircle.fisheye_valid_mask if split == "train" else fullcircle.fisheye_valid_mask_tripod
            )
            suffix = "" if downsample_factor == 1 else f"_{downsample_factor}"
            self.fullcircle_distractor_mask_root = os.path.join(f"masks{suffix}", f"masks-{dilation}{suffix}")
            self.distractor_mask_white_is_valid = fullcircle.get("distractor_mask_white_is_valid", False)
        self._valid_mask_cache: dict[tuple[int, int], np.ndarray] = {}
        self.world_normalization_transform = np.eye(4, dtype=np.float32)

        # Worker-based GPU cache for multiprocessing compatibility
        self._worker_gpu_cache = {}

        # (Re)load intrinsics and extrinsics
        self.reload()

    def reload(self):
        # GPU cache of processed camera intrinsics - now per camera ID
        self.intrinsics = {}
        self.time_indices = None
        self.world_normalization_transform = np.eye(4, dtype=np.float32)

        # Get the scene data
        self.load_intrinsics_and_extrinsics()
        frame_indices_before_split = self._filter_cameras()

        # Build mapping from COLMAP camera_id to 0-based contiguous index
        # This is needed for post-processing which expects 0-based camera indices
        sorted_camera_ids = sorted(self.cam_intrinsics.keys())
        self._camera_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(sorted_camera_ids)}

        self.n_frames = len(self.cam_extrinsics)
        self.load_camera_data()
        if self.normalize_world_space:
            self._apply_world_space_normalization()
        indices = np.ones(self.n_frames, dtype=bool)
        split_name = "test" if self.split == "val" else self.split
        if split_name not in {"train", "test", "base_state", "novel_view", "novel_combo"}:
            raise ValueError(f"Unknown COLMAP split: {self.split}")

        test_frames_file = getattr(self, "test_frames_file", None)
        if test_frames_file:
            image_names = [extr.name for extr in self.cam_extrinsics]
            if split_name == "train":
                indices = ~_test_frame_mask_from_file(self.path, test_frames_file, image_names)
                indices &= self.source_eval_splits == "regular"
            else:
                manifest_split = "test" if split_name == "base_state" else split_name
                indices = _test_frame_mask_from_file(
                    self.path,
                    test_frames_file,
                    image_names,
                    manifest_split,
                )
                indices &= self.source_eval_splits == ("regular" if manifest_split == "test" else manifest_split)
        # A named suffix is the FullCircle dataset convention and takes
        # precedence over interval sampling.  This lets the README override
        # work without requiring users to also disable test_split_interval.
        elif self.test_frame_suffix:
            image_stems = [os.path.splitext(os.path.basename(path))[0] for path in self.image_paths]
            is_test_frame = np.asarray([stem.endswith(self.test_frame_suffix) for stem in image_stems])
            indices = ~is_test_frame if split_name == "train" else is_test_frame
        # If test_split_interval is set, every test_split_interval frame will be excluded from the training set.
        # If it is non-positive, all images are used for training and validation.
        elif self.test_split_interval > 0:
            frame_indices = np.arange(self.n_frames)
            if split_name == "train":
                indices = np.mod(frame_indices, self.test_split_interval) != 0
            else:
                indices = np.mod(frame_indices, self.test_split_interval) == 0

        if not test_frames_file:
            source_split = split_name if split_name.startswith("novel_") else "regular"
            indices &= self.source_eval_splits == source_split

        if split_name == "base_state":
            intervals_path = next(
                (
                    path
                    for name in ("interaction_times.txt", "transition_frames.txt")
                    if os.path.exists(path := os.path.join(self.path, name))
                ),
                None,
            )
            if intervals_path is None:
                logger.warning(f"Skipping base_state: no interaction times found in {self.path}")
                indices &= False
            else:
                intervals = np.atleast_2d(np.loadtxt(intervals_path, dtype=np.int64))
                if intervals.shape[1] == 4:
                    intervals = np.stack(
                        [(intervals[:, 0] + intervals[:, 1] + 1) // 2, (intervals[:, 2] + intervals[:, 3] + 1) // 2],
                        axis=1,
                    )
                if intervals.shape[1] != 2 or np.any(intervals[:, 0] > intervals[:, 1]):
                    raise ValueError(f"Invalid interaction intervals: {intervals_path}")
                for start, end in intervals:
                    indices &= (self.time_indices < start) | (self.time_indices > end)

        self.cam_extrinsics = [self.cam_extrinsics[i] for i in np.where(indices)[0]]
        self.poses = self.poses[indices].astype(np.float32)

        # numpy str array of image paths and mask paths
        self.image_paths = self.image_paths[indices]
        self.mask_paths = self.mask_paths[indices]
        self.fullcircle_distractor_mask_paths = self.fullcircle_distractor_mask_paths[indices]
        self.source_eval_splits = self.source_eval_splits[indices]
        self.state_code_strings = self.state_code_strings[indices]
        self.time_indices = self.time_indices[indices]
        if np.any(self.time_indices < 0):
            self.time_indices = None

        self.camera_centers = self.camera_centers[indices]
        if np.any(indices):
            self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()
        else:
            self.center = torch.zeros(3)
            self.length_scale = torch.tensor(0.0)
            self.scene_bbox = (self.center, self.center)

        # Apply split indices to EXIF exposures
        if self._all_exif_exposures is not None:
            frame_exif_exposures = [self._all_exif_exposures[i] for i in frame_indices_before_split]
            self.exif_exposures: Optional[list[Optional[float]]] = [
                frame_exif_exposures[i] for i in np.where(indices)[0]
            ]
        else:
            self.exif_exposures = None

        # Update the number of frames to only include the samples from the split
        self.n_frames = self.poses.shape[0]

        # Clear existing worker caches to force recreation with new intrinsics
        self._worker_gpu_cache.clear()

        self._preloaded_data = None
        if getattr(self, "preload_data", False):
            self._preload_decoded_data()

    def _load_points_for_world_normalization(self) -> np.ndarray:
        points_candidates = [
            (os.path.join(self.path, "sparse/0", "points3D.bin"), read_colmap_points3D_binary),
            (os.path.join(self.path, "sparse/0", "points3D.txt"), read_colmap_points3D_text),
            (os.path.join(self.path, "colmap", "points3D.txt"), read_colmap_points3D_text),
        ]
        for points_file, reader in points_candidates:
            if os.path.isfile(points_file):
                points, _, _ = reader(points_file)
                if points.shape[0] == 0:
                    raise ValueError(f"COLMAP point file is empty: {points_file}")
                return points.astype(np.float64)
        raise FileNotFoundError(f"Could not find COLMAP points3D file under {self.path}")

    def _apply_world_space_normalization(self) -> None:
        points = self._load_points_for_world_normalization()
        camtoworlds, _, transform = normalize_world_space(self.poses, points)

        self.poses = camtoworlds.astype(np.float32)
        self.camera_centers = self.poses[:, :3, 3].copy()
        self.cameras_extent = scene_scale(self.poses) * 1.1
        self.world_normalization_transform = transform.astype(np.float32)
        logger.info(
            "Applied GSplat-style COLMAP world normalization "
            f"(scene_extent={self.cameras_extent:.6f}, split={self.split})"
        )

    def load_intrinsics_and_extrinsics(self):
        try:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)

    def _camera_names_by_id(self) -> dict[int, str]:
        sorted_camera_ids = sorted(self.cam_intrinsics.keys())
        camera_id_to_idx = {camera_id: idx for idx, camera_id in enumerate(sorted_camera_ids)}
        names: dict[int, str] = {camera_id: f"camera_{camera_id_to_idx[camera_id]}" for camera_id in sorted_camera_ids}

        for extr in self.cam_extrinsics:
            parent_folder = os.path.dirname(extr.name)
            if parent_folder:
                names[extr.camera_id] = parent_folder

        return names

    def _filter_cameras(self) -> list[int]:
        selected_camera_ids = set(self.cam_intrinsics.keys())
        names_by_id = self._camera_names_by_id()

        if self.camera_ids is not None:
            requested_ids = set(self.camera_ids)
            missing_ids = sorted(requested_ids - selected_camera_ids)
            if missing_ids:
                available_ids = sorted(selected_camera_ids)
                raise ValueError(f"COLMAP camera_ids {missing_ids} not found. Available camera_ids: {available_ids}")
            selected_camera_ids &= requested_ids

        if self.camera_names is not None:
            name_to_ids: dict[str, set[int]] = {}
            for camera_id, camera_name in names_by_id.items():
                name_to_ids.setdefault(camera_name, set()).add(camera_id)

            requested_names = set(self.camera_names)
            missing_names = sorted(requested_names - set(name_to_ids))
            if missing_names:
                available_names = sorted(name_to_ids)
                raise ValueError(
                    f"COLMAP camera_names {missing_names} not found. Available camera_names: {available_names}"
                )

            selected_camera_ids &= set().union(*(name_to_ids[camera_name] for camera_name in requested_names))

        if not selected_camera_ids:
            raise ValueError("COLMAP camera selection is empty.")

        frame_indices = [
            frame_idx for frame_idx, extr in enumerate(self.cam_extrinsics) if extr.camera_id in selected_camera_ids
        ]
        if not frame_indices:
            selected_names = [names_by_id[camera_id] for camera_id in sorted(selected_camera_ids)]
            raise ValueError(f"COLMAP camera selection {selected_names} has no frames.")

        if self.camera_names is not None or self.camera_ids is not None:
            selected_names = [names_by_id[camera_id] for camera_id in sorted(selected_camera_ids)]
            logger.info(
                f"Using COLMAP cameras: {selected_names} "
                f"(camera_ids={sorted(selected_camera_ids)}, frames={len(frame_indices)})"
            )

        self.cam_extrinsics = [self.cam_extrinsics[i] for i in frame_indices]
        self.cam_intrinsics = {
            camera_id: intr for camera_id, intr in self.cam_intrinsics.items() if camera_id in selected_camera_ids
        }
        return frame_indices

    def get_images_folder(self):
        downsample_suffix = "" if self.downsample_factor == 1 else f"_{self.downsample_factor}"
        return f"images{downsample_suffix}"

    def load_camera_data(self):
        """
        Load the camera data and generate rays for each camera.
        This function is called on CPU for multiprocessing compatibility
        GPU tensors will be created per-worker as needed
        """
        self._camera_data_params = {}
        self._store_camera_params_cpu()

    def _store_camera_params_cpu(self):
        """Store camera parameters on CPU for multiprocessing compatibility."""

        image_dir = os.path.join(self.path, self.get_images_folder())
        colmap_to_image = {}
        if self.gsplat_image_downscale:
            colmap_image_dir = os.path.join(self.path, "images")
            for directory in [image_dir, colmap_image_dir]:
                if not os.path.exists(directory):
                    raise ValueError(f"Image folder {directory} does not exist.")

            colmap_files = sorted(_get_relative_paths(colmap_image_dir))
            image_files = sorted(_get_relative_paths(image_dir))
            if self.downsample_factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
                image_dir = _resize_image_folder(
                    colmap_image_dir,
                    image_dir + "_png",
                    factor=self.downsample_factor,
                )
                image_files = sorted(_get_relative_paths(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))

        def create_pinhole_camera(focalx, focaly, w, h, cx=None, cy=None):
            cx = cx if cx is not None else w / 2.0
            cy = cy if cy is not None else h / 2.0
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([w, h], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([cx, cy], dtype=np.float32),
                focal_length=np.array([focalx, focaly], dtype=np.float32),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            rays_o_cam, rays_d_cam = pinhole_camera_rays(u, v, focalx, focaly, w, h, self.ray_jitter, cx=cx, cy=cy)
            pixel_coords = create_pixel_coords(w, h)
            return (
                params.to_dict(),
                torch.tensor(rays_o_cam, dtype=torch.float32).reshape(out_shape),
                torch.tensor(rays_d_cam, dtype=torch.float32).reshape(out_shape),
                type(params).__name__,
                pixel_coords,
            )

        def create_opencv_pinhole_camera(
            focal_length,
            principal_point,
            w,
            h,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        ):
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([w, h], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.asarray(principal_point, dtype=np.float32),
                focal_length=np.asarray(focal_length, dtype=np.float32),
                radial_coeffs=np.asarray(radial_coeffs, dtype=np.float32),
                tangential_coeffs=np.asarray(tangential_coeffs, dtype=np.float32),
                thin_prism_coeffs=np.asarray(thin_prism_coeffs, dtype=np.float32),
            )
            camera_model = ncore.sensors.CameraModel.from_parameters(params, device="cpu", dtype=torch.float32)
            int_pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
            image_points = camera_model.pixels_to_image_points(int_pixel_coords)
            rays_d_cam = camera_model.image_points_to_camera_rays(image_points)
            rays_o_cam = torch.zeros_like(rays_d_cam)
            pixel_coords = create_pixel_coords(w, h)
            return (
                params.to_dict(),
                rays_o_cam.to(torch.float32).reshape(out_shape),
                rays_d_cam.to(torch.float32).reshape(out_shape),
                type(params).__name__,
                pixel_coords,
            )

        def create_fisheye_camera(params, w, h):
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            resolution = np.array([w, h]).astype(np.uint64)
            principal_point = params[2:4].astype(np.float32)
            focal_length = params[0:2].astype(np.float32)
            radial_coeffs = params[4:].astype(np.float32)
            # COLMAP OPENCV_FISHEYE images use an optical circle. The square
            # raster corners lie outside that circle and must not expand its FOV.
            max_radius_pixels = compute_inscribed_radius(resolution.astype(np.float64), principal_point)
            fov_angle_x = 2.0 * max_radius_pixels / focal_length[0]
            fov_angle_y = 2.0 * max_radius_pixels / focal_length[1]
            max_angle = np.max([fov_angle_x, fov_angle_y]) / 2.0

            params = OpenCVFisheyeCameraModelParameters(
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=radial_coeffs,
                resolution=resolution,
                max_angle=max_angle,
                shutter_type=ShutterType.GLOBAL,
            )
            camera_model = ncore.sensors.CameraModel.from_parameters(params, device="cpu", dtype=torch.float32)
            int_pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
            image_points = camera_model.pixels_to_image_points(int_pixel_coords)
            rays_d_cam = camera_model.image_points_to_camera_rays(image_points)
            rays_o_cam = torch.zeros_like(rays_d_cam)
            valid_mask = self._get_valid_mask(w, h)
            if valid_mask is not None:
                rays_d_cam[~torch.from_numpy(valid_mask.reshape(-1))] = float("nan")
            pixel_coords = create_pixel_coords(w, h)
            return (
                params.to_dict(),
                rays_o_cam.to(torch.float32).reshape(out_shape),
                rays_d_cam.to(torch.float32).reshape(out_shape),
                type(params).__name__,
                pixel_coords,
            )

        cam_id_to_image_name = {extr.camera_id: extr.name for extr in self.cam_extrinsics}

        for intr in self.cam_intrinsics.values():
            full_width = intr.width
            full_height = intr.height

            image_name = cam_id_to_image_name[intr.id]
            if self.gsplat_image_downscale:
                image_path = os.path.join(image_dir, colmap_to_image[image_name])
            else:
                image_name = (
                    os.path.join(os.path.split(image_name)[1], "")
                    if self.get_images_folder() in image_name
                    else image_name
                )
                image_path = os.path.join(self.path, self.get_images_folder(), image_name)

            try:
                # Load the image to get its actual dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
            except FileNotFoundError:
                logger.error(f"Image {image_path} not found. Cannot determine dimensions for intrinsic ID {intr.id}.")
                continue

            if self.gsplat_image_downscale:
                scaling_factor = self.downsample_factor
                image_scale = np.array(
                    [width / (full_width // scaling_factor), height / (full_height // scaling_factor)],
                    dtype=np.float32,
                )
            else:
                # Calculate scaling factor to match the image dimensions to the intrinsic dimensions
                scaling_factor = int(round(intr.height / height))
                expected_size = f"{full_width / scaling_factor}x{full_height / scaling_factor}"
                assert (
                    abs(full_width / scaling_factor - width) <= 1
                ), f"Scaled image dimension {expected_size} (factor {scaling_factor}x) does not match the actual image dimensions {width}x{height}"
                assert (
                    abs(full_height / scaling_factor - height) <= 1
                ), f"Scaled image dimension {expected_size} (factor {scaling_factor}x) does not match the actual image dimensions {width}x{height}"

            if intr.model == "SIMPLE_PINHOLE":
                focal_length = intr.params[0] / scaling_factor
                cx = intr.params[1] / scaling_factor
                cy = intr.params[2] / scaling_factor
                if self.gsplat_image_downscale:
                    self.intrinsics[intr.id] = create_pinhole_camera(
                        focal_length * image_scale[0],
                        focal_length * image_scale[1],
                        width,
                        height,
                        cx=cx * image_scale[0],
                        cy=cy * image_scale[1],
                    )
                else:
                    self.intrinsics[intr.id] = create_pinhole_camera(
                        focal_length, focal_length, width, height, cx=cx, cy=cy
                    )

            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0] / scaling_factor
                focal_length_y = intr.params[1] / scaling_factor
                cx = intr.params[2] / scaling_factor
                cy = intr.params[3] / scaling_factor
                if self.gsplat_image_downscale:
                    focal_length_x *= image_scale[0]
                    focal_length_y *= image_scale[1]
                    cx *= image_scale[0]
                    cy *= image_scale[1]
                self.intrinsics[intr.id] = create_pinhole_camera(
                    focal_length_x, focal_length_y, width, height, cx=cx, cy=cy
                )

            elif intr.model in {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"}:
                focal_length, principal_point, radial_coeffs, tangential_coeffs, thin_prism_coeffs = (
                    _opencv_pinhole_intrinsics_from_colmap(intr.model, intr.params, scaling_factor)
                )
                if self.gsplat_image_downscale:
                    focal_length *= image_scale
                    principal_point *= image_scale
                self.intrinsics[intr.id] = create_opencv_pinhole_camera(
                    focal_length,
                    principal_point,
                    width,
                    height,
                    radial_coeffs,
                    tangential_coeffs,
                    thin_prism_coeffs,
                )

            elif intr.model == "OPENCV_FISHEYE":
                params = copy.deepcopy(intr.params)
                params[:4] = params[:4] / scaling_factor
                if self.gsplat_image_downscale:
                    params[:4] *= image_scale[[0, 1, 0, 1]]
                self.intrinsics[intr.id] = create_fisheye_camera(params, width, height)

            else:
                assert False, (
                    f"Colmap camera model '{intr.model}' not handled: supported camera models are "
                    "PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, and OPENCV_FISHEYE."
                )

        # Load poses and paths
        self.poses = []
        self.image_paths = []
        self.mask_paths = []
        self.fullcircle_distractor_mask_paths = []

        cam_centers = []
        for extr in logger.track(
            self.cam_extrinsics,
            description=f"Load Dataset ({self.split})",
            color="salmon1",
        ):
            R = qvec_to_so3(extr.qvec)
            T = np.array(extr.tvec)
            W2C = np.zeros((4, 4), dtype=np.float32)
            W2C[:3, 3] = T
            W2C[:3, :3] = R
            W2C[3, 3] = 1.0
            C2W = np.linalg.inv(W2C)
            self.poses.append(C2W)
            cam_centers.append(C2W[:3, 3])

            if self.gsplat_image_downscale:
                image_path = os.path.join(image_dir, colmap_to_image[extr.name])
            else:
                image_path = os.path.join(self.path, self.get_images_folder(), extr.name)
            self.image_paths.append(image_path)
            self.mask_paths.append(self._mask_path(image_path))
            self.fullcircle_distractor_mask_paths.append(
                self._mask_path(image_path, self.fullcircle_distractor_mask_root)
                if self.fullcircle_distractor_mask_root
                else ""
            )

        self.camera_centers = np.array(cam_centers)
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        self.poses = np.stack(self.poses)

        self.image_paths = np.stack(self.image_paths, dtype=str)
        self.mask_paths = np.stack(self.mask_paths, dtype=str)
        self.fullcircle_distractor_mask_paths = np.stack(self.fullcircle_distractor_mask_paths, dtype=str)
        metadata = [_novel_eval_metadata(extr.name) for extr in self.cam_extrinsics]
        self.source_eval_splits = np.asarray(
            [item[0] if item is not None else "regular" for item in metadata],
            dtype=str,
        )
        self.state_code_strings = np.asarray(
            [item[1] if item is not None else "" for item in metadata],
            dtype=str,
        )
        self.time_indices = np.asarray(
            [
                parsed[0] if (parsed := parse_frame_time_indices([path])) is not None else -1
                for path in self.image_paths
            ],
            dtype=np.int64,
        )

    def _lazy_worker_intrinsics_cache(self):
        """Create intrinsics cache for a specific worker."""
        worker_id = get_worker_id()

        # Check if this worker already has cached tensors
        if worker_id not in self._worker_gpu_cache:
            # For now, fall back to the original approach for each worker
            # This ensures each worker creates its own GPU tensors
            worker_intrinsics = {}
            for intr_id, (
                params_dict,
                rays_ori,
                rays_dir,
                camera_name,
                pixel_coords,
            ) in self.intrinsics.items():
                # Create new GPU tensors for this worker
                worker_rays_ori = rays_ori.to(self.device, non_blocking=True)
                worker_rays_dir = rays_dir.to(self.device, non_blocking=True)
                worker_pixel_coords = pixel_coords.to(self.device, non_blocking=True)
                worker_intrinsics[intr_id] = (
                    params_dict,
                    worker_rays_ori,
                    worker_rays_dir,
                    camera_name,
                    worker_pixel_coords,
                )
            self._worker_gpu_cache[worker_id] = worker_intrinsics

        return self._worker_gpu_cache[worker_id]

    @torch.no_grad()
    def compute_spatial_extents(self):
        camera_origins = torch.FloatTensor(self.poses[:, :3, 3])
        center = camera_origins.mean(dim=0)
        dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
        mean_dist = torch.mean(dists)  # mean distance between of cameras from center
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values
        return center, mean_dist, (bbox_min, bbox_max)

    def get_length_scale(self):
        return self.length_scale

    def get_center(self):
        return self.center

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene_bbox

    def get_scene_extent(self):
        return self.cameras_extent

    def get_observer_points(self):
        return self.camera_centers

    def get_world_normalization_transform(self) -> np.ndarray:
        return self.world_normalization_transform.copy()

    def get_poses(self) -> np.ndarray:
        """Get camera poses as 4x4 transformation matrices.

        COLMAP Dataset Implementation:
        COLMAP naturally provides poses in a coordinate system compatible with
        3DGRUT's "right down front" convention, so no coordinate conversion is needed.

        The poses are constructed from COLMAP's world-to-camera matrices by:
        1. Building W2C from rotation (qvec_to_so3) and translation (tvec)
        2. Inverting to get camera-to-world: C2W = inv(W2C)

        Returns:
            np.ndarray: Camera poses with shape (N, 4, 4) in "right down front" convention
        """
        return self.poses

    def get_intrinsics_idx(self, extr_idx: int):
        return self.cam_extrinsics[extr_idx].camera_id

    def get_camera_idx(self, frame_idx: int) -> int:
        """Return 0-based camera index for a given frame index.

        Maps from COLMAP's potentially non-contiguous camera_id to a
        0-based contiguous index.
        """
        colmap_camera_id = self.cam_extrinsics[frame_idx].camera_id
        return self._camera_id_to_idx[colmap_camera_id]

    def get_frames_per_camera(self) -> list[int]:
        """Return list of frame counts per camera.

        Returns a list where index i contains the number of frames captured
        by camera i (using 0-based camera indices). Derived values:
        - num_cameras = len(frames_per_camera)
        - num_frames = sum(frames_per_camera)
        """
        num_cameras = len(self.cam_intrinsics)
        counts = [0] * num_cameras
        for extr in self.cam_extrinsics:
            camera_idx = self._camera_id_to_idx[extr.camera_id]
            counts[camera_idx] += 1
        return counts

    def get_camera_names(self) -> list[str]:
        """Return list of camera names.

        For multi-camera setups where images are organized in subfolders by camera,
        returns the folder names. For single-camera setups (images directly in images
        folder), returns default names like "camera_0".
        """
        num_cameras = len(self.cam_intrinsics)
        names: list[str | None] = [None] * num_cameras

        # Find one image path for each camera to determine folder name
        for extr in self.cam_extrinsics:
            camera_idx = self._camera_id_to_idx[extr.camera_id]
            if names[camera_idx] is not None:
                continue  # Already have a name for this camera

            # extr.name is relative path from images folder
            # e.g., "cam_front/image001.jpg" or just "image001.jpg"
            parent_folder = os.path.dirname(extr.name)
            if parent_folder:
                names[camera_idx] = parent_folder
            else:
                names[camera_idx] = f"camera_{camera_idx}"

        return names

    def __len__(self) -> int:
        return self.n_frames

    def _get_valid_mask(self, width: int, height: int) -> Optional[np.ndarray]:
        if not self.valid_mask_path:
            return None

        key = (width, height)
        if key not in self._valid_mask_cache:
            if not os.path.isfile(self.valid_mask_path):
                raise FileNotFoundError(f"Valid mask not found: {self.valid_mask_path}")
            mask = Image.open(self.valid_mask_path).convert("L")
            if mask.size != key:
                mask = mask.resize(key, Image.Resampling.NEAREST)
            self._valid_mask_cache[key] = np.asarray(mask) > 127
        return self._valid_mask_cache[key]

    def _mask_path(self, image_path: str, mask_root: str = "") -> str:
        if not mask_root:
            return os.path.splitext(image_path)[0] + "_mask.png"

        mask_root = mask_root if os.path.isabs(mask_root) else os.path.join(self.path, mask_root)
        image_root = os.path.join(self.path, self.get_images_folder())
        relative_image = os.path.relpath(image_path, image_root)
        return os.path.join(mask_root, os.path.splitext(relative_image)[0] + "_mask.png")

    def _load_data_and_mask(self, idx: int) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode one RGB image and its combined mask into CPU uint8 tensors."""
        with Image.open(self.image_paths[idx]) as image:
            image_data = np.asarray(image).copy()
        actual_h, actual_w = image_data.shape[:2]

        if image_data.dtype != np.uint8:
            raise TypeError(f"Image data must be uint8, got {image_data.dtype}")

        mask = None
        if os.path.exists(mask_path := self.mask_paths[idx]):
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L")) > 127

        if os.path.exists(mask_path := self.fullcircle_distractor_mask_paths[idx]):
            with Image.open(mask_path) as image:
                source_mask = np.asarray(image.convert("L")) > 127
            valid_mask = source_mask if self.distractor_mask_white_is_valid else ~source_mask
            mask = valid_mask if mask is None else mask & valid_mask

        valid_mask = self._get_valid_mask(actual_w, actual_h)
        if valid_mask is not None:
            mask = valid_mask if mask is None else mask & valid_mask

        data_tensor = torch.from_numpy(image_data).unsqueeze(0)
        mask_tensor = None
        if mask is not None:
            mask_tensor = torch.from_numpy(mask.astype(np.uint8) * 255).reshape(1, actual_h, actual_w, 1)
        return data_tensor, mask_tensor

    def _preload_decoded_data(self) -> None:
        """Decode the selected split once and retain compact uint8 tensors in host RAM."""
        started = time.perf_counter()
        logger.info(f"Preloading {len(self)} decoded COLMAP samples into host RAM")
        self._preloaded_data = [self._load_data_and_mask(idx) for idx in range(len(self))]
        cached_bytes = sum(
            data.numel() * data.element_size()
            + (0 if mask is None else mask.numel() * mask.element_size())
            for data, mask in self._preloaded_data
        )
        logger.info(
            "COLMAP_RAM_CACHE_READY "
            f"samples={len(self._preloaded_data)} gib={cached_bytes / 2**30:.2f} "
            f"seconds={time.perf_counter() - started:.2f}"
        )

    @torch.cuda.nvtx.range("colmap_dataset::_getitem")
    def __getitem__(self, idx) -> dict:
        if self._preloaded_data is None:
            data_tensor, mask_tensor = self._load_data_and_mask(idx)
        else:
            data_tensor, mask_tensor = self._preloaded_data[idx]

        output_dict = {
            "data": data_tensor,
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "intr": self.get_intrinsics_idx(idx),
            "camera_idx": self.get_camera_idx(idx),
            "frame_idx": idx,
        }
        if self.time_indices is not None:
            output_dict["time_idx"] = int(self.time_indices[idx])
        if self.state_code_strings[idx]:
            state_code = remap_novel_state_code(self.state_code_strings[idx], self.state_code_mapping)
        else:
            state_code = []
        if state_code:
            output_dict["state_code"] = torch.tensor(
                state_code,
                dtype=torch.float32,
            )

        if mask_tensor is not None:
            output_dict["mask"] = mask_tensor

        # Add EXIF exposure if available for this frame
        if self.exif_exposures is not None and self.exif_exposures[idx] is not None:
            output_dict["exposure"] = torch.tensor(self.exif_exposures[idx], dtype=torch.float32)

        return output_dict

    def get_gpu_batch_with_intrinsics(self, batch):
        """Add the intrinsics to the batch and move data to GPU."""

        batch_size = int(batch["data"].shape[0])
        if batch_size > 1:
            data = batch["data"][:, 0].to(self.device, non_blocking=True) / 255.0
            pose = batch["pose"][:, 0].to(self.device, non_blocking=True)
            assert data.dtype == torch.float32
            assert pose.dtype == torch.float32

            intrinsics_ids = batch["intr"].reshape(-1)
            if intrinsics_ids.numel() != batch_size:
                raise ValueError(
                    f"Expected one COLMAP intrinsic ID per view, got {tuple(batch['intr'].shape)} for B={batch_size}"
                )

            worker_intrinsics = self._lazy_worker_intrinsics_cache()
            camera_entries = [worker_intrinsics[int(intrinsic_id.item())] for intrinsic_id in intrinsics_ids]
            camera_names = [entry[3] for entry in camera_entries]
            if any(name != camera_names[0] for name in camera_names[1:]):
                raise ValueError(
                    "All COLMAP views in a logical batch must use the same camera-model kind; "
                    f"received {camera_names}"
                )

            ray_shapes = [tuple(entry[2].shape[1:3]) for entry in camera_entries]
            if any(shape != ray_shapes[0] for shape in ray_shapes[1:]) or ray_shapes[0] != tuple(data.shape[1:3]):
                raise ValueError(
                    "Mixed image resolutions are not supported in one logical COLMAP batch; "
                    f"RGB is {tuple(data.shape[1:3])}, camera rays are {ray_shapes}"
                )

            camera_idx = torch.as_tensor(batch["camera_idx"]).reshape(-1)
            frame_idx = torch.as_tensor(batch["frame_idx"]).reshape(-1)
            if camera_idx.numel() != batch_size or frame_idx.numel() != batch_size:
                raise ValueError(f"Expected one camera/frame index per COLMAP view for B={batch_size}")

            sample = {
                "rgb_gt": data,
                "rays_ori": torch.cat([entry[1] for entry in camera_entries], dim=0),
                "rays_dir": torch.cat([entry[2] for entry in camera_entries], dim=0),
                "T_to_world": pose,
                f"intrinsics_{camera_names[0]}": stack_camera_parameter_dicts([entry[0] for entry in camera_entries]),
                "camera_idx": camera_idx,
                "frame_idx": frame_idx,
                "pixel_coords": torch.cat([entry[4] for entry in camera_entries], dim=0),
            }

            if "mask" in batch:
                mask = batch["mask"][:, 0].to(self.device, non_blocking=True) / 255.0
                sample["mask"] = (mask > 0.5).to(torch.float32)

            if "exposure" in batch:
                sample["exposure"] = batch["exposure"].to(self.device, non_blocking=True)

            if "time_idx" in batch:
                time_idx = torch.as_tensor(batch["time_idx"]).reshape(-1)
                if time_idx.numel() != batch_size:
                    raise ValueError(f"Expected one time index per COLMAP view for B={batch_size}")
                sample["time_idx"] = time_idx.to(self.device, non_blocking=True)
            if "state_code" in batch:
                sample["state_code"] = batch["state_code"].to(self.device, non_blocking=True)

            return Batch(**sample)

        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        intr = batch["intr"][0].item()

        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        # Get intrinsics for current worker
        worker_intrinsics = self._lazy_worker_intrinsics_cache()

        camera_params_dict, rays_ori, rays_dir, camera_name, pixel_coords = worker_intrinsics[intr]

        sample = {
            "rgb_gt": data,
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": pose,
            f"intrinsics_{camera_name}": camera_params_dict,
            "camera_idx": batch["camera_idx"][0].item(),
            "frame_idx": batch["frame_idx"][0].item(),
            "pixel_coords": pixel_coords,
        }

        if "mask" in batch:
            mask = batch["mask"][0].to(self.device, non_blocking=True) / 255.0
            mask = (mask > 0.5).to(torch.float32)
            sample["mask"] = mask

        # Add exposure prior from EXIF if available (move to GPU)
        if "exposure" in batch and batch["exposure"][0] is not None:
            sample["exposure"] = batch["exposure"].to(self.device)

        if "time_idx" in batch:
            time_idx = torch.as_tensor(batch["time_idx"]).reshape(-1)
            if time_idx.numel() != 1:
                raise ValueError(f"Expected one time index per COLMAP view, got {time_idx.numel()}")
            sample["time_idx"] = time_idx.to(self.device, non_blocking=True)
        if "state_code" in batch:
            sample["state_code"] = batch["state_code"].to(self.device, non_blocking=True)

        return Batch(**sample)

    def create_dataset_camera_visualization(self):
        """Create a visualization of the dataset cameras."""

        cam_list = []

        for i_cam, pose in enumerate(self.poses):
            trans_mat = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # Camera convention rotation
            camera_convention_rot = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            # Get camera ID and corresponding intrinsics
            camera_id = self.get_intrinsics_idx(i_cam)
            intr, _, _, _, _ = self.intrinsics[camera_id]

            # Load actual image to get dimensions
            image_data = np.asarray(Image.open(self.image_paths[i_cam]))
            h, w = image_data.shape[:2]

            f_w = intr["focal_length"][0]
            f_h = intr["focal_length"][1]

            fov_w = 2.0 * np.arctan(0.5 * w / f_w)
            fov_h = 2.0 * np.arctan(0.5 * h / f_h)

            assert image_data.dtype == np.uint8, "Image data must be of type uint8"
            rgb = image_data.reshape(h, w, 3) / np.float32(255.0)
            assert rgb.dtype == np.float32, f"RGB image must be float32, got {rgb.dtype}"

            cam_list.append(
                {
                    "ext_mat": trans_mat_world_to_camera,
                    "w": w,
                    "h": h,
                    "fov_w": fov_w,
                    "fov_h": fov_h,
                    "rgb_img": rgb,
                    "split": self.split,
                }
            )

        create_camera_visualization(cam_list)
