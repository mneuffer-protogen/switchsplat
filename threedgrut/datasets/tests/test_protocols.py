# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from threedgrut.datasets.protocols import Batch, slice_batch
from threedgrut.datasets.utils import collate_same_resolution, stack_camera_parameter_dicts


def test_batch_supports_static_and_rolling_shutter_poses():
    rays = torch.zeros((1, 1, 1, 3))
    pose_start = torch.eye(4).unsqueeze(0)

    static = Batch(rays_ori=rays, rays_dir=rays, T_to_world=pose_start)
    assert static.T_to_world_end is None
    assert not static.rays_in_world_space
    assert static.time_idx is None

    pose_end = pose_start.clone()
    rolling = Batch(
        rays_ori=rays,
        rays_dir=rays,
        T_to_world=pose_start,
        T_to_world_end=pose_end,
        rays_in_world_space=True,
    )
    assert rolling.T_to_world_end is pose_end
    assert rolling.rays_in_world_space


def test_slice_batch_preserves_image_axes_and_slices_camera_metadata():
    batch_size, height, width = 2, 3, 4
    rays = torch.arange(batch_size * height * width * 3, dtype=torch.float32).reshape(
        batch_size, height, width, 3
    )
    poses = torch.eye(4).repeat(batch_size, 1, 1)
    poses[1, 0, 3] = 0.25
    batch = Batch(
        rays_ori=torch.zeros_like(rays),
        rays_dir=rays,
        T_to_world=poses,
        T_to_world_end=poses.clone(),
        rgb_gt=torch.zeros(batch_size, height, width, 3),
        mask=torch.ones(batch_size, height, width, 1),
        intrinsics=[
            torch.tensor([10.0, 11.0]),
            12.0,
            torch.tensor([1.5, 1.75]),
            2.0,
        ],
        intrinsics_OpenCVPinholeCameraModelParameters=None,
        camera_idx=torch.tensor([3, 7]),
        frame_idx=[20, 21],
        time_idx=torch.tensor([1, 41]),
        state_code=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        pixel_coords=torch.zeros(batch_size, height, width, 2),
        exposure=torch.tensor([0.1, -0.2]),
    )

    sliced = slice_batch(batch, 1)
    assert sliced.batch_size == 1
    assert sliced.rays_dir.shape == (1, height, width, 3)
    assert sliced.T_to_world.shape == (1, 4, 4)
    assert sliced.T_to_world_end.shape == (1, 4, 4)
    assert sliced.rgb_gt.shape == (1, height, width, 3)
    assert sliced.mask.shape == (1, height, width, 1)
    assert sliced.pixel_coords.shape == (1, height, width, 2)
    assert sliced.exposure.shape == (1,)
    assert sliced.intrinsics[0].item() == 11.0
    assert sliced.intrinsics[1] == 12.0
    assert sliced.camera_idx == 7
    assert sliced.frame_idx == 21
    torch.testing.assert_close(sliced.time_idx, torch.tensor([41]))
    torch.testing.assert_close(sliced.state_code, torch.tensor([[1.0, 0.0]]))


def test_slice_batch_handles_nested_batched_and_shared_camera_parameters():
    batch_size, height, width = 2, 2, 3
    camera_parameters = {
        "resolution": np.array([width, height], dtype=np.uint64),
        "shutter_type": ["GLOBAL", "GLOBAL"],
        "principal_point": torch.tensor([[1.2, 0.8], [1.3, 0.9]]),
        "focal_length": torch.tensor([[5.0, 5.5], [6.0, 6.5]]),
        "radial_coeffs": torch.zeros(batch_size, 6),
        "tangential_coeffs": torch.zeros(batch_size, 2),
        "thin_prism_coeffs": torch.zeros(batch_size, 4),
        "extra": {"per_camera": torch.tensor([4.0, 8.0]), "shared": "value"},
    }
    batch = Batch(
        rays_ori=torch.zeros(batch_size, height, width, 3),
        rays_dir=torch.zeros(batch_size, height, width, 3),
        T_to_world=torch.eye(4).repeat(batch_size, 1, 1),
        intrinsics_OpenCVPinholeCameraModelParameters=camera_parameters,
    )

    sliced = slice_batch(batch, 1)
    parameters = sliced.intrinsics_OpenCVPinholeCameraModelParameters
    assert parameters["resolution"] is camera_parameters["resolution"]
    assert parameters["shutter_type"] == "GLOBAL"
    torch.testing.assert_close(parameters["principal_point"], torch.tensor([1.3, 0.9]))
    torch.testing.assert_close(parameters["focal_length"], torch.tensor([6.0, 6.5]))
    assert parameters["extra"]["per_camera"].item() == 8.0
    assert parameters["extra"]["shared"] == "value"


def test_batch_rejects_mismatched_image_domains():
    with pytest.raises(ValueError, match="share one H x W resolution"):
        Batch(
            rays_ori=torch.zeros(2, 3, 4, 3),
            rays_dir=torch.zeros(2, 4, 4, 3),
            T_to_world=torch.eye(4).repeat(2, 1, 1),
        )


def test_collate_rejects_mixed_resolution_with_clear_error():
    samples = [
        {"data": torch.zeros(1, 3, 4, 3), "pose": torch.eye(4).unsqueeze(0)},
        {"data": torch.zeros(1, 5, 4, 3), "pose": torch.eye(4).unsqueeze(0)},
    ]
    with pytest.raises(ValueError, match="Mixed image resolutions"):
        collate_same_resolution(samples)


def test_camera_parameter_batching_preserves_shared_none_fields():
    stacked = stack_camera_parameter_dicts(
        [
            {"camera_model": "fisheye", "external_distortion_parameters": None},
            {"camera_model": "fisheye", "external_distortion_parameters": None},
        ]
    )
    assert stacked["camera_model"] == ["fisheye", "fisheye"]
    assert stacked["external_distortion_parameters"] is None

    with pytest.raises(ValueError, match="missing for only some views"):
        stack_camera_parameter_dicts([{"value": torch.tensor(1.0)}, {"value": None}])


def test_stack_camera_parameter_dicts_preserves_shared_none_values():
    stacked = stack_camera_parameter_dicts(
        [
            {"focal_length": [1.0, 2.0], "radial_coeffs": None},
            {"focal_length": [3.0, 4.0], "radial_coeffs": None},
        ]
    )

    torch.testing.assert_close(stacked["focal_length"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert stacked["radial_coeffs"] is None


def test_stack_camera_parameter_dicts_rejects_mixed_none_values():
    with pytest.raises(ValueError, match="missing for only some views"):
        stack_camera_parameter_dicts(
            [
                {"radial_coeffs": None},
                {"radial_coeffs": [0.1, 0.2]},
            ]
        )
