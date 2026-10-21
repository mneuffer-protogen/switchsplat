# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import warnings

import pytest
import torch
from omegaconf import OmegaConf

gsplat = pytest.importorskip("gsplat")
if not hasattr(gsplat, "rasterize_num_contributing_gaussians"):
    pytest.skip("The pinned gsplat commit is not installed", allow_module_level=True)

from gsplat_tracer import Tracer
import gsplat_tracer.tracer as gsplat_tracer_module
from threedgrut.datasets.protocols import Batch, slice_batch
from threedgrut.model.features import Features
from threedgrut.model.switch_splat_controller import SwitchSplatController
from threedgrut.utils.render import C0

REQUIRED_OUTPUTS = {
    "pred_features",
    "pred_opacity",
    "pred_dist",
    "pred_normals",
    "hits_count",
    "frame_time_ms",
    "mog_visibility",
}


def _config(method: str, *, enable_normals: bool = False):
    return OmegaConf.create(
        {
            "render": {
                "method": method,
                "enable_kernel_timings": False,
                "enable_normals": enable_normals,
                "gsplat": {
                    "packed": method == "gsplat_3dgs",
                    "sparse_grad": False,
                    "absgrad": False,
                    "rasterize_mode": "classic",
                    "radius_clip": 0.0,
                    "eps2d": 0.3,
                    "global_z_order": True,
                },
                "splat": {
                    "ut_alpha": 1.0,
                    "ut_beta": 2.0,
                    "ut_kappa": 0.0,
                    "ut_in_image_margin_factor": 0.1,
                    "ut_require_all_sigma_points_valid": False,
                    "global_z_order": True,
                },
            }
        }
    )


class _Gaussians(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.positions = torch.nn.Parameter(
            torch.tensor(
                [
                    [-0.35, -0.18, 2.7],
                    [0.28, 0.22, 3.1],
                    [0.05, -0.36, 3.5],
                    [0.46, -0.08, 3.8],
                ],
                device=device,
            )
        )
        self.rotation = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.96, 0.12, -0.20, 0.10],
                    [0.91, -0.24, 0.14, 0.29],
                    [0.88, 0.31, 0.22, -0.18],
                    [0.93, -0.12, 0.30, 0.17],
                ],
                device=device,
            )
        )
        self.scale = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    [
                        [0.22, 0.10, 0.14],
                        [0.11, 0.25, 0.13],
                        [0.28, 0.09, 0.17],
                        [0.15, 0.12, 0.27],
                    ],
                    device=device,
                )
            )
        )
        self.density = torch.nn.Parameter(torch.tensor([[1.2], [0.8], [1.5], [0.5]], device=device))

        rgb = torch.tensor(
            [[0.85, 0.18, 0.12], [0.12, 0.82, 0.20], [0.16, 0.22, 0.88], [0.75, 0.65, 0.16]],
            device=device,
        )
        self.features_albedo = torch.nn.Parameter((rgb - 0.5) / C0)
        self.features_specular = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.09, -0.04, 0.02, -0.03, 0.08, 0.01, 0.02, -0.01, 0.06],
                    [-0.05, 0.03, 0.07, 0.06, -0.02, 0.04, -0.01, 0.05, -0.03],
                    [0.04, 0.07, -0.03, -0.02, 0.01, 0.09, 0.05, -0.04, 0.02],
                    [0.03, -0.06, 0.05, 0.07, 0.02, -0.04, -0.05, 0.03, 0.08],
                ],
                device=device,
            )
        )
        self.feature_type = Features.Type.SH
        self.ray_feature_dim = 3
        self.n_active_features = 1

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_positions(self):
        return self.positions

    def get_rotation(self):
        return torch.nn.functional.normalize(self.rotation, dim=-1)

    def get_scale(self):
        return torch.exp(self.scale)

    def get_density(self):
        return torch.sigmoid(self.density)

    def get_features(self):
        return torch.cat((self.features_albedo, self.features_specular), dim=1)


def _batch(
    device: torch.device,
    *,
    height: int = 36,
    width: int = 44,
    fx: float = 38.0,
    fy: float = 41.0,
    cx: float = 17.25,
    cy: float = 20.5,
    c2w: torch.Tensor | None = None,
) -> Batch:
    if c2w is None:
        c2w = torch.eye(4, device=device)
    else:
        c2w = c2w.to(device)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    rays_dir = torch.stack(((xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)), dim=-1)
    rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1).unsqueeze(0)
    rays_ori = torch.zeros_like(rays_dir)
    return Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=c2w.unsqueeze(0),
        rgb_gt=torch.zeros((1, height, width, 3), device=device),
        intrinsics=[fx, fy, cx, cy],
    )


def _camera_batch(device: torch.device, count: int = 2) -> Batch:
    cameras = []
    intrinsics = [[], [], [], []]
    for index in range(count):
        c2w = torch.eye(4, device=device)
        c2w[0, 3] = 0.12 * index
        angle = 0.025 * index
        c2w[:3, :3] = torch.tensor(
            [
                [math.cos(angle), 0.0, math.sin(angle)],
                [0.0, 1.0, 0.0],
                [-math.sin(angle), 0.0, math.cos(angle)],
            ],
            device=device,
        )
        values = (38.0 + index, 41.0 + 0.5 * index, 17.25 + 0.2 * index, 20.5 - 0.1 * index)
        cameras.append(
            _batch(
                device,
                fx=values[0],
                fy=values[1],
                cx=values[2],
                cy=values[3],
                c2w=c2w,
            )
        )
        for component, value in zip(intrinsics, values):
            component.append(value)

    return Batch(
        rays_ori=torch.cat([camera.rays_ori for camera in cameras], dim=0),
        rays_dir=torch.cat([camera.rays_dir for camera in cameras], dim=0),
        T_to_world=torch.cat([camera.T_to_world for camera in cameras], dim=0),
        rgb_gt=torch.cat([camera.rgb_gt for camera in cameras], dim=0),
        intrinsics=[torch.tensor(values, device=device) for values in intrinsics],
        camera_idx=torch.arange(count, device=device),
        frame_idx=torch.arange(10, 10 + count, device=device),
    )


def _centroid(image: torch.Tensor) -> torch.Tensor:
    height, width = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=image.dtype) + 0.5,
        torch.arange(width, device=image.device, dtype=image.dtype) + 0.5,
        indexing="ij",
    )
    mass = image.clamp_min(0)
    return torch.stack(((mass * xs).sum(), (mass * ys).sum())) / mass.sum().clamp_min(1.0e-8)


def test_camera_conventions_and_intrinsics():
    tracer = Tracer(_config("gsplat_3dgs"))
    angle = 0.37
    rotation = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    c2w = torch.eye(4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = torch.tensor([1.2, -0.4, 0.7])
    batch = _batch(torch.device("cpu"), height=29, width=37, fx=31.0, fy=35.0, cx=10.25, cy=18.75, c2w=c2w)

    width, height, camera = tracer._camera_args(batch, device=torch.device("cpu"), dtype=torch.float32)
    assert (width, height) == (37, 29)
    torch.testing.assert_close(camera["viewmats"], torch.linalg.inv(c2w).unsqueeze(0))
    torch.testing.assert_close(
        camera["Ks"],
        torch.tensor([[[31.0, 0.0, 10.25], [0.0, 35.0, 18.75], [0.0, 0.0, 1.0]]]),
    )

    point_camera = torch.tensor([0.45, -0.30, 3.0, 1.0])
    point_world = c2w @ point_camera
    recovered = camera["viewmats"][0] @ point_world
    torch.testing.assert_close(recovered, point_camera)
    pixel = camera["Ks"][0] @ recovered[:3]
    pixel = pixel[:2] / pixel[2]
    torch.testing.assert_close(pixel, torch.tensor([14.9, 15.25]))
    assert pixel[0] > camera["Ks"][0, 0, 2]
    assert pixel[1] < camera["Ks"][0, 1, 2]


def test_distortion_and_rolling_shutter_mapping():
    batch = _batch(torch.device("cpu"))
    batch.intrinsics = None
    batch.intrinsics_OpenCVPinholeCameraModelParameters = {
        "resolution": [44, 36],
        "shutter_type": "ROLLING_LEFT_TO_RIGHT",
        "principal_point": [17.25, 20.5],
        "focal_length": [38.0, 41.0],
        "radial_coeffs": [0.01, -0.02, 0.003, 0.0, 0.0, 0.0],
        "tangential_coeffs": [0.001, -0.002],
        "thin_prism_coeffs": [0.0, 0.0, 0.0, 0.0],
    }
    end_pose = torch.eye(4)
    end_pose[0, 3] = 0.03
    batch.T_to_world_end = end_pose.unsqueeze(0)

    tracer = Tracer(_config("gsplat_3dgut"))
    _, _, camera = tracer._camera_args(batch, device=torch.device("cpu"), dtype=torch.float32)
    assert camera["camera_model"] == "pinhole"
    assert camera["rolling_shutter"] == gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT
    torch.testing.assert_close(camera["viewmats_rs"], torch.linalg.inv(batch.T_to_world_end))
    torch.testing.assert_close(camera["radial_coeffs"], torch.tensor([[0.01, -0.02, 0.003, 0.0, 0.0, 0.0]]))

    with pytest.raises(NotImplementedError, match="require render=gsplat_3dgut"):
        Tracer(_config("gsplat_3dgs"))._camera_args(
            batch,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_ftheta_camera_mapping():
    batch = _batch(torch.device("cpu"))
    batch.intrinsics = None
    batch.intrinsics_FThetaCameraModelParameters = {
        "resolution": [44, 36],
        "shutter_type": "GLOBAL",
        "principal_point": [17.25, 20.5],
        "reference_poly": "PIXELDIST_TO_ANGLE",
        "pixeldist_to_angle_poly": [0.0, 0.01, 0.0, 0.0, 0.0, 0.0],
        "angle_to_pixeldist_poly": [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        "max_angle": 1.2,
        "linear_cde": [1.0, 0.0, 0.0],
    }

    _, _, camera = Tracer(_config("gsplat_3dgut"))._camera_args(
        batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert camera["camera_model"] == "ftheta"
    assert camera["ftheta_coeffs"].reference_poly == gsplat.rendering.FThetaPolynomialType.PIXELDIST_TO_ANGLE
    torch.testing.assert_close(camera["Ks"][0, :2, 2], torch.tensor([17.25, 20.5]))


def test_batched_camera_mapping():
    batch = _camera_batch(torch.device("cpu"), count=2)
    width, height, camera = Tracer(_config("gsplat_3dgs"))._camera_args(
        batch, device=torch.device("cpu"), dtype=torch.float32
    )
    assert (width, height) == (44, 36)
    assert camera["viewmats"].shape == (2, 4, 4)
    assert camera["Ks"].shape == (2, 3, 3)
    torch.testing.assert_close(camera["viewmats"], torch.linalg.inv(batch.T_to_world))
    torch.testing.assert_close(camera["Ks"][:, 0, 0], torch.tensor([38.0, 39.0]))
    torch.testing.assert_close(camera["Ks"][:, 1, 1], torch.tensor([41.0, 41.5]))


def test_batched_distortion_and_rolling_shutter_mapping():
    batch = _camera_batch(torch.device("cpu"), count=2)
    batch.intrinsics = None
    batch.intrinsics_OpenCVPinholeCameraModelParameters = {
        "resolution": torch.tensor([[44, 36], [44, 36]]),
        "shutter_type": ["ROLLING_LEFT_TO_RIGHT", "ROLLING_LEFT_TO_RIGHT"],
        "principal_point": torch.tensor([[17.25, 20.5], [17.5, 20.25]]),
        "focal_length": torch.tensor([[38.0, 41.0], [39.0, 41.5]]),
        "radial_coeffs": torch.tensor(
            [[0.01, -0.02, 0.003, 0.0, 0.0, 0.0], [0.02, -0.01, 0.004, 0.0, 0.0, 0.0]]
        ),
        "tangential_coeffs": torch.tensor([[0.001, -0.002], [0.003, -0.001]]),
        "thin_prism_coeffs": torch.zeros(2, 4),
    }
    batch.T_to_world_end = batch.T_to_world.clone()
    batch.T_to_world_end[:, 0, 3] += 0.03

    _, _, camera = Tracer(_config("gsplat_3dgut"))._camera_args(
        batch, device=torch.device("cpu"), dtype=torch.float32
    )
    assert camera["Ks"].shape == (2, 3, 3)
    assert camera["radial_coeffs"].shape == (2, 6)
    assert camera["tangential_coeffs"].shape == (2, 2)
    assert camera["thin_prism_coeffs"].shape == (2, 4)
    assert camera["viewmats_rs"].shape == (2, 4, 4)
    torch.testing.assert_close(camera["viewmats_rs"], torch.linalg.inv(batch.T_to_world_end))

    batch.intrinsics_OpenCVPinholeCameraModelParameters["shutter_type"][1] = "GLOBAL"
    with pytest.raises(NotImplementedError, match="one shared shutter_type"):
        Tracer(_config("gsplat_3dgut"))._camera_args(
            batch, device=torch.device("cpu"), dtype=torch.float32
        )


def test_gaussian_conversion_preserves_layout_and_autograd():
    gaussians = _Gaussians(torch.device("cpu"))
    converted = Tracer._gaussian_args(gaussians)
    assert converted["quats"].shape == (4, 4)
    assert converted["opacities"].shape == (4,)
    assert converted["colors"].shape == (4, 4, 3)
    assert converted["sh_degree"] == 1
    torch.testing.assert_close(converted["quats"], torch.nn.functional.normalize(gaussians.rotation, dim=-1))
    torch.testing.assert_close(converted["colors"][:, 0], gaussians.features_albedo)
    torch.testing.assert_close(converted["colors"][:, 1:].reshape(4, 9), gaussians.features_specular)

    sum(value.sum() for key, value in converted.items() if key != "sh_degree").backward()
    for parameter in gaussians.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    "batch_size,shape,expected_shape",
    [
        (1, (4,), (4,)),
        (1, (4, 1), (4,)),
        (2, (2, 4), (2, 4)),
        (2, (2, 4, 1), (2, 4)),
    ],
)
def test_opacity_override_shapes_are_canonicalized(batch_size, shape, expected_shape):
    opacity = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape).requires_grad_()
    canonical = Tracer._validate_opacity_override(opacity, batch_size, num_gaussians=4)
    assert canonical.shape == expected_shape
    assert canonical.requires_grad
    canonical.sum().backward()
    torch.testing.assert_close(opacity.grad, torch.ones_like(opacity))


@pytest.mark.parametrize("batch_size,shape", [(1, (1, 4, 1)), (2, (4, 1)), (2, (2, 5))])
def test_opacity_override_rejects_wrong_shapes(batch_size, shape):
    with pytest.raises(ValueError, match="opacity_override must have shape"):
        Tracer._validate_opacity_override(torch.ones(shape), batch_size, num_gaussians=4)


def test_paired_layout_expands_geometry_and_cameras_but_shares_sh_coefficients():
    gaussians = _Gaussians(torch.device("cpu"))
    gaussian_args = Tracer._gaussian_args(
        gaussians,
        opacity_override=torch.full((2, gaussians.num_gaussians), 0.5),
    )
    camera_args = {
        "viewmats": torch.eye(4).repeat(2, 1, 1),
        "Ks": torch.eye(3).repeat(2, 1, 1),
        "radial_coeffs": torch.zeros(2, 6),
        "camera_model": "pinhole",
        "viewmats_rs": None,
    }

    paired_gaussians, paired_cameras = Tracer._pair_gaussians_and_cameras(
        gaussian_args, camera_args, batch_size=2
    )

    assert paired_gaussians["means"].shape == (2, 4, 3)
    assert paired_gaussians["quats"].shape == (2, 4, 4)
    assert paired_gaussians["scales"].shape == (2, 4, 3)
    assert paired_gaussians["opacities"].shape == (2, 4)
    # The pinned gsplat SH contract requires [N,K,D], shared across leading batches.
    assert paired_gaussians["colors"].shape == (4, 4, 3)
    assert paired_cameras["viewmats"].shape == (2, 1, 4, 4)
    assert paired_cameras["Ks"].shape == (2, 1, 3, 3)
    assert paired_cameras["radial_coeffs"].shape == (2, 1, 6)
    assert paired_cameras["camera_model"] == "pinhole"
    assert paired_cameras["viewmats_rs"] is None


def test_non_sh_features_fail_explicitly():
    gaussians = _Gaussians(torch.device("cpu"))
    gaussians.feature_type = Features.Type.NHT
    with pytest.raises(NotImplementedError, match="feature_type='sh'"):
        Tracer._gaussian_args(gaussians)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization requires CUDA")
@pytest.mark.parametrize("method", ["gsplat_3dgs", "gsplat_3dgut"])
def test_output_contract_and_gradients(method: str):
    device = torch.device("cuda:0")
    gaussians = _Gaussians(device)
    tracer = Tracer(_config(method, enable_normals=method == "gsplat_3dgut"))
    batch = _batch(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        outputs = tracer.render(gaussians, batch, train=True, frame_id=7)

    assert set(outputs) == REQUIRED_OUTPUTS
    assert outputs["pred_features"].shape == (1, 36, 44, 3)
    for key in ("pred_opacity", "pred_dist", "hits_count"):
        assert outputs[key].shape == (1, 36, 44, 1)
    assert outputs["pred_normals"].shape == (1, 36, 44, 3)
    assert outputs["mog_visibility"].shape == gaussians.density.shape
    assert outputs["mog_visibility"].dtype == torch.bool
    assert tracer.timings == {}
    for key in ("pred_features", "pred_opacity", "pred_dist", "pred_normals", "hits_count"):
        assert outputs[key].device == device
        assert torch.isfinite(outputs[key]).all()

    ys = torch.linspace(0.7, 1.3, 36, device=device).view(1, 36, 1, 1)
    xs = torch.linspace(0.8, 1.4, 44, device=device).view(1, 1, 44, 1)
    channels = torch.tensor([0.7, 1.1, 1.5], device=device).view(1, 1, 1, 3)
    loss = (
        (outputs["pred_features"] * ys * xs * channels).mean()
        + 0.17 * (outputs["pred_opacity"] * xs).mean()
        + 0.03 * (outputs["pred_dist"] * ys).mean()
    )
    loss.backward()

    for name in ("positions", "rotation", "scale", "density", "features_albedo", "features_specular"):
        gradient = getattr(gaussians, name).grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization requires CUDA")
@pytest.mark.parametrize("method", ["gsplat_3dgs", "gsplat_3dgut"])
def test_synthetic_projection_and_camera_motion(method: str):
    device = torch.device("cuda:0")
    gaussians = _Gaussians(device)
    with torch.no_grad():
        gaussians.positions.copy_(
            torch.tensor(
                [[0.0, 0.0, 3.0], [0.42, 0.25, 3.2], [-0.38, -0.28, 3.4], [0.8, 0.1, 4.0]],
                device=device,
            )
        )
        gaussians.features_specular.zero_()
        rgb = torch.tensor(
            [[0.95, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.95], [0.0, 0.0, 0.0]],
            device=device,
        )
        gaussians.features_albedo.copy_((rgb - 0.5) / C0)

    tracer = Tracer(_config(method))
    batch = _batch(device, height=40, width=48, fx=43.0, fy=45.0, cx=19.0, cy=23.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        outputs = tracer.render(gaussians, batch)
    image = outputs["pred_features"][0]
    red = _centroid(image[..., 0])
    green = _centroid(image[..., 1])
    blue = _centroid(image[..., 2])
    assert torch.linalg.vector_norm(red - torch.tensor([19.0, 23.0], device=device)) < 2.0
    assert green[0] > red[0] and green[1] > red[1]
    assert blue[0] < red[0] and blue[1] < red[1]
    assert torch.isfinite(outputs["pred_opacity"]).all()
    assert torch.isfinite(outputs["pred_dist"]).all()

    translated = torch.eye(4, device=device)
    translated[0, 3] = 0.2
    moved_batch = _batch(device, height=40, width=48, fx=43.0, fy=45.0, cx=19.0, cy=23.0, c2w=translated)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        moved = tracer.render(gaussians, moved_batch)
    moved_red = _centroid(moved["pred_features"][0, ..., 0])
    assert moved_red[0] < red[0]

    angle = 0.08
    rotated = torch.eye(4, device=device)
    rotated[:3, :3] = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        device=device,
    )
    rotated_batch = _batch(
        device,
        height=40,
        width=48,
        fx=43.0,
        fy=45.0,
        cx=19.0,
        cy=23.0,
        c2w=rotated,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        turned = tracer.render(gaussians, rotated_batch)
    turned_red = _centroid(turned["pred_features"][0, ..., 0])
    assert turned_red[0] < red[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization requires CUDA")
@pytest.mark.parametrize("method", ["gsplat_3dgs", "gsplat_3dgut"])
def test_camera_batch_matches_single_view_loop(method: str, monkeypatch):
    device = torch.device("cuda:0")
    gaussians = _Gaussians(device)
    tracer = Tracer(_config(method, enable_normals=method == "gsplat_3dgut"))
    batch = _camera_batch(device, count=2)

    calls = 0
    batched_shapes = None
    rasterization = gsplat_tracer_module.rasterization

    def counted_rasterization(*args, **kwargs):
        nonlocal calls, batched_shapes
        calls += 1
        if calls == 1:
            batched_shapes = {
                key: tuple(kwargs[key].shape)
                for key in ("means", "opacities", "viewmats")
            }
        return rasterization(*args, **kwargs)

    monkeypatch.setattr(gsplat_tracer_module, "rasterization", counted_rasterization)
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", UserWarning)
        batched = tracer.render(gaussians, batch, train=False, frame_id=11)
    assert calls == 1
    assert batched_shapes == {
        "means": (batch.batch_size, gaussians.num_gaussians, 3),
        "opacities": (batch.batch_size, gaussians.num_gaussians),
        "viewmats": (batch.batch_size, 1, 4, 4),
    }

    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", UserWarning)
        per_view = [
            tracer.render(gaussians, slice_batch(batch, index), train=False, frame_id=11 + index)
            for index in range(batch.batch_size)
        ]

    for key in ("pred_features", "pred_opacity", "pred_dist", "pred_normals"):
        expected = torch.cat([output[key] for output in per_view], dim=0)
        torch.testing.assert_close(batched[key], expected, rtol=2.0e-4, atol=2.0e-5)
    torch.testing.assert_close(
        batched["hits_count"],
        torch.cat([output["hits_count"] for output in per_view], dim=0),
        rtol=0.0,
        atol=0.0,
    )
    visibility = torch.stack([output["mog_visibility"].bool() for output in per_view]).any(dim=0)
    torch.testing.assert_close(batched["mog_visibility"], visibility, rtol=0.0, atol=0.0)
    assert batched["mog_visibility"].shape == gaussians.density.shape


def _weighted_render_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    height, width = outputs["pred_features"].shape[1:3]
    ys = torch.linspace(0.7, 1.3, height, device=outputs["pred_features"].device).view(1, height, 1, 1)
    xs = torch.linspace(0.8, 1.4, width, device=outputs["pred_features"].device).view(1, 1, width, 1)
    channels = torch.tensor([0.7, 1.1, 1.5], device=outputs["pred_features"].device).view(1, 1, 1, 3)
    return (
        (outputs["pred_features"] * ys * xs * channels).mean()
        + 0.17 * (outputs["pred_opacity"] * xs).mean()
        + 0.03 * (outputs["pred_dist"] * ys).mean()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization requires CUDA")
@pytest.mark.parametrize("method", ["gsplat_3dgs", "gsplat_3dgut"])
def test_camera_batch_gradient_matches_single_view_loop(method: str):
    device = torch.device("cuda:0")
    gaussians = _Gaussians(device)
    tracer = Tracer(_config(method))
    batch = _camera_batch(device, count=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _weighted_render_loss(tracer.render(gaussians, batch, train=True, frame_id=21)).backward()
    batched_gradients = {name: parameter.grad.detach().clone() for name, parameter in gaussians.named_parameters()}
    gaussians.zero_grad(set_to_none=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loop_loss = sum(
            _weighted_render_loss(
                tracer.render(gaussians, slice_batch(batch, index), train=True, frame_id=21 + index)
            )
            for index in range(batch.batch_size)
        ) / batch.batch_size
        loop_loss.backward()

    for name, parameter in gaussians.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(batched_gradients[name], parameter.grad, rtol=3.0e-3, atol=3.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization requires CUDA")
@pytest.mark.parametrize("method", ["gsplat_3dgs", "gsplat_3dgut"])
def test_camera_specific_opacity_batch_matches_single_view_loop_and_gradients(method: str):
    device = torch.device("cuda:0")
    gaussians = _Gaussians(device)
    tracer = Tracer(_config(method, enable_normals=method == "gsplat_3dgut"))
    batch = _camera_batch(device, count=2)

    switch_config = OmegaConf.create(
        {
            "num_slots": 1,
            "num_frames": 256,
            "init_windows": [[20.0, 80.0]],
            "temperature": 10.0,
        }
    )
    initial_base_opacity = gaussians.get_density().detach()
    initial_switch = SwitchSplatController(switch_config).to(device)
    initial_switch_logits = initial_switch.initialize_logits(
        gaussians.num_gaussians, device=device, dtype=initial_base_opacity.dtype
    ).detach()
    with torch.no_grad():
        initial_switch_logits[:, 1] += torch.linspace(-0.4, 0.2, gaussians.num_gaussians, device=device)
        initial_switch_logits[:, 2] += torch.linspace(0.3, -0.1, gaussians.num_gaussians, device=device)
    times = torch.tensor([27.0, 68.0], device=device)

    def make_switch_inputs():
        switch = SwitchSplatController(switch_config).to(device)
        switch.load_state_dict(initial_switch.state_dict())
        base_opacity = initial_base_opacity.clone().requires_grad_()
        switch_logits = initial_switch_logits.clone().requires_grad_()
        opacity = switch.compute_effective_opacity(base_opacity, switch_logits, times)
        return switch, base_opacity, switch_logits, opacity

    batched_switch, batched_base_opacity, batched_switch_logits, opacity_override = make_switch_inputs()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        batched = tracer.render(
            gaussians,
            batch,
            train=True,
            frame_id=31,
            opacity_override=opacity_override,
        )
        _weighted_render_loss(batched).backward()

    batched_parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in gaussians.named_parameters()
        if parameter.grad is not None
    }
    batched_upstream_gradients = {
        "base_opacity": batched_base_opacity.grad.detach().clone(),
        "switch_logits": batched_switch_logits.grad.detach().clone(),
        "endpoints_raw": batched_switch.endpoints_raw.grad.detach().clone(),
        "temperature_raw": batched_switch.temperature_raw.grad.detach().clone(),
    }
    assert gaussians.density.grad is None
    gaussians.zero_grad(set_to_none=True)

    loop_switch, loop_base_opacity, loop_switch_logits, loop_opacity = make_switch_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        per_view = [
            tracer.render(
                gaussians,
                slice_batch(batch, index),
                train=True,
                frame_id=31 + index,
                opacity_override=loop_opacity[index],
            )
            for index in range(batch.batch_size)
        ]
        loop_loss = sum(_weighted_render_loss(output) for output in per_view) / batch.batch_size
        loop_loss.backward()

    for key in ("pred_features", "pred_opacity", "pred_dist", "pred_normals"):
        torch.testing.assert_close(
            batched[key],
            torch.cat([output[key] for output in per_view], dim=0),
            rtol=2.0e-4,
            atol=2.0e-5,
        )
    torch.testing.assert_close(
        batched["hits_count"],
        torch.cat([output["hits_count"] for output in per_view], dim=0),
        rtol=0.0,
        atol=0.0,
    )
    expected_visibility = torch.stack(
        [output["mog_visibility"].bool() for output in per_view], dim=0
    ).any(dim=0)
    torch.testing.assert_close(batched["mog_visibility"], expected_visibility, rtol=0.0, atol=0.0)
    loop_upstream_gradients = {
        "base_opacity": loop_base_opacity.grad,
        "switch_logits": loop_switch_logits.grad,
        "endpoints_raw": loop_switch.endpoints_raw.grad,
        "temperature_raw": loop_switch.temperature_raw.grad,
    }
    for name, batched_gradient in batched_upstream_gradients.items():
        loop_gradient = loop_upstream_gradients[name]
        assert loop_gradient is not None, name
        assert torch.isfinite(batched_gradient).all(), name
        assert torch.count_nonzero(batched_gradient), name
        torch.testing.assert_close(batched_gradient, loop_gradient, rtol=3.0e-3, atol=3.0e-5)

    assert gaussians.density.grad is None
    for name, parameter in gaussians.named_parameters():
        if name == "density":
            continue
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            batched_parameter_gradients[name], parameter.grad, rtol=3.0e-3, atol=3.0e-5
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat rasterization requires CUDA")
@pytest.mark.parametrize("method", ["gsplat_3dgs", "gsplat_3dgut"])
def test_camera_batch_visibility_is_union_of_different_view_subsets(method: str):
    device = torch.device("cuda:0")
    gaussians = _Gaussians(device)
    with torch.no_grad():
        gaussians.positions.copy_(
            torch.tensor([[-4.0, 0.0, 3.0], [4.0, 0.0, 3.0], [-4.0, 0.2, 3.2], [4.0, 0.2, 3.2]], device=device)
        )
        gaussians.scale.fill_(-3.0)

    cameras = []
    for camera_x in (-4.0, 4.0):
        c2w = torch.eye(4, device=device)
        c2w[0, 3] = camera_x
        cameras.append(_batch(device, c2w=c2w))
    batch = Batch(
        rays_ori=torch.cat([camera.rays_ori for camera in cameras]),
        rays_dir=torch.cat([camera.rays_dir for camera in cameras]),
        T_to_world=torch.cat([camera.T_to_world for camera in cameras]),
        intrinsics=[38.0, 41.0, 17.25, 20.5],
    )

    tracer = Tracer(_config(method))
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        per_view = [tracer.render(gaussians, slice_batch(batch, index)) for index in range(2)]
        batched = tracer.render(gaussians, batch)

    view_visibility = [output["mog_visibility"].bool() for output in per_view]
    assert not torch.equal(view_visibility[0], view_visibility[1])
    expected = torch.stack(view_visibility).any(dim=0)
    torch.testing.assert_close(batched["mog_visibility"], expected, rtol=0.0, atol=0.0)
    assert batched["mog_visibility"].shape == (gaussians.num_gaussians, 1)
