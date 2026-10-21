# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Any

import torch
from gsplat import (
    RendererConfig_MixedBatch,
    RendererConfig_ParallelBatch,
    RollingShutterType,
    rasterization,
    rasterize_num_contributing_gaussians,
)
from gsplat.rendering import (
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    UnscentedTransformParameters,
)

from threedgrut.datasets.protocols import Batch


class Tracer:
    """Use gsplat as the rasterizer for the existing 3DGRUT Gaussian model."""

    _METHOD_FLAGS = {
        "gsplat_3dgs": (False, False),
        "gsplat_3dgut": (True, True),
    }

    def __init__(self, conf):
        self.conf = conf
        self.method = str(conf.render.method)
        try:
            self.with_ut, self.with_eval3d = self._METHOD_FLAGS[self.method]
        except KeyError as exc:
            raise ValueError(f"Unsupported gsplat rendering method: {self.method}") from exc

        self.gsplat_conf = conf.render.gsplat
        renderer_config_name = str(getattr(self.gsplat_conf, "renderer_config", "mixed_batch"))
        renderer_configs = {
            "mixed_batch": RendererConfig_MixedBatch,
            "parallel_batch": RendererConfig_ParallelBatch,
        }
        try:
            self.renderer_config = renderer_configs[renderer_config_name]()
        except KeyError as exc:
            raise ValueError(
                "render.gsplat.renderer_config must be one of "
                f"{sorted(renderer_configs)}, got {renderer_config_name!r}"
            ) from exc
        self._timing_enabled = bool(conf.render.enable_kernel_timings)
        self._timings: dict[str, float] = {}
        self._warned_missing_hit_counts = False
        self._warned_fisheye_max_angle = False

        if self.with_eval3d:
            if bool(self.gsplat_conf.packed):
                raise ValueError("gsplat_3dgut requires render.gsplat.packed=false")
            if bool(self.gsplat_conf.sparse_grad):
                raise ValueError("gsplat_3dgut requires render.gsplat.sparse_grad=false")
            if str(self.gsplat_conf.rasterize_mode) != "classic":
                raise ValueError("gsplat_3dgut requires render.gsplat.rasterize_mode=classic")

    @property
    def timings(self) -> dict[str, float]:
        return self._timings

    def build_acc(self, gaussians, rebuild=True):
        pass

    @staticmethod
    def _as_camera_values(
        value: Any,
        size: int,
        batch_size: int,
        *,
        device,
        dtype,
        name: str,
    ) -> torch.Tensor:
        values = torch.as_tensor(value, device=device, dtype=dtype)
        if values.ndim == 0 and size == 1:
            return values.reshape(1, 1).expand(batch_size, 1)
        if values.ndim == 1:
            if values.numel() == size:
                return values.reshape(1, size).expand(batch_size, size)
            if size == 1 and values.numel() == batch_size:
                return values.reshape(batch_size, 1)
        if values.ndim == 2:
            if values.shape == (1, size):
                return values.expand(batch_size, size)
            if values.shape == (batch_size, size):
                return values
        raise ValueError(
            f"{name} must be shared [{size}] or per-camera [{batch_size},{size}], "
            f"got {tuple(values.shape)}"
        )

    def _check_resolution(self, parameters: dict, batch_size: int, width: int, height: int) -> None:
        resolution = self._as_camera_values(
            parameters["resolution"],
            2,
            batch_size,
            device=torch.device("cpu"),
            dtype=torch.int64,
            name="resolution",
        )
        actual = torch.tensor([width, height], dtype=resolution.dtype).expand_as(resolution)
        if not torch.equal(resolution.cpu(), actual):
            raise ValueError(
                "All cameras in a logical batch must match the common ray resolution "
                f"[width, height]=[{width}, {height}], got {resolution.tolist()}"
            )

    @staticmethod
    def _shared_name(value: Any, batch_size: int, *, field: str) -> str:
        if isinstance(value, (list, tuple)):
            names = [getattr(item, "name", str(item)) for item in value]
            if len(names) not in (1, batch_size):
                raise ValueError(f"{field} must be shared or contain {batch_size} values, got {len(names)}")
        else:
            names = [getattr(value, "name", str(value))]
        if any(name != names[0] for name in names[1:]):
            raise NotImplementedError(
                f"gsplat camera batching requires one shared {field}; got {names}"
            )
        return names[0]

    def _shutter(self, parameters: dict, batch_size: int) -> RollingShutterType:
        name = self._shared_name(parameters.get("shutter_type", "GLOBAL"), batch_size, field="shutter_type")
        try:
            return RollingShutterType[name]
        except KeyError as exc:
            raise NotImplementedError(f"Unsupported shutter type for gsplat: {name}") from exc

    @staticmethod
    def _make_K(focal: torch.Tensor, principal: torch.Tensor, *, device, dtype) -> torch.Tensor:
        batch_size = focal.shape[0]
        K = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, 3, 3).clone()
        K[:, 0, 0] = focal[:, 0]
        K[:, 1, 1] = focal[:, 1]
        K[:, 0, 2] = principal[:, 0]
        K[:, 1, 2] = principal[:, 1]
        return K

    def _basic_intrinsics(self, intrinsics: list, batch_size: int, *, device, dtype) -> torch.Tensor:
        if len(intrinsics) != 4:
            raise ValueError("intrinsics must contain [fx, fy, cx, cy]")
        components = [
            self._as_camera_values(
                value,
                1,
                batch_size,
                device=device,
                dtype=dtype,
                name=name,
            ).squeeze(-1)
            for name, value in zip(("fx", "fy", "cx", "cy"), intrinsics)
        ]
        return torch.stack(components, dim=-1)

    def _camera_args(self, gpu_batch: Batch, *, device, dtype) -> tuple[int, int, dict[str, Any]]:
        rays = gpu_batch.rays_dir
        if rays.ndim != 4 or rays.shape[-1] != 3:
            raise ValueError(f"The gsplat renderers require image rays [B,H,W,3], got {tuple(rays.shape)}")
        batch_size = int(rays.shape[0])
        if gpu_batch.T_to_world.shape != (batch_size, 4, 4):
            raise ValueError(
                f"The gsplat renderers require camera-to-world matrices [B,4,4], "
                f"got {tuple(gpu_batch.T_to_world.shape)}"
            )
        if bool(getattr(gpu_batch, "rays_in_world_space", False)):
            raise NotImplementedError("gsplat does not support batches with rays_in_world_space=True in this tracer")

        height, width = int(rays.shape[1]), int(rays.shape[2])
        if gpu_batch.rgb_gt is not None and gpu_batch.rgb_gt.shape != (batch_size, height, width, 3):
            raise ValueError(
                f"RGB shape {tuple(gpu_batch.rgb_gt.shape)} does not match "
                f"the camera batch {(batch_size, height, width, 3)}"
            )

        c2w = gpu_batch.T_to_world.to(device=device, dtype=dtype)
        viewmats = torch.linalg.inv(c2w)
        camera_model = "pinhole"
        radial_coeffs = None
        tangential_coeffs = None
        thin_prism_coeffs = None
        ftheta_coeffs = None
        shutter = RollingShutterType.GLOBAL

        camera_fields = [
            gpu_batch.intrinsics,
            gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters,
            gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters,
            gpu_batch.intrinsics_FThetaCameraModelParameters,
        ]
        if sum(field is not None for field in camera_fields) != 1:
            raise ValueError("Exactly one supported camera-intrinsics representation must be present in the batch")

        if gpu_batch.intrinsics is not None:
            intrinsics = self._basic_intrinsics(gpu_batch.intrinsics, batch_size, device=device, dtype=dtype)
            Ks = self._make_K(intrinsics[:, :2], intrinsics[:, 2:], device=device, dtype=dtype)
        elif gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters is not None:
            parameters = gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters
            self._check_resolution(parameters, batch_size, width, height)
            focal = self._as_camera_values(
                parameters["focal_length"], 2, batch_size, device=device, dtype=dtype, name="focal_length"
            )
            principal = self._as_camera_values(
                parameters["principal_point"],
                2,
                batch_size,
                device=device,
                dtype=dtype,
                name="principal_point",
            )
            Ks = self._make_K(focal, principal, device=device, dtype=dtype)
            radial_coeffs = self._as_camera_values(
                parameters["radial_coeffs"], 6, batch_size, device=device, dtype=dtype, name="radial_coeffs"
            )
            tangential_coeffs = self._as_camera_values(
                parameters["tangential_coeffs"],
                2,
                batch_size,
                device=device,
                dtype=dtype,
                name="tangential_coeffs",
            )
            thin_prism_coeffs = self._as_camera_values(
                parameters.get("thin_prism_coeffs", [0.0] * 4),
                4,
                batch_size,
                device=device,
                dtype=dtype,
                name="thin_prism_coeffs",
            )
            shutter = self._shutter(parameters, batch_size)
        elif gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters is not None:
            parameters = gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters
            self._check_resolution(parameters, batch_size, width, height)
            focal = self._as_camera_values(
                parameters["focal_length"], 2, batch_size, device=device, dtype=dtype, name="focal_length"
            )
            principal = self._as_camera_values(
                parameters["principal_point"],
                2,
                batch_size,
                device=device,
                dtype=dtype,
                name="principal_point",
            )
            Ks = self._make_K(focal, principal, device=device, dtype=dtype)
            camera_model = "fisheye"
            radial_coeffs = self._as_camera_values(
                parameters["radial_coeffs"], 4, batch_size, device=device, dtype=dtype, name="radial_coeffs"
            )
            shutter = self._shutter(parameters, batch_size)
            if not self._warned_fisheye_max_angle and "max_angle" in parameters:
                warnings.warn(
                    "gsplat has no rasterization argument equivalent to the 3DGRUT fisheye max_angle; "
                    "the distortion coefficients and camera model are still applied.",
                    stacklevel=2,
                )
                self._warned_fisheye_max_angle = True
        else:
            parameters = gpu_batch.intrinsics_FThetaCameraModelParameters
            self._check_resolution(parameters, batch_size, width, height)
            principal = self._as_camera_values(
                parameters["principal_point"],
                2,
                batch_size,
                device=device,
                dtype=dtype,
                name="principal_point",
            )
            Ks = self._make_K(torch.ones_like(principal), principal, device=device, dtype=dtype)
            camera_model = "ftheta"
            reference_name = self._shared_name(
                parameters["reference_poly"], batch_size, field="F-theta reference_poly"
            )
            try:
                reference_poly = FThetaPolynomialType[reference_name]
            except KeyError as exc:
                raise NotImplementedError(f"Unsupported F-theta polynomial type: {reference_name}") from exc

            def shared_ftheta_values(key: str, size: int) -> tuple[float, ...]:
                values = self._as_camera_values(
                    parameters[key], size, batch_size, device=device, dtype=dtype, name=key
                )
                if not torch.equal(values, values[:1].expand_as(values)):
                    raise NotImplementedError(
                        f"gsplat F-theta batching requires shared {key}; per-camera values are not supported"
                    )
                return tuple(float(value) for value in values[0])

            max_angles = self._as_camera_values(
                parameters["max_angle"],
                1,
                batch_size,
                device=device,
                dtype=dtype,
                name="max_angle",
            )
            if not torch.equal(max_angles, max_angles[:1].expand_as(max_angles)):
                raise NotImplementedError(
                    "gsplat F-theta batching requires a shared max_angle; per-camera values are not supported"
                )
            ftheta_coeffs = FThetaCameraDistortionParameters(
                reference_poly=reference_poly,
                pixeldist_to_angle_poly=shared_ftheta_values("pixeldist_to_angle_poly", 6),
                angle_to_pixeldist_poly=shared_ftheta_values("angle_to_pixeldist_poly", 6),
                max_angle=float(max_angles[0, 0]),
                linear_cde=shared_ftheta_values("linear_cde", 3),
            )
            shutter = self._shutter(parameters, batch_size)

        has_distortion = (
            any(
                coefficients is not None and bool(torch.count_nonzero(coefficients))
                for coefficients in (radial_coeffs, tangential_coeffs, thin_prism_coeffs)
            )
            or ftheta_coeffs is not None
        )
        if not self.with_ut and (camera_model != "pinhole" or has_distortion or shutter != RollingShutterType.GLOBAL):
            raise NotImplementedError(
                "Distorted, fisheye, F-theta, and rolling-shutter cameras require render=gsplat_3dgut"
            )

        viewmats_rs = None
        if shutter != RollingShutterType.GLOBAL:
            if gpu_batch.T_to_world_end is None:
                raise ValueError(f"Rolling shutter {shutter.name} requires Batch.T_to_world_end")
            if gpu_batch.T_to_world_end.shape != (batch_size, 4, 4):
                raise ValueError(
                    f"Rolling shutter requires end poses [B,4,4], got {tuple(gpu_batch.T_to_world_end.shape)}"
                )
            viewmats_rs = torch.linalg.inv(gpu_batch.T_to_world_end.to(device=device, dtype=dtype))

        return (
            width,
            height,
            {
                "viewmats": viewmats,
                "Ks": Ks,
                "camera_model": camera_model,
                "radial_coeffs": radial_coeffs if self.with_ut else None,
                "tangential_coeffs": tangential_coeffs if self.with_ut else None,
                "thin_prism_coeffs": thin_prism_coeffs if self.with_ut else None,
                "ftheta_coeffs": ftheta_coeffs if self.with_ut else None,
                "rolling_shutter": shutter,
                "viewmats_rs": viewmats_rs,
            },
        )

    @staticmethod
    def _validate_opacity_override(
        opacity_override: torch.Tensor,
        batch_size: int,
        num_gaussians: int,
    ) -> torch.Tensor:
        if not isinstance(opacity_override, torch.Tensor):
            raise TypeError("opacity_override must be a torch.Tensor")
        expected_shapes = (
            {(num_gaussians,), (num_gaussians, 1)}
            if batch_size == 1
            else {(batch_size, num_gaussians), (batch_size, num_gaussians, 1)}
        )
        if tuple(opacity_override.shape) not in expected_shapes:
            expected = " or ".join(str(shape) for shape in sorted(expected_shapes))
            raise ValueError(
                f"opacity_override must have shape {expected} for batch size {batch_size}, "
                f"got {tuple(opacity_override.shape)}"
            )
        if batch_size == 1:
            return opacity_override.reshape(num_gaussians)
        return opacity_override.reshape(batch_size, num_gaussians)

    @staticmethod
    def _gaussian_args(
        gaussians,
        opacity_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int]:
        feature_type = getattr(gaussians.feature_type, "name", str(gaussians.feature_type)).lower()
        if feature_type != "sh":
            raise NotImplementedError(
                f"The gsplat renderers support only 3DGRUT feature_type='sh', got '{feature_type}'"
            )
        if int(gaussians.ray_feature_dim) != 3:
            raise NotImplementedError(
                f"The gsplat renderers require three-channel SH output, got ray_feature_dim={gaussians.ray_feature_dim}"
            )

        means = gaussians.get_positions()
        quats = gaussians.get_rotation()
        scales = gaussians.get_scale()
        opacities = gaussians.get_density().squeeze(-1) if opacity_override is None else opacity_override
        flat_features = gaussians.get_features()
        if flat_features.shape[1] % 3:
            raise ValueError(f"Flattened SH feature dimension must be divisible by 3, got {flat_features.shape[1]}")
        sh_coefficients = flat_features.view(flat_features.shape[0], -1, 3)
        sh_degree = int(gaussians.n_active_features)
        if (sh_degree + 1) ** 2 > sh_coefficients.shape[1]:
            raise ValueError(
                f"Active SH degree {sh_degree} needs {(sh_degree + 1) ** 2} coefficients, "
                f"but only {sh_coefficients.shape[1]} are stored"
            )
        return {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "colors": sh_coefficients,
            "sh_degree": sh_degree,
        }

    @staticmethod
    def _pair_gaussians_and_cameras(
        gaussian_args: dict[str, torch.Tensor | int],
        camera_args: dict[str, Any],
        batch_size: int,
    ) -> tuple[dict[str, torch.Tensor | int], dict[str, Any]]:
        # The pinned gsplat API treats leading dimensions as independent scenes.
        # SH coefficients stay shared by contract; geometry is expanded and is
        # materialized only by gsplat's existing contiguous conversion.
        paired_gaussians = dict(gaussian_args)
        for key in ("means", "quats", "scales"):
            value = paired_gaussians[key]
            assert isinstance(value, torch.Tensor)
            paired_gaussians[key] = value.unsqueeze(0).expand(batch_size, *value.shape)

        opacities = paired_gaussians["opacities"]
        assert isinstance(opacities, torch.Tensor)
        if opacities.ndim == 1:
            paired_gaussians["opacities"] = opacities.unsqueeze(0).expand(batch_size, -1)

        paired_cameras = {
            key: value.unsqueeze(1) if isinstance(value, torch.Tensor) else value
            for key, value in camera_args.items()
        }
        return paired_gaussians, paired_cameras

    def _ut_parameters(self):
        if not self.with_ut:
            return None
        splat = self.conf.render.splat
        return UnscentedTransformParameters(
            alpha=float(splat.ut_alpha),
            beta=float(splat.ut_beta),
            kappa=float(splat.ut_kappa),
            in_image_margin_factor=float(splat.ut_in_image_margin_factor),
            require_all_sigma_points_valid=bool(splat.ut_require_all_sigma_points_valid),
        )

    @staticmethod
    def _visibility(info: dict, num_gaussians: int, device) -> torch.Tensor:
        visibility = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        gaussian_ids = info.get("gaussian_ids")
        if gaussian_ids is not None:
            if gaussian_ids.numel():
                visibility[gaussian_ids.long().unique()] = True
            return visibility.unsqueeze(-1)

        radii = info["radii"]
        if radii.ndim >= 3 and radii.shape[-2] == num_gaussians:
            visible = (radii > 0).all(dim=-1).reshape(-1, num_gaussians).any(dim=0)
        elif radii.shape[-1] == num_gaussians:
            visible = (radii > 0).reshape(-1, num_gaussians).any(dim=0)
        else:
            raise ValueError(f"Unexpected gsplat radii shape {tuple(radii.shape)} for {num_gaussians} Gaussians")
        visibility.copy_(visible)
        return visibility.unsqueeze(-1)

    def _hit_counts(self, info: dict, pred_opacity: torch.Tensor, width: int, height: int) -> torch.Tensor:
        if self.with_eval3d:
            if not self._warned_missing_hit_counts:
                warnings.warn(
                    "gsplat does not expose native-equivalent per-pixel hit counts for eval-3D; returning zeros.",
                    stacklevel=2,
                )
                self._warned_missing_hit_counts = True
            return torch.zeros_like(pred_opacity)

        counts, _ = rasterize_num_contributing_gaussians(
            info["means2d"],
            info["conics"],
            info["opacities"],
            info["isect_offsets"],
            info["flatten_ids"],
            width,
            height,
            int(info["tile_size"]),
        )
        return counts.unsqueeze(-1).to(dtype=pred_opacity.dtype)

    def render(
        self,
        gaussians,
        gpu_batch: Batch,
        train=False,
        frame_id=0,
        opacity_override: torch.Tensor | None = None,
    ):
        paired_batch = gpu_batch.batch_size > 1
        if opacity_override is None:
            gaussian_args = self._gaussian_args(gaussians)
        else:
            opacity_override = self._validate_opacity_override(
                opacity_override, gpu_batch.batch_size, gaussians.num_gaussians
            )
            gaussian_args = self._gaussian_args(gaussians, opacity_override=opacity_override)
        means = gaussian_args["means"]
        width, height, camera_args = self._camera_args(gpu_batch, device=means.device, dtype=means.dtype)
        if paired_batch:
            gaussian_args, camera_args = self._pair_gaussians_and_cameras(
                gaussian_args, camera_args, gpu_batch.batch_size
            )

        enable_normals = bool(self.conf.render.enable_normals)
        if enable_normals and not self.with_eval3d:
            raise NotImplementedError(
                "gsplat_3dgs cannot provide the real normals requested by render.enable_normals=true"
            )

        start_event = end_event = None
        if self._timing_enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        render_mode = "RGB-d" if self.with_eval3d else "RGB+D"
        rendered, pred_opacity, info = rasterization(
            **gaussian_args,
            **camera_args,
            width=width,
            height=height,
            tile_size=int(getattr(self.gsplat_conf, "tile_size", 16)),
            near_plane=0.01,
            far_plane=1.0e10,
            radius_clip=float(self.gsplat_conf.radius_clip),
            eps2d=float(self.gsplat_conf.eps2d),
            packed=bool(self.gsplat_conf.packed),
            backgrounds=None,
            render_mode=render_mode,
            sparse_grad=bool(self.gsplat_conf.sparse_grad),
            absgrad=bool(self.gsplat_conf.absgrad),
            rasterize_mode=str(self.gsplat_conf.rasterize_mode),
            with_ut=self.with_ut,
            with_eval3d=self.with_eval3d,
            return_normals=enable_normals,
            global_z_order=bool(
                self.conf.render.splat.global_z_order if self.with_eval3d else self.gsplat_conf.global_z_order
            ),
            ut_params=self._ut_parameters(),
            renderer_config=self.renderer_config,
        )

        paired_pred_opacity = pred_opacity
        if paired_batch:
            rendered = rendered[:, 0]
            pred_opacity = pred_opacity[:, 0]

        frame_time_ms = 0.0
        if self._timing_enabled:
            end_event.record()
            end_event.synchronize()
            frame_time_ms = start_event.elapsed_time(end_event)
            self._timings = {"forward_render": frame_time_ms}
        else:
            self._timings = {}

        pred_features = rendered[..., :3].contiguous()
        pred_dist = rendered[..., 3:].contiguous()
        if not self.with_eval3d:
            ray_directions = gpu_batch.rays_dir.to(device=pred_dist.device, dtype=pred_dist.dtype)
            distance_per_z = ray_directions.norm(dim=-1, keepdim=True) / ray_directions[..., 2:].clamp_min(1.0e-8)
            pred_dist = pred_dist * distance_per_z

        if enable_normals:
            pred_normals = (
                info["normals"][:, 0].contiguous()
                if paired_batch
                else info["normals"].contiguous()
            )
        else:
            pred_normals = torch.nn.functional.normalize(torch.ones_like(pred_features), dim=-1)

        return {
            "pred_features": pred_features,
            "pred_opacity": pred_opacity.contiguous(),
            "pred_dist": pred_dist,
            "pred_normals": pred_normals,
            "hits_count": (
                self._hit_counts(info, paired_pred_opacity, width, height)[:, 0]
                if paired_batch
                else self._hit_counts(info, pred_opacity, width, height)
            ),
            "frame_time_ms": frame_time_ms,
            "mog_visibility": self._visibility(info, gaussians.num_gaussians, means.device),
        }
