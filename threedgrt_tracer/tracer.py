# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
from enum import IntEnum

import torch
import torch.utils.cpp_extension

from threedgrut.datasets.protocols import Batch, slice_batch
from threedgrut.model.features import Features
from threedgrut.utils.timer import CudaTimer

logger = logging.getLogger(__name__)

_BATCH_IMAGE_OUTPUTS = ("pred_features", "pred_opacity", "pred_dist", "pred_normals", "hits_count")
_FIRST_INTERNAL_FRAME_ID = 1 << 31
_FRAME_ID_LIMIT = 1 << 32


# ----------------------------------------------------------------------------
#

_3dgrt_plugin = None


def load_3dgrt_plugin(conf):
    global _3dgrt_plugin
    if _3dgrt_plugin is None:
        try:
            from . import lib3dgrt_cc as tdgrt  # type: ignore
        except ImportError:
            from .setup_3dgrt import setup_3dgrt

            tdgrt = setup_3dgrt(conf)
        _3dgrt_plugin = tdgrt


# ----------------------------------------------------------------------------
#
class Tracer:
    class _Autograd(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            tracer_wrapper,
            frame_id,
            ray_to_world,
            ray_ori,
            ray_dir,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph,
            render_opts,
            sph_degree,
            min_transmittance,
        ):
            particle_density = torch.concat([mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1)
            ray_features, ray_density, ray_hit_distance, ray_normals, hits_count, mog_visibility = tracer_wrapper.trace(
                frame_id,
                ray_to_world,
                ray_ori,
                ray_dir,
                particle_density,
                mog_sph,
                render_opts,
                sph_degree,
                min_transmittance,
            )
            ctx.save_for_backward(
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_features,
                ray_density,
                ray_hit_distance,
                ray_normals,
                particle_density,
                mog_sph,
            )
            ctx.frame_id = frame_id
            ctx.render_opts = render_opts
            ctx.sph_degree = sph_degree
            ctx.min_transmittance = min_transmittance
            ctx.tracer_wrapper = tracer_wrapper
            return (
                ray_features.float(),  # always fp32 to caller; fp16 saved in ctx for trace_bwd
                ray_density,
                ray_hit_distance[:, :, :, 0:1],  # return only the hit distance
                ray_normals,
                hits_count,
                mog_visibility,
            )

        @staticmethod
        def backward(
            ctx,
            ray_features_grd,
            ray_density_grd,
            ray_hit_distance_grd,
            ray_normals_grd,
            ray_hits_count_grd_UNUSED,
            mog_visibility_grd_UNUSED,
        ):
            (
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_features,
                ray_density,
                ray_hit_distance,
                ray_normals,
                particle_density,
                mog_sph,
            ) = ctx.saved_variables
            frame_id = ctx.frame_id
            particle_density_grd, mog_sph_grd = ctx.tracer_wrapper.trace_bwd(
                frame_id,
                ray_to_world,
                ray_ori,
                ray_dir,
                ray_features,
                ray_density,
                ray_hit_distance,
                ray_normals,
                particle_density,
                mog_sph,
                ray_features_grd,
                ray_density_grd,
                ray_hit_distance_grd,
                ray_normals_grd,
                ctx.render_opts,
                ctx.sph_degree,
                ctx.min_transmittance,
            )
            mog_pos_grd, mog_dns_grd, mog_rot_grd, mog_scl_grd, _ = torch.split(
                particle_density_grd, [3, 1, 4, 3, 1], dim=1
            )
            return (
                None,
                None,
                None,
                None,
                None,
                mog_pos_grd,
                mog_rot_grd,
                mog_scl_grd,
                mog_dns_grd,
                mog_sph_grd,
                None,
                None,
                None,
            )

    class RenderOpts(IntEnum):
        NONE = 0
        DEFAULT = NONE

    def __init__(self, conf):
        self.device = "cuda"
        self.conf = conf
        self.num_update_bvh = 0
        self.feature_transform_type = Features(conf).transform_type

        logger.info(f'🔆 Creating Optix tracing pipeline.. Using CUDA path: "{torch.utils.cpp_extension.CUDA_HOME}"')
        torch.zeros(1, device=self.device)  # Create a dummy tensor to force cuda context init
        load_3dgrt_plugin(conf)

        self.tracer_wrapper = _3dgrt_plugin.OptixTracer(
            os.path.dirname(__file__),
            torch.utils.cpp_extension.CUDA_HOME,
            self.conf.render.pipeline_type,
            self.conf.render.backward_pipeline_type,
            self.conf.render.primitive_type,
            self.conf.render.particle_kernel_degree,
            self.conf.render.particle_kernel_min_response,
            self.conf.render.particle_kernel_max_alpha,
            self.conf.render.particle_kernel_density_clamping,
            self.conf.render.particle_radiance_sph_degree,
            self.conf.render.enable_normals,
            self.conf.render.enable_hitcounts,
        )

        self.frame_timer = CudaTimer() if self.conf.render.enable_kernel_timings else None
        self.timings = {}
        self._next_subview_frame_id = _FIRST_INTERNAL_FRAME_ID

    def build_acc(self, gaussians, rebuild=True):
        with torch.cuda.nvtx.range(f"build-bvh-full-build-{rebuild}"):
            allow_bvh_update = (
                self.conf.render.max_consecutive_bvh_update > 1
            ) and not self.conf.render.particle_kernel_density_clamping
            rebuild_bvh = (
                rebuild
                or self.conf.render.particle_kernel_density_clamping
                or self.num_update_bvh >= self.conf.render.max_consecutive_bvh_update
            )
            self.tracer_wrapper.build_bvh(
                gaussians.positions.view(-1, 3).contiguous(),
                gaussians.rotation_activation(gaussians.rotation).view(-1, 4).contiguous(),
                gaussians.scale_activation(gaussians.scale).view(-1, 3).contiguous(),
                gaussians.density_activation(gaussians.density).view(-1, 1).contiguous(),
                rebuild_bvh,
                allow_bvh_update,
            )
            self.num_update_bvh = 0 if rebuild_bvh else self.num_update_bvh + 1

    def _allocate_subview_frame_ids(self, batch_size: int) -> range:
        end = self._next_subview_frame_id + batch_size
        if end > _FRAME_ID_LIMIT:
            raise RuntimeError("Native subview frame IDs exhausted the uint32 range")
        frame_ids = range(self._next_subview_frame_id, end)
        self._next_subview_frame_id = end
        return frame_ids

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
        return opacity_override

    def render(
        self,
        gaussians,
        gpu_batch: Batch,
        train=False,
        frame_id=0,
        opacity_override: torch.Tensor | None = None,
    ):
        if gpu_batch.batch_size == 1:
            if opacity_override is not None:
                opacity_override = self._validate_opacity_override(
                    opacity_override, gpu_batch.batch_size, gaussians.num_gaussians
                )
                return self._render_single(
                    gaussians,
                    gpu_batch,
                    train=train,
                    frame_id=frame_id,
                    opacity_override=opacity_override,
                )
            return self._render_single(gaussians, gpu_batch, train=train, frame_id=frame_id)

        # Native 3DGRT has one camera transform per call; all subviews share the existing GAS.
        if opacity_override is None:
            outputs = [
                self._render_single(
                    gaussians,
                    slice_batch(gpu_batch, index),
                    train=train,
                    frame_id=subview_frame_id,
                )
                for index, subview_frame_id in enumerate(self._allocate_subview_frame_ids(gpu_batch.batch_size))
            ]
        else:
            opacity_override = self._validate_opacity_override(
                opacity_override, gpu_batch.batch_size, gaussians.num_gaussians
            )
            outputs = [
                self._render_single(
                    gaussians,
                    slice_batch(gpu_batch, index),
                    train=train,
                    frame_id=subview_frame_id,
                    opacity_override=opacity_override[index],
                )
                for index, subview_frame_id in enumerate(self._allocate_subview_frame_ids(gpu_batch.batch_size))
            ]

        frame_time_ms = sum(output["frame_time_ms"] for output in outputs)
        if self.frame_timer is not None:
            self.timings["forward_render"] = frame_time_ms

        first_visibility = outputs[0]["mog_visibility"]
        return {
            **{key: torch.cat([output[key] for output in outputs], dim=0) for key in _BATCH_IMAGE_OUTPUTS},
            "frame_time_ms": frame_time_ms,
            "mog_visibility": torch.stack(
                [output["mog_visibility"].bool() for output in outputs], dim=0
            )
            .any(dim=0)
            .to(dtype=first_visibility.dtype),
        }

    def _render_single(
        self,
        gaussians,
        gpu_batch: Batch,
        train=False,
        frame_id=0,
        opacity_override: torch.Tensor | None = None,
    ):
        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):

            if self.frame_timer is not None:
                self.frame_timer.start()

            opacity = gaussians.get_density() if opacity_override is None else opacity_override
            if opacity.ndim == 1:
                opacity = opacity.unsqueeze(-1)

            pred_features, pred_opacity, pred_dist, pred_normals, hits_count, mog_visibility = Tracer._Autograd.apply(
                self.tracer_wrapper,
                frame_id,
                gpu_batch.T_to_world.contiguous(),
                gpu_batch.rays_ori.contiguous(),
                gpu_batch.rays_dir.contiguous(),
                gaussians.positions.contiguous(),
                gaussians.get_rotation().contiguous(),
                gaussians.get_scale().contiguous(),
                opacity.contiguous(),
                gaussians.get_features().contiguous(),
                Tracer.RenderOpts.DEFAULT,
                gaussians.n_active_features,
                self.conf.render.min_transmittance,
            )

            if self.frame_timer is not None:
                self.frame_timer.end()

        if self.frame_timer is not None:
            self.timings["forward_render"] = self.frame_timer.timing()

        return {
            "pred_features": pred_features,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": torch.nn.functional.normalize(pred_normals, dim=3),
            "hits_count": hits_count,
            "frame_time_ms": self.frame_timer.timing() if self.frame_timer is not None else 0.0,
            "mog_visibility": mog_visibility,
        }
