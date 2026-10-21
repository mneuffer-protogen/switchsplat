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

import torch
from fused_ssim import fused_ssim


@torch.cuda.nvtx.range("l1_loss")
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


@torch.cuda.nvtx.range("l2_loss")
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


@torch.cuda.nvtx.range("ssim")
def ssim(img1, img2, window_size=11, size_average=True):
    # predicted_image, gt_image: [BS, CH, H, W], predicted_image is differentiable
    return fused_ssim(img1, img2, padding="valid")


def switch_splat_mode_a_loss(logits: torch.Tensor) -> torch.Tensor:
    base_probability = torch.softmax(logits, dim=1)[:, 0].clamp_min(1e-8)
    return -torch.log(base_probability).mean()


def switch_splat_spatial_loss(
    positions: torch.Tensor,
    opacity: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    weights = opacity[:, None] * torch.softmax(logits, dim=1)[:, 1:]
    normalized_weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

    state2_weights = weights[:, 1::2]
    state2_weights = state2_weights / (state2_weights.sum(dim=0, keepdim=True) + 1e-8)
    slot_centroids = state2_weights.t() @ positions
    centroids = slot_centroids.repeat_interleave(2, dim=0)

    position_sq = positions.square().sum(dim=-1, keepdim=True)
    centroid_sq = centroids.square().sum(dim=-1)
    distances_sq = position_sq - 2.0 * positions @ centroids.t() + centroid_sq
    return (normalized_weights * distances_sq).sum(dim=0).mean()
