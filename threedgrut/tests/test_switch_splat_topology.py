# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.strategy.gs import GSStrategy
from threedgrut.strategy.mcmc import MCMCStrategy


class _SyntheticGaussianModel(torch.nn.Module):
    def __init__(self, num_gaussians: int, device: torch.device):
        super().__init__()
        self.device = device
        self.positions = torch.nn.Parameter(
            torch.arange(num_gaussians * 3, dtype=torch.float32, device=device).reshape(num_gaussians, 3) / 100.0
        )
        self.rotation = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_gaussians, 1))
        self.scale = torch.nn.Parameter(torch.full((num_gaussians, 3), 0.1, device=device))
        self.density = torch.nn.Parameter(torch.full((num_gaussians, 1), 0.8, device=device))
        self.features = torch.nn.Parameter(
            torch.arange(num_gaussians * 2, dtype=torch.float32, device=device).reshape(num_gaussians, 2) / 50.0
        )
        self.switch_logits = torch.nn.Parameter(
            torch.arange(num_gaussians * 5, dtype=torch.float32, device=device).reshape(num_gaussians, 5) / 10.0
        )
        self.scale_activation = lambda value: value
        self.scale_activation_inv = lambda value: value
        self.optimizer = torch.optim.Adam(
            [{"params": [parameter], "name": name} for name, parameter in self.named_parameters()],
            lr=1e-2,
        )

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_scale(self) -> torch.Tensor:
        return self.scale

    def get_density(self) -> torch.Tensor:
        return self.density


def _populate_optimizer_state(model: _SyntheticGaussianModel) -> None:
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)


def _optimizer_group(model: _SyntheticGaussianModel, name: str) -> dict:
    return next(group for group in model.optimizer.param_groups if group["name"] == name)


def _assert_gaussian_optimizer_state_shapes(model: _SyntheticGaussianModel) -> None:
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        assert getattr(model, group["name"]) is parameter
        assert parameter.shape[0] == model.num_gaussians
        for key, value in model.optimizer.state[parameter].items():
            if key != "step":
                assert value.shape == parameter.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GS split currently allocates its samples on CUDA")
def test_gs_clone_split_and_prune_preserve_switch_logits_and_optimizer_state():
    model = _SyntheticGaussianModel(4, torch.device("cuda"))
    with torch.no_grad():
        model.scale[:] = torch.tensor([[0.10], [0.40], [0.10], [0.20]], device=model.device)
        model.density[:] = torch.tensor([[0.90], [0.70], [0.10], [0.80]], device=model.device)
    _populate_optimizer_state(model)

    strategy = GSStrategy.__new__(GSStrategy)
    strategy.model = model
    strategy.split_n_gaussians = 2
    strategy.relative_size_threshold = 0.25
    strategy.clone_grad_threshold = 0.5
    strategy.split_grad_threshold = 0.5
    strategy.prune_density_threshold = 0.5
    strategy.conf = OmegaConf.create({"strategy": {"print_stats": False}})
    strategy.densify_grad_norm_accum = torch.zeros((4, 1), device=model.device)
    strategy.densify_grad_norm_denom = torch.zeros((4, 1), dtype=torch.int32, device=model.device)

    original_logits = model.switch_logits.detach().clone()
    densify_gradient = torch.tensor([1.0, 1.0, 0.0, 0.0], device=model.device)

    strategy.clone_gaussians(densify_gradient, scene_extent=1.0)
    torch.testing.assert_close(
        model.switch_logits,
        torch.cat((original_logits, original_logits[[0]])),
    )
    _assert_gaussian_optimizer_state_shapes(model)

    strategy.split_gaussians(densify_gradient, scene_extent=1.0)
    after_split = torch.cat((original_logits[[0, 2, 3, 0]], original_logits[[1, 1]]))
    torch.testing.assert_close(model.switch_logits, after_split)
    _assert_gaussian_optimizer_state_shapes(model)

    valid_mask = model.get_density().squeeze(-1) >= strategy.prune_density_threshold
    expected_after_prune = after_split[valid_mask]
    strategy.prune_gaussians_opacity()
    torch.testing.assert_close(model.switch_logits, expected_after_prune)
    _assert_gaussian_optimizer_state_shapes(model)


def test_mcmc_relocation_and_addition_preserve_switch_logits_and_optimizer_state():
    model = _SyntheticGaussianModel(20, torch.device("cpu"))
    with torch.no_grad():
        model.density[[1, 3]] = 1e-4
    _populate_optimizer_state(model)

    strategy = MCMCStrategy.__new__(MCMCStrategy)
    strategy.model = model
    strategy.conf = OmegaConf.create(
        {
            "strategy": {
                "opacity_threshold": 0.005,
                "print_stats": False,
                "add": {"max_n_gaussians": 1000},
            }
        }
    )

    def sample_new_gaussians(self, num_gaussians, valid_indices=None):
        if valid_indices is not None:
            assert num_gaussians == 2
            sampled = torch.tensor([2, 4], dtype=torch.long)
        else:
            assert num_gaussians == 1
            sampled = torch.tensor([5], dtype=torch.long)
        densities = torch.full((num_gaussians, 1), 0.75)
        scales = torch.full((num_gaussians, 3), 0.15)
        return sampled, densities, scales

    strategy.sample_new_gaussians = MethodType(sample_new_gaussians, strategy)

    original_logits = model.switch_logits.detach().clone()
    strategy.relocate_gaussians()
    expected_after_relocation = original_logits.clone()
    expected_after_relocation[[1, 3]] = original_logits[[2, 4]]
    torch.testing.assert_close(model.switch_logits, expected_after_relocation)
    _assert_gaussian_optimizer_state_shapes(model)

    strategy.add_new_gaussians()
    expected_after_add = torch.cat((expected_after_relocation, expected_after_relocation[[5]]))
    torch.testing.assert_close(model.switch_logits, expected_after_add)
    _assert_gaussian_optimizer_state_shapes(model)

    switch_group = _optimizer_group(model, "switch_logits")
    switch_parameter = switch_group["params"][0]
    for key, value in model.optimizer.state[switch_parameter].items():
        if key != "step":
            torch.testing.assert_close(value[-1], torch.zeros_like(value[-1]))
