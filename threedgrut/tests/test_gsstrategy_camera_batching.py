# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.datasets.protocols import Batch, slice_batch
from threedgrut.strategy.gs import GSStrategy
from threedgrut.trainer import Trainer3DGRUT, _slice_camera_outputs

_PARAMETER_NAMES = ("positions", "rotation", "scale", "density", "albedo", "specular")


class _SyntheticGaussianModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.positions = torch.nn.Parameter(torch.randn(3, 3, generator=generator, dtype=torch.float64) * 0.08)
        self.rotation = torch.nn.Parameter(torch.randn(3, 4, generator=generator, dtype=torch.float64) * 0.05)
        self.scale = torch.nn.Parameter(torch.randn(3, 3, generator=generator, dtype=torch.float64) * 0.04)
        self.density = torch.nn.Parameter(torch.randn(3, 1, generator=generator, dtype=torch.float64) * 0.03)
        self.albedo = torch.nn.Parameter(torch.randn(3, 3, generator=generator, dtype=torch.float64) * 0.06)
        self.specular = torch.nn.Parameter(torch.randn(3, 2, 3, generator=generator, dtype=torch.float64) * 0.02)
        self.forward_calls = 0

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        camera = batch.T_to_world[:, :3, 3].to(self.positions)
        gaussian_weight = torch.tensor([0.7, 1.1, 1.4], dtype=self.positions.dtype)
        position_signal = torch.einsum(
            "nc,bc,n->b",
            self.positions,
            camera + torch.tensor([1.2, -0.7, 0.5], dtype=self.positions.dtype),
            gaussian_weight,
        )
        camera_factor = 1.0 + 0.1 * camera[:, 0] - 0.05 * camera[:, 1]
        rotation_signal = self.rotation.sum() * camera_factor
        scale_signal = self.scale.sum() * (1.0 - 0.08 * camera[:, 2])
        density_signal = self.density.sum() * (1.0 + 0.04 * camera[:, 1])
        albedo_signal = self.albedo.sum() * (1.0 + 0.03 * camera[:, 0])
        specular_signal = self.specular.sum() * (1.0 - 0.02 * camera[:, 2])
        rgb = torch.stack(
            (
                0.07 * position_signal + 0.03 * rotation_signal + 0.02 * density_signal,
                -0.04 * position_signal + 0.02 * scale_signal + 0.015 * albedo_signal,
                0.05 * position_signal + 0.01 * specular_signal + 0.01 * density_signal,
            ),
            dim=-1,
        )
        return {"pred_features": rgb[:, None, None, :].expand(-1, 2, 3, -1)}


def _batch(batch_size: int = 2, device: str = "cpu") -> Batch:
    poses = torch.eye(4, dtype=torch.float64, device=device).repeat(batch_size, 1, 1)
    translations = torch.tensor([[0.2, -0.3, 0.5], [-0.6, 0.4, 0.9]], dtype=torch.float64, device=device)[:batch_size]
    poses[:, :3, 3] = translations
    rays = torch.zeros(batch_size, 2, 3, 3, dtype=torch.float64, device=device)
    target = torch.linspace(-0.15, 0.2, batch_size * 2 * 3 * 3, dtype=torch.float64, device=device)
    return Batch(
        rays_ori=rays,
        rays_dir=rays + 1,
        T_to_world=poses,
        rgb_gt=target.reshape(batch_size, 2, 3, 3),
    )


def _losses(self, batch: Batch, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    image_loss = (outputs["pred_features"] - batch.rgb_gt).square().mean()
    regularizer = 0.013 * sum(
        getattr(self.model, name).square().mean() for name in _PARAMETER_NAMES if name != "positions"
    )
    return {
        "total_loss": image_loss + regularizer,
        "image_loss": image_loss,
        "regularizer_loss": regularizer,
    }


def _strategy(model: _SyntheticGaussianModel) -> GSStrategy:
    strategy = GSStrategy.__new__(GSStrategy)
    strategy.conf = OmegaConf.create(
        {"strategy": {"densify": {"end_iteration": 100}}, "model": {"density_activation": "sigmoid"}}
    )
    strategy.model = model
    strategy._suspended = False
    strategy.densify_grad_norm_accum = torch.zeros(model.positions.shape[0], 1, dtype=model.positions.dtype)
    strategy.densify_grad_norm_denom = torch.zeros(model.positions.shape[0], 1, dtype=torch.int32)
    return strategy


def _trainer(model: _SyntheticGaussianModel, strategy: GSStrategy) -> Trainer3DGRUT:
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.model = model
    trainer.strategy = strategy
    trainer.get_losses = MethodType(_losses, trainer)
    return trainer


def _parameter_grads(model: _SyntheticGaussianModel) -> dict[str, torch.Tensor]:
    return {name: getattr(model, name).grad.detach().clone() for name in _PARAMETER_NAMES}


def test_batch_size_one_keeps_original_gradient_buffer_path():
    model = _SyntheticGaussianModel()
    strategy = _strategy(model)
    batch = _batch(batch_size=1)
    grad = torch.tensor([[0.2, -0.1, 0.3], [0.0, 0.0, 0.0], [-0.4, 0.2, 0.1]], dtype=model.positions.dtype)
    model.positions.grad = grad.clone()
    mask = (grad != 0).max(dim=1)[0]
    distance = (model.positions.detach()[mask] - batch.T_to_world[0, :3, 3]).norm(dim=1, keepdim=True)
    expected = torch.zeros_like(strategy.densify_grad_norm_accum)
    expected[mask] = torch.norm(grad[mask] * distance, dim=-1, keepdim=True) / 2

    supplied_gradients = []
    original_update = strategy.update_gradient_buffer

    def record_update(sensor_position, params_grad=None):
        supplied_gradients.append(params_grad)
        original_update(sensor_position, params_grad)

    strategy.update_gradient_buffer = record_update
    strategy._post_backward(1, scene_extent=1.0, train_dataset=None, batch=batch)

    assert supplied_gradients == [None]
    torch.testing.assert_close(strategy.densify_grad_norm_accum, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        strategy.densify_grad_norm_denom,
        mask.to(strategy.densify_grad_norm_denom.dtype).unsqueeze(-1),
        rtol=0.0,
        atol=0.0,
    )


def test_batched_backward_matches_mean_gradient_and_sequential_gs_buffers():
    batch = _batch()
    initial = _SyntheticGaussianModel().state_dict()

    full_model = _SyntheticGaussianModel()
    full_model.load_state_dict(initial)
    full_trainer = _trainer(full_model, _strategy(full_model))
    full_outputs = full_model(batch)
    full_losses = full_trainer.get_losses(batch, full_outputs)
    full_losses["total_loss"].backward()
    expected_grads = _parameter_grads(full_model)

    batched_model = _SyntheticGaussianModel()
    batched_model.load_state_dict(initial)
    batched_strategy = _strategy(batched_model)
    observed_sensors = []
    original_update = batched_strategy.update_gradient_buffer

    def record_update(sensor_position, params_grad=None):
        observed_sensors.append(sensor_position.detach().clone())
        original_update(sensor_position, params_grad)

    batched_strategy.update_gradient_buffer = record_update
    batched_trainer = _trainer(batched_model, batched_strategy)
    batched_outputs = batched_model(batch)
    per_view_losses, logged_losses = batched_trainer._prepare_per_view_losses(batch, batched_outputs)

    for name, expected in full_losses.items():
        torch.testing.assert_close(logged_losses[name], expected.detach(), rtol=1e-12, atol=1e-12)

    batched_trainer._backward_per_view_position_gradients(1, per_view_losses)
    actual_grads = _parameter_grads(batched_model)
    for name in _PARAMETER_NAMES:
        assert torch.isfinite(actual_grads[name]).all()
        assert torch.count_nonzero(actual_grads[name]) > 0
        torch.testing.assert_close(actual_grads[name], expected_grads[name], rtol=1e-11, atol=1e-12)

    assert batched_model.forward_calls == 1
    assert len(observed_sensors) == batch.batch_size
    torch.testing.assert_close(torch.stack(observed_sensors), batch.T_to_world[:, :3, 3])

    sequential_model = _SyntheticGaussianModel()
    sequential_model.load_state_dict(initial)
    sequential_strategy = _strategy(sequential_model)
    sequential_trainer = _trainer(sequential_model, sequential_strategy)
    for view_index in range(batch.batch_size):
        view_batch = slice_batch(batch, view_index)
        view_losses = sequential_trainer.get_losses(view_batch, sequential_model(view_batch))
        view_losses["total_loss"].backward()
        sequential_strategy.update_gradient_buffer(view_batch.T_to_world[0, :3, 3])
        for parameter in sequential_model.parameters():
            parameter.grad = None

    torch.testing.assert_close(
        batched_strategy.densify_grad_norm_accum,
        sequential_strategy.densify_grad_norm_accum,
        rtol=1e-11,
        atol=1e-12,
    )
    torch.testing.assert_close(
        batched_strategy.densify_grad_norm_denom,
        sequential_strategy.densify_grad_norm_denom,
        rtol=0.0,
        atol=0.0,
    )

    accum_before = batched_strategy.densify_grad_norm_accum.clone()
    denom_before = batched_strategy.densify_grad_norm_denom.clone()
    batched_strategy._post_backward(1, scene_extent=1.0, train_dataset=None, batch=batch)
    torch.testing.assert_close(batched_strategy.densify_grad_norm_accum, accum_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(batched_strategy.densify_grad_norm_denom, denom_before, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused SSIM requires CUDA")
def test_same_resolution_per_view_loss_mean_matches_full_batch_loss():
    device = "cuda"
    batch_size, height, width = 2, 16, 17
    generator = torch.Generator(device=device).manual_seed(19)
    mask = torch.ones(batch_size, height, width, 1, device=device)
    mask[1, :4, :6] = 0
    batch = Batch(
        rays_ori=torch.zeros(batch_size, height, width, 3, device=device),
        rays_dir=torch.ones(batch_size, height, width, 3, device=device),
        T_to_world=torch.eye(4, device=device).repeat(batch_size, 1, 1),
        rgb_gt=torch.rand(batch_size, height, width, 3, generator=generator, device=device),
        mask=mask,
    )
    outputs = {
        "pred_features": torch.rand(
            batch_size, height, width, 3, generator=generator, device=device, requires_grad=True
        )
    }

    class _LossModel:
        def __init__(self):
            self.density = torch.tensor([[0.2], [-0.4]], device=device, requires_grad=True)
            self.scale = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1]], device=device, requires_grad=True)

        def get_density(self):
            return self.density

        def get_scale(self):
            return self.scale

    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.device = device
    trainer.model = _LossModel()
    trainer._in_color_refine = False
    trainer.conf = OmegaConf.create(
        {
            "loss": {
                "use_l1": True,
                "lambda_l1": 0.8,
                "use_l2": True,
                "lambda_l2": 0.1,
                "use_ssim": True,
                "lambda_ssim": 0.2,
                "use_opacity": True,
                "lambda_opacity": 0.03,
                "use_scale": True,
                "lambda_scale": 0.04,
            }
        }
    )

    full_losses = trainer.get_losses(batch, outputs)
    mean_losses = {}
    for view_index in range(batch_size):
        view_losses = trainer.get_losses(
            slice_batch(batch, view_index),
            _slice_camera_outputs(outputs, view_index, batch_size),
        )
        for name, value in view_losses.items():
            mean_losses[name] = mean_losses.get(name, 0) + value / batch_size

    for name, expected in full_losses.items():
        torch.testing.assert_close(mean_losses[name], expected, rtol=2e-6, atol=2e-7)
