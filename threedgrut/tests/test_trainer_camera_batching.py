# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import torch
from omegaconf import OmegaConf

import threedgrut.trainer as trainer_module
from threedgrut.datasets.protocols import Batch
from threedgrut.trainer import Trainer3DGRUT


class _Counter:
    def __init__(self):
        self.calls = 0

    def step(self, *args, **kwargs):
        self.calls += 1

    def zero_grad(self):
        self.zero_grad_calls += 1


class _Optimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self):
        self.step_calls += 1

    def zero_grad(self):
        self.zero_grad_calls += 1
        self.parameter.grad = None


class _Profiler:
    enabled = False

    def __init__(self):
        self.starts = 0
        self.ends = 0

    def start(self):
        self.starts += 1

    def end(self):
        self.ends += 1


class _Strategy:
    requires_per_view_position_gradients = True

    def __init__(self):
        self.pre_backward_calls = 0
        self.post_backward_calls = 0
        self.post_optimizer_calls = 0
        self.gradient_buffer_calls = []

    def pre_backward(self, **kwargs):
        self.pre_backward_calls += 1

    def post_backward(self, **kwargs):
        self.post_backward_calls += 1
        return False

    def post_optimizer_step(self, **kwargs):
        self.post_optimizer_calls += 1
        return False

    def should_update_gradient_buffer(self, step):
        return True

    def update_gradient_buffer(self, sensor_position, params_grad=None):
        self.gradient_buffer_calls.append((sensor_position.clone(), params_grad.clone()))


class _FeatureDecoder:
    def __init__(self):
        self.ema_calls = 0

    def ema_update(self, step):
        self.ema_calls += 1


class _Model:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor([[1.0]]))
        self.positions = self.parameter
        self.density = torch.zeros(1, 1)
        self.optimizer = _Optimizer(self.parameter)
        self.renderer = SimpleNamespace(timings={})
        self.background = None
        self.progressive_training = False
        self.scheduler_calls = 0
        self.build_acc_calls = 0
        self.forward_calls = 0

    def __call__(self, batch, train=False, frame_id=0):
        self.forward_calls += 1
        batch_size, height, width = batch.rays_dir.shape[:3]
        features = self.parameter * torch.ones(batch_size, height, width, 3)
        scalar_image = torch.zeros(batch_size, height, width, 1)
        return {
            "pred_features": features,
            "pred_opacity": scalar_image,
            "pred_dist": scalar_image,
            "pred_normals": torch.zeros_like(features),
            "hits_count": scalar_image,
            "frame_time_ms": 0.0,
            "mog_visibility": torch.ones_like(self.density, dtype=torch.bool),
        }

    def scheduler_step(self, step):
        self.scheduler_calls += 1

    def build_acc(self, rebuild=True):
        self.build_acc_calls += 1


def _logical_batch() -> Batch:
    return Batch(
        rays_ori=torch.zeros(2, 2, 3, 3),
        rays_dir=torch.ones(2, 2, 3, 3),
        T_to_world=torch.eye(4).repeat(2, 1, 1),
        rgb_gt=torch.zeros(2, 2, 3, 3),
    )


def test_one_logical_batch_advances_every_training_component_once(monkeypatch):
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.conf = OmegaConf.create(
        {
            "validate_first": False,
            "model": {"bvh_update_frequency": 1, "nht_decoder": {"center_ray_encoding": False}},
            "checkpoint": {"iterations": []},
            "with_viser_gui": False,
            "with_gui": False,
        }
    )
    trainer.global_step = 0
    trainer.val_frequency = 100
    trainer.scene_extent = 1.0
    trainer.tracking = SimpleNamespace(writer=None)
    trainer._color_refine_start_step = 100
    trainer._in_color_refine = False
    trainer._distillation_start_step = -1
    trainer.model = _Model()
    trainer.strategy = _Strategy()
    trainer.train_dataset = SimpleNamespace(get_gpu_batch_with_intrinsics=lambda batch: batch)
    trainer.feature_decoder = _FeatureDecoder()
    trainer.feature_decoder_optimizer = _Counter()
    trainer.feature_decoder_optimizer.zero_grad_calls = 0
    trainer.feature_decoder_scheduler = _Counter()
    trainer.post_processing = None
    trainer.post_processing_optimizers = None
    trainer.post_processing_schedulers = None

    trainer.get_losses = MethodType(
        lambda self, batch, outputs: {"total_loss": outputs["pred_features"].mean()}, trainer
    )
    trainer.get_metrics = MethodType(lambda self, *args, **kwargs: {}, trainer)
    trainer.log_training_iter = MethodType(lambda self, *args, **kwargs: None, trainer)
    trainer.save_checkpoint = MethodType(lambda self, *args, **kwargs: None, trainer)
    monkeypatch.setattr(trainer_module, "apply_feature_decoder", lambda decoder, outputs, *args, **kwargs: outputs)
    monkeypatch.setattr(trainer_module, "apply_background", lambda background, outputs, *args, **kwargs: outputs)

    profilers = {name: _Profiler() for name in ("step_total", "inference", "backward", "build_as", "feature_decoder")}
    metrics = []
    trainer.run_train_iter(0, _logical_batch(), profilers, metrics, trainer.conf)

    assert trainer.global_step == 1
    assert trainer.model.optimizer.step_calls == 1
    assert trainer.model.optimizer.zero_grad_calls == 1
    assert trainer.model.scheduler_calls == 1
    assert trainer.feature_decoder_optimizer.calls == 1
    assert trainer.feature_decoder_optimizer.zero_grad_calls == 1
    assert trainer.feature_decoder_scheduler.calls == 1
    assert trainer.feature_decoder.ema_calls == 1
    assert trainer.strategy.pre_backward_calls == 1
    assert trainer.strategy.post_backward_calls == 1
    assert trainer.strategy.post_optimizer_calls == 1
    assert len(trainer.strategy.gradient_buffer_calls) == 2
    assert trainer.model.forward_calls == 1
    assert trainer.model.build_acc_calls == 1


def test_batch_strategy_and_post_processing_guards():
    mcmc = OmegaConf.create(
        {"batch_size": 2, "strategy": {"method": "MCMCStrategy"}, "post_processing": {"method": None}}
    )
    Trainer3DGRUT._validate_batching_configuration(mcmc)

    gs = OmegaConf.create({"batch_size": 2, "strategy": {"method": "GSStrategy"}, "post_processing": {"method": None}})
    Trainer3DGRUT._validate_batching_configuration(gs)

    post_processing = OmegaConf.create(
        {"batch_size": 2, "strategy": {"method": "MCMCStrategy"}, "post_processing": {"method": "ppisp"}}
    )
    try:
        Trainer3DGRUT._validate_batching_configuration(post_processing)
    except ValueError as error:
        assert "camera/frame controllers are single-view" in str(error)
    else:
        raise AssertionError("Post-processing with B>1 must fail explicitly")
