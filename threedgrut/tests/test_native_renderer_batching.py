# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import threedgrt_tracer.tracer as grt_module
import threedgut_tracer.tracer as gut_module
from threedgrut.datasets.protocols import Batch
from threedgrut.model.switch_splat_controller import SwitchSplatController

IMAGE_OUTPUTS = ("pred_features", "pred_opacity", "pred_dist", "pred_normals", "hits_count")
NATIVE_MODULES = pytest.mark.parametrize("renderer_module", [grt_module, gut_module])


def _make_batch(batch_size: int) -> Batch:
    height, width = 2, 3
    transforms = torch.eye(4).repeat(batch_size, 1, 1)
    transforms[:, 0, 3] = torch.arange(batch_size, dtype=torch.float32)
    return Batch(
        rays_ori=torch.zeros(batch_size, height, width, 3),
        rays_dir=torch.ones(batch_size, height, width, 3),
        T_to_world=transforms,
        camera_idx=torch.arange(batch_size),
        frame_idx=torch.arange(10, 10 + batch_size),
    )


def _make_output(view_index: int) -> dict[str, torch.Tensor | float]:
    value = float(view_index + 1)
    return {
        "pred_features": torch.full((1, 2, 3, 3), value),
        "pred_opacity": torch.full((1, 2, 3, 1), value + 10),
        "pred_dist": torch.full((1, 2, 3, 1), value + 20),
        "pred_normals": torch.full((1, 2, 3, 3), value + 30),
        "hits_count": torch.full((1, 2, 3, 1), view_index + 40, dtype=torch.int32),
        "frame_time_ms": 1.25 * value,
        "mog_visibility": torch.tensor([[0, 2, 0], [4, 0, 7]][view_index], dtype=torch.uint32).reshape(3, 1),
    }


def _bare_tracer(renderer_module):
    tracer = renderer_module.Tracer.__new__(renderer_module.Tracer)
    tracer.tracer_wrapper = object()
    tracer._next_subview_frame_id = 1 << 31
    if renderer_module is grt_module:
        tracer.frame_timer = object()
        tracer.timings = {}
    else:
        tracer.conf = gut_module.OmegaConf.create({"render": {"enable_kernel_timings": True}})
        tracer._batch_tracer_wrappers = []
        tracer._batch_timings = None
    return tracer


@NATIVE_MODULES
def test_single_view_render_forwards_exact_batch_and_frame_id(monkeypatch, renderer_module):
    tracer = _bare_tracer(renderer_module)
    batch = _make_batch(1)
    gaussians = object()
    sentinel = object()
    calls = []

    def fake_render_single(gaussians_arg, batch_arg, train=False, frame_id=0, **kwargs):
        calls.append((gaussians_arg, batch_arg, train, frame_id, kwargs))
        return sentinel

    monkeypatch.setattr(tracer, "_render_single", fake_render_single)

    result = tracer.render(gaussians, batch, train=True, frame_id=123456789)

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][0] is gaussians
    assert calls[0][1] is batch
    assert calls[0][2:4] == (True, 123456789)
    assert calls[0][4] == {}


@NATIVE_MODULES
def test_single_view_render_forwards_opacity_override(monkeypatch, renderer_module):
    tracer = _bare_tracer(renderer_module)
    batch = _make_batch(1)
    gaussians = SimpleNamespace(num_gaussians=3)
    opacity_override = torch.tensor([[0.1], [0.4], [0.8]], requires_grad=True)
    sentinel = object()
    calls = []

    def fake_render_single(gaussians_arg, batch_arg, train=False, frame_id=0, **kwargs):
        calls.append((gaussians_arg, batch_arg, train, frame_id, kwargs))
        return sentinel

    monkeypatch.setattr(tracer, "_render_single", fake_render_single)

    result = tracer.render(
        gaussians,
        batch,
        train=True,
        frame_id=123456789,
        opacity_override=opacity_override,
    )

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][:4] == (gaussians, batch, True, 123456789)
    assert calls[0][4] == {"opacity_override": opacity_override}


@NATIVE_MODULES
def test_two_view_render_aggregates_outputs_and_uses_safe_wrappers(monkeypatch, renderer_module):
    tracer = _bare_tracer(renderer_module)
    primary_wrapper = tracer.tracer_wrapper
    created_wrappers = []

    if renderer_module is gut_module:

        def make_wrapper(_conf):
            wrapper = object()
            created_wrappers.append(wrapper)
            return wrapper

        monkeypatch.setattr(
            gut_module,
            "_3dgut_plugin",
            SimpleNamespace(SplatRaster=make_wrapper),
        )

    batch = _make_batch(2)
    gaussians = object()
    calls = []

    def fake_render_single(gaussians_arg, batch_arg, train=False, frame_id=0, **kwargs):
        wrapper = kwargs.get("tracer_wrapper", tracer.tracer_wrapper)
        calls.append((gaussians_arg, batch_arg, train, frame_id, wrapper))
        return _make_output(len(calls) - 1)

    monkeypatch.setattr(tracer, "_render_single", fake_render_single)

    result = tracer.render(gaussians, batch, train=True, frame_id=17)

    assert len(calls) == 2
    assert all(call[0] is gaussians and call[2] is True for call in calls)
    assert [call[1].batch_size for call in calls] == [1, 1]
    assert [call[1].camera_idx for call in calls] == [0, 1]
    assert [call[1].frame_idx for call in calls] == [10, 11]

    internal_frame_ids = [call[3] for call in calls]
    assert internal_frame_ids == [1 << 31, (1 << 31) + 1]
    assert len(set(internal_frame_ids)) == 2
    assert all(0 <= frame_id < (1 << 32) for frame_id in internal_frame_ids)

    expected_outputs = [_make_output(0), _make_output(1)]
    for key in IMAGE_OUTPUTS:
        torch.testing.assert_close(
            result[key],
            torch.cat([output[key] for output in expected_outputs], dim=0),
        )

    expected_visibility = torch.tensor([[1], [1], [1]], dtype=torch.uint32)
    torch.testing.assert_close(result["mog_visibility"], expected_visibility)
    assert result["mog_visibility"].shape == (3, 1)
    assert result["mog_visibility"].dtype == expected_outputs[0]["mog_visibility"].dtype
    assert result["frame_time_ms"] == pytest.approx(3.75)
    assert tracer.timings["forward_render"] == pytest.approx(3.75)

    wrappers = [call[4] for call in calls]
    if renderer_module is gut_module:
        assert wrappers == created_wrappers
        assert len(created_wrappers) == 2
        assert wrappers[0] is not wrappers[1]
        assert all(wrapper is not primary_wrapper for wrapper in wrappers)
    else:
        assert wrappers == [primary_wrapper, primary_wrapper]


@NATIVE_MODULES
@pytest.mark.parametrize("trailing_dimension", [False, True])
def test_two_view_render_slices_camera_specific_opacity(
    monkeypatch,
    renderer_module,
    trailing_dimension,
):
    tracer = _bare_tracer(renderer_module)
    if renderer_module is gut_module:
        monkeypatch.setattr(
            gut_module,
            "_3dgut_plugin",
            SimpleNamespace(SplatRaster=lambda _conf: object()),
        )

    batch = _make_batch(2)
    gaussians = SimpleNamespace(num_gaussians=3)
    opacity_override = torch.tensor(
        [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
        requires_grad=True,
    )
    if trailing_dimension:
        opacity_override = opacity_override.unsqueeze(-1)
    received_opacities = []

    def fake_render_single(gaussians_arg, batch_arg, train=False, frame_id=0, **kwargs):
        received_opacities.append(kwargs.pop("opacity_override"))
        return _make_output(len(received_opacities) - 1)

    monkeypatch.setattr(tracer, "_render_single", fake_render_single)

    tracer.render(gaussians, batch, opacity_override=opacity_override)

    assert len(received_opacities) == batch.batch_size
    for index, received in enumerate(received_opacities):
        torch.testing.assert_close(received, opacity_override[index])
        assert received.requires_grad


@NATIVE_MODULES
def test_two_view_opacity_slicing_preserves_switch_splat_gradient_chain(monkeypatch, renderer_module):
    tracer = _bare_tracer(renderer_module)
    if renderer_module is gut_module:
        monkeypatch.setattr(
            gut_module,
            "_3dgut_plugin",
            SimpleNamespace(SplatRaster=lambda _conf: object()),
        )

    switch = SwitchSplatController(
        OmegaConf.create(
            {
                "num_slots": 1,
                "num_frames": 256,
                "init_windows": [[20.0, 80.0]],
                "temperature": 10.0,
            }
        )
    )
    base_opacity = torch.tensor([[0.25], [0.55], [0.85]], requires_grad=True)
    switch_logits = switch.initialize_logits(3, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        switch_logits[:, 1] += torch.tensor([-0.3, 0.1, 0.4])
        switch_logits[:, 2] += torch.tensor([0.2, -0.2, 0.3])
    opacity_override = switch.compute_effective_opacity(
        base_opacity,
        switch_logits,
        torch.tensor([27.0, 68.0]),
    )
    received_opacities = []

    def fake_render_single(gaussians_arg, batch_arg, train=False, frame_id=0, **kwargs):
        received_opacities.append(kwargs.pop("opacity_override"))
        return _make_output(len(received_opacities) - 1)

    monkeypatch.setattr(tracer, "_render_single", fake_render_single)
    tracer.render(
        SimpleNamespace(num_gaussians=3),
        _make_batch(2),
        opacity_override=opacity_override,
    )

    loss = received_opacities[0].sum() + 2 * received_opacities[1].sum()
    loss.backward()
    for parameter in (base_opacity, switch_logits, switch.endpoints_raw, switch.temperature_raw):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)


@NATIVE_MODULES
@pytest.mark.parametrize(
    "batch_size,shape",
    [(1, (1, 3, 1)), (2, (3, 1)), (2, (2, 4))],
)
def test_opacity_override_shape_is_validated(renderer_module, batch_size, shape):
    tracer = _bare_tracer(renderer_module)
    gaussians = SimpleNamespace(num_gaussians=3)

    with pytest.raises(ValueError, match="opacity_override must have shape"):
        tracer.render(gaussians, _make_batch(batch_size), opacity_override=torch.ones(shape))


@NATIVE_MODULES
def test_internal_frame_id_exhaustion_is_reported(renderer_module):
    tracer = renderer_module.Tracer.__new__(renderer_module.Tracer)
    tracer._next_subview_frame_id = (1 << 32) - 1

    with pytest.raises(RuntimeError, match="exhausted the uint32 range"):
        tracer._allocate_subview_frame_ids(2)

    assert tracer._next_subview_frame_id == (1 << 32) - 1


def test_native_gut_camera_parameters_convert_batched_tensors_for_pybind():
    vector = torch.tensor([720, 720])
    scalar = torch.tensor(1.8452)
    original = np.array([360.0, 360.0], dtype=np.float32)

    assert gut_module._as_native_camera_parameter(vector) == [720, 720]
    assert gut_module._as_native_camera_parameter(scalar) == pytest.approx(1.8452)
    assert gut_module._as_native_camera_parameter(original) is original
