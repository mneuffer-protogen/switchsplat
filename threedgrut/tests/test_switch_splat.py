import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.model.losses import switch_splat_mode_a_loss, switch_splat_spatial_loss
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.switch_splat_controller import (
    SwitchSplatController,
    make_switch_splat_controller,
)
from threedgrut.trainer import Trainer3DGRUT


def _config(num_slots: int, windows=None):
    if windows is None:
        windows = [[20.0 + 40.0 * index, 50.0 + 40.0 * index] for index in range(num_slots)]
    return OmegaConf.create(
        {
            "enabled": True,
            "num_slots": num_slots,
            "num_frames": 256,
            "init_windows": windows,
            "temperature": 10.0,
            "logits_lr": 0.05,
            "endpoints_lr": 0.0005,
            "temperature_lr": 0.002,
        }
    )


def test_original_window_parameterization_without_leak_matches_reference():
    switch = SwitchSplatController(_config(1, [[20.0, 80.0]]))
    starts, ends = switch.window_bounds()
    torch.testing.assert_close(starts, torch.tensor([20.0]))
    torch.testing.assert_close(ends, torch.tensor([80.0]))
    torch.testing.assert_close(switch.temperatures(), torch.tensor([10.0]))

    values = switch.windows(torch.tensor([-100.0, 50.0, 200.0]))
    torch.testing.assert_close(values[1], torch.ones(1), atol=1e-6, rtol=0)
    assert values[0].item() < 1e-5
    assert values[2].item() < 1e-5

    times = torch.linspace(0, 255, 37)
    lo, hi = switch.window_bounds()
    temperature = switch.temperatures().clamp_min(1e-6)
    raw = torch.sigmoid((times[:, None] - lo) / temperature) * torch.sigmoid((hi - times[:, None]) / temperature)
    midpoint = 0.5 * (lo + hi)
    midpoint_value = torch.sigmoid((midpoint - lo) / temperature) * torch.sigmoid((hi - midpoint) / temperature)
    torch.testing.assert_close(switch.windows(times), (raw / midpoint_value.clamp_min(1e-12)).clamp(0, 1))

    loss = switch.windows(torch.tensor([30.0])).sum()
    loss.backward()
    assert switch.endpoints_raw.grad is not None and switch.endpoints_raw.grad.abs().sum() > 0
    assert switch.temperature_raw.grad is not None and switch.temperature_raw.grad.abs().sum() > 0


@pytest.mark.parametrize(("batch_size", "num_slots"), [(1, 1), (2, 1), (1, 3), (2, 3)])
def test_matrix_gate_matches_explicit_state_loop(batch_size: int, num_slots: int):
    generator = torch.Generator().manual_seed(5)
    switch = SwitchSplatController(_config(num_slots))
    num_gaussians = 4
    base_opacity = torch.rand((num_gaussians, 1), generator=generator, requires_grad=True)
    logits = torch.randn((num_gaussians, switch.num_states), generator=generator, requires_grad=True)
    times = torch.linspace(10, 100, batch_size)

    actual = switch.compute_effective_opacity(base_opacity, logits, times)
    activations = switch.state_activations(times)
    direct = switch.compute_effective_opacity(base_opacity, logits, state_activations=activations)
    probabilities = torch.softmax(logits, dim=-1)
    reference = torch.empty((batch_size, num_gaussians))
    for camera in range(batch_size):
        for gaussian in range(num_gaussians):
            reference[camera, gaussian] = (
                sum(activations[camera, state] * probabilities[gaussian, state] for state in range(switch.num_states))
                * base_opacity[gaussian, 0]
            )

    expected = reference[0, :, None] if batch_size == 1 else reference[..., None]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(direct, expected)
    assert torch.all(actual >= 0)
    if batch_size == 1:
        assert torch.all(actual <= base_opacity)
    else:
        assert torch.all(actual <= base_opacity[None])


def test_uniform_logit_initialization_and_identity_modes():
    switch = SwitchSplatController(_config(3))
    logits = switch.initialize_logits(5, device="cpu", dtype=torch.float32)
    expected = torch.full_like(logits, 1.0 / switch.num_states)
    torch.testing.assert_close(torch.softmax(logits, dim=-1), expected)

    assert make_switch_splat_controller(None) is None
    assert make_switch_splat_controller(OmegaConf.create({"enabled": False})) is None
    identity = make_switch_splat_controller(OmegaConf.create({"enabled": True, "num_slots": 0}))
    assert identity is None


def test_state_codes_map_to_slot_pairs_and_bypass_time():
    switch = SwitchSplatController(_config(2))
    codes = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    torch.testing.assert_close(
        switch.state_code_activations(codes),
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0, 0.0]]),
    )

    density = torch.nn.Parameter(torch.tensor([[0.8], [0.4]]))
    logits = switch.initialize_logits(2, device="cpu", dtype=torch.float32)
    renderer = SimpleNamespace(
        render=lambda _model, _batch, _train, _frame_id, *, opacity_override: {"pred_opacity": opacity_override}
    )
    model = SimpleNamespace(
        switch_splat_controller=switch,
        switch_logits=logits,
        renderer=renderer,
        get_density=lambda: density,
    )

    outputs = MixtureOfGaussians.forward(model, SimpleNamespace(state_code=codes, time_idx=None))
    expected = switch.compute_effective_opacity(
        density,
        logits,
        state_activations=switch.state_code_activations(codes),
    )
    torch.testing.assert_close(outputs["pred_opacity"], expected)


def test_timeline_loads_dataset_windows_and_derives_slot_count(tmp_path):
    (tmp_path / "interaction_times.txt").write_text("20 80\n100 170\n")
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.train_dataset = SimpleNamespace(time_indices=torch.tensor([2, 104, 511]))
    trainer.val_dataset = SimpleNamespace(time_indices=torch.tensor([1, 533]))
    config = OmegaConf.create(
        {
            "path": str(tmp_path),
            "switch_splat": {
                "enabled": True,
                "num_slots": None,
                "num_frames": None,
                "init_windows_file": "interaction_times.txt",
                "init_windows": None,
            },
        }
    )

    trainer._configure_switch_splat_timeline(config)

    assert config.switch_splat.num_frames == 534
    assert config.switch_splat.num_slots == 2
    assert config.switch_splat.init_windows == [[20.0, 80.0], [100.0, 170.0]]

    config.switch_splat.init_windows_file = None
    config.switch_splat.init_windows = [[30.0, 40.0]]
    config.switch_splat.num_frames = None
    trainer._configure_switch_splat_timeline(config)
    assert config.switch_splat.num_slots == 1

    config.switch_splat.init_windows = None
    trainer._configure_switch_splat_timeline(config)
    assert config.switch_splat.num_slots == 0
    assert config.switch_splat.init_windows == []


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("", "empty"),
        ("20 80 90\n", "shape"),
        ("80 20\n", "starts"),
        ("20 600\n", "within"),
    ],
)
def test_timeline_rejects_invalid_window_files(tmp_path, contents, match):
    (tmp_path / "windows.txt").write_text(contents)
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.train_dataset = trainer.val_dataset = SimpleNamespace(time_indices=torch.tensor([0, 533]))
    config = OmegaConf.create(
        {
            "path": str(tmp_path),
            "switch_splat": {
                "enabled": True,
                "num_frames": None,
                "init_windows_file": "windows.txt",
            },
        }
    )

    with pytest.raises(ValueError, match=match):
        trainer._configure_switch_splat_timeline(config)


def test_paper_losses_match_reference_and_gradient_targets():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.0, 0.0]],
        requires_grad=True,
    )
    opacity = torch.tensor([0.8, 0.6, 0.7, 0.9], requires_grad=True)
    logits = torch.tensor(
        [
            [0.2, 1.0, -0.3, 0.1, -0.2],
            [0.1, -0.1, 1.2, 0.0, -0.4],
            [0.3, -0.5, 0.0, 1.1, -0.2],
            [0.0, -0.2, -0.1, 0.2, 1.3],
        ],
        requires_grad=True,
    )

    probabilities = torch.softmax(logits, dim=1)
    expected_mode_a = -torch.log(probabilities[:, 0].clamp_min(1e-8)).mean()
    torch.testing.assert_close(switch_splat_mode_a_loss(logits), expected_mode_a)

    weights = opacity[:, None] * probabilities[:, 1:]
    normalized = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)
    state2 = weights[:, 1::2]
    state2 = state2 / (state2.sum(dim=0, keepdim=True) + 1e-8)
    centroids = (state2.t() @ positions).repeat_interleave(2, dim=0)
    distances = (positions[:, None] - centroids[None]).square().sum(dim=-1)
    expected_spatial = (normalized * distances).sum(dim=0).mean()
    actual_spatial = switch_splat_spatial_loss(positions, opacity, logits)
    torch.testing.assert_close(actual_spatial, expected_spatial)

    (switch_splat_mode_a_loss(logits) + actual_spatial).backward()
    for parameter in (positions, opacity, logits):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)


def test_paper_loss_schedule_boundaries():
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.conf = OmegaConf.create(
        {
            "loss": {
                "mode_a_schedule": {
                    "start_iteration": 18000,
                    "end_iteration": -1,
                    "ramp_iterations": 1000,
                }
            }
        }
    )

    expected = {17999: 0.0, 18000: 0.001, 18999: 1.0, 21000: 1.0}
    for step, multiplier in expected.items():
        trainer.global_step = step
        assert trainer._loss_schedule_multiplier("loss.mode_a_schedule") == pytest.approx(multiplier)


def test_scheduled_logit_softening_preserves_argmax_and_resets_only_its_moments():
    density = torch.nn.Parameter(torch.tensor([[0.4], [0.7]]))
    logits = torch.nn.Parameter(torch.tensor([[4.0, 1.0, -2.0], [0.0, 3.0, 6.0]]))
    optimizer = torch.optim.Adam(
        [
            {"params": [density], "name": "density", "lr": 0.01},
            {"params": [logits], "name": "switch_logits", "lr": 0.01},
        ]
    )
    (density.square().sum() + logits.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad()

    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.conf = OmegaConf.create(
        {
            "switch_splat": {
                "logit_softening": {
                    "enabled": True,
                    "iterations": [18000, 21000],
                    "factor": 0.3,
                }
            }
        }
    )
    trainer.model = SimpleNamespace(density=density, switch_logits=logits, optimizer=optimizer)
    density_before = density.detach().clone()
    logits_before = logits.detach().clone()
    density_state_before = copy.deepcopy(optimizer.state[density])
    switch_step_before = optimizer.state[logits]["step"].clone()

    trainer._maybe_soften_switch_logits(17999)
    torch.testing.assert_close(logits, logits_before, rtol=0, atol=0)

    trainer._maybe_soften_switch_logits(18000)
    expected = (logits_before - logits_before.mean(dim=1, keepdim=True)) * 0.3
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(density, density_before, rtol=0, atol=0)
    torch.testing.assert_close(logits.argmax(dim=1), logits_before.argmax(dim=1), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[logits]["step"], switch_step_before, rtol=0, atol=0)
    for name, value in optimizer.state[logits].items():
        if name != "step":
            torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
    for name, value in density_state_before.items():
        torch.testing.assert_close(optimizer.state[density][name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"init_windows": []}, "exactly one"),
        ({"num_frames": 1}, "greater than 1"),
    ],
)
def test_invalid_window_configuration_is_rejected(updates, message):
    config = _config(1)
    for key, value in updates.items():
        config[key] = value
    with pytest.raises(ValueError, match=message):
        SwitchSplatController(config)


def test_disabled_model_forward_preserves_original_renderer_call_and_gradients():
    class Renderer:
        def __init__(self):
            self.args = None

        def render(self, *args):
            self.args = args
            return {"value": args[0].density.square()}

    density = torch.nn.Parameter(torch.tensor([[2.0]]))
    model = SimpleNamespace(switch_splat_controller=None, density=density, renderer=Renderer())
    batch = object()
    outputs = MixtureOfGaussians.forward(model, batch, train=True, frame_id=17)
    assert model.renderer.args == (model, batch, True, 17)
    outputs["value"].sum().backward()
    torch.testing.assert_close(density.grad, torch.tensor([[4.0]]))


def test_active_model_passes_isolated_per_camera_opacity_and_gradients():
    class Renderer:
        def __init__(self):
            self.opacity = None

        def render(self, _model, _batch, _train, _frame_id, *, opacity_override):
            self.opacity = opacity_override
            return {"pred_opacity": opacity_override}

    switch = SwitchSplatController(_config(1, [[20.0, 80.0]]))
    density = torch.nn.Parameter(torch.tensor([[0.8], [0.4]]))
    logits = switch.initialize_logits(2, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        logits[:, 1] = 0.3
        logits[:, 2] = -0.2
    renderer = Renderer()
    model = SimpleNamespace(
        switch_splat_controller=switch,
        switch_logits=logits,
        renderer=renderer,
        get_density=lambda: density,
    )

    batch = SimpleNamespace(time_idx=torch.tensor([20, 50]))
    outputs = MixtureOfGaussians.forward(model, batch, train=True, frame_id=4)
    expected = switch.compute_effective_opacity(density, logits, batch.time_idx)
    torch.testing.assert_close(outputs["pred_opacity"], expected)
    assert outputs["pred_opacity"].shape == (2, 2, 1)

    original_second = outputs["pred_opacity"][1].detach().clone()
    changed = MixtureOfGaussians.forward(
        model,
        SimpleNamespace(time_idx=torch.tensor([-100, 50])),
        train=True,
        frame_id=5,
    )
    torch.testing.assert_close(changed["pred_opacity"][1], original_second)

    outputs["pred_opacity"].sum().backward()
    for parameter in (density, logits, switch.endpoints_raw, switch.temperature_raw):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_enabled_state_and_both_optimizers_resume_exactly_then_advance():
    config = _config(1, [[20.0, 80.0]])

    def make_training_state():
        switch = SwitchSplatController(config)
        density = torch.nn.Parameter(torch.tensor([[0.8], [0.4]]))
        logits = switch.initialize_logits(2, device="cpu", dtype=torch.float32)
        gaussian_optimizer = torch.optim.Adam(
            [
                {"params": [density], "name": "density", "lr": 0.01},
                {"params": [logits], "name": "switch_logits", "lr": 0.01},
            ]
        )
        trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
        trainer.model = SimpleNamespace(switch_splat_controller=switch)
        trainer.conf = SimpleNamespace(
            switch_splat=config,
            optimizer=SimpleNamespace(eps=1e-15),
        )
        trainer._setup_switch_splat_optimizer()
        return switch, density, logits, gaussian_optimizer, trainer.switch_splat_optimizer

    def step(state):
        switch, density, logits, gaussian_optimizer, window_optimizer = state
        opacity = switch.compute_effective_opacity(
            density.sigmoid(),
            logits,
            torch.tensor([27.0, 68.0]),
        )
        loss = (opacity * torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])).sum()
        loss.backward()
        gaussian_optimizer.step()
        window_optimizer.step()
        gaussian_optimizer.zero_grad()
        window_optimizer.zero_grad()
        return loss.detach()

    source = make_training_state()
    assert [(group["name"], group["lr"]) for group in source[-1].param_groups] == [
        ("endpoints_raw", 0.0005),
        ("temperature_raw", 0.002),
    ]
    assert all(group["eps"] == 1e-15 for group in source[-1].param_groups)
    step(source)
    step(source)
    switch, density, logits, gaussian_optimizer, window_optimizer = source
    checkpoint = {
        "global_step": 2,
        "density": density.detach().clone(),
        "switch_logits": logits.detach().clone(),
        "switch_splat": copy.deepcopy(switch.state_dict()),
        "optimizer": copy.deepcopy(gaussian_optimizer.state_dict()),
        "switch_splat_optimizer": copy.deepcopy(window_optimizer.state_dict()),
    }

    restored = make_training_state()
    restored_switch, restored_density, restored_logits, restored_gaussian_optimizer, restored_window_optimizer = (
        restored
    )
    with torch.no_grad():
        restored_density.copy_(checkpoint["density"])
        restored_logits.copy_(checkpoint["switch_logits"])
    restored_switch.load_state_dict(checkpoint["switch_splat"])
    restored_gaussian_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_window_optimizer.load_state_dict(checkpoint["switch_splat_optimizer"])

    torch.testing.assert_close(restored_density, density, rtol=0, atol=0)
    torch.testing.assert_close(restored_logits, logits, rtol=0, atol=0)
    for name, value in switch.state_dict().items():
        torch.testing.assert_close(restored_switch.state_dict()[name], value, rtol=0, atol=0)

    def assert_optimizer_equal(actual, expected):
        actual_state = actual.state_dict()
        expected_state = expected.state_dict()
        assert actual_state["param_groups"] == expected_state["param_groups"]
        assert actual_state["state"].keys() == expected_state["state"].keys()
        for parameter_id, state in expected_state["state"].items():
            assert state.keys() == actual_state["state"][parameter_id].keys()
            for name, value in state.items():
                torch.testing.assert_close(actual_state["state"][parameter_id][name], value, rtol=0, atol=0)

    assert_optimizer_equal(restored_gaussian_optimizer, gaussian_optimizer)
    assert_optimizer_equal(restored_window_optimizer, window_optimizer)

    assert torch.isfinite(step(restored))
    assert torch.isfinite(step(restored))
    for parameter in (restored_density, restored_logits, *restored_switch.parameters()):
        assert torch.isfinite(parameter).all()
    assert all(int(state["step"]) == 4 for state in restored_gaussian_optimizer.state.values())
    assert all(int(state["step"]) == 4 for state in restored_window_optimizer.state.values())
