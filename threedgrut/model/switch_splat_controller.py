# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def _encode_bounded(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    probability = ((value - lower) / (upper - lower)).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probability)


def make_switch_splat_controller(config) -> "SwitchSplatController | None":
    if config is None or not bool(config.get("enabled", False)):
        return None
    if config.get("num_slots", None) is None:
        raise ValueError("switch_splat.num_slots must be resolved before model construction")
    num_slots = int(config.num_slots)
    if num_slots < 0:
        raise ValueError("switch_splat.num_slots must be non-negative")
    if num_slots == 0:
        return None
    return SwitchSplatController(config)


class SwitchSplatController(torch.nn.Module):
    """Global temporal windows and opacity-gating math for SwitchSplat."""

    def __init__(self, config) -> None:
        super().__init__()
        self.num_slots = int(config.num_slots)
        self.num_frames = int(config.num_frames)

        if self.num_slots <= 0:
            raise ValueError("SwitchSplat requires num_slots > 0")
        if self.num_frames <= 1:
            raise ValueError("switch_splat.num_frames must be greater than 1")
        windows = torch.as_tensor(config.init_windows, dtype=torch.float32)
        if windows.shape != (self.num_slots, 2):
            raise ValueError("switch_splat.init_windows must contain exactly one [start, end] pair per slot")

        frame_max = float(self.num_frames - 1)
        temperature = torch.full((self.num_slots,), float(config.temperature), dtype=torch.float32)
        self.endpoints_raw = torch.nn.Parameter(_encode_bounded(windows, 0.0, frame_max))
        self.temperature_raw = torch.nn.Parameter(_encode_bounded(temperature, -1.0, 35.0))

    @property
    def num_states(self) -> int:
        return 1 + 2 * self.num_slots

    def endpoints(self) -> torch.Tensor:
        return float(self.num_frames - 1) * torch.sigmoid(self.endpoints_raw)

    def temperatures(self) -> torch.Tensor:
        return -1.0 + 36.0 * torch.sigmoid(self.temperature_raw)

    def window_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        endpoints = self.endpoints()
        return torch.minimum(endpoints[:, 0], endpoints[:, 1]), torch.maximum(endpoints[:, 0], endpoints[:, 1])

    def windows(self, times: torch.Tensor) -> torch.Tensor:
        times = torch.as_tensor(times, device=self.endpoints_raw.device, dtype=self.endpoints_raw.dtype).reshape(-1)
        starts, ends = self.window_bounds()
        temperature = self.temperatures().clamp_min(1e-6)
        raw_windows = torch.sigmoid((times[:, None] - starts) / temperature) * torch.sigmoid(
            (ends - times[:, None]) / temperature
        )
        midpoint = 0.5 * (starts + ends)
        midpoint_value = torch.sigmoid((midpoint - starts) / temperature) * torch.sigmoid(
            (ends - midpoint) / temperature
        )
        return (raw_windows / midpoint_value.clamp_min(1e-12)).clamp(0, 1)

    def state_activations(self, times: torch.Tensor) -> torch.Tensor:
        windows = self.windows(times)
        activations = torch.empty(
            (windows.shape[0], self.num_states),
            dtype=windows.dtype,
            device=windows.device,
        )
        activations[:, 0] = 1
        activations[:, 1::2] = 1 - windows
        activations[:, 2::2] = windows
        return activations

    def state_code_activations(self, state_code: torch.Tensor) -> torch.Tensor:
        state_code = torch.as_tensor(state_code, device=self.endpoints_raw.device, dtype=self.endpoints_raw.dtype)
        if state_code.ndim != 2 or state_code.shape[1] != self.num_slots:
            raise ValueError(f"state_code must have shape [B,{self.num_slots}], got {tuple(state_code.shape)}")
        return torch.cat(
            [torch.ones_like(state_code[:, :1]), torch.stack([1 - state_code, state_code], dim=-1).flatten(1)],
            dim=1,
        )

    def initialize_logits(self, num_gaussians: int, *, device, dtype) -> torch.nn.Parameter:
        logits = torch.zeros((num_gaussians, self.num_states), device=device, dtype=dtype)
        return torch.nn.Parameter(logits)

    def compute_effective_opacity(
        self,
        base_opacity: torch.Tensor,
        switch_logits: torch.Tensor,
        times: torch.Tensor | None = None,
        state_activations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_opacity.ndim != 2 or base_opacity.shape[1] != 1:
            raise ValueError(f"base_opacity must have shape [N, 1], got {tuple(base_opacity.shape)}")
        if switch_logits.shape != (base_opacity.shape[0], self.num_states):
            raise ValueError(
                f"switch_logits must have shape [{base_opacity.shape[0]}, {self.num_states}], "
                f"got {tuple(switch_logits.shape)}"
            )

        activations = (self.state_activations(times) if state_activations is None else state_activations).to(
            switch_logits
        )
        state_probs = torch.softmax(switch_logits, dim=-1)
        gate = activations @ state_probs.transpose(0, 1)
        effective_opacity = gate * base_opacity.transpose(0, 1)
        if effective_opacity.shape[0] == 1:
            return effective_opacity.transpose(0, 1)
        return effective_opacity.unsqueeze(-1)
