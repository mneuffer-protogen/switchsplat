# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch

from threedgrut.utils import viser_gui_util
from threedgrut.utils.viser_gui_util import ViserGUI


class _Model:
    def __init__(self):
        self.positions = torch.zeros((1, 3))
        self.state_activations = []

    def __call__(self, inputs, train=False, state_activations=None):
        self.state_activations.append(state_activations)
        height, width = inputs.rays_dir.shape[1:3]
        value = 0.0 if state_activations is None else float(state_activations.sum())
        return {
            "pred_features": torch.full((1, height, width, 3), value),
            "pred_opacity": torch.ones((1, height, width, 1)),
            "pred_dist": torch.ones((1, height, width, 1)),
            "pred_normals": torch.ones((1, height, width, 3)),
            "hits_count": torch.ones((1, height, width, 1)),
        }


def _gui(model):
    gui = ViserGUI.__new__(ViserGUI)
    gui.conf = SimpleNamespace(writer=SimpleNamespace(max_num_hits=128))
    gui.model = model
    gui.feature_decoder = None
    gui.feature_decoder_center_ray_encoding = False
    gui.adjust_resolution_checkbox = SimpleNamespace(value=False)
    gui.resolution_slider = SimpleNamespace(value=8)
    gui.viz_render_subsample = 1
    gui.viz_render_train_view = False
    gui.render_timer = SimpleNamespace(start=lambda: None, end=lambda: None)
    gui.render_width = gui.render_height = 8
    return gui


def _client():
    return SimpleNamespace(
        camera=SimpleNamespace(
            aspect=1.0,
            fov=np.pi / 2,
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.zeros(3),
        )
    )


def test_state_controls_toggle_groups_and_build_activation_vector():
    gui = ViserGUI.__new__(ViserGUI)
    gui.model = SimpleNamespace(positions=torch.zeros(1, 3))
    gui.switch_state_checkboxes = [SimpleNamespace(value=value) for value in (True, True, False, True, False)]

    torch.testing.assert_close(gui._switch_state_activations(), torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]]))
    gui._toggle_switch_states((1, 2))
    torch.testing.assert_close(gui._switch_state_activations(), torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0]]))
    gui._toggle_switch_states((0,))
    torch.testing.assert_close(gui._switch_state_activations(), torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]))


def test_viser_render_passes_direct_states_and_preserves_disabled_path(monkeypatch):
    monkeypatch.setattr(viser_gui_util, "DEFAULT_DEVICE", "cpu")
    model = _Model()
    gui = _gui(model)

    gui.switch_state_checkboxes = []
    gui.render_from_current_view(_client())
    assert model.state_activations[-1] is None

    gui.switch_state_checkboxes = [SimpleNamespace(value=value) for value in (True, False, True)]
    gui.render_from_current_view(_client())
    torch.testing.assert_close(model.state_activations[-1], torch.tensor([[1.0, 0.0, 1.0]]))


def test_timetable_streams_image_and_freezes_with_live_update(monkeypatch):
    gui = ViserGUI.__new__(ViserGUI)
    gui.server = SimpleNamespace(get_clients=lambda: {0: object()})
    gui.switch_timetable_image = SimpleNamespace(image=None)
    gui._render_switch_timetable = lambda step: np.full((2, 3, 3), step, dtype=np.uint8)
    gui._last_timetable_update = 0.0
    gui.live_update = True
    monkeypatch.setattr(viser_gui_util.time, "monotonic", lambda: 6.0)

    gui.update_switch_timetable(7)
    np.testing.assert_array_equal(gui.switch_timetable_image.image, np.full((2, 3, 3), 7, dtype=np.uint8))

    frozen = gui.switch_timetable_image.image
    gui.live_update = False
    monkeypatch.setattr(viser_gui_util.time, "monotonic", lambda: 12.0)
    gui.update_switch_timetable(8)
    assert gui.switch_timetable_image.image is frozen
