# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch

from threedgrut.utils.switch_splat_timetable import render_switch_splat_timetable


def test_timetable_renders_controller_windows_without_mutating_them():
    controller = SimpleNamespace(
        num_slots=2,
        windows=lambda times: torch.stack((times / 10.0, 1.0 - times / 10.0), dim=-1),
        window_bounds=lambda: (torch.tensor([1.0, 5.0]), torch.tensor([4.0, 9.0])),
        temperatures=lambda: torch.tensor([2.0, 3.0]),
    )
    times = np.arange(11, dtype=np.float32)

    image = render_switch_splat_timetable(controller, times, [[1.0, 4.0], [5.0, 9.0]], step=12)

    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.shape[1] > image.shape[0]
    assert np.unique(image.reshape(-1, 3), axis=0).shape[0] > 20
