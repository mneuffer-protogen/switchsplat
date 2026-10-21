# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import torch

from threedgrut.utils.misc import to_np


@torch.no_grad()
def render_switch_splat_timetable(controller, times, gt_windows, step=None) -> np.ndarray:
    """Render the learned SwitchSplat windows as an old-style assignment table."""
    times = np.asarray(times, dtype=np.float32)
    windows = to_np(controller.windows(torch.from_numpy(times))).T
    starts, ends = (to_np(value) for value in controller.window_bounds())
    temperatures = to_np(controller.temperatures())
    gt_windows = np.asarray(gt_windows, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, max(2.0, 0.75 * controller.num_slots + 0.8)), dpi=120)
    ax.imshow(
        windows,
        cmap="PiYG",
        norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
        extent=(times[0] - 0.5, times[-1] + 0.5, controller.num_slots - 0.5, -0.5),
        interpolation="nearest",
        aspect="auto",
    )
    for index, (window, start, end, temperature, ground_truth) in enumerate(
        zip(windows, starts, ends, temperatures, gt_windows)
    ):
        ax.plot(times, index + 0.4 - 0.8 * window, color="#67E8F9", linewidth=1.6)
        ax.add_patch(
            Rectangle(
                (start, index - 0.45),
                end - start,
                0.9,
                fill=False,
                edgecolor="#06B6D4",
                linewidth=1.5,
            )
        )
        ax.add_patch(
            Rectangle(
                (ground_truth[0], index - 0.47),
                ground_truth[1] - ground_truth[0],
                0.94,
                fill=False,
                edgecolor="#F59E0B",
                linewidth=2.0,
                linestyle="--",
            )
        )
        ax.text(
            0.5 * (start + end),
            index,
            f"{start:.1f}–{end:.1f}  T={temperature:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    ax.set_yticks(range(controller.num_slots), [chr(ord("B") + index) for index in range(controller.num_slots)])
    ax.set_xlim(times[0] - 0.5, times[-1] + 0.5)
    ax.set_xlabel("raw frame time")
    ax.set_title(f"SwitchSplat timetable{'' if step is None else f' · step {step:,}'}    cyan: learned    orange: GT")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image
