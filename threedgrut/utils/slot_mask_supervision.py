# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask-guided opacity supervision for a SwitchSplat fine-tuning stage.

Adds one term to the training loss: the Gaussians owned by an object slot must not
paint anything outside that object's mask. Slot *k* owns two states, 2k-1 ("before")
and 2k ("after"); the weak masks cover the symmetric difference of the two
configurations, so the same mask constrains both, and both are rendered with their
temporal gate forced on so each gets a gradient from the frames that have masks.

Masks come from one camera only. Other cameras are ignored by default; the
optional ``other_cameras_empty`` setting instead penalizes the whole frame.
Multiple slot masks may share one frame when object intervals overlap.

Supports two modes controlled by ``soft_gating``:
  - **hard** (default, soft_gating=false): state membership is decided once by argmax
    over the frozen logits; only opacity is differentiable.
  - **soft** (soft_gating=true): each Gaussian's rendered opacity for a state is
    ``density × softmax(logits)[:, state]``, so gradients flow through both density
    *and* the logits.  A Gaussian can escape the mask penalty by shifting its logit
    toward the base state rather than being deleted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from threedgrut.datasets.protocols import Batch, slice_batch


class GaussianSubset:
    """Read-only view over a fixed subset of a model's Gaussians.

    The rasterizer reads Gaussians only through these accessors, so a view is enough to
    render one SwitchSplat state on its own at a fraction of the cost of the full scene.
    Indexing the activated tensors keeps the subset identical to what the full render
    would produce, and leaves the gather in the graph so gradients reach the shared
    density parameter.
    """

    def __init__(self, model, indices: torch.Tensor) -> None:
        self._model = model
        self._indices = indices
        self.feature_type = model.feature_type
        self.ray_feature_dim = model.ray_feature_dim
        self.n_active_features = model.n_active_features
        self.num_gaussians = int(indices.numel())

    def get_positions(self) -> torch.Tensor:
        return self._model.get_positions()[self._indices]

    def get_rotation(self) -> torch.Tensor:
        return self._model.get_rotation()[self._indices]

    def get_scale(self) -> torch.Tensor:
        return self._model.get_scale()[self._indices]

    def get_density(self) -> torch.Tensor:
        return self._model.get_density()[self._indices]

    def get_features(self) -> torch.Tensor:
        return self._model.get_features()[self._indices]


class SlotMaskSupervision:
    """Per-frame lookup of object masks plus the rendered-alpha penalty they define."""

    def __init__(self, conf, model, train_dataset) -> None:
        config = conf.loss.slot_mask
        controller = model.switch_splat_controller
        if controller is None:
            raise ValueError("loss.use_slot_mask requires a SwitchSplat model")

        self.soft_gating = bool(getattr(config, "soft_gating", False))
        self.num_states = int(controller.num_states)

        self.mask_run = Path(config.mask_run)
        self.mask_key = str(config.mask_key)
        self.other_cameras_empty = bool(config.other_cameras_empty)
        self.use_label_valid = bool(getattr(config, "use_label_valid", False))
        self.mask_dilate_pixels = int(getattr(config, "mask_dilate_pixels", 0))
        if self.mask_dilate_pixels < 0:
            raise ValueError("mask_dilate_pixels must be non-negative")
        self.num_slots = int(controller.num_slots)

        camera_names = list(train_dataset.get_camera_names())
        target = str(config.camera_id)
        if target not in camera_names:
            raise ValueError(f"mask camera {config.camera_id!r} is not one of {camera_names}")
        self.mask_camera_indices = frozenset(
            i for i, name in enumerate(camera_names) if name == target
        )

        if not self.soft_gating:
            self.render_subset = str(conf.render.method).startswith("gsplat")
            assignment = torch.softmax(model.switch_logits.detach().float(), dim=-1).argmax(dim=-1)
            self.state_indices = {
                state: torch.nonzero(assignment == state, as_tuple=False).squeeze(-1)
                for state in range(1, controller.num_states)
            }

        masks, label_valids, self.slots_of_time, self.row_of_time = [], [], {}, {}
        for slot in range(1, self.num_slots + 1):
            path = self.mask_run / f"slot{slot:02d}" / "mask" / "masks.npz"
            if not path.is_file():
                raise FileNotFoundError(f"missing slot masks: {path}")
            bundle = np.load(path)
            frames = bundle[self.mask_key] > 0.5
            if self.use_label_valid:
                labels_path = self.mask_run / f"slot{slot:02d}" / "labels" / "labels.npz"
                if not labels_path.is_file():
                    raise FileNotFoundError(f"missing label validity masks: {labels_path}")
                labels = np.load(labels_path)
                if not np.array_equal(bundle["time_indices"], labels["time_indices"]):
                    raise ValueError(f"mask and label time indices differ for slot {slot}")
                valid_frames = labels["valid"] > 0.5
            for position, time_index in enumerate(bundle["time_indices"]):
                time_index = int(time_index)
                self.slots_of_time.setdefault(time_index, []).append(slot)
                self.row_of_time[(slot, time_index)] = len(masks)
                masks.append(frames[position])
                if self.use_label_valid:
                    label_valids.append(valid_frames[position])
        self.masks = torch.from_numpy(np.stack(masks).astype(np.uint8))
        self.label_valids = (
            torch.from_numpy(np.stack(label_valids).astype(np.uint8))
            if self.use_label_valid else None
        )

    def _render_alpha_hard(self, model, view: Batch, indices: torch.Tensor) -> torch.Tensor:
        if self.render_subset:
            return model.renderer.render(GaussianSubset(model, indices), view, train=True)["pred_opacity"]
        opacity = torch.zeros_like(model.get_density())
        opacity[indices] = model.get_density()[indices]
        return model.renderer.render(model, view, train=True, opacity_override=opacity)["pred_opacity"]

    def _render_alpha_soft(self, model, view: Batch, state: int) -> torch.Tensor:
        probs = torch.softmax(model.switch_logits, dim=-1)[:, state]
        density = model.get_density()
        if density.ndim == 2 and density.shape[-1] == 1:
            opacity = density.squeeze(-1) * probs
        else:
            opacity = density * probs
        return model.renderer.render(model, view, train=True, opacity_override=opacity)["pred_opacity"]

    def describe(self) -> str:
        counts = {}
        for slots in self.slots_of_time.values():
            for slot in slots:
                counts[slot] = counts.get(slot, 0) + 1
        mode = "soft" if self.soft_gating else "hard"
        cam_str = f"mask camera indices {sorted(self.mask_camera_indices)}"
        safety = (
            f"label-valid={self.use_label_valid}, "
            f"mask dilation={self.mask_dilate_pixels}px@{self.masks.shape[-1]}"
        )
        if self.soft_gating:
            return (
                f"slot-mask supervision ({mode}): {len(self.slots_of_time)} masked times "
                f"over {len(counts)} slots, {self.num_states} states, "
                f"{cam_str}, other cameras empty={self.other_cameras_empty}, {safety}"
            )
        sizes = {state: int(index.numel()) for state, index in self.state_indices.items()}
        return (
            f"slot-mask supervision ({mode}): {len(self.slots_of_time)} masked times over {len(counts)} slots, "
            f"{cam_str}, other cameras empty={self.other_cameras_empty}, "
            f"render_subset={self.render_subset}, {safety}, "
            f"state sizes {sizes}"
        )

    def _weight(self, view: Batch, slot: int, time_index: int, camera_idx: int, shape) -> torch.Tensor | None:
        valid = view.mask.to(dtype=torch.float32) if view.mask is not None else None
        if camera_idx in self.mask_camera_indices:
            row = self.row_of_time[(slot, time_index)]
            mask = self.masks[row].to(device=shape.device, dtype=torch.float32)[None, None]
            if self.mask_dilate_pixels:
                radius = self.mask_dilate_pixels
                mask = F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)
            mask = F.interpolate(mask, size=shape.shape[1:3], mode="bilinear", align_corners=False)
            outside = 1.0 - (mask[0, 0] > 0.5).to(torch.float32)
            outside = outside[None, ..., None]
            if self.label_valids is not None:
                label_valid = self.label_valids[row].to(
                    device=shape.device, dtype=torch.float32
                )[None, None]
                label_valid = F.interpolate(label_valid, size=shape.shape[1:3], mode="nearest")
                outside = outside * label_valid[0, 0][None, ..., None]
        elif self.other_cameras_empty:
            outside = torch.ones_like(shape)
        else:
            return None
        return outside if valid is None else outside * valid

    def loss(self, model, batch: Batch) -> torch.Tensor:
        total = torch.zeros((), device=model.device)
        views_with_masks = 0
        for view_index in range(batch.batch_size):
            view = slice_batch(batch, view_index) if batch.batch_size > 1 else batch
            if view.time_idx is None:
                continue
            time_index = int(view.time_idx.reshape(-1)[0])
            slots = self.slots_of_time.get(time_index)
            if not slots:
                continue
            camera_idx = int(torch.as_tensor(view.camera_idx).reshape(-1)[0])
            if camera_idx not in self.mask_camera_indices and not self.other_cameras_empty:
                continue

            slot_losses = []
            for slot in slots:
                state_losses = []
                for state in (2 * slot - 1, 2 * slot):
                    if self.soft_gating:
                        alpha = self._render_alpha_soft(model, view, state)
                    else:
                        indices = self.state_indices[state]
                        if indices.numel() == 0:
                            continue
                        alpha = self._render_alpha_hard(model, view, indices)
                    weight = self._weight(view, slot, time_index, camera_idx, alpha)
                    if weight is not None:
                        state_losses.append((alpha * weight).sum() / weight.sum().clamp_min(1.0))
                if state_losses:
                    slot_losses.append(torch.stack(state_losses).mean())
            if slot_losses:
                # One overlapped photo must not carry two or three times the
                # mask weight of a photo containing only one active slot.
                total = total + torch.stack(slot_losses).mean()
                views_with_masks += 1
        return total / views_with_masks if views_with_masks else total
