# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import (
    BoundedMultiViewDataset,
    get_dataset_world_transform,
    slice_batch,
)
from threedgrut.datasets.utils import (
    DEFAULT_DEVICE,
    MultiEpochsDataLoader,
    PointCloud,
    collate_same_resolution,
)
from threedgrut.model.losses import (
    ssim,
    switch_splat_mode_a_loss,
    switch_splat_spatial_loss,
)
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.optimizers import SelectiveAdam
from threedgrut.render import Renderer
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import check_step_condition, create_summary_writer, jet_map
from threedgrut.utils.render import (
    apply_background,
    apply_feature_decoder,
    apply_post_processing,
)
from threedgrut.utils.timer import CudaTimer

_CAMERA_BATCHED_OUTPUT_KEYS = (
    "pred_features",
    "pred_opacity",
    "pred_dist",
    "pred_normals",
    "hits_count",
)


def _slice_camera_outputs(outputs: dict[str, Any], index: int, batch_size: int) -> dict[str, Any]:
    """Slice camera-indexed render outputs while preserving shared scene outputs."""
    sliced = outputs.copy()
    for key in _CAMERA_BATCHED_OUTPUT_KEYS:
        if key not in sliced:
            continue
        value = sliced[key]
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(f"Output {key!r} must have a leading camera dimension of {batch_size}")
        sliced[key] = value[index : index + 1]
    return sliced


class Trainer3DGRUT:
    """Trainer for paper: "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes" """

    model: MixtureOfGaussians
    """ Gaussian Model """

    train_dataset: BoundedMultiViewDataset
    val_dataset: BoundedMultiViewDataset

    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader

    scene_extent: float = 1.0
    """TODO: Add docstring"""

    scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
    """TODO: Add docstring"""

    strategy: BaseStrategy
    """ Strategy for optimizing the Gaussian model in terms of densification, pruning, etc. """

    gui = None
    """ If GUI is enabled, references the GUI interface """

    criterions: Dict
    """ Contains functors required to compute evaluation metrics, i.e. psnr, ssim, lpips """

    tracking: Dict
    """ Contains all components used to report progress of training """

    post_processing: Optional[nn.Module] = None
    """ Post-processing module """

    post_processing_optimizers: Optional[list] = None
    """ Optimizers for post-processing module """

    post_processing_schedulers: Optional[list] = None
    """ Schedulers for post-processing module optimizers """

    switch_splat_optimizer: Optional[torch.optim.Optimizer] = None
    """ Optimizer for global SwitchSplat window parameters. """

    _distillation_start_step: int = -1
    """ Step at which distillation starts (-1 means disabled) """

    _color_refine_frozen_param_names = frozenset(("positions", "scale", "rotation", "density"))
    """ Gaussian optimizer parameter groups frozen during NHT color refinement """

    @staticmethod
    def create_from_checkpoint(resume: str, conf: DictConfig):
        """Create a new trainer from a checkpoint file"""

        conf.resume = resume
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ply(ply_path: str, conf: DictConfig):
        """Create a new trainer from a PLY file"""

        conf.resume = ""
        conf.import_ply.enabled = True
        conf.import_ply.path = ply_path
        return Trainer3DGRUT(conf)

    @torch.cuda.nvtx.range("setup-trainer")
    def __init__(self, conf: DictConfig, device=None):
        """Set up a new training session, or continue an existing one based on configuration"""

        self._validate_batching_configuration(conf)

        # Keep track of useful fields
        self.conf = conf
        """ Global configuration of model, scene, optimization, etc"""
        self.device = device if device is not None else DEFAULT_DEVICE
        """ Device used for training and visualizations """
        self.global_step = 0
        """ Current global iteration of the trainer """
        self.n_iterations = conf.n_iterations
        """ Total number of train iterations to take (for multiple passes over the dataset) """
        self.n_epochs = 0
        """ Total number of train epochs / passes, e.g. single pass over the dataset."""
        self.val_frequency = conf.val_frequency
        """ Validation frequency, in terms on global steps """
        self._color_refine_start_step = self._get_color_refine_start_step(conf)
        """ Step at which NHT color refinement starts """
        self._in_color_refine = False
        """ Whether NHT color refinement is active """

        # Setup the trainer and components
        logger.log_rule("Load Datasets")
        self.init_dataloaders(conf)
        self._configure_switch_splat_timeline(conf)
        self.init_scene_extents(self.train_dataset)
        logger.log_rule("Initialize Model")
        self.init_model(conf, self.scene_extent)
        self.init_densification_and_pruning_strategy(conf)
        logger.log_rule("Setup Model Weights & Training")
        self.init_metrics()
        # Feature decoder and post-processing must exist before setup_training so resume can load their state.
        self.init_feature_decoder(conf)
        self.init_post_processing(conf)
        self.setup_training(conf, self.model, self.train_dataset)
        self.init_experiments_tracking(conf)
        self.init_gui(conf, self.model, self.train_dataset, self.val_dataset, self.scene_bbox)

    @staticmethod
    def _validate_batching_configuration(conf: DictConfig) -> None:
        batch_size = int(OmegaConf.select(conf, "batch_size", default=1))
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if batch_size > 1 and OmegaConf.select(conf, "post_processing.method") is not None:
            raise ValueError(
                "Post-processing with batch_size > 1 is not supported yet because its camera/frame "
                "controllers are single-view. Use post_processing.method=null or batch_size=1."
            )

    def _configure_switch_splat_timeline(self, conf: DictConfig) -> None:
        if not bool(OmegaConf.select(conf, "switch_splat.enabled", default=False)):
            return

        windows_file = OmegaConf.select(conf, "switch_splat.init_windows_file", default=None)
        if windows_file:
            windows_path = Path(str(windows_file)).expanduser()
            if not windows_path.is_absolute():
                windows_path = Path(conf.path) / windows_path
            if not windows_path.is_file():
                raise FileNotFoundError(f"SwitchSplat window file not found: {windows_path}")
            if windows_path.stat().st_size == 0:
                raise ValueError(f"SwitchSplat window file is empty: {windows_path}")
            windows = np.atleast_2d(np.loadtxt(windows_path, dtype=np.float64))
        else:
            windows_path = None
            inline_windows = OmegaConf.select(conf, "switch_splat.init_windows", default=None)
            windows = np.asarray([] if inline_windows is None else inline_windows, dtype=np.float64)

        if windows.size == 0:
            if windows_path is not None:
                raise ValueError(f"SwitchSplat window file is empty: {windows_path}")
            conf.switch_splat.num_slots = 0
            conf.switch_splat.init_windows = []
            return
        if windows.ndim != 2 or windows.shape[1] != 2 or not np.all(np.isfinite(windows)):
            raise ValueError("SwitchSplat initial windows must be a finite array with shape [K,2]")
        if np.any(windows[:, 0] > windows[:, 1]):
            raise ValueError("SwitchSplat initial window starts must not exceed their ends")

        time_indices = []
        for split, dataset in (("train", self.train_dataset), ("test", self.val_dataset)):
            split_times = getattr(dataset, "time_indices", None)
            if split_times is None:
                raise ValueError(f"SwitchSplat requires raw numeric filename times in the {split} dataset")
            time_indices.extend(np.asarray(split_times, dtype=np.int64).reshape(-1).tolist())
        if not time_indices or min(time_indices) < 0:
            raise ValueError("SwitchSplat time indices must be non-negative")

        num_frames = max(time_indices) + 1
        if np.any(windows < 0) or np.any(windows > num_frames - 1):
            raise ValueError(f"SwitchSplat initial windows must lie within [0,{num_frames - 1}]")
        configured_num_frames = OmegaConf.select(conf, "switch_splat.num_frames", default=None)
        if configured_num_frames is not None and int(configured_num_frames) != num_frames:
            raise ValueError(
                f"switch_splat.num_frames={configured_num_frames} does not match max(time_idx) + 1 = {num_frames}"
            )
        conf.switch_splat.num_slots = len(windows)
        conf.switch_splat.init_windows = windows.tolist()
        conf.switch_splat.num_frames = num_frames
        logger.info(
            f"SwitchSplat initialized {len(windows)} slots from "
            f"{windows_path if windows_path is not None else 'inline config'}: {windows.tolist()}"
        )

    def _get_color_refine_start_step(self, conf: DictConfig) -> int:
        """Return the first step of the NHT color-only refinement phase."""
        feature_type = str(OmegaConf.select(conf, "model.feature_type", default="sh")).lower()
        if feature_type != "nht":
            return conf.n_iterations

        color_refine_steps = int(OmegaConf.select(conf, "model.nht_decoder.color_refine_steps", default=0) or 0)
        if color_refine_steps <= 0:
            return conf.n_iterations

        return max(0, conf.n_iterations - color_refine_steps)

    def _is_color_refine_active(self, global_step: int) -> bool:
        return global_step >= self._color_refine_start_step and self._color_refine_start_step < self.conf.n_iterations

    def _apply_color_refine_freeze(self, global_step: int) -> None:
        """Freeze Gaussian geometry/opacity optimizer groups while colors keep training."""
        if not self._is_color_refine_active(global_step):
            return

        if not self._in_color_refine:
            self._in_color_refine = True
            self.strategy.suspend()
            logger.info(
                f"🎨 [step {global_step}] Entering NHT color refinement: "
                "freezing geometry + opacity and disabling scale/opacity regularization."
            )

        if self.model.optimizer is None:
            return

        for param_group in self.model.optimizer.param_groups:
            if param_group.get("name") in self._color_refine_frozen_param_names:
                param_group["lr"] = 0.0

    def _zero_color_refine_frozen_grads(self) -> None:
        if not self._in_color_refine or self.model.optimizer is None:
            return

        for param_group in self.model.optimizer.param_groups:
            if param_group.get("name") in self._color_refine_frozen_param_names:
                for param in param_group["params"]:
                    param.grad = None

    def init_dataloaders(self, conf: DictConfig):
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        train_dataset, val_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        batch_size = int(OmegaConf.select(conf, "batch_size", default=1))
        train_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": batch_size,
                "shuffle": True,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
                "collate_fn": collate_same_resolution,
            }
        )

        val_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": False,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
                "collate_fn": collate_same_resolution,
            }
        )

        train_dataloader = MultiEpochsDataLoader(train_dataset, **train_dataloader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

    def teardown_dataloaders(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.val_dataloader is not None:
            del self.val_dataloader
        if self.train_dataset is not None:
            del self.train_dataset
        if self.val_dataset is not None:
            del self.val_dataset

    def init_scene_extents(self, train_dataset: BoundedMultiViewDataset) -> None:
        scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
        scene_extent = train_dataset.get_scene_extent()
        scene_bbox = train_dataset.get_scene_bbox()
        self.scene_extent = scene_extent
        self.scene_bbox = scene_bbox

    def init_model(self, conf: DictConfig, scene_extent=None) -> None:
        """Initializes the gaussian model and the optix context"""
        self.model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    def init_densification_and_pruning_strategy(self, conf: DictConfig) -> None:
        """Set pre-train / post-train iteration logic. i.e. densification and pruning"""
        assert self.model is not None
        match self.conf.strategy.method:
            case "GSStrategy":
                from threedgrut.strategy.gs import GSStrategy

                self.strategy = GSStrategy(conf, self.model)
                logger.info("🔆 Using GS strategy")
            case "MCMCStrategy":
                from threedgrut.strategy.mcmc import MCMCStrategy

                self.strategy = MCMCStrategy(conf, self.model)
                logger.info("🔆 Using MCMC strategy")
            case _:
                raise ValueError(f"unrecognized model.strategy {conf.strategy.method}")

    def setup_training(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
    ):
        """
        Performs required steps to setup the optimization:
        1. Initialize the gaussian model fields: load previous weights from checkpoint, or initialize from scratch.
        2. Build BVH acceleration structure for gaussian model, if not loaded with checkpoint
        3. Set up the optimizer to optimize the gaussian model params
        4. Initialize the densification buffers in the densificaiton strategy
        """

        resume_checkpoint = None

        # Initialize
        if conf.resume:  # Load a checkpoint
            logger.info(f"🤸 Loading a pretrained checkpoint from {conf.resume}!")
            checkpoint = torch.load(conf.resume, weights_only=False)
            resume_checkpoint = checkpoint
            model.init_from_checkpoint(checkpoint)
            self.strategy.init_densification_buffer(checkpoint)
            global_step = checkpoint["global_step"]

            # Restore feature decoder state (skip if architecture drifted vs checkpoint)
            if "feature_decoder" in checkpoint and self.feature_decoder is not None:
                fd_ckpt = checkpoint["feature_decoder"]
                self.feature_decoder.load_state_dict(fd_ckpt["module"])
                self.feature_decoder_optimizer.load_state_dict(fd_ckpt["optimizer"])
                self.feature_decoder_scheduler.load_state_dict(fd_ckpt["scheduler"])
                ema_state = fd_ckpt.get("ema")
                if ema_state is not None:
                    self.feature_decoder.load_ema_state_dict(ema_state)
                logger.info("🎨 Feature decoder state restored from checkpoint")

            # Restore post-processing state
            if "post_processing" in checkpoint and self.post_processing is not None:
                self.post_processing.load_state_dict(checkpoint["post_processing"]["module"])
                for opt, opt_state in zip(
                    self.post_processing_optimizers,
                    checkpoint["post_processing"]["optimizers"],
                ):
                    opt.load_state_dict(opt_state)
                for sched, sched_state in zip(
                    self.post_processing_schedulers,
                    checkpoint["post_processing"]["schedulers"],
                ):
                    sched.load_state_dict(sched_state)
                logger.info("📷 Post-processing state restored from checkpoint")
            model.build_acc()
        elif conf.import_ply.enabled:
            ply_path = (
                conf.import_ply.path
                if conf.import_ply.path
                else f"{conf.out_dir}/{conf.experiment_name}/export_last.ply"
            )
            logger.info(f"Loading a ply model from {ply_path}!")
            model.init_from_ply(ply_path)
            self.strategy.init_densification_buffer()
            model.build_acc()
            global_step = conf.import_ply.init_global_step
        else:
            logger.info("🤸 Initiating new 3dgrut training..")
            initialization_method = conf.initialization.method
            points_transform = (
                get_dataset_world_transform(train_dataset)
                if initialization_method in {"colmap", "fused_point_cloud", "point_cloud"}
                else None
            )

            match initialization_method:
                case "random":
                    model.init_from_random_point_cloud(
                        num_gaussians=conf.initialization.num_gaussians,
                        xyz_max=conf.initialization.xyz_max,
                        xyz_min=conf.initialization.xyz_min,
                    )
                case "colmap":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    model.init_from_colmap(conf.path, observer_points, points_transform=points_transform)
                case "fused_point_cloud":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    ply_path = conf.initialization.fused_point_cloud_path
                    logger.info(f"Initializing from accumulated point cloud: {ply_path}")
                    model.init_from_fused_point_cloud(
                        ply_path,
                        observer_points,
                        points_transform=points_transform,
                    )
                case "point_cloud":
                    if points_transform is not None:
                        raise ValueError(
                            "initialization.method=point_cloud loads complete Gaussian geometry and cannot safely "
                            "apply the dataset world transform. Use COLMAP/fused-point initialization, disable "
                            "world normalization, or import a model already expressed in normalized coordinates."
                        )
                    try:
                        ply_path = os.path.join(conf.path, "point_cloud.ply")
                        model.init_from_pretrained_point_cloud(ply_path)
                    except FileNotFoundError as e:
                        logger.error(e)
                        raise e
                case "checkpoint":
                    checkpoint = torch.load(conf.initialization.path, weights_only=False)
                    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
                case "lidar":
                    assert isinstance(
                        train_dataset, datasets.NCoreDataset
                    ), "can only initialize from lidar with NCoreDataset"
                    pc = PointCloud.from_sequence(
                        list(train_dataset.get_point_clouds(step_frame=1, non_dynamic_points_only=True)),
                        device="cpu",
                    )
                    if conf.initialization.num_points < len(pc.xyz_end):
                        # Deterministically random subsample points if there are more points than the specified number of gaussians
                        rng = torch.Generator().manual_seed(conf.seed_initialization)
                        idxs = torch.randperm(len(pc.xyz_end), generator=rng)[: conf.initialization.num_points]
                        pc = pc.selected_idxs(idxs)
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    model.init_from_lidar(pc, observer_points)
                case _:
                    raise ValueError(
                        f"unrecognized initialization.method {initialization_method}, choose from "
                        "[colmap, fused_point_cloud, point_cloud, random, checkpoint, lidar]"
                    )

            self.strategy.init_densification_buffer()

            model.build_acc()
            model.setup_optimizer()
            global_step = 0

        self._setup_switch_splat_optimizer(resume_checkpoint)

        self.global_step = global_step
        steps_per_epoch = len(self.train_dataloader)
        self.n_epochs = int((conf.n_iterations + steps_per_epoch - 1) / steps_per_epoch)

    def _setup_switch_splat_optimizer(self, checkpoint: Optional[dict] = None) -> None:
        if self.model.switch_splat_controller is None:
            self.switch_splat_optimizer = None
            return

        controller = self.model.switch_splat_controller
        self.switch_splat_optimizer = torch.optim.Adam(
            [
                {
                    "params": [controller.endpoints_raw],
                    "name": "endpoints_raw",
                    "lr": float(self.conf.switch_splat.endpoints_lr),
                },
                {
                    "params": [controller.temperature_raw],
                    "name": "temperature_raw",
                    "lr": float(self.conf.switch_splat.temperature_lr),
                },
            ],
            lr=0.0,
            eps=float(self.conf.optimizer.eps),
        )
        if checkpoint is None:
            return
        optimizer_state = checkpoint.get("switch_splat_optimizer")
        if optimizer_state is not None:
            self.switch_splat_optimizer.load_state_dict(optimizer_state)
        elif "switch_splat" in checkpoint:
            raise ValueError("enabled SwitchSplat checkpoint is missing its window optimizer state")

    @torch.no_grad()
    def _maybe_soften_switch_logits(self, step: int) -> None:
        config = OmegaConf.select(self.conf, "switch_splat.logit_softening", default=None)
        if config is None or not config.enabled or step not in config.iterations:
            return

        logits = self.model.switch_logits
        logits.copy_((logits - logits.mean(dim=1, keepdim=True)) * float(config.factor))
        for name, value in self.model.optimizer.state.get(logits, {}).items():
            if name != "step" and torch.is_tensor(value):
                value.zero_()
        logger.info(f"Step {step}: softened SwitchSplat logits by factor={config.factor}")

    def init_gui(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
        val_dataset: BoundedMultiViewDataset,
        scene_bbox,
    ):
        gui = None

        feature_decoder = getattr(self, "feature_decoder", None)
        if conf.with_gui:
            from threedgrut.utils.gui import GUI

            gui = GUI(conf, model, train_dataset, val_dataset, scene_bbox, feature_decoder=feature_decoder)

        elif conf.with_viser_gui:
            from threedgrut.utils.viser_gui_util import ViserGUI

            gui = ViserGUI(conf, model, train_dataset, val_dataset, scene_bbox, feature_decoder=feature_decoder)

        self.gui = gui

    def init_metrics(self):
        self.criterions = Dict(
            psnr=PeakSignalNoiseRatio(data_range=1).to(self.device),
            ssim=StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            lpips=LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(self.device),
        )

    def init_experiments_tracking(self, conf: DictConfig):
        # Initialize the tensorboard writer
        object_name = Path(conf.path).stem
        writer, out_dir, run_name = create_summary_writer(
            conf, object_name, conf.out_dir, conf.experiment_name, conf.use_wandb
        )
        logger.info(f"📊 Training logs & will be saved to: {out_dir}")

        # Store parsed config for reference
        with open(os.path.join(out_dir, "parsed.yaml"), "w") as fp:
            OmegaConf.save(config=conf, f=fp)

        # Pack all components used to track progress of training
        self.tracking = Dict(
            writer=writer,
            run_name=run_name,
            object_name=object_name,
            output_dir=out_dir,
        )

    def init_post_processing(self, conf: DictConfig):
        """Initialize post-processing module based on config."""
        method = conf.post_processing.method

        if method is None:
            return

        if method == "ppisp":
            from ppisp import PPISP, PPISPConfig

            frames_per_camera = self.train_dataset.get_frames_per_camera()
            num_cameras = len(frames_per_camera)
            num_frames = sum(frames_per_camera)

            use_controller = conf.post_processing.get("use_controller", True)

            # Distillation mode: controller activates after main training
            # Total iterations = n_iterations, distillation starts at n_iterations - n_distillation_steps
            n_distillation_steps = conf.post_processing.get("n_distillation_steps", 5000)
            if use_controller and n_distillation_steps > 0:
                main_training_steps = conf.n_iterations - n_distillation_steps
                controller_activation_ratio = main_training_steps / conf.n_iterations
                controller_distillation = True
                self._distillation_start_step = main_training_steps
                logger.info(f"📷 PPISP distillation mode: controller activates at step {main_training_steps}")
            elif use_controller:
                controller_activation_ratio = 0.8
                controller_distillation = False
                self._distillation_start_step = -1
            else:
                controller_activation_ratio = 0.0
                controller_distillation = False
                self._distillation_start_step = -1

            ppisp_config = PPISPConfig(
                use_controller=use_controller,
                controller_distillation=controller_distillation,
                controller_activation_ratio=controller_activation_ratio,
            )

            self.post_processing = PPISP(
                num_cameras=num_cameras,
                num_frames=num_frames,
                config=ppisp_config,
            ).to(self.device)

            self.post_processing_optimizers = self.post_processing.create_optimizers()
            self.post_processing_schedulers = self.post_processing.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=conf.n_iterations,
            )

            logger.info(f"📷 {method.upper()} initialized: {num_cameras} cameras, {num_frames} frames")
        elif method == "linear-to-srgb":
            from threedgrut.utils.post_processing_linear_to_srgb import (
                LinearToSrgbPostProcessing,
            )

            self.post_processing = LinearToSrgbPostProcessing().to(self.device)
            self.post_processing_optimizers = []
            self.post_processing_schedulers = []
            logger.info("Post-processing: linear-to-sRGB (no trainable parameters)")
        else:
            raise ValueError(f"Unknown post-processing method: {method}")

    def init_feature_decoder(self, conf: DictConfig):
        """Initialize feature decoder for learned features mode."""
        from threedgrut.model.features import Features

        if self.model.feature_type != Features.Type.NHT:
            self.feature_decoder = None
            self.feature_decoder_optimizer = None
            self.feature_decoder_scheduler = None
            return

        dec_conf = conf.model.nht_decoder
        if not getattr(dec_conf, "enabled", True):
            self.feature_decoder = None
            self.feature_decoder_optimizer = None
            self.feature_decoder_scheduler = None
            return

        from threedgrut.model.feature_decoder import FeatureDecoder

        ray_feature_dim = self.model.ray_feature_dim
        dec = conf.model.nht_decoder
        hidden_dim = dec.hidden_dim
        num_layers = getattr(dec, "num_layers", 4)
        dir_encoding = getattr(dec, "dir_encoding", "SphericalHarmonics")
        dir_encoding_degree = getattr(dec, "dir_encoding_degree", 3)
        sh_scale = getattr(dec, "sh_scale", 1.0)
        output_activation = getattr(dec, "output_activation", "Sigmoid")
        unpremultiply_alpha = getattr(dec, "unpremultiply_alpha", False)
        ema_decay = getattr(dec_conf, "ema_decay", 0.0)
        ema_start_step = getattr(dec_conf, "ema_start_step", 0)
        logger.info(f"Initializing FeatureDecoder: {ray_feature_dim} -> 3 RGB")
        self.feature_decoder = FeatureDecoder(
            ray_feature_dim=ray_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dir_encoding=dir_encoding,
            dir_encoding_degree=dir_encoding_degree,
            sh_scale=sh_scale,
            output_activation=output_activation,
            ema_decay=ema_decay,
            ema_start_step=ema_start_step,
            unpremultiply_alpha=unpremultiply_alpha,
        ).to(self.device)

        lr = dec.learning_rate
        weight_decay = getattr(dec, "reg_weight", 0.0)
        self.feature_decoder_optimizer = torch.optim.Adam(
            self.feature_decoder.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        scheduler_conf = dec.scheduler
        max_steps = int(getattr(scheduler_conf, "max_steps", getattr(conf, "n_iterations", 30000)))
        decay_final = float(getattr(scheduler_conf, "decay_final", 0.001))
        if scheduler_conf.type == "exponential":
            gamma = decay_final ** (1.0 / max_steps)
            self.feature_decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.feature_decoder_optimizer,
                gamma=gamma,
            )
        elif scheduler_conf.type == "cosine":
            eta_min = lr * decay_final
            self.feature_decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.feature_decoder_optimizer,
                T_max=max_steps,
                eta_min=eta_min,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_conf.type}")

        if ema_decay > 0:
            logger.info(f"🎨 FeatureDecoder EMA: decay={ema_decay}, start_step={ema_start_step}")
        logger.info(
            f"🎨 FeatureDecoder optimizer: lr={lr}, " f"weight_decay={weight_decay}, scheduler={scheduler_conf.type}"
        )

    @torch.cuda.nvtx.range("get_metrics")
    def get_metrics(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        losses: dict[str, torch.Tensor],
        profilers: dict[str, CudaTimer],
        split: str = "training",
        iteration: Optional[int] = None,
    ) -> dict[str, Union[int, float]]:
        """Computes dictionary of single batch metrics based on current batch output.
        Args:
            gpu_batch: GT data of current batch
            output: model prediction for current batch
            losses: dictionary of loss terms computed for current batch
            split: name of split metrics are computed for - 'training' or 'validation'
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        step = self.global_step

        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_features"]

        rgb_gt = gpu_batch.masked(rgb_gt)
        rgb_pred = gpu_batch.masked(rgb_pred)

        psnr = self.criterions["psnr"]
        ssim = self.criterions["ssim"]
        lpips = self.criterions["lpips"]

        # Move losses to cpu once
        metrics["losses"] = {k: v.detach().item() for k, v in losses.items()}

        is_compute_train_hit_metrics = (split == "training") and (step % self.conf.writer.hit_stat_frequency == 0)
        is_compute_validation_metrics = split == "validation"

        if is_compute_train_hit_metrics or is_compute_validation_metrics:
            metrics["hits_mean"] = outputs["hits_count"].mean().item()
            metrics["hits_std"] = outputs["hits_count"].std().item()
            metrics["hits_min"] = outputs["hits_count"].min().item()
            metrics["hits_max"] = outputs["hits_count"].max().item()

        if is_compute_validation_metrics:
            with torch.cuda.nvtx.range(f"criterions_psnr"):
                metrics["psnr"] = psnr(rgb_pred, rgb_gt).item()

            rgb_gt_full = rgb_gt.permute(0, 3, 1, 2)
            pred_features_full = rgb_pred.permute(0, 3, 1, 2)
            pred_features_full_clipped = rgb_pred.clip(0, 1).permute(0, 3, 1, 2)

            with torch.cuda.nvtx.range(f"criterions_ssim"):
                metrics["ssim"] = ssim(pred_features_full, rgb_gt_full).item()
            with torch.cuda.nvtx.range(f"criterions_lpips"):
                metrics["lpips"] = lpips(pred_features_full_clipped, rgb_gt_full).item()

            if iteration in self.conf.writer.log_image_views:
                metrics["img_hit_counts"] = jet_map(outputs["hits_count"][-1], self.conf.writer.max_num_hits)
                metrics["img_gt"] = gpu_batch.rgb_gt[-1].clip(0, 1.0)
                metrics["img_pred"] = outputs["pred_features"][-1].clip(0, 1.0)
                metrics["img_pred_dist"] = jet_map(outputs["pred_dist"][-1], 100)
                metrics["img_pred_opacity"] = jet_map(outputs["pred_opacity"][-1], 1)

        if profilers:
            timings = {}
            for key, timer in profilers.items():
                if timer.enabled:
                    timings[key] = timer.timing()
            if timings:
                metrics["timings"] = timings

        return metrics

    @torch.cuda.nvtx.range("get_losses")
    def get_losses(
        self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Computes dictionary of losses for current batch.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
        Returns:
            losses: dictionary of loss terms computed for current batch.
        """
        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_features"]
        rgb_gt = gpu_batch.masked(rgb_gt)
        rgb_pred = gpu_batch.masked(rgb_pred)

        # L1 loss
        loss_l1 = torch.zeros(1, device=self.device)
        lambda_l1 = 0.0
        if self.conf.loss.use_l1:
            with torch.cuda.nvtx.range(f"loss-l1"):
                loss_l1 = torch.abs(rgb_pred - rgb_gt).mean()
                lambda_l1 = self.conf.loss.lambda_l1

        # L2 loss
        loss_l2 = torch.zeros(1, device=self.device)
        lambda_l2 = 0.0
        if self.conf.loss.use_l2:
            with torch.cuda.nvtx.range(f"loss-l2"):
                loss_l2 = torch.nn.functional.mse_loss(rgb_pred, rgb_gt)
                lambda_l2 = self.conf.loss.lambda_l2

        # DSSIM loss
        loss_ssim = torch.zeros(1, device=self.device)
        lambda_ssim = 0.0
        if self.conf.loss.use_ssim:
            with torch.cuda.nvtx.range(f"loss-ssim"):
                rgb_gt_full = torch.permute(rgb_gt, (0, 3, 1, 2))
                pred_features_full = torch.permute(rgb_pred, (0, 3, 1, 2))
                loss_ssim = 1.0 - ssim(pred_features_full, rgb_gt_full)
                lambda_ssim = self.conf.loss.lambda_ssim

        # Opacity regularization
        loss_opacity = torch.zeros(1, device=self.device)
        lambda_opacity = 0.0
        if self.conf.loss.use_opacity and not self._in_color_refine:
            with torch.cuda.nvtx.range(f"loss-opacity"):
                loss_opacity = torch.abs(self.model.get_density()).mean()
                lambda_opacity = self.conf.loss.lambda_opacity

        # Scale regularization
        loss_scale = torch.zeros(1, device=self.device)
        lambda_scale = 0.0
        if self.conf.loss.use_scale and not self._in_color_refine:
            with torch.cuda.nvtx.range(f"loss-scale"):
                loss_scale = torch.abs(self.model.get_scale()).mean()
                lambda_scale = self.conf.loss.lambda_scale

        loss_mode_a = torch.zeros(1, device=self.device)
        lambda_mode_a = 0.0
        if bool(OmegaConf.select(self.conf, "loss.use_mode_a", default=False)):
            multiplier = self._loss_schedule_multiplier("loss.mode_a_schedule")
            if multiplier > 0:
                loss_mode_a = switch_splat_mode_a_loss(self.model.switch_logits)
                lambda_mode_a = float(self.conf.loss.lambda_mode_a) * multiplier

        loss_spatial = torch.zeros(1, device=self.device)
        lambda_spatial = 0.0
        if bool(OmegaConf.select(self.conf, "loss.use_slot_spatial_conc", default=False)):
            multiplier = self._loss_schedule_multiplier("loss.slot_spatial_conc_schedule")
            if multiplier > 0:
                loss_spatial = switch_splat_spatial_loss(
                    self.model.positions,
                    self.model.get_density().squeeze(-1),
                    self.model.switch_logits,
                )
                lambda_spatial = float(self.conf.loss.lambda_slot_spatial_conc) * multiplier

        loss_slot_mask = torch.zeros(1, device=self.device)
        lambda_slot_mask = 0.0
        if bool(OmegaConf.select(self.conf, "loss.use_slot_mask", default=False)):
            multiplier = self._loss_schedule_multiplier("loss.slot_mask_schedule")
            if multiplier > 0:
                loss_slot_mask = self._slot_mask_supervision().loss(self.model, gpu_batch)
                lambda_slot_mask = float(self.conf.loss.lambda_slot_mask) * multiplier

        # Total loss
        loss = (
            lambda_l1 * loss_l1
            + lambda_ssim * loss_ssim
            + lambda_opacity * loss_opacity
            + lambda_scale * loss_scale
            + lambda_mode_a * loss_mode_a
            + lambda_spatial * loss_spatial
            + lambda_slot_mask * loss_slot_mask
        )
        return dict(
            total_loss=loss,
            l1_loss=lambda_l1 * loss_l1,
            l2_loss=lambda_l2 * loss_l2,
            ssim_loss=lambda_ssim * loss_ssim,
            opacity_loss=lambda_opacity * loss_opacity,
            scale_loss=lambda_scale * loss_scale,
            mode_a_loss=lambda_mode_a * loss_mode_a,
            slot_spatial_conc_loss=lambda_spatial * loss_spatial,
            slot_mask_loss=lambda_slot_mask * loss_slot_mask,
        )

    def _slot_mask_supervision(self):
        """Mask supervisor for the fine-tuning stage, built on first use and then cached."""
        supervision = getattr(self, "_slot_mask_supervision_cache", None)
        if supervision is None:
            from threedgrut.utils.slot_mask_supervision import SlotMaskSupervision

            supervision = SlotMaskSupervision(self.conf, self.model, self.train_dataset)
            logger.info(f"\U0001f3ad {supervision.describe()}")
            self._slot_mask_supervision_cache = supervision
        return supervision

    def _loss_schedule_multiplier(self, schedule_path: str) -> float:
        schedule = OmegaConf.select(self.conf, schedule_path)
        start = int(schedule.start_iteration)
        end = int(schedule.end_iteration)
        if self.global_step < start or (end >= 0 and self.global_step >= end):
            return 0.0
        ramp = int(schedule.ramp_iterations)
        return min((self.global_step - start + 1) / ramp, 1.0) if ramp > 0 else 1.0

    @torch.cuda.nvtx.range("log_validation_iter")
    def log_validation_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single validation iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        logger.log_progress(
            task_name="Validation",
            advance=1,
            iteration=f"{str(iteration)}",
            psnr=batch_metrics["psnr"],
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range("log_validation_pass")
    def log_validation_pass(self, metrics: dict[str, Any]) -> None:
        """Log information after a single validation pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if "img_pred" in metrics:
            writer.add_images(
                "image/pred/val",
                torch.stack(metrics["img_pred"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_gt" in metrics:
            writer.add_images(
                "image/gt",
                torch.stack(metrics["img_gt"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_hit_counts" in metrics:
            writer.add_images(
                "image/hit_counts/val",
                torch.stack(metrics["img_hit_counts"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_pred_dist" in metrics:
            writer.add_images(
                "image/dist/val",
                torch.stack(metrics["img_pred_dist"]),
                global_step,
                dataformats="NHWC",
            )
        if "img_pred_opacity" in metrics:
            writer.add_images(
                "image/opacity/val",
                torch.stack(metrics["img_pred_opacity"]),
                global_step,
                dataformats="NHWC",
            )

        mean_timings = {}
        if "timings" in metrics:
            for time_key in metrics["timings"]:
                mean_timings[time_key] = np.mean(metrics["timings"][time_key])
                writer.add_scalar("time/" + time_key + "/val", mean_timings[time_key], global_step)

        writer.add_scalar("num_particles/val", self.model.num_gaussians, self.global_step)

        mean_psnr = np.mean(metrics["psnr"])
        writer.add_scalar("psnr/val", mean_psnr, global_step)
        writer.add_scalar("ssim/val", np.mean(metrics["ssim"]), global_step)
        writer.add_scalar("lpips/val", np.mean(metrics["lpips"]), global_step)
        writer.add_scalar("hits/min/val", np.mean(metrics["hits_min"]), global_step)
        writer.add_scalar("hits/max/val", np.mean(metrics["hits_max"]), global_step)
        writer.add_scalar("hits/mean/val", np.mean(metrics["hits_mean"]), global_step)

        loss = np.mean(metrics["losses"]["total_loss"])
        writer.add_scalar("loss/total/val", loss, global_step)
        if self.conf.loss.use_l1:
            l1_loss = np.mean(metrics["losses"]["l1_loss"])
            writer.add_scalar("loss/l1/val", l1_loss, global_step)
        if self.conf.loss.use_l2:
            l2_loss = np.mean(metrics["losses"]["l2_loss"])
            writer.add_scalar("loss/l2/val", l2_loss, global_step)
        if self.conf.loss.use_ssim:
            ssim_loss = np.mean(metrics["losses"]["ssim_loss"])
            writer.add_scalar("loss/ssim/val", ssim_loss, global_step)

        table = {k: np.mean(v) for k, v in metrics.items() if k in ("psnr", "ssim", "lpips")}
        for time_key in mean_timings:
            table[time_key] = f"{'{:.2f}'.format(mean_timings[time_key])}" + " ms/it"
        logger.log_table(f"📊 Validation Metrics - Step {global_step}", record=table)

    @torch.cuda.nvtx.range(f"log_training_iter")
    def log_training_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single training iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if self.conf.enable_writer and global_step > 0 and global_step % self.conf.log_frequency == 0:
            loss = np.mean(batch_metrics["losses"]["total_loss"])
            writer.add_scalar("loss/total/train", loss, global_step)
            if self.conf.loss.use_l1:
                l1_loss = np.mean(batch_metrics["losses"]["l1_loss"])
                writer.add_scalar("loss/l1/train", l1_loss, global_step)
            if self.conf.loss.use_l2:
                l2_loss = np.mean(batch_metrics["losses"]["l2_loss"])
                writer.add_scalar("loss/l2/train", l2_loss, global_step)
            if self.conf.loss.use_ssim:
                ssim_loss = np.mean(batch_metrics["losses"]["ssim_loss"])
                writer.add_scalar("loss/ssim/train", ssim_loss, global_step)
            if self.conf.loss.use_opacity:
                opacity_loss = np.mean(batch_metrics["losses"]["opacity_loss"])
                writer.add_scalar("loss/opacity/train", opacity_loss, global_step)
            if self.conf.loss.use_scale:
                scale_loss = np.mean(batch_metrics["losses"]["scale_loss"])
                writer.add_scalar("loss/scale/train", scale_loss, global_step)
            if bool(OmegaConf.select(self.conf, "loss.use_mode_a", default=False)):
                writer.add_scalar("loss/mode_a/train", batch_metrics["losses"]["mode_a_loss"], global_step)
            if bool(OmegaConf.select(self.conf, "loss.use_slot_spatial_conc", default=False)):
                writer.add_scalar(
                    "loss/slot_spatial_conc/train",
                    batch_metrics["losses"]["slot_spatial_conc_loss"],
                    global_step,
                )
            if bool(OmegaConf.select(self.conf, "loss.use_slot_mask", default=False)):
                writer.add_scalar(
                    "loss/slot_mask/train",
                    batch_metrics["losses"]["slot_mask_loss"],
                    global_step,
                )
            if self.post_processing is not None and "post_processing_reg_loss" in batch_metrics["losses"]:
                post_processing_reg_loss = np.mean(batch_metrics["losses"]["post_processing_reg_loss"])
                writer.add_scalar(
                    "loss/post_processing_reg/train",
                    post_processing_reg_loss,
                    global_step,
                )
            if self._color_refine_start_step < self.conf.n_iterations:
                writer.add_scalar("train/color_refine", float(self._in_color_refine), global_step)
            if "psnr" in batch_metrics:
                writer.add_scalar("psnr/train", batch_metrics["psnr"], self.global_step)
            if "ssim" in batch_metrics:
                writer.add_scalar("ssim/train", batch_metrics["ssim"], self.global_step)
            if "lpips" in batch_metrics:
                writer.add_scalar("lpips/train", batch_metrics["lpips"], self.global_step)
            if "hits_mean" in batch_metrics:
                writer.add_scalar("hits/mean/train", batch_metrics["hits_mean"], self.global_step)
            if "hits_std" in batch_metrics:
                writer.add_scalar("hits/std/train", batch_metrics["hits_std"], self.global_step)
            if "hits_min" in batch_metrics:
                writer.add_scalar("hits/min/train", batch_metrics["hits_min"], self.global_step)
            if "hits_max" in batch_metrics:
                writer.add_scalar("hits/max/train", batch_metrics["hits_max"], self.global_step)

            if "timings" in batch_metrics:
                for time_key in batch_metrics["timings"]:
                    writer.add_scalar(
                        "time/" + time_key + "/train",
                        batch_metrics["timings"][time_key],
                        self.global_step,
                    )

            writer.add_scalar("num_particles/train", self.model.num_gaussians, self.global_step)
            writer.add_scalar("train/num_GS", self.model.num_gaussians, self.global_step)

            # # NOTE: hack to easily compare with 3DGS
            # writer.add_scalar("train_loss_patches/total_loss", loss, global_step)
            # writer.add_scalar("gaussians/count", self.model.num_gaussians, self.global_step)

        logger.log_progress(
            task_name="Training",
            advance=1,
            step=f"{str(self.global_step)}",
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range(f"log_training_pass")
    def log_training_pass(self, metrics):
        """Log information after a single training pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        pass

    @torch.cuda.nvtx.range(f"on_training_end")
    def on_training_end(self):
        """Callback that prompts at the end of training."""
        conf = self.conf
        out_dir = self.tracking.output_dir

        # Export the mixture-of-3d-gaussians
        logger.log_rule("Exporting Models")

        if conf.export_ply.enabled:
            from threedgrut.export import PLYExporter

            ply_path = conf.export_ply.path if conf.export_ply.path else os.path.join(out_dir, "export_last.ply")
            exporter = PLYExporter()
            exporter.export(self.model, Path(ply_path), dataset=self.train_dataset, conf=conf)

        if conf.export_usd.enabled:
            from threedgrut.export import NuRecExporter, USDExporter

            # Determine format for filename suffix
            usdz_format = getattr(conf.export_usd, "format", "nurec")
            if usdz_format == "standard":
                format_suffix = "lightfield"
                exporter = USDExporter.from_config(conf)
            else:
                format_suffix = "nurec"
                exporter = NuRecExporter()

            # Handle path: if not set or relative, put in output directory
            if conf.export_usd.path:
                usdz_path = conf.export_usd.path
                if not os.path.isabs(usdz_path):
                    usdz_path = os.path.join(out_dir, usdz_path)
            else:
                # Default filename includes format suffix
                usdz_path = os.path.join(out_dir, f"export_last_{format_suffix}.usdz")

            exporter.export(
                self.model,
                Path(usdz_path),
                dataset=self.train_dataset,
                conf=conf,
                background=getattr(self, "background", None),
                post_processing=getattr(self, "post_processing", None),
            )

        # Export post-processing report (PPISP-based)
        if self.post_processing is not None and conf.post_processing.method == "ppisp":
            from ppisp.report import export_ppisp_report

            logger.info("📊 Exporting PPISP report...")

            ppisp_report_dir = Path(out_dir) / "ppisp_report"
            frames_per_camera = self.train_dataset.get_frames_per_camera()

            # Get camera names if available
            camera_names = None
            if hasattr(self.train_dataset, "get_camera_names"):
                camera_names = self.train_dataset.get_camera_names()

            export_ppisp_report(
                self.post_processing,
                frames_per_camera=frames_per_camera,
                output_dir=ppisp_report_dir,
                camera_names=camera_names,
            )
            logger.info(f"📊 PPISP report saved to: {ppisp_report_dir}")

        self.teardown_dataloaders()
        self.save_checkpoint(last_checkpoint=True)

        # Evaluate on test set
        if conf.test_last:
            logger.log_rule("Evaluation on Test Set")

            # Renderer test split
            if self.feature_decoder is not None:
                self.feature_decoder.apply_ema_shadow()
            try:
                renderer = Renderer.from_preloaded_model(
                    model=self.model,
                    out_dir=out_dir,
                    path=conf.path,
                    save_gt=False,
                    writer=self.tracking.writer,
                    global_step=self.global_step,
                    compute_extra_metrics=conf.compute_extra_metrics,
                    post_processing=self.post_processing,
                    feature_decoder=self.feature_decoder,
                )
                renderer.render_all()
            finally:
                if self.feature_decoder is not None:
                    self.feature_decoder.restore_ema()

    @torch.cuda.nvtx.range(f"save_checkpoint")
    def save_checkpoint(self, last_checkpoint: bool = False):
        """Saves checkpoint to a path under {conf.out_dir}/{conf.experiment_name}.
        Args:
            last_checkpoint: If true, will update checkpoint title to 'last'.
                             Otherwise uses global step
        """
        global_step = self.global_step
        out_dir = self.tracking.output_dir
        parameters = self.model.get_model_parameters()
        parameters |= {"global_step": self.global_step, "epoch": self.n_epochs - 1}
        if self.switch_splat_optimizer is not None:
            parameters["switch_splat_optimizer"] = self.switch_splat_optimizer.state_dict()

        strategy_parameters = self.strategy.get_strategy_parameters()
        parameters = {**parameters, **strategy_parameters}

        # Add feature decoder state to checkpoint (module + optimizer + scheduler + EMA)
        if self.feature_decoder is not None:
            dec = self.feature_decoder
            parameters["feature_decoder"] = {
                "module": dec.state_dict(),
                "optimizer": self.feature_decoder_optimizer.state_dict(),
                "scheduler": self.feature_decoder_scheduler.state_dict(),
                "arch": {
                    "ray_feature_dim": dec.ray_feature_dim,
                    "hidden_dim": dec.hidden_dim,
                    "num_layers": dec.num_layers,
                    "sh_scale": dec.sh_scale,
                    "output_activation": dec.output_activation,
                    "unpremultiply_alpha": dec.unpremultiply_alpha,
                },
            }
            ema_state = self.feature_decoder.ema_state_dict()
            if ema_state:
                parameters["feature_decoder"]["ema"] = ema_state

        # Add post-processing state to checkpoint (module + optimizers + schedulers)
        if self.post_processing is not None:
            parameters["post_processing"] = {
                "module": self.post_processing.state_dict(),
                "optimizers": [opt.state_dict() for opt in self.post_processing_optimizers],
                "schedulers": [sched.state_dict() for sched in self.post_processing_schedulers],
            }

        os.makedirs(os.path.join(out_dir, f"ours_{int(global_step)}"), exist_ok=True)
        if not last_checkpoint:
            ckpt_path = os.path.join(out_dir, f"ours_{int(global_step)}", f"ckpt_{global_step}.pt")
        else:
            ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
        torch.save(parameters, ckpt_path)
        logger.info(f'💾 Saved checkpoint to: "{os.path.abspath(ckpt_path)}"')

    def render_gui(self, scene_updated):
        """Render & refresh a single frame for the gui"""
        gui = self.gui
        if gui is not None:
            import polyscope as ps

            if gui.live_update:
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_cloud_viz()
                gui.update_render_view_viz()

            ps.frame_tick()
            while not gui.viz_do_train:
                ps.frame_tick()

            if ps.window_requests_close():
                logger.warning(
                    "Terminating training from GUI window is not supported. Please terminate it from the terminal."
                )

    def render_gui_viser(self, scene_updated):
        gui = self.gui
        if gui is not None:
            if gui.live_update:
                gui.update_switch_timetable(self.global_step)
                # update render view
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_point_cloud()
                for client in gui.server.get_clients().values():
                    gui.update_render_view(client, force=True)
                while not gui.viz_do_train:
                    time.sleep(0.0001)

    def _prepare_per_view_losses(self, gpu_batch, outputs):
        """Build per-camera losses and their detached mean for normal logging."""
        batch_size = gpu_batch.batch_size
        weight = 1.0 / batch_size
        per_view = []
        batch_losses: dict[str, torch.Tensor] = {}

        for view_index in range(batch_size):
            view_batch = slice_batch(gpu_batch, view_index)
            view_outputs = _slice_camera_outputs(outputs, view_index, batch_size)
            view_losses = self.get_losses(view_batch, view_outputs)
            per_view.append((view_batch, view_losses))
            for name, value in view_losses.items():
                weighted_value = value.detach() * weight
                batch_losses[name] = batch_losses.get(name, 0) + weighted_value

        return per_view, batch_losses

    def _backward_per_view_position_gradients(self, global_step: int, per_view_losses) -> None:
        """Backpropagate a logical batch while recording each camera's GS statistic."""
        batch_size = len(per_view_losses)
        weight = 1.0 / batch_size
        update_buffer = self.strategy.should_update_gradient_buffer(global_step)

        if update_buffer:
            for view_batch, view_losses in per_view_losses:
                (view_position_grad,) = torch.autograd.grad(
                    view_losses["total_loss"],
                    self.model.positions,
                    retain_graph=True,
                )
                self.strategy.update_gradient_buffer(
                    sensor_position=view_batch.T_to_world[0, :3, 3],
                    params_grad=view_position_grad.detach(),
                )

        # A single terminal backward traverses every per-view loss branch and
        # releases the complete shared render graph. Backwarding each branch
        # separately with retain_graph=True leaves earlier branches awaiting
        # cyclic GC, which can retain many full-resolution batches on CUDA.
        mean_total_loss = sum(view_losses["total_loss"] * weight for _, view_losses in per_view_losses)
        mean_total_loss.backward()

    @torch.cuda.nvtx.range(f"run_train_iter")
    def run_train_iter(
        self,
        global_step: int,
        batch: dict,
        profilers: dict,
        metrics: list,
        conf: DictConfig,
    ):
        self._apply_color_refine_freeze(global_step)

        # Freeze Gaussians and suspend strategy when distillation starts
        if self._distillation_start_step >= 0 and global_step >= self._distillation_start_step:
            self.model.freeze_gaussians()
            self.strategy.suspend()

        # Access the GPU-cache batch data
        with torch.cuda.nvtx.range(f"train_iter{global_step}_get_gpu_batch"):
            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)

        profilers["step_total"].start()

        # Perform validation if required
        is_time_to_validate = (global_step > 0 or conf.validate_first) and (global_step % self.val_frequency == 0)
        if is_time_to_validate:
            self.run_validation_pass(conf)

        # Compute the outputs of a single batch
        with torch.cuda.nvtx.range(f"train_{global_step}_fwd"):
            profilers["inference"].start()
            outputs = self.model(gpu_batch, train=True, frame_id=global_step)
            profilers["inference"].end()

        # Apply feature decoder to convert N-dimensional features to RGB
        if self.feature_decoder is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_feature_decoder"):
                profilers["feature_decoder"].start()
                outputs = apply_feature_decoder(
                    self.feature_decoder,
                    outputs,
                    gpu_batch,
                    training=True,
                    center_ray_encoding=bool(getattr(self.conf.model.nht_decoder, "center_ray_encoding", False)),
                )
                profilers["feature_decoder"].end()
        outputs = apply_background(self.model.background, outputs, gpu_batch, training=True)

        # Apply post-processing to rendered output
        if self.post_processing is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing"):
                outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=True)

        use_per_view_position_gradients = gpu_batch.batch_size > 1 and getattr(
            self.strategy, "requires_per_view_position_gradients", False
        )

        # Compute the losses of a single batch
        with torch.cuda.nvtx.range(f"train_{global_step}_loss"):
            if use_per_view_position_gradients:
                per_view_losses, batch_losses = self._prepare_per_view_losses(gpu_batch, outputs)
            else:
                batch_losses = self.get_losses(gpu_batch, outputs)

            # Add post-processing regularization loss
            if self.post_processing is not None:
                post_processing_reg_loss = self.post_processing.get_regularization_loss()
                batch_losses["total_loss"] = batch_losses["total_loss"] + post_processing_reg_loss
                batch_losses["post_processing_reg_loss"] = post_processing_reg_loss

        # Backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_pre_bwd"):
            self.strategy.pre_backward(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Back-propagate the gradients and update the parameters
        with torch.cuda.nvtx.range(f"train_{global_step}_bwd"):
            profilers["backward"].start()
            if use_per_view_position_gradients:
                self._backward_per_view_position_gradients(global_step, per_view_losses)
            else:
                batch_losses["total_loss"].backward()
            profilers["backward"].end()

        # Post backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_post_bwd"):
            scene_updated = self.strategy.post_backward(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Optimizer step
        with torch.cuda.nvtx.range(f"train_{global_step}_backprop"):
            self._zero_color_refine_frozen_grads()
            if isinstance(self.model.optimizer, SelectiveAdam):
                assert (
                    outputs["mog_visibility"].shape == self.model.density.shape
                ), f"Visibility shape {outputs['mog_visibility'].shape} does not match density shape {self.model.density.shape}"
                self.model.optimizer.step(outputs["mog_visibility"])
            else:
                self.model.optimizer.step()
            if self.switch_splat_optimizer is not None:
                self.switch_splat_optimizer.step()
            self.model.optimizer.zero_grad()
            if self.switch_splat_optimizer is not None:
                self.switch_splat_optimizer.zero_grad()

        # Scheduler step
        with torch.cuda.nvtx.range(f"train_{global_step}_scheduler"):
            self.model.scheduler_step(global_step)
            self._apply_color_refine_freeze(global_step)

        # Feature decoder optimizer/scheduler step
        if self.feature_decoder_optimizer is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_feature_decoder_opt"):
                self.feature_decoder_optimizer.step()
                self.feature_decoder_optimizer.zero_grad()
                self.feature_decoder_scheduler.step()
                self.feature_decoder.ema_update(global_step)

        # Post-processing optimizer/scheduler step
        if self.post_processing_optimizers is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing_opt"):
                for opt in self.post_processing_optimizers:
                    opt.step()
                    opt.zero_grad()
                for sched in self.post_processing_schedulers:
                    sched.step()

        # Post backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_post_opt_step"):
            scene_updated = self.strategy.post_optimizer_step(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )
            self._maybe_soften_switch_logits(global_step)

        # Update the SH if required
        if self.model.progressive_training and check_step_condition(
            global_step, 0, 1e6, self.model.feature_dim_increase_interval
        ):
            self.model.increase_num_active_features()

        # Update the BVH if required
        if scene_updated or (
            conf.model.bvh_update_frequency > 0 and global_step % conf.model.bvh_update_frequency == 0
        ):
            with torch.cuda.nvtx.range(f"train_{global_step}_bvh"):
                profilers["build_as"].start()
                self.model.build_acc(rebuild=True)
                profilers["build_as"].end()

        profilers["step_total"].end()

        # Increment the global step
        global_step += 1
        self.global_step = global_step

        # Compute metrics
        batch_metrics = self.get_metrics(
            gpu_batch,
            outputs,
            batch_losses,
            profilers,
            split="training",
            iteration=iter,
        )
        if "forward_render" in self.model.renderer.timings:
            batch_metrics["timings"]["forward_render_cuda"] = self.model.renderer.timings["forward_render"]
        if "backward_render" in self.model.renderer.timings:
            batch_metrics["timings"]["backward_render_cuda"] = self.model.renderer.timings["backward_render"]
        metrics.append(batch_metrics)

        # !!! Below global step has been incremented !!!
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_log_iter"):
            self.log_training_iter(gpu_batch, outputs, batch_metrics, iter)
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_save_ckpt"):
            if global_step in conf.checkpoint.iterations:
                self.save_checkpoint()

        # Updating the GUI
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_update_gui"):
            if self.conf.with_viser_gui:
                self.render_gui_viser(scene_updated)
            elif self.conf.with_gui:
                self.render_gui(scene_updated)

    @torch.cuda.nvtx.range(f"run_train_pass")
    def run_train_pass(self, conf: DictConfig):
        """Runs a single train epoch over the dataset."""
        metrics = []
        profilers = {
            "inference": CudaTimer(enabled=self.conf.enable_frame_timings),
            "backward": CudaTimer(enabled=self.conf.enable_frame_timings),
            "build_as": CudaTimer(enabled=self.conf.enable_frame_timings),
            "step_total": CudaTimer(enabled=self.conf.enable_frame_timings),
        }
        if self.feature_decoder is not None:
            profilers["feature_decoder"] = CudaTimer(enabled=self.conf.enable_frame_timings)

        for iter, batch in enumerate(self.train_dataloader):
            # Check if we have reached the maximum number of iterations
            if self.global_step >= conf.n_iterations:
                return

            # Step for training iteration
            self.run_train_iter(self.global_step, batch, profilers, metrics, conf)

        self.log_training_pass(metrics)

    @torch.cuda.nvtx.range(f"run_validation_pass")
    @torch.no_grad()
    def run_validation_pass(self, conf: DictConfig) -> dict[str, Any]:
        """Runs a single validation epoch over the dataset.
        Returns:
             dictionary of metrics computed and aggregated over validation set.
        """

        if self.feature_decoder is not None:
            self.feature_decoder.apply_ema_shadow()
        profilers = {
            "inference": CudaTimer(),
        }
        metrics = []
        logger.info(f"Step {self.global_step} -- Running validation..")
        logger.start_progress(
            task_name="Validation",
            total_steps=len(self.val_dataloader),
            color="medium_purple3",
        )

        for val_iteration, batch_idx in enumerate(self.val_dataloader):
            # Access the GPU-cache batch data
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(batch_idx)

            # Compute the outputs of a single batch
            with torch.cuda.nvtx.range(f"train.validation_step_{self.global_step}"):
                profilers["inference"].start()
                outputs = self.model(gpu_batch, train=False)
                # Apply feature decoder to convert N-dimensional features to RGB
                if self.feature_decoder is not None:
                    outputs = apply_feature_decoder(
                        self.feature_decoder,
                        outputs,
                        gpu_batch,
                        training=False,
                        center_ray_encoding=bool(getattr(self.conf.model.nht_decoder, "center_ray_encoding", False)),
                    )
                outputs = apply_background(self.model.background, outputs, gpu_batch, training=False)
                # Apply post-processing for validation (novel view mode)
                if self.post_processing is not None:
                    outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)
                profilers["inference"].end()

                batch_losses = self.get_losses(gpu_batch, outputs)
                batch_metrics = self.get_metrics(
                    gpu_batch,
                    outputs,
                    batch_losses,
                    profilers,
                    split="validation",
                    iteration=val_iteration,
                )

                self.log_validation_iter(gpu_batch, outputs, batch_metrics, iteration=val_iteration)
                metrics.append(batch_metrics)

        logger.end_progress(task_name="Validation")
        if self.feature_decoder is not None:
            self.feature_decoder.restore_ema()

        metrics = self._flatten_list_of_dicts(metrics)
        self.log_validation_pass(metrics)
        return metrics

    @staticmethod
    def _flatten_list_of_dicts(list_of_dicts):
        """
        Converts list of dicts -> dict of lists.
        Supports flattening of up to 2 levels of dict hierarchies
        """
        flat_dict = defaultdict(list)
        for d in list_of_dicts:
            for k, v in d.items():
                if isinstance(v, dict):
                    flat_dict[k] = defaultdict(list) if k not in flat_dict else flat_dict[k]
                    for inner_k, inner_v in v.items():
                        flat_dict[k][inner_k].append(inner_v)
                else:
                    flat_dict[k].append(v)
        return flat_dict

    def run_training(self):
        """Initiate training logic for n_epochs.
        Training and validation are controlled by the config.
        """
        assert self.model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
        conf = self.conf

        logger.log_rule(f"Training {conf.render.method.upper()}")

        # Training loop
        logger.start_progress(task_name="Training", total_steps=conf.n_iterations, color="spring_green1")

        for epoch_idx in range(self.n_epochs):
            self.run_train_pass(conf)

        logger.end_progress(task_name="Training")

        # Report training statistics
        stats = logger.finished_tasks["Training"]
        table = dict(
            n_steps=f"{self.global_step}",
            n_epochs=f"{self.n_epochs}",
            training_time=f"{stats['elapsed']:.2f} s",
            iteration_speed=f"{self.global_step / stats['elapsed']:.2f} it/s",
        )
        logger.log_table(f"🎊 Training Statistics", record=table)

        # Perform testing
        self.on_training_end()
        logger.info(f"🥳 Training Complete.")

        # Updating the GUI
        if self.gui is not None:
            self.gui.training_done = True
            logger.info(f"🎨 GUI Blocking... Terminate GUI to Stop.")
            self.gui.block_in_rendering_loop(fps=60)
