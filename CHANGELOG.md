# Changelog

## SwitchSplat (2026)

SwitchSplat is built on 3DGRUT v2.0.0 (commit a37ef72) and FullCircle. Relative to 3DGRUT it adds:

- SwitchSplat scene model: per-Gaussian state logits, learnable interaction windows ("slots") with
  soft temporal gates, logit softening, the sparsity and compactness losses, and state-combination
  rendering (`threedgrut/model/switch_splat_controller.py`, trainer and loss changes).
- FullCircle data handling: fisheye and pinhole COLMAP scenes with capturer masks, test-frame lists
  and the base-state / novel-view / novel-combination evaluation splits.
- gsplat 3DGUT and 3DGS renderers with camera batching (`gsplat_tracer/`).
- App configs `configs/apps/colmap_switchsplat*.yaml` (pinhole, native fisheye, slot Gaussian
  fine-tuning with object masks) and the paper's FullCircle baselines and ablations in
  `configs/paper/switchsplat/`; `threedgrut/utils/slot_mask_supervision.py` for the mask loss.
- One installer for the whole pipeline (`install.sh`).
- One-step preprocessing of a capture (`scripts/preprocess.sh`): FullCircle's masking of the camera
  holder (`masking/`, `thirdparty/sam-ui`), COLMAP, half-size frames and a test split.
- Tools: VLM interaction windows
  (`tools/vlm_windows`), object masks (`tools/object_masks`), cross-camera evaluation
  (`tools/eval_on_dataset.py`).

Removed from 3DGRUT because SwitchSplat does not use them: the playground (`playground.py`,
`threedgrut_playground/`), the benchmark scripts (`scripts/benchmark/`), the 3DGRUT paper configs
(`configs/paper/`), the app configs of other datasets, the CI workflow, the Dockerfile, the Windows
installation scripts, `validate.py`, `tools/ppisp_export/`, `CONTRIBUTING.md`, the 3DGRUT media in
`assets/`, and the formatting, rebase and Kaolin helper scripts. The installer no longer installs
Kaolin, which only the playground used.

Files of 3DGRUT a37ef72 changed by the SwitchSplat and FullCircle authors (59):

- `.gitignore`
- `ATTRIBUTIONS.md`
- `CHANGELOG.md`
- `README.md`
- `configs/base_gs.yaml`
- `configs/dataset/colmap.yaml`
- `configs/initialization/colmap.yaml`
- `configs/render/3dgut.yaml`
- `install_env_uv.sh`
- `pyproject.toml`
- `requirements_extra.txt`
- `scripts/create_conda.sh`
- `threedgrt_tracer/__init__.py`
- `threedgrt_tracer/include/3dgrt/optixTracer.h`
- `threedgrt_tracer/src/kernels/cuda/referenceOptix.cu`
- `threedgrt_tracer/src/optixTracer.cpp`
- `threedgrt_tracer/src/particlePrimitives.cu`
- `threedgrt_tracer/tracer.py`
- `threedgrut/datasets/__init__.py`
- `threedgrut/datasets/datasetNcore.py`
- `threedgrut/datasets/dataset_colmap.py`
- `threedgrut/datasets/dataset_nerf.py`
- `threedgrut/datasets/protocols.py`
- `threedgrut/datasets/tests/test_dataset_colmap.py`
- `threedgrut/datasets/tests/test_dataset_colmap_lifecycle.py`
- `threedgrut/datasets/utils.py`
- `threedgrut/export/README.md`
- `threedgrut/export/base.py`
- `threedgrut/export/scripts/ply_to_usd.py`
- `threedgrut/export/usd/post_processing/README.md`
- `threedgrut/gui/bindings.cpp`
- `threedgrut/gui/include/gl_interop.h`
- `threedgrut/gui/ps_extension.py`
- `threedgrut/gui/setup_gui.py`
- `threedgrut/model/losses.py`
- `threedgrut/model/model.py`
- `threedgrut/optimizers/__init__.py`
- `threedgrut/optimizers/optimizers.cpp`
- `threedgrut/optimizers/optimizers.cu`
- `threedgrut/optimizers/setup_optimizers.py`
- `threedgrut/render.py`
- `threedgrut/strategy/base.py`
- `threedgrut/strategy/gs.py`
- `threedgrut/strategy/mcmc.py`
- `threedgrut/strategy/src/__init__.py`
- `threedgrut/strategy/src/bindings.cpp`
- `threedgrut/strategy/src/gaussian_mcmc.cu`
- `threedgrut/strategy/src/gaussian_mcmc.h`
- `threedgrut/strategy/src/setup_mcmc.py`
- `threedgrut/trainer.py`
- `threedgrut/utils/gui.py`
- `threedgrut/utils/misc.py`
- `threedgrut/utils/viser_gui_util.py`
- `threedgut_tracer/include/3dgut/kernels/cuda/models/shGaussianModel.cuh`
- `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh`
- `threedgut_tracer/include/3dgut/kernels/cuda/renderers/gutProjector.cuh`
- `threedgut_tracer/setup_3dgut.py`
- `threedgut_tracer/tracer.py`
- `train.py`

## v2.0.0 - 2026-06-10

- Added Neural Harmonic Textures support for 3DGRT and 3DGUT.
- Added NHT training examples, citation, and Mip-NeRF 360 validation results.
- Updated installation dependencies for the NHT pipeline, including tiny-cuda-nn bindings.

## v1.1.0 - 2026-06-10

- Set the Python package metadata version to `1.1.0`.
- Added runtime version access through `threedgrut.__version__`.
- Moved package metadata into `pyproject.toml` and reduced `setup.py` to a setuptools compatibility shim.
- Updated PPISP to v1.2.1 for CUDA 13 compatibility.

## v1.0.0 - 2025-04

- Public stable release tag for 3DGRUT. Python package metadata was still pre-1.0 at that time and is aligned in v1.1.0.
