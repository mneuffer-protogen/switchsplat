# SwitchSplat: Learning Discrete Object States from Casually Captured Scenes
[![Webpage](https://img.shields.io/badge/Webpage-1F6FEB?style=for-the-badge&logo=googlechrome&logoColor=white)](https://switchsplat.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.24812)
[![Viewer](https://img.shields.io/badge/Interactive%20Viewer-1F6F4A?style=for-the-badge&logo=webgpu&logoColor=white)](https://switchsplat.github.io/viewer/)
[![Data](https://img.shields.io/badge/Data-0F766E?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/xieyizheng/switchsplat)

[Yizheng Xie](https://xieyizheng.com)\*,
[Yalda Foroutan](https://youlenda.github.io)\*,
George Shramko,
[Daniel Rebain](http://drebain.com),
[Shrisudhan Govindarajan](https://shrisudhan.github.io)†,
[Andrea Tagliasacchi](https://taiya.github.io)†

Official implementation of **SwitchSplat**, a method for learning discrete object states from casually captured scenes.

![Teaser](assets/teaser.jpg)

## Installation

```bash
git clone --recursive https://github.com/theialab/switchsplat.git
cd switchsplat
bash install.sh
conda activate switchsplat
```

## Data

Download the [scenes](https://huggingface.co/datasets/xieyizheng/switchsplat) and unzip them under `data/`. Each scene is organized as follows:
```
data/switchsplat_fisheye/<scene_name>/
├── images_2/     # frames (camera1/, camera2/)
├── masks_2/      # masks of the person holding the camera
└── sparse/0/     # COLMAP reconstruction
```

## Usage

### 1. Preprocessing

The released scenes are already preprocessed. For a new capture, place its fisheye frames in `data/<scene_name>/images/` and its 360° panoramas in `data/<scene_name>/omni/images/`, as in [FullCircle](https://github.com/theialab/fullcircle), then run:

```bash
bash scripts/preprocess.sh data/<scene_name>
```

### 2. Slot window initialization

Each slot starts from a time window, one `start end` line (in frames) per object in `vlm_prediction.txt`; the released scenes include them. The windows can be written by hand, or predicted with a VLM:

```bash
python tools/vlm_windows/predict_windows.py data/<scene_name>
```

### 3. Training

```bash
python train.py --config-name apps/colmap_switchsplat_fisheye.yaml path=data/switchsplat_fisheye/<scene_name>
```

For the pinhole version of a scene, use `apps/colmap_switchsplat.yaml`; other renderers can be selected with `render=3dgrt` or `render=3dgut`.

### 4. Slot Gaussian fine-tuning

Optionally, compute the object masks of a trained model:

```bash
python tools/object_masks/make_masks.py --checkpoint runs/<run>/ckpt_last.pt
```

Then fine-tune its slot Gaussians with them:

```bash
python train.py --config-name apps/colmap_switchsplat_fisheye_finetune.yaml path=data/switchsplat_fisheye/<scene_name> \
    resume=runs/<run>/ckpt_last.pt loss.slot_mask.mask_run=runs/<run>/object_masks/settled
```

## Acknowledgements

This code is built on [3DGRUT](https://github.com/nv-tlabs/3dgrut), [FullCircle](https://github.com/theialab/fullcircle) and [gsplat](https://github.com/nerfstudio-project/gsplat).

## BibTeX
```bibtex
@inproceedings{xie2026switchsplat,
  title     = {SwitchSplat: Learning Discrete Object States from Casually Captured Scenes},
  author    = {Xie, Yizheng and Foroutan, Yalda and Shramko, George and Rebain, Daniel and Govindarajan, Shrisudhan and Tagliasacchi, Andrea},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  doi       = {10.1145/3829340.3842325}
}
```
