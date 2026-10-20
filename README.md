# SwitchSplat: Learning Discrete Object States from Casually Captured Scenes
[![Webpage](https://img.shields.io/badge/Webpage-1F6FEB?style=for-the-badge&logo=googlechrome&logoColor=white)](https://switchsplat.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.24812)
[![Viewer](https://img.shields.io/badge/Interactive%20Viewer-6D28D9?style=for-the-badge&logo=webgpu&logoColor=white)](https://switchsplat.github.io/viewer/)

[Yizheng Xie](https://xieyizheng.com)\*,
[Yalda Foroutan](https://youlenda.github.io)\*,
George Shramko,
[Daniel Rebain](http://drebain.com),
[Shrisudhan Govindarajan](https://shrisudhan.github.io)†,
[Andrea Tagliasacchi](https://taiya.github.io)†

\* Equal contribution &nbsp; † Equal advising

Official implementation of **SwitchSplat** (SIGGRAPH Asia 2026), a method that learns the discrete
states of objects (a door open or closed, a drawer in or out) from one casual video, so each object
can be switched on its own at render time.

![Top: frames of a casual video of a car whose door, hood and trunk are opened one at a time. Bottom: SwitchSplat renders of the car from a new viewpoint, with the trunk, door and hood switched open or closed independently.](assets/teaser.jpg)

**Code and dataset coming soon.**

## BibTeX
```bibtex
@inproceedings{xie2026switchsplat,
  title     = {SwitchSplat: Learning Discrete Object States from Casually Captured Scenes},
  author    = {Xie, Yizheng and Foroutan, Yalda and Shramko, George and Rebain, Daniel
               and Govindarajan, Shrisudhan and Tagliasacchi, Andrea},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  doi       = {10.1145/3829340.3842325}
}
```
