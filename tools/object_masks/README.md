<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Object masks for the slot Gaussian fine-tuning

```bash
python tools/object_masks/make_masks.py --checkpoint runs/<run>/ckpt_last.pt
```

**Input:** a trained SwitchSplat model and its scene folder.

**Output:** `runs/<run>/object_masks/settled/slotNN/mask/masks.npz`, the input of the fine-tuning: for each object, a 512 × 512 mask of the region that
changes with it (the object in both states and what it uncovers), in every front-lens training
frame where the object is switched.

**Pipeline:**

1. The interaction windows the model learned choose, for each object, the frames inside its window
   and 40 control frames outside all windows.
2. The model renders each of these views with only this object switched back to its base state.
3. Where photo and render differ (1 − SSIM above the 99.5% quantile of the control frames) is
   labelled object, and where they agree (below the median) background.
4. A small MLP on DINOv3 features and pixel positions learns these labels and predicts a mask for
   every frame; SAM 2.1 then refines each mask by propagating masks from neighbouring frames.
5. Frames within 10 frames of a window's start or end, where an object may still be moving, are
   dropped.
