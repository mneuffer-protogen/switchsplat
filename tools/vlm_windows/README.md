# Slot window initialization with a VLM

```bash
python tools/vlm_windows/predict_windows.py data/<scene>
```

**Input:** a scene folder with the front-lens frames `images_2/camera1/frame_NNNN.png`.

**Output:** `vlm_prediction.txt`, one `start end` line per object interaction, in frame numbers. It
is also copied into the scene folder if the scene has none; training starts one object slot from
each line.

**Pipeline:**

1. Every 8th frame, plus the first and the last, is shrunk to 480 px wide and stamped with its frame
   number.
2. One call of GPT-5.6 Sol through the Codex CLI, at maximum reasoning effort, with
   `prompt_template.txt` and the answer format `output_schema.json`: for every interaction, the
   object and the frames where it is switched on and off.
3. Windows shorter than 20 frames are dropped, and near-duplicates (temporal overlap of 0.8 or more)
   are merged.

`--replay tools/vlm_windows/results/gpt-5.6-sol/max/trial1/<scene>` reuses the paper's recorded
answer instead of calling the model, which gives the released windows. `results/` holds the answers of the paper's
VLM study (5 reasoning efforts × 10 runs), `evaluate.py` scores windows against annotated times,
and `figures/` draws the supplement's VLM figures.
