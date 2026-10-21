# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden tests: the released code reproduces the paper's VLM windows and numbers, offline.

Run from the repository root:  python -m pytest tools/vlm_windows/tests
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL_DIR / "figures"))
sys.path.insert(0, str(TOOL_DIR))

import detect_windows  # noqa: E402
import evaluate  # noqa: E402
import plot_effort_scaling  # noqa: E402
import plot_failure_modes  # noqa: E402
import plot_interaction_patterns  # noqa: E402
import postprocess  # noqa: E402

RESULTS = TOOL_DIR / "results"
ROLLOUTS = RESULTS / "gpt-5.6-sol"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
TRIALS = tuple(range(1, 11))
SCENES = (
    "balcony",
    "car",
    "kitchen1",
    "flat1",
    "flat2",
    "flat3",
    "chair",
    "m_kitchen1",
    "m_kitchen2",
    "m_kitchen3",
    "kitchen2",
    "oslo",
)
# Last line of the prompt's rule list, as sent for each scene.
VALID_RANGE_N = {
    "balcony": 929,
    "chair": 756,
    "flat2": 2307,
    "kitchen1": 2126,
    "m_kitchen1": 533,
    "m_kitchen2": 686,
    "m_kitchen3": 824,
    "car": 1155,
    "flat3": 2115,
    "kitchen2": 867,
    "oslo": 683,
    "flat1": 1331,
}
OUTPUT_SCHEMA_SHA256 = "6e4b6eeae19a7c96e54031b1cb064ec2802489e1d99468ec0774c6a6d2bb95bb"
# Per-effort means over 10 rollouts (low, medium, high, xhigh, max), as in the paper figure.
MEAN_EVENT_JACCARD = (97.12, 96.61, 97.26, 98.13, 99.49)
MEAN_ENDPOINT_MAE = (4.255, 3.855, 3.882, 3.321, 2.780)
# sha256 of vlm_failure_mode_frequency_gpt56sol_n10_data.json written for the paper figure.
FAILURE_DATA_SHA256 = "b0588649d257a915c3cdb99ae880b4f2aa2747c9729873fe7ead28273d5b0f7b"
ROLLOUT_IDS = [(effort, trial) for effort in EFFORTS for trial in TRIALS]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scene_rows() -> dict[str, dict]:
    return {row["scene"]: row for row in load(RESULTS / "scenes.json")["scenes"]}


def test_scene_table() -> None:
    assert tuple(evaluate.paper_scenes()) == SCENES
    for effort, trial in ROLLOUT_IDS:
        rollout = ROLLOUTS / effort / f"trial{trial}"
        assert set(load(rollout / "index.json")) == set(SCENES)
        assert all((rollout / scene / "raw_observation.json").is_file() for scene in SCENES)


def test_prompt_template_reproduces_every_sent_prompt() -> None:
    rows = scene_rows()
    for scene in SCENES:
        row = rows[scene]
        prompt = detect_windows.render_prompt(row["frame_min"], row["frame_max"], 8)
        assert f"- Valid frame range is 1 to {VALID_RANGE_N[scene]}.\n" in prompt
        assert detect_windows.text_sha256(prompt) == row["prompt_sha256"]
        for effort, trial in ROLLOUT_IDS:
            manifest = load(ROLLOUTS / effort / f"trial{trial}" / scene / "manifest.json")
            assert manifest["prompt_sha256"] == row["prompt_sha256"], (scene, effort, trial)


def test_output_schema_is_the_one_sent() -> None:
    assert sha256(TOOL_DIR / "output_schema.json") == OUTPUT_SCHEMA_SHA256
    for effort, trial in ROLLOUT_IDS:
        for scene in SCENES:
            manifest = load(ROLLOUTS / effort / f"trial{trial}" / scene / "manifest.json")
            assert manifest["output_schema_sha256"] == OUTPUT_SCHEMA_SHA256


def test_codex_command_and_inputs_match_every_paper_run() -> None:
    rows = scene_rows()
    for scene in SCENES:
        inputs = load(RESULTS / "inputs" / f"{scene}.json")
        assert detect_windows.input_digest(inputs["input_sha256"]) == inputs["input_sha256_digest"]
        assert inputs["input_sha256_digest"] == rows[scene]["input_sha256_digest"]
        frames = [item["frame"] for item in inputs["items"]]
        assert frames[0] == rows[scene]["frame_min"] and frames[-1] == rows[scene]["frame_max"]
        assert all(frame % 8 == 0 for frame in frames[1:-1]) and frames == sorted(set(frames))
        assert len(frames) == rows[scene]["input_images"] == len(inputs["input_sha256"])
        for effort, trial in ROLLOUT_IDS:
            manifest = load(ROLLOUTS / effort / f"trial{trial}" / scene / "manifest.json")
            assert manifest["input_sha256_digest"] == inputs["input_sha256_digest"]
            count = manifest["sampling"]["input_image_count"]
            assert count == len(frames)
            expected = detect_windows.codex_command(
                executable="codex",
                model="gpt-5.6-sol",
                reasoning_effort=effort,
                reasoning_summary="detailed",
                image_count=count,
                work_dir=detect_windows.WORKDIR_PLACEHOLDER,
            )
            call = manifest["model_call"]
            assert call["command"] == expected[: expected.index("--image")]
            assert call["command_images"].endswith(f"0001..{count:04d}")
            assert manifest["codex"]["version"] == detect_windows.PAPER_CODEX_VERSION
            assert manifest["reasoning_effort"] == effort and manifest["model"] == "gpt-5.6-sol"


def test_raw_observations_are_the_recorded_ones() -> None:
    for effort, trial in ROLLOUT_IDS:
        for scene in SCENES:
            folder = ROLLOUTS / effort / f"trial{trial}" / scene
            manifest = load(folder / "manifest.json")
            assert sha256(folder / "raw_observation.json") == manifest["raw_observation_sha256"]
            assert len(load(folder / "raw_observation.json")["events"]) == manifest["event_count"]


def test_postprocess_reproduces_paper_vlm_predictions() -> None:
    manifest = (RESULTS / "paper_trial1" / "MANIFEST.txt").read_text(encoding="utf-8")
    for scene in SCENES:
        events = load(ROLLOUTS / "max" / "trial1" / scene / "raw_observation.json")["events"]
        kept, _ = postprocess.process_events(events)
        released = RESULTS / "paper_trial1" / scene / "vlm_prediction.txt"
        assert postprocess.format_windows(kept).encode("utf-8") == released.read_bytes(), scene
        assert f"{sha256(released)}  {scene}/vlm_prediction.txt" in manifest


def test_postprocess_defaults_are_the_paper_settings() -> None:
    assert postprocess.DEFAULT_MINIMUM_DURATION == 20
    assert postprocess.DEFAULT_NMS_IOU == 0.8


@pytest.mark.parametrize("effort,trial", ROLLOUT_IDS)
def test_evaluations_reproduce(effort: str, trial: int) -> None:
    rollout = ROLLOUTS / effort / f"trial{trial}"
    raw_evaluation = load(rollout / "evaluation_raw.json")
    for scene, recorded in raw_evaluation["scenes"].items():
        ground_truth = evaluate.read_intervals(RESULTS / "ground_truth" / f"{scene}.txt")
        assert recorded["gt"] == ground_truth
        observation = load(RESULTS / recorded["raw_observation"])
        scored, _ = evaluate.score_scene(evaluate.raw_windows(observation), ground_truth)
        assert scored == {key: recorded[key] for key in scored}, scene
    reproduced = postprocess.postprocess_evaluation(raw_evaluation, 20, 0.8, RESULTS)
    assert reproduced == load(rollout / "evaluation_min20_nms_iou80.json")


def test_paper_trial1_evaluation() -> None:
    for root, postprocessed in ((RESULTS / "paper_trial1", False), (ROLLOUTS / "max" / "trial1", True)):
        total = evaluate.evaluate_directory(root, scenes=list(SCENES), postprocess=postprocessed)["aggregate"]
        assert (total["matched_events"], total["ground_truth_events"]) == (59, 59)
        assert (total["extras"], total["misses"]) == (0, 0)
        assert total["endpoint_mae_frames"] == 2.559
        assert total["mean_event_jaccard"] == 1.0


def test_paper_numbers() -> None:
    points = plot_effort_scaling.load_points(list(EFFORTS), list(TRIALS))
    for effort, jaccard, mae in zip(EFFORTS, MEAN_EVENT_JACCARD, MEAN_ENDPOINT_MAE):
        group = [point for point in points if point["effort"] == effort]
        assert round(mean(point["macro_event_jaccard"] for point in group), 2) == jaccard
        assert round(mean(point["endpoint_mae_frames"] for point in group), 3) == mae
    max_points = [point for point in points if point["effort"] == "max"]
    perfect = [point["trial"] for point in max_points if point["extras"] == 0 and point["misses"] == 0]
    assert perfect == [1, 2, 4, 5, 7, 9]  # "6 of 10" at max effort
    best = max(max_points, key=lambda point: (point["macro_event_jaccard"], -point["endpoint_mae_frames"]))
    assert best["trial"] == 1 and best["endpoint_mae_frames"] == 2.559
    windows = sum(
        load(ROLLOUTS / effort / f"trial{trial}" / "evaluation_min20_nms_iou80.json")["aggregate"]["predicted_events"]
        for effort, trial in ROLLOUT_IDS
    )
    assert windows == 2934  # post-processed windows over 600 scene-runs


def test_failure_mode_figure_data() -> None:
    data = plot_failure_modes.figure_data()
    assert data["failure_modes"] == ("Missing slot", "Over-segmentation", "Incorrect merging", "False positive")
    assert [sum(row) for row in data["error_matrix"]] == [34, 27, 12, 3]
    assert (data["count_perfect_scene_runs"], data["scene_runs"]) == (533, 600)
    text = json.dumps(data, indent=2) + "\n"
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == FAILURE_DATA_SHA256


def test_interaction_pattern_figure_inputs() -> None:
    _, compiled, provenance = plot_interaction_patterns.load_frozen_inputs(
        plot_interaction_patterns.DEFAULT_INPUTS, plot_interaction_patterns.DEFAULT_DATASET_ROOT
    )
    assert [scene["scene_id"] for scene in plot_interaction_patterns.canonical_scenes(compiled)] == list(SCENES)
    assert sum(len(scene["intervals"]) for scene in compiled) == 59
    assert sum(record["verified_mapping_count"] for record in provenance.values()) == 59


def _require_label_font() -> None:
    from PIL import ImageFont

    try:
        ImageFont.truetype(detect_windows.DEFAULT_FONT, 40)
    except OSError:
        pytest.skip("DejaVu Sans Bold is not installed")


def _make_scene(root: Path, frame_count: int) -> Path:
    from PIL import Image

    camera = root / "images_2" / "camera1"
    camera.mkdir(parents=True)
    for frame in range(1, frame_count + 1):
        Image.new("RGB", (8, 6), (frame % 256, 40, 90)).save(camera / f"frame_{frame:04d}.png")
    Image.new("RGB", (8, 6)).save(camera / "frame_test_novel_view_01_0001.png")  # ignored
    (root / "images_2" / "camera2").mkdir()
    return root


def test_dry_run_builds_prompt_images_and_command(tmp_path: Path) -> None:
    _require_label_font()
    scene = _make_scene(tmp_path / "toy", 20)
    out_dir = tmp_path / "out"
    detect_windows.run(detect_windows.build_parser().parse_args([str(scene), "--out-dir", str(out_dir)]))
    assert (out_dir / "prompt.txt").read_text(encoding="utf-8") == detect_windows.render_prompt(1, 20, 8)
    assert sha256(out_dir / "output_schema.json") == OUTPUT_SCHEMA_SHA256
    inputs = load(out_dir / "input_manifest.json")
    assert [item["frame"] for item in inputs["items"]] == [1, 8, 16, 20]
    assert [(item["width"], item["height"]) for item in inputs["items"]] == [(480, 360)] * 4
    command = load(out_dir / "codex_command.json")["command"]
    assert command == detect_windows.codex_command(
        executable="codex",
        model="gpt-5.6-sol",
        reasoning_effort="max",
        reasoning_summary="detailed",
        image_count=4,
        work_dir=detect_windows.WORKDIR_PLACEHOLDER,
    )
    manifest = load(out_dir / "manifest.json")
    assert manifest["status"] == "dry_run" and manifest["label_font"]["size_px"] == 40
    assert not (out_dir / "raw_observation.json").exists()
    for name in ("manifest.json", "input_manifest.json", "codex_command.json"):
        assert str(tmp_path) not in (out_dir / name).read_text(encoding="utf-8")


def test_replay_reuses_a_recorded_answer(tmp_path: Path) -> None:
    _require_label_font()
    scene = _make_scene(tmp_path / "balcony", VALID_RANGE_N["balcony"])
    record = ROLLOUTS / "max" / "trial1" / "balcony"
    out_dir = tmp_path / "out"
    args = [str(scene), "--out-dir", str(out_dir), "--replay", str(record)]
    detect_windows.run(detect_windows.build_parser().parse_args(args))
    assert (out_dir / "raw_observation.json").read_bytes() == (record / "raw_observation.json").read_bytes()
    replay = load(out_dir / "manifest.json")["replay"]
    assert replay["prompt_identical"] is True and replay["input_images_identical"] is False
    windows = postprocess.format_windows(
        postprocess.process_events(load(out_dir / "raw_observation.json")["events"])[0]
    )
    assert windows == (RESULTS / "paper_trial1" / "balcony" / "vlm_prediction.txt").read_text(encoding="utf-8")


def test_no_private_paths_or_hosts() -> None:
    forbidden = [
        "/" + "home/",
        "/" + "data/",
        "/" + "localscratch",
        "mini" + "conda",
        "conda/" + "envs",
        "theia" + "00s",
        "cs-" + "theia",
        "xieyi" + "zheng",
        "sfu" + ".ca",
        "/" + "tmp/",
    ]
    local_only = {"__pycache__", ".pytest_cache", "out", "fonts"}  # caches, rendered figures, local fonts
    for path in sorted(TOOL_DIR.rglob("*")):
        if not path.is_file() or local_only.intersection(path.relative_to(TOOL_DIR).parts):
            continue
        if path.suffix not in {".py", ".json", ".txt", ".md", ""}:
            continue
        text = path.read_text(encoding="utf-8")
        for word in forbidden:
            assert word not in text, f"{path.relative_to(TOOL_DIR)} contains {word!r}"
