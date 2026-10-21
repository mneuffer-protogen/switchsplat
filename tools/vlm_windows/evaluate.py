# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score interaction windows against annotated interaction times (the paper's metric).

Matching is one-to-one (Hungarian assignment). A predicted and an annotated
window may be paired only if their temporal IoU is at least 0.3; each admissible
pair scores 1000 + IoU, so the assignment maximises the number of matches first
and their overlap second. Per scene:

  event Jaccard = matched / (matched + extras + misses)

and the paper reports its mean over scenes ("event-set Jaccard"). The endpoint
MAE is the mean absolute frame error over both endpoints of every matched
window, pooled over all scenes.

Usage:
  python evaluate.py results/paper_trial1                 # vlm_prediction.txt per scene
  python evaluate.py results/gpt-5.6-sol/max/trial1 --postprocess   # raw_observation.json
  python evaluate.py PRED_ROOT --gt-root <dataset>/pinhole        # <scene>/interaction_times.txt

PRED_ROOT/<scene>/ must hold vlm_prediction.txt or raw_observation.json. The
default ground truth, results/ground_truth/<scene>.txt, holds the annotations
the paper evaluated against (see README.md for Balcony).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
RESULTS_DIR = TOOL_DIR / "results"
DEFAULT_GT_ROOT = RESULTS_DIR / "ground_truth"
SCENES_FILE = RESULTS_DIR / "scenes.json"
IOU_THRESHOLD = 0.3


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def paper_scenes() -> list[str]:
    """The 12 scene ids, in the paper's order."""
    return [row["scene"] for row in load_json(SCENES_FILE)["scenes"]]


def read_intervals(path: Path) -> list[list[int]]:
    """One "on off" frame pair per line, as in interaction_times.txt and vlm_prediction.txt."""
    intervals: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        start, end = line.split()
        intervals.append([int(start), int(end)])
    return intervals


def interval_iou(first: list[int], second: list[int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else 0.0


def match(
    predictions: list[list[int]],
    ground_truth: list[list[int]],
) -> list[tuple[int, int]]:
    """(prediction index, ground-truth index) pairs of the one-to-one assignment."""
    size = max(len(predictions), len(ground_truth))
    scores = [[0.0] * size for _ in range(size)]
    for prediction_index, prediction in enumerate(predictions):
        for ground_truth_index, truth in enumerate(ground_truth):
            overlap = interval_iou(prediction, truth)
            if overlap >= IOU_THRESHOLD:
                scores[prediction_index][ground_truth_index] = 1000.0 + overlap
    rows, columns = linear_sum_assignment([[-value for value in row] for row in scores])
    return sorted(
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if row < len(predictions) and column < len(ground_truth) and scores[row][column] > 0
    )


def score_scene(
    predictions: list[list[int]],
    ground_truth: list[list[int]],
) -> tuple[dict[str, Any], list[int]]:
    """Per-scene result (same fields as the paper's evaluation files) and endpoint errors."""
    pairs = match(predictions, ground_truth)
    matched_predictions = {prediction_index for prediction_index, _ in pairs}
    matched_ground_truth = {ground_truth_index for _, ground_truth_index in pairs}
    endpoint_errors: list[int] = []
    matches: list[dict[str, Any]] = []
    for prediction_index, ground_truth_index in pairs:
        prediction = predictions[prediction_index]
        truth = ground_truth[ground_truth_index]
        endpoint_errors.extend([abs(prediction[0] - truth[0]), abs(prediction[1] - truth[1])])
        matches.append(
            {
                "prediction_index": prediction_index,
                "ground_truth_index": ground_truth_index,
                "iou": round(interval_iou(prediction, truth), 6),
                "prediction": prediction,
                "ground_truth": truth,
            }
        )
    result = {
        "gt": ground_truth,
        "prediction": predictions,
        "predicted_count": len(predictions),
        "ground_truth_count": len(ground_truth),
        "matched_count": len(matches),
        "extra_count": len(predictions) - len(matches),
        "miss_count": len(ground_truth) - len(matches),
        "matches": matches,
        "extra_prediction_indices": [index for index in range(len(predictions)) if index not in matched_predictions],
        "missed_ground_truth_indices": [
            index for index in range(len(ground_truth)) if index not in matched_ground_truth
        ],
        "endpoint_mae_frames": (round(sum(endpoint_errors) / len(endpoint_errors), 3) if endpoint_errors else None),
    }
    return result, endpoint_errors


def event_jaccard(scene: dict[str, Any]) -> float:
    return scene["matched_count"] / (scene["matched_count"] + scene["extra_count"] + scene["miss_count"])


def aggregate(scenes: dict[str, dict[str, Any]], endpoint_errors: list[int]) -> dict[str, Any]:
    predicted_events = sum(scene["predicted_count"] for scene in scenes.values())
    ground_truth_events = sum(scene["ground_truth_count"] for scene in scenes.values())
    matched_events = sum(scene["matched_count"] for scene in scenes.values())
    return {
        "scene_count": len(scenes),
        "predicted_events": predicted_events,
        "ground_truth_events": ground_truth_events,
        "matched_events": matched_events,
        "extras": predicted_events - matched_events,
        "misses": ground_truth_events - matched_events,
        "precision": round(matched_events / predicted_events, 6) if predicted_events else None,
        "recall": round(matched_events / ground_truth_events, 6) if ground_truth_events else None,
        "endpoint_mae_frames": (round(sum(endpoint_errors) / len(endpoint_errors), 3) if endpoint_errors else None),
        "iou_match_threshold": IOU_THRESHOLD,
        "mean_event_jaccard": sum(event_jaccard(scene) for scene in scenes.values()) / len(scenes),
    }


def raw_windows(observation: dict[str, Any]) -> list[list[int]]:
    return [[int(event["on_frame"]), int(event["off_frame"])] for event in observation["events"]]


def load_prediction(
    scene_dir: Path,
    *,
    postprocess: bool,
    minimum_duration: int,
    nms_iou: float,
) -> list[list[int]]:
    text_path = scene_dir / "vlm_prediction.txt"
    raw_path = scene_dir / "raw_observation.json"
    if text_path.is_file():
        if postprocess:
            raise SystemExit(f"{text_path} is already post-processed; drop --postprocess")
        return read_intervals(text_path)
    if raw_path.is_file():
        observation = load_json(raw_path)
        if not postprocess:
            return raw_windows(observation)
        from postprocess import process_events

        kept, _ = process_events(observation["events"], minimum_duration, nms_iou)
        return raw_windows({"events": kept})
    raise FileNotFoundError(f"No vlm_prediction.txt or raw_observation.json in {scene_dir}")


def ground_truth_path(gt_root: Path, scene: str) -> Path:
    for candidate in (gt_root / f"{scene}.txt", gt_root / scene / "interaction_times.txt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No ground truth for {scene} under {gt_root}")


def evaluate_directory(
    prediction_root: Path,
    *,
    scenes: list[str],
    gt_root: Path = DEFAULT_GT_ROOT,
    postprocess: bool = False,
    minimum_duration: int = 20,
    nms_iou: float = 0.8,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    endpoint_errors: list[int] = []
    for scene in scenes:
        predictions = load_prediction(
            prediction_root / scene,
            postprocess=postprocess,
            minimum_duration=minimum_duration,
            nms_iou=nms_iou,
        )
        ground_truth = read_intervals(ground_truth_path(gt_root, scene))
        results[scene], errors = score_scene(predictions, ground_truth)
        endpoint_errors.extend(errors)
    return {
        "configuration": {
            "postprocessing": postprocess,
            "postprocessing_parameters": (
                {"minimum_duration_frames": minimum_duration, "temporal_nms_iou_threshold": nms_iou}
                if postprocess
                else None
            ),
        },
        "aggregate": aggregate(results, endpoint_errors),
        "scenes": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--scenes", nargs="+", default=None, help="Default: the 12 paper scenes")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply postprocess.py to raw_observation.json before scoring",
    )
    parser.add_argument("--minimum-duration", type=int, default=20)
    parser.add_argument("--nms-iou", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=None, help="Write the full result as JSON")
    args = parser.parse_args()
    result = evaluate_directory(
        args.prediction_root,
        scenes=args.scenes or paper_scenes(),
        gt_root=args.gt_root,
        postprocess=args.postprocess,
        minimum_duration=args.minimum_duration,
        nms_iou=args.nms_iou,
    )
    for scene, row in result["scenes"].items():
        print(
            f"{scene:11s} predicted {row['predicted_count']:2d}  annotated {row['ground_truth_count']:2d}  "
            f"matched {row['matched_count']:2d}  extras {row['extra_count']}  misses {row['miss_count']}  "
            f"Jaccard {100 * event_jaccard(row):6.2f}%  endpoint MAE {row['endpoint_mae_frames']}"
        )
    total = result["aggregate"]
    print(
        f"all        matched {total['matched_events']}/{total['ground_truth_events']} annotated, "
        f"{total['extras']} extras, {total['misses']} misses; mean event Jaccard "
        f"{100 * total['mean_event_jaccard']:.2f}%; endpoint MAE {total['endpoint_mae_frames']} frames"
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
