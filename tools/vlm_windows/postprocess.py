# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a raw VLM observation into interaction windows (the paper's post-processing).

1. Drop windows shorter than --minimum-duration frames (off_frame - on_frame < 20).
2. Visit the rest by decreasing confidence (ties: earlier on_frame, then earlier
   off_frame) and keep a window unless its temporal IoU with an already kept
   window is at least --nms-iou (0.8).
3. Sort the kept windows by on_frame, then off_frame.

The output has one "on off" frame pair per line, the format of the dataset's
vlm_prediction.txt and interaction_times.txt.

Usage:
  python postprocess.py results/gpt-5.6-sol/max/trial1/balcony/raw_observation.json
  python postprocess.py RUN_DIR/raw_observation.json --output RUN_DIR/vlm_prediction.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from evaluate import interval_iou, load_json, match  # noqa: E402

DEFAULT_MINIMUM_DURATION = 20
DEFAULT_NMS_IOU = 0.8


def process_events(
    events: list[dict[str, Any]],
    min_duration: int = DEFAULT_MINIMUM_DURATION,
    iou_threshold: float = DEFAULT_NMS_IOU,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Duration filter plus confidence-ranked temporal NMS; returns kept events and an audit."""
    short_indices = [
        index for index, event in enumerate(events) if int(event["off_frame"]) - int(event["on_frame"]) < min_duration
    ]
    candidates = [index for index in range(len(events)) if index not in short_indices]
    kept_indices: list[int] = []
    suppressed: list[dict[str, Any]] = []
    for index in sorted(
        candidates,
        key=lambda event_index: (
            -float(events[event_index]["confidence"]),
            int(events[event_index]["on_frame"]),
            int(events[event_index]["off_frame"]),
        ),
    ):
        interval = [int(events[index]["on_frame"]), int(events[index]["off_frame"])]
        overlaps = [
            (
                kept_index,
                interval_iou(
                    interval,
                    [
                        int(events[kept_index]["on_frame"]),
                        int(events[kept_index]["off_frame"]),
                    ],
                ),
            )
            for kept_index in kept_indices
        ]
        duplicate = max(overlaps, key=lambda item: item[1], default=None)
        if duplicate is not None and duplicate[1] >= iou_threshold:
            suppressed.append(
                {
                    "event_index": index,
                    "kept_event_index": duplicate[0],
                    "iou": round(duplicate[1], 6),
                }
            )
        else:
            kept_indices.append(index)

    kept_indices.sort(key=lambda index: (events[index]["on_frame"], events[index]["off_frame"]))
    return [events[index] for index in kept_indices], {
        "short_event_indices": short_indices,
        "nms_suppressions": suppressed,
        "kept_event_indices": kept_indices,
    }


def format_windows(events: list[dict[str, Any]]) -> str:
    return "".join(f"{int(event['on_frame'])} {int(event['off_frame'])}\n" for event in events)


def score_scene(
    original: dict[str, Any],
    events: list[dict[str, Any]],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    """Re-score one scene of a raw evaluation after post-processing (paper evaluation files)."""
    ground_truth = original["gt"]
    predictions = [[int(event["on_frame"]), int(event["off_frame"])] for event in events]
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

    result = copy.deepcopy(original)
    result.update(
        {
            "prediction": predictions,
            "predicted_count": len(predictions),
            "matched_count": len(matches),
            "extra_count": len(predictions) - len(matches),
            "miss_count": len(ground_truth) - len(matches),
            "matches": matches,
            "extra_prediction_indices": [
                index for index in range(len(predictions)) if index not in matched_predictions
            ],
            "missed_ground_truth_indices": [
                index for index in range(len(ground_truth)) if index not in matched_ground_truth
            ],
            "endpoint_mae_frames": (round(sum(endpoint_errors) / len(endpoint_errors), 3) if endpoint_errors else None),
            "postprocessing_audit": audit,
        }
    )
    return result, endpoint_errors


def postprocess_evaluation(
    evaluation: dict[str, Any],
    min_duration: int,
    iou_threshold: float,
    base_dir: Path,
) -> dict[str, Any]:
    """Post-process every scene of a raw evaluation file (results/.../evaluation_raw.json).

    Raw observations are read from the paths stored in the file, relative to base_dir.
    """
    result = copy.deepcopy(evaluation)
    scenes: dict[str, Any] = {}
    endpoint_errors: list[int] = []
    for scene, original in evaluation["scenes"].items():
        raw_events = load_json(base_dir / original["raw_observation"])["events"]
        raw_intervals = [[int(event["on_frame"]), int(event["off_frame"])] for event in raw_events]
        if raw_intervals != original["prediction"]:
            raise ValueError(f"{scene}: raw events do not match evaluation predictions")
        events, audit = process_events(raw_events, min_duration, iou_threshold)
        scenes[scene], scene_errors = score_scene(original, events, audit)
        endpoint_errors.extend(scene_errors)

    predicted_events = sum(scene["predicted_count"] for scene in scenes.values())
    ground_truth_events = sum(scene["ground_truth_count"] for scene in scenes.values())
    matched_events = sum(scene["matched_count"] for scene in scenes.values())
    result["batch"] = f"{evaluation['batch']}_min{min_duration}_nms_iou{round(iou_threshold * 100):02d}"
    result["configuration"]["postprocessing"] = True
    result["configuration"]["postprocessing_parameters"] = {
        "minimum_duration_frames": min_duration,
        "temporal_nms_iou_threshold": iou_threshold,
        "nms_ranking": "descending confidence",
    }
    result["aggregate"] = {
        "scene_count": len(scenes),
        "predicted_events": predicted_events,
        "ground_truth_events": ground_truth_events,
        "matched_events": matched_events,
        "extras": predicted_events - matched_events,
        "misses": ground_truth_events - matched_events,
        "precision": round(matched_events / predicted_events, 6),
        "recall": round(matched_events / ground_truth_events, 6),
        "endpoint_mae_frames": round(sum(endpoint_errors) / len(endpoint_errors), 3),
        "iou_match_threshold": evaluation["aggregate"]["iou_match_threshold"],
        "all_runs_verified": evaluation["aggregate"].get("all_runs_verified", False),
    }
    result["scenes"] = scenes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_observation", type=Path)
    parser.add_argument("--minimum-duration", type=int, default=DEFAULT_MINIMUM_DURATION)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--output", type=Path, default=None, help="Default: print to stdout")
    parser.add_argument("--audit", type=Path, default=None, help="Write the filter/NMS audit as JSON")
    args = parser.parse_args()
    if args.minimum_duration < 0:
        raise SystemExit("--minimum-duration must be non-negative")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise SystemExit("--nms-iou must be between 0 and 1")
    events = load_json(args.raw_observation)["events"]
    kept, audit = process_events(events, args.minimum_duration, args.nms_iou)
    text = format_windows(kept)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.audit:
        args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
