# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper figure "VLM slot-prediction failure modes" (supplement).

Counts every extra and missed window of the 50 GPT-5.6 Sol rollouts (5 efforts x
10 trials x 12 scenes, after post-processing) by failure mode, read from
results/gpt-5.6-sol/<effort>/trial<N>/evaluation_min20_nms_iou80.json.

Errors are assigned to the four modes per scene, as decided by inspecting the
failures, not computed from the windows:
  * Missing slot: every miss in Flat 3 (a closing hidden by occlusion) and in
    Chair (one of two similar chairs omitted);
  * Incorrect merging: every miss in Car (two cycles of the same door merged);
  * Over-segmentation: every extra window in any scene except Balcony;
  * Balcony: its seven errors were classified one by one (BALCONY_* below);
    they include the only three false positives.
The script stops if a miss occurs in any other scene or the totals change
(533 count-perfect scene-runs of 600; 76 errors).

Needs LinLibertine_Rah.ttf (--font). --data-only writes the data file only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any

FIGURE_DIR = Path(__file__).resolve().parent
ROLLOUT_ROOT = FIGURE_DIR.parent / "results" / "gpt-5.6-sol"
DEFAULT_FONT = FIGURE_DIR / "fonts" / "LinLibertine_Rah.ttf"
DEFAULT_OUTPUT_STEM = FIGURE_DIR / "out" / "vlm_failure_mode_frequency_gpt56sol_n10"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
TAXONOMY = (
    "Occluded closure\n(miss)",
    "Temporal over-segmentation\n(extra)",
    "Repeated-cycle merge\n(miss)",
    "Similar-object omission\n(miss)",
    "Articulated-part ambiguity\n(mixed)",
)
COLORS = {
    "low": "#2F86CF",
    "medium": "#2AA876",
    "high": "#E28A2B",
    "xhigh": "#8A63C7",
    "max": "#D45454",
}
EFFORT_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "XHigh",
    "max": "Max",
}
DISPLAY_LABELS = (
    "Missing slot",
    "Over-segmentation",
    "Incorrect merging",
    "False positive",
)

# The original audit grouped every Balcony error together. Reinspection of the
# four failed Balcony scene-runs separates its seven errors as follows, in
# low/medium/high/xhigh/max effort order.
BALCONY_MISSING = (0, 2, 0, 0, 0)
BALCONY_OVERSEGMENTED = (1, 0, 0, 0, 0)
BALCONY_MERGED = (1, 0, 0, 0, 0)
BALCONY_FALSE_DETECTION = (0, 3, 0, 0, 0)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def evaluation_path(effort: str, trial: int, rollout_root: Path = ROLLOUT_ROOT) -> Path:
    return rollout_root / effort / f"trial{trial}" / "evaluation_min20_nms_iou80.json"


def ci95(values: list[float]) -> float:
    from scipy.stats import t

    return float(t.ppf(0.975, len(values) - 1)) * stdev(values) / sqrt(len(values))


def collect(trials: list[int], rollout_root: Path = ROLLOUT_ROOT) -> tuple[dict[str, Any], list[list[int]]]:
    """Per-effort count-perfect rates and the per-scene error taxonomy (rows = TAXONOMY)."""
    success: dict[str, Any] = {}
    error_matrix: list[list[int]] = [[0 for _ in EFFORTS] for _ in TAXONOMY]
    for effort_index, effort in enumerate(EFFORTS):
        trial_rates: list[float] = []
        aggregate_errors = 0
        for trial in trials:
            evaluation = load_json(evaluation_path(effort, trial, rollout_root))
            scenes = evaluation["scenes"]
            exact = sum(scene["extra_count"] == 0 and scene["miss_count"] == 0 for scene in scenes.values())
            trial_rates.append(100 * exact / len(scenes))

            error_matrix[0][effort_index] += scenes["flat3"]["miss_count"]
            error_matrix[1][effort_index] += sum(
                scene["extra_count"] for name, scene in scenes.items() if name != "balcony"
            )
            error_matrix[2][effort_index] += scenes["car"]["miss_count"]
            error_matrix[3][effort_index] += scenes["chair"]["miss_count"]
            error_matrix[4][effort_index] += scenes["balcony"]["extra_count"] + scenes["balcony"]["miss_count"]
            aggregate_errors += evaluation["aggregate"]["extras"] + evaluation["aggregate"]["misses"]

            known_misses = sum(scenes[name]["miss_count"] for name in ("flat3", "car", "chair", "balcony"))
            known_extras = sum(scene["extra_count"] for scene in scenes.values())
            if known_misses != evaluation["aggregate"]["misses"]:
                raise ValueError(
                    f"Unclassified miss in {effort} trial {trial}: "
                    f"known={known_misses}, total={evaluation['aggregate']['misses']}"
                )
            if known_extras != evaluation["aggregate"]["extras"]:
                raise ValueError(
                    f"Unclassified extra in {effort} trial {trial}: "
                    f"known={known_extras}, total={evaluation['aggregate']['extras']}"
                )

        classified_errors = sum(row[effort_index] for row in error_matrix)
        if classified_errors != aggregate_errors:
            raise ValueError(
                f"Taxonomy mismatch for {effort}: classified={classified_errors}, aggregate={aggregate_errors}"
            )
        success[effort] = {
            "trial_rates": trial_rates,
            "mean": mean(trial_rates),
            "ci95": ci95(trial_rates),
            "exact_scene_runs": round(sum(trial_rates) * 12 / 100),
            "scene_runs": 12 * len(trials),
        }
    return success, error_matrix


def intuitive_error_matrix(coarse_matrix: list[list[int]]) -> list[list[int]]:
    """Map the five per-scene modes onto the four modes shown in the paper (rows = DISPLAY_LABELS)."""
    occluded, oversegmented, merged, omitted, balcony = coarse_matrix
    balcony_breakdown = [
        sum(values)
        for values in zip(
            BALCONY_MISSING,
            BALCONY_OVERSEGMENTED,
            BALCONY_MERGED,
            BALCONY_FALSE_DETECTION,
        )
    ]
    if balcony != balcony_breakdown:
        raise ValueError("The audited Balcony error breakdown no longer matches")
    return [
        [occluded[index] + omitted[index] + BALCONY_MISSING[index] for index in range(len(EFFORTS))],
        [oversegmented[index] + BALCONY_OVERSEGMENTED[index] for index in range(len(EFFORTS))],
        [merged[index] + BALCONY_MERGED[index] for index in range(len(EFFORTS))],
        list(BALCONY_FALSE_DETECTION),
    ]


def figure_data(rollout_root: Path = ROLLOUT_ROOT) -> dict[str, Any]:
    trials = list(range(1, 11))
    success, coarse_error_matrix = collect(trials, rollout_root)
    error_matrix = intuitive_error_matrix(coarse_error_matrix)
    total_scene_runs = sum(row["scene_runs"] for row in success.values())
    exact_scene_runs = sum(row["exact_scene_runs"] for row in success.values())
    if (exact_scene_runs, total_scene_runs, sum(map(sum, error_matrix))) != (533, 600, 76):
        raise ValueError("The frozen 50-rollout failure totals changed; re-audit the plot")
    return {
        "model": "gpt-5.6-sol",
        "trials_per_effort": 10,
        "scene_runs": total_scene_runs,
        "count_perfect_scene_runs": exact_scene_runs,
        "efforts": EFFORTS,
        "failure_modes": DISPLAY_LABELS,
        "error_matrix": error_matrix,
    }


def style() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Linux Libertine"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    }


def render(
    error_matrix: list[list[int]],
    output_stem: Path,
    dpi: int,
    font: Path,
    pdf_creation_date: datetime | None = None,
) -> None:
    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    if not font.is_file():
        raise FileNotFoundError(
            f"Required paper font is missing: {font}. Pass --font /path/to/LinLibertine_Rah.ttf "
            "(Linux Libertine 5.3.0 TrueType; see README.md)."
        )
    font_manager.fontManager.addfont(font)

    totals = [sum(row) for row in error_matrix]
    order = sorted(range(len(totals)), key=lambda index: totals[index], reverse=True)
    matrix = np.asarray([error_matrix[index] for index in order], dtype=int)
    labels = [DISPLAY_LABELS[index] for index in order]

    with mpl.rc_context(style()):
        figure, axis = plt.subplots(figsize=(5.15, 2.85), dpi=dpi)
        figure.subplots_adjust(left=0.185, right=0.97, bottom=0.16, top=0.78)

        y = np.arange(len(labels))
        left = np.zeros(len(labels), dtype=int)
        for effort_index, effort in enumerate(EFFORTS):
            values = matrix[:, effort_index]
            axis.barh(
                y,
                values,
                left=left,
                height=0.58,
                color=COLORS[effort],
                edgecolor="white",
                linewidth=0.5,
                label=EFFORT_LABELS[effort],
                zorder=2,
            )
            for row, value in enumerate(values):
                if not value:
                    continue
                axis.text(
                    left[row] + value / 2,
                    row,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    fontweight="bold",
                    color="#202020",
                    zorder=3,
                )
            left += values

        axis.set_xlim(0, 35.5)
        axis.set_xticks((0, 5, 10, 15, 20, 25, 30, 35))
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Number of errors")
        axis.xaxis.grid(True, color="#E5E5E5", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines["left"].set_visible(False)
        axis.set_title("VLM slot prediction errors", loc="left", pad=28)

        handles = [Patch(facecolor=COLORS[effort], edgecolor="none", label=EFFORT_LABELS[effort]) for effort in EFFORTS]
        axis.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=5,
            handlelength=0.9,
            handleheight=0.7,
            handletextpad=0.3,
            columnspacing=0.9,
            borderaxespad=0,
            frameon=False,
        )

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "svg", "png"):
            figure.savefig(
                output_stem.with_suffix(f".{suffix}"),
                dpi=dpi if suffix == "png" else None,
                facecolor="white",
                bbox_inches=None,
                pad_inches=0,
                metadata=(
                    {"CreationDate": pdf_creation_date} if suffix == "pdf" and pdf_creation_date is not None else None
                ),
            )
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT, help="LinLibertine_Rah.ttf")
    parser.add_argument(
        "--pdf-creation-date",
        type=datetime.fromisoformat,
        default=None,
        help="Naive local time stored as the PDF CreationDate (default: now); see README.md",
    )
    parser.add_argument("--data-only", action="store_true", help="Write only <stem>_data.json")
    args = parser.parse_args()

    data = figure_data()
    output_stem = args.output_stem.resolve()
    if not args.data_only:
        render(data["error_matrix"], output_stem, args.dpi, args.font, args.pdf_creation_date)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_stem.with_name(f"{output_stem.name}_data").with_suffix(".json").write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
