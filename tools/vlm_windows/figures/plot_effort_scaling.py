# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper figure "VLM performance across reasoning-effort levels" (supplement).

Event-set Jaccard and endpoint MAE of the post-processed windows against the
mean reasoning tokens per scene, for 10 GPT-5.6 Sol rollouts at each of five
reasoning efforts, read from results/gpt-5.6-sol/<effort>/trial<N>/.

Needs the Linux Libertine 5.3.0 TrueType file LinLibertine_Rah.ttf (--font).
--data-only prints the plotted numbers without rendering (no font needed).
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
DEFAULT_OUTPUT_STEM = FIGURE_DIR / "out" / "vlm_event_jaccard_mae_scaling_gpt56sol_n10_aligned"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
EFFORT_CODES = {
    "low": "L",
    "medium": "M",
    "high": "H",
    "xhigh": "X",
    "max": "MAX",
}
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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def load_points(efforts: list[str], trials: list[int], rollout_root: Path = ROLLOUT_ROOT) -> list[dict[str, Any]]:
    """One point per rollout: reasoning tokens per scene, mean event Jaccard, endpoint MAE."""
    points: list[dict[str, Any]] = []
    for effort in efforts:
        for trial in trials:
            rollout = rollout_root / effort / f"trial{trial}"
            index = load_json(rollout / "index.json")
            evaluation = load_json(rollout / "evaluation_min20_nms_iou80.json")
            aggregate = evaluation["aggregate"]
            scenes = list(evaluation["scenes"].values())
            scene_jaccards = [
                scene["matched_count"] / (scene["matched_count"] + scene["extra_count"] + scene["miss_count"])
                for scene in scenes
            ]
            reasoning_tokens = [int(row["usage"]["reasoning_output_tokens"]) for row in index.values()]
            points.append(
                {
                    "effort": effort,
                    "trial": trial,
                    "label": f"{EFFORT_CODES[effort]}{trial}",
                    "reasoning_tokens_per_scene": sum(reasoning_tokens) / len(reasoning_tokens),
                    "macro_event_jaccard": 100 * sum(scene_jaccards) / len(scene_jaccards),
                    "endpoint_mae_frames": float(aggregate["endpoint_mae_frames"]),
                    "extras": int(aggregate["extras"]),
                    "misses": int(aggregate["misses"]),
                }
            )
    return points


def uncertainty_errors(values: list[float], kind: str) -> tuple[float, float]:
    average = sum(values) / len(values)
    if kind == "range":
        return average - min(values), max(values) - average
    from scipy.stats import t

    half_width = float(t.ppf(0.975, len(values) - 1)) * stdev(values) / sqrt(len(values))
    return half_width, half_width


def style() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Linux Libertine"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.55,
        "lines.linewidth": 0.9,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
    }


def render(
    points: list[dict[str, Any]],
    output_stem: Path,
    dpi: int,
    font: Path,
    pdf_creation_date: datetime | None = None,
) -> None:
    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not font.is_file():
        raise FileNotFoundError(
            f"Required paper font is missing: {font}. Pass --font /path/to/LinLibertine_Rah.ttf "
            "(Linux Libertine 5.3.0 TrueType; see README.md)."
        )
    font_manager.fontManager.addfont(font)

    with mpl.rc_context(style()):
        figure, axes = plt.subplots(
            2,
            1,
            figsize=(3.45, 3.35),
            dpi=dpi,
            sharex=True,
        )
        figure.subplots_adjust(
            left=0.19,
            right=0.99,
            bottom=0.13,
            top=0.91,
            hspace=0.16,
        )

        panels = (
            (
                "macro_event_jaccard",
                "Event-set Jaccard (%)",
                (90, 100.5),
                (90, 92, 94, 96, 98, 100),
            ),
            (
                "endpoint_mae_frames",
                "Endpoint MAE (frames)",
                (2, 6),
                (2, 3, 4, 5, 6),
            ),
        )
        for axis, (metric, label, limits, ticks) in zip(axes, panels):
            mean_token_values = []
            mean_metric_values = []
            for effort in EFFORTS:
                group = [point for point in points if point["effort"] == effort]
                token_values = [float(point["reasoning_tokens_per_scene"]) / 1000 for point in group]
                metric_values = [float(point[metric]) for point in group]
                mean_tokens = mean(token_values)
                mean_metric = mean(metric_values)
                mean_token_values.append(mean_tokens)
                mean_metric_values.append(mean_metric)
                metric_ci = uncertainty_errors(metric_values, "ci95")
                if metric == "macro_event_jaccard":
                    metric_ci = (
                        min(metric_ci[0], mean_metric),
                        min(metric_ci[1], 100 - mean_metric),
                    )

                axis.scatter(
                    token_values,
                    metric_values,
                    s=18,
                    marker="o",
                    facecolor=COLORS[effort],
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.58,
                    zorder=2,
                )
                axis.errorbar(
                    mean_tokens,
                    mean_metric,
                    yerr=[[metric_ci[0]], [metric_ci[1]]],
                    fmt="D",
                    markersize=4.4,
                    markerfacecolor="white",
                    markeredgecolor=COLORS[effort],
                    markeredgewidth=1.0,
                    ecolor=COLORS[effort],
                    elinewidth=0.8,
                    capsize=2.0,
                    zorder=4,
                )

            axis.plot(
                mean_token_values,
                mean_metric_values,
                color="#777777",
                linewidth=0.65,
                alpha=0.55,
                zorder=1,
            )
            axis.set_ylim(*limits)
            axis.set_yticks(ticks)
            axis.set_ylabel(label)
            axis.yaxis.set_label_coords(-0.13, 0.5)
            axis.yaxis.grid(True, color="#E5E5E5", linewidth=0.45)
            axis.set_axisbelow(True)

        axes[-1].set_xlim(0, 7.2)
        axes[-1].set_xticks((0, 1, 2, 3, 4, 5, 6, 7))
        axes[-1].set_xlabel("Reasoning tokens / scene (×10³)")

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=4.5,
                markerfacecolor=COLORS[effort],
                markeredgecolor="white",
                markeredgewidth=0.4,
                label=EFFORT_LABELS[effort],
            )
            for effort in EFFORTS
        ]
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.59, 0.995),
            ncol=5,
            handletextpad=0.25,
            columnspacing=0.75,
            borderaxespad=0,
        )

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "svg", "png"):
            figure.savefig(
                output_stem.with_suffix(f".{suffix}"),
                dpi=dpi if suffix == "png" else None,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.02,
                metadata=(
                    {"CreationDate": pdf_creation_date} if suffix == "pdf" and pdf_creation_date is not None else None
                ),
            )
        plt.close(figure)


def summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for effort in EFFORTS:
        group = [point for point in points if point["effort"] == effort]
        rows[effort] = {
            "mean_reasoning_tokens_per_scene": mean(point["reasoning_tokens_per_scene"] for point in group),
            "mean_event_jaccard_percent": mean(point["macro_event_jaccard"] for point in group),
            "mean_endpoint_mae_frames": mean(point["endpoint_mae_frames"] for point in group),
            "count_perfect_trials": [
                point["trial"] for point in group if point["extras"] == 0 and point["misses"] == 0
            ],
        }
    return {"efforts": rows, "points": points}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--trials", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT, help="LinLibertine_Rah.ttf")
    parser.add_argument(
        "--pdf-creation-date",
        type=datetime.fromisoformat,
        default=None,
        help="Naive local time stored as the PDF CreationDate (default: now); see README.md",
    )
    parser.add_argument("--data-only", action="store_true", help="Print the plotted numbers as JSON")
    args = parser.parse_args()
    if not args.trials or len(set(args.trials)) != len(args.trials):
        raise SystemExit("--trials must be nonempty and contain no duplicates")
    points = load_points(list(EFFORTS), args.trials)
    if args.data_only:
        print(json.dumps(summary(points), indent=2))
        return
    render(points, args.output_stem.resolve(), args.dpi, args.font, args.pdf_creation_date)


if __name__ == "__main__":
    main()
