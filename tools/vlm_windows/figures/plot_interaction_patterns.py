# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper figure "SwitchSplat dataset interaction events across 12 scenes" (supplement).

Bars are the annotated interaction windows (results/ground_truth/<scene>.txt,
checked against the sha256 values in interaction_patterns_inputs.json). The short
object names are visualization-only aliases of the GPT-5.6 Sol descriptions
(max effort, trial 1) matched to each annotated window; the script checks every
alias against results/gpt-5.6-sol/max/trial1/evaluation_raw.json and the raw
observations before drawing.

Needs Liberation Serif 2.x (regular and bold), installed or given with
--font-regular/--font-bold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import string
from datetime import datetime
from pathlib import Path
from typing import Any

FIGURE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = FIGURE_DIR.parent / "results"
DEFAULT_INPUTS = FIGURE_DIR / "interaction_patterns_inputs.json"
DEFAULT_DATASET_ROOT = RESULTS_DIR / "ground_truth"
DEFAULT_OUTPUT_PREFIX = FIGURE_DIR / "out" / "v7_dataset_interaction_patterns_paper_v3_grid_structured"
FIGURE_WIDTH_IN = 7.1
FIGURE_HEIGHT_IN = 7.15
EXPECTED_SCENES = 12
EXPECTED_INTERVALS = 59

INTERACTION_COLOR = "#0072B2"  # Okabe-Ito blue
TEXT_COLOR = "#202428"
MUTED_COLOR = "#606A73"
GRID_COLOR = "#D9DEE2"

# The paper figure uses Liberation Serif, a metric-compatible stand-in for Times.
PREVIEW_FONT = "Liberation Serif"

# Paper-facing names.
PAPER_GROUPS = (
    (
        "STANDARD SCENES",
        "",
        (
            ("balcony", "Balcony"),
            ("car", "Car"),
            ("kitchen1", "Kitchen 1"),
            ("flat1", "Flat 1"),
            ("flat2", "Flat 2"),
            ("flat3", "Flat 3"),
            ("chair", "Chair"),
            ("m_kitchen1", "M Kitchen 1"),
            ("m_kitchen2", "M Kitchen 2"),
            ("m_kitchen3", "M Kitchen 3"),
        ),
    ),
    (
        "GOLDEN SCENES",
        "",
        (
            ("kitchen2", "Kitchen 2"),
            ("oslo", "Oslo"),
        ),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def read_intervals(path: Path) -> list[list[int]]:
    intervals: list[list[int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected two frame values")
        start, end = map(int, fields)
        if start >= end:
            raise ValueError(f"{path}:{line_number}: expected start < end")
        intervals.append([start, end])
    return intervals


def verify_semantic_sources(spec: dict[str, Any]) -> dict[str, Any]:
    """Check that each alias still belongs to the VLM event matched to that annotated window."""
    source = spec["semantic_source_evaluation"]
    evaluation_path = RESULTS_DIR / source["path"]
    if sha256_file(evaluation_path) != source["sha256"]:
        raise ValueError(f"Semantic source hash mismatch: {evaluation_path}")
    evaluation = load_json(evaluation_path)

    records: dict[str, Any] = {}
    for scene in spec["scenes"]:
        scene_id = scene["scene_id"]
        evaluated_scene = evaluation["scenes"][scene_id]
        raw_path = RESULTS_DIR / evaluated_scene["raw_observation"]
        raw_hash = sha256_file(raw_path)
        if raw_hash != scene["semantic_source_sha256"]:
            raise ValueError(f"Semantic observation hash mismatch: {raw_path}")
        events = load_json(raw_path)["events"]
        matches = {
            int(match["ground_truth_index"]): int(match["prediction_index"]) for match in evaluated_scene["matches"]
        }
        for gt_index, label in enumerate(scene["labels"]):
            prediction_index = int(label["prediction_index"])
            if matches.get(gt_index) != prediction_index:
                raise ValueError(f"{scene_id} GT {gt_index}: frozen prediction index no longer matches")
            if events[prediction_index]["object_description"] != label["source"]:
                raise ValueError(f"{scene_id} GT {gt_index}: frozen semantic description mismatch")
        records[scene_id] = {
            "evaluation_path": source["path"],
            "evaluation_sha256": source["sha256"],
            "raw_observation": evaluated_scene["raw_observation"],
            "raw_observation_sha256": raw_hash,
            "verified_mapping_count": len(scene["labels"]),
        }
    return records


def load_frozen_inputs(
    spec_path: Path,
    dataset_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    spec = load_json(spec_path)
    scenes = spec.get("scenes", [])
    if len(scenes) != EXPECTED_SCENES:
        raise ValueError(f"Expected {EXPECTED_SCENES} scenes, found {len(scenes)}")
    semantic_provenance = verify_semantic_sources(spec)

    compiled: list[dict[str, Any]] = []
    endpoint_count = 0
    for scene in scenes:
        endpoint_path = dataset_root / scene["endpoint_file"]
        endpoint_hash = sha256_file(endpoint_path)
        if endpoint_hash != scene["endpoint_sha256"]:
            raise ValueError(f"Endpoint hash mismatch: {endpoint_path}")
        intervals = read_intervals(endpoint_path)
        if len(intervals) != len(scene["labels"]):
            raise ValueError(
                f"{scene['scene_id']}: {len(intervals)} endpoints but {len(scene['labels'])} semantic labels"
            )
        frame_min, frame_max = map(int, scene["camera1_frame_range"])
        for start, end in intervals:
            if not (frame_min <= start < end <= frame_max):
                raise ValueError(f"{scene['scene_id']}: interval {start, end} outside frame range")
        endpoint_count += len(intervals)
        compiled.append(
            {
                **scene,
                "endpoint_path": str(endpoint_path),
                "endpoint_sha256": endpoint_hash,
                "intervals": intervals,
            }
        )
    if endpoint_count != EXPECTED_INTERVALS:
        raise ValueError(f"Expected {EXPECTED_INTERVALS} intervals, found {endpoint_count}")
    return spec, compiled, semantic_provenance


def progress(frame: int, frame_min: int, frame_max: int) -> float:
    return (frame - frame_min) / (frame_max - frame_min)


def verify_layout(figure: Any, essential_text: list[Any]) -> dict[str, Any]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_box = figure.bbox
    boxes = [artist.get_window_extent(renderer=renderer) for artist in essential_text]
    clipped = [
        index
        for index, box in enumerate(boxes)
        if box.x0 < figure_box.x0 or box.y0 < figure_box.y0 or box.x1 > figure_box.x1 or box.y1 > figure_box.y1
    ]
    collisions: list[list[int]] = []
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            overlap = boxes[first].intersection(boxes[first], boxes[second])
            if overlap is not None and overlap.width > 0.5 and overlap.height > 0.5:
                collisions.append([first, second])
    if clipped:
        clipped_text = [essential_text[index].get_text() for index in clipped]
        raise ValueError(f"Essential text clipped at final size: {clipped_text}")
    if collisions:
        raise ValueError(f"Essential text collisions at final size: {collisions}")
    return {
        "essential_text_count": len(essential_text),
        "clipped_text_count": len(clipped),
        "text_collision_count": len(collisions),
    }


def canonical_scenes(compiled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {scene["scene_id"]: scene for scene in compiled}
    ordered: list[dict[str, Any]] = []
    for group_name, group_note, entries in PAPER_GROUPS:
        for scene_id, paper_name in entries:
            ordered.append(
                {
                    **by_id[scene_id],
                    "display_name": paper_name,
                    "paper_group": group_name,
                    "paper_group_note": group_note,
                }
            )
    return ordered


def draw_panel(
    figure: Any,
    position: Any,
    scene: dict[str, Any],
    panel_index: int,
    show_ticks: bool,
    horizontal_position: str,
    guide_style: str,
) -> tuple[Any, list[Any]]:
    x_by_position = {"left": 0.235, "center": 0.480, "right": 0.725}
    axis = figure.add_axes([x_by_position[horizontal_position], position.y0, 0.260, position.height])
    intervals = scene["intervals"]
    labels = scene["labels"]
    frame_min, frame_max = map(int, scene["camera1_frame_range"])
    count = len(intervals)
    essential_text: list[Any] = []

    if guide_style == "full":
        for y in range(count):
            axis.plot(
                [0.0, 1.0],
                [y, y],
                color=GRID_COLOR,
                linewidth=0.35,
                zorder=0,
            )

    for interval_index, ((start, end), label) in enumerate(zip(intervals, labels)):
        y = count - interval_index - 1
        axis.plot(
            [progress(start, frame_min, frame_max), progress(end, frame_min, frame_max)],
            [y, y],
            color=INTERACTION_COLOR,
            linewidth=5.0,
            solid_capstyle="round",
            zorder=3,
        )
        essential_text.append(
            axis.text(
                -0.045,
                y,
                f"{label['short']}  {start}–{end}",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=6.35,
                color=TEXT_COLOR,
                clip_on=False,
            )
        )

    if guide_style == "full":
        for x in (0.0, 0.5, 1.0):
            axis.axvline(x, color=GRID_COLOR, linewidth=0.45, zorder=0)
    elif guide_style == "minimal":
        axis.axvline(0.5, color=GRID_COLOR, linewidth=0.55, zorder=0)
    axis.set_xlim(-0.01, 1.01)
    axis.set_ylim(-0.55, count - 0.45)
    axis.set_yticks([])
    axis.set_xticks([0.0, 0.5, 1.0])
    if show_ticks:
        axis.set_xticklabels(["0%", "50%", "100%"], fontsize=6.2)
        axis.tick_params(
            axis="x",
            length=0 if guide_style == "none" else 2.0,
            pad=1.5,
            colors=MUTED_COLOR,
        )
    else:
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    if guide_style == "full" or (guide_style == "minimal" and show_ticks):
        axis.spines["bottom"].set_color(MUTED_COLOR)
        axis.spines["bottom"].set_linewidth(0.55)
    else:
        axis.spines["bottom"].set_visible(False)
    axis.spines[["top", "right", "left"]].set_visible(False)

    panel_letter = string.ascii_lowercase[panel_index]
    essential_text.append(
        axis.text(
            -0.045,
            1.055,
            f"({panel_letter}) {scene['display_name']}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color=TEXT_COLOR,
            clip_on=False,
        )
    )
    return axis, essential_text


def section_header(figure: Any, position: Any, title: str, note: str, draw_rule: bool) -> None:
    y = (position.y0 + position.y1) / 2
    note_x = 0.270 if title == "OVERLAPPING INTERACTIONS" else 0.170
    figure.text(
        0.018,
        y,
        title,
        ha="left",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    line_start = 0.170 if not note else note_x + 0.255
    if note:
        figure.text(
            note_x,
            y,
            note,
            ha="left",
            va="center",
            fontsize=6.15,
            fontstyle="italic",
            color=MUTED_COLOR,
        )
    if draw_rule:
        figure.lines.append(
            __import__("matplotlib").lines.Line2D(
                [line_start, 0.985],
                [y, y],
                transform=figure.transFigure,
                color=GRID_COLOR,
                linewidth=0.65,
            )
        )


def build_figure(scenes: list[dict[str, Any]], dpi: int, guide_style: str) -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": PREVIEW_FONT,
            "font.size": 7.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=dpi)
    row_ratios = [1.0, 5.5, 9.5, 9.5, 3.8, 3.8, 1.0, 3.8]
    grid = figure.add_gridspec(
        len(row_ratios),
        1,
        left=0.0,
        right=1.0,
        top=0.975,
        bottom=0.060,
        hspace=0.30,
        height_ratios=row_ratios,
    )

    section_header(
        figure,
        grid[0].get_position(figure),
        "STANDARD SCENES",
        "",
        guide_style == "full",
    )
    placements = (
        (1, "left", False),
        (1, "right", False),
        (2, "left", False),
        (2, "right", False),
        (3, "left", False),
        (3, "right", False),
        (4, "left", False),
        (4, "right", False),
        (5, "left", True),
        (5, "right", True),
        (7, "left", True),
        (7, "right", True),
    )

    essential_text: list[Any] = []
    for panel_index, (scene, (row, horizontal_position, show_ticks)) in enumerate(zip(scenes, placements)):
        position = grid[row].get_position(figure)
        _, panel_text = draw_panel(
            figure,
            position,
            scene,
            panel_index,
            show_ticks,
            horizontal_position,
            guide_style,
        )
        essential_text.extend(panel_text)

    section_header(
        figure,
        grid[6].get_position(figure),
        "GOLDEN SCENES",
        "",
        guide_style != "none",
    )
    figure.text(
        0.985,
        0.018,
        "Normalized scene progress",
        ha="right",
        va="center",
        fontsize=6.5,
        color=TEXT_COLOR,
    )
    verify_layout(figure, essential_text)
    return figure


def register_fonts(paths: list[Path]) -> None:
    """Add given font files, then make sure Liberation Serif resolves (no silent fallback)."""
    import matplotlib.font_manager as font_manager

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Font file not found: {path}")
        font_manager.fontManager.addfont(path)
    for weight in ("normal", "bold"):
        try:
            found = font_manager.findfont(
                font_manager.FontProperties(family=PREVIEW_FONT, weight=weight),
                fallback_to_default=False,
            )
        except ValueError as exc:
            raise SystemExit(
                f"{PREVIEW_FONT} is not available to matplotlib. Install Liberation Serif 2.x "
                "(e.g. the fonts-liberation2 package) or pass --font-regular/--font-bold."
            ) from exc
        if font_manager.get_font(found).family_name != PREVIEW_FONT:
            raise SystemExit(f"matplotlib resolved {PREVIEW_FONT} ({weight}) to {found}")


def render(args: argparse.Namespace) -> list[Path]:
    import matplotlib.pyplot as plt

    register_fonts([path for path in (args.font_regular, args.font_bold) if path is not None])
    _, compiled, _ = load_frozen_inputs(args.inputs.resolve(), args.dataset_root.resolve())
    scenes = canonical_scenes(compiled)
    figure = build_figure(scenes, args.dpi, args.guide_style)
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_prefix.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    figure.savefig(outputs[0], dpi=args.dpi, facecolor="white")
    figure.savefig(
        outputs[1],
        facecolor="white",
        metadata={"CreationDate": args.pdf_creation_date} if args.pdf_creation_date is not None else None,
    )
    figure.savefig(outputs[2], facecolor="white")
    plt.close(figure)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Folder holding the endpoint files named in --inputs (default: results/ground_truth)",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--guide-style",
        choices=("minimal", "full", "none"),
        default="full",
        help="The paper figure uses 'full'",
    )
    parser.add_argument("--font-regular", type=Path, default=None, help="LiberationSerif-Regular.ttf")
    parser.add_argument("--font-bold", type=Path, default=None, help="LiberationSerif-Bold.ttf")
    parser.add_argument(
        "--pdf-creation-date",
        type=datetime.fromisoformat,
        default=None,
        help="Naive local time stored as the PDF CreationDate (default: now); see README.md",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for output in render(args):
        print(output)


if __name__ == "__main__":
    main()
