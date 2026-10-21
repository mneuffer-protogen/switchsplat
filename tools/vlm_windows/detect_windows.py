# SPDX-FileCopyrightText: Copyright (c) 2026 The SwitchSplat Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ask a vision-language model for the interaction windows of one scene.

This is the observer used for the SwitchSplat paper, reduced to the single
configuration the paper used: camera-1 frames whose index is a multiple of the
stride (8), plus the first and last frame, are resized to 480-pixel-wide
thumbnails stamped with "t=FRAME" and attached, in order, to one Codex CLI
rollout of GPT-5.6 Sol together with prompt_template.txt and the JSON schema
output_schema.json.

Modes:
  (default) dry run   write the prompt, the input images, their manifest and the
                      exact Codex command to --out-dir; no model is called.
  --run               additionally run `codex exec` once and store the model's
                      JSON answer as raw_observation.json.
  --replay RECORD     instead of calling a model, reuse a recorded answer such
                      as results/gpt-5.6-sol/max/trial1/balcony, after checking
                      that the rebuilt prompt is byte-identical to the recorded one.

Turn raw_observation.json into windows with postprocess.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
PROMPT_TEMPLATE = TOOL_DIR / "prompt_template.txt"
OUTPUT_SCHEMA = TOOL_DIR / "output_schema.json"

OBSERVER_VERSION = "v7_codex_raw_structured_observation"
INPUT_FORMAT = "direct_frames_labeled_stride"
PROMPT_PROFILE = "semantic_parts"
DEFAULT_CAMERA = "camera1"
DEFAULT_STRIDE = 8
DEFAULT_TILE_WIDTH = 480
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "max"
DEFAULT_REASONING_SUMMARY = "detailed"
PAPER_CODEX_VERSION = "codex-cli 0.148.0"
DEFAULT_FONT = "DejaVuSans-Bold.ttf"
EXPECTED_FONT_NAME = ("DejaVu Sans", "Bold")
CANVAS_COLOR = (18, 18, 18)
LABEL_BOX_COLOR = (0, 0, 0)
LABEL_TEXT_COLOR = (255, 226, 92)
JPEG_QUALITY = 92
WORKDIR_PLACEHOLDER = "<WORKDIR>"
WORKDIR_PREFIX = "df360-v7-direct-"
ISOLATED_ROLLOUT_DISABLED_FEATURES = (
    "shell_tool",
    "view_image",
    "browser_use",
    "computer_use",
    "apps",
    "plugins",
    "multi_agent",
    "image_generation",
)
CONTRACT = {
    "api_calls": 0,
    "codex_rollouts": 1,
    "structured_output": True,
    "refinement": False,
    "merge": False,
    "gap_recovery": False,
    "episode_postprocessing": False,
    "ground_truth_used_for_observation": False,
}
REGULAR_FRAME_RE = re.compile(r"^frame_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


@dataclass(frozen=True)
class Frame:
    time_index: int
    path: Path
    camera_id: str


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def input_digest(input_sha256: list[str]) -> str:
    """One hash for an ordered list of input-image hashes (stored in results/)."""
    return text_sha256("\n".join(input_sha256) + "\n")


def render_prompt(frame_min: int, frame_max: int, stride: int) -> str:
    """Fill prompt_template.txt; the result is the exact text sent on stdin."""
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    for name in ("frame_min", "frame_max", "stride"):
        placeholder = "{" + name + "}"
        if placeholder not in template:
            raise RuntimeError(f"{PROMPT_TEMPLATE.name} lacks the placeholder {placeholder}")
    prompt = (
        template.replace("{frame_min}", str(frame_min))
        .replace("{frame_max}", str(frame_max))
        .replace("{stride}", str(stride))
    )
    if "{" in prompt or "}" in prompt:
        raise RuntimeError(f"{PROMPT_TEMPLATE.name} contains an unknown placeholder")
    return prompt


def parse_regular_frame(path: Path) -> int | None:
    match = REGULAR_FRAME_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def default_image_root(scene_root: Path) -> Path:
    candidates = [
        scene_root / "undistort" / "images_2",
        scene_root / "undistort" / "images",
        scene_root / "images_2",
        scene_root / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find an image directory under {scene_root}. Pass --image-root explicitly.")


def discover_frames(image_root: Path, camera: str) -> list[Frame]:
    """Regular frames frame_<N>.<ext> of one camera, sorted by N.

    Novel-view test images (frame_test_novel_...) do not match and are ignored.
    """
    camera_dirs = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    if camera_dirs and camera not in camera_dirs:
        raise FileNotFoundError(f"Camera {camera!r} not found. Available: {', '.join(camera_dirs)}")
    camera_id = camera if camera_dirs else ""
    search_root = image_root / camera_id if camera_id else image_root
    frames: list[Frame] = []
    for path in search_root.iterdir():
        if not path.is_file():
            continue
        time_index = parse_regular_frame(path)
        if time_index is None:
            continue
        frames.append(Frame(time_index=time_index, path=path, camera_id=camera_id))
    frames.sort(key=lambda item: item.time_index)
    if not frames:
        raise FileNotFoundError(f"No regular frame_#### images found in {search_root}")
    indices = [frame.time_index for frame in frames]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate frame numbers in {search_root}")
    return frames


def select_stride_frames(frames: list[Frame], stride: int) -> list[Frame]:
    """Frames whose number is a multiple of the stride, plus the first and last frame."""
    if stride <= 0:
        raise ValueError("--stride must be positive")
    selected = [frame for frame in frames if frame.time_index % stride == 0]
    if frames[0] not in selected:
        selected.insert(0, frames[0])
    if frames[-1] not in selected:
        selected.append(frames[-1])
    return sorted(set(selected), key=lambda item: item.time_index)


def label_font_size(tile_width: int) -> int:
    """30 px at a 360-pixel tile, scaled with the tile width (40 px at 480)."""
    return max(1, round(30 * tile_width / 360))


def load_label_font(font: str | None, size: int) -> Any:
    """DejaVu Sans Bold with Pillow's basic text layout, as in the paper runs.

    The paper environment had no libraqm, so Pillow laid labels out without
    OpenType kerning. With libraqm, Pillow's default, multi-digit labels come out
    about one pixel narrower; forcing the basic layout keeps the paper's labels.
    """
    from PIL import ImageFont

    requested = font or DEFAULT_FONT
    layout = getattr(getattr(ImageFont, "Layout", None), "BASIC", getattr(ImageFont, "LAYOUT_BASIC", 0))
    try:
        loaded = ImageFont.truetype(requested, size, layout_engine=layout)
    except OSError as exc:
        # matplotlib ships DejaVu Sans Bold too; it draws the frame stamps pixel for pixel like the
        # system copy used for the paper.
        bundled = None if font else matplotlib_dejavu_bold()
        if bundled is not None:
            return ImageFont.truetype(bundled, size, layout_engine=layout)
        raise SystemExit(
            f"Cannot load the label font {requested!r} ({exc}). The paper's input images use "
            "DejaVu Sans Bold. Install it (for example the Debian/Ubuntu package fonts-dejavu-core) "
            "or pass --font /path/to/DejaVuSans-Bold.ttf. This script does not fall back to "
            "Pillow's built-in font, because that would silently change the images."
        ) from exc
    if tuple(loaded.getname()) != EXPECTED_FONT_NAME:
        print(
            f"warning: label font is {' '.join(loaded.getname())}, not DejaVu Sans Bold; "
            "the input images will differ from the paper's",
            file=sys.stderr,
        )
    return loaded


def matplotlib_dejavu_bold() -> str | None:
    try:
        import matplotlib
    except ImportError:
        return None
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / DEFAULT_FONT
    return str(path) if path.is_file() else None


def font_record(font: Any, size: int) -> dict[str, Any]:
    font_path = Path(str(getattr(font, "path", "")))
    return {
        "name": " ".join(font.getname()),
        "size_px": size,
        "layout": "basic",
        "sha256": file_sha256(font_path) if font_path.is_file() else None,
    }


def relative_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def build_input_frames(
    frames: list[Frame],
    out_dir: Path,
    *,
    stride: int,
    tile_width: int,
    font: Any,
    source_base: Path,
) -> tuple[list[Path], list[dict[str, Any]], list[Frame]]:
    """Write one stamped JPEG per selected frame, exactly as the paper runs did."""
    from PIL import Image, ImageDraw

    selected = select_stride_frames(frames, stride)
    with Image.open(selected[0].path) as first:
        tile_height = max(1, int(tile_width * first.height / first.width))
    scale = tile_width / 360
    font_size = label_font_size(tile_width)

    paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for attachment_index, frame in enumerate(selected, start=1):
        path = out_dir / "input_frames" / f"image_{attachment_index:03d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(frame.path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width, tile_height))
            canvas = Image.new("RGB", (tile_width, tile_height), CANVAS_COLOR)
            canvas.paste(
                image,
                ((tile_width - image.width) // 2, (tile_height - image.height) // 2),
            )
            draw = ImageDraw.Draw(canvas)
            label = f"t={frame.time_index}"
            bounds = draw.textbbox((0, 0), label, font=font)
            box_width = bounds[2] - bounds[0] + round(16 * scale)
            box_height = bounds[3] - bounds[1] + round(12 * scale)
            draw.rectangle([0, 0, box_width, box_height], fill=LABEL_BOX_COLOR)
            text_position = (round(8 * scale), round(6 * scale) - bounds[1])
            draw.text(text_position, label, fill=LABEL_TEXT_COLOR, font=font)
            canvas.save(path, quality=JPEG_QUALITY)
        paths.append(path)
        records.append(
            {
                "attachment_index": attachment_index,
                "frame": frame.time_index,
                "source_path": relative_path(frame.path, source_base),
                "path": path.relative_to(out_dir).as_posix(),
                "width": tile_width,
                "height": tile_height,
                "timestamp_rendered": True,
                "timestamp_style": f"compact_t_only_{font_size}px",
            }
        )
    return paths, records, selected


def codex_command(
    *,
    executable: str,
    model: str,
    reasoning_effort: str,
    reasoning_summary: str,
    image_count: int,
    work_dir: str,
) -> list[str]:
    """The `codex exec` argument list of the paper runs; the prompt goes to stdin."""
    command = [
        executable,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--config",
        f'model_reasoning_summary="{reasoning_summary}"',
        "--config",
        "hide_agent_reasoning=false",
        "--output-schema",
        f"{work_dir}/output_schema.json",
        "--output-last-message",
        f"{work_dir}/result.json",
        "--cd",
        work_dir,
    ]
    for feature in ISOLATED_ROLLOUT_DISABLED_FEATURES:
        command.extend(["--disable", feature])
    command.append("--json")
    for index in range(1, image_count + 1):
        command.extend(["--image", f"{work_dir}/input_images/image_{index:04d}.jpg"])
    return command


def codex_version(executable: str) -> str | None:
    try:
        result = subprocess.run([executable, "--version"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def observed_codex_field(stderr: str, field: str) -> str | None:
    prefix = f"{field}: "
    for line in stderr.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def codex_tokens_used(stderr: str) -> int | None:
    lines = stderr.splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip() == "tokens used":
            try:
                return int(lines[index + 1].replace(",", "").strip())
            except ValueError:
                return None
    return None


def trace_details(stdout: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Token usage and any command executions from the `codex exec --json` event stream."""
    usage: dict[str, Any] | None = None
    commands: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item", {})
        if event.get("type") == "item.completed" and isinstance(item, dict) and item.get("type") == "command_execution":
            commands.append(str(item.get("command", "")))
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
    return usage, commands


def call_codex_observer(
    *,
    executable: str,
    model: str,
    reasoning_effort: str,
    reasoning_summary: str,
    prompt: str,
    image_paths: list[Path],
    schema_path: Path,
    output_path: Path,
    call_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One isolated Codex rollout: inputs are staged into a fresh empty work directory."""
    call_dir.mkdir(parents=True)
    # Same prefix as the paper runs: Codex sees this directory as its working directory.
    temporary_work_dir = tempfile.TemporaryDirectory(prefix=WORKDIR_PREFIX)
    work_dir = Path(temporary_work_dir.name)
    try:
        staged_input_dir = work_dir / "input_images"
        staged_input_dir.mkdir()
        for index, image_path in enumerate(image_paths, start=1):
            staged_path = staged_input_dir / f"image_{index:04d}{image_path.suffix}"
            try:
                os.link(image_path, staged_path)
            except OSError:
                shutil.copy2(image_path, staged_path)
        shutil.copy2(schema_path, work_dir / "output_schema.json")
        command = codex_command(
            executable=executable,
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            image_count=len(image_paths),
            work_dir=str(work_dir),
        )
        sanitized_env = {key: value for key, value in os.environ.items() if not key.startswith("OPENAI_")}
        started = datetime.now(timezone.utc)
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                input=prompt.encode("utf-8"),
                env=sanitized_env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Codex executable not found: {executable}") from exc
        ended = datetime.now(timezone.utc)
        result_path = work_dir / "result.json"
        if result_path.exists():
            shutil.copy2(result_path, output_path)
    finally:
        temporary_work_dir.cleanup()
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    (call_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (call_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    usage, command_executions = trace_details(stdout)
    metadata = {
        "backend": "codex",
        "requested_model": model,
        "observed_model": observed_codex_field(stderr, "model"),
        "observed_provider": observed_codex_field(stderr, "provider"),
        "reasoning_effort": reasoning_effort,
        "reasoning_summary": reasoning_summary,
        "json_events": True,
        "image_count": len(image_paths),
        "command": [part.replace(str(work_dir), WORKDIR_PLACEHOLDER) for part in command] + ["<stdin: prompt.txt>"],
        "isolated_temporary_workdir": True,
        "model_tools_isolated": True,
        "disabled_features": list(ISOLATED_ROLLOUT_DISABLED_FEATURES),
        "openai_environment_removed": True,
        "start_time_utc": started.isoformat(),
        "end_time_utc": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "returncode": result.returncode,
        "tokens_used": codex_tokens_used(stderr),
        "usage": usage,
        "command_executions": command_executions,
    }
    if result.returncode != 0:
        write_json(call_dir / "metadata.json", metadata)
        raise RuntimeError(f"Codex rollout failed with exit code {result.returncode}; see {call_dir.name}/stderr.log")
    if metadata["observed_model"] is None and metadata["observed_provider"] is None:
        metadata["provenance_validation"] = "requested_only_json_events"
    elif metadata["observed_model"] != model or metadata["observed_provider"] != "openai":
        write_json(call_dir / "metadata.json", metadata)
        raise RuntimeError(
            "Codex provenance mismatch: "
            f"requested={model!r}, observed={metadata['observed_model']!r}, "
            f"provider={metadata['observed_provider']!r}"
        )
    else:
        metadata["provenance_validation"] = "observed_match"
    write_json(call_dir / "metadata.json", metadata)
    if not output_path.exists():
        raise RuntimeError(f"Codex did not write its final response to {output_path.name}")
    try:
        parsed = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Codex returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Codex returned a non-object response")
    return parsed, metadata


def replay_observation(
    record_dir: Path,
    *,
    prompt_sha256: str,
    input_sha256: list[str],
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy a recorded raw_observation.json after checking that its prompt matches ours."""
    record_manifest = json.loads((record_dir / "manifest.json").read_text(encoding="utf-8"))
    if record_manifest.get("prompt_sha256") != prompt_sha256:
        raise SystemExit(
            f"Prompt mismatch: the record was produced with prompt sha256 "
            f"{record_manifest.get('prompt_sha256')}, this scene gives {prompt_sha256}. "
            "Check the scene, camera and stride."
        )
    raw_bytes = (record_dir / "raw_observation.json").read_bytes()
    observation = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(observation, dict):
        raise SystemExit("The recorded observation is not a JSON object")
    output_path.write_bytes(raw_bytes)
    recorded_digest = record_manifest.get("input_sha256_digest")
    info = {
        "record_run_name": record_manifest.get("run_name"),
        "record_model": record_manifest.get("model"),
        "record_reasoning_effort": record_manifest.get("reasoning_effort"),
        "record_raw_observation_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "prompt_identical": True,
        "input_images_identical": (None if recorded_digest is None else recorded_digest == input_digest(input_sha256)),
    }
    return observation, info


def run(args: argparse.Namespace) -> None:
    if args.run and args.replay:
        raise SystemExit("--run and --replay are mutually exclusive")
    scene_root = args.scene_root.resolve()
    image_root = (args.image_root or default_image_root(scene_root)).resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {out_dir}")

    frames = discover_frames(image_root, args.camera)
    font_size = label_font_size(args.tile_width)
    font = load_label_font(args.font, font_size)
    out_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc)

    image_paths, input_records, selected = build_input_frames(
        frames,
        out_dir,
        stride=args.stride,
        tile_width=args.tile_width,
        font=font,
        source_base=image_root,
    )
    prompt = render_prompt(frames[0].time_index, frames[-1].time_index, args.stride)
    prompt_path = out_dir / "prompt.txt"
    prompt_path.write_bytes(prompt.encode("utf-8"))
    schema_path = out_dir / "output_schema.json"
    shutil.copyfile(OUTPUT_SCHEMA, schema_path)
    input_sha256 = [file_sha256(path) for path in image_paths]
    write_json(
        out_dir / "input_manifest.json",
        {
            "input_format": INPUT_FORMAT,
            "stride": args.stride,
            "tile_width": args.tile_width,
            "items": input_records,
            "input_sha256": input_sha256,
        },
    )
    command = codex_command(
        executable=args.codex_executable,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary,
        image_count=len(image_paths),
        work_dir=WORKDIR_PLACEHOLDER,
    )
    write_json(
        out_dir / "codex_command.json",
        {
            "command": command,
            "stdin": "prompt.txt",
            "work_dir": (
                f"{WORKDIR_PLACEHOLDER} is a fresh empty temporary directory holding only "
                "output_schema.json (a copy of output_schema.json) and input_images/image_NNNN.jpg "
                "(input_frames/image_NNN.jpg in attachment order); the model's answer is read "
                f"from {WORKDIR_PLACEHOLDER}/result.json."
            ),
            "environment": "the caller's environment with every OPENAI_* variable removed",
        },
    )

    import PIL

    manifest: dict[str, Any] = {
        "observer_version": OBSERVER_VERSION,
        "status": "in_progress",
        "scene": scene_root.name,
        "image_root": relative_path(image_root, scene_root),
        "camera": frames[0].camera_id,
        "frame_count": len(frames),
        "frame_min": frames[0].time_index,
        "frame_max": frames[-1].time_index,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "prompt_profile": PROMPT_PROFILE,
        "backend": "codex",
        "sampling": {
            "input_format": INPUT_FORMAT,
            "stride": args.stride,
            "selected_frame_count": len(selected),
            "input_image_count": len(image_paths),
            "tile_width": args.tile_width,
            "tile_height": input_records[0]["height"],
            "image_detail": "codex_default_high",
        },
        "label_font": font_record(font, font_size),
        "contract": CONTRACT,
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "prompt_template_sha256": file_sha256(PROMPT_TEMPLATE),
        "prompt": "prompt.txt",
        "prompt_sha256": file_sha256(prompt_path),
        "output_schema": "output_schema.json",
        "output_schema_sha256": file_sha256(schema_path),
        "input_manifest": "input_manifest.json",
        "input_sha256_digest": input_digest(input_sha256),
        "codex_command": "codex_command.json",
        "codex": {
            "executable": Path(args.codex_executable).name,
            "version": None,
            "paper_version": PAPER_CODEX_VERSION,
            "ephemeral": True,
            "ignore_user_config": True,
            "sandbox": "read-only",
            "model_tools_isolated": True,
            "disabled_features": list(ISOLATED_ROLLOUT_DISABLED_FEATURES),
            "reasoning_summary": args.reasoning_summary,
            "json_events": True,
            "openai_environment_removed": True,
            "isolated_temporary_workdir": True,
        },
        "python": sys.version.split()[0],
        "pillow": PIL.__version__,
        "start_time_utc": started.isoformat(),
    }

    print(f"scene: {scene_root.name}")
    print(f"model/effort: {args.model}/{args.reasoning_effort}")
    print(f"regular frames: {len(frames)} ({frames[0].time_index}..{frames[-1].time_index})")
    print(f"input images: {len(image_paths)} at stride {args.stride}, {args.tile_width} px wide")
    print(f"prompt sha256: {manifest['prompt_sha256']}")

    observation_path = out_dir / "raw_observation.json"
    if not args.run and not args.replay:
        manifest["status"] = "dry_run"
        manifest["end_time_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out_dir / "manifest.json", manifest)
        print(f"dry run: no model called; prompt, images and command are in {out_dir}")
        return

    if args.replay:
        try:
            observation, info = replay_observation(
                args.replay.resolve(),
                prompt_sha256=manifest["prompt_sha256"],
                input_sha256=input_sha256,
                output_path=observation_path,
            )
        except SystemExit as exc:
            manifest.update({"status": "failed", "failure": str(exc)})
            manifest["end_time_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(out_dir / "manifest.json", manifest)
            raise
        manifest.update({"status": "replayed", "replay": info})
        if info["input_images_identical"] is False:
            print(
                "note: the input images differ from the ones the recorded answer was produced from "
                "(the prompt is identical)"
            )
    else:
        manifest["codex"]["version"] = codex_version(args.codex_executable)
        if manifest["codex"]["version"] != PAPER_CODEX_VERSION:
            print(
                f"warning: Codex CLI is {manifest['codex']['version']!r}; the paper used " f"{PAPER_CODEX_VERSION!r}",
                file=sys.stderr,
            )
        write_json(out_dir / "manifest.json", manifest)
        try:
            observation, call_metadata = call_codex_observer(
                executable=args.codex_executable,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                reasoning_summary=args.reasoning_summary,
                prompt=prompt,
                image_paths=image_paths,
                schema_path=schema_path,
                output_path=observation_path,
                call_dir=out_dir / "model_call",
            )
        except RuntimeError as exc:
            manifest["status"] = "failed"
            manifest["end_time_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["failure"] = str(exc)
            write_json(out_dir / "manifest.json", manifest)
            raise SystemExit(str(exc)) from exc
        manifest.update(
            {
                "status": "complete",
                "observed_model": call_metadata["observed_model"],
                "observed_provider": call_metadata["observed_provider"],
                "tokens_used": call_metadata["tokens_used"],
                "usage": call_metadata["usage"],
                "command_executions": len(call_metadata["command_executions"]),
                "model_call_metadata": "model_call/metadata.json",
                "reasoning_trace": "model_call/stdout.log",
            }
        )
    ended = datetime.now(timezone.utc)
    manifest.update(
        {
            "end_time_utc": ended.isoformat(),
            "elapsed_seconds": (ended - started).total_seconds(),
            "event_count": len(observation.get("events", [])),
            "raw_observation": "raw_observation.json",
            "raw_observation_sha256": file_sha256(observation_path),
        }
    )
    write_json(out_dir / "manifest.json", manifest)
    print(f"raw events: {manifest['event_count']}")
    print(f"wrote: {observation_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_root", type=Path, help="Scene folder, e.g. <dataset>/pinhole/balcony")
    parser.add_argument("--out-dir", type=Path, required=True, help="New folder for this rollout")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Image folder; default: the first of undistort/images_2, undistort/images, images_2, images",
    )
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--tile-width", type=int, default=DEFAULT_TILE_WIDTH)
    parser.add_argument(
        "--font",
        default=None,
        help="Path to DejaVuSans-Bold.ttf if Pillow cannot find it in the system font folders",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--reasoning-summary",
        choices=["none", "auto", "concise", "detailed"],
        default=DEFAULT_REASONING_SUMMARY,
    )
    parser.add_argument("--codex-executable", default="codex")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Call the model once through `codex exec`")
    mode.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Reuse a recorded answer, e.g. results/gpt-5.6-sol/max/trial1/balcony",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
