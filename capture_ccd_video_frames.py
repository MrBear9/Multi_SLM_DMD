"""Interactively extract registered 640x640 CCD intensity frames from a video.

The hardware-export ``input`` directory defines the sample order.  The user
selects the sensor-plane region independently on several later, clearly
modulated frames.  Their median ROI is confirmed before the video restarts from
the beginning for one-frame-per-sample capture.  Every confirmed frame is
perspective-rectified to the detector feature size and saved with the matching
hardware-export ID (0001.png, 0002.png, ...).

This script intentionally does not apply per-frame contrast normalization.  A
Light detector trained on optical intensity should see a consistent camera
response across samples; per-frame min/max stretching would destroy that
relationship.  Use ``--black-level`` and ``--white-level`` only after camera
calibration (or to remove a known fixed black offset).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "output" / "Tv2_dmd640_scratch" / "hardware_export_100"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WINDOW_ROI = "Select CCD optical region"
WINDOW_VERIFY = "Verify corrected 640x640 region"
WINDOW_CAPTURE = "Capture CCD frames"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the effective CCD optical region and interactively save one "
            "registered 640x640 intensity image for each hardware-export input."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_EXPORT_ROOT / "2026-09-01_21-50-28_435.wmv",
        help="CCD video path.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Hardware export containing input/ and manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <export-root>/ccd).",
    )
    parser.add_argument(
        "--selection",
        choices=("quad", "rect"),
        default="quad",
        help="ROI selection: four-corner perspective correction or axis-aligned rectangle.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=640,
        help="Square Light-head input size (default: 640).",
    )
    parser.add_argument(
        "--channel",
        choices=("green", "gray", "red", "blue"),
        default="green",
        help="CCD intensity channel (default: green for the 532 nm source).",
    )
    parser.add_argument(
        "--black-level",
        type=float,
        default=0.0,
        help="Fixed camera black level in 8-bit units (default: 0).",
    )
    parser.add_argument(
        "--white-level",
        type=float,
        default=255.0,
        help="Fixed camera white level in 8-bit units (default: 255).",
    )
    parser.add_argument(
        "--roi-start-time",
        "--start-time",
        dest="roi_start_time",
        type=float,
        default=5.0,
        help="Start ROI calibration after this many seconds (default: 5; --start-time is an alias).",
    )
    parser.add_argument(
        "--roi-samples",
        type=int,
        default=3,
        help="Number of independently selected frames used to estimate a stable ROI (default: 3).",
    )
    parser.add_argument(
        "--roi-sample-gap",
        type=float,
        default=30.0,
        help="Suggested time gap between ROI calibration frames in seconds (default: 30).",
    )
    parser.add_argument(
        "--capture-start-time",
        type=float,
        default=0.0,
        help="Restart video here after ROI confirmation (default: 0).",
    )
    parser.add_argument(
        "--start-id",
        default=None,
        help="Start at a specific exported ID, for example 0021.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only capture this many IDs from the selected start position.",
    )
    parser.add_argument(
        "--reselect-roi",
        action="store_true",
        help="Ignore a saved ROI in ccd_manifest.json and select it again.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow confirmed captures to replace existing CCD PNG files.",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=30.0,
        help="Maximum interactive playback refresh rate (default: 30).",
    )
    parser.add_argument(
        "--auto-pause-seconds",
        type=float,
        default=7.0,
        help=(
            "Pause after this much played video time without another capture; "
            "armed only after the first successful capture of each run (default: 7; 0 disables)."
        ),
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Maximum displayed video width in pixels (default: 1600).",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=980,
        help="Maximum displayed video height in pixels (default: 980).",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.output_size < 1:
        raise ValueError("--output-size must be positive.")
    if args.roi_start_time < 0 or args.capture_start_time < 0:
        raise ValueError("--roi-start-time and --capture-start-time must be non-negative.")
    if args.roi_samples < 2:
        raise ValueError("--roi-samples must be at least 2 for stable ROI estimation.")
    if args.roi_sample_gap < 0:
        raise ValueError("--roi-sample-gap must be non-negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.preview_fps <= 0:
        raise ValueError("--preview-fps must be positive.")
    if args.auto_pause_seconds < 0:
        raise ValueError("--auto-pause-seconds must be non-negative.")
    if args.window_width < 640 or args.window_height < 480:
        raise ValueError("--window-width/--window-height must be at least 640x480.")
    if not 0 <= args.black_level < args.white_level <= 255:
        raise ValueError("Require 0 <= --black-level < --white-level <= 255.")


def load_sample_order(export_root: Path) -> list[dict[str, str]]:
    """Load canonical IDs and input paths, preferring the export manifest."""
    manifest_path = export_root / "manifest.json"
    samples: list[dict[str, str]] = []
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for sample in payload.get("samples", []):
            sample_id = str(sample.get("id", "")).strip()
            relative_input = str(sample.get("input_letterboxed", f"input/{sample_id}.png"))
            if sample_id:
                samples.append({"id": sample_id, "input": relative_input.replace("\\", "/")})
    if samples:
        return samples

    input_dir = export_root / "input"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Hardware input directory not found: {input_dir}")
    files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No numbered input images found in: {input_dir}")
    return [{"id": path.stem, "input": str(path.relative_to(export_root)).replace("\\", "/")} for path in files]


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    # Some high-speed-camera WMV files advertise 1000 FPS while their decoded
    # presentation timestamps are about 60 FPS.  Using the header value makes
    # preview playback skip hundreds of milliseconds at every refresh.  Probe
    # a short sequential prefix and prefer its real timestamp cadence.
    timestamp_samples: list[float] = []
    for _ in range(64):
        ok, _frame = capture.read()
        if not ok:
            break
        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if np.isfinite(timestamp_ms):
            timestamp_samples.append(timestamp_ms)
    timestamp_fps = 0.0
    if len(timestamp_samples) >= 3:
        elapsed_ms = timestamp_samples[-1] - timestamp_samples[0]
        if elapsed_ms > 0:
            timestamp_fps = (len(timestamp_samples) - 1) * 1000.0 / elapsed_ms
    if 1.0 <= timestamp_fps <= 240.0:
        playback_fps = timestamp_fps
        fps_source = "decoded_presentation_timestamps"
    else:
        playback_fps = reported_fps if reported_fps > 0 else 30.0
        fps_source = "container_header"
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    metadata = {
        "fps": float(playback_fps),
        "reported_fps": float(reported_fps),
        "fps_source": fps_source,
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "frame_size_wh": [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ],
    }
    return capture, metadata


def read_frame_at(
    capture: cv2.VideoCapture,
    frame_index: int,
    frame_count: int,
    playback_fps: float,
) -> tuple[int, np.ndarray]:
    """Seek on the presentation timeline, not the WMV's broken frame index."""
    frame_index = max(int(frame_index), 0)
    # CAP_PROP_POS_FRAMES is unreliable for this camera WMV because its header
    # advertises 1000 FPS while decoded timestamps are about 61 FPS. Convert
    # our timeline frame to milliseconds and let FFmpeg seek by timestamp.
    requested_seconds = frame_index / max(float(playback_fps), 1e-9)
    capture.set(cv2.CAP_PROP_POS_MSEC, requested_seconds * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read video frame {frame_index}.")
    actual_seconds = max(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0, 0.0)
    actual_index = max(int(round(actual_seconds * playback_fps)), 0)
    return actual_index, frame


def fit_for_display(frame: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    if abs(scale - 1.0) < 0.001:
        return frame.copy(), 1.0
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    display = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=interpolation)
    return display, scale


def overlay_lines(image: np.ndarray, lines: list[str], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 25
    panel_height = 12 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], min(panel_height, image.shape[0])), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)
    for line_index, line in enumerate(lines):
        cv2.putText(image, line, (10, 23 + line_index * line_height), font, 0.58, color, 1, cv2.LINE_AA)


def seek_delta(
    capture: cv2.VideoCapture,
    current_index: int,
    delta_frames: int,
    frame_count: int,
    playback_fps: float,
) -> tuple[int, np.ndarray]:
    return read_frame_at(capture, current_index + delta_frames, frame_count, playback_fps)


def advance_sequential(
    capture: cv2.VideoCapture,
    current_index: int,
    step_frames: int,
    frame_count: int,
) -> tuple[int, np.ndarray | None, bool]:
    """Advance without random seeking so WMV decoding cannot jump to a keyframe."""
    step_frames = max(int(step_frames), 1)
    if frame_count > 0:
        step_frames = min(step_frames, max(frame_count - 1 - current_index, 0))
    if step_frames <= 0:
        return current_index, None, False
    for _ in range(step_frames - 1):
        if not capture.grab():
            return current_index, None, False
    ok, frame = capture.read()
    if not ok or frame is None:
        return current_index, None, False
    return current_index + step_frames, frame, True


def select_reference_frame(
    capture: cv2.VideoCapture,
    metadata: dict[str, Any],
    start_time: float,
    window_width: int,
    window_height: int,
    sample_number: int,
    sample_count: int,
) -> tuple[int, np.ndarray]:
    """Navigate to a clear frame before selecting the optical-plane ROI."""
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    current_index, frame = read_frame_at(capture, round(start_time * fps), frame_count, fps)
    cv2.namedWindow(WINDOW_ROI, cv2.WINDOW_NORMAL)
    # Clear the point-selection callback left by a previous ROI sample.
    cv2.setMouseCallback(WINDOW_ROI, lambda *_args: None)
    while True:
        display, _ = fit_for_display(frame, window_width, window_height)
        cv2.resizeWindow(WINDOW_ROI, display.shape[1], display.shape[0])
        overlay_lines(
            display,
            [
                f"ROI sample {sample_number}/{sample_count}: choose a clear modulated frame | frame {current_index} | {current_index / fps:.3f} s",
                "A/D: -/+0.1 s   Z/X: -/+1 s   ,/.: previous/next frame",
                "ENTER: use this frame   Q/ESC: quit",
            ],
        )
        cv2.imshow(WINDOW_ROI, display)
        key = cv2.waitKeyEx(0)
        if key in (13, 10):
            return current_index, frame
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt
        delta = 0
        if key in (ord("a"), ord("A")):
            delta = -max(1, round(fps * 0.1))
        elif key in (ord("d"), ord("D")):
            delta = max(1, round(fps * 0.1))
        elif key in (ord("z"), ord("Z")):
            delta = -max(1, round(fps))
        elif key in (ord("x"), ord("X")):
            delta = max(1, round(fps))
        elif key in (ord(","), 2424832):
            delta = -1
        elif key in (ord("."), 2555904):
            delta = 1
        if delta:
            current_index, frame = seek_delta(capture, current_index, delta, frame_count, fps)


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    if len({tuple(point) for point in ordered.tolist()}) != 4:
        raise ValueError("The four ROI corners are ambiguous; please select them again.")
    area = abs(float(cv2.contourArea(ordered)))
    if area < 100.0:
        raise ValueError("Selected ROI is too small.")
    return ordered


def select_quad(frame: np.ndarray, window_width: int, window_height: int) -> np.ndarray:
    points: list[tuple[float, float]] = []
    display_base, display_scale = fit_for_display(frame, window_width, window_height)
    cv2.resizeWindow(WINDOW_ROI, display_base.shape[1], display_base.shape[0])

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / display_scale, y / display_scale))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.setMouseCallback(WINDOW_ROI, on_mouse)
    while True:
        display = display_base.copy()
        display_points = np.asarray(points, dtype=np.float32) * display_scale if points else np.empty((0, 2))
        for index, (x, y) in enumerate(display_points.astype(int)):
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(display, str(index + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        if len(display_points) > 1:
            cv2.polylines(display, [display_points.astype(np.int32)], len(display_points) == 4, (0, 0, 255), 2)
        overlay_lines(
            display,
            [
                "Click the four optical-plane corners (any order).",
                f"Selected: {len(points)}/4   Right-click: undo   R: reset",
                "ENTER: confirm four corners   Q/ESC: quit",
            ],
        )
        cv2.imshow(WINDOW_ROI, display)
        key = cv2.waitKeyEx(20)
        if key in (13, 10) and len(points) == 4:
            return order_quad_points(np.asarray(points, dtype=np.float32))
        if key in (ord("r"), ord("R")):
            points.clear()
        elif key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt


def select_rectangle(frame: np.ndarray, window_width: int, window_height: int) -> np.ndarray:
    display, scale = fit_for_display(frame, window_width, window_height)
    cv2.resizeWindow(WINDOW_ROI, display.shape[1], display.shape[0])
    roi = cv2.selectROI(WINDOW_ROI, display, showCrosshair=True, fromCenter=False)
    x, y, width, height = [float(value) / scale for value in roi]
    if width < 2 or height < 2:
        raise KeyboardInterrupt
    return np.asarray(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float32,
    )


def perspective_matrix(points: np.ndarray, output_size: int) -> np.ndarray:
    destination = np.asarray(
        [[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(np.asarray(points, dtype=np.float32), destination)


def rectify_frame(frame: np.ndarray, matrix: np.ndarray, output_size: int) -> np.ndarray:
    return cv2.warpPerspective(
        frame,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def intensity_image(
    corrected_bgr: np.ndarray,
    channel: str,
    black_level: float,
    white_level: float,
) -> np.ndarray:
    if channel == "blue":
        intensity = corrected_bgr[..., 0]
    elif channel == "green":
        intensity = corrected_bgr[..., 1]
    elif channel == "red":
        intensity = corrected_bgr[..., 2]
    else:
        intensity = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2GRAY)
    intensity_float = intensity.astype(np.float32)
    intensity_float = (intensity_float - black_level) * (255.0 / (white_level - black_level))
    return np.rint(np.clip(intensity_float, 0.0, 255.0)).astype(np.uint8)


def verify_roi(
    frame: np.ndarray,
    points: np.ndarray,
    output_size: int,
    channel: str,
    black_level: float,
    white_level: float,
) -> bool:
    matrix = perspective_matrix(points, output_size)
    corrected = rectify_frame(frame, matrix, output_size)
    preview = intensity_image(corrected, channel, black_level, white_level)
    cv2.namedWindow(WINDOW_VERIFY, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_VERIFY, min(output_size + 320, 1200), min(output_size + 240, 900))
    while True:
        display = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        overlay_lines(display, ["Corrected detector input", "ENTER: accept ROI   R: reselect   Q/ESC: quit"])
        cv2.imshow(WINDOW_VERIFY, display)
        key = cv2.waitKeyEx(0)
        if key in (13, 10):
            cv2.destroyWindow(WINDOW_VERIFY)
            return True
        if key in (ord("r"), ord("R")):
            cv2.destroyWindow(WINDOW_VERIFY)
            return False
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt


def verify_stable_roi(
    frames: list[np.ndarray],
    frame_indices: list[int],
    points: np.ndarray,
    point_std: np.ndarray,
    fps: float,
    args: argparse.Namespace,
) -> bool:
    """Show the median ROI on every calibration frame before final acceptance."""
    corrected_images = []
    matrix = perspective_matrix(points, args.output_size)
    for frame, frame_index in zip(frames, frame_indices):
        corrected = rectify_frame(frame, matrix, args.output_size)
        intensity = intensity_image(
            corrected, args.channel, args.black_level, args.white_level
        )
        image = cv2.cvtColor(intensity, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            image,
            f"frame {frame_index} | {frame_index / fps:.3f} s",
            (12, args.output_size - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        corrected_images.append(image)

    columns = min(3, len(corrected_images))
    rows = (len(corrected_images) + columns - 1) // columns
    tile_size = min(args.output_size, max(240, (args.window_width - 20 * (columns + 1)) // columns))
    header_height = 86
    canvas = np.zeros(
        (header_height + rows * (tile_size + 12) + 12, columns * (tile_size + 12) + 12, 3),
        dtype=np.uint8,
    )
    for index, image in enumerate(corrected_images):
        row, column = divmod(index, columns)
        tile = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        x = 12 + column * (tile_size + 12)
        y = header_height + row * (tile_size + 12)
        canvas[y : y + tile_size, x : x + tile_size] = tile

    max_std = float(point_std.max())
    overlay_lines(
        canvas,
        [
            f"Stable ROI = coordinate-wise median of {len(frames)} selections | max corner std {max_std:.2f} px",
            "Inspect all corrected frames. ENTER: accept and restart video at capture start   R: redo all ROI samples",
            "Q/ESC: quit",
        ],
    )
    display, _ = fit_for_display(canvas, args.window_width, args.window_height)
    cv2.namedWindow(WINDOW_VERIFY, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_VERIFY, display.shape[1], display.shape[0])
    while True:
        cv2.imshow(WINDOW_VERIFY, display)
        key = cv2.waitKeyEx(0)
        if key in (13, 10):
            cv2.destroyWindow(WINDOW_VERIFY)
            return True
        if key in (ord("r"), ord("R")):
            cv2.destroyWindow(WINDOW_VERIFY)
            return False
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt


def collect_stable_roi(
    capture: cv2.VideoCapture,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Select several independent ROIs and robustly combine their corners."""
    fps = float(metadata["fps"])
    while True:
        selected_points: list[np.ndarray] = []
        selected_frames: list[np.ndarray] = []
        selected_indices: list[int] = []
        suggested_time = args.roi_start_time
        for sample_index in range(args.roi_samples):
            frame_index, frame = select_reference_frame(
                capture,
                metadata,
                suggested_time,
                args.window_width,
                args.window_height,
                sample_index + 1,
                args.roi_samples,
            )
            while True:
                points = (
                    select_quad(frame, args.window_width, args.window_height)
                    if args.selection == "quad"
                    else select_rectangle(frame, args.window_width, args.window_height)
                )
                if verify_roi(
                    frame,
                    points,
                    args.output_size,
                    args.channel,
                    args.black_level,
                    args.white_level,
                ):
                    break
            selected_points.append(order_quad_points(points))
            selected_frames.append(frame.copy())
            selected_indices.append(frame_index)
            suggested_time = frame_index / fps + args.roi_sample_gap

        point_stack = np.stack(selected_points, axis=0)
        stable_points = order_quad_points(np.median(point_stack, axis=0).astype(np.float32))
        point_std = point_stack.std(axis=0)
        if verify_stable_roi(
            selected_frames,
            selected_indices,
            stable_points,
            point_std,
            fps,
            args,
        ):
            selections = [
                {
                    "video_frame_index": int(frame_index),
                    "video_time_seconds": float(frame_index / fps),
                    "roi_points_xy": points.astype(float).tolist(),
                }
                for frame_index, points in zip(selected_indices, selected_points)
            ]
            return stable_points, selections, float(point_std.max())


def load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid CCD manifest: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_png_atomic(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    os.replace(temporary, path)


def first_pending_position(samples: list[dict[str, str]], output_dir: Path) -> int:
    for index, sample in enumerate(samples):
        if not (output_dir / f"{sample['id']}.png").is_file():
            return index
    return len(samples)


def resume_capture_entry(
    samples: list[dict[str, str]],
    start_position: int,
    previous_manifest: dict[str, Any] | None,
    recapture_current: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Find a saved video position close to the requested hardware ID."""
    entries_by_id = {
        str(entry["id"]): entry
        for entry in (previous_manifest or {}).get("captures", [])
        if isinstance(entry, dict) and "id" in entry
    }
    if recapture_current:
        current_entry = entries_by_id.get(samples[start_position]["id"])
        if current_entry is not None:
            return current_entry, "current"
    for position in range(start_position - 1, -1, -1):
        previous_entry = entries_by_id.get(samples[position]["id"])
        if previous_entry is not None:
            return previous_entry, "previous"
    return None, None


def draw_capture_display(
    frame: np.ndarray,
    points: np.ndarray,
    sample: dict[str, str],
    sample_position: int,
    sample_count: int,
    frame_index: int,
    fps: float,
    playing: bool,
    auto_pause_status: str,
    last_message: str,
    window_width: int,
    window_height: int,
) -> tuple[np.ndarray, float]:
    display, scale = fit_for_display(frame, window_width, window_height)
    polygon = np.rint(points * scale).astype(np.int32)
    cv2.polylines(display, [polygon], True, (0, 0, 255), 2, cv2.LINE_AA)
    status = "PLAY" if playing else "PAUSE"
    lines = [
        f"Next: {sample['id']} ({sample_position + 1}/{sample_count}) | frame {frame_index} | {frame_index / fps:.3f} s | {status}",
        "Left-click / ENTER / C: capture   SPACE/P: play-pause",
        "A/D: -/+0.1 s   Z/X: -/+1 s   ,/.: previous/next frame   Q/ESC: save and quit",
        auto_pause_status,
    ]
    if last_message:
        lines.append(last_message)
    overlay_lines(display, lines, color=(255, 255, 255))
    return display, scale


def capture_samples(
    capture: cv2.VideoCapture,
    video_path: Path,
    export_root: Path,
    output_dir: Path,
    samples: list[dict[str, str]],
    start_position: int,
    frame_index: int,
    frame: np.ndarray,
    points: np.ndarray,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    manifest_path: Path,
    previous_manifest: dict[str, Any] | None,
    roi_selections: list[dict[str, Any]],
    roi_max_corner_std_px: float,
) -> int:
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    matrix = perspective_matrix(points, args.output_size)
    entries_by_id = {
        str(entry["id"]): entry
        for entry in (previous_manifest or {}).get("captures", [])
        if isinstance(entry, dict) and "id" in entry
    }
    manifest = {
        "video": str(video_path),
        **metadata,
        "export_root": str(export_root),
        "output_directory": str(output_dir),
        "output_size_hw": [args.output_size, args.output_size],
        "selection": args.selection,
        "roi_points_xy": points.astype(float).tolist(),
        "roi_estimator": "coordinate-wise median",
        "roi_samples": roi_selections,
        "roi_max_corner_std_px": roi_max_corner_std_px,
        "roi_start_time_seconds": args.roi_start_time,
        "roi_sample_gap_seconds": args.roi_sample_gap,
        "capture_start_time_seconds": args.capture_start_time,
        "auto_pause_seconds": args.auto_pause_seconds,
        "channel": args.channel,
        "black_level": args.black_level,
        "white_level": args.white_level,
        "preprocessing": (
            "CCD ROI perspective-warped directly to the full detector feature grid; "
            "fixed black/white calibration only; no per-frame normalization or letterbox padding."
        ),
        "captures": list(entries_by_id.values()),
    }

    current_position = start_position
    captured_this_run = 0
    playing = False
    capture_requested = False
    last_message = ""
    auto_pause_armed = False
    auto_pause_anchor_frame: int | None = None
    auto_pause_triggered = False

    def on_mouse(event: int, _x: int, _y: int, _flags: int, _param: Any) -> None:
        nonlocal capture_requested
        if event == cv2.EVENT_LBUTTONDOWN:
            capture_requested = True

    cv2.namedWindow(WINDOW_CAPTURE, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_CAPTURE, on_mouse)

    while current_position < len(samples):
        if args.limit is not None and captured_this_run >= args.limit:
            break
        sample = samples[current_position]
        if (
            playing
            and auto_pause_armed
            and auto_pause_anchor_frame is not None
            and args.auto_pause_seconds > 0
            and (frame_index - auto_pause_anchor_frame) / fps >= args.auto_pause_seconds
        ):
            playing = False
            auto_pause_triggered = True
            last_message = (
                f"Auto-paused: no image saved for {args.auto_pause_seconds:g} s of video. "
                "SPACE resumes; capture also resumes playback."
            )

        if args.auto_pause_seconds <= 0:
            auto_pause_status = "Auto-pause: disabled"
        elif not auto_pause_armed:
            auto_pause_status = (
                f"Auto-pause: starts after this run's first save ({args.auto_pause_seconds:g} s timeout)"
            )
        elif auto_pause_triggered:
            auto_pause_status = (
                f"Auto-pause: TRIGGERED after {args.auto_pause_seconds:g} s | SPACE or capture to continue"
            )
        elif playing and auto_pause_anchor_frame is not None:
            elapsed = max(0.0, (frame_index - auto_pause_anchor_frame) / fps)
            remaining = max(0.0, args.auto_pause_seconds - elapsed)
            auto_pause_status = f"Auto-pause: armed | {remaining:.1f} s remaining"
        else:
            auto_pause_status = "Auto-pause: armed | timer paused"
        display, _ = draw_capture_display(
            frame,
            points,
            sample,
            current_position,
            len(samples),
            frame_index,
            fps,
            playing,
            auto_pause_status,
            last_message,
            args.window_width,
            args.window_height,
        )
        cv2.resizeWindow(WINDOW_CAPTURE, display.shape[1], display.shape[0])
        cv2.imshow(WINDOW_CAPTURE, display)
        wait_ms = max(1, round(1000.0 / args.preview_fps)) if playing else 20
        key = cv2.waitKeyEx(wait_ms)

        if capture_requested or key in (13, 10, ord("c"), ord("C")):
            capture_requested = False
            output_path = output_dir / f"{sample['id']}.png"
            if output_path.exists() and not args.overwrite:
                last_message = f"Not overwritten: {output_path.name} (use --overwrite or resume at first missing ID)"
                continue
            corrected = rectify_frame(frame, matrix, args.output_size)
            intensity = intensity_image(corrected, args.channel, args.black_level, args.white_level)
            save_png_atomic(output_path, intensity)
            entries_by_id[sample["id"]] = {
                "id": sample["id"],
                "input": sample["input"],
                "ccd_image": output_path.name,
                "video_frame_index": int(frame_index),
                "video_time_seconds": float(frame_index / fps),
            }
            manifest["captures"] = [
                entries_by_id[item["id"]]
                for item in samples
                if item["id"] in entries_by_id
            ]
            write_json_atomic(manifest_path, manifest)
            captured_this_run += 1
            current_position += 1
            # A confirmed capture is a marker within one continuous playback,
            # not a pause/seek operation. Continue from the saved frame; SPACE
            # remains the only play/pause toggle.
            playing = True
            auto_pause_armed = args.auto_pause_seconds > 0
            auto_pause_anchor_frame = frame_index if auto_pause_armed else None
            auto_pause_triggered = False
            last_message = f"Saved {output_path.name}; continuing from {frame_index / fps:.3f} s"
            if current_position >= len(samples):
                break
            continue

        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("p"), ord("P"), 32):
            playing = not playing
            if playing and auto_pause_armed:
                # A manual resume gets a fresh timeout window instead of
                # immediately pausing again at the same video position.
                auto_pause_anchor_frame = frame_index
            auto_pause_triggered = False
            continue

        delta = 0
        if key in (ord("a"), ord("A")):
            delta = -max(1, round(fps * 0.1))
        elif key in (ord("d"), ord("D")):
            delta = max(1, round(fps * 0.1))
        elif key in (ord("z"), ord("Z")):
            delta = -max(1, round(fps))
        elif key in (ord("x"), ord("X")):
            delta = max(1, round(fps))
        elif key in (ord(","), 2424832):
            delta = -1
        elif key in (ord("."), 2555904):
            delta = 1

        if delta:
            frame_index, frame = seek_delta(capture, frame_index, delta, frame_count, fps)
            if auto_pause_armed:
                auto_pause_anchor_frame = frame_index
                auto_pause_triggered = False
        elif playing:
            next_index, next_frame, ok = advance_sequential(
                capture,
                frame_index,
                max(1, round(fps / args.preview_fps)),
                frame_count,
            )
            if ok:
                frame_index, frame = next_index, next_frame
            else:
                playing = False
                last_message = "Reached the end of the video; playback paused."

    manifest["captures"] = [entries_by_id[item["id"]] for item in samples if item["id"] in entries_by_id]
    write_json_atomic(manifest_path, manifest)
    return captured_this_run


def main() -> None:
    args = parse_args()
    validate_args(args)
    video_path = resolve_path(args.video)
    export_root = resolve_path(args.export_root)
    output_dir = resolve_path(args.output) if args.output is not None else export_root / "ccd"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not export_root.is_dir():
        raise FileNotFoundError(f"Hardware export directory not found: {export_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_sample_order(export_root)
    manifest_path = output_dir / "ccd_manifest.json"
    existing_manifest = load_existing_manifest(manifest_path)

    if args.start_id is not None:
        start_position = next(
            (index for index, sample in enumerate(samples) if sample["id"] == args.start_id),
            -1,
        )
        if start_position < 0:
            raise ValueError(f"--start-id {args.start_id!r} is not present in the hardware export.")
    else:
        start_position = first_pending_position(samples, output_dir)
    if start_position >= len(samples):
        print(f"All {len(samples)} CCD images already exist in: {output_dir}")
        print("Use --start-id together with --overwrite to recapture selected IDs.")
        return

    capture, metadata = open_video(video_path)
    try:
        saved_points = (existing_manifest or {}).get("roi_points_xy")
        saved_selections = (existing_manifest or {}).get("roi_samples", [])
        saved_roi_gap = float((existing_manifest or {}).get("roi_sample_gap_seconds", 0.0))
        saved_roi_is_stable = (
            saved_points is not None
            and isinstance(saved_selections, list)
            and len(saved_selections) >= args.roi_samples
            and saved_roi_gap >= args.roi_sample_gap
        )
        if saved_roi_is_stable and not args.reselect_roi:
            reference_index, reference_frame = select_reference_frame(
                capture,
                metadata,
                args.roi_start_time,
                args.window_width,
                args.window_height,
                1,
                1,
            )
            points = order_quad_points(np.asarray(saved_points, dtype=np.float32))
            if not verify_roi(
                reference_frame,
                points,
                args.output_size,
                args.channel,
                args.black_level,
                args.white_level,
            ):
                saved_roi_is_stable = False
            else:
                roi_selections = saved_selections
                roi_max_corner_std_px = float(
                    (existing_manifest or {}).get("roi_max_corner_std_px", 0.0)
                )
                print(
                    f"Reused stable ROI from {len(roi_selections)} saved calibration samples "
                    f"(verified at frame {reference_index})."
                )
        if not saved_roi_is_stable or args.reselect_roi:
            points, roi_selections, roi_max_corner_std_px = collect_stable_roi(
                capture, metadata, args
            )
            print(
                f"Stable ROI accepted from {len(roi_selections)} samples; "
                f"maximum corner standard deviation={roi_max_corner_std_px:.2f} px."
            )
        cv2.destroyWindow(WINDOW_ROI)

        # Resume the video together with the first pending numbered ID.  When
        # recapturing an existing ID, seek to its own saved frame; otherwise
        # resume at the nearest preceding saved capture so no transition is lost.
        resume_entry, resume_kind = resume_capture_entry(
            samples,
            start_position,
            existing_manifest,
            recapture_current=args.start_id is not None and args.overwrite,
        )
        if resume_entry is not None and "video_time_seconds" in resume_entry:
            playback_start_seconds = float(resume_entry["video_time_seconds"])
        elif resume_entry is not None and "video_frame_index" in resume_entry:
            playback_start_seconds = float(resume_entry["video_frame_index"]) / float(metadata["fps"])
        else:
            playback_start_seconds = args.capture_start_time
        frame_index, frame = read_frame_at(
            capture,
            round(playback_start_seconds * float(metadata["fps"])),
            int(metadata["frame_count"]),
            float(metadata["fps"]),
        )
        if resume_entry is None:
            print(
                f"Capture starts at configured time {frame_index / float(metadata['fps']):.3f} s "
                f"for ID {samples[start_position]['id']}."
            )
        else:
            relation = (
                "its saved position"
                if resume_kind == "current"
                else f"saved ID {resume_entry['id']}"
            )
            print(
                f"Resuming ID {samples[start_position]['id']} from {relation} at "
                f"{frame_index / float(metadata['fps']):.3f} s."
            )

        captured = capture_samples(
            capture=capture,
            video_path=video_path,
            export_root=export_root,
            output_dir=output_dir,
            samples=samples,
            start_position=start_position,
            frame_index=frame_index,
            frame=frame,
            points=points,
            metadata=metadata,
            args=args,
            manifest_path=manifest_path,
            previous_manifest=existing_manifest,
            roi_selections=roi_selections,
            roi_max_corner_std_px=roi_max_corner_std_px,
        )
        print(f"Captured {captured} image(s) in this run.")
        print(f"CCD images: {output_dir}")
        print(f"Registration manifest: {manifest_path}")
        next_position = first_pending_position(samples, output_dir)
        if next_position < len(samples):
            print(f"Next pending ID: {samples[next_position]['id']}")
        else:
            print(f"Completed all {len(samples)} hardware-export IDs.")
    except KeyboardInterrupt:
        print("Cancelled; existing captured images were kept.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
