#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DVP2 industrial-camera preview, capture, and recording utility.

This tool calls the official DVP2 DLL through ctypes.  It intentionally does
not import the SDK's ``dvp.pyd`` because the bundled Python binding only
supports Python 3.6, while this project uses a newer 64-bit Python runtime.

Preview controls
----------------
Space / S : save the current full-resolution frame as PNG
R         : start or stop AVI recording
A         : toggle continuous auto exposure
[ / ]     : reduce or increase manual exposure by 10 percent
Esc / Q   : exit
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ..devices.dvp2_camera.sdk import (
    AE_OP_OFF,
    AE_OP_CONTINUOUS,
    WINDOW_NAME,
    DvpError,
    DvpFrame,
    default_dll_path,
    check_status,
    DvpApi,
    CameraSession,
)
from ..devices.dvp2_camera.controller import (
    print_devices,
    frame_to_image,
    image_for_preview,
    next_frame_index,
    save_frame,
    VideoRecorder,
    timestamped_video_path,
    get_exposure,
    get_ae_operation,
    configure_camera,
    print_camera_parameters,
)


def render_preview(
    image: np.ndarray,
    frame: DvpFrame,
    recorder: VideoRecorder,
    window_scale: float,
) -> np.ndarray:
    preview = image_for_preview(image).copy()
    state = "REC" if recorder.active else "preview"
    text = (
        f"{state} | frame={frame.uFrameID} | {frame.iWidth}x{frame.iHeight} | "
        f"exp={frame.fExposure:.1f} us | gain={frame.fAGain:.2f}"
    )
    cv2.putText(
        preview,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255) if recorder.active else (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    if not np.isclose(window_scale, 1.0):
        preview = cv2.resize(
            preview,
            None,
            fx=window_scale,
            fy=window_scale,
            interpolation=cv2.INTER_AREA if window_scale < 1.0 else cv2.INTER_NEAREST,
        )
    return preview


def run_camera(api: DvpApi, args: argparse.Namespace) -> None:
    devices = api.list_devices()
    print_devices(devices)
    if not devices:
        raise RuntimeError(
            "No DVP2 camera was found. Check camera power/cable and the DVP2 driver."
        )
    if args.camera < 0 or args.camera >= len(devices):
        raise ValueError(f"--camera must be in [0, {len(devices) - 1}]")

    output_dir = args.output.expanduser().resolve()
    frame_index = next_frame_index(output_dir)
    recorder = VideoRecorder(args.fps, args.codec)
    started_at = time.monotonic()
    next_auto_capture = started_at
    auto_captured = 0
    preview_window_sized = False

    with CameraSession(api, args.camera) as session:
        handle = session.handle.value
        configure_camera(api, handle, args)
        print_camera_parameters(api, handle)
        session.start()
        print("Camera stream started.")
        if args.preview:
            print("Keys: Space/S=save, R=record, A=auto exposure, [ / ]=exposure, Esc/Q=exit")
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while True:
                try:
                    frame, raw = session.get_frame(args.timeout_ms)
                except TimeoutError:
                    print("Frame timeout; retrying...", file=sys.stderr)
                    continue
                image = frame_to_image(frame, raw)
                now = time.monotonic()

                if args.capture > 0 and auto_captured < args.capture and now >= next_auto_capture:
                    save_frame(image, output_dir, frame_index)
                    frame_index += 1
                    auto_captured += 1
                    next_auto_capture = now + args.capture_interval

                if args.record is not None and not recorder.active:
                    video_path = (
                        timestamped_video_path(output_dir)
                        if args.record.name == "__AUTO__"
                        else args.record
                    )
                    recorder.start(video_path, image)
                    args.record = None
                recorder.write(image)

                key = -1
                if args.preview:
                    preview = render_preview(image, frame, recorder, args.window_scale)
                    if not preview_window_sized:
                        preview_height, preview_width = preview.shape[:2]
                        cv2.resizeWindow(WINDOW_NAME, preview_width, preview_height)
                        preview_window_sized = True
                    cv2.imshow(WINDOW_NAME, preview)
                    key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord(" "), ord("s"), ord("S")):
                    save_frame(image, output_dir, frame_index)
                    frame_index += 1
                elif key in (ord("r"), ord("R")):
                    if recorder.active:
                        recorder.stop()
                    else:
                        recorder.start(timestamped_video_path(output_dir), image)
                elif key in (ord("a"), ord("A")):
                    current = get_ae_operation(api, handle)
                    new_value = AE_OP_OFF if current == AE_OP_CONTINUOUS else AE_OP_CONTINUOUS
                    check_status(api.dll.dvpSetAeOperation(handle, new_value), "toggle AE")
                    print(f"Auto exposure: {'continuous' if new_value else 'off'}")
                elif key in (ord("["), ord("]")):
                    current = get_exposure(api, handle)
                    factor = 0.9 if key == ord("[") else 1.1
                    check_status(api.dll.dvpSetAeOperation(handle, AE_OP_OFF), "disable AE")
                    check_status(
                        api.dll.dvpSetExposure(handle, current * factor),
                        "adjust exposure",
                    )
                    print(f"Exposure: {get_exposure(api, handle):.3f} us")

                elapsed = now - started_at
                if args.duration is not None and elapsed >= args.duration:
                    break
                capture_complete = args.capture > 0 and auto_captured >= args.capture
                if capture_complete and args.duration is None:
                    break
        finally:
            recorder.stop()
            if args.preview:
                cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control a Do3think DVP2 camera using the bundled 64-bit SDK DLL."
    )
    parser.add_argument("--list", action="store_true", help="List cameras and exit.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--dll", type=Path, default=default_dll_path(), help="Path to DVPCamera64.dll."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/dvp2_camera"),
        help="Directory for captured PNG images and default AVI files.",
    )
    parser.add_argument(
        "--target-format",
        choices=("native", "bgr8", "mono8"),
        default="native",
        help="SDK output format; native keeps the current camera setting.",
    )
    parser.add_argument(
        "--exposure-us", type=float, help="Manual exposure time in microseconds."
    )
    parser.add_argument("--gain", type=float, help="Manual analog gain.")
    exposure_group = parser.add_mutually_exclusive_group()
    exposure_group.add_argument(
        "--auto-exposure", action="store_true", help="Enable continuous auto exposure."
    )
    exposure_group.add_argument(
        "--manual-exposure", action="store_true", help="Disable auto exposure."
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Camera hardware ROI. Values must satisfy the camera's alignment limits.",
    )
    parser.add_argument(
        "--dialog", action="store_true", help="Open the SDK property dialog before capture."
    )
    parser.add_argument("--load-config", type=Path, help="Load an SDK .ini configuration.")
    parser.add_argument("--save-config", type=Path, help="Save the resulting SDK config.")
    parser.add_argument(
        "--capture",
        type=int,
        default=0,
        help="Automatically save this many PNG frames; 0 means keyboard-only capture.",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=0.0,
        help="Seconds between automatic PNG captures (default: 0).",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const=Path("__AUTO__"),
        type=Path,
        help="Start AVI recording, optionally to the specified path.",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="AVI frame rate.")
    parser.add_argument(
        "--codec", default="MJPG", help="Four-character OpenCV video codec (default: MJPG)."
    )
    parser.add_argument(
        "--duration", type=float, help="Stop automatically after this many seconds."
    )
    parser.add_argument(
        "--timeout-ms", type=int, default=4000, help="Frame timeout in milliseconds."
    )
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Run without the OpenCV preview window.",
    )
    parser.set_defaults(preview=True)
    parser.add_argument(
        "--window-scale",
        type=float,
        default=0.75,
        help=(
            "Initial preview scale relative to the camera frame (default: 0.75); "
            "saved images remain at native resolution."
        ),
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.capture < 0:
        parser.error("--capture must be >= 0")
    if args.capture_interval < 0:
        parser.error("--capture-interval must be >= 0")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be > 0")
    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if len(args.codec) != 4:
        parser.error("--codec must contain exactly four characters")
    if args.window_scale <= 0:
        parser.error("--window-scale must be > 0")
    if args.exposure_us is not None and args.exposure_us <= 0:
        parser.error("--exposure-us must be > 0")
    if args.gain is not None and args.gain <= 0:
        parser.error("--gain must be > 0")
    if args.auto_exposure and (args.exposure_us is not None or args.gain is not None):
        parser.error("--auto-exposure cannot be combined with --exposure-us or --gain")
    if args.roi is not None and any(value < 0 for value in args.roi[:2]):
        parser.error("ROI X and Y must be >= 0")
    if args.roi is not None and any(value <= 0 for value in args.roi[2:]):
        parser.error("ROI W and H must be > 0")
    if not args.preview and args.capture == 0 and args.record is None:
        parser.error("--no-preview requires --capture or --record")
    if not args.preview and args.record is not None and args.duration is None and args.capture == 0:
        parser.error("headless recording requires --duration (or --capture)")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        api = DvpApi(args.dll)
        if args.list:
            print_devices(api.list_devices())
            return 0
        run_camera(api, args)
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except (DvpError, FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
