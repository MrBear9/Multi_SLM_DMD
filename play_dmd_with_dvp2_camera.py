#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synchronize native-pixel DMD playback with a DVP2 camera.

Each DMD image is displayed for two seconds by default.  One second after the
image becomes visible, the latest continuously acquired camera frame is saved
with the same filename stem.  If no DVP2 camera is available, DMD playback
continues normally and image capture is disabled.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import cv2
import tkinter as tk
from PIL import Image, ImageTk

import control_dvp2_camera as dvp2
import play_dmd_input_fullscreen as dmd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = dmd.DEFAULT_INPUT_DIR
MANIFEST_NAME = "dvp2_capture_manifest.json"


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def collect_images(
    input_path: Path,
    start_id: Optional[str],
    limit: Optional[int],
) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in dmd.IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported DMD image format: {input_path}")
        if start_id is not None:
            raise ValueError("--start-id can be used only when --input is a directory.")
        paths = [input_path]
    elif input_path.is_dir():
        paths = dmd.load_image_paths(input_path)
        if start_id is not None:
            start_index = next(
                (index for index, path in enumerate(paths) if path.stem == start_id),
                -1,
            )
            if start_index < 0:
                raise ValueError(f"--start-id {start_id!r} was not found in {input_path}")
            paths = paths[start_index:]
    else:
        raise FileNotFoundError(f"DMD input image or directory not found: {input_path}")

    if limit is not None:
        paths = paths[:limit]
    stems = [path.stem.casefold() for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("DMD inputs contain duplicate filename stems; capture names would collide.")
    return paths


def validate_dmd_images(paths: Sequence[Path], monitor: dmd.Monitor) -> None:
    for path in paths:
        with Image.open(path) as image:
            image.verify()
            if image.width > monitor.width or image.height > monitor.height:
                raise ValueError(
                    f"{path.name} is {image.width}x{image.height}, larger than monitor "
                    f"{monitor.width}x{monitor.height}; resizing is intentionally disabled."
                )


def default_capture_dir(input_path: Path) -> Path:
    container = input_path if input_path.is_dir() else input_path.parent
    return container.parent / "ccd"


@dataclass
class LatestFrame:
    image: object
    frame_id: int
    received_monotonic: float
    exposure_us: float
    gain: float


class Dvp2FrameWorker:
    """Continuously drain the SDK stream so captures cannot use stale frames."""

    def __init__(
        self,
        api: dvp2.DvpApi,
        camera_index: int,
        target_format: str,
        auto_exposure: bool,
        manual_exposure: bool,
        exposure_us: Optional[float],
        gain: Optional[float],
        timeout_ms: int,
    ) -> None:
        self.api = api
        self.camera_index = camera_index
        self.target_format = target_format
        self.auto_exposure = auto_exposure
        self.manual_exposure = manual_exposure
        self.exposure_us = exposure_us
        self.gain = gain
        self.timeout_ms = timeout_ms
        self.session = dvp2.CameraSession(api, camera_index)
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.latest: Optional[LatestFrame] = None
        self.error: Optional[BaseException] = None
        self.thread: Optional[threading.Thread] = None

    def start(self, ready_timeout: float) -> None:
        self.session.open()
        handle = self.session.handle.value
        dvp2.check_status(
            self.api.dll.dvpSetTriggerState(handle, False),
            "disable DVP2 trigger mode",
        )
        target_formats = {"bgr8": dvp2.S_BGR24, "mono8": dvp2.S_MONO8}
        if self.target_format in target_formats:
            dvp2.check_status(
                self.api.dll.dvpSetTargetFormat(handle, target_formats[self.target_format]),
                f"set DVP2 target format {self.target_format}",
            )

        if self.auto_exposure:
            dvp2.check_status(
                self.api.dll.dvpSetAeOperation(handle, dvp2.AE_OP_CONTINUOUS),
                "enable DVP2 auto exposure",
            )
        elif self.manual_exposure or self.exposure_us is not None or self.gain is not None:
            dvp2.check_status(
                self.api.dll.dvpSetAeOperation(handle, dvp2.AE_OP_OFF),
                "disable DVP2 auto exposure",
            )
        if self.exposure_us is not None:
            dvp2.check_status(
                self.api.dll.dvpSetExposure(handle, self.exposure_us),
                "set DVP2 exposure",
            )
        if self.gain is not None:
            dvp2.check_status(
                self.api.dll.dvpSetAnalogGain(handle, self.gain),
                "set DVP2 analog gain",
            )

        self.session.start()
        self.thread = threading.Thread(
            target=self._run,
            name="DVP2-frame-worker",
            daemon=True,
        )
        self.thread.start()
        deadline = time.monotonic() + ready_timeout
        with self.condition:
            while self.latest is None and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
        if self.error is not None:
            raise RuntimeError(f"DVP2 acquisition failed: {self.error}") from self.error
        if self.latest is None:
            raise TimeoutError(f"No DVP2 frame received within {ready_timeout:g} seconds")

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    frame, raw = self.session.get_frame(self.timeout_ms)
                except TimeoutError:
                    continue
                image = dvp2.frame_to_image(frame, raw)
                latest = LatestFrame(
                    image=image,
                    frame_id=int(frame.uFrameID),
                    received_monotonic=time.monotonic(),
                    exposure_us=float(frame.fExposure),
                    gain=float(frame.fAGain),
                )
                with self.condition:
                    self.latest = latest
                    self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def snapshot_after(self, minimum_time: float, timeout: float) -> Optional[LatestFrame]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while (
                self.error is None
                and (self.latest is None or self.latest.received_monotonic < minimum_time)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            if self.error is not None:
                return None
            if self.latest is None or self.latest.received_monotonic < minimum_time:
                return None
            item = self.latest
            return LatestFrame(
                image=item.image.copy(),
                frame_id=item.frame_id,
                received_monotonic=item.received_monotonic,
                exposure_us=item.exposure_us,
                gain=item.gain,
            )

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.timeout_ms / 1000.0 + 1.0))
        self.session.close()


class SynchronizedDmdPlayer:
    def __init__(
        self,
        root: tk.Tk,
        monitor: dmd.Monitor,
        paths: list[Path],
        display_seconds: float,
        settle_seconds: float,
        camera: Optional[Dvp2FrameWorker],
        capture_dir: Optional[Path],
        overwrite: bool,
        manifest: dict,
        show_cursor: bool,
    ) -> None:
        self.root = root
        self.monitor = monitor
        self.paths = paths
        self.display_ms = round(display_seconds * 1000.0)
        self.settle_ms = round(settle_seconds * 1000.0)
        self.camera = camera
        self.capture_dir = capture_dir
        self.overwrite = overwrite
        self.manifest = manifest
        self.index = 0
        self.playing = False
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.capture_after_id: Optional[str] = None
        self.advance_after_id: Optional[str] = None
        self.period_token = 0
        self.period_started = 0.0
        self.captured_indices: set[int] = set()
        self.camera_warning_printed = False

        root.configure(background="black", cursor="" if show_cursor else "none")
        root.overrideredirect(True)
        root.geometry(f"{monitor.width}x{monitor.height}{monitor.left:+d}{monitor.top:+d}")
        root.attributes("-topmost", True)
        self.canvas = tk.Canvas(
            root,
            width=monitor.width,
            height=monitor.height,
            background="black",
            borderwidth=0,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        root.bind("<space>", self.toggle_playback)
        root.bind("<Escape>", self.close)
        root.bind("q", self.close)
        root.bind("Q", self.close)
        root.bind("<Right>", self.next_image)
        root.bind("<Left>", self.previous_image)
        root.bind("<Home>", self.first_image)

        self.show_current()
        root.after(100, self.focus_window)

    def focus_window(self) -> None:
        self.root.lift()
        self.root.focus_force()

    def cancel_timers(self) -> None:
        for timer_name in ("capture_after_id", "advance_after_id"):
            timer_id = getattr(self, timer_name)
            if timer_id is not None:
                self.root.after_cancel(timer_id)
                setattr(self, timer_name, None)

    def show_current(self) -> None:
        path = self.paths[self.index]
        with Image.open(path) as source:
            source.load()
            image = source.copy()
        self.photo = ImageTk.PhotoImage(image=image, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.monitor.width // 2,
            self.monitor.height // 2,
            image=self.photo,
            anchor="center",
        )
        state = "PLAY" if self.playing else "PAUSE"
        camera_state = "capture" if self.camera is not None else "DMD-only"
        self.root.title(
            f"DMD + DVP2 | {path.name} | {self.index + 1}/{len(self.paths)} | "
            f"{state} | {camera_state}"
        )
        print(
            f"[{self.index + 1}/{len(self.paths)}] DMD={path.name} | "
            f"{image.width}x{image.height} native pixels | {state} | {camera_state}",
            flush=True,
        )

    def begin_period(self) -> None:
        self.cancel_timers()
        self.period_token += 1
        token = self.period_token
        self.show_current()
        self.root.update_idletasks()
        self.period_started = time.monotonic()
        if self.camera is not None and self.index not in self.captured_indices:
            self.capture_after_id = self.root.after(
                self.settle_ms,
                lambda: self.capture_current(token),
            )
        self.advance_after_id = self.root.after(
            self.display_ms,
            lambda: self.advance(token),
        )

    def capture_current(self, token: int) -> None:
        self.capture_after_id = None
        if not self.playing or token != self.period_token or self.camera is None:
            return
        path = self.paths[self.index]
        minimum_time = self.period_started + self.settle_ms / 1000.0 * 0.95
        snapshot = self.camera.snapshot_after(minimum_time, timeout=0.25)
        if snapshot is None:
            reason = self.camera.error or "no sufficiently recent frame"
            print(f"WARNING: capture skipped for {path.name}: {reason}", flush=True)
            if self.camera.error is not None and not self.camera_warning_printed:
                print("DVP2 capture is unavailable; DMD playback will continue.", flush=True)
                self.camera_warning_printed = True
            return

        assert self.capture_dir is not None
        output_path = self.capture_dir / f"{path.stem}.png"
        if output_path.exists() and not self.overwrite:
            print(f"WARNING: capture exists and was not overwritten: {output_path}", flush=True)
            self.captured_indices.add(self.index)
            return
        if not cv2.imwrite(str(output_path), snapshot.image):
            print(f"WARNING: failed to save camera frame: {output_path}", flush=True)
            return

        self.captured_indices.add(self.index)
        capture_delay = snapshot.received_monotonic - self.period_started
        self.manifest["captures"].append(
            {
                "dmd_image": str(path),
                "camera_image": str(output_path),
                "dmd_name": path.name,
                "camera_name": output_path.name,
                "camera_frame_id": snapshot.frame_id,
                "capture_delay_seconds": capture_delay,
                "exposure_us": snapshot.exposure_us,
                "analog_gain": snapshot.gain,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        manifest_path = self.capture_dir / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Captured: {path.name} -> {output_path.name} "
            f"(after {capture_delay:.3f} s, camera frame {snapshot.frame_id})",
            flush=True,
        )

    def advance(self, token: int) -> None:
        self.advance_after_id = None
        if not self.playing or token != self.period_token:
            return
        if self.index + 1 >= len(self.paths):
            self.playing = False
            self.show_current()
            print("DMD sequence complete; final image remains displayed.", flush=True)
            return
        self.index += 1
        self.begin_period()

    def toggle_playback(self, _event: Optional[tk.Event] = None) -> str:
        self.playing = not self.playing
        if self.playing:
            self.begin_period()
        else:
            self.cancel_timers()
            self.show_current()
        return "break"

    def move(self, delta: int) -> None:
        self.cancel_timers()
        self.index = min(max(self.index + delta, 0), len(self.paths) - 1)
        if self.playing:
            self.begin_period()
        else:
            self.show_current()

    def next_image(self, _event: Optional[tk.Event] = None) -> str:
        self.move(1)
        return "break"

    def previous_image(self, _event: Optional[tk.Event] = None) -> str:
        self.move(-1)
        return "break"

    def first_image(self, _event: Optional[tk.Event] = None) -> str:
        self.cancel_timers()
        self.index = 0
        if self.playing:
            self.begin_period()
        else:
            self.show_current()
        return "break"

    def close(self, _event: Optional[tk.Event] = None) -> str:
        self.cancel_timers()
        self.root.destroy()
        return "break"


def print_monitors(monitors: Sequence[dmd.Monitor], selected: int) -> None:
    print("Available monitors:")
    for monitor in monitors:
        suffix = "  <-- selected DMD" if monitor.index == selected else ""
        print(
            f"  Monitor {monitor.index}: {monitor.width}x{monitor.height} "
            f"at ({monitor.left}, {monitor.top}){suffix}"
        )


def prepare_capture_output(
    output_dir: Path,
    paths: Sequence[Path],
    overwrite: bool,
) -> None:
    source_paths = {path.resolve() for path in paths}
    planned = [output_dir / f"{path.stem}.png" for path in paths]
    if any(path.resolve() in source_paths for path in planned):
        raise ValueError("Capture output would overwrite the DMD source images.")
    conflicts = [path for path in planned if path.exists()]
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        conflicts.append(manifest_path)
    if conflicts and not overwrite:
        preview = ", ".join(path.name for path in conflicts[:5])
        more = " ..." if len(conflicts) > 5 else ""
        raise FileExistsError(
            f"Capture outputs already exist ({preview}{more}); use --overwrite "
            "or choose a different --output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def connect_camera(args: argparse.Namespace) -> tuple[Optional[Dvp2FrameWorker], Optional[dict]]:
    if args.no_camera:
        print("Camera disabled by --no-camera; continuing with DMD-only playback.")
        return None, None

    worker: Optional[Dvp2FrameWorker] = None
    try:
        api = dvp2.DvpApi(resolve_path(args.dll))
        devices = api.list_devices()
        dvp2.print_devices(devices)
        if not devices:
            raise RuntimeError("No DVP2 camera found")
        if args.camera < 0 or args.camera >= len(devices):
            raise ValueError(f"--camera must be in [0, {len(devices) - 1}]")
        worker = Dvp2FrameWorker(
            api=api,
            camera_index=args.camera,
            target_format=args.target_format,
            auto_exposure=args.auto_exposure,
            manual_exposure=args.manual_exposure,
            exposure_us=args.exposure_us,
            gain=args.gain,
            timeout_ms=args.camera_timeout_ms,
        )
        worker.start(args.camera_ready_timeout)
        info = devices[args.camera]
        camera_info = {
            "index": args.camera,
            "friendly_name": dvp2.decode_sdk_string(info.FriendlyName),
            "model": dvp2.decode_sdk_string(info.Model),
            "serial_number": dvp2.decode_sdk_string(info.SerialNumber),
            "target_format": args.target_format,
        }
        print("DVP2 camera is streaming; synchronized capture is enabled.")
        return worker, camera_info
    except Exception as exc:
        if worker is not None:
            worker.stop()
        if args.require_camera:
            raise
        print(f"WARNING: DVP2 camera unavailable: {exc}")
        print("Continuing with DMD-only playback; no camera images will be saved.")
        return None, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play native DMD images and capture synchronized DVP2 camera frames."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="One DMD image or a flat image directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Camera PNG directory (default: a sibling ccd/ directory beside input/).",
    )
    parser.add_argument("--monitor", type=int, default=0, help="DMD monitor index.")
    parser.add_argument(
        "--display-seconds",
        type=float,
        default=2.0,
        help="Total display time for each DMD image (default: 2 seconds).",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Delay after a DMD change before camera capture (default: 1 second).",
    )
    parser.add_argument("--start-id", help="First DMD filename stem, for example 0021.")
    parser.add_argument("--limit", type=int, help="Use only the first N selected images.")
    parser.add_argument("--show-cursor", action="store_true", help="Show the DMD cursor.")
    parser.add_argument("--list-monitors", action="store_true", help="List monitors and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan only.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing same-named camera PNGs and manifest to be replaced.",
    )

    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not initialize DVP2; run DMD-only.",
    )
    camera_group.add_argument(
        "--require-camera",
        action="store_true",
        help="Stop instead of falling back when the DVP2 camera is unavailable.",
    )
    parser.add_argument("--camera", type=int, default=0, help="DVP2 camera index.")
    parser.add_argument(
        "--dll",
        type=Path,
        default=dvp2.default_dll_path(),
        help="Path to the 64-bit DVP2 camera DLL.",
    )
    parser.add_argument(
        "--target-format",
        choices=("native", "bgr8", "mono8"),
        default="bgr8",
        help="DVP2 output format (default: bgr8).",
    )
    ae_group = parser.add_mutually_exclusive_group()
    ae_group.add_argument("--auto-exposure", action="store_true")
    ae_group.add_argument("--manual-exposure", action="store_true")
    parser.add_argument("--exposure-us", type=float, help="Manual exposure in microseconds.")
    parser.add_argument("--gain", type=float, help="Manual analog gain.")
    parser.add_argument(
        "--camera-timeout-ms",
        type=int,
        default=500,
        help="Timeout for each background DVP2 frame request (default: 500 ms).",
    )
    parser.add_argument(
        "--camera-ready-timeout",
        type=float,
        default=5.0,
        help="Wait for the first camera frame before DMD-only fallback (default: 5 s).",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.display_seconds <= 0:
        parser.error("--display-seconds must be > 0")
    if args.settle_seconds < 0 or args.settle_seconds >= args.display_seconds:
        parser.error("--settle-seconds must satisfy 0 <= settle < display-seconds")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.camera_timeout_ms <= 0:
        parser.error("--camera-timeout-ms must be > 0")
    if args.camera_ready_timeout <= 0:
        parser.error("--camera-ready-timeout must be > 0")
    if args.exposure_us is not None and args.exposure_us <= 0:
        parser.error("--exposure-us must be > 0")
    if args.gain is not None and args.gain <= 0:
        parser.error("--gain must be > 0")
    if args.auto_exposure and (args.exposure_us is not None or args.gain is not None):
        parser.error("--auto-exposure cannot be combined with --exposure-us or --gain")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    dmd.enable_native_dpi()
    monitors = dmd.enumerate_monitors()
    print_monitors(monitors, args.monitor)
    if args.list_monitors:
        return 0
    if args.monitor < 0 or args.monitor >= len(monitors):
        raise ValueError(f"--monitor must be in [0, {len(monitors) - 1}]")
    monitor = monitors[args.monitor]

    input_path = resolve_path(args.input)
    paths = collect_images(input_path, args.start_id, args.limit)
    validate_dmd_images(paths, monitor)
    output_dir = (
        resolve_path(args.output)
        if args.output is not None
        else default_capture_dir(input_path).resolve()
    )

    print(f"DMD input: {input_path}")
    print(f"Images: {len(paths)} | first={paths[0].name} | last={paths[-1].name}")
    print(
        f"Timing: display {args.display_seconds:g} s/image; "
        f"capture after {args.settle_seconds:g} s"
    )
    print(f"Camera output: {output_dir}")
    if args.dry_run:
        if not args.no_camera:
            try:
                api = dvp2.DvpApi(resolve_path(args.dll))
                dvp2.print_devices(api.list_devices())
            except Exception as exc:
                print(f"WARNING: DVP2 check failed: {exc}")
        print("Dry run complete; no DMD window or camera stream was opened.")
        return 0

    camera, camera_info = connect_camera(args)
    if camera is not None:
        try:
            prepare_capture_output(output_dir, paths, args.overwrite)
        except Exception:
            camera.stop()
            raise

    manifest = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dmd_input": str(input_path),
        "camera_output": str(output_dir) if camera is not None else None,
        "display_seconds": args.display_seconds,
        "settle_seconds": args.settle_seconds,
        "camera": camera_info,
        "captures": [],
    }
    if camera is not None:
        (output_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("SPACE: start/pause/resume | LEFT/RIGHT: previous/next | HOME: first | ESC/Q: quit")
    print("The first DMD image is ready but paused; press Space to begin the 2-second sequence.")
    try:
        root = tk.Tk()
        SynchronizedDmdPlayer(
            root=root,
            monitor=monitor,
            paths=paths,
            display_seconds=args.display_seconds,
            settle_seconds=args.settle_seconds,
            camera=camera,
            capture_dir=output_dir if camera is not None else None,
            overwrite=args.overwrite,
            manifest=manifest,
            show_cursor=args.show_cursor,
        )
        root.mainloop()
    finally:
        if camera is not None:
            camera.stop()
            print("DVP2 camera stream closed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
