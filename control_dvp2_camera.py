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
import ctypes
import os
import re
import sys
import time
from ctypes import POINTER, Structure, byref, c_bool, c_char, c_double
from ctypes import c_float, c_int, c_int32, c_uint16, c_uint32, c_uint64, c_void_p
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


DVP_STATUS_OK = 1
DVP_STATUS_TIME_OUT = -1000

OPEN_NORMAL = 1 << 0

FORMAT_MONO = 0
FORMAT_BAYER_BG = 1
FORMAT_BAYER_GB = 2
FORMAT_BAYER_GR = 3
FORMAT_BAYER_RG = 4
FORMAT_BGR24 = 10
FORMAT_BGR32 = 11
FORMAT_BGR48 = 12
FORMAT_BGR64 = 13
FORMAT_RGB24 = 14
FORMAT_RGB32 = 15
FORMAT_RGB48 = 16
FORMAT_RGB64 = 17

BITS_8 = 0

S_BGR24 = 10
S_MONO8 = 30

AE_OP_OFF = 0
AE_OP_CONTINUOUS = 2

WINDOW_NAME = "DVP2 Camera"
FRAME_NAME_PATTERN = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


class DvpError(RuntimeError):
    """An error returned by the DVP2 API."""


class DvpCameraInfo(Structure):
    _fields_ = [
        ("Vendor", c_char * 64),
        ("Manufacturer", c_char * 64),
        ("Model", c_char * 64),
        ("Family", c_char * 64),
        ("LinkName", c_char * 64),
        ("SensorInfo", c_char * 64),
        ("HardwareVersion", c_char * 64),
        ("FirmwareVersion", c_char * 64),
        ("KernelVersion", c_char * 64),
        ("DscamVersion", c_char * 64),
        ("FriendlyName", c_char * 64),
        ("PortInfo", c_char * 64),
        ("SerialNumber", c_char * 64),
        ("CameraInfo", c_char * 128),
        ("UserID", c_char * 128),
        ("OriginalSerialNumber", c_char * 64),
        ("reserved", c_char * 64),
    ]


class DvpRegion(Structure):
    _fields_ = [
        ("X", c_int32),
        ("Y", c_int32),
        ("W", c_int32),
        ("H", c_int32),
        ("reserved", c_uint32 * 32),
    ]


class DvpFrame(Structure):
    # Layout copied from the bundled DVPCamera.h (64-bit SDK).
    _fields_ = [
        ("format", c_int),
        ("bits", c_int),
        ("uBytes", c_uint32),
        ("iWidth", c_int32),
        ("iHeight", c_int32),
        ("uFrameID", c_uint64),
        ("uTimestamp", c_uint64),
        ("fExposure", c_double),
        ("fAGain", c_float),
        ("position", c_int),
        ("bFlipHorizontalState", c_bool),
        ("bFlipVerticalState", c_bool),
        ("bRotateState", c_bool),
        ("bRotateOpposite", c_bool),
        ("internalFlags", c_uint32),
        ("internalValue", c_uint32),
        ("uTriggerId", c_uint64),
        ("uLineLevelStatus", c_uint16),
        ("reserved1", ctypes.c_ubyte * 6),
        ("pExtra", c_void_p),
        ("reserved2", c_uint32 * 22),
    ]


def decode_sdk_string(value: bytes) -> str:
    raw = bytes(value).split(b"\0", 1)[0]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def encode_sdk_path(path: Path) -> bytes:
    encoding = "mbcs" if os.name == "nt" else sys.getfilesystemencoding()
    return str(path).encode(encoding, errors="strict")


def default_dll_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "3rdparty"
        / "DVP2 SDK CN"
        / "library"
        / "Visual C++"
        / "bin"
        / "x64"
        / "DVPCamera64.dll"
    )


def check_status(status: int, operation: str) -> None:
    if status != DVP_STATUS_OK:
        raise DvpError(f"{operation} failed: DVP2 status={status}")


class DvpApi:
    def __init__(self, dll_path: Path) -> None:
        if os.name != "nt" or ctypes.sizeof(c_void_p) != 8:
            raise RuntimeError("This script requires 64-bit Python on Windows.")
        dll_path = dll_path.expanduser().resolve()
        if not dll_path.is_file():
            raise FileNotFoundError(f"DVP2 DLL not found: {dll_path}")

        self._dll_dir_handle = os.add_dll_directory(str(dll_path.parent))
        try:
            self.dll = ctypes.CDLL(str(dll_path))
        except OSError as exc:
            self._dll_dir_handle.close()
            raise RuntimeError(
                f"Unable to load DVP2 DLL: {dll_path}. Install the DVP2 camera "
                "driver/runtime and confirm that 64-bit dependencies are available."
            ) from exc
        self._bind()

    def _bind(self) -> None:
        dll = self.dll
        dll.dvpRefresh.argtypes = [POINTER(c_uint32)]
        dll.dvpRefresh.restype = c_int
        dll.dvpEnum.argtypes = [c_uint32, POINTER(DvpCameraInfo)]
        dll.dvpEnum.restype = c_int
        dll.dvpOpen.argtypes = [c_uint32, c_int, POINTER(c_uint32)]
        dll.dvpOpen.restype = c_int
        dll.dvpClose.argtypes = [c_uint32]
        dll.dvpClose.restype = c_int
        dll.dvpStart.argtypes = [c_uint32]
        dll.dvpStart.restype = c_int
        dll.dvpStop.argtypes = [c_uint32]
        dll.dvpStop.restype = c_int
        dll.dvpGetFrame.argtypes = [
            c_uint32,
            POINTER(DvpFrame),
            POINTER(c_void_p),
            c_uint32,
        ]
        dll.dvpGetFrame.restype = c_int

        dll.dvpSetTriggerState.argtypes = [c_uint32, c_bool]
        dll.dvpSetTriggerState.restype = c_int
        dll.dvpGetTargetFormat.argtypes = [c_uint32, POINTER(c_int)]
        dll.dvpGetTargetFormat.restype = c_int
        dll.dvpSetTargetFormat.argtypes = [c_uint32, c_int]
        dll.dvpSetTargetFormat.restype = c_int

        dll.dvpGetExposure.argtypes = [c_uint32, POINTER(c_double)]
        dll.dvpGetExposure.restype = c_int
        dll.dvpSetExposure.argtypes = [c_uint32, c_double]
        dll.dvpSetExposure.restype = c_int
        dll.dvpGetAnalogGain.argtypes = [c_uint32, POINTER(c_float)]
        dll.dvpGetAnalogGain.restype = c_int
        dll.dvpSetAnalogGain.argtypes = [c_uint32, c_float]
        dll.dvpSetAnalogGain.restype = c_int
        dll.dvpGetAeOperation.argtypes = [c_uint32, POINTER(c_int)]
        dll.dvpGetAeOperation.restype = c_int
        dll.dvpSetAeOperation.argtypes = [c_uint32, c_int]
        dll.dvpSetAeOperation.restype = c_int

        dll.dvpGetRoi.argtypes = [c_uint32, POINTER(DvpRegion)]
        dll.dvpGetRoi.restype = c_int
        dll.dvpSetRoi.argtypes = [c_uint32, DvpRegion]
        dll.dvpSetRoi.restype = c_int
        dll.dvpSetRoiState.argtypes = [c_uint32, c_bool]
        dll.dvpSetRoiState.restype = c_int

        dll.dvpLoadConfig.argtypes = [c_uint32, ctypes.c_char_p]
        dll.dvpLoadConfig.restype = c_int
        dll.dvpSaveConfig.argtypes = [c_uint32, ctypes.c_char_p]
        dll.dvpSaveConfig.restype = c_int
        dll.dvpShowPropertyModalDialog.argtypes = [c_uint32, c_void_p]
        dll.dvpShowPropertyModalDialog.restype = c_int

    def list_devices(self) -> list[DvpCameraInfo]:
        count = c_uint32()
        check_status(self.dll.dvpRefresh(byref(count)), "dvpRefresh")
        devices: list[DvpCameraInfo] = []
        for index in range(count.value):
            info = DvpCameraInfo()
            check_status(self.dll.dvpEnum(index, byref(info)), f"dvpEnum({index})")
            devices.append(info)
        return devices


class CameraSession:
    def __init__(self, api: DvpApi, index: int) -> None:
        self.api = api
        self.index = index
        self.handle = c_uint32()
        self.opened = False
        self.started = False

    def open(self) -> None:
        check_status(
            self.api.dll.dvpOpen(self.index, OPEN_NORMAL, byref(self.handle)),
            f"dvpOpen({self.index})",
        )
        self.opened = True

    def start(self) -> None:
        check_status(self.api.dll.dvpStart(self.handle.value), "dvpStart")
        self.started = True

    def get_frame(self, timeout_ms: int) -> tuple[DvpFrame, np.ndarray]:
        frame = DvpFrame()
        buffer_ptr = c_void_p()
        status = self.api.dll.dvpGetFrame(
            self.handle.value, byref(frame), byref(buffer_ptr), timeout_ms
        )
        if status == DVP_STATUS_TIME_OUT:
            raise TimeoutError("DVP2 frame timeout")
        check_status(status, "dvpGetFrame")
        if not buffer_ptr.value or frame.uBytes == 0:
            raise DvpError("DVP2 returned an empty frame buffer")
        raw_type = ctypes.c_ubyte * frame.uBytes
        raw = np.ctypeslib.as_array(raw_type.from_address(buffer_ptr.value)).copy()
        return frame, raw

    def close(self) -> None:
        if self.started:
            status = self.api.dll.dvpStop(self.handle.value)
            self.started = False
            if status != DVP_STATUS_OK:
                print(f"Warning: dvpStop status={status}", file=sys.stderr)
        if self.opened:
            status = self.api.dll.dvpClose(self.handle.value)
            self.opened = False
            if status != DVP_STATUS_OK:
                print(f"Warning: dvpClose status={status}", file=sys.stderr)

    def __enter__(self) -> "CameraSession":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def print_devices(devices: Sequence[DvpCameraInfo]) -> None:
    print("Available DVP2 cameras:")
    if not devices:
        print("  (none)")
        return
    for index, info in enumerate(devices):
        friendly = decode_sdk_string(info.FriendlyName) or "(unnamed)"
        model = decode_sdk_string(info.Model) or "unknown model"
        serial = decode_sdk_string(info.SerialNumber) or "unknown serial"
        port = decode_sdk_string(info.PortInfo) or "unknown port"
        print(f"  Camera {index}: {friendly} | model={model} | serial={serial} | {port}")


def frame_to_image(frame: DvpFrame, raw: np.ndarray) -> np.ndarray:
    width = int(frame.iWidth)
    height = int(frame.iHeight)
    if width <= 0 or height <= 0:
        raise DvpError(f"Invalid frame dimensions: {width}x{height}")

    is_8_bit = frame.bits == BITS_8
    dtype = np.uint8 if is_8_bit else np.uint16
    values = raw.view(dtype)
    pixel_format = int(frame.format)

    if pixel_format in {
        FORMAT_MONO,
        FORMAT_BAYER_BG,
        FORMAT_BAYER_GB,
        FORMAT_BAYER_GR,
        FORMAT_BAYER_RG,
    }:
        needed = width * height
        if values.size < needed:
            raise DvpError(f"Frame buffer is too small: {values.size} < {needed}")
        image = values[:needed].reshape(height, width)
        bayer_codes = {
            FORMAT_BAYER_BG: cv2.COLOR_BayerBG2BGR,
            FORMAT_BAYER_GB: cv2.COLOR_BayerGB2BGR,
            FORMAT_BAYER_GR: cv2.COLOR_BayerGR2BGR,
            FORMAT_BAYER_RG: cv2.COLOR_BayerRG2BGR,
        }
        if pixel_format in bayer_codes:
            image = cv2.cvtColor(image, bayer_codes[pixel_format])
        return image.copy()

    format_channels = {
        FORMAT_BGR24: (3, False),
        FORMAT_BGR32: (4, False),
        FORMAT_BGR48: (3, False),
        FORMAT_BGR64: (4, False),
        FORMAT_RGB24: (3, True),
        FORMAT_RGB32: (4, True),
        FORMAT_RGB48: (3, True),
        FORMAT_RGB64: (4, True),
    }
    if pixel_format not in format_channels:
        raise DvpError(
            f"Unsupported DVP2 image format={pixel_format}, bits={frame.bits}"
        )
    channels, is_rgb = format_channels[pixel_format]
    needed = width * height * channels
    if values.size < needed:
        raise DvpError(f"Frame buffer is too small: {values.size} < {needed}")
    image = values[:needed].reshape(height, width, channels)
    if channels == 4:
        code = cv2.COLOR_RGBA2BGR if is_rgb else cv2.COLOR_BGRA2BGR
        image = cv2.cvtColor(image, code)
    elif is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image.copy()


def image_for_preview(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        preview = image
    else:
        max_value = float(np.max(image)) if image.size else 0.0
        scale = 255.0 / max(max_value, 1.0)
        preview = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if preview.ndim == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    return preview


def next_frame_index(output_dir: Path) -> int:
    maximum = 0
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            match = FRAME_NAME_PATTERN.match(path.name)
            if match:
                maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def save_frame(image: np.ndarray, output_dir: Path, index: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"frame_{index:06d}.png"
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Unable to save frame: {path}")
    print(f"Saved: {path}")
    return path


class VideoRecorder:
    def __init__(self, fps: float, codec: str) -> None:
        self.fps = fps
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None
        self.path: Optional[Path] = None

    @property
    def active(self) -> bool:
        return self.writer is not None

    def start(self, path: Path, image: np.ndarray) -> None:
        if self.writer is not None:
            return
        path = path.expanduser().resolve()
        if not path.suffix:
            path = path.with_suffix(".avi")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = image_for_preview(image)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*self.codec), self.fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Unable to create video: {path} (codec={self.codec}, fps={self.fps})"
            )
        self.writer = writer
        self.path = path
        print(f"Recording started: {path}")

    def write(self, image: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(image_for_preview(image))

    def stop(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        print(f"Recording stopped: {self.path}")
        self.writer = None
        self.path = None


def timestamped_video_path(output_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"dvp2_{stamp}.avi"


def get_exposure(api: DvpApi, handle: int) -> float:
    value = c_double()
    check_status(api.dll.dvpGetExposure(handle, byref(value)), "dvpGetExposure")
    return float(value.value)


def get_gain(api: DvpApi, handle: int) -> float:
    value = c_float()
    check_status(api.dll.dvpGetAnalogGain(handle, byref(value)), "dvpGetAnalogGain")
    return float(value.value)


def get_ae_operation(api: DvpApi, handle: int) -> int:
    value = c_int()
    check_status(api.dll.dvpGetAeOperation(handle, byref(value)), "dvpGetAeOperation")
    return int(value.value)


def configure_camera(api: DvpApi, handle: int, args: argparse.Namespace) -> None:
    if args.load_config is not None:
        config_path = args.load_config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Camera config not found: {config_path}")
        check_status(
            api.dll.dvpLoadConfig(handle, encode_sdk_path(config_path)),
            "dvpLoadConfig",
        )

    # Configuration files may contain a trigger setting.  Force continuous
    # acquisition after loading so the preview cannot silently wait for a trigger.
    check_status(api.dll.dvpSetTriggerState(handle, False), "disable trigger mode")

    target_formats = {"bgr8": S_BGR24, "mono8": S_MONO8}
    if args.target_format in target_formats:
        check_status(
            api.dll.dvpSetTargetFormat(handle, target_formats[args.target_format]),
            f"set target format {args.target_format}",
        )

    if args.auto_exposure:
        check_status(
            api.dll.dvpSetAeOperation(handle, AE_OP_CONTINUOUS),
            "enable continuous auto exposure",
        )
    elif args.manual_exposure or args.exposure_us is not None or args.gain is not None:
        check_status(
            api.dll.dvpSetAeOperation(handle, AE_OP_OFF),
            "disable auto exposure",
        )

    if args.exposure_us is not None:
        check_status(
            api.dll.dvpSetExposure(handle, args.exposure_us),
            f"set exposure to {args.exposure_us} us",
        )
    if args.gain is not None:
        check_status(
            api.dll.dvpSetAnalogGain(handle, args.gain),
            f"set analog gain to {args.gain}",
        )

    if args.roi is not None:
        x, y, width, height = args.roi
        roi = DvpRegion(X=x, Y=y, W=width, H=height)
        check_status(api.dll.dvpSetRoi(handle, roi), "dvpSetRoi")
        check_status(api.dll.dvpSetRoiState(handle, True), "enable ROI")

    if args.dialog:
        check_status(
            api.dll.dvpShowPropertyModalDialog(handle, None),
            "dvpShowPropertyModalDialog",
        )

    if args.save_config is not None:
        config_path = args.save_config.expanduser().resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        check_status(
            api.dll.dvpSaveConfig(handle, encode_sdk_path(config_path)),
            "dvpSaveConfig",
        )
        print(f"Camera configuration saved: {config_path}")


def print_camera_parameters(api: DvpApi, handle: int) -> None:
    target = c_int()
    roi = DvpRegion()
    check_status(api.dll.dvpGetTargetFormat(handle, byref(target)), "dvpGetTargetFormat")
    check_status(api.dll.dvpGetRoi(handle, byref(roi)), "dvpGetRoi")
    ae = get_ae_operation(api, handle)
    print(
        "Camera parameters: "
        f"target_format={target.value}, AE={ae}, "
        f"exposure={get_exposure(api, handle):.3f} us, "
        f"gain={get_gain(api, handle):.3f}, "
        f"ROI=({roi.X}, {roi.Y}, {roi.W}, {roi.H})"
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
