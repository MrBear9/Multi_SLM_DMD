"""Reusable dvp2_camera image and device operations."""

from __future__ import annotations
import argparse
from ctypes import byref, c_double
from ctypes import c_float, c_int
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
import cv2
import numpy as np
from .sdk import (
    FORMAT_MONO,
    FORMAT_BAYER_BG,
    FORMAT_BAYER_GB,
    FORMAT_BAYER_GR,
    FORMAT_BAYER_RG,
    FORMAT_BGR24,
    FORMAT_BGR32,
    FORMAT_BGR48,
    FORMAT_BGR64,
    FORMAT_RGB24,
    FORMAT_RGB32,
    FORMAT_RGB48,
    FORMAT_RGB64,
    BITS_8,
    S_BGR24,
    S_MONO8,
    AE_OP_OFF,
    AE_OP_CONTINUOUS,
    FRAME_NAME_PATTERN,
    DvpError,
    DvpCameraInfo,
    DvpRegion,
    DvpFrame,
    decode_sdk_string,
    encode_sdk_path,
    default_dll_path,
    check_status,
    DvpApi,
    CameraSession,
)


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
