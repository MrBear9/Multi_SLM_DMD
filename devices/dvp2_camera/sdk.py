"""Typed SDK bindings and device discovery for dvp2_camera. No DLL is loaded on import."""

from __future__ import annotations
import ctypes
import os
import re
import sys
from ctypes import POINTER, Structure, byref, c_bool, c_char, c_double
from ctypes import c_float, c_int, c_int32, c_uint16, c_uint32, c_uint64, c_void_p
from pathlib import Path
import numpy as np
from ...utils.paths import THIRD_PARTY_ROOT

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
        THIRD_PARTY_ROOT
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
        try:
            self._bind()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Release the DLL search directory after all camera sessions close."""
        if self._dll_dir_handle is not None:
            self._dll_dir_handle.close()
            self._dll_dir_handle = None

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
