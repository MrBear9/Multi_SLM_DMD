"""Typed SDK bindings and device discovery for zkwx_slm. No DLL is loaded on import."""

from __future__ import annotations
import ctypes
import os
import struct
from ctypes import wintypes
from pathlib import Path
from typing import Any, NamedTuple
from ...utils.paths import PROJECT_ROOT

DEFAULT_INPUT = PROJECT_ROOT / "tools" / "slm1"

DEFAULT_SDK_DIR = PROJECT_ROOT / "tools" / "3rdparty" / "python_VS2015_x64"

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

MONITORINFOF_PRIMARY = 1

class Monitor(NamedTuple):
    index: int
    left: int
    top: int
    width: int
    height: int
    primary: bool

class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]

def enable_native_dpi() -> None:
    """Use physical desktop pixels, as required by the SDK coordinates."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

def enumerate_monitors() -> list[Monitor]:
    if os.name != "nt":
        raise OSError("Zhongke Weixing SLM SDK 2.1 is supported only on Windows.")

    found: list[tuple[int, int, int, int, bool]] = []
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )

    def collect_monitor(
        monitor_handle: int,
        _device_context: int,
        _rectangle_pointer: int,
        _data: int,
    ) -> bool:
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if not ctypes.windll.user32.GetMonitorInfoW(monitor_handle, ctypes.byref(info)):
            return True
        rectangle = info.rcMonitor
        found.append(
            (
                int(rectangle.left),
                int(rectangle.top),
                int(rectangle.right - rectangle.left),
                int(rectangle.bottom - rectangle.top),
                bool(info.dwFlags & MONITORINFOF_PRIMARY),
            )
        )
        return True

    callback = callback_type(collect_monitor)
    if not ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0):
        raise RuntimeError("Windows monitor enumeration failed.")
    if not found:
        raise RuntimeError("No display monitor was detected.")
    return [Monitor(index, *values) for index, values in enumerate(found)]

class ZhongkeTimeoutSDK:
    """Minimal ctypes wrapper for the documented SDK 2.1 Timeout API."""

    def __init__(self, sdk_dir: Path) -> None:
        if struct.calcsize("P") != 8:
            raise RuntimeError("python_VS2015_x64 requires a 64-bit Python interpreter.")
        if not sdk_dir.is_dir():
            raise NotADirectoryError(f"SDK directory not found: {sdk_dir}")
        dll_path = sdk_dir / "SecondDll.dll"
        if not dll_path.is_file():
            raise FileNotFoundError(f"SLM SDK DLL not found: {dll_path}")

        self._dll_directory: Any = None
        if hasattr(os, "add_dll_directory"):
            self._dll_directory = os.add_dll_directory(str(sdk_dir))
        self.dll = ctypes.CDLL(str(dll_path))
        self.dll.Timeout_CreateWindow.argtypes = []
        self.dll.Timeout_CreateWindow.restype = None
        self.dll.Timeout_ShowWindow.argtypes = [ctypes.c_bool]
        self.dll.Timeout_ShowWindow.restype = None
        self.dll.Timeout_ShowImageFromFilePath.argtypes = [
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_int,
        ]
        self.dll.Timeout_ShowImageFromFilePath.restype = ctypes.c_bool
        self.dll.Timeout_CloseWindow.argtypes = []
        self.dll.Timeout_CloseWindow.restype = None
        self.created = False

    def open(self) -> None:
        self.dll.Timeout_CreateWindow()
        self.created = True
        self.dll.Timeout_ShowWindow(True)

    def show(self, path: Path, monitor: Monitor, interval_ms: int) -> None:
        succeeded = self.dll.Timeout_ShowImageFromFilePath(
            str(path).encode("utf-8"),
            False,  # IsStretch: preserve native pixels and centre the image.
            monitor.left,
            monitor.top,
            monitor.width,
            monitor.height,
            False,  # IsRGB: phase patterns are grayscale.
            interval_ms,
        )
        if not succeeded:
            raise RuntimeError(f"SLM SDK failed to display: {path}")

    def close(self) -> None:
        if self.created:
            self.dll.Timeout_CloseWindow()
            self.created = False
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None
