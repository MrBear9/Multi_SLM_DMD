"""Typed SDK bindings and device discovery for magicholo_slm. No DLL is loaded on import."""

from __future__ import annotations
import ctypes
import os
import struct
from ctypes import wintypes
from pathlib import Path
from typing import Any, NamedTuple
from ...utils.paths import PROJECT_ROOT

DEFAULT_INPUT = PROJECT_ROOT / "tools" / "slm2"

DEFAULT_SDK_DIR = (
    PROJECT_ROOT / "tools" / "3rdparty" / "HDSLM_SDK_0612" / "HDSLM_API" / "Bin"
)

IMAGE_EXTENSIONS = {".bmp", ".png"}

HDSLM45R_SIZE = (1920, 1080)

HDSLM_8BIT_FLAG = 256

HDSLM_OK = 1

MONITORINFOF_PRIMARY = 1

PM_REMOVE = 1

STATUS_NAMES = {
    1: "HDSLM_OK",
    -1: "HDSLM_NG (no display/unknown error)",
    -2: "HDSLM_NOT_OPEN_MONITOR",
    -3: "HDSLM_OPEN_WINDOW_ERR",
    -4: "HDSLM_DATA_FORMAT_ERR",
    -5: "HDSLM_NOT_SUPPORT_IMAGE",
    -6: "HDSLM_FILE_READ_ERR",
    -7: "HDSLM_GAMMAFILE_READ_ERR",
    -8: "HDSLM_GAMMAFILE_SUPPORT_ERR",
    -9: "HDSLM_IMAGESPLICE_ERROR",
}

class DesktopMonitor(NamedTuple):
    left: int
    top: int
    width: int
    height: int
    primary: bool

class SDKDisplay(NamedTuple):
    index: int
    name: str
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
    """Make Windows coordinates correspond to physical display pixels."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

def enumerate_desktop_monitors() -> list[DesktopMonitor]:
    if os.name != "nt":
        raise OSError("The HDSLM SDK is supported only on Windows.")

    found: list[DesktopMonitor] = []
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
        if ctypes.windll.user32.GetMonitorInfoW(monitor_handle, ctypes.byref(info)):
            rect = info.rcMonitor
            found.append(
                DesktopMonitor(
                    left=int(rect.left),
                    top=int(rect.top),
                    width=int(rect.right - rect.left),
                    height=int(rect.bottom - rect.top),
                    primary=bool(info.dwFlags & MONITORINFOF_PRIMARY),
                )
            )
        return True

    callback = callback_type(collect_monitor)
    if not ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0):
        raise RuntimeError("Windows monitor enumeration failed.")
    return found

class HDSLM8BitSDK:
    """Small, typed wrapper around the official 64-bit HDSLMFunc.dll."""

    def __init__(self, sdk_dir: Path) -> None:
        if os.name != "nt":
            raise OSError("The HDSLM SDK is supported only on Windows.")
        if struct.calcsize("P") != 8:
            raise RuntimeError("HDSLMFunc.dll requires a 64-bit Python interpreter.")
        if not sdk_dir.is_dir():
            raise NotADirectoryError(f"SDK directory not found: {sdk_dir}")
        dll_path = sdk_dir / "HDSLMFunc.dll"
        if not dll_path.is_file():
            raise FileNotFoundError(f"HDSLM SDK DLL not found: {dll_path}")

        self._dll_directory: Any = None
        if hasattr(os, "add_dll_directory"):
            self._dll_directory = os.add_dll_directory(str(sdk_dir))
        try:
            self.dll = ctypes.WinDLL(str(dll_path), use_last_error=True)
        except Exception:
            if self._dll_directory is not None:
                self._dll_directory.close()
                self._dll_directory = None
            raise

        status = ctypes.c_int
        self.dll.SLM_Disp_Info_NumberName.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_char),
        ]
        self.dll.SLM_Disp_Info_NumberName.restype = status
        self.dll.SLM_Disp_Info.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.dll.SLM_Disp_Info.restype = status
        self.dll.SLM_Disp_CoordInfo.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.dll.SLM_Disp_CoordInfo.restype = status
        self.dll.SLM_Disp_Open.argtypes = [ctypes.c_int]
        self.dll.SLM_Disp_Open.restype = status
        self.dll.SLM_Disp_Close.argtypes = [ctypes.c_int]
        self.dll.SLM_Disp_Close.restype = status
        self.dll.SLM_Disp_Data.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.dll.SLM_Disp_Data.restype = status
        self.opened_display: int | None = None

    @staticmethod
    def check(status: int, operation: str) -> None:
        if status != HDSLM_OK:
            detail = STATUS_NAMES.get(status, "unknown SDK status")
            raise RuntimeError(f"{operation} failed: {status} ({detail}).")

    def list_displays(self) -> list[SDKDisplay]:
        count = ctypes.c_int()
        names_buffer = ctypes.create_string_buffer(4096)
        self.check(
            self.dll.SLM_Disp_Info_NumberName(ctypes.byref(count), names_buffer),
            "SLM_Disp_Info_NumberName",
        )
        if count.value <= 0:
            raise RuntimeError("The HDSLM SDK did not find any display.")

        raw_names = names_buffer.value.decode("utf-8", errors="replace")
        names = [name.strip() for name in raw_names.split(";") if name.strip()]
        desktop_monitors = enumerate_desktop_monitors()
        displays: list[SDKDisplay] = []
        for index in range(count.value):
            width = ctypes.c_int()
            height = ctypes.c_int()
            left = ctypes.c_int()
            top = ctypes.c_int()
            self.check(
                self.dll.SLM_Disp_Info(index, ctypes.byref(width), ctypes.byref(height)),
                f"SLM_Disp_Info({index})",
            )
            self.check(
                self.dll.SLM_Disp_CoordInfo(index, ctypes.byref(left), ctypes.byref(top)),
                f"SLM_Disp_CoordInfo({index})",
            )
            primary = any(
                monitor.primary
                and (monitor.left, monitor.top, monitor.width, monitor.height)
                == (left.value, top.value, width.value, height.value)
                for monitor in desktop_monitors
            )
            name = names[index] if index < len(names) else f"Display {index}"
            displays.append(
                SDKDisplay(
                    index=index,
                    name=name,
                    left=left.value,
                    top=top.value,
                    width=width.value,
                    height=height.value,
                    primary=primary,
                )
            )
        return displays

    def open(self, display_number: int) -> None:
        self.check(self.dll.SLM_Disp_Open(display_number), "SLM_Disp_Open")
        self.opened_display = display_number

    def show_8bit(self, display_number: int, width: int, height: int, data: bytes) -> None:
        expected = width * height
        if len(data) != expected:
            raise ValueError(f"8-bit frame has {len(data)} bytes; expected {expected}.")
        buffer = (ctypes.c_ubyte * expected).from_buffer_copy(data)
        self.check(
            self.dll.SLM_Disp_Data(
                display_number,
                width,
                height,
                HDSLM_8BIT_FLAG,
                ctypes.cast(buffer, ctypes.c_void_p),
            ),
            "SLM_Disp_Data(8-bit)",
        )

    def close(self) -> None:
        if self.opened_display is not None:
            status = self.dll.SLM_Disp_Close(self.opened_display)
            if status != HDSLM_OK:
                print(
                    f"Warning: SLM_Disp_Close returned {status} "
                    f"({STATUS_NAMES.get(status, 'unknown SDK status')})."
                )
            self.opened_display = None
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None
