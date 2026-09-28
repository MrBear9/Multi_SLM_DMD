"""Typed SDK bindings and device discovery for zkwx_slm. No DLL is loaded on import."""

from __future__ import annotations
import ctypes
import os
import struct
import tempfile
from ctypes import wintypes
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4
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

    # Bound disk use if the SDK retains old file handles indefinitely.
    _MAX_CANVAS_FILES = 8

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
        try:
            self.dll = ctypes.CDLL(str(dll_path))
        except Exception:
            if self._dll_directory is not None:
                self._dll_directory.close()
                self._dll_directory = None
            raise
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
        # SDK 2.1 manual section 1.3 declares this function as void.
        self.dll.Timeout_ShowImageFromFilePath.restype = None
        self.dll.Timeout_CloseWindow.argtypes = []
        self.dll.Timeout_CloseWindow.restype = None
        self.created = False
        self._canvas_directory: tempfile.TemporaryDirectory | None = None
        self._active_canvas: Path | None = None

    def open(self) -> None:
        self.dll.Timeout_CreateWindow()
        self.created = True
        self.dll.Timeout_ShowWindow(True)

    def show(
        self,
        path: Path,
        monitor: Monitor,
        interval_ms: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        """Display native pixels, optionally shifted from the panel centre.

        SDK DesX/DesY position the output window, not the image inside it. For
        offsets we therefore keep that window on the complete monitor and pass
        a lossless, full-panel image. Zero offset retains the original SDK path.
        """
        output_path = path
        next_canvas = None
        if offset_x or offset_y:
            # Imported here because controller also exposes the SDK types.
            from .controller import prepare_canvas

            with prepare_canvas(path, monitor, offset_x, offset_y) as canvas:
                if getattr(self, "_canvas_directory", None) is None:
                    self._canvas_directory = tempfile.TemporaryDirectory(prefix="slm1_canvas_")
                self._cleanup_canvases(getattr(self, "_active_canvas", None))
                canvas_directory = Path(self._canvas_directory.name)
                if len(list(canvas_directory.iterdir())) >= self._MAX_CANVAS_FILES:
                    raise RuntimeError(
                        "SLM1 SDK is keeping previous canvas files locked. "
                        "Reconnect SLM1 before continuing playback."
                    )
                # TIFF preserves 16/32-bit samples; BMP preserves 1/8-bit samples.
                suffix = ".bmp" if canvas.mode in {"1", "L"} else ".tiff"
                # The SDK does not document file-handle release or path caching.
                # Never overwrite a file it may still be displaying.
                output_path = canvas_directory / (uuid4().hex + suffix)
                canvas.save(output_path)
                next_canvas = output_path
        self.dll.Timeout_ShowImageFromFilePath(
            str(output_path).encode("utf-8"),
            False,  # IsStretch: preserve native pixels and centre the image.
            monitor.left,
            monitor.top,
            monitor.width,
            monitor.height,
            False,  # IsRGB: phase patterns are grayscale.
            interval_ms,
        )
        self._active_canvas = next_canvas
        self._cleanup_canvases(next_canvas)

    def _cleanup_canvases(self, keep: Path | None) -> None:
        """Delete retired frames; retry locked files on later frames or close."""
        if getattr(self, "_canvas_directory", None) is None:
            return
        for path in Path(self._canvas_directory.name).iterdir():
            if path == keep:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # Windows rejects deletion while the vendor keeps a handle open.
                # The bounded file count above prevents unlimited accumulation.
                pass

    def close(self) -> None:
        if self.created:
            self.dll.Timeout_CloseWindow()
            self.created = False
        try:
            if getattr(self, "_canvas_directory", None) is not None:
                self._canvas_directory.cleanup()
                self._canvas_directory = None
                self._active_canvas = None
        finally:
            if self._dll_directory is not None:
                self._dll_directory.close()
                self._dll_directory = None
