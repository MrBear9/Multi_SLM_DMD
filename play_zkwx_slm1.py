"""Display one or more phase images on a Zhongke Weixing SLM via SDK 2.1.

The implementation follows the vendor's Timeout API for images sent from a
Python loop. Images are always shown with ``IsStretch=False`` so their native
pixels are centred in the selected SLM display region without resizing.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import struct
from ctypes import wintypes
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display a single image or an ordered image directory on Zhongke "
            "Weixing SLM1 using the vendor SDK 2.1 Timeout API."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="One phase image or a flat directory of phase images (default: tools/slm1).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds per image for multi-image playback (default: 5).",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="Target monitor index; default is the first non-primary monitor.",
    )
    parser.add_argument(
        "--start-id",
        default=None,
        help="Start at this filename stem for a directory, for example 0021.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Display only the first N selected images.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the selected sequence until Ctrl+C.",
    )
    parser.add_argument(
        "--close-after",
        action="store_true",
        help="Close the SLM window after the sequence instead of holding the final image.",
    )
    parser.add_argument(
        "--sdk-dir",
        type=Path,
        default=DEFAULT_SDK_DIR,
        help="Directory containing SecondDll.dll and its dependencies.",
    )
    parser.add_argument(
        "--list-monitors",
        action="store_true",
        help="List display indices and coordinates, then exit without loading the SDK.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned SDK calls without opening the SLM window.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def print_monitors(monitors: list[Monitor], selected_index: int | None = None) -> None:
    print("Available monitors:")
    for monitor in monitors:
        flags = []
        if monitor.primary:
            flags.append("primary")
        if monitor.index == selected_index:
            flags.append("selected SLM")
        suffix = f"  <-- {', '.join(flags)}" if flags else ""
        print(
            f"  Monitor {monitor.index}: {monitor.width}x{monitor.height} "
            f"at ({monitor.left}, {monitor.top}){suffix}"
        )


def select_monitor(monitors: list[Monitor], requested_index: int | None) -> Monitor:
    if requested_index is not None:
        if not 0 <= requested_index < len(monitors):
            raise ValueError(
                f"--monitor {requested_index} is unavailable; choose 0..{len(monitors) - 1}."
            )
        return monitors[requested_index]
    for monitor in monitors:
        if not monitor.primary:
            return monitor
    raise RuntimeError(
        "Only the primary display is available. Connect the SLM as an extended display, "
        "set scaling to 100%, and run again."
    )


def natural_image_key(path: Path) -> tuple[int, int | str, str]:
    if path.stem.isdigit():
        return 0, int(path.stem), path.name.casefold()
    return 1, path.stem.casefold(), path.name.casefold()


def collect_images(input_path: Path, start_id: str | None, limit: int | None) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {input_path}")
        if start_id is not None:
            raise ValueError("--start-id can be used only when --input is a directory.")
        paths = [input_path]
    elif input_path.is_dir():
        paths = sorted(
            (
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=natural_image_key,
        )
        if not paths:
            raise RuntimeError(f"No supported phase images found in: {input_path}")
        if start_id is not None:
            start_index = next(
                (index for index, path in enumerate(paths) if path.stem == start_id),
                -1,
            )
            if start_index < 0:
                raise ValueError(f"--start-id {start_id!r} was not found in: {input_path}")
            paths = paths[start_index:]
    else:
        raise FileNotFoundError(f"Input image or directory not found: {input_path}")
    return paths if limit is None else paths[:limit]


def validate_images(paths: list[Path], monitor: Monitor) -> tuple[int, int]:
    expected_size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            image.verify()
            size = (int(image.width), int(image.height))
            mode = image.mode
        if size[0] > monitor.width or size[1] > monitor.height:
            raise ValueError(
                f"{path.name} is {size[0]}x{size[1]}, larger than selected SLM region "
                f"{monitor.width}x{monitor.height}. Stretching/cropping is disabled."
            )
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError(
                f"All phase images must have one native size; {path.name} is {size}, "
                f"expected {expected_size}."
            )
        if mode not in {"1", "L", "I", "I;16"}:
            raise ValueError(
                f"{path.name} uses mode {mode!r}; SLM phase images must be single-channel grayscale."
            )
    if expected_size is None:
        raise RuntimeError("No image was selected.")
    return expected_size


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


def print_plan(
    paths: list[Path],
    monitor: Monitor,
    image_size: tuple[int, int],
    interval_seconds: float,
) -> None:
    print(
        f"Native phase size: {image_size[0]}x{image_size[1]}; "
        f"SDK display region: ({monitor.left}, {monitor.top}, "
        f"{monitor.width}, {monitor.height})"
    )
    print("IsStretch=False; images will be centred without resizing.")
    print(f"Images: {len(paths)} | interval: {interval_seconds:g} s")
    for index, path in enumerate(paths, start=1):
        print(f"  {index:04d}: {path.name}")


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    enable_native_dpi()
    monitors = enumerate_monitors()
    if args.list_monitors:
        print_monitors(monitors)
        return
    monitor = select_monitor(monitors, args.monitor)
    print_monitors(monitors, selected_index=monitor.index)

    input_path = resolve_path(args.input)
    paths = collect_images(input_path, args.start_id, args.limit)
    image_size = validate_images(paths, monitor)
    print_plan(paths, monitor, image_size, args.interval)
    print("Reminder: Windows main display and SLM scaling must both be 100%.")
    if args.dry_run:
        print("Dry run complete; SDK was not loaded and no image was sent to the SLM.")
        return

    sdk_dir = resolve_path(args.sdk_dir)
    interval_ms = max(1, round(args.interval * 1000.0))
    sdk = ZhongkeTimeoutSDK(sdk_dir)
    try:
        sdk.open()
        print("SLM window opened. Press Ctrl+C in this terminal to stop.")
        while True:
            for index, path in enumerate(paths, start=1):
                print(f"Displaying [{index}/{len(paths)}] {path.name}", flush=True)
                sdk.show(path, monitor, interval_ms)
            if not args.loop:
                break
        if not args.close_after:
            print("Final image remains on the SLM. Press Enter to close the SLM window.")
            input()
    except (KeyboardInterrupt, EOFError):
        print("Stopping SLM playback.")
    finally:
        sdk.close()
        print("SLM window closed.")


if __name__ == "__main__":
    main()
