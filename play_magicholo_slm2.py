"""Display 8-bit phase images on a MagicHolo HDSLM45R through its SDK.

The HDSLM45R is a 1920x1080, 8-bit phase SLM.  Images are never resized:
smaller phase maps (for example 768x768) are centred, pixel for pixel, on an
8-bit zero-valued canvas matching the selected display.  This preserves the
physical 4.5 um SLM pixel pitch used by the optical configuration.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import struct
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Display one 8-bit phase image or an ordered image directory on a "
            "MagicHolo HDSLM45R through the vendor HDSLM SDK."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="One 8-bit BMP/PNG or a flat directory (default: tools/slm2).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds per image during multi-image playback (default: 5).",
    )
    parser.add_argument(
        "--display",
        "--monitor",
        dest="display",
        type=int,
        default=None,
        help=(
            "HDSLM SDK display ID. By default, automatically select the only "
            "non-primary 1920x1080 display."
        ),
    )
    parser.add_argument(
        "--offset-x",
        type=int,
        default=0,
        help="Move the native phase image right (+) or left (-), in SLM pixels.",
    )
    parser.add_argument(
        "--offset-y",
        type=int,
        default=0,
        help="Move the native phase image down (+) or up (-), in SLM pixels.",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=0,
        help="8-bit canvas value outside a smaller phase image (default: 0).",
    )
    parser.add_argument(
        "--start-id",
        default=None,
        help="Start at this filename stem for directory input, for example 0021.",
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
        help="Repeat the image sequence until Ctrl+C.",
    )
    parser.add_argument(
        "--close-after",
        action="store_true",
        help=(
            "Close after the final image has been shown for --interval seconds; "
            "otherwise hold the final image until Enter."
        ),
    )
    parser.add_argument(
        "--sdk-dir",
        type=Path,
        default=DEFAULT_SDK_DIR,
        help="Directory containing the official HDSLMFunc.dll.",
    )
    parser.add_argument(
        "--list-displays",
        "--list-monitors",
        dest="list_displays",
        action="store_true",
        help="List HDSLM SDK display IDs, resolutions and positions, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load the SDK, enumerate displays and validate images, but do not "
            "open a display window or send image data."
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


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


def natural_image_key(path: Path) -> tuple[int, int | str, str]:
    if path.stem.isdigit():
        return 0, int(path.stem), path.name.casefold()
    return 1, path.stem.casefold(), path.name.casefold()


def collect_images(input_path: Path, start_id: str | None, limit: int | None) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {input_path}; use 8-bit BMP or PNG.")
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
            raise RuntimeError(f"No 8-bit BMP/PNG phase images found in: {input_path}")
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


def validate_images(paths: list[Path], display: SDKDisplay) -> tuple[int, int]:
    expected_size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            image.load()
            size = (int(image.width), int(image.height))
            mode = image.mode
        if mode != "L":
            raise ValueError(
                f"{path.name} uses Pillow mode {mode!r}; HDSLM45R input must be "
                "an 8-bit single-channel grayscale image (mode 'L')."
            )
        if size[0] > display.width or size[1] > display.height:
            raise ValueError(
                f"{path.name} is {size[0]}x{size[1]}, larger than display "
                f"{display.width}x{display.height}; resizing/cropping is disabled."
            )
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError(
                f"All phase images must have one native size; {path.name} is {size}, "
                f"expected {expected_size}."
            )
    if expected_size is None:
        raise RuntimeError("No image was selected.")
    return expected_size


def canvas_origin(
    image_size: tuple[int, int],
    display: SDKDisplay,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int]:
    left = (display.width - image_size[0]) // 2 + offset_x
    top = (display.height - image_size[1]) // 2 + offset_y
    if left < 0 or top < 0:
        raise ValueError("The requested offset moves the phase image outside the display.")
    if left + image_size[0] > display.width or top + image_size[1] > display.height:
        raise ValueError("The requested offset moves the phase image outside the display.")
    return left, top


def prepare_canvas(
    path: Path,
    display: SDKDisplay,
    origin: tuple[int, int],
    background: int,
) -> bytes:
    with Image.open(path) as image:
        image.load()
        if image.mode != "L":
            raise ValueError(f"Image changed or is not 8-bit grayscale: {path}")
        canvas = Image.new("L", (display.width, display.height), color=background)
        canvas.paste(image, origin)
        return canvas.tobytes()


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


def print_displays(displays: list[SDKDisplay], selected: int | None = None) -> None:
    print("Available HDSLM SDK displays:")
    for display in displays:
        flags = []
        if display.primary:
            flags.append("primary")
        if display.index == selected:
            flags.append("selected HDSLM45R")
        suffix = f"  <-- {', '.join(flags)}" if flags else ""
        print(
            f"  Display {display.index}: {display.name} | "
            f"{display.width}x{display.height} at ({display.left}, {display.top}){suffix}"
        )


def select_display(displays: list[SDKDisplay], requested: int | None) -> SDKDisplay:
    if requested is not None:
        match = next((display for display in displays if display.index == requested), None)
        if match is None:
            available = ", ".join(str(display.index) for display in displays)
            raise ValueError(f"--display {requested} is unavailable; choose one of: {available}.")
        return match

    native_candidates = [
        display
        for display in displays
        if not display.primary and (display.width, display.height) == HDSLM45R_SIZE
    ]
    if len(native_candidates) == 1:
        return native_candidates[0]
    if len(native_candidates) > 1:
        raise RuntimeError(
            "More than one non-primary 1920x1080 display was found; pass --display explicitly."
        )
    raise RuntimeError(
        "No unique non-primary 1920x1080 HDSLM45R display was found. Connect the SLM "
        "in Windows extended-display mode at native resolution and 100% scaling, then "
        "use --list-displays and select it with --display."
    )


def pump_window_messages() -> None:
    msg = wintypes.MSG()
    user32 = ctypes.windll.user32
    while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


def wait_with_messages(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        pump_window_messages()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.01, remaining))


def hold_until_enter() -> None:
    import msvcrt

    while True:
        pump_window_messages()
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in {"\r", "\n", "\x1b"}:
                return
        time.sleep(0.01)


def print_plan(
    paths: list[Path],
    display: SDKDisplay,
    image_size: tuple[int, int],
    origin: tuple[int, int],
    args: argparse.Namespace,
) -> None:
    print(
        f"8-bit native phase size: {image_size[0]}x{image_size[1]}; "
        f"HDSLM canvas: {display.width}x{display.height}"
    )
    print(
        f"Placement: ({origin[0]}, {origin[1]}); background gray: {args.background}. "
        "No resizing or interpolation will be applied."
    )
    print(f"Images: {len(paths)} | interval: {args.interval:g} s")
    for index, path in enumerate(paths, start=1):
        print(f"  {index:04d}: {path.name}")


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if not 0 <= args.background <= 255:
        raise ValueError("--background must be between 0 and 255.")
    if args.loop and args.close_after:
        raise ValueError("--loop and --close-after cannot be used together.")

    enable_native_dpi()
    sdk_dir = resolve_path(args.sdk_dir)
    sdk = HDSLM8BitSDK(sdk_dir)
    try:
        displays = sdk.list_displays()
        if args.list_displays:
            print_displays(displays)
            print(
                "Note: SDK display IDs are ordered by display name and may differ "
                "from another tool's Monitor indices."
            )
            return

        display = select_display(displays, args.display)
        print_displays(displays, selected=display.index)
        if (display.width, display.height) != HDSLM45R_SIZE:
            print(
                "Warning: selected display is not the HDSLM45R native 1920x1080 "
                "resolution; verify --display before sending images."
            )
        if display.primary:
            print("Warning: the selected display is the Windows primary display.")

        paths = collect_images(resolve_path(args.input), args.start_id, args.limit)
        image_size = validate_images(paths, display)
        origin = canvas_origin(image_size, display, args.offset_x, args.offset_y)
        print_plan(paths, display, image_size, origin, args)
        print("Reminder: use Windows extended-display mode, native resolution and 100% scaling.")
        if args.dry_run:
            print("Dry run complete; no HDSLM window was opened and no image was sent.")
            return

        sdk.open(display.index)
        print("HDSLM45R window opened. Press Ctrl+C in this terminal to stop.")
        while True:
            for index, path in enumerate(paths, start=1):
                print(f"Displaying [{index}/{len(paths)}] {path.name}", flush=True)
                frame = prepare_canvas(path, display, origin, args.background)
                sdk.show_8bit(display.index, display.width, display.height, frame)
                is_final = index == len(paths)
                if args.loop or args.close_after or not is_final:
                    wait_with_messages(args.interval)
            if not args.loop:
                break

        if not args.close_after:
            print("Final image remains on the SLM. Press Enter or Esc to close the window.")
            hold_until_enter()
    except KeyboardInterrupt:
        print("\nStopping HDSLM45R playback.")
    finally:
        sdk.close()
        print("HDSLM SDK closed.")


if __name__ == "__main__":
    main()
