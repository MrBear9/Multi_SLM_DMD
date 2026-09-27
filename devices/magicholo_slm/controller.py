"""Reusable magicholo_slm image and device operations."""

from __future__ import annotations
import ctypes
import time
from ctypes import wintypes
from pathlib import Path
from PIL import Image
from .sdk import (
    DEFAULT_SDK_DIR,
    IMAGE_EXTENSIONS,
    HDSLM45R_SIZE,
    PM_REMOVE,
    SDKDisplay,
    HDSLM8BitSDK,
)
from ...utils.images import natural_image_key


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
