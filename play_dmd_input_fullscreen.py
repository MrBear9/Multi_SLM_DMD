"""Play numbered DMD input images fullscreen without changing their pixels.

Each source image is placed at its native pixel size in the centre of a black
fullscreen canvas.  No resize, stretch, interpolation, letterbox, contrast
adjustment, or colour conversion is applied to the image data.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import tkinter as tk
from ctypes import wintypes
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "output" / "Tv2_dmd640_scratch" / "hardware_export_100" / "input"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Monitor(NamedTuple):
    index: int
    left: int
    top: int
    width: int
    height: int


def enable_native_dpi() -> None:
    """Prevent Windows DPI scaling from changing the pixel-to-screen mapping."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def enumerate_monitors() -> list[Monitor]:
    if os.name != "nt":
        root = tk.Tk()
        root.withdraw()
        monitors = [Monitor(0, 0, 0, root.winfo_screenwidth(), root.winfo_screenheight())]
        root.destroy()
        return monitors

    rectangles: list[tuple[int, int, int, int]] = []
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.RECT),
        ctypes.c_void_p,
    )

    def collect_monitor(
        _monitor: int,
        _device_context: int,
        rectangle_pointer: ctypes.POINTER(wintypes.RECT),
        _data: int,
    ) -> bool:
        rectangle = rectangle_pointer.contents
        rectangles.append(
            (
                int(rectangle.left),
                int(rectangle.top),
                int(rectangle.right - rectangle.left),
                int(rectangle.bottom - rectangle.top),
            )
        )
        return True

    callback = callback_type(collect_monitor)
    if not ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0):
        raise RuntimeError("Windows monitor enumeration failed.")
    if not rectangles:
        raise RuntimeError("No display monitor was detected.")
    return [Monitor(index, *rectangle) for index, rectangle in enumerate(rectangles)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fullscreen DMD image sequence player with native-size, unscaled pixels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing DMD input images.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds each image remains visible while playing (default: 5).",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Zero-based target monitor index reported by --list-monitors (default: 1).",
    )
    parser.add_argument(
        "--start-id",
        default=None,
        help="Optional first filename stem, for example 0021.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Return to the first image after the final image instead of pausing.",
    )
    parser.add_argument(
        "--list-monitors",
        action="store_true",
        help="Print available monitor indices and resolutions, then exit.",
    )
    parser.add_argument(
        "--show-cursor",
        action="store_true",
        help="Keep the mouse cursor visible over the DMD window.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def natural_image_key(path: Path) -> tuple[int, int | str, str]:
    if path.stem.isdigit():
        return 0, int(path.stem), path.name.casefold()
    return 1, path.stem.casefold(), path.name.casefold()


def load_image_paths(input_dir: Path) -> list[Path]:
    paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_image_key,
    )
    if not paths:
        raise RuntimeError(f"No supported images found in: {input_dir}")
    return paths


class DMDPlayer:
    def __init__(
        self,
        root: tk.Tk,
        monitor: Monitor,
        image_paths: list[Path],
        start_index: int,
        interval_seconds: float,
        loop: bool,
        show_cursor: bool,
    ) -> None:
        self.root = root
        self.monitor = monitor
        self.image_paths = image_paths
        self.index = start_index
        self.interval_ms = max(1, round(interval_seconds * 1000.0))
        self.loop = loop
        self.playing = False
        self.after_id: str | None = None
        self.photo: ImageTk.PhotoImage | None = None

        root.configure(background="black", cursor="" if show_cursor else "none")
        root.overrideredirect(True)
        root.geometry(
            f"{monitor.width}x{monitor.height}{monitor.left:+d}{monitor.top:+d}"
        )
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
        root.bind("<F11>", self.toggle_topmost)

        self.show_current()
        root.after(100, self.focus_window)

    def focus_window(self) -> None:
        self.root.lift()
        self.root.focus_force()

    def cancel_timer(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def schedule_next(self) -> None:
        self.cancel_timer()
        if self.playing:
            self.after_id = self.root.after(self.interval_ms, self.advance_after_interval)

    def read_native_image(self, path: Path) -> Image.Image:
        with Image.open(path) as source:
            source.load()
            return source.copy()

    def show_current(self) -> None:
        path = self.image_paths[self.index]
        image = self.read_native_image(path)
        if image.width > self.monitor.width or image.height > self.monitor.height:
            raise RuntimeError(
                f"Image {path.name} is {image.width}x{image.height}, larger than monitor "
                f"{self.monitor.width}x{self.monitor.height}. Refusing to resize or crop it."
            )
        self.photo = ImageTk.PhotoImage(image=image, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.monitor.width // 2,
            self.monitor.height // 2,
            image=self.photo,
            anchor="center",
        )
        state = "PLAY" if self.playing else "PAUSE"
        self.root.title(
            f"DMD native pixels | {path.name} | {self.index + 1}/{len(self.image_paths)} | {state}"
        )
        print(
            f"[{self.index + 1}/{len(self.image_paths)}] {path.name} | "
            f"{image.width}x{image.height} native pixels | {state}",
            flush=True,
        )

    def toggle_playback(self, _event: tk.Event | None = None) -> str:
        self.playing = not self.playing
        self.show_current()
        self.schedule_next()
        return "break"

    def advance_after_interval(self) -> None:
        self.after_id = None
        if not self.playing:
            return
        if self.index + 1 >= len(self.image_paths):
            if self.loop:
                self.index = 0
            else:
                self.playing = False
                self.show_current()
                print("Reached final image; playback paused.", flush=True)
                return
        else:
            self.index += 1
        self.show_current()
        self.schedule_next()

    def move(self, delta: int) -> None:
        self.index = min(max(self.index + delta, 0), len(self.image_paths) - 1)
        self.show_current()
        self.schedule_next()

    def next_image(self, _event: tk.Event | None = None) -> str:
        self.move(1)
        return "break"

    def previous_image(self, _event: tk.Event | None = None) -> str:
        self.move(-1)
        return "break"

    def first_image(self, _event: tk.Event | None = None) -> str:
        self.index = 0
        self.show_current()
        self.schedule_next()
        return "break"

    def toggle_topmost(self, _event: tk.Event | None = None) -> str:
        current = bool(self.root.attributes("-topmost"))
        self.root.attributes("-topmost", not current)
        return "break"

    def close(self, _event: tk.Event | None = None) -> str:
        self.cancel_timer()
        self.root.destroy()
        return "break"


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive.")

    enable_native_dpi()
    monitors = enumerate_monitors()
    print("Available monitors:")
    for monitor in monitors:
        selected = "  <-- selected" if monitor.index == args.monitor else ""
        print(
            f"  Monitor {monitor.index}: {monitor.width}x{monitor.height} "
            f"at ({monitor.left}, {monitor.top}){selected}"
        )
    if args.list_monitors:
        return
    if not 0 <= args.monitor < len(monitors):
        raise ValueError(
            f"--monitor {args.monitor} is unavailable; choose 0..{len(monitors) - 1} "
            "or run with --list-monitors."
        )

    input_dir = resolve_path(args.input)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"DMD input directory not found: {input_dir}")
    image_paths = load_image_paths(input_dir)
    start_index = 0
    if args.start_id is not None:
        start_index = next(
            (index for index, path in enumerate(image_paths) if path.stem == args.start_id),
            -1,
        )
        if start_index < 0:
            raise ValueError(f"--start-id {args.start_id!r} was not found in: {input_dir}")

    monitor = monitors[args.monitor]
    print(f"Input directory: {input_dir}")
    print(f"Images: {len(image_paths)}")
    print(
        f"Using monitor {monitor.index}: {monitor.width}x{monitor.height} "
        f"at ({monitor.left}, {monitor.top})"
    )
    print(f"Interval: {args.interval:g} seconds")
    print("SPACE: play/pause | LEFT/RIGHT: previous/next | HOME: first | ESC/Q: quit")

    root = tk.Tk()
    DMDPlayer(
        root=root,
        monitor=monitor,
        image_paths=image_paths,
        start_index=start_index,
        interval_seconds=args.interval,
        loop=args.loop,
        show_cursor=args.show_cursor,
    )
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
