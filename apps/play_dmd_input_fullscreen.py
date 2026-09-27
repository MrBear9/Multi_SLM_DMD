"""Play numbered DMD input images fullscreen without changing their pixels.

Each source image is placed at its native pixel size in the centre of a black
fullscreen canvas.  No resize, stretch, interpolation, letterbox, contrast
adjustment, or colour conversion is applied to the image data.
"""

from __future__ import annotations
import argparse
import sys
import tkinter as tk
from pathlib import Path
from ..devices.dmd_display.controller import (
    DEFAULT_INPUT_DIR,
    enable_native_dpi,
    enumerate_monitors,
    load_image_paths,
    DMDPlayer,
)
from ..utils.paths import resolve_path


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
