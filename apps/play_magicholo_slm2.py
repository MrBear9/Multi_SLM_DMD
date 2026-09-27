"""Display 8-bit phase images on a MagicHolo HDSLM45R through its SDK.

The HDSLM45R is a 1920x1080, 8-bit phase SLM.  Images are never resized:
smaller phase maps (for example 768x768) are centred, pixel for pixel, on an
8-bit zero-valued canvas matching the selected display.  This preserves the
physical 4.5 um SLM pixel pitch used by the optical configuration.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from ..devices.magicholo_slm.sdk import (
    DEFAULT_INPUT,
    DEFAULT_SDK_DIR,
    HDSLM45R_SIZE,
    SDKDisplay,
    enable_native_dpi,
    HDSLM8BitSDK,
)
from ..utils.paths import resolve_path
from ..devices.magicholo_slm.controller import (
    collect_images,
    validate_images,
    canvas_origin,
    prepare_canvas,
    print_displays,
    select_display,
    wait_with_messages,
    hold_until_enter,
)


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
