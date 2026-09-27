"""Display one or more phase images on a Zhongke Weixing SLM via SDK 2.1.

The implementation follows the vendor's Timeout API for images sent from a
Python loop. Images are always shown with ``IsStretch=False`` so their native
pixels are centred in the selected SLM display region without resizing.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from ..devices.zkwx_slm.sdk import (
    DEFAULT_INPUT,
    DEFAULT_SDK_DIR,
    Monitor,
    enable_native_dpi,
    enumerate_monitors,
    ZhongkeTimeoutSDK,
)
from ..utils.paths import resolve_path
from ..devices.zkwx_slm.controller import (
    print_monitors,
    select_monitor,
    collect_images,
    validate_images,
)


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
