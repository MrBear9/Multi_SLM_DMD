"""Synchronously display paired phase images on both project SLMs.

SLM1 is driven by the Zhongke Weixing SDK and reads ``tools/slm1`` by
default. SLM2 is the 8-bit MagicHolo HDSLM45R and reads ``tools/slm2`` by
default. Files are paired by filename stem, displayed without interpolation,
and advanced together at one shared interval.  By default, both SDK windows
are opened first and the first phase pair is not sent until Space is pressed
in the terminal.
"""

from __future__ import annotations

import argparse
import msvcrt
import time
from pathlib import Path

import play_magicholo_slm2 as magicholo
import play_zkwx_slm1 as zkwx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZKWX_INPUT = PROJECT_ROOT / "tools" / "slm1"
DEFAULT_MAGICHolo_INPUT = PROJECT_ROOT / "tools" / "slm2"
ZKWX_DISPATCH_TIMEOUT_MS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronously display paired phase images on Zhongke Weixing SLM1 "
            "and the 8-bit MagicHolo HDSLM45R SLM2."
        )
    )
    parser.add_argument(
        "--zkwx-input",
        type=Path,
        default=DEFAULT_ZKWX_INPUT,
        help="SLM1 image or flat image directory (default: tools/slm1).",
    )
    parser.add_argument(
        "--magicholo-input",
        type=Path,
        default=DEFAULT_MAGICHolo_INPUT,
        help="SLM2 8-bit BMP/PNG or flat directory (default: tools/slm2).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds before both SLMs advance to the next pair (default: 5).",
    )
    parser.add_argument(
        "--zkwx-monitor",
        type=int,
        default=None,
        help=(
            "Windows Monitor index for Zhongke Weixing SLM1. By default, use "
            "the unique non-primary monitor that is not the HDSLM45R."
        ),
    )
    parser.add_argument(
        "--magicholo-display",
        type=int,
        default=None,
        help=(
            "HDSLM SDK display ID for MagicHolo SLM2. By default, identify the "
            "display whose SDK name is HDSLM45R."
        ),
    )
    parser.add_argument(
        "--start-id",
        default=None,
        help="Start both directories at this filename stem, for example 0021.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Display only the first N paired images.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the paired sequence until Ctrl+C.",
    )
    parser.add_argument(
        "--close-after",
        action="store_true",
        help=(
            "Close both SLM windows after the final pair has been shown for "
            "--interval seconds; otherwise hold the final pair until Enter."
        ),
    )
    parser.add_argument(
        "--magicholo-offset-x",
        type=int,
        default=0,
        help="Move the SLM2 phase map right (+) or left (-), in HDSLM45R pixels.",
    )
    parser.add_argument(
        "--magicholo-offset-y",
        type=int,
        default=0,
        help="Move the SLM2 phase map down (+) or up (-), in HDSLM45R pixels.",
    )
    parser.add_argument(
        "--magicholo-background",
        type=int,
        default=0,
        help="8-bit background value around the SLM2 phase map (default: 0).",
    )
    parser.add_argument(
        "--zkwx-sdk-dir",
        type=Path,
        default=zkwx.DEFAULT_SDK_DIR,
        help="Directory containing the Zhongke Weixing SecondDll.dll.",
    )
    parser.add_argument(
        "--magicholo-sdk-dir",
        type=Path,
        default=magicholo.DEFAULT_SDK_DIR,
        help="Directory containing the MagicHolo HDSLMFunc.dll.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List both SDK/display index systems and the automatic choices, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect devices and validate both image sequences without opening either SLM.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start immediately after opening both SDK windows instead of waiting for Space.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def same_monitor(left: int, top: int, width: int, height: int, display: magicholo.SDKDisplay) -> bool:
    return (left, top, width, height) == (
        display.left,
        display.top,
        display.width,
        display.height,
    )


def select_magicholo_display(
    displays: list[magicholo.SDKDisplay], requested: int | None
) -> tuple[magicholo.SDKDisplay, str]:
    if requested is not None:
        display = magicholo.select_display(displays, requested)
        return display, f"manually selected with --magicholo-display {requested}"

    named = [
        display
        for display in displays
        if display.name.strip().casefold() == "hdslm45r"
    ]
    if len(named) == 1:
        return named[0], "automatically identified by SDK name 'HDSLM45R'"
    if len(named) > 1:
        raise RuntimeError(
            "More than one SDK display is named HDSLM45R; pass --magicholo-display."
        )

    partial = [
        display for display in displays if "hdslm45r" in display.name.casefold()
    ]
    if len(partial) == 1:
        return partial[0], "automatically identified because its SDK name contains 'HDSLM45R'"
    if len(partial) > 1:
        raise RuntimeError(
            "More than one SDK display name contains HDSLM45R; pass --magicholo-display."
        )

    try:
        display = magicholo.select_display(displays, None)
    except RuntimeError as error:
        raise RuntimeError(
            "The MagicHolo SDK did not report a unique HDSLM45R name or display; "
            "use --list-devices and pass --magicholo-display explicitly."
        ) from error
    return display, "fallback: unique non-primary 1920x1080 display"


def select_zkwx_monitor(
    monitors: list[zkwx.Monitor],
    magicholo_display: magicholo.SDKDisplay,
    requested: int | None,
) -> tuple[zkwx.Monitor, str]:
    if requested is not None:
        monitor = zkwx.select_monitor(monitors, requested)
        if same_monitor(
            monitor.left,
            monitor.top,
            monitor.width,
            monitor.height,
            magicholo_display,
        ):
            raise ValueError(
                "--zkwx-monitor points to the same physical display as "
                "--magicholo-display; choose the Zhongke Weixing screen."
            )
        return monitor, f"manually selected with --zkwx-monitor {requested}"

    candidates = [
        monitor
        for monitor in monitors
        if not monitor.primary
        and not same_monitor(
            monitor.left,
            monitor.top,
            monitor.width,
            monitor.height,
            magicholo_display,
        )
    ]
    if len(candidates) == 1:
        return candidates[0], "automatically identified as the remaining non-primary monitor"
    if not candidates:
        raise RuntimeError(
            "No non-primary monitor remains after excluding the HDSLM45R; "
            "connect Zhongke Weixing SLM1 in extended-display mode."
        )
    raise RuntimeError(
        "Several non-primary monitors remain after excluding the HDSLM45R; "
        "use --list-devices and pass --zkwx-monitor explicitly."
    )


def validate_pairs(zkwx_paths: list[Path], magicholo_paths: list[Path]) -> list[tuple[Path, Path]]:
    zkwx_stems = [path.stem for path in zkwx_paths]
    magicholo_stems = [path.stem for path in magicholo_paths]
    if zkwx_stems != magicholo_stems:
        zkwx_only = sorted(set(zkwx_stems) - set(magicholo_stems))
        magicholo_only = sorted(set(magicholo_stems) - set(zkwx_stems))
        order_mismatch = not zkwx_only and not magicholo_only
        detail_parts = []
        if zkwx_only:
            detail_parts.append(f"only in SLM1: {', '.join(zkwx_only[:10])}")
        if magicholo_only:
            detail_parts.append(f"only in SLM2: {', '.join(magicholo_only[:10])}")
        if order_mismatch:
            detail_parts.append("the two filename orders differ")
        details = "; ".join(detail_parts)
        raise ValueError(
            "SLM1 and SLM2 images must have matching filename stems so the optical "
            f"phases cannot become misaligned ({details})."
        )
    return list(zip(zkwx_paths, magicholo_paths))


def print_selection(
    monitors: list[zkwx.Monitor],
    displays: list[magicholo.SDKDisplay],
    zkwx_monitor: zkwx.Monitor,
    magicholo_display: magicholo.SDKDisplay,
    zkwx_reason: str,
    magicholo_reason: str,
) -> None:
    zkwx.print_monitors(monitors, selected_index=zkwx_monitor.index)
    print(f"Zhongke Weixing selection: {zkwx_reason}.")
    print()
    magicholo.print_displays(displays, selected=magicholo_display.index)
    print(f"MagicHolo selection: {magicholo_reason}.")


def read_console_key() -> str | None:
    """Read one Windows console key without blocking; ignore navigation keys."""
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key in {"\x00", "\xe0"}:
        if msvcrt.kbhit():
            msvcrt.getwch()
        return None
    return key


def wait_for_space(message: str) -> None:
    """Keep both SDK windows responsive until Space; Esc stops playback."""
    print(message, flush=True)
    while True:
        magicholo.pump_window_messages()
        key = read_console_key()
        if key == " ":
            return
        if key == "\x1b":
            raise KeyboardInterrupt
        time.sleep(0.01)


def wait_interval_with_controls(seconds: float) -> None:
    """Wait one shared interval while Space toggles pause/resume."""
    deadline = time.monotonic() + seconds
    while True:
        magicholo.pump_window_messages()
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            return
        key = read_console_key()
        if key == " ":
            wait_for_space(
                "Playback paused. Press Space in this terminal to continue; Esc stops."
            )
            print("Playback resumed.", flush=True)
            deadline = time.monotonic() + remaining
        elif key == "\x1b":
            raise KeyboardInterrupt
        time.sleep(min(0.01, remaining))


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if not 0 <= args.magicholo_background <= 255:
        raise ValueError("--magicholo-background must be between 0 and 255.")
    if args.loop and args.close_after:
        raise ValueError("--loop and --close-after cannot be used together.")

    zkwx.enable_native_dpi()
    monitors = zkwx.enumerate_monitors()

    magicholo_sdk_dir = resolve_path(args.magicholo_sdk_dir)
    magicholo_sdk = magicholo.HDSLM8BitSDK(magicholo_sdk_dir)
    zkwx_sdk: zkwx.ZhongkeTimeoutSDK | None = None
    try:
        displays = magicholo_sdk.list_displays()
        magicholo_display, magicholo_reason = select_magicholo_display(
            displays, args.magicholo_display
        )
        zkwx_monitor, zkwx_reason = select_zkwx_monitor(
            monitors, magicholo_display, args.zkwx_monitor
        )
        print_selection(
            monitors,
            displays,
            zkwx_monitor,
            magicholo_display,
            zkwx_reason,
            magicholo_reason,
        )

        if args.list_devices:
            print(
                "Note: Zhongke uses Windows Monitor indices; MagicHolo uses HDSLM "
                "SDK Display IDs. The numbers can differ even for the same screen."
            )
            return

        zkwx_paths = zkwx.collect_images(
            resolve_path(args.zkwx_input), args.start_id, args.limit
        )
        magicholo_paths = magicholo.collect_images(
            resolve_path(args.magicholo_input), args.start_id, args.limit
        )
        pairs = validate_pairs(zkwx_paths, magicholo_paths)

        zkwx_size = zkwx.validate_images(zkwx_paths, zkwx_monitor)
        magicholo_size = magicholo.validate_images(
            magicholo_paths, magicholo_display
        )
        magicholo_origin = magicholo.canvas_origin(
            magicholo_size,
            magicholo_display,
            args.magicholo_offset_x,
            args.magicholo_offset_y,
        )

        print()
        print(
            f"SLM1 native phase: {zkwx_size[0]}x{zkwx_size[1]}, centred by the "
            "Zhongke SDK without resizing."
        )
        print(
            f"SLM2 native phase: {magicholo_size[0]}x{magicholo_size[1]}, placed "
            f"at {magicholo_origin} on a {magicholo_display.width}x"
            f"{magicholo_display.height} 8-bit canvas without resizing."
        )
        print(
            f"Paired images: {len(pairs)} | shared interval: {args.interval:g} s | "
            f"first pair: {pairs[0][0].stem} | last pair: {pairs[-1][0].stem}"
        )
        print("Reminder: all three Windows displays must use native resolution and 100% scaling.")
        if args.dry_run:
            print("Dry run complete; neither SLM window was opened and no image was sent.")
            return

        zkwx_sdk_dir = resolve_path(args.zkwx_sdk_dir)
        zkwx_sdk = zkwx.ZhongkeTimeoutSDK(zkwx_sdk_dir)
        zkwx_sdk.open()
        magicholo_sdk.open(magicholo_display.index)
        print("Both SLM windows opened. Press Ctrl+C or Esc in this terminal to stop.")
        if args.auto_start:
            print("Automatic start enabled; sending the first phase pair now.")
        else:
            wait_for_space(
                "Ready: press Space in this terminal to display the first phase pair."
            )
            print("Playback started.", flush=True)

        while True:
            for index, (zkwx_path, magicholo_path) in enumerate(pairs, start=1):
                magicholo_frame = magicholo.prepare_canvas(
                    magicholo_path,
                    magicholo_display,
                    magicholo_origin,
                    args.magicholo_background,
                )
                pair_started = time.monotonic()
                print(
                    f"Displaying pair [{index}/{len(pairs)}] "
                    f"SLM1={zkwx_path.name} | SLM2={magicholo_path.name}",
                    flush=True,
                )
                magicholo_sdk.show_8bit(
                    magicholo_display.index,
                    magicholo_display.width,
                    magicholo_display.height,
                    magicholo_frame,
                )
                # The Zhongke Timeout API blocks for uMill.  Use only one
                # nominal video frame here, then let the shared clock below
                # define the requested pair interval for both SLMs.
                zkwx_sdk.show(
                    zkwx_path,
                    zkwx_monitor,
                    ZKWX_DISPATCH_TIMEOUT_MS,
                )

                is_final = index == len(pairs)
                if args.loop or args.close_after or not is_final:
                    elapsed = time.monotonic() - pair_started
                    wait_interval_with_controls(max(0.0, args.interval - elapsed))
            if not args.loop:
                break

        if not args.close_after:
            print("Final pair remains displayed. Press Enter or Esc to close both windows.")
            magicholo.hold_until_enter()
    except KeyboardInterrupt:
        print("\nStopping synchronized SLM playback.")
    finally:
        magicholo_sdk.close()
        if zkwx_sdk is not None:
            zkwx_sdk.close()
        print("Both SLM SDKs closed.")


if __name__ == "__main__":
    main()
