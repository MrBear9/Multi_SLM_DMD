"""Reusable zkwx_slm image and device operations."""

from __future__ import annotations
from pathlib import Path
from PIL import Image
from .sdk import (
    DEFAULT_SDK_DIR,
    IMAGE_EXTENSIONS,
    Monitor,
    enable_native_dpi,
    enumerate_monitors,
    ZhongkeTimeoutSDK,
)
from ...utils.images import natural_image_key
from ...utils.image_transforms import ImageTransform


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

def select_monitor(monitors: list[Monitor], requested_index: int | None,
                   *, occupied=()) -> Monitor:
    if requested_index is not None:
        match = next((monitor for monitor in monitors if monitor.index == requested_index), None)
        if match is None:
            raise ValueError(
                f"--monitor {requested_index} is unavailable; choose from {[m.index for m in monitors]}."
            )
        return match
    candidates = [monitor for monitor in monitors if not monitor.primary
                  and (monitor.left, monitor.top, monitor.width, monitor.height) not in occupied]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            'SLM1 自动选择存在多个未占用的副屏，请指定 Windows monitor 编号：'
            + '; '.join(f'monitor {m.index}: {m.width}x{m.height} @ ({m.left}, {m.top})' for m in candidates)
        )
    raise RuntimeError(
        'SLM1 没有可自动选择的未占用副屏。请检查扩展显示连接，或按设备枚举结果手动指定 Windows monitor 编号。'
    )


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

def validate_images(paths: list[Path], monitor: Monitor,
                    transform: ImageTransform | None = None) -> tuple[int, int]:
    transform = transform or ImageTransform()
    expected_size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            image.verify()
            size = (int(image.width), int(image.height))
            mode = image.mode
        output_size = transform.output_size(size)
        if output_size[0] > monitor.width or output_size[1] > monitor.height:
            raise ValueError(
                f"{path.name} after rotation is {output_size[0]}x{output_size[1]}, larger than selected SLM region "
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
    return transform.output_size(expected_size)


def canvas_origin(
    image_size: tuple[int, int],
    monitor: Monitor,
    offset_x: int = 0,
    offset_y: int = 0,
) -> tuple[int, int]:
    """Return centred native-pixel placement; shifted pixels may leave the panel."""
    width, height = image_size
    left = (monitor.width - width) // 2 + offset_x
    top = (monitor.height - height) // 2 + offset_y
    if width <= 0 or height <= 0 or width > monitor.width or height > monitor.height:
        raise ValueError("The SLM1 phase image size exceeds the display.")
    return left, top


def prepare_canvas(
    path: Path,
    monitor: Monitor,
    offset_x: int = 0,
    offset_y: int = 0,
    transform: ImageTransform | None = None,
) -> Image.Image:
    """Keep source grayscale values on a full-size black panel without resampling.

    The caller owns the returned image. Keeping the whole panel covered also
    clears pixels occupied by a previous placement when the offset changes.
    Paste clips shifted pixels at panel edges; exposed pixels remain zero gray.
    """
    with Image.open(path) as image:
        image.load()
        if image.mode not in {"1", "L", "I", "I;16"}:
            raise ValueError(f"SLM1 phase image must be single-channel grayscale: {path}")
        transform = transform or ImageTransform()
        origin = canvas_origin(transform.output_size(image.size), monitor, offset_x, offset_y)
        with transform.apply_pil(image) as oriented:
            canvas = Image.new(image.mode, (monitor.width, monitor.height), color=0)
            canvas.paste(oriented, origin)
            return canvas
