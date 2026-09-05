"""Generate physically registered DMD/SLM alignment patterns.

The three devices do not have the same pixel pitch. Their patterns therefore
use different pixel shapes while covering the same DMD aperture and containing
the same number of physical checkerboard cells. SLM sizes are derived only
from ``models.SLM.physical_defaults`` hardware pitches; empirical numerical
sampling pitches are not used to size hardware images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.SLM.physical_defaults import (
    DEFAULT_DMD_PIXEL_PITCH,
    DEFAULT_DMD_RESOLUTION,
    DEFAULT_SLM_LAYER_PROFILES,
    SLM_PROFILES,
    active_pixel_shape,
    aperture_size,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Optical_yolo_detect" / "DMD_SLM_checkerborad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DMD, SLM1, and SLM2 checkerboards registered to the same "
            "physical aperture."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for alignment PNGs and metadata.",
    )
    parser.add_argument(
        "--blocks-y",
        type=int,
        default=10,
        help="Checkerboard cells across the physical aperture height (default: 10).",
    )
    parser.add_argument(
        "--blocks-x",
        type=int,
        default=10,
        help="Checkerboard cells across the physical aperture width (default: 10).",
    )
    parser.add_argument(
        "--slm-phase-low",
        type=int,
        default=0,
        help="8-bit gray value for the low SLM phase state (default: 0).",
    )
    parser.add_argument(
        "--slm-phase-high",
        type=int,
        default=128,
        help="8-bit high phase gray (default: 128, nominal pi for a linear LUT).",
    )
    return parser.parse_args()


def resolve_output(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.blocks_y < 2 or args.blocks_x < 2:
        raise ValueError("--blocks-y and --blocks-x must both be at least 2.")
    for name in ("slm_phase_low", "slm_phase_high"):
        value = getattr(args, name)
        if not 0 <= value <= 255:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 255.")
    if args.slm_phase_low == args.slm_phase_high:
        raise ValueError("--slm-phase-low and --slm-phase-high must be different.")


def physical_checkerboard(
    shape_hw: tuple[int, int],
    blocks_hw: tuple[int, int],
    low: int,
    high: int,
) -> np.ndarray:
    """Rasterize equal physical cells on an arbitrary hardware pixel grid.

    Cell membership is evaluated at each hardware pixel centre. When the
    device shape is not divisible by the number of cells, adjacent cells differ
    by at most one pixel instead of cropping the aperture.
    """
    height, width = (int(value) for value in shape_hw)
    blocks_y, blocks_x = (int(value) for value in blocks_hw)
    if height < blocks_y or width < blocks_x:
        raise ValueError(
            f"Checkerboard {blocks_y}x{blocks_x} does not fit hardware shape {height}x{width}."
        )
    row_cells = np.floor(
        (np.arange(height, dtype=np.float64) + 0.5) * blocks_y / height
    ).astype(np.int32)
    col_cells = np.floor(
        (np.arange(width, dtype=np.float64) + 0.5) * blocks_x / width
    ).astype(np.int32)
    parity = (row_cells[:, None] + col_cells[None, :]) % 2
    return np.where(parity == 0, high, low).astype(np.uint8)


def uniform_image(shape_hw: tuple[int, int], value: int) -> np.ndarray:
    return np.full(tuple(int(item) for item in shape_hw), value, dtype=np.uint8)


def cell_pixel_widths(pixel_count: int, block_count: int) -> list[int]:
    labels = np.floor(
        (np.arange(pixel_count, dtype=np.float64) + 0.5) * block_count / pixel_count
    ).astype(np.int32)
    return np.bincount(labels, minlength=block_count).astype(int).tolist()


def save_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save alignment image: {path}")


def device_metadata(
    name: str,
    shape_hw: tuple[int, int],
    pitch_m: float,
    blocks_hw: tuple[int, int],
    files: dict[str, str],
    profile: str | None = None,
    effective_sampling_pitch_m: float | None = None,
) -> dict:
    height, width = shape_hw
    blocks_y, blocks_x = blocks_hw
    result = {
        "device": name,
        "profile": profile,
        "active_shape_hw": [height, width],
        "hardware_pixel_pitch_m": pitch_m,
        "active_aperture_hw_m": [height * pitch_m, width * pitch_m],
        "checkerboard_cells_hw": [blocks_y, blocks_x],
        "cell_heights_pixels": cell_pixel_widths(height, blocks_y),
        "cell_widths_pixels": cell_pixel_widths(width, blocks_x),
        "files": files,
    }
    if effective_sampling_pitch_m is not None:
        result["effective_sampling_pitch_m_not_used_for_hardware_raster"] = effective_sampling_pitch_m
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = resolve_output(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dmd_shape = tuple(int(value) for value in DEFAULT_DMD_RESOLUTION)
    dmd_aperture = aperture_size(dmd_shape, DEFAULT_DMD_PIXEL_PITCH)
    blocks_hw = (args.blocks_y, args.blocks_x)

    slm_layers = []
    for layer_index in sorted(DEFAULT_SLM_LAYER_PROFILES):
        profile_name = DEFAULT_SLM_LAYER_PROFILES[layer_index]
        profile = SLM_PROFILES[profile_name]
        shape_hw = active_pixel_shape(profile_name, dmd_aperture)
        slm_layers.append((layer_index, profile_name, profile, shape_hw))

    dmd_files = {
        "checkerboard": "dmd_checkerboard.png",
        "white": "dmd_white.png",
        "black": "dmd_black.png",
    }
    save_png(
        output_dir / dmd_files["checkerboard"],
        physical_checkerboard(dmd_shape, blocks_hw, low=0, high=255),
    )
    save_png(output_dir / dmd_files["white"], uniform_image(dmd_shape, 255))
    save_png(output_dir / dmd_files["black"], uniform_image(dmd_shape, 0))

    devices = [
        device_metadata(
            "DMD",
            dmd_shape,
            DEFAULT_DMD_PIXEL_PITCH,
            blocks_hw,
            dmd_files,
        )
    ]
    for layer_index, profile_name, profile, shape_hw in slm_layers:
        prefix = f"slm{layer_index}"
        files = {
            "phase_checkerboard": f"{prefix}_phase_checkerboard.png",
            "uniform_phase": f"{prefix}_uniform_phase.png",
        }
        save_png(
            output_dir / files["phase_checkerboard"],
            physical_checkerboard(
                shape_hw,
                blocks_hw,
                low=args.slm_phase_low,
                high=args.slm_phase_high,
            ),
        )
        save_png(
            output_dir / files["uniform_phase"],
            uniform_image(shape_hw, args.slm_phase_low),
        )
        devices.append(
            device_metadata(
                f"SLM{layer_index}",
                shape_hw,
                float(profile["hardware_pixel_pitch"]),
                blocks_hw,
                files,
                profile=profile_name,
                effective_sampling_pitch_m=float(profile["effective_sampling_pitch"]),
            )
        )

    metadata = {
        "reference_aperture_hw_m": list(dmd_aperture),
        "reference_aperture_hw_mm": [value * 1e3 for value in dmd_aperture],
        "checkerboard_cells_hw": list(blocks_hw),
        "physical_cell_size_hw_m": [
            dmd_aperture[0] / args.blocks_y,
            dmd_aperture[1] / args.blocks_x,
        ],
        "slm_phase_gray": {
            "low": args.slm_phase_low,
            "high": args.slm_phase_high,
            "warning": (
                "Gray 128 is pi only for an ideal linear 0..2pi response; "
                "use the calibrated LUT value on hardware."
            ),
        },
        "placement": (
            "These SLM PNGs are active-region rasters. Display them at 1:1 hardware pixels and "
            "centre the active region on the physical panel; do not stretch them to the panel resolution."
        ),
        "devices": devices,
    }
    metadata_path = output_dir / "alignment_geometry.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Reference DMD aperture: {dmd_aperture[0] * 1e3:.3f} x "
        f"{dmd_aperture[1] * 1e3:.3f} mm"
    )
    print(
        f"DMD:  {dmd_shape[0]} x {dmd_shape[1]} pixels @ "
        f"{DEFAULT_DMD_PIXEL_PITCH * 1e6:.1f} um"
    )
    for layer_index, profile_name, profile, shape_hw in slm_layers:
        print(
            f"SLM{layer_index}: {shape_hw[0]} x {shape_hw[1]} active pixels @ "
            f"{float(profile['hardware_pixel_pitch']) * 1e6:.1f} um ({profile_name})"
        )
    print(f"Physical checkerboard: {args.blocks_y} x {args.blocks_x} cells")
    print(f"Saved alignment patterns to: {output_dir}")
    print(f"Geometry metadata: {metadata_path}")
    legacy_files = [
        output_dir / "slm_phase_checkerboard.png",
        output_dir / "slm_uniform_phase.png",
    ]
    if any(path.exists() for path in legacy_files):
        print(
            "WARNING: old ambiguous 640x640 SLM files still exist; "
            "do not load them onto either current SLM."
        )


if __name__ == "__main__":
    main()
