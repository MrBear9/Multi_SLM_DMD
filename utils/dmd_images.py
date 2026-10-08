"""Shared DMD input conversion; never writes to or rescales source files.

Conversion happens before ImageTransform. Display adapters may save the returned
image to a temporary BMP: mode ``1`` produces a 1-bit BMP and mode ``L`` an
8-bit BMP. The caller owns and must close each returned PIL image.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral
from pathlib import Path

import numpy as np
from PIL import Image


PIXEL_MODES = {'保持原图': 'original', '8 位灰度': 'gray8', '1 位二值': 'binary1'}
INPUT_RANGES = {'按位深范围': 'bit_depth', '自定义范围': 'custom'}


@dataclass(frozen=True)
class DmdImageOptions:
    """Fixed input mapping shared by every image in a playback sequence.

    A threshold compares the converted 8-bit value with ``>= threshold``.
    Floyd–Steinberg dithering uses Pillow's fixed midpoint and ignores threshold.
    Custom black/white levels are expressed in original grayscale sample units.
    ``original`` bypasses all conversion, including the range settings.
    """

    pixel_mode: str = 'original'
    threshold: int = 128
    dither: bool = False
    input_range: str = 'bit_depth'
    black_level: float = 0.0
    white_level: float = 65535.0

    def __post_init__(self):
        if self.pixel_mode not in PIXEL_MODES.values():
            raise ValueError(f'不支持的 DMD 像素模式：{self.pixel_mode}')
        if self.input_range not in INPUT_RANGES.values():
            raise ValueError(f'不支持的 DMD 输入范围：{self.input_range}')
        if not isinstance(self.threshold, Integral) or not 0 <= self.threshold <= 255:
            raise ValueError('DMD 二值阈值必须为 0～255 的整数。')
        if not isinstance(self.dither, bool):
            raise ValueError('DMD 抖动选项必须为布尔值。')
        if not math.isfinite(self.black_level) or not math.isfinite(self.white_level):
            raise ValueError('DMD 黑、白电平必须为有限数值。')
        if self.white_level <= self.black_level:
            raise ValueError('DMD 白电平必须大于黑电平。')

    @classmethod
    def from_settings(cls, settings):
        pixel_mode = settings.get('dmdPixelModeComboBox', '保持原图')
        input_range = settings.get('dmdInputRangeComboBox', '按位深范围')
        # Reject an unrecognized saved value instead of silently using defaults.
        if pixel_mode not in PIXEL_MODES or input_range not in INPUT_RANGES:
            raise ValueError('DMD 图像转换配置包含未知选项。')
        return cls(PIXEL_MODES[pixel_mode], settings.get('dmdThresholdSpinBox', 128),
                   settings.get('dmdDitherCheckBox', False), INPUT_RANGES[input_range],
                   float(settings.get('dmdBlackLevelSpinBox', 0)),
                   float(settings.get('dmdWhiteLevelSpinBox', 65535)))

    def as_dict(self):
        """Return serializable conversion metadata for a captured frame."""
        return asdict(self)


def _automatic_white(image: Image.Image) -> float:
    if image.mode == '1':
        return 1.0
    if image.mode in ('I;16', 'I;16L', 'I;16B', 'I;16N'):
        return 65535.0
    if image.mode == 'I':
        # Pillow releases before 11 load unsigned 16-bit PNG as mode I, even
        # though ndarray then uses int32. PNG does not contain 32-bit samples.
        if image.format == 'PNG':
            return 65535.0
        tags = getattr(image, 'tag_v2', {})
        bits, sample_format = tags.get(258), tags.get(339, 1)
        if isinstance(bits, (tuple, list)):
            bits = bits[0] if len(bits) == 1 else None
        if isinstance(sample_format, (tuple, list)):
            sample_format = sample_format[0] if len(sample_format) == 1 else None
        if bits == 16 and sample_format == 1:
            return 65535.0
        raise ValueError('32 位整数图像的有效范围不明确，请选择 DMD 自定义范围并设置黑、白电平。')
    if image.mode == 'F':
        raise ValueError('浮点图像需要选择 DMD 自定义范围并设置黑、白电平。')
    if image.mode in ('L', 'LA', 'P', 'PA', 'RGB', 'RGBA', 'RGBX', 'CMYK', 'YCbCr'):
        return 255.0
    raise ValueError(f'不支持自动确定 {image.mode} 图像的 DMD 位深范围，请先转换图像格式。')


def prepare_image(path: str | Path, options: DmdImageOptions) -> Image.Image:
    """Read source pixels and return original, 8-bit grayscale or 1-bit pixels.

    Auto mapping uses bit depth, never per-image min/max. Custom ranges clip to
    the selected endpoints, then linearly map to 0..255 using nearest integer.
    Color conversion follows Pillow luminance; alpha does not alter luminance.
    The source is closed before returning, and never saved or modified.
    """
    with Image.open(path) as source:
        source.load()
        if options.pixel_mode == 'original':
            return source.copy()
        white = _automatic_white(source) if options.input_range == 'bit_depth' else options.white_level
        black = 0.0 if options.input_range == 'bit_depth' else options.black_level
        if source.mode in ('1', 'L', 'I', 'F') or source.mode.startswith('I;16'):
            values = np.asarray(source, dtype=np.float64)
        else:
            with source.convert('L') as grayscale:
                values = np.asarray(grayscale, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('DMD 输入图像包含 NaN 或无穷值，不能转换为整数灰度。')
        # Clipping before subtraction avoids overflow for out-of-range samples.
        normalized = (np.clip(values, black, white) - black) / (white - black)
        pixels = np.floor(normalized * 255.0 + 0.5).astype(np.uint8)
        gray = Image.fromarray(pixels)
        if options.pixel_mode == 'gray8':
            return gray
        try:
            if options.dither:
                return gray.convert('1', dither=Image.Dither.FLOYDSTEINBERG)
            return gray.point([255 if value >= options.threshold else 0
                               for value in range(256)], mode='1')
        finally:
            gray.close()
