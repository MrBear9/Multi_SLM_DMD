"""Native-pixel orientation shared by previews, device output and camera frames."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageTransform:
    """Flip in source axes, then rotate clockwise without interpolating samples."""

    angle: float = 0.0
    horizontal: bool = False
    vertical: bool = False

    def __post_init__(self):
        if not math.isfinite(self.angle):
            raise ValueError('Image rotation angle must be finite.')

    @classmethod
    def from_settings(cls, settings, key):
        return cls(float(settings.get(f'{key}RotationSpinBox', 0.0)),
                   bool(settings.get(f'{key}HorizontalFlipCheckBox', False)),
                   bool(settings.get(f'{key}VerticalFlipCheckBox', False)))

    @property
    def is_identity(self):
        return self.angle % 360 == 0 and not self.horizontal and not self.vertical

    def as_dict(self):
        return {'angle': self.angle, 'horizontal': self.horizontal,
                'vertical': self.vertical}

    def output_size(self, size):
        """Match Pillow's expanded, centre-based nearest-neighbour rotation."""
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError('Image dimensions must be positive.')
        angle = (-self.angle) % 360
        if angle in (0, 180):
            return width, height
        if angle in (90, 270):
            return height, width
        radians = -math.radians(angle)
        cosine, sine = round(math.cos(radians), 15), round(math.sin(radians), 15)
        cx, cy = width / 2, height / 2
        tx, ty = -cx * cosine - cy * sine + cx, cx * sine - cy * cosine + cy
        corners = [(cosine * x + sine * y + tx, -sine * x + cosine * y + ty)
                   for x, y in ((0, 0), (width, 0), (width, height), (0, height))]
        xs, ys = zip(*corners)
        return math.ceil(max(xs)) - math.floor(min(xs)), math.ceil(max(ys)) - math.floor(min(ys))

    def apply_pil(self, image, fill=0):
        """Return a caller-owned image; preserve mode, bit depth and phase values."""
        result = image.copy()
        try:
            for enabled, operation in ((self.horizontal, Image.Transpose.FLIP_LEFT_RIGHT),
                                       (self.vertical, Image.Transpose.FLIP_TOP_BOTTOM)):
                if enabled:
                    previous, result = result, result.transpose(operation)
                    previous.close()
            if self.angle % 360:
                previous, result = result, result.rotate(
                    -self.angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=fill)
                previous.close()
            return result
        except Exception:
            result.close()
            raise

    def apply_array(self, image, fill=0):
        """Return owned contiguous data without changing dtype or channel order."""
        source = np.asarray(image)
        if source.ndim not in (2, 3) or not source.shape[0] or not source.shape[1]:
            raise ValueError('Expected a nonempty H×W or H×W×C image array.')
        if self.horizontal:
            source = source[:, ::-1]
        if self.vertical:
            source = source[::-1]
        angle = self.angle % 360
        if angle in (0, 90, 180, 270):
            return np.array(np.rot90(source, -int(angle // 90)), copy=True, order='C')
        # Rotate pixel indices once, then gather untouched samples. This also
        # preserves uint16 colour frames and other dtypes unsupported by Pillow.
        height, width = source.shape[:2]
        if height * width > np.iinfo(np.int32).max:
            raise ValueError('Image is too large for native-pixel transformation.')
        indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
        with Image.fromarray(indices) as index_image:
            with index_image.rotate(-self.angle, resample=Image.Resampling.NEAREST,
                                    expand=True, fillcolor=-1) as rotated:
                locations = np.array(rotated)
        output = np.empty(locations.shape + source.shape[2:], dtype=source.dtype)
        output[...] = fill
        valid = locations >= 0
        output[valid] = source.reshape((-1,) + source.shape[2:])[locations[valid]]
        return output


def from_settings(settings, key):
    return ImageTransform.from_settings(settings, key)
