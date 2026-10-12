"""Asymmetric intensity targets; these are not SLM phase images."""
from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image


TARGETS = ('dots', 'arrow', 'corners', 'holdout')


def illumination_mask(config):
    """White DMD aperture on SLM1's numerical coordinates after the 1:1 relay."""
    n = config.slm1_grid_size
    axis = (np.arange(n, dtype=np.float32) - (n - 1) / 2) * config.slm1_sampling_um
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    theta = math.radians(config.dmd_rotation)
    x = -xx * math.cos(theta) - yy * math.sin(theta)
    y = xx * math.sin(theta) - yy * math.cos(theta)
    if config.dmd_horizontal_flip:
        x = -x
    height, width = config.aperture_um
    return (np.abs(x) < width / 2) & (np.abs(y) < height / 2)


def target_intensity(config, name):
    n = config.output_grid_size
    out = np.zeros((n, n), np.float32)
    # Fit the central target into a conservative portion of the illuminated field.
    scale = min(config.aperture_um) / config.output_sampling_um * 0.27
    cx = (n - 1) / 2 + config.target_x_um / config.output_sampling_um
    cy = (n - 1) / 2 + config.target_y_um / config.output_sampling_um
    radius = max(2, round(scale * 0.09))
    points = {
        'dots': [(-.8, -.7), (.7, -.4), (-.4, .7)],
        'corners': [(-.7, -.7), (.7, -.7), (-.7, .7), (.3, .5)],
        'holdout': [(-.6, -.8), (.8, -.1), (.2, .7), (-.7, .3), (.0, -.2)],
    }
    if name == 'arrow':
        vertices = [(-.8, -.18), (.2, -.18), (.2, -.55), (.85, .0),
                    (.2, .55), (.2, .18), (-.8, .18)]
        polygon = np.array([(round(cx + x * scale), round(cy + y * scale))
                            for x, y in vertices], np.int32)
        cv2.fillPoly(out, [polygon], 1.0)
    else:
        for i, (x, y) in enumerate(points[name]):
            cv2.circle(out, (round(cx + x * scale), round(cy + y * scale)),
                       radius + (i % 2), 1.0 - 0.10 * i, -1)
    if min(cx, cy, n - 1 - cx, n - 1 - cy) < 1.1 * scale:
        raise ValueError('相机中心对应位置超出训练计算区域，请扩大计算网格或先粗调光路。')
    return cv2.GaussianBlur(out, (0, 0), .7)


def save_gray(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)
    return path


def dmd_patterns(config, root):
    root = Path(root)
    shape = config.dmd_height, config.dmd_width
    dark = save_gray(root / 'dark.png', np.zeros(shape, np.uint8))
    white = save_gray(root / 'white.png', np.full(shape, 255, np.uint8))
    probes = []
    # Individual points have identities through the acquisition sequence, even
    # when the sensor only sees part of the full input area.
    for fraction in (.18, .32):
        for y, x in ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0),
                     (1, 0), (-1, 1), (0, 1), (1, 1)):
            cx = round((shape[1] - 1) / 2 + x * shape[1] * fraction)
            cy = round((shape[0] - 1) / 2 + y * shape[0] * fraction)
            array = np.zeros(shape, np.uint8)
            cv2.circle(array, (cx, cy), max(3, round(min(shape) * .018)), 255, -1)
            path = save_gray(root / f'probe_{len(probes):02d}.png', array)
            probes.append((path, (cx, cy)))
    return dark, white, probes

