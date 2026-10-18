"""Image fitting in physical coordinates, shared by the UI and optimizer."""
import cv2
import numpy as np
from PIL import Image


def prepare_target(config):
    with Image.open(config.target) as image:
        source = np.asarray(image.convert('L'), dtype=np.float32) / 255.
    h, w = source.shape
    width, height = config.target_width_um, config.target_height_um
    nx, ny = config.output_width, config.output_height
    if config.target_fit == 'stretch':
        return cv2.resize(source, (nx, ny), interpolation=cv2.INTER_AREA)
    scale = (min if config.target_fit == 'contain' else max)(width / w, height / h)
    x = (((np.arange(nx)+.5)/nx-.5)*width / scale + (w-1)/2).astype(np.float32)
    y = (((np.arange(ny)+.5)/ny-.5)*height / scale + (h-1)/2).astype(np.float32)
    xx, yy = np.meshgrid(x, y)
    return cv2.remap(source, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
