"""Image fitting in physical coordinates, shared by the UI and optimizer."""
import cv2
import numpy as np
from PIL import Image


def prepare_target(config):
    if config.target_kind != 'file':
        # Draw in physical coordinates so line thickness has the same meaning
        # along both axes, including rectangular targets and numerical grids.
        x=((np.arange(config.output_width)+.5)/config.output_width-.5)*config.target_width_um
        y=((np.arange(config.output_height)+.5)/config.output_height-.5)*config.target_height_um
        xx,yy=np.meshgrid(x,y)
        side=min(config.target_width_um,config.target_height_um)
        half=.04*side
        if config.target_kind=='cross':
            mask=((abs(xx)<=half)&(abs(yy)<=.35*side))|((abs(yy)<=half)&(abs(xx)<=.35*side))
        elif config.target_kind=='frame':
            mask=(np.maximum(abs(xx),abs(yy))<=.35*side)&(np.maximum(abs(xx),abs(yy))>=.27*side)
        elif config.target_kind=='asymmetric':
            mask=((abs(xx+.25*side)<=half)&(abs(yy)<=.3*side))|((abs(yy-.26*side)<=half)&(abs(xx)<=.29*side))
            mask|=((xx-.25*side)**2+(yy+.25*side)**2 <= (.07*side)**2)
        else:
            raise ValueError('未知测试图案。')
        return mask.astype(np.float32)
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
