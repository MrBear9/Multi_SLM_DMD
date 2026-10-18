"""Camera-to-target coordinates; no per-image alignment or contrast fitting."""
import cv2
import numpy as np
from .geometry import observation_matrix, project, validate_profile


def rectify(image, config):
    raw = np.asarray(image)
    signal = raw[..., 1] if raw.ndim == 3 else raw
    if signal.ndim != 2:
        raise ValueError('需要灰度或 BGR 相机帧。')
    if np.issubdtype(raw.dtype,np.integer) and config.camera_white > np.iinfo(raw.dtype).max:
        raise ValueError('饱和电平超过相机输出数据类型范围，请检查输出位深。')
    nx, ny = config.output_width, config.output_height
    if config.geometry_profile:
        validate_profile(config.geometry_profile, config, raw_shape=signal.shape)
    x = ((np.arange(nx, dtype=np.float32)+.5)/nx-.5)*config.target_width_um
    y = ((np.arange(ny, dtype=np.float32)+.5)/ny-.5)*config.target_height_um
    xx, yy = np.meshgrid(x, y)
    mapped = project(observation_matrix(config), np.stack([xx, yy], axis=-1))
    mx, my = mapped[...,0], mapped[...,1]
    if mx.min() < 1 or my.min() < 1 or mx.max() > signal.shape[1]-2 or my.max() > signal.shape[0]-2:
        raise ValueError('目标观察区域超出相机画面，请修正中心或减小目标物理视野；不会补造缺失区域。')
    observed = cv2.remap(signal.astype(np.float32), mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
    if np.mean(observed >= config.camera_white*.99) > .01:
        raise ValueError('目标区域过曝或白电平设置错误，请检查相机原始数据位深及曝光。')
    observed = np.maximum(observed-config.camera_dark, 0)
    if observed.mean() < max((config.camera_white-config.camera_dark)*1e-5, 1e-6):
        raise ValueError('目标区域没有足够信号，请检查映射、曝光和光路。')
    return observed
