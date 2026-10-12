"""Geometry and fixed-coordinate camera metrics; never refit per candidate."""
import cv2
import numpy as np


def intensity(image):
    array = np.asarray(image)
    # Fixed green channel for this colour sensor / green-laser bench. No
    # per-frame contrast stretching or automatic white balance is performed.
    return (array[..., 1] if array.ndim == 3 else array).astype(np.float32)


def centroid(image, dark):
    signal = np.maximum(intensity(image) - intensity(dark), 0)
    smooth = cv2.GaussianBlur(signal, (0, 0), 2)
    background = float(np.median(smooth))
    noise = max(float(np.median(np.abs(smooth - background))) * 1.4826, .5)
    if smooth.max() < background + 8 * noise:
        raise ValueError('DMD 探测点不可见或信噪比不足。')
    mask = (smooth > max(background + 5 * noise, float(smooth.max()) * .30)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        raise ValueError('未检测到完整探测点。')
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = 1 + int(np.argmax(areas))
    x, y, w, h, area = stats[label]
    if x <= 2 or y <= 2 or x + w >= mask.shape[1] - 2 or y + h >= mask.shape[0] - 2:
        raise ValueError('探测点被相机边界裁切。')
    if area < 5 or area > .15 * mask.size:
        raise ValueError('探测点尺寸异常，可能不在可直接定位的成像平面。')
    if len(areas) > 1 and np.sort(areas)[-2] > .45 * area:
        raise ValueError('探测点存在多个相近亮斑，无法唯一定位。')
    weights = np.where(labels == label, signal, 0)
    yy, xx = np.indices(mask.shape)
    return np.array([(weights * xx).sum(), (weights * yy).sum()]) / weights.sum()


def fit_geometry(source, destination, max_error):
    source, destination = np.asarray(source, np.float64), np.asarray(destination, np.float64)
    if len(source) < 8 or np.linalg.matrix_rank(source - source.mean(0)) < 2:
        raise ValueError('有效 DMD 点不足或分布退化；请先让更大的输入区域进入相机视野。')
    # Every fourth visible point is held out from fitting.
    held = np.arange(len(source)) % 4 == 1
    matrix, inliers = cv2.estimateAffine2D(source[~held], destination[~held],
        method=cv2.RANSAC, ransacReprojThreshold=max_error, maxIters=3000)
    if matrix is None or inliers.sum() < 6 or np.linalg.cond(matrix[:, :2]) > 20:
        raise ValueError('DMD 坐标关系不可稳定拟合，不能按几何成像继续自动标定。')
    errors = np.linalg.norm(source @ matrix[:, :2].T + matrix[:, 2] - destination, axis=1)
    held_error = float(np.sqrt(np.mean(errors[held] ** 2)))
    if held_error > max_error:
        raise ValueError(f'DMD 独立验证误差 {held_error:.2f} px 超过阈值；请检查衍射、裁切和成像关系。')
    u, _, vt = np.linalg.svd(matrix[:, :2])
    orientation = (u @ vt).T
    mirrored = np.linalg.det(orientation) < 0
    sign = -1 if mirrored else 1
    angle = float(np.degrees(np.arctan2(orientation[1, 0] * sign, orientation[0, 0] * sign)))
    return {'matrix': matrix.tolist(), 'heldout_rms_px': held_error,
            'errors_px': errors.tolist(), 'source_points': source.tolist(),
            'camera_points': destination.tolist(), 'horizontal_flip': bool(mirrored),
            'rotation_deg': angle}


def camera_map(config, geometry):
    """Final (SLM2) propagation grid -> DMD coordinates -> raw sensor coordinates.

    The equal-focal-length 4f relay has signed magnification -1. Pixel centres
    are used consistently; optical propagation remains in the training model.
    """
    n = config.output_grid_size
    ratio = config.output_sampling_um / config.dmd_pitch_um
    grid_to_dmd = np.array([[-ratio, 0, (config.dmd_width - 1) / 2 + ratio * (n - 1) / 2],
                            [0, -ratio, (config.dmd_height - 1) / 2 + ratio * (n - 1) / 2],
                            [0, 0, 1.]])
    return np.asarray(geometry['matrix']) @ grid_to_dmd


def correlation(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 1e-12 else -1.0


def reference_coverage(expected, transform, camera_shape, config):
    h, w = camera_shape[:2]
    expected = np.asarray(expected, np.float32)
    support = (expected > expected.max() * .08).astype(np.uint8)
    support = cv2.dilate(support, np.ones((9, 9), np.uint8))
    y, x = np.nonzero(support)
    projected = np.column_stack((x, y)) @ transform[:, :2].T + transform[:, 2]
    coverage = float(np.mean((projected[:, 0] > 3) & (projected[:, 0] < w - 4)
                            & (projected[:, 1] > 3) & (projected[:, 1] < h - 4)))
    if coverage < config.minimum_coverage:
        raise ValueError(f'目标有效覆盖率仅 {coverage:.1%}；需调整光路倍率或可见区域。')
    return coverage, support


def compare_capture(image, dark, expected, transform, config):
    signal = np.maximum(intensity(image) - intensity(dark), 0)
    h, w = signal.shape
    expected = np.asarray(expected, np.float32)
    coverage, support = reference_coverage(expected, transform, signal.shape, config)
    reference = cv2.warpAffine(expected, transform, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(support, transform, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    # Include a guard region so a bright background cannot pass as a good spot.
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    if mask.sum() < 25 or np.max(signal[mask]) < 3:
        raise ValueError('目标区域没有足够的有效光信号。')
    maximum = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 255
    saturated = float(np.mean(intensity(image)[mask] >= maximum * .99))
    if saturated > config.max_saturated_fraction:
        raise ValueError('标定区域过曝，请降低固定曝光或光强后重新标定。')
    measured = cv2.GaussianBlur(signal, (0, 0), 1.2)
    predicted = cv2.GaussianBlur(reference, (0, 0), 1.2)
    return {'score': correlation(measured[mask], predicted[mask]),
            'coverage': coverage, 'saturated_fraction': saturated}, reference
