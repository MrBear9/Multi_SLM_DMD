"""Physical target coordinates (micrometres) to untransformed camera pixels."""
import math
import copy
import cv2
import numpy as np


def signature(config):
    names = ('mode', 'wavelength_nm', 'gray_inverted', 'slm1_distance_mm',
             'slm2_distance_mm', 'inter_slm_mm', 'target_width_um', 'target_height_um',
             'camera_optical_path', 'sensor_pitch_x_um', 'sensor_pitch_y_um',
             'binning_x', 'binning_y', 'relay_magnification', 'camera_profile')
    result = {name: getattr(config, name) for name in names}
    if config.geometry_probe != 'focus':
        result.update(geometry_probe=config.geometry_probe,
                      geometry_probe_radius_um=config.geometry_probe_radius_um,
                      geometry_axicon_period_um=config.geometry_axicon_period_um,
                      probe_coordinates='fixed_pattern_fresnel_tilt')
    for slm in config.active_slms:
        for suffix in ('width', 'height', 'pitch_um'):
            name = slm + '_' + suffix
            result[name] = getattr(config, name)
    return result


def project(matrix, points):
    points = np.asarray(points, np.float64)
    p = np.c_[points.reshape(-1, 2), np.ones(points.size//2)] @ np.asarray(matrix).T
    if not np.isfinite(p).all() or np.any(np.abs(p[:, 2]) < 1e-9):
        raise ValueError('映射包含无穷远或无效坐标。')
    return (p[:, :2] / p[:, 2:]).reshape(points.shape)


def corners(config):
    return np.array([[-1,-1], [1,-1], [1,1], [-1,1]], float)*[
        config.target_width_um/2, config.target_height_um/2]


def observation_matrix(config):
    if config.geometry_profile:
        validate_profile(config.geometry_profile, config)
        return np.asarray(config.geometry_profile['matrix'], float)
    angle = math.radians(config.camera_angle_deg)
    pitch_y = config.camera_pitch_um * config.sensor_pitch_y_um*config.binning_y / (config.sensor_pitch_x_um*config.binning_x)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    matrix = np.eye(3)
    matrix[:2,:2] = rotation @ np.diag([
        (-1 if config.camera_flip_x else 1)/config.camera_pitch_um,
        (-1 if config.camera_flip_y else 1)/pitch_y])
    matrix[:2,2] = config.camera_center_x, config.camera_center_y
    return matrix


def roi_info(matrix, config, raw_shape):
    denominator = np.c_[corners(config),np.ones(4)] @ np.asarray(matrix)[2]
    if denominator.min()*denominator.max() <= 0:
        raise ValueError('透视映射的无穷远线穿过目标视野。')
    quad = project(matrix, corners(config))
    if not cv2.isContourConvex(quad.astype(np.float32)):
        raise ValueError('目标 ROI 映射不是有效凸四边形。')
    h, w = raw_shape[:2]
    frame = np.array([[1,1], [w-2,1], [w-2,h-2], [1,h-2]], np.float32)
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    intersection, _ = cv2.intersectConvexConvex(quad.astype(np.float32), frame)
    coverage = min(1., max(0., intersection/max(area, 1e-12)))
    centre = project(matrix, [[0,0], [1,0], [0,1]])
    jacobian = (centre[1:]-centre[0]).T
    if abs(np.linalg.det(jacobian)) < 1e-12:
        raise ValueError('目标映射退化。')
    return {'quad': quad.tolist(), 'bounds': [*quad.min(0), *quad.max(0)],
            'coverage': coverage, 'sampling_um': (1/np.linalg.norm(jacobian, axis=0)).tolist(),
            'rotation_deg': math.degrees(math.atan2(jacobian[1,0], jacobian[0,0])),
            'mirrored': bool(np.linalg.det(jacobian) < 0)}


def geometry_hardware(hardware):
    """Geometric identity excludes only photometric camera settings.

    Model calibration still compares the full hardware identity in worker.py.
    """
    value=copy.deepcopy(hardware)
    if not isinstance(value,dict):
        return value
    settings=value.get('settings',{})
    for key in ('ccdExposureSpinBox','ccdGainSpinBox','ccdAutoExposureCheckBox'):
        settings.pop(key,None)
    camera=value.get('camera')
    if isinstance(camera,dict):
        for key in ('exposure','gain','auto'):
            camera.pop(key,None)
    return value


def validate_profile(profile, config, hardware=None, raw_shape=None):
    if profile.get('schema') == 2 and profile.get('local_validated') and not profile.get('target_validation',{}).get('passed'):
        raise ValueError('局部标定通过，但完整目标边缘尚未验证通过，不能应用于全视野重建。')
    if profile.get('schema') == 2 and profile.get('local_validated') and profile.get('roi',{}).get('coverage',0)<.99999:
        raise ValueError('完整重建目标未被 CCD 覆盖，请调整相机端成像倍率或光路；目标尺寸未缩小。')
    if profile.get('schema') not in (1,2) or profile.get('kind') != 'holography_geometry' or profile.get('validated') is not True:
        raise ValueError('几何标定未通过验证或文件格式无效。')
    if profile.get('signature') != signature(config):
        raise ValueError('几何标定与当前 SLM、距离、目标尺寸或成像设置不一致，请重新标定。')
    matrix = np.asarray(profile.get('matrix'), float)
    if matrix.shape != (3,3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('几何映射矩阵无效。')
    if hardware is not None and geometry_hardware(profile.get('hardware')) != geometry_hardware(hardware):
        raise ValueError('几何标定的设备、分辨率、ROI、方向或显示设置与当前会话不同。')
    shape = profile.get('raw_shape', [])
    if len(shape) != 2 or min(shape) < 4:
        raise ValueError('几何标定缺少原始采集尺寸。')
    if raw_shape is not None and list(raw_shape[:2]) != shape:
        raise ValueError('相机采集尺寸已改变，请重新标定。')
    info = roi_info(matrix, config, shape)
    if info['coverage'] < .99999:
        raise ValueError('目标 ROI 未被相机完整覆盖，请减小目标视野或调整光路后重新标定。')
    errors = np.asarray(profile.get('validation_errors_px', []), float)
    if errors.size < 4 or not np.isfinite(errors).all() or errors.max() > config.geometry_tolerance:
        raise ValueError('独立验证点的误差超过当前允许值。')
    training = np.asarray(profile.get('training_errors_px', []),float)
    if training.size < 6 or not np.isfinite(training).all() or training.max() > config.geometry_tolerance:
        raise ValueError('拟合点误差超过当前允许值。')
    if profile.get('schema') == 2:
        from .geometry_limits import target_edge_points
        edge = profile.get('target_validation',{})
        physical=np.asarray(edge.get('physical_um',[]),float)
        measured=np.asarray(edge.get('camera_px',[]),float)
        if not edge.get('passed') or physical.shape!=(8,2) or measured.shape!=(8,2) or not np.isfinite(measured).all():
            raise ValueError('完整目标缺少 8 个有效的边缘验证点。')
        if not np.allclose(physical,target_edge_points(config),rtol=0,atol=1e-6):
            raise ValueError('边缘验证点与当前完整目标尺寸不一致。')
        if np.linalg.norm(project(matrix,physical)-measured,axis=1).max()>config.geometry_tolerance:
            raise ValueError('完整目标边缘验证误差超限。')
    return info


def fit_mapping(physical, camera, mode):
    src, dst = np.asarray(physical, float), np.asarray(camera, float)
    if len(src) < 6 or np.linalg.matrix_rank(np.c_[src, np.ones(len(src))]) < 3:
        raise ValueError('定位点不足或共线。')
    matrix = np.eye(3)
    if mode == 'similarity':
        # Orthogonal Procrustes allows either handedness; no guessed flip label.
        x, y = src-src.mean(0), dst-dst.mean(0)
        u, s, vt = np.linalg.svd(x.T @ y)
        transform = (u @ vt) * s.sum()/np.square(x).sum()
        matrix[:2,:2] = transform.T
        matrix[:2,2] = dst.mean(0)-src.mean(0) @ transform
    elif mode == 'affine':
        matrix[:2] = np.linalg.lstsq(np.c_[src,np.ones(len(src))], dst, rcond=None)[0].T
    elif mode == 'perspective':
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise ValueError('无法拟合透视映射。')
    else:
        raise ValueError('未知几何映射。')
    # Fail on ambiguous point matches instead of hiding them as RANSAC outliers.
    if abs(np.linalg.det(matrix[:2,:2])) < 1e-12:
        raise ValueError('定位点没有形成有效二维映射。')
    return matrix


# Kept here as the geometry API; photometric camera checks remain separate.
from .spot_detection import locate_spot
