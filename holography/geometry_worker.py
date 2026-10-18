"""Single-SLM focal probes and independent geometric validation.

Runs in the external environment; hardware transactions use the existing RPC.
"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from PIL import Image

from .geometry import fit_mapping, locate_spot, project, roi_info, signature


def probe_points(config):
    train = np.array([(x,y) for y in (-.4,0,.4) for x in (-.4,0,.4)])
    if config.geometry_pattern == 'asymmetric':
        train[1] = [-.08,-.32]
        train[5] = [.32,.08]
    held = np.array([[-.31,-.19], [.23,-.29], [.29,.27], [-.27,.31], [.06,-.07]])
    return np.vstack([train, held])*[config.target_width_um,config.target_height_um], len(train)


def focus_phase(config, point):
    key = config.active_slms[0]
    h, w = config.shape(key)
    pitch = getattr(config,key+'_pitch_um')
    distance = getattr(config,key+'_distance_mm')*1000
    wavelength = config.wavelength_nm/1000
    x,y = np.meshgrid((np.arange(w)-(w-1)/2)*pitch-point[0],
                      (np.arange(h)-(h-1)/2)*pitch-point[1])
    r = np.sqrt(distance*distance+x*x+y*y)
    if max(np.abs(x/r).max(),np.abs(y/r).max())*pitch/wavelength >= .48:
        raise ValueError('定位相位超过 SLM 像素的采样带宽；请增大传播距离或减小目标视野/有效口径。')
    # Stable evaluation of r-z avoids cancellation for small apertures.
    phase = -2*np.pi/wavelength*((x*x+y*y)/(r+distance))
    gray = np.remainder(phase/(2*np.pi),1)
    if config.gray_inverted:
        gray = 1-gray
    return np.uint8(np.rint(gray*255))


def run_geometry(job, config, rpc, emit):
    root = Path(job['output'])
    root.mkdir(parents=True,exist_ok=True)
    key = config.active_slms[0]
    points, training_count = probe_points(config)
    # Check every probe before changing the physical display.
    phases = [focus_phase(config,p) for p in points]

    def capture(array, name):
        folder = root/'captures'/name
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/(key+'.png')
        Image.fromarray(array).save(path)
        payload = {'paths':{key:str(path)},'capture_path':str(folder/'raw.npy')}
        result = rpc(payload)
        if result.get('error'):
            raise RuntimeError(result['error'])
        image = np.load(result['path'],allow_pickle=False)
        emit('preview',kind='camera_raw',path=str(folder/'raw.npy'))
        return image

    emit('log',message='单 SLM 定位：先采集平相位背景，再逐点聚焦；最后 5 点仅用于独立验证。')
    background = capture(np.zeros(config.shape(key),np.uint8),'background')
    measured = []
    for i, phase in enumerate(phases):
        raw = capture(phase, f'probe_{i:02d}')
        try:
            position = locate_spot(raw,background,config.camera_white)
        except ValueError as exc:
            raise ValueError(f'定位点 {i+1}（目标 {points[i].tolist()} μm）：{exc}') from exc
        measured.append(position)
        # Keep partial observations even when a later capture fails.
        (root/'points.json').write_text(json.dumps({'physical_um':points[:i+1].tolist(),
            'camera_px':measured,'training_count':training_count},indent=2),encoding='utf-8')
        emit('progress',value=int((i+1)/len(phases)*95),
             message=f'定位 {i+1}/{len(phases)}：相机 ({position[0]:.2f}, {position[1]:.2f}) px')
    matrix = fit_mapping(points[:training_count],measured[:training_count],config.geometry_mapping)
    errors = np.linalg.norm(project(matrix,points)-measured,axis=1)
    roi = roi_info(matrix,config,background.shape)
    passed = bool(errors.max() <= config.geometry_tolerance and roi['coverage'] >= .99999)
    result = {'schema':1,'kind':'holography_geometry','validated':passed,
        'signature':signature(config),'hardware':job.get('hardware'), 'config':asdict(config),
        'matrix':matrix.tolist(),'raw_shape':list(background.shape[:2]),'roi':roi,
        'physical_points_um':points.tolist(),'camera_points_px':measured,
        'training_errors_px':errors[:training_count].tolist(),
        'validation_errors_px':errors[training_count:].tolist(),
        'interpretation':'Effective mapping from model-generated probes; absolute scale needs an independent physical reference.'}
    path = root/'geometry.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    emit('result',path=str(path),message=f'几何验证最大误差 {errors[training_count:].max():.3f} px；'
        f'目标覆盖 {roi["coverage"]*100:.2f}%。'+('可应用映射。' if passed else '未通过，保留诊断结果；请调整视野、曝光或光路。'))
    return result
