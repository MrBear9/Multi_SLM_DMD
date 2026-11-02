"""Portable, native-gray phase response. No torch or GUI dependency.

Measured profiles use signed phase relative to gray zero. The sign is an
explicit experimental convention: a single Fourier sideband cannot determine
the absolute phase sign without knowing the reference/test beam arrangement.
"""
import hashlib
import json
import math
import numpy as np


def validate_settings(config):
    if not isinstance(config.lut_profiles,dict) or not isinstance(config.lut_roi,dict):
        raise ValueError('LUT 曲线和条纹 ROI 格式无效。')
    for key in ('slm1','slm2'):
        source = getattr(config,key+'_lut_source')
        if source not in ('model','ideal','measured'):
            raise ValueError('未知 LUT 来源。')
        if source == 'measured' and key in config.active_slms and config.operation!='lut':
            validate_profile(config.lut_profiles.get(key),config,key)
    if config.operation != 'lut':return
    if config.mode not in ('slm1','slm2'):
        raise ValueError('LUT 请分别标定单片 SLM，不能同时扫描两片。')
    if not config.lut_roi:
        raise ValueError('请先在第三页框选直线干涉条纹区域，再读取为 LUT 条纹 ROI。')
    if type(config.lut_gray_step) is not int or not 1<=config.lut_gray_step<=8:
        raise ValueError('LUT 灰度步长需要整数 1～8。')
    if config.lut_split not in ('vertical','horizontal') or type(config.lut_negative) is not bool:
        raise ValueError('LUT 分区或相位方向无效。')
    for name in ('wavelength_nm','camera_white','capture_timeout','geometry_exposure_us',
                 'reconstruction_exposure_us',config.mode+'_pitch_um'):
        value=getattr(config,name)
        if not math.isfinite(value) or value<=0:raise ValueError(name+' 必须为有限正数。')
    for name in ('lut_carrier_x','lut_carrier_y'):
        value=getattr(config,name)
        if not math.isfinite(value) or abs(value)>512:raise ValueError('条纹载频需要 -512～512 周期/ROI。')
    for name,low,high in (('average_frames',1,100),('discard_frames',0,100),
                          (config.mode+'_width',8,1920),(config.mode+'_height',8,1080)):
        value=getattr(config,name)
        if type(value) is not int or not low<=value<=high:raise ValueError(name+' 超出有效范围。')
    if not math.isfinite(config.settle_seconds) or config.settle_seconds<0:
        raise ValueError('稳定等待时间无效。')
    if not math.isfinite(config.camera_dark) or not 0<=config.camera_dark<config.camera_white:
        raise ValueError('相机黑白电平无效。')


def validate_profile(profile, config=None, key=None):
    if not isinstance(profile, dict) or profile.get('kind') != 'slm_phase_lut' or profile.get('version') != 1:
        raise ValueError('请选择独立 LUT 标定生成的 lut.json；模型拟合曲线不属于独立测量。')
    if profile.get('validated') is not True:
        raise ValueError('LUT 未通过重复扫描与条纹质量验证。')
    curve = np.asarray(profile.get('phase_rad'), dtype=float)
    if curve.shape != (256,) or not np.isfinite(curve).all():
        raise ValueError('LUT 必须包含 256 个有限相位值。')
    span = curve[-1]-curve[0]
    if abs(curve[0]) > 1e-6 or not .5 <= abs(span) <= 8*math.pi:
        raise ValueError('LUT 必须以灰度 0 为相位参考，调制范围需为 0.5～8π rad。')
    if np.any(np.diff(curve)*np.sign(span) < -1e-7):
        raise ValueError('LUT 必须是连续的单调相位响应。')
    wave = profile.get('wavelength_nm')
    pitch = profile.get('pitch_um')
    if not isinstance(wave,(float,int)) or not math.isfinite(wave) or wave <= 0:
        raise ValueError('LUT 缺少有效工作波长。')
    if not isinstance(pitch,(float,int)) or not math.isfinite(pitch) or pitch <= 0:
        raise ValueError('LUT 缺少有效 SLM 像素间距。')
    if config is not None:
        if profile.get('slm') != key or abs(wave-config.wavelength_nm) > .01:
            raise ValueError(f'{key} LUT 的 SLM 或波长不匹配，请重新选择。')
        if abs(pitch-getattr(config,key+'_pitch_um')) > 1e-6:
            raise ValueError(f'{key} LUT 的硬件像素间距不匹配。')
    return curve


def signature(config):
    """Keep legacy model signatures unchanged for the default automatic mode."""
    result = {}
    for key in config.active_slms:
        source = getattr(config,key+'_lut_source')
        if source == 'model':
            continue
        entry = {'source':source}
        if source == 'measured':
            curve = validate_profile(config.lut_profiles.get(key),config,key)
            entry['sha256'] = hashlib.sha256(curve.astype('<f8').tobytes()).hexdigest()
        result[key] = entry
    return result


def phase_to_gray(phase, config, key):
    """Convert relative radians once; never apply to already optimized gray."""
    if getattr(config,key+'_lut_source') == 'measured':
        curve = validate_profile(config.lut_profiles.get(key),config,key)
    else:
        curve = np.linspace(0,(-1 if config.gray_inverted else 1)*2*np.pi,256)
    angles = np.remainder(curve,2*np.pi)
    order = np.argsort(angles);angles = angles[order]
    wanted = np.remainder(phase,2*np.pi)
    right = np.searchsorted(angles,wanted) % 256;left = (right-1) % 256
    distance = lambda x: np.abs(np.angle(np.exp(1j*(x-wanted))))
    return order[np.where(distance(angles[left]) <= distance(angles[right]),left,right)].astype(np.uint8)


def save_profile(profile, folder):
    from pathlib import Path
    folder = Path(folder)
    curve = validate_profile(profile)
    (folder/'lut.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savetxt(folder/'lut.csv',np.column_stack((np.arange(256),curve)),delimiter=',',
               header='gray,phase_rad',comments='',fmt=['%d','%.9g'])
    # Circular nearest-realizable inverse, including devices with <2pi range.
    phase = np.linspace(0,2*np.pi,1024,endpoint=False)
    error = np.abs(np.angle(np.exp(1j*(curve[:,None]-phase))))
    gray = error.argmin(axis=0)
    np.savetxt(folder/'inverse_lut.csv',np.column_stack((phase,gray,error[gray,np.arange(len(gray))])),
               delimiter=',',header='desired_phase_rad,gray,error_rad',comments='',fmt=['%.9g','%d','%.9g'])
    values = np.abs(curve);span = max(values.max(),2*np.pi)
    points = ' '.join(f'{50+g*700/255:.2f},{260-v/span*220:.2f}' for g,v in enumerate(values))
    (folder/'lut.svg').write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="310">'
        '<rect width="800" height="310" fill="white"/>'
        '<path d="M50 30 V260 H760" fill="none" stroke="black"/>'
        f'<polyline points="{points}" fill="none" stroke="#1269d3" stroke-width="2"/>'
        f'<text x="50" y="20">Phase magnitude (rad), span = {values[-1]:.4f}</text>'
        '<text x="50" y="285">0</text><text x="700" y="285">255 gray</text></svg>',encoding='utf-8')
