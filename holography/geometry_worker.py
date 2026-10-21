"""Single-SLM focal probes and independent geometric validation.

Runs in the external environment; hardware transactions use the existing RPC.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

from .geometry import fit_mapping, locate_spot, project, roi_info, signature
from .diagnostics import GUIDE, save_capture_preview, save_spot_diagnostics
from .geometry_limits import probe_points, validate_probe_bandwidth, target_edge_bandwidth, LIMIT
from .spot_detection import search_rect
from .ring_detection import locate_ring
from .geometry_exposure import prepare_exposure


from .probe_phase import focus_phase


def run_geometry(job, config, rpc, emit):
    root = Path(job['output'])
    root.mkdir(parents=True,exist_ok=True)
    (root/'captures').mkdir(exist_ok=True)
    (root/'captures'/'README.txt').write_text(GUIDE,encoding='utf-8')
    key = config.active_slms[0]
    validate_probe_bandwidth(config)
    if config.geometry_search_roi:
        size = job.get('hardware',{}).get('camera',{}).get('geometry',{}).get('raw_size',[])
        if len(size)!=2:
            raise ValueError('无法核对定位搜索 ROI 的相机信息，请重新连接并读取 ROI。')
        search_rect(config.geometry_search_roi,size[::-1],job['hardware'])
    points, training_count = probe_points(config)
    local_count = len(probe_points(config,include_target=False)[0])
    target_edges_available = len(points)>local_count
    target_edge_error = '' if target_edges_available else '目标边缘定位相位超过像素采样带宽，尚未验证完整目标。'
    preflight_count=5 if config.geometry_movement_check else 0
    if preflight_count:
        preflight=np.array([[0,0],[-.2,0],[.2,0],[0,-.2],[0,.2]])*config.probe_size()
        points=np.vstack((preflight,points))
    # Check every probe before changing the physical display.
    phases = [focus_phase(config,p) for p in points]

    exposure_override = None
    last_exposure = None

    def capture(array, name):
        nonlocal last_exposure
        folder = root/'captures'/name
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/(key+'.png')
        Image.fromarray(array).save(path)
        payload = {'paths':{key:str(path)},'capture_path':str(folder/'raw.npy')}
        if exposure_override is not None:
            payload['exposure_us'] = exposure_override
        result = rpc(payload)
        if result.get('error'):
            raise RuntimeError(result['error'])
        last_exposure=result.get('exposure_us',exposure_override)
        image = np.load(result['path'],allow_pickle=False)
        try:
            save_capture_preview(folder,image,config.camera_white)
        except OSError as exc:
            emit('log',message=f'采集预览图保存失败，raw.npy 已保留：{exc}')
        emit('preview',kind='camera_raw',path=str(folder/'raw.npy'))
        return image

    emit('log',message=f'单 SLM 定位：{config.geometry_probe} 图案 / {config.geometry_detector} 检测；9 点拟合、5 点局部独立验证。')
    emit('log',message=f'标定探测范围 {config.probe_size()[0]:g}×{config.probe_size()[1]:g} μm；'
         f'重建目标保持 {config.target_width_um:g}×{config.target_height_um:g} μm。'+
         ('另采集 8 个目标角点及边中点，仅用于全视野验证。' if target_edges_available else target_edge_error))
    emit('log',message=f'采集预览、差分和候选标注将保存到：{root / "captures"}（含说明 README.txt）。')
    background = capture(np.zeros(config.shape(key),np.uint8),'background')
    class Recapture(Exception):
        pass

    def choose_region(raw, folder, i, repeat, error, attempt):
        nonlocal exposure_override
        token = f'{folder.name}:{attempt}'
        response = rpc({'request_type':'geometry_select','request_id':token,
            'path':str(folder/'raw.npy'),'point_index':i,'point_count':len(points),
            'repeat_index':repeat,'target_um':points[i].tolist(),'reason':error or '',
            'exposure_us':exposure_override})
        if response.get('error'):
            raise RuntimeError(response['error'])
        if response.get('request_id') != token:
            raise ValueError('定位选择回复与当前采集图不匹配。')
        (folder/f'selection_{attempt:02d}.json').write_text(
            json.dumps(response,ensure_ascii=False,indent=2),encoding='utf-8')
        action=response.get('action')
        if action == 'cancel':
            raise InterruptedError('用户取消几何定位。')
        if action == 'recapture':
            value=float(response['exposure_us'])
            if not math.isfinite(value) or value<=0:
                raise ValueError('重拍曝光必须为有限正数。')
            exposure_override=value
            raise Recapture()
        if action != 'select':
            raise ValueError('未知定位辅助操作。')
        document=response['roi']
        search_rect(document,raw.shape,job.get('hardware'))
        return document

    def locate_capture(phase, i, repeat, retry, selected_roi):
        suffix=(f'_retry_{retry:02d}' if retry else '')
        name=f'probe_{i:02d}'+suffix+(f'_repeat_{repeat:02d}' if repeat else '')
        folder=root/'captures'/name
        raw = capture(phase, name)
        if config.geometry_assistance == 'manual' and repeat == 0 and selected_roi is None:
            selected_roi=choose_region(raw,folder,i,repeat,None,0)
        attempt=0
        while True:
            diagnostic = {};error = None;position=None
            try:
                options=dict(diagnostics=diagnostic,stability=config.geometry_spot_stability,
                             search_roi=selected_roi or config.geometry_search_roi)
                if config.geometry_detector=='ring':
                    position=locate_ring(raw,background,config.camera_white,radius=config.geometry_ring_radius,**options)
                else:
                    position=locate_spot(raw,background,config.camera_white,max_fwhm=config.geometry_spot_max_fwhm,**options)
            except ValueError as exc:
                error=str(exc)
            diagnostic['background_path']=str(background_path)
            diagnostic['search_roi_document']=selected_roi or config.geometry_search_roi
            try:
                destination=folder if attempt==0 else folder/f'analysis_{attempt:02d}'
                destination.mkdir(exist_ok=True)
                save_spot_diagnostics(destination,raw,diagnostic,points[i].tolist(),error)
            except (OSError,ValueError) as exc:
                emit('log',message=f'诊断文件保存失败，raw.npy 已保留：{exc}')
            if error is None:
                return position,diagnostic['localization'],selected_roi
            message=f'定位点 {i+1}，第 {repeat+1} 次采集（目标 {points[i].tolist()} μm）：{error}'
            emit('log',message=message+f'；诊断已保存到 {folder}。')
            if config.geometry_assistance == 'automatic':
                raise ValueError(message)
            attempt+=1
            selected_roi=choose_region(raw,folder,i,repeat,error,attempt)

    background_path=root/'captures'/'background'/'raw.npy'
    initial_roi=None
    if config.geometry_auto_exposure:
        camera=job.get('hardware',{}).get('camera',{})
        initial=config.exposure_for(camera.get('exposure',config.geometry_exposure_us))
        if config.geometry_detector=='ring' and not config.geometry_search_roi:
            if config.geometry_assistance=='automatic':
                raise ValueError('圆环自动曝光需要搜索 ROI；请读取第三页 ROI 或启用逐点辅助。')
            attempt=0
            while initial_roi is None:
                name='exposure_selection'+(f'_retry_{attempt:02d}' if attempt else '')
                raw=capture(phases[0],name)
                try:
                    initial_roi=choose_region(raw,root/'captures'/name,0,0,'请粗框中心及内圈，用于曝光准备。',attempt)
                except Recapture:
                    initial=exposure_override
                    attempt+=1
        exposure_records=[]
        def auto_capture(phase,name,exposure):
            nonlocal exposure_override
            exposure_override=exposure
            raw=capture(phase,name)
            if last_exposure is None:
                raise ValueError('相机未返回实际曝光，无法确认自动曝光已锁定。')
            return raw,last_exposure
        def record(item):
            exposure_records.append(item)
            (root/'exposure_preparation.json').write_text(json.dumps(exposure_records,indent=2),encoding='utf-8')
            emit('log',message=f'曝光准备 {item["iteration"]+1}：{item["exposure_us"]:g} μs；稳定计数 {item["stable_count"]}/2。')
        limits=camera.get('exposure_range')
        if limits:
            limits=[limits[0],min(limits[1],config.capture_timeout*1e6/max(config.average_frames+config.discard_frames+2,1))]
        # A manually selected local ring ROI applies only at this first position.
        # Other positions are checked by the detector and movement validation.
        exposure_override=prepare_exposure(auto_capture,[phases[0]],initial,limits,
            config.camera_white,config.geometry_exposure_target,
            initial_roi or config.geometry_search_roi,record)
        background=capture(np.zeros(config.shape(key),np.uint8),'background_locked')
        background_path=root/'captures'/'background_locked'/'raw.npy'
        emit('geometry_exposure_locked',exposure_us=exposure_override)
        emit('log',message=f'几何标定曝光已锁定 {exposure_override:g} μs；检测阈值仍由图像独立计算。')
    measured = [];quality=[]
    emit('log',message=f'中心检测：每点重复 {config.geometry_repeats} 次，'
         f'中心稳定性限值 {config.geometry_spot_stability:g} px；允许有限中心饱和，使用未饱和外围拟合。')
    for i, phase in enumerate(phases):
        try:
            retry=0;selected_roi=initial_roi if i==0 else None
            while True:
                repeats=[]
                try:
                    for j in range(config.geometry_repeats):
                        position,localization,selected_roi=locate_capture(phase,i,j,retry,selected_roi)
                        repeats.append((position,localization))
                    positions=np.asarray([r[0] for r in repeats])
                    separation=float(np.linalg.norm(positions[:,None]-positions[None,:],axis=-1).max())
                    item={'positions_xy':positions.tolist(),'max_separation_px':separation,
                          'limit_px':config.geometry_spot_stability,'localizations':[r[1] for r in repeats],
                          'passed':separation<=config.geometry_spot_stability,
                          'retry':retry,'search_roi':selected_roi,'exposure_override_us':exposure_override}
                    name=f'probe_{i:02d}'+(f'_retry_{retry:02d}' if retry else '')
                    (root/'captures'/name/'repeatability.json').write_text(
                        json.dumps(item,ensure_ascii=False,indent=2),encoding='utf-8')
                    if item['passed']:
                        break
                    message=(f'定位点 {i+1} 重复采集位置变化 {separation:.3f} px，超过 '
                             f'{config.geometry_spot_stability:g} px；请检查光斑与光路稳定性。')
                    if config.geometry_assistance == 'automatic':
                        raise ValueError(message)
                    folder=root/'captures'/(name+f'_repeat_{config.geometry_repeats-1:02d}')
                    raw=np.load(folder/'raw.npy',allow_pickle=False)
                    selected_roi=choose_region(raw,folder,i,config.geometry_repeats-1,message,1000)
                    # A new region must be checked on a fresh set of repeated frames.
                    retry+=1
                except Recapture:
                    # Exposure changes invalidate the subtraction background and every
                    # repeat of this point. Keep all previous attempts on disk.
                    retry+=1;selected_roi=None
                    name=f'background_probe_{i:02d}_retry_{retry:02d}'
                    background=capture(np.zeros(config.shape(key),np.uint8),name)
                    background_path=root/'captures'/name/'raw.npy'
                    emit('log',message=f'重新采集定位点 {i+1} 的全部重复帧；曝光 {exposure_override:g} μs，已更新同曝光背景。')
            position=positions.mean(0).tolist()
            measured.append(position);quality.append(item)
            if preflight_count and i==preflight_count-1:
                movement=np.asarray(measured)
                horizontal=movement[2]-movement[1];vertical=movement[4]-movement[3]
                distances=np.linalg.norm(movement[1:]-movement[0],axis=1)
                area=abs(float(np.linalg.det(np.array([horizontal,vertical]))))
                sine=area/max(float(np.linalg.norm(horizontal)*np.linalg.norm(vertical)),1e-9)
                # Five probes suffice for this preliminary affine consistency check;
                # the final calibration still requires the full nine fit points.
                pre_matrix=np.eye(3)
                pre_matrix[:2]=np.linalg.lstsq(np.c_[points[:5],np.ones(5)],movement,rcond=None)[0].T
                residual=float(np.linalg.norm(project(pre_matrix,points[:5])-movement,axis=1).max())
                passed=bool(distances.min()>max(3.,config.geometry_spot_stability*3) and sine>.15 and residual<=config.geometry_tolerance)
                movement_result={'passed':passed,'physical_um':points[:5].tolist(),'camera_px':measured,
                    'minimum_motion_px':float(distances.min()),'axis_sine':sine,'affine_error_px':residual}
                (root/'movement_check.json').write_text(json.dumps(movement_result,indent=2),encoding='utf-8')
                if not passed:
                    raise ValueError('五点移动验证未通过：位移过小、方向退化或中心不随指令一致移动；详见 movement_check.json。')
                emit('log',message='中心/左右/上下移动验证通过，开始多位置拟合及独立验证。')
            # Keep partial observations even when a later capture fails.
            (root/'points.json').write_text(json.dumps({'physical_um':points[:i+1].tolist(),
                'camera_px':measured,'training_count':training_count,
                'movement_check_count':preflight_count},indent=2),encoding='utf-8')
            emit('progress',value=int((i+1)/len(phases)*95),
                 message=f'定位 {i+1}/{len(phases)}：相机 ({position[0]:.2f}, {position[1]:.2f}) px')
        except ValueError as exc:
            if i < preflight_count+local_count:
                raise
            target_edge_error=str(exc)
            emit('log',message='目标边缘验证未完成：'+target_edge_error+'；保留局部映射，目标尺寸不变。')
            break
    points=points[preflight_count:len(measured)];measured=measured[preflight_count:];quality=quality[preflight_count:]
    matrix = fit_mapping(points[:training_count],measured[:training_count],config.geometry_mapping)
    errors = np.linalg.norm(project(matrix,points)-measured,axis=1)
    roi = roi_info(matrix,config,background.shape)
    local_passed = bool(errors[:local_count].max() <= config.geometry_tolerance)
    edge_errors = errors[local_count:]
    edges_passed = bool(len(edge_errors)==8 and edge_errors.max()<=config.geometry_tolerance)
    if len(edge_errors)==8 and not edges_passed:
        target_edge_error='完整目标边缘定位误差超限。'
    passed = bool(local_passed and edges_passed and roi['coverage'] >= .99999)
    result = {'schema':2,'kind':'holography_geometry','validated':passed,
        'local_validated':local_passed,'probe_size_um':list(config.probe_size()),
        'target_validation':{'passed':edges_passed,'error':target_edge_error,
            'physical_um':points[local_count:].tolist(),'camera_px':measured[local_count:],
            'errors_px':edge_errors.tolist(),'phase_peak':float(target_edge_bandwidth(config)),
            'phase_limit':LIMIT},
        'signature':signature(config),'hardware':job.get('hardware'), 'config':asdict(config),
        'matrix':matrix.tolist(),'raw_shape':list(background.shape[:2]),'roi':roi,
        'physical_points_um':points.tolist(),'camera_points_px':measured,
        'training_errors_px':errors[:training_count].tolist(),
        'validation_errors_px':errors[training_count:].tolist(),
        'detector_version':3,'localization_quality':quality,'detector':config.geometry_detector,
        'probe':config.geometry_probe,'locked_exposure_us':exposure_override,
        'movement_check':preflight_count>0,
        'interpretation':'Effective mapping from model-generated probes; absolute scale needs an independent physical reference.'}
    path = root/'geometry.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    emit('result',path=str(path),message=f'几何验证最大误差 {errors[training_count:].max():.3f} px；'
        f'完整目标在 CCD 内覆盖 {roi["coverage"]*100:.2f}%。'+('完整目标验证通过，可应用映射。' if passed else
            ('局部映射通过；完整目标未验证通过，不可应用。' if local_passed else '局部映射未通过。')+target_edge_error))
    return result
