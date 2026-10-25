"""Measured SLM-pattern coordinates, deliberately separate from propagation probes.

A compact central board avoids requiring sharp corners at the panel boundary.
Two commanded translations resolve checkerboard handedness; a fourth exposure
at a different position validates the mapping without refitting it.
"""
from dataclasses import asdict, replace
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .diagnostics import gray_channel, save_capture_preview
from .geometry import fit_mapping, project, roi_info
from .geometry_exposure import prepare_exposure
from .spot_detection import search_rect


def board(config, offset=(0, 0)):
    h, w = config.shape(config.active_slms[0])
    cell, cols, rows = config.checker_cell_px, config.checker_columns, config.checker_rows
    bw, bh = (cols+1)*cell, (rows+1)*cell
    # Reserve a complete white border, direction marks and translation space.
    if bw+4*cell>w or bh+4*cell>h:
        raise ValueError('中心棋盘格及移动余量超出有效区域；请减小格宽或角点行列数。')
    x0, y0 = (w-bw)//2+int(offset[0]), (h-bh)//2+int(offset[1])
    image = np.full((h,w), 127, np.uint8)
    image[y0-cell:y0+bh+cell, x0-cell:x0+bw+cell] = 255
    yy,xx=np.indices((bh,bw))
    image[y0:y0+bh,x0:x0+bw] = ((xx//cell+yy//cell)%2*255).astype(np.uint8)
    # One mark at +X, two at +Y, outside the regular corner lattice.
    r=max(2,cell//10)
    for cx,cy in ((x0+bw+cell//2,y0+bh//2),
                  (x0+bw//2-cell//4,y0+bh+cell//2),
                  (x0+bw//2+cell//4,y0+bh+cell//2)):
        cv2.circle(image,(cx,cy),r,0,-1)
    x,y=np.meshgrid(x0+np.arange(1,cols+1)*cell-.5,
                    y0+np.arange(1,rows+1)*cell-.5)
    pitch=getattr(config,config.active_slms[0]+'_pitch_um')
    physical=(np.stack((x,y),axis=-1).reshape(-1,2)-[(w-1)/2,(h-1)/2])*pitch
    return image, physical


def detect(raw, config, roi=None):
    gray=gray_channel(raw).astype(np.float32)
    if not np.isfinite(gray).all():
        raise ValueError('棋盘格采集包含无效像素。')
    x,y,w,h=search_rect(roi,gray.shape)
    crop=gray[y:y+h,x:x+w]
    low,high=np.percentile(crop,[1,99])
    if high-low<max(1.,config.camera_white*.015):
        raise ValueError('棋盘格对比度不足，请调整曝光或相位灰度响应。')
    image=np.uint8(np.clip((crop-low)*255/(high-low),0,255))
    pattern=(config.checker_columns,config.checker_rows)
    found, points=cv2.findChessboardCornersSB(image,pattern,flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise ValueError(f'未识别完整的中心 {pattern[0]}×{pattern[1]} 角点。请框住整个小棋盘及白边，或减小格数/调整格宽；不需要框住整个 SLM。')
    points=points.reshape(-1,2).astype(float)+[x,y]
    return points


def orientations(points, config):
    grid=np.asarray(points).reshape(config.checker_rows,config.checker_columns,2)
    grids=[grid]
    if config.checker_rows==config.checker_columns: grids.append(grid.transpose(1,0,2))
    return [value[::sy,::sx].reshape(-1,2) for value in grids for sy in (1,-1) for sx in (1,-1)]


def repeat_consensus(observations, tolerance):
    """Reject isolated detections only with a unique strict temporal majority.

    Whole-frame motion, widespread bad corners and ambiguous two-frame changes
    still fail. No fitted grid positions or missing corners are manufactured.
    """
    stack=np.asarray(observations,dtype=float)
    if stack.ndim!=3 or stack.shape[2]!=2 or not np.isfinite(stack).all():
        raise ValueError('棋盘格重复角点数据无效。')
    frames,count,_=stack.shape
    distances=np.linalg.norm(stack[:,None]-stack[None,:],axis=-1)
    motion=np.linalg.norm(np.median(stack[:,None]-stack[None,:],axis=2),axis=-1)
    report={'passed':False,'limit_px':float(tolerance),
        'raw_max_px':float(distances.max()),'whole_frame_motion_px':float(motion.max()),
        'accepted_max_px':0.,'excluded':[],'unresolved_corner_indices':[],
        'max_excluded_per_frame':max(1,count//10)}
    average=stack.mean(0)
    for corner in range(count):
        pair=distances[:,:,corner]
        groups=[]
        # A strict majority must agree pairwise within the original limit.
        for size in range(frames,frames//2,-1):
            groups=[group for group in combinations(range(frames),size)
                    if pair[np.ix_(group,group)].max()<=tolerance]
            if groups: break
        if len(groups)!=1:
            report['unresolved_corner_indices'].append(corner)
            continue
        keep=groups[0]
        average[corner]=stack[list(keep),corner].mean(0)
        report['accepted_max_px']=max(report['accepted_max_px'],float(pair[np.ix_(keep,keep)].max()))
        for frame in set(range(frames))-set(keep):
            report['excluded'].append({'frame_index':frame,'corner_index':corner,
                'camera_px':stack[frame,corner].tolist(),
                'distance_to_consensus_px':float(np.linalg.norm(stack[frame,corner]-average[corner]))})
    removed=[sum(item['frame_index']==i for item in report['excluded']) for i in range(frames)]
    report['excluded_per_frame']=removed
    if motion.max()>tolerance:
        report['error']='多角点共同移动，不能按单点误识别排除。'
    elif report['unresolved_corner_indices']:
        report['error']='部分角点没有唯一的多数帧一致位置；请增加到至少 3 次重复采集或改善成像。'
    elif max(removed)>report['max_excluded_per_frame']:
        report['error']='同一帧异常角点过多，请检查图像清晰度、振动和曝光。'
    else:
        report['passed']=True
    return average,report


def solve(config, physical, measured):
    candidates=[]
    for origin in orientations(measured[0],config):
        matrix=fit_mapping(physical[0],origin,config.geometry_mapping)
        ordered=[origin]; axis_errors=[]
        for index in (1,2):
            expected=project(matrix,physical[index])
            choices=orientations(measured[index],config)
            errors=[np.linalg.norm(p-expected,axis=1) for p in choices]
            best=int(np.argmin([e.mean() for e in errors]))
            ordered.append(choices[best]);axis_errors.extend(errors[best])
        candidates.append((float(np.max(axis_errors)),matrix,ordered))
    candidates.sort(key=lambda item:item[0])
    score,matrix,ordered=candidates[0]
    if score>config.geometry_tolerance:
        raise ValueError(f'棋盘格 X/Y 移动检查失败（最大 {score:.3f} px）；图案可能不是 SLM 面的稳定成像，或角点误配。')
    if candidates[1][0]-score<max(1.,config.geometry_tolerance):
        raise ValueError('棋盘格方向存在歧义；移动量太小或成像不稳定。')
    # This diagonal shift did not participate in fitting or direction selection.
    expected=project(matrix,physical[3])
    choices=orientations(measured[3],config)
    ordered.append(min(choices,key=lambda p:float(np.square(p-expected).sum())))
    errors=[np.linalg.norm(project(matrix,p)-q,axis=1) for p,q in zip(physical,ordered)]
    return matrix,ordered,errors


def run_checkerboard(job,config,rpc,emit):
    root=Path(job['output']);root.mkdir(parents=True,exist_ok=True)
    key=config.active_slms[0]; cell=config.checker_cell_px
    offsets=[(0,0),(cell//2,0),(0,cell//2),(-cell//3,cell//4)]
    patterns,physical=zip(*(board(config,offset) for offset in offsets))
    exposure=None; actual=None; roi=config.geometry_search_roi
    emit('log',message='棋盘格：中心角点拟合，X/Y 平移确定方向，另一次斜向移动独立验证。结果为 SLM 面映射，完整区域由中心外推，不自动用于全息回环。')

    def capture(pattern,name):
        nonlocal actual
        folder=root/'captures'/name;folder.mkdir(parents=True,exist_ok=True)
        path=folder/(key+'.png');Image.fromarray(pattern).save(path)
        payload={'paths':{key:str(path)},'capture_path':str(folder/'raw.npy')}
        if exposure is not None: payload['exposure_us']=exposure
        result=rpc(payload)
        if result.get('error'): raise RuntimeError(result['error'])
        actual=result.get('exposure_us',exposure)
        raw=np.load(result['path'],allow_pickle=False)
        # Preserve a canonical raw path even if a hardware adapter returns another.
        if Path(result['path']) != folder/'raw.npy': np.save(folder/'raw.npy',raw)
        save_capture_preview(folder,raw,config.camera_white)
        emit('preview',kind='camera_raw',path=str(folder/'raw.npy'))
        return raw,folder

    def select(raw,folder,index,reason,attempt):
        nonlocal exposure
        token=f'{folder.name}:{attempt}'
        response=rpc({'request_type':'geometry_select','request_id':token,
            'path':str(folder/'raw.npy'),'point_index':index,'point_count':4,
            'repeat_index':0,'target_um':[0,0], 'reason':reason,
            'exposure_us':exposure})
        if response.get('error'): raise RuntimeError(response['error'])
        if response.get('request_id')!=token: raise ValueError('棋盘格框选回复与当前图像不匹配。')
        (folder/f'selection_{attempt:02d}.json').write_text(json.dumps(response,ensure_ascii=False,indent=2),encoding='utf-8')
        if response.get('action')=='cancel': raise InterruptedError('用户取消棋盘格标定。')
        if response.get('action')=='recapture':
            value=float(response['exposure_us'])
            if not np.isfinite(value) or value<=0: raise ValueError('曝光必须为有限正数。')
            exposure=value
            return None
        if response.get('action')!='select': raise ValueError('未知框选操作。')
        selected=response['roi'];search_rect(selected,raw.shape,job.get('hardware'))
        return selected

    raw,folder=capture(patterns[0],'initial')
    search_rect(roi,raw.shape,job.get('hardware') if roi else None)
    if config.geometry_assistance=='manual':
        attempt=0
        while True:
            roi=select(raw,folder,0,'请框住中心小棋盘全部角点及白边，并为后续 X/Y 移动预留空间。',attempt)
            if roi is not None: break
            attempt+=1;raw,folder=capture(patterns[0],f'initial_{attempt}')
    if config.geometry_auto_exposure:
        camera=job.get('hardware',{}).get('camera',{})
        records=[]
        def auto_capture(pattern,name,value):
            nonlocal exposure
            exposure=value;raw,_=capture(pattern,name)
            if actual is None: raise ValueError('相机未返回实际曝光。')
            return raw,actual
        def record(value):
            records.append(value)
            (root/'exposure_preparation.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        limits=camera.get('exposure_range')
        if limits: limits=[limits[0],min(limits[1],config.capture_timeout*1e6/max(config.average_frames+config.discard_frames+2,1))]
        exposure=prepare_exposure(auto_capture,[patterns[0]],exposure or config.exposure_for(camera.get('exposure',config.geometry_exposure_us)),
            limits,config.camera_white,config.geometry_exposure_target,roi,record)
        emit('geometry_exposure_locked',exposure_us=exposure)
    measured=[]; raw_shape=None; stability=[]; repeat_reports=[]
    for index,pattern in enumerate(patterns):
        observations=[];attempt=0
        for repeat in range(config.geometry_repeats):
            while True:
                name=f'board_{index:02d}_repeat_{repeat:02d}_attempt_{attempt:02d}'
                raw,folder=capture(pattern,name)
                if raw_shape is not None and list(raw.shape[:2])!=raw_shape: raise ValueError('棋盘格采集期间相机分辨率发生变化。')
                raw_shape=list(raw.shape[:2])
                try:
                    points=detect(raw,config,roi)
                    break
                except ValueError as exc:
                    (folder/'error.txt').write_text(str(exc),encoding='utf-8')
                    if config.geometry_assistance=='automatic': raise
                    selected=select(raw,folder,index,str(exc),attempt)
                    if selected is not None: roi=selected
                    attempt+=1
            if observations:
                points=min(orientations(points,config),key=lambda p:float(np.square(p-observations[0]).sum()))
            observations.append(points)
            canvas=np.uint8(np.clip(gray_channel(raw)/config.camera_white*255,0,255))
            canvas=cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(canvas,(config.checker_columns,config.checker_rows),points.astype(np.float32).reshape(-1,1,2),True)
            Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)).save(folder/'corners.png')
            (folder/'corners.json').write_text(json.dumps({'camera_px_detector_order':points.tolist(),
                'note':'Raw detector order; oriented correspondences are in geometry.json.'},indent=2),encoding='utf-8')
        position,report=repeat_consensus(observations,config.geometry_spot_stability)
        repeat_reports.append(report)
        (root/f'board_{index:02d}_repeatability.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if not report['passed']:
            raise ValueError(f'棋盘格重复采集不稳定：原始最大 {report["raw_max_px"]:.3f} px。'+report['error'])
        stability.append(report['accepted_max_px'])
        if report['excluded']:
            emit('log',message=f'棋盘格 {index+1}：多数帧确认后排除 {len(report["excluded"])} 个单次角点误识别；'
                f'保留观测最大变化 {report["accepted_max_px"]:.3f} px，限值仍为 {config.geometry_spot_stability:g} px。')
        measured.append(position)
        emit('progress',value=(index+1)*23,message=f'棋盘格 {index+1}/4')
    (root/'observations.json').write_text(json.dumps({'physical_um':[p.tolist() for p in physical],
        'camera_px_detector_order':[p.tolist() for p in measured]},indent=2),encoding='utf-8')
    try:
        matrix,ordered,errors=solve(config,physical,measured)
    except ValueError as exc:
        (root/'validation_error.txt').write_text(str(exc),encoding='utf-8')
        raise
    panel=replace(config,target_width_um=config.shape(key)[1]*getattr(config,key+'_pitch_um'),
                  target_height_um=config.shape(key)[0]*getattr(config,key+'_pitch_um'))
    info=roi_info(matrix,panel,raw_shape)
    passed=bool(max(float(e.max()) for e in errors)<=config.geometry_tolerance)
    local_hull=cv2.convexHull(np.concatenate(physical).astype(np.float32)).reshape(-1,2)
    result={'schema':1,'kind':'slm_pattern_geometry','validated':passed,'coordinate_space':'slm_surface_um',
        'matrix':matrix.tolist(),'raw_shape':raw_shape,'roi':info,'hardware':job.get('hardware'),
        'config':asdict(config),'panel_size_um':[panel.target_width_um,panel.target_height_um],
        'physical_points_um':[p.tolist() for p in physical], 'camera_points_px':[p.tolist() for p in ordered],
        'training_errors_px':errors[0].tolist(),'validation_errors_px':np.concatenate(errors[1:]).tolist(),
        'independent_errors_px':errors[3].tolist(),'repeatability_px':stability,
        'repeatability_details':repeat_reports,
        'measured_hull_um':local_hull.tolist(),'full_panel_extrapolated':True,
        'target_validation':{'passed':False,'error':'SLM 面棋盘格映射未验证全息目标平面。'},
        'locked_exposure_us':exposure,
        'interpretation':'SLM surface coordinates to raw camera pixels; full panel ROI extrapolated from central corners. Not a holographic target-plane map.'}
    path=root/'geometry.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    emit('result',path=str(path),message=f'棋盘格独立验证最大误差 {errors[3].max():.3f} px；'+
        ('SLM 面映射通过。' if passed else 'SLM 面映射未通过。')+'完整面板 ROI 为外推参考，不能直接用于目标平面回环。')
    return result
