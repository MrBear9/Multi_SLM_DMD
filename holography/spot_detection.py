"""Compact focal-spot localization; never used for photometric training checks."""
import math

import cv2
import numpy as np


def search_rect(document, shape, hardware=None):
    h, w = shape[:2]
    if not document:
        return 0, 0, w, h
    if document.get('coordinates') != 'raw_frame_xywh' or document.get('frame_size') != [w,h]:
        raise ValueError('定位搜索 ROI 的原始帧尺寸或坐标系已改变，请从第三页重新读取。')
    values = [document.get('roi',{}).get(k) for k in ('x','y','width','height')]
    if any(type(v) is not int for v in values):
        raise ValueError('定位搜索 ROI 需要整数坐标。')
    x,y,rw,rh = values
    if min(x,y)<0 or min(rw,rh)<8 or x+rw>w or y+rh>h:
        raise ValueError('定位搜索 ROI 超出原始画面或尺寸太小。')
    if hardware is not None:
        camera = hardware.get('camera',{})
        identity = camera.get('camera',{})
        stored_identity = document.get('camera',{})
        if not any(stored_identity.get(k) and stored_identity.get(k) == identity.get(k)
                   for k in ('serial','id')):
            raise ValueError('定位搜索 ROI 的相机身份无法匹配，请重新读取第三页 ROI。')
        for key in ('serial','id','model'):
            stored = stored_identity.get(key)
            if stored != identity.get(key):
                raise ValueError('定位搜索 ROI 的相机身份无法匹配，请重新读取第三页 ROI。')
        if document.get('camera_geometry') != camera.get('geometry'):
            raise ValueError('定位搜索 ROI 的采集分辨率或硬件 ROI 已改变，请重新读取。')
    return x,y,rw,rh


def _gaussian(parameters, x, y):
    a,cx,cy,sx,sy,r,b,bx,by = parameters
    dx=(x-cx)/np.exp(sx);dy=(y-cy)/np.exp(sy);rho=np.tanh(r)
    exponent=(dx*dx+dy*dy-2*rho*dx*dy)/(2*(1-rho*rho))
    return np.exp(a-exponent)+b+bx*x+by*y


def _fit(values, x, y, valid, initial, iterations=24):
    # Small bounded least-squares fit. No scipy dependency in the GUI runtime.
    p=initial.copy(); xx=x[valid];yy=y[valid];z=values[valid]
    if len(z)<24:
        raise ValueError('未饱和的光斑外围数据不足。')
    damping=.01
    for _ in range(iterations):
        pred=_gaussian(p,xx,yy);res=z-pred
        jac=[]
        for i in range(9):
            step=1e-4*max(1.,abs(p[i]));q=p.copy();q[i]+=step
            jac.append((_gaussian(q,xx,yy)-pred)/step)
        j=np.array(jac).T
        # Huber weights reduce isolated bad pixels without hiding broad residuals.
        scale=max(float(np.median(np.abs(res)))*1.4826, np.exp(p[0])*.025, .01)
        weights=np.minimum(1.,2.5*scale/np.maximum(abs(res),1e-12))
        normal=j.T@(j*weights[:,None]);rhs=j.T@(res*weights)
        try:dp=np.linalg.solve(normal+damping*np.diag(np.maximum(np.diag(normal),1e-6)),rhs)
        except np.linalg.LinAlgError:break
        q=p+dp
        q[0]=np.clip(q[0],-12,20);q[1:3]=np.clip(q[1:3],-2,2)
        q[3:5]=np.clip(q[3:5],-7,2);q[5]=np.clip(q[5],-1.2,1.2)
        if np.mean((z-_gaussian(q,xx,yy))**2)<np.mean(res**2):
            p=q;damping=max(1e-6,damping/3)
            if np.linalg.norm(dp)<1e-6:break
        else:damping=min(1e6,damping*5)
    return p


def _analyze(image, signed, compact, box, max_fwhm, white, stability):
    x,y,w,h=box
    if max(w,h)>max_fwhm*2.2:
        raise ValueError('亮区过大，未形成紧凑定位光斑。')
    pad=max(5,int(max(w,h)*.9))
    x0=max(0,x-pad);y0=max(0,y-pad);x1=min(image.shape[1],x+w+pad);y1=min(image.shape[0],y+h+pad)
    if x0==0 or y0==0 or x1==image.shape[1] or y1==image.shape[0]:
        raise ValueError('定位光斑外围被相机或搜索 ROI 边界截断。')
    raw=image[y0:y1,x0:x1];data=signed[y0:y1,x0:x1].astype(float)
    weight=compact[y0:y1,x0:x1].copy()
    yy,xx=np.indices(weight.shape,dtype=float)
    # Initial moments only use the central candidate, not the broad pedestal.
    inside=(xx>=x-x0)&(xx<x+w-x0)&(yy>=y-y0)&(yy<y+h-y0)
    weight*=inside;total=weight.sum()
    cx=(xx*weight).sum()/total;cy=(yy*weight).sum()/total
    scale=max(data.shape)/2
    gx=(xx-cx)/scale;gy=(yy-cy)/scale
    sx=max(1.,w/3)/scale;sy=max(1.,h/3)/scale
    baseline=float(np.median(data[~inside]))
    p=np.array([np.log(max(float(data.max())-baseline,1e-6)),0,0,np.log(sx),np.log(sy),0,baseline,0,0])
    sat=raw>=white*.99
    fraction=float(np.mean(sat[inside]))
    if fraction>.35:
        raise ValueError('定位光斑饱和范围过大，外围不足以可靠定位，请降低曝光。')
    # Uniform subsampling bounds fitting work for full-resolution cameras.
    step=max(1,int(math.ceil(math.sqrt(data.size/5000))))
    sampled=np.zeros(data.shape,bool);sampled[::step,::step]=True
    valid=sampled & ~sat
    p=_fit(data,gx,gy,valid,p)
    amplitude=float(np.exp(p[0]));sigma=np.exp(p[3:5])*scale
    rho=np.tanh(p[5]);cov=np.array([[sigma[0]**2,rho*np.prod(sigma)],[rho*np.prod(sigma),sigma[1]**2]])
    eig=np.linalg.eigvalsh(cov);fwhm=2.35482*np.sqrt(eig)
    center=np.array([cx+p[1]*scale,cy+p[2]*scale])
    if fwhm.max()>max_fwhm or fwhm.min()<1.3 or fwhm.max()/fwhm.min()>4:
        raise ValueError('光斑尺寸或形状不符合紧凑聚焦点，请检查聚焦位置。')
    if not (x-x0<=center[0]<x+w-x0 and y-y0<=center[1]<y+h-y0):
        raise ValueError('拟合中心离开候选亮区。')
    prediction=_gaussian(p,gx,gy)
    plane=p[6]+p[7]*gx+p[8]*gy
    gaussian=prediction-plane
    relevant=(gaussian>.08*amplitude)&~sat
    if relevant.sum()<24:
        raise ValueError('未饱和光斑外围不足。')
    residual=float(np.sqrt(np.mean((data[relevant]-prediction[relevant])**2))/max(amplitude,1e-12))
    if residual>.18:
        raise ValueError('亮区不是稳定的单个光斑：外围拟合残差过大。')
    sectors=((np.arctan2(yy-center[1],xx-center[0])+np.pi)/(2*np.pi)*8).astype(int)%8
    wings=(gaussian>.1*amplitude)&(gaussian<.75*amplitude)&~sat
    if sum(np.count_nonzero(wings&(sectors==i))>=2 for i in range(8))<7:
        raise ValueError('光斑未饱和外围覆盖不完整，不能可靠推断中心。')
    second=_fit(data,gx,gy,valid&(gaussian<.75*amplitude),p,iterations=12)
    shift=float(np.linalg.norm(second[1:3]-p[1:3])*scale)
    if shift>stability:
        raise ValueError('光斑中心随拟合区域变化，定位不稳定。')
    # A saturated core must actually lie near the fitted peak, not in a second lobe.
    if np.any(sat&inside&(gaussian<.5*amplitude)):
        raise ValueError('饱和区域偏离光斑中心，不能使用外围拟合定位。')
    return dict(position_xy=[float(x0+center[0]),float(y0+center[1])],
                fwhm_px=fwhm.tolist(),fit_residual=residual,fit_center_shift_px=shift,
                saturation_fraction=fraction,saturated_pixels=int(np.count_nonzero(sat&inside)),
                fitted_amplitude=amplitude,method='unsaturated_wings_gaussian' if sat[inside].any() else 'gaussian_with_planar_background')


def locate_spot(raw, background, white, diagnostics=None, *, max_fwhm=0., stability=1., search_roi=None):
    report=diagnostics if diagnostics is not None else {}
    report.update(status='rejected',white_level=float(white),candidates=[],detector_version=2,
                  ambiguity_ratio_limit=.45,minimum_component_area=3)
    raw=np.asarray(raw);background=np.asarray(background)
    image=raw[...,1] if raw.ndim==3 else raw
    base=background[...,1] if background.ndim==3 else background
    if image.ndim!=2 or image.shape!=base.shape or not np.isfinite(image).all() or not np.isfinite(base).all():
        raise ValueError('采集尺寸改变或包含无效像素。')
    if not np.isfinite(white) or white<=0 or (np.issubdtype(raw.dtype,np.integer) and white>np.iinfo(raw.dtype).max):
        raise ValueError('白电平超过相机位深范围或无效。')
    if image.max()>white*1.01:
        raise ValueError('定位数据超过白电平，无法判断过曝，请核对实际位深。')
    rx,ry,rw,rh=search_rect(search_roi,image.shape)
    limit=float(max_fwhm or max(12,min(160,min(image.shape)*.04)))
    signed=image.astype(np.float32)-base.astype(np.float32)
    delta=np.maximum(signed,0);smooth=cv2.GaussianBlur(delta,(5,5),1)
    # Estimate broad illumination at a bounded image size, then subtract it.
    shrink=min(1.,1536/max(image.shape));small=cv2.resize(smooth,None,fx=shrink,fy=shrink,interpolation=cv2.INTER_AREA)
    broad=cv2.GaussianBlur(small,(0,0),max(2.,limit/2*shrink))
    broad=cv2.resize(broad,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)
    compact=np.maximum(smooth-broad,0)
    scope=compact[ry:ry+rh,rx:rx+rw]
    peak=float(scope.max());noise=max(float(np.median(np.abs(scope-np.median(scope))))*1.4826,white*1e-4)
    threshold=max(peak*.3,noise*8)
    mask=np.zeros(image.shape,np.uint8);mask[ry:ry+rh,rx:rx+rw]=np.uint8(scope>threshold)
    report.update(delta=delta,smooth=smooth,compact=compact,mask=mask,peak=peak,noise=noise,
                  component_threshold=threshold,signal_threshold=max(noise*12,white*.005),
                  max_fwhm_px=limit,search_roi_xywh=[rx,ry,rw,rh],center_stability_limit_px=stability,
                  saturation_fraction_limit=.35)
    if peak<max(noise*12,white*.005):
        raise ValueError('定位光斑信号不足；请检查照明和传播距离。')
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    report['components_before_area_filter']=count-1
    components=[]
    for i in range(1,count):
        x,y,w,h,area=stats[i]
        if area<3:continue
        weights=compact[y:y+h,x:x+w]*(labels[y:y+h,x:x+w]==i)
        components.append((float(weights.max()),i,int(x),int(y),int(w),int(h),int(area),float(weights.sum())))
    components.sort(reverse=True)
    report['candidate_count']=len(components);report['candidate_fit_limit']=32
    candidates=[]
    for rank,(strength,i,x,y,w,h,area,total) in enumerate(components[:32],1):
        candidate=dict(rank=rank,component_id=i,bbox_xywh=[x,y,w,h],area_px=area,total_signal=total,
                       relative_signal=strength/max(components[0][0],1e-12),peak_signal=strength,
                       centroid_xy=[x+(w-1)/2,y+(h-1)/2],quality='rejected')
        try:
            fit=_analyze(image[ry:ry+rh,rx:rx+rw],signed[ry:ry+rh,rx:rx+rw],scope,
                         (x-rx,y-ry,w,h),limit,white,stability)
            fit['position_xy'][0]+=rx;fit['position_xy'][1]+=ry
            candidate.update(fit,quality='accepted',centroid_xy=fit['position_xy'])
            candidates.append(candidate)
        except ValueError as exc:candidate['rejection_reason']=str(exc)
        report['candidates'].append(candidate)
    if not candidates:
        reasons=list(dict.fromkeys(c['rejection_reason'] for c in report['candidates']))
        raise ValueError('没有可靠的紧凑定位光斑。'+'；'.join(reasons[:3]))
    candidates.sort(key=lambda c:c['peak_signal'],reverse=True)
    chosen=candidates[0]
    ratio=candidates[1]['peak_signal']/chosen['peak_signal'] if len(candidates)>1 else None
    report['runner_up_ratio']=ratio
    if ratio is not None and ratio>.45:
        raise ValueError('多个相近强度光斑或定位不明确，请设置搜索 ROI 或调整光路。')
    # Too many strong unexamined peaks must not silently become a unique match.
    if len(components)>32 and components[32][0]>chosen['peak_signal']*.45:
        raise ValueError('强候选过多，无法确认唯一定位光斑，请限制搜索区域。')
    report.update(status='accepted',selected_rank=chosen['rank'],position_xy=chosen['position_xy'],
                  localization=chosen)
    return chosen['position_xy']
