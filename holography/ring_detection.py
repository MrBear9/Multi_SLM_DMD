"""Robust gradient-line intersection for locally concentric fringes.

Works on signed background subtraction: dark rings carry as much information
as bright rings. Coordinates always refer to the original camera frame.
"""
import cv2
import numpy as np

from .spot_detection import search_rect
from .diagnostics import gray_channel


def _solve(x, y, nx, ny, weights, center, radius):
    for _ in range(20):
        distance=np.hypot(x-center[0],y-center[1])
        residual=nx*(center[0]-x)+ny*(center[1]-y)
        # Tukey weights reject edges whose gradient does not point at the center.
        cutoff=np.maximum(2.,distance*.16)
        robust=np.maximum(0.,1-(residual/cutoff)**2)**2
        weight=weights*robust*(distance>radius*.12)*(distance<radius)
        a=np.column_stack((nx,ny)); b=nx*x+ny*y
        normal=a.T@(a*weight[:,None])
        if weight.sum()<1e-8 or np.linalg.cond(normal)>30:
            raise ValueError('圆环有效弧段不足或方向分布不完整，请重新框选中心及数圈环纹。')
        updated=np.linalg.solve(normal,a.T@(weight*b))
        if np.linalg.norm(updated-center)<.005:
            center=updated;break
        center=updated
    return center,weight


def _symmetry(data, initial, radius, band=(.15,.95), search=True):
    """Refine using 180-degree symmetry, also valid for concentric ellipses."""
    angles=np.linspace(0,2*np.pi,80,endpoint=False)
    radii=np.linspace(radius*band[0],radius*band[1],48)
    dx=(radii[:,None]*np.cos(angles)).astype(np.float32)
    dy=(radii[:,None]*np.sin(angles)).astype(np.float32)
    def score(c):
        if min(c[0],c[1],data.shape[1]-1-c[0],data.shape[0]-1-c[1])<radius*band[1]:
            return -1.
        values=cv2.remap(data,dx+np.float32(c[0]),dy+np.float32(c[1]),cv2.INTER_LINEAR)
        values=values-values.mean()
        values=np.clip(values,-np.std(values)*2.5,np.std(values)*2.5)
        opposite=np.roll(values,40,axis=1)
        return float(np.sum(values*opposite)/max(float(np.sum(values*values)),1e-9))
    center=initial.copy();best=score(center);evaluated=[]
    if search:
        step=max(2.,radius*.06)
        for y in np.arange(-radius*.35,radius*.35+step/2,step):
            for x in np.arange(-radius*.35,radius*.35+step/2,step):
                candidate=initial+[x,y];value=score(candidate)
                evaluated.append((candidate,value))
                if value>best:best=value;center=candidate
    for step in (2.,1.,.5,.25):
        for _ in range(6):
            candidates=[center+[dx,dy] for dx in (-step,0,step) for dy in (-step,0,step)]
            values=[score(c) for c in candidates];index=int(np.argmax(values))
            if values[index]<=best+1e-7:break
            center=candidates[index];best=values[index]
    if search and any(np.linalg.norm(candidate-center)>radius*.25 and value>max(.5,best*.95)
                      for candidate,value in evaluated):
        raise ValueError('搜索区域存在多个相近的对称中心，请排除周期网格并重新框选完整内环。')
    return center,best


def locate_ring(raw, background, white, diagnostics=None, *, radius=100., stability=1., search_roi=None):
    report=diagnostics if diagnostics is not None else {}
    if search_roi and search_roi.get('ring_selection')=='ellipse_boundary':
        from .selected_ring import locate_selected_ring
        return locate_selected_ring(raw,background,white,report,stability=stability,search_roi=search_roi)
    # Try complete inner bands first; never force outer panel edges into a fit.
    attempts=[];accepted=[]
    if search_roi:
        _,_,w,h=search_rect(search_roi,np.asarray(raw).shape)
        maximum=min(float(radius),min(w,h)*.48)
        radii=sorted(set(max(20.,maximum*f) for f in (.45,.65,.82,1.)))
    else:
        radii=[radius]
    for current in radii:
        local={}
        try:
            result=_locate_ring(raw,background,white,local,radius=current,stability=stability,search_roi=search_roi)
        except ValueError as exc:
            attempts.append({'radius_px':float(current),'error':str(exc)})
            report.update(local)
            last_error=exc
        else:
            attempts.append({'radius_px':float(current),'center_xy':result})
            accepted.append((current,result,local))
            if len(accepted)>1 and np.linalg.norm(np.asarray(result)-accepted[-2][1])<=stability:
                _,position,selected=accepted[-2]
                report.update(selected);report['inside_out_attempts']=attempts
                report['cross_radius_centers_xy']=[position,result]
                return position
    report['inside_out_attempts']=attempts
    report['status']='rejected'
    if accepted:
        raise ValueError('内圈搜索在不同半径下尚未得到一致中心；可勾选手动指定圆环边界。')
    raise last_error


def _locate_ring(raw, background, white, diagnostics=None, *, radius=100., stability=1., search_roi=None):
    report=diagnostics if diagnostics is not None else {}
    report.update(white_level=float(white),status='rejected',detector='radial_rings')
    image=gray_channel(raw).astype(np.float32)
    base=gray_channel(background).astype(np.float32)
    if image.shape!=base.shape or not np.isfinite(image).all() or not np.isfinite(base).all():
        raise ValueError('定位图与背景尺寸不一致或包含无效数值。')
    if not np.isfinite(white) or white<=0 or image.max()>white*1.01:
        raise ValueError('相机白电平与采集数据不一致。')
    # A coarse selection prevents arbitrary rectangle/speckle centers from being
    # mistaken for the optical ring. It is NOT a measured calibration point.
    if not search_roi:
        raise ValueError('圆环模式请先粗框中心及数圈完整环纹；可选择逐点辅助或失败时辅助。')
    ox,oy,w,h=search_rect(search_roi,image.shape)
    report['search_roi_xywh']=[ox,oy,w,h]
    data=image[oy:oy+h,ox:ox+w]-base[oy:oy+h,ox:ox+w]
    report['delta']=np.maximum(image-base,0)
    sigma=2.
    smooth=cv2.GaussianBlur(data,(0,0),sigma)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)/8
    mag=np.hypot(gx,gy)
    noise=max(float(np.median(abs(data-smooth)))*1.4826,.03)
    yy,xx=np.indices(data.shape,dtype=float)
    valid=(mag>max(noise/(2*sigma*sigma),float(np.percentile(mag,60))))
    valid&=(xx>4)&(yy>4)&(xx<w-5)&(yy<h-5)
    valid&=image[oy:oy+h,ox:ox+w]<white*.99
    if valid.sum()<80:
        raise ValueError('可用环纹边缘不足；检查曝光、饱和及粗选区域。')
    x,y=xx[valid],yy[valid]; g=mag[valid]
    nx,ny=-gy[valid]/g,gx[valid]/g
    weights=np.minimum(g,np.percentile(g,85))
    weights/=max(float(weights.max()),1e-9)
    limit=min(float(radius),min(w,h)*.48)
    # Multiple starts avoid relying on exact placement of the hand-drawn box.
    solutions=[]
    for dx,dy in ((0,0),(-.15,0),(.15,0),(0,-.15),(0,.15)):
        initial=np.array([w*.5+dx*limit,h*.5+dy*limit])
        try:
            center,weight=_solve(x,y,nx,ny,weights,initial,limit)
            solutions.append((float(weight.sum()),center,weight))
        except ValueError:
            pass
    if not solutions:
        raise ValueError('未找到方向完整的圆环中心。')
    _,center,weight=max(solutions,key=lambda item:item[0])
    # Remove the broad illumination pedestal; concentric elliptical rings are
    # centrally symmetric even when their gradients are not exactly radial.
    texture=smooth-cv2.GaussianBlur(smooth,(0,0),max(3.,limit*.2))
    center,symmetry=_symmetry(texture,center,limit)
    limit=min(limit,center[0]-4,w-5-center[0],center[1]-4,h-5-center[1])
    if limit<12:
        raise ValueError('圆环中心太靠近搜索区域边界，请扩大或重新框选。')
    report['symmetry_score']=symmetry
    distance=np.hypot(x-center[0],y-center[1])
    residual=nx*(center[0]-x)+ny*(center[1]-y)
    weight=weights*np.maximum(0.,1-(residual/np.maximum(2.,distance*.16))**2)**2
    weight*=(distance>limit*.12)&(distance<limit)
    angular=np.mod(np.arctan2(y-center[1],x-center[0]),2*np.pi)
    support=weight>weights*.35
    sectors=np.unique((angular[support]*12/(2*np.pi)).astype(int))
    fraction=float(weight.sum()/max(weights[(distance<limit)&(distance>limit*.12)].sum(),1e-9))
    report.update(radial_support=fraction,angular_sectors=len(sectors),analysis_radius_px=float(limit))
    if len(sectors)<10 or fraction<.07 or symmetry<.5:
        raise ValueError('环纹方向一致性不足，可能混入矩形边缘、亮斑或多个中心。')
    # Independent inner/outer bands must agree; do not accept a stable but
    # arbitrary intensity centroid of the illuminated rectangle.
    centers=[]
    for lo,hi in ((.18,.6),(.55,.95)):
        c,_=_symmetry(texture,center.copy(),limit,(lo,hi),search=False)
        centers.append(c)
    spread=float(np.linalg.norm(centers[0]-centers[1]))
    # Radial profile must contain multiple alternating bright/dark rings.
    angles=np.linspace(0,2*np.pi,120,endpoint=False)
    radii=np.arange(3.,limit-2)
    mx=(center[0]+radii[:,None]*np.cos(angles)).astype(np.float32)
    my=(center[1]+radii[:,None]*np.sin(angles)).astype(np.float32)
    polar=cv2.remap(smooth,mx,my,cv2.INTER_LINEAR)
    profile=np.median(polar,axis=1)
    blurred=cv2.GaussianBlur(profile[:,None],(1,5),.8).ravel()
    derivative=np.diff(blurred)
    turns=np.flatnonzero(derivative[:-1]*derivative[1:]<0)+1
    significant=[int(t) for t in turns if abs(float(blurred[t]-np.median(blurred[max(0,t-4):t+5])))>max(noise*.1,.03)]
    report.update(band_centers_xy=[(c+[ox,oy]).tolist() for c in centers],
                  band_center_difference_px=spread,radial_extrema=len(significant),noise_dn=noise)
    if len(significant)<4:
        raise ValueError('完整明暗环数量不足，不能可靠区分圆环与孤立亮斑。')
    if spread>stability:
        raise ValueError(f'内外环中心相差 {spread:.3f} px，超过 {stability:g} px；请缩小到完整内圈或检查环纹畸变。')
    position=(center+[ox,oy]).tolist()
    report.update(status='accepted',position_xy=position,selected_rank=1,
        candidates=[{'rank':1,'bbox_xywh':[ox,oy,w,h],'centroid_xy':position,
                     'quality':'ring center','peak_signal':float(data.max())}],
        localization={'center_xy':position,'band_center_difference_px':spread,
                      'angular_sectors':len(sectors),'radial_support':fraction,'radius_px':float(limit),
                      'symmetry_score':symmetry})
    return position
