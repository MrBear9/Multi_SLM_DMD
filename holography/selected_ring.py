"""Refine a user-selected ring boundary; the hand-drawn centre is only a seed."""
import cv2
import numpy as np
from .diagnostics import gray_channel
from .spot_detection import search_rect


def _ellipse(points):
    if len(points)<20: raise ValueError('手选圆环有效边缘不足。')
    return cv2.fitEllipse(np.asarray(points,np.float32).reshape(-1,1,2))


def _residual(points, ellipse):
    (cx,cy),(w,h),angle=ellipse
    theta=np.deg2rad(angle);c,s=np.cos(theta),np.sin(theta)
    delta=points-[cx,cy]
    xy=delta@np.array([[c,-s],[s,c]])
    return np.abs(np.hypot(xy[:,0]/(w/2),xy[:,1]/(h/2))-1)*min(w,h)/2


def locate_selected_ring(raw,background,white,report,*,stability,search_roi):
    image=gray_channel(raw).astype(np.float32);base=gray_channel(background).astype(np.float32)
    report.update(detector='radial_rings',method='selected_ellipse',status='rejected',white_level=float(white))
    if image.shape!=base.shape or not np.isfinite(image).all() or not np.isfinite(base).all() or white<=0 or image.max()>white*1.01:
        raise ValueError('圆环采集、背景或白电平无效。')
    x,y,w,h=search_rect(search_roi,image.shape)
    if min(w,h)<24: raise ValueError('请用椭圆框圈出一条完整环纹，直径至少 24 原始相机像素。')
    report['search_roi_xywh']=[x,y,w,h];report['delta']=np.maximum(image-base,0)
    seed=np.array([x+w/2,y+h/2]);axes=np.array([w/2,h/2])
    # Work locally but preserve access to a narrow band outside the drawn ring.
    left=max(0,int(seed[0]-axes[0]*1.35));top=max(0,int(seed[1]-axes[1]*1.35))
    right=min(image.shape[1],int(seed[0]+axes[0]*1.35)+1);bottom=min(image.shape[0],int(seed[1]+axes[1]*1.35)+1)
    data=image[top:bottom,left:right]-base[top:bottom,left:right]
    smooth=cv2.GaussianBlur(data,(0,0),3.)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)
    angles=np.linspace(0,2*np.pi,144,endpoint=False)
    scale=np.linspace(.72,1.28,max(40,int(min(axes)*1.12)))
    dx=axes[0]*np.cos(angles);dy=axes[1]*np.sin(angles)
    mx=seed[0]-left+scale[:,None]*dx;my=seed[1]-top+scale[:,None]*dy
    polar=cv2.remap(smooth,mx.astype(np.float32),my.astype(np.float32),cv2.INTER_LINEAR)
    derivative=np.gradient(polar,axis=0)
    radial_profile=np.median(polar,axis=1)
    radial_derivative=np.gradient(radial_profile)
    report['radial_profile']=radial_profile.tolist()
    report['edge_fit_attempts']=[]
    solutions=[]
    for sign in (-1,1):
        # Select a coherent ring radially before tracking its edge by angle;
        # strong horizontal/vertical lines cannot independently choose radii.
        coherent=sign*radial_derivative*np.exp(-((scale-1)/.22)**2)
        selected=int(np.argmax(coherent))
        band=np.abs(scale-scale[selected])<max(.04,5/min(axes))
        score=np.where(band[:,None],sign*derivative,-np.inf)
        indices=score.argmax(0);a=np.arange(len(angles))
        peak=sign*derivative[indices,a]
        px=seed[0]+scale[indices]*dx;py=seed[1]+scale[indices]*dy
        valid=(indices>1)&(indices<len(scale)-2)&(peak>max(float(coherent[selected])*.2,.04))
        valid&=(px>2)&(py>2)&(px<image.shape[1]-3)&(py<image.shape[0]-3)
        ix=np.clip(np.rint(px).astype(int),0,image.shape[1]-1);iy=np.clip(np.rint(py).astype(int),0,image.shape[0]-1)
        valid&=image[iy,ix]<white*.99
        mxp=(px-left).astype(np.float32)[None,:];myp=(py-top).astype(np.float32)[None,:]
        ex=cv2.remap(gx,mxp,myp,cv2.INTER_LINEAR).ravel()
        ey=cv2.remap(gy,mxp,myp,cv2.INTER_LINEAR).ravel()
        nx=(px-seed[0])/axes[0]**2;ny=(py-seed[1])/axes[1]**2
        alignment=np.abs(ex*nx+ey*ny)/np.maximum(np.hypot(ex,ey)*np.hypot(nx,ny),1e-8)
        valid&=alignment>.8
        points=np.column_stack((px,py));keep=valid.copy()
        try:
            for _ in range(5):
                fitted=_ellipse(points[keep]);res=_residual(points,fitted)
                limit=max(1.,min(3.,float(np.median(res[keep]))*2.5))
                keep=valid&(res<limit)
            fitted=_ellipse(points[keep]);res=_residual(points,fitted)
            sectors=np.unique((angles[keep]*12/(2*np.pi)).astype(int))
            center=np.array(fitted[0]);sizes=np.array(fitted[1])/2
            report['edge_fit_attempts'].append({'polarity':sign,'radius_fraction':float(scale[selected]),'points':int(keep.sum()),'sectors':len(sectors),'center_xy':center.tolist(),'diameters_px':list(map(float,fitted[1]))})
            if len(sectors)<10 or keep.sum()<72 or np.linalg.norm((center-seed)/axes)>.3: continue
            if min(sizes)<min(axes)*.65 or max(sizes)>max(axes)*1.4: continue
            # Interleaved angular groups independently estimate the same centre.
            labels=np.flatnonzero(keep)
            c1=np.array(_ellipse(points[labels[::2]])[0]);c2=np.array(_ellipse(points[labels[1::2]])[0])
            spread=float(np.linalg.norm(c1-c2))
            rms=float(np.sqrt(np.mean(res[keep]**2)))
            if spread>stability or rms>max(1.,min(axes)*.025): continue
            solutions.append((int(keep.sum()),center,fitted,spread,rms,len(sectors),points[keep]))
        except (ValueError,cv2.error):
            continue
    if not solutions:
        raise ValueError('手选环边缘不完整或拟合不稳定。请让红色椭圆贴近一条完整明暗环边界，避开横竖条纹；不要只圈中心亮点。')
    _,center,fitted,spread,rms,sectors,points=max(solutions,key=lambda s:s[0])
    # Two contrast polarities can delimit one thick ring; their centres must agree.
    if len(solutions)>1 and np.linalg.norm(solutions[0][1]-solutions[1][1])>stability:
        raise ValueError('手选环的内外边缘中心不一致，请重新选择完整内环。')
    position=center.tolist();radius=float(min(fitted[1])/2)
    report.update(status='accepted',selected_rank=1,position_xy=position,analysis_radius_px=radius,
        selected_ellipse={'center_xy':position,'diameters_px':list(map(float,fitted[1])),'angle_deg':float(fitted[2])},
        edge_points_xy=points.tolist(),angular_sectors=sectors,edge_rms_px=rms,
        candidates=[{'rank':1,'bbox_xywh':[x,y,w,h],'centroid_xy':position,'quality':'selected ring','peak_signal':float(data.max())}],
        localization={'center_xy':position,'band_center_difference_px':spread,'radius_px':radius,
                      'angular_sectors':sectors,'edge_rms_px':rms,'method':'selected_ellipse'})
    return position
