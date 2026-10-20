"""Image-guided exposure preparation, locked before geometric measurements."""
import cv2
import numpy as np
from .diagnostics import gray_channel
from .spot_detection import search_rect


def brightness(raw, white, roi=None):
    image=gray_channel(raw).astype(np.float32)
    if not np.isfinite(image).all() or image.max()>white*1.01:
        raise ValueError('自动曝光：采集值无效或相机白电平不匹配。')
    x,y,w,h=search_rect(roi,image.shape)
    image=image[y:y+h,x:x+w]
    # Suppress isolated defective pixels, but retain small real focal spots.
    peak=float(cv2.GaussianBlur(image,(3,3),.6).max())/white
    return peak,float(np.mean(image>=white*.99))


def prepare_exposure(capture, phases, current, limits, white, target, roi, record):
    if not limits or len(limits)<2 or not 0<limits[0]<=limits[1]:
        raise ValueError('自动准备曝光需要 SDK 提供曝光范围；当前相机请使用手动曝光。')
    lower,upper=map(float,limits[:2])
    # Very long exposures cannot finish the existing capture transaction.
    exposure=float(np.clip(current,lower,upper));stable=0
    for iteration in range(14):
        metrics=[];actual=[]
        for index,phase in enumerate(phases):
            raw,readback=capture(phase,f'exposure_{iteration:02d}_{index:02d}',exposure)
            peak,saturated=brightness(raw,white,roi)
            metrics.append({'peak_fraction':peak,'saturated_fraction':saturated})
            actual.append(float(readback))
        if max(actual)-min(actual)>max(1.,exposure*.01):
            raise ValueError('自动曝光准备期间相机曝光读回不稳定。')
        exposure=float(np.mean(actual))
        peak=max(m['peak_fraction'] for m in metrics)
        saturated=max(m['saturated_fraction'] for m in metrics)
        acceptable=target*.85<=peak<=min(.92,target*1.15) and saturated<.001
        stable=stable+1 if acceptable else 0
        record({'iteration':iteration,'exposure_us':exposure,'probes':metrics,'stable_count':stable})
        if stable>=2:
            return exposure
        if acceptable:
            continue
        factor=float(np.clip(target/max(peak,.001),.25,4.))
        if saturated>=.001:factor=min(factor,.7)
        proposed=float(np.clip(exposure*factor,lower,upper))
        if abs(proposed-exposure)<max(.01,exposure*.001):
            raise ValueError('自动曝光已到相机允许范围仍未达到亮度目标；请检查光源/衰减或使用手动曝光。')
        exposure=proposed
    raise ValueError('自动曝光在限定次数内未稳定；采集记录已保留，请检查光源稳定性或使用手动曝光。')
