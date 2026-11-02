"""Two-region interferometric LUT scan with forward/reverse validation.

Requires an optical arrangement that overlaps reference and test waves into
straight carrier fringes on the camera. A normal SLM image/ring is not enough.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .lut import save_profile
from .spot_detection import search_rect


def split_pattern(config, gray):
    image = np.zeros(config.shape(config.mode),np.uint8)
    if config.lut_split == 'vertical':
        image[:,image.shape[1]//2:] = gray
    else:
        image[image.shape[0]//2:,:] = gray
    return image


class FringeMeter:
    def __init__(self, sample, white, dark, carrier=(0.,0.)):
        self.shape = sample.shape
        h,w = self.shape
        if min(h,w)<32:
            raise ValueError('LUT 条纹 ROI 至少需要 32×32 原始像素。')
        if h*w>1024*1024:
            raise ValueError('LUT 条纹 ROI 请控制在 100 万像素以内，选择局部清晰条纹即可。')
        self.white,self.dark = white,dark
        y,x = np.meshgrid(np.arange(h)/h,np.arange(w)/w,indexing='ij')
        self.weight = np.outer(np.hanning(h),np.hanning(w))
        self.plane = np.stack((np.ones_like(x),x-.5,y-.5),-1).reshape(-1,3)
        self.plane_inverse = np.linalg.pinv(self.plane)
        values = self.detrend(sample)
        cx,cy = carrier
        if cx == cy == 0:
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2(values*self.weight)))
            fy = np.fft.fftshift(np.fft.fftfreq(h))*h
            fx = np.fft.fftshift(np.fft.fftfreq(w))*w
            yy,xx = np.meshgrid(fy,fx,indexing='ij')
            mask = (np.hypot(xx,yy)>=4)&(np.abs(xx)<w/4)&(np.abs(yy)<h/4)&((xx>0)|((xx==0)&(yy>0)))
            spectrum[~mask]=0
            iy,ix = np.unravel_index(spectrum.argmax(),spectrum.shape)
            cx,cy = float(fx[ix]),float(fy[iy])
        if np.hypot(cx,cy)<3 or abs(cx)>=w/4 or abs(cy)>=h/4:
            raise ValueError('条纹载频需在 ROI 内至少 3 周期，且每周期至少 4 像素。')
        self.carrier = [cx,cy]
        angle = 2*np.pi*(cx*x+cy*y)
        self.design = np.column_stack((self.plane,np.cos(angle).ravel(),np.sin(angle).ravel()))
        weight = np.sqrt(self.weight.ravel())
        self.inverse = np.linalg.pinv(self.design*weight[:,None])*weight[None,:]

    def detrend(self, image):
        return (image.ravel()-self.plane@(self.plane_inverse@image.ravel())).reshape(self.shape)

    def measure(self, image):
        image = np.asarray(image,float)
        if image.shape!=self.shape or not np.isfinite(image).all():
            raise ValueError('LUT 采集尺寸变化或包含无效数值。')
        if np.mean(image>=self.white*.995)>.001:
            raise ValueError('LUT 条纹 ROI 存在饱和，请降低曝光后重新扫描。')
        coefficients = self.inverse@image.ravel()
        amplitude = np.hypot(*coefficients[-2:])
        dc = max(float(coefficients[0]-self.dark),1e-8)
        residual = image.ravel()-self.design@coefficients
        signal = self.detrend(image).ravel()
        error = float(np.sqrt(np.sum(self.weight.ravel()*residual**2)/max(np.sum(self.weight.ravel()*signal**2),1e-12)))
        visibility = float(amplitude/dc)
        if visibility<.05 or amplitude<(self.white-self.dark)*.005 or error>.65:
            raise ValueError(f'未检测到稳定直线干涉条纹（对比度 {visibility:.3f}，拟合残差 {error:.3f}）。请调整干涉光路/ROI/载频；不能使用普通圆环或亮度图。')
        return float(np.arctan2(-coefficients[-1],coefficients[-2])), {'visibility':visibility,'residual':error}


def fit_scans(grays, phases, negative, tolerance=.15):
    phases = np.asarray(phases,float)
    curves = np.unwrap(phases,axis=1)
    curves -= curves[:,:1]
    if np.max(np.abs(np.diff(curves,axis=1))) >= .7*np.pi:
        raise ValueError('相邻灰度相位差过大，存在解包裹歧义；请减小扫描步长。')
    repeat_error = float(np.max(np.ptp(curves,axis=0)))
    if repeat_error>tolerance:
        raise ValueError(f'正反扫描不一致（最大差 {repeat_error:.3f} rad），请检查漂移、稳定等待和液晶迟滞。')
    curve = curves.mean(axis=0)
    curve *= -1 if curve[-1]<0 else 1
    if not .5<=curve[-1]<=8*np.pi:
        raise ValueError('未测得有效相位扫描范围；请检查参考光与测试光是否真正干涉。')
    # Only remove small measurement noise; substantial nonmonotonicity fails.
    monotonic = np.maximum.accumulate(np.maximum(curve,0))
    correction = float(np.max(abs(monotonic-curve)))
    if correction>tolerance:
        raise ValueError('响应明显非单调，无法可靠生成单调 LUT；请检查条纹、漂移或缩小测量范围。')
    result = np.interp(np.arange(256),grays,monotonic)*(-1 if negative else 1)
    return result, {'repeat_max_error_rad':repeat_error,'monotonic_correction_rad':correction}


def run_lut(job, config, rpc, emit):
    root = Path(job['output']);root.mkdir(parents=True,exist_ok=True)
    grays = sorted(set(range(0,256,config.lut_gray_step))|{255})
    meter = None;rect = None;records=[];scans=[]
    total = len(grays)*2
    emit('log',message='独立 LUT：参考半区固定灰度 0，测试半区正反扫描；固定曝光。需要参考/测试光重叠形成直线干涉条纹。')
    try:
        for sweep,sequence in enumerate((grays,grays[::-1])):
            measured = {}
            for gray in sequence:
                folder = root/'captures'/f'scan_{sweep}_gray_{gray:03d}';folder.mkdir(parents=True)
                path = folder/(config.mode+'.png')
                Image.fromarray(split_pattern(config,gray)).save(path)
                response = rpc({'paths':{config.mode:str(path)},'capture_path':str(folder/'raw.npy')})
                if response.get('error'):raise RuntimeError(response['error'])
                raw = np.load(response['path'],allow_pickle=False)
                emit('preview',kind='camera_raw',path=str(folder/'raw.npy'))
                if raw.ndim not in (2,3):raise ValueError('LUT 需要二维相机强度图。')
                current = search_rect(config.lut_roi,raw.shape,job.get('hardware'))
                if rect is not None and current!=rect:raise ValueError('LUT 扫描期间 ROI 变化。')
                rect = current;x,y,w,h = rect
                sample = np.asarray(raw[y:y+h,x:x+w],float)
                saturated = sample>=config.camera_white*.995
                if saturated.ndim==3:saturated=np.any(saturated,axis=2)
                if np.mean(saturated)>.001:
                    raise ValueError('LUT 条纹 ROI 存在饱和，请降低曝光后重新扫描。')
                if sample.ndim==3:sample=sample.mean(axis=2)
                Image.fromarray(np.uint8(np.clip((sample-config.camera_dark)/(config.camera_white-config.camera_dark),0,1)*255)).save(folder/'roi.png')
                if meter is None:
                    meter = FringeMeter(sample,config.camera_white,config.camera_dark,(config.lut_carrier_x,config.lut_carrier_y))
                    emit('log',message=f'锁定条纹载频 {meter.carrier} 周期/ROI；后续不重新选频。')
                phase,quality = meter.measure(sample)
                measured[gray]=phase
                records.append({'sweep':sweep,'gray':gray,'wrapped_phase_rad':phase,**quality})
                (root/'measurements.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
                emit('progress',value=int(95*len(records)/total),message=f'LUT 扫描 {len(records)}/{total}，灰度 {gray}')
            scans.append([measured[g] for g in grays])
        curve,quality = fit_scans(grays,scans,config.lut_negative)
        profile = {'kind':'slm_phase_lut','version':1,'validated':True,'slm':config.mode,
            'wavelength_nm':config.wavelength_nm,'pitch_um':getattr(config,config.mode+'_pitch_um'),
            'phase_rad':curve.tolist(),'method':'split_reference_fourier_fringe',
            'sign_source':'user_negative' if config.lut_negative else 'user_positive',
            'hardware':job.get('hardware'),'roi':config.lut_roi,'carrier_cycles':meter.carrier,
            'gray_samples':grays,'quality':quality,'covers_2pi':bool(abs(curve[-1])>=2*np.pi),
            'scope':'global response at measured illumination/polarization; not spatially resolved'}
        save_profile(profile,root)
        if not profile['covers_2pi']:
            emit('log',message='实测调制范围不足 2π；保留真实范围，不强行拉伸，部分目标相位只能近似。')
        emit('progress',value=100,message='LUT 扫描完成。')
        emit('result',path=str(root/'lut.json'),message='LUT 已保存；请查看曲线、核对相位方向后手动应用。')
        return profile
    except Exception as exc:
        (root/'lut_failure.json').write_text(json.dumps({'error':str(exc),'measurements':records},ensure_ascii=False,indent=2),encoding='utf-8')
        raise
