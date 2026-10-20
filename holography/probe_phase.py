"""Native-pixel calibration probe phases shared by GUI and external worker.

Requires NumPy only; importing this module never loads the training runtime.
"""
import math
import numpy as np

from .geometry_limits import probe_frequency, LIMIT


def manual_fresnel_phase(config, period_px):
    """Quadratic phase, P is the center-to-first-2pi-ring radius in pixels.

    This explicit convention is independent of vendor-specific 'period' units.
    It is for manual display tests, not the calibrated propagation geometry.
    """
    if config.mode not in ('slm1','slm2'):
        raise ValueError('手动菲涅尔测试需要指定单片 SLM。')
    h,w=config.shape(config.mode)
    if type(h) is not int or type(w) is not int or not (8<=h<=1080 and 8<=w<=1920):
        raise ValueError('SLM 有效区域应为宽 8～1920、高 8～1080 像素。')
    if not math.isfinite(period_px) or not 1<=period_px<=10000:
        raise ValueError('手动周期参数需要在 1～10000 px 之间。')
    # Maximum axial phase slope, in cycles per native hardware pixel.
    if max(w-1,h-1)/period_px**2>=LIMIT:
        minimum=math.sqrt(max(w-1,h-1)/LIMIT)
        raise ValueError(f'手动菲涅尔相位超过像素采样带宽；当前有效区域的周期参数需大于 {minimum:.2f} px。')
    y,x=np.ogrid[:h,:w]
    cycles=-((x-(w-1)/2)**2+(y-(h-1)/2)**2)/period_px**2
    gray=np.remainder(cycles,1)
    if config.gray_inverted:
        gray=1-gray
    return np.uint8(np.rint(gray*255))


def focus_phase(config, point):
    if config.mode not in ('slm1','slm2'):
        raise ValueError('定位测试图案需要指定单片 SLM。')
    key = config.active_slms[0]
    h, w = config.shape(key)
    if type(h) is not int or type(w) is not int or not (8<=h<=1080 and 8<=w<=1920):
        raise ValueError('SLM 有效区域应为宽 8～1920、高 8～1080 像素。')
    pitch = getattr(config,key+'_pitch_um')
    distance = getattr(config,key+'_distance_mm')*1000
    wavelength = config.wavelength_nm/1000
    if any(not math.isfinite(v) or v<=0 for v in (pitch,distance,wavelength)):
        raise ValueError('请在全息参数中设置有效的波长、像素间距及该 SLM 到目标平面的距离。')
    if config.geometry_probe not in ('focus','disk','axicon'):
        raise ValueError('未知定位探测相位。')
    point=np.asarray(point,dtype=float)
    if point.shape!=(2,) or not np.isfinite(point).all():
        raise ValueError('定位目标需要两个有限坐标。')
    if config.geometry_probe=='axicon' and (not math.isfinite(config.geometry_axicon_period_um) or config.geometry_axicon_period_um<=0):
        raise ValueError('请在全息参数中设置有效的轴锥径向周期。')
    if config.geometry_probe=='disk':
        limit=(min(h,w)-5)*pitch/2
        if not math.isfinite(config.geometry_probe_radius_um) or not 0<config.geometry_probe_radius_um<limit:
            raise ValueError(f'圆形相位区半径需要大于零且小于 {limit:g} μm。')
    x,y = np.meshgrid((np.arange(w)-(w-1)/2)*pitch-point[0],
                      (np.arange(h)-(h-1)/2)*pitch-point[1])
    r = np.sqrt(distance*distance+x*x+y*y)
    if probe_frequency(config,point) >= LIMIT:
        raise ValueError('定位相位超过 SLM 像素的采样带宽；请增大传播距离或减小目标视野/有效口径。')
    # Stable evaluation of r-z avoids cancellation for small apertures.
    phase = -2*np.pi/wavelength*((x*x+y*y)/(r+distance))
    if config.geometry_probe != 'focus':
        # Keep the radial pattern fixed on the SLM. A ramp shifts its Fresnel
        # diffraction pattern; physically moving a disk would use another map.
        sx=x+point[0];sy=y+point[1];radius=np.hypot(sx,sy)
        if config.geometry_probe=='axicon':
            phase=-2*np.pi*radius/config.geometry_axicon_period_um
        else:
            edge=np.clip((config.geometry_probe_radius_um+2*pitch-radius)/(4*pitch),0,1)
            phase=np.pi*edge*edge*(3-2*edge)
        phase+=2*np.pi*(sx*point[0]+sy*point[1])/(wavelength*distance)
    gray = np.remainder(phase/(2*np.pi),1)
    if config.gray_inverted:
        gray = 1-gray
    return np.uint8(np.rint(gray*255))

