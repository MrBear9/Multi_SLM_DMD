"""Hardware sampling check for focal probes; no torch or full phase allocation."""
import math
import numpy as np

LIMIT = .48


def probe_points(config):
    train=np.array([(x,y) for y in (-.4,0,.4) for x in (-.4,0,.4)])
    if config.geometry_pattern=='asymmetric':
        train[1]=[-.08,-.32];train[5]=[.32,.08]
    held=np.array([[-.31,-.19],[.23,-.29],[.29,.27],[-.27,.31],[.06,-.07]])
    return np.vstack([train,held])*[config.target_width_um,config.target_height_um],len(train)


def probe_frequency(config, point):
    key=config.active_slms[0]
    h,w=config.shape(key);p=getattr(config,key+'_pitch_um')
    z=getattr(config,key+'_distance_mm')*1000;wave=config.wavelength_nm/1000
    if config.geometry_probe != 'focus':
        # Fixed radial pattern plus a phase ramp. Under Fresnel propagation,
        # the ramp translates the observed field by z * wavelength * frequency.
        carrier=max(abs(float(point[0])),abs(float(point[1])))*p/(z*wave)
        radial=p/config.geometry_axicon_period_um if config.geometry_probe=='axicon' else .1875
        return carrier+radial
    def bounds(count, centre):
        edge=(count-1)*p/2
        nearest=min(count-1,max(0,round(centre/p+(count-1)/2)))
        return max(abs(-edge-centre),abs(edge-centre)),abs(nearest*p-edge-centre)
    xmax,xmin=bounds(w,point[0]);ymax,ymin=bounds(h,point[1])
    return max(xmax/math.sqrt(z*z+xmax*xmax+ymin*ymin),
               ymax/math.sqrt(z*z+ymax*ymax+xmin*xmin))*p/wave


def bandwidth_report(config):
    if config.mode=='dual':
        raise ValueError('此范围检查适用于单 SLM 几何标定。')
    points,_=probe_points(config)
    def peak(scale):
        return max(probe_frequency(config,point*scale) for point in points)
    current=peak(1.)
    if peak(0.)>=LIMIT:
        scale=0.
    else:
        lo,hi=0.,1.
        while peak(hi)<LIMIT and hi<1e6:
            lo,hi=hi,hi*2
        for _ in range(55):
            mid=(lo+hi)/2
            if peak(mid)<LIMIT:lo=mid
            else:hi=mid
        scale=lo
    return {'peak':current,'limit':LIMIT,'fits':current<LIMIT,
            'max_width_um':config.target_width_um*scale,'max_height_um':config.target_height_um*scale,
            'safe_width_um':config.target_width_um*scale*.98,'safe_height_um':config.target_height_um*scale*.98}


def validate_probe_bandwidth(config):
    report=bandwidth_report(config)
    if not report['fits']:
        raise ValueError(f'定位相位超过 SLM 像素的采样带宽：{report["peak"]:.3f} 周期/像素 ≥ {LIMIT}。'
                         f'当前比例下估算最大视野 {report["max_width_um"]:.1f}×{report["max_height_um"]:.1f} μm；'
                         '请使用“调整到可标定范围”，或核对实际传播距离与有效口径。')
    return report
