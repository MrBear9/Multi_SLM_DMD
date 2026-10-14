"""Shared camera selection, used by UI, acquisition and calibration."""


def selected_camera(settings):
    dvp = bool(settings.get('ccdEnabledCheckBox'))
    toup = bool(settings.get('toupcamEnabledCheckBox'))
    if dvp and toup:
        raise ValueError('当前共用一路相机预览，请只启用度申或 ToupTek 中的一种。')
    return 'toupcam' if toup else 'dvp2' if dvp else None


def camera_prefix(settings):
    return 'toupcam' if selected_camera(settings) == 'toupcam' else 'ccd'


def camera_value(settings, suffix, default=None):
    return settings.get(camera_prefix(settings) + suffix, default)


def camera_roi(settings):
    prefix = camera_prefix(settings)
    return ([settings[prefix+'Roi'+key+'SpinBox'] for key in ('X', 'Y', 'Width', 'Height')]
            if settings.get(prefix+'RoiCheckBox') else None)
