"""Applied holography geometry shown over the unmodified CCD detail frame."""
import copy
from dataclasses import replace
import json

from PyQt5 import QtCore

from ..holography.geometry import geometry_hardware, roi_info, validate_profile


def geometric_identity(hardware):
    value = geometry_hardware(hardware)
    if isinstance(value, dict):
        # Illumination and CCD *display* orientation do not move raw pixels.
        value.pop('dmd_sha256', None)
        value.pop('dmd_input', None)
        settings = value.get('settings', {})
        for name in ('ccdRotationSpinBox', 'ccdHorizontalFlipCheckBox', 'ccdVerticalFlipCheckBox'):
            settings.pop(name, None)
    return value


class CcdGeometryOverlay(QtCore.QObject):
    def __init__(self, detail):
        super().__init__(detail)
        self.detail = detail
        self.window = w = detail.window
        self.config = None
        self.cache_key = None
        self.frame_key = None
        self.target = self.fov = None
        w.ccdDetailCalibrationTargetCheckBox.toggled.connect(self.render)
        w.ccdDetailCalibrationFovCheckBox.toggled.connect(self.render)
        w.ccdDetailCalibrationTargetButton.clicked.connect(lambda: detail.run_action(lambda: self.focus(False)))
        w.ccdDetailCalibrationFovButton.clicked.connect(lambda: detail.run_action(lambda: self.focus(True)))
        # Validate hardware changes even while the preview is frozen; no raw
        # images are copied or DMD buffers hashed by this inexpensive check.
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def sync(self, settings):
        if settings.loading:
            return
        self.config = copy.deepcopy(settings.ui.values().resolve_target_size()) if settings.profile else None
        self.cache_key = None
        self.refresh()

    def frame_changed(self):
        if self.detail.displayed is None:
            return
        raw, metadata, _ = self.detail.displayed
        key = json.dumps([list(raw.shape[:2]),metadata.get('camera'),metadata.get('camera_geometry')],sort_keys=True)
        if key != self.frame_key:
            self.frame_key = key
            self.refresh()

    def refresh(self):
        d, w = self.detail, self.window
        if self.config is None or d.displayed is None:
            self.cache_key = None
            self.frame_key = None
            self.target = self.fov = None
            self.status('未应用全息标定' if self.config is None else '已应用标定 · 等待相机画面')
            self.render()
            return
        c = self.config
        raw, metadata, _ = d.displayed
        hardware = geometric_identity(w.holography.hardware_signature(include_illumination=False))
        frame_identity = {k:metadata.get(k) for k in ('camera','camera_geometry')}
        key = json.dumps([hardware,list(raw.shape[:2]),frame_identity],sort_keys=True)
        if key == self.cache_key:
            return
        self.cache_key = key
        self.target = self.fov = None
        try:
            profile = dict(c.geometry_profile)
            profile['hardware'] = geometric_identity(profile.get('hardware'))
            info = validate_profile(profile, c, hardware=hardware, raw_shape=raw.shape)
            camera = hardware.get('camera', {})
            identity = metadata.get('camera') or {}
            expected = camera.get('camera', {})
            for name in ('serial','model'):
                if identity.get(name) and expected.get(name) and identity[name] != expected[name]:
                    raise ValueError('冻结帧来自另一台相机。')
            geometry = metadata.get('camera_geometry')
            if geometry and json.dumps(geometry,sort_keys=True) != json.dumps(camera.get('geometry'),sort_keys=True):
                raise ValueError('当前帧与相机采集几何不一致。')
            self.target = info['quad']
            note = '红实线：目标 ROI；橙虚线：外围视野'
            if profile.get('target_plane_assumption'):
                note += '\n棋盘格映射：沿用目标平面 1:1 假设'
            try:
                outer = roi_info(profile['matrix'],replace(c,
                    target_width_um=c.target_width_um*c.observation_fov_scale,
                    target_height_um=c.target_height_um*c.observation_fov_scale),raw.shape)
                self.fov = outer['quad']
                if outer['coverage'] < .99999:
                    note += '\n外围视野部分超出画面，仅显示可见部分'
            except ValueError as exc:
                note += '\n外围视野不可显示：'+str(exc)
            self.status(note)
        except (ValueError, KeyError, TypeError) as exc:
            self.status('标定框已隐藏，请核验并重新应用', str(exc))
        self.render()

    def status(self, text, reason=None):
        label = self.window.ccdDetailCalibrationStatusLabel
        label.setText(text)
        label.setToolTip(reason or text)

    def render(self, *_):
        w = self.window
        c = self.config
        polygons = []
        for points,check,button,color,style,label in (
            (self.target,w.ccdDetailCalibrationTargetCheckBox,w.ccdDetailCalibrationTargetButton,
             '#ff3030',QtCore.Qt.SolidLine,'标定目标 ROI'),
            (self.fov,w.ccdDetailCalibrationFovCheckBox,w.ccdDetailCalibrationFovButton,
             '#ffb020',QtCore.Qt.DashLine,f'外围视野 ×{c.observation_fov_scale:g}' if c else '外围视野')):
            check.setEnabled(points is not None)
            button.setEnabled(points is not None)
            if points is not None and check.isChecked():
                polygons.append((points,color,style,label))
        if polygons != self.detail.view.calibration_polygons:
            self.detail.view.set_calibration_polygons(polygons)

    def focus(self, outer):
        self.refresh()
        polygon = self.fov if outer else self.target
        if polygon is None:
            raise ValueError('没有与当前帧匹配的标定区域，请核验并应用全息标定。')
        check = self.window.ccdDetailCalibrationFovCheckBox if outer else self.window.ccdDetailCalibrationTargetCheckBox
        check.setChecked(True)
        self.detail.view.focus_polygon(polygon)
