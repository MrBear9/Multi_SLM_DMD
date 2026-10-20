"""Target sizing, observation settings and geometry profiles for the child UI."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

from ..holography.config import HolographyConfig
from ..holography.geometry import observation_matrix, roi_info, signature, validate_profile
from ..holography.target import prepare_target
from ..holography.camera import rectify


CONTROLS = {
    'geometry_detector': ('geometry_detector', ('spot','ring')),
    'geometry_probe': ('geometry_probe', ('focus','disk','axicon')),
    'geometry_assistance': ('geometryAssistanceComboBox', ('automatic','manual','on_failure')),
    'initialization': ('initialization', ('random','gs')),
    'target_kind': ('target_kind', ('file','cross','frame','asymmetric')),
    'target_fit': ('targetFitModeComboBox', ('contain','cover','stretch')),
    'reference_source': ('referenceSourceComboBox', ('slm','dmd','custom')),
    'sizing_mode': ('targetSizingModeComboBox', ('reference','manual')),
    'reference_width': ('referenceWidthSpinBox', None),
    'reference_height': ('referenceHeightSpinBox', None),
    'reference_pitch_um': ('referencePitchSpinBox', None),
    'reference_physical_width_um': ('customWidthSpinBox', None),
    'reference_physical_height_um': ('customHeightSpinBox', None),
    'reconstruction_scale': ('reconstructionScaleSpinBox', None),
    'camera_optical_path': ('cameraOpticalPathComboBox', ('direct','relay','unsupported')),
    'camera_profile': ('cameraProfileComboBox', ('connected','toup','dvp')),
    'sensor_pitch_x_um': ('sensorPitchSpinBox', None),
    'binning_x': ('binningXSpinBox', None), 'binning_y': ('binningYSpinBox', None),
    'relay_magnification': ('relayMagnificationSpinBox', None),
    'magnification_source': ('magnificationSourceComboBox', ('manual','calibrated')),
    'geometry_mapping': ('geometryMappingComboBox', ('similarity','affine','perspective')),
    'geometry_pattern': ('geometryPatternComboBox', ('sequential','asymmetric')),
    'geometry_tolerance': ('geometryToleranceSpinBox', None),
}
STORED_FIELDS = {'sensor_pitch_y_um', 'geometry_search_roi'}
CUSTOM_SIZE = ('reference_physical_width_um', 'reference_physical_height_um')


class HolographySettings(QtCore.QObject):
    def __init__(self, ui):
        super().__init__(ui)
        self.ui = u = ui
        self.profile = {}
        self.pending = None
        self.raw_path = None
        self.loading = False
        self.last_geometry_signature = None
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_target)
        self.set_values(HolographyConfig())
        for key, (name, choices) in CONTROLS.items():
            control = getattr(u,name)
            signal = control.currentIndexChanged if choices else control.valueChanged
            signal.connect(lambda _, k=key: self.changed(k))
            if not choices:
                control.setMinimum(1 if key in ('reference_width','reference_height','binning_x','binning_y') else .001)
        for name in ('mode','slm1_width','slm1_height','slm1_pitch_um','slm2_width','slm2_height',
                     'slm2_pitch_um','target_width_um','target_height_um','wavelength_nm',
                     'slm1_distance_mm','slm2_distance_mm','inter_slm_mm','gray_inverted','output_width','output_height'):
            control = getattr(u,name)
            signal = (control.currentIndexChanged if isinstance(control,QtWidgets.QComboBox) else
                      control.toggled if isinstance(control,QtWidgets.QCheckBox) else control.valueChanged)
            signal.connect(lambda _, k=name: self.changed(k))
        for name in ('geometry_axicon_period_um','geometry_probe_radius_um'):
            getattr(u,name).valueChanged.connect(lambda _: self.changed())
        for name in ('camera_center_x','camera_center_y','camera_angle_deg','camera_pitch_um','camera_flip_x','camera_flip_y'):
            control = getattr(u,name)
            signal = control.toggled if isinstance(control,QtWidgets.QCheckBox) else control.valueChanged
            signal.connect(self.manual_changed)
        u.target.textChanged.connect(lambda _: self.preview_timer.start(150))
        u.geometryStartButton.clicked.connect(lambda: u.guarded(lambda: u.controller.start('geometry')))
        u.geometryLoadButton.clicked.connect(lambda: u.guarded(self.load_profile))
        u.geometrySaveButton.clicked.connect(lambda: u.guarded(self.save_profile))
        u.applyGeometryButton.clicked.connect(lambda: u.guarded(self.apply_profile))
        u.showTargetRoiCheckBox.toggled.connect(self.update_camera)
        u.rectifyTargetCheckBox.toggled.connect(self.select_camera_view)
        for name in ('targetDesignViewComboBox','cameraDesignViewComboBox'):
            getattr(u,name).setEnabled(True)
        u.targetDesignViewComboBox.currentIndexChanged.connect(u.refresh_previews)
        u.cameraDesignViewComboBox.currentIndexChanged.connect(self.update_camera)
        u.showTargetRoiCheckBox.setEnabled(True)
        u.showTargetRoiCheckBox.setChecked(True)
        u.targetDesignViewComboBox.setCurrentIndex(2)
        u.rectifyTargetCheckBox.setEnabled(True)
        u.geometryProfileComboBox.clear()
        u.geometryProfileComboBox.addItem('当前光路 · 未应用标定')
        u.geometryProfileComboBox.setEnabled(False)
        self.changed()

    def values(self):
        result = {'geometry_profile':self.profile, 'sensor_pitch_y_um':self.sensor_y_um,
                  'geometry_search_roi':self.search_roi}
        for key, (name, choices) in CONTROLS.items():
            control = getattr(self.ui,name)
            result[key] = choices[control.currentIndex()] if choices else control.value()
        result.update(zip(CUSTOM_SIZE,self.custom_size_um))
        return result

    def set_values(self, config):
        self.loading = True
        try:
            self.profile = dict(config.geometry_profile)
            self.search_roi = dict(config.geometry_search_roi)
            self.sensor_y_um = config.sensor_pitch_y_um
            # Preserve exact physical values in previously saved configurations.
            # Only an explicit pixel/pitch edit recomputes them from UI values.
            self.custom_size_um = [getattr(config,key) for key in CUSTOM_SIZE]
            for key, (name, choices) in CONTROLS.items():
                control, value = getattr(self.ui,name), getattr(config,key)
                if key in CUSTOM_SIZE:
                    value /= config.reference_pitch_um
                with QtCore.QSignalBlocker(control):
                    if choices:
                        control.setCurrentIndex(choices.index(value))
                    else:
                        control.setValue(value)
        finally:
            self.loading = False

    def changed(self, key=None):
        if self.loading or not hasattr(self.ui,'settings'):
            return
        u = self.ui
        self.loading = True
        try:
            camera_read = False
            if key in (*CUSTOM_SIZE,'reference_pitch_um'):
                self.custom_size_um = [getattr(u,CONTROLS[k][0]).value()*u.referencePitchSpinBox.value() for k in CUSTOM_SIZE]
            if key == 'sensor_pitch_x_um':
                self.sensor_y_um = u.sensorPitchSpinBox.value()
            if key == 'camera_profile':
                camera_read = self.read_camera()
            if key == 'magnification_source' and u.magnificationSourceComboBox.currentIndex()==0:
                self.profile = {}
                u.mapping_confirmed.setChecked(False)
            config = u.values().resolve_target_size()
            reference = config.reference_size()
            if config.reference_source == 'custom':
                pixel_note = f'自定义 {u.customWidthSpinBox.value():g} × {u.customHeightSpinBox.value():g} px × {config.reference_pitch_um:g} μm/px；'
            else:
                pixel_note = ''
            if config.sizing_mode == 'reference':
                for name in ('target_width_um','target_height_um'):
                    with QtCore.QSignalBlocker(getattr(u,name)):
                        getattr(u,name).setValue(getattr(config,name))
            prefix = (f'参考 {reference[0]:g} × {reference[1]:g} μm × {config.reconstruction_scale:g} → '
                      if config.sizing_mode=='reference' else f'参考 {reference[0]:g} × {reference[1]:g} μm；手动指定 ')
            u.targetSizeFormulaLabel.setText(pixel_note + prefix +
                f'目标 {config.target_width_um:g} × {config.target_height_um:g} μm '
                f'（{config.target_width_um/1000:g} × {config.target_height_um/1000:g} mm）')
            u.targetSamplingLabel.setText(
                f'目标计算：{config.output_width}×{config.output_height}；'
                f'目标采样：{config.target_width_um/config.output_width:.4g}×'
                f'{config.target_height_um/config.output_height:.4g} μm')
            mag = config.relay_magnification if config.camera_optical_path == 'relay' else 1.
            px = config.sensor_pitch_x_um*config.binning_x/max(mag,1e-9)
            py = config.sensor_pitch_y_um*config.binning_y/max(mag,1e-9)
            if (key in ('sensor_pitch_x_um','sensor_pitch_y_um','binning_x','binning_y',
                       'relay_magnification','camera_optical_path') or camera_read) and not self.profile:
                with QtCore.QSignalBlocker(u.camera_pitch_um):
                    u.camera_pitch_um.setValue(px)
                u.mapping_confirmed.setChecked(False)
            u.imagingFormulaLabel.setText(f'名义目标采样：{px:g} × {py:g} μm/px（像元 × Binning ÷ 成像倍率 {mag:g}）；应用后使用实测矩阵。')
            square = abs(config.sensor_pitch_x_um-config.sensor_pitch_y_um)<1e-6
            u.sensorPitchNoteLabel.setText('方形像元 · X / Y 共用' if square else
                f'读取值：X {config.sensor_pitch_x_um:g} / Y {config.sensor_pitch_y_um:g} μm（非方形）')
            current_signature = signature(config)
            if self.last_geometry_signature is not None and current_signature != self.last_geometry_signature:
                u.mapping_confirmed.setChecked(False)
            self.last_geometry_signature = current_signature
            self.update_bandwidth(config)
            u.update_grid_report()
            rect=self.search_roi.get('roi')
            u.geometrySearchLabel.setText('搜索：整个原始画面' if not rect else
                f"搜索 ROI：({rect['x']}, {rect['y']})，{rect['width']}×{rect['height']} px；需包含全部定位点")
        finally:
            self.loading = False
        self.update_controls()
        self.preview_timer.start(150)
        self.update_profile_status()
        self.update_camera()

    def update_bandwidth(self, config):
        from ..holography.geometry_limits import bandwidth_report
        u=self.ui
        try:
            report=bandwidth_report(config)
            status='通过' if report['fits'] else '超限'
            u.bandwidthLabel.setText(f'定位采样：{report["peak"]:.3f} 周期/像素（限值 0.48，{status}）；'
                f'保持当前比例的最大视野估算：{report["max_width_um"]:.1f}×{report["max_height_um"]:.1f} μm。')
            u.fitGeometryRangeButton.setEnabled(not u.controller.active and not report['fits'])
        except (ValueError, ZeroDivisionError):
            u.bandwidthLabel.setText('请选择单 SLM，并填写有效的波长、距离、口径与目标尺寸。')
            u.fitGeometryRangeButton.setEnabled(False)

    def fit_geometry_range(self):
        from ..holography.geometry_limits import bandwidth_report
        c=self.ui.values().resolve_target_size();r=bandwidth_report(c);u=self.ui
        if r['fits']:
            return
        if min(r['safe_width_um'],r['safe_height_um'])<max(u.target_width_um.minimum(),u.target_height_um.minimum()):
            raise ValueError('即使目标缩到中心，当前口径与距离仍无法满足采样限制；请减小有效口径或调整实际传播距离。')
        with QtCore.QSignalBlocker(u.targetSizingModeComboBox), QtCore.QSignalBlocker(u.target_width_um), QtCore.QSignalBlocker(u.target_height_um):
            u.targetSizingModeComboBox.setCurrentIndex(1)
            u.target_width_um.setValue(r['safe_width_um'])
            u.target_height_um.setValue(r['safe_height_um'])
        self.changed()
        u.log.appendPlainText('已按原比例缩小目标物理视野，并在带宽边界内保留 2% 尺寸余量；请重新进行几何标定。')

    def read_camera(self):
        u = self.ui
        selected = u.cameraProfileComboBox.currentIndex()
        parameters = getattr(u.controller.window,'camera_parameters',{})
        identity = parameters.get('camera',{})
        pitch = identity.get('pixel_size_um') if selected == 0 else [3.76,3.76] if selected == 1 else None
        if pitch and len(pitch) == 2 and min(pitch) > 0:
            with QtCore.QSignalBlocker(u.sensorPitchSpinBox):
                u.sensorPitchSpinBox.setValue(pitch[0])
            self.sensor_y_um = pitch[1]
            u.cameraProfileComboBox.setToolTip('SDK 像元已读取；请核对当前分辨率及 Binning。' if selected==0 else 'IUC26000KMA 规格预设：3.76 μm；连接相机时可选择 SDK 读取。')
            geometry = parameters.get('geometry',{}) if selected == 0 else {}
            if geometry:
                factor = geometry.get('hardware_binning',1)*geometry.get('software_binning',1)
                # Lower-resolution SDK modes may also change effective sampling;
                # do not silently derive it from rounded resolution ratios.
                for name in ('binningXSpinBox','binningYSpinBox'):
                    with QtCore.QSignalBlocker(getattr(u,name)):
                        getattr(u,name).setValue(factor)
                if geometry.get('resolution_index',0):
                    u.log.appendPlainText('相机使用非原生分辨率：请核对总采样倍率；几何标定以实测坐标为准。')
            return True
        else:
            u.cameraProfileComboBox.setToolTip('当前 SDK 未提供像元尺寸；保留手动值，请查相机规格。DVP 不自动套用 4 μm。')
            u.log.appendPlainText('未读取到该相机的像元尺寸，保留手动值，请核对规格。')
            return False

    def manual_changed(self, *_):
        if self.loading:
            return
        if self.profile:
            self.profile = {}
            with QtCore.QSignalBlocker(self.ui.magnificationSourceComboBox):
                self.ui.magnificationSourceComboBox.setCurrentIndex(0)
            self.ui.log.appendPlainText('手动修改映射：已退出实测矩阵，请重新确认或应用标定文件。')
        self.update_profile_status()
        self.update_camera()

    def update_controls(self):
        u = self.ui
        running = u.controller.active
        ring=u.geometry_detector.currentIndex()==1
        u.geometry_spot_max_fwhm.setEnabled(not ring)
        u.geometry_ring_radius.setEnabled(ring)
        u.geometry_probe_radius_um.setEnabled(u.geometry_probe.currentIndex()==1)
        u.geometry_axicon_period_um.setEnabled(u.geometry_probe.currentIndex()==2)
        reference = u.referenceSourceComboBox.currentIndex()
        automatic = u.targetSizingModeComboBox.currentIndex()==0
        for name in ('referenceWidthSpinBox','referenceHeightSpinBox'):
            getattr(u,name).setEnabled(reference==1)
        u.referencePitchSpinBox.setEnabled(reference in (1,2))
        for name in ('customWidthSpinBox','customHeightSpinBox'):
            getattr(u,name).setEnabled(reference==2)
        for name in ('target_width_um','target_height_um'):
            getattr(u,name).setEnabled(not automatic)
        u.reconstructionScaleSpinBox.setEnabled(automatic)
        u.relayMagnificationSpinBox.setEnabled(u.cameraOpticalPathComboBox.currentIndex()==1)
        u.geometryStartButton.setEnabled(not running and u.mode.currentIndex()!=2 and u.cameraOpticalPathComboBox.currentIndex()!=2)
        u.geometryStartButton.setToolTip('逐点显示定位相位，采集并验证；单 SLM 模式无需 DMD。双 SLM 请使用原光路标定入口。')
        u.geometryLoadButton.setEnabled(not running)
        u.geometrySaveButton.setEnabled(not running and bool(self.pending or self.profile))
        u.applyGeometryButton.setEnabled(not running and bool(self.pending and self.pending.get('validated')))
        u.group1.setEnabled(True)  # Capture parameters also govern the geometry button.
        u.cameraCenterButton.setEnabled(not running)
        u.manualMappingGroup.setTitle('手动映射（编辑后退出实测矩阵）')
        file_target=u.target_kind.currentIndex()==0
        for name in ('target','targetBrowse','targetFitModeComboBox'):
            getattr(u,name).setEnabled(file_target and u.operation.currentIndex()!=2)

    def current_hardware(self):
        w = self.ui.controller.window
        return self.ui.controller.hardware_signature() if getattr(w,'camera_ready',False) else None

    def show_profile(self, profile):
        if profile.get('schema') != 1 or profile.get('kind') != 'holography_geometry':
            raise ValueError('请选择本窗口保存的 geometry.json。')
        self.pending = profile
        self.update_profile_status()
        self.update_controls()
        self.update_camera()

    def update_profile_status(self):
        u = self.ui
        profile = self.pending or self.profile
        if not profile:
            u.geometryStatusDisplay.setText('未应用标定 · 可使用手动映射')
            u.geometryProfileComboBox.setItemText(0,'当前光路 · 未应用标定')
            return
        try:
            config = u.values().resolve_target_size()
            info = roi_info(profile['matrix'],config,profile['raw_shape'])
            u.roiPhysicalSizeDisplay.setText(f'{config.target_width_um:g} × {config.target_height_um:g} μm')
            u.roiCameraBoundsDisplay.setText(' / '.join(f'{x:.1f}' for x in info['bounds'])+' px（xmin/ymin/xmax/ymax）')
            u.calibratedSamplingDisplay.setText(' × '.join(f'{x:.4g}' for x in info['sampling_um'])+' μm/px（中心）')
            error = np.asarray(profile['validation_errors_px'])
            u.geometryErrorDisplay.setText(f'独立验证 RMS {np.sqrt(np.mean(error**2)):.3f} / 最大 {error.max():.3f} px')
            u.geometryCoverageDisplay.setText(f'{info["coverage"]*100:.2f}% · 旋转 {info["rotation_deg"]:.2f}° · 镜像：'+('是' if info['mirrored'] else '否'))
            validate_profile(profile,config,hardware=self.current_hardware())
            u.geometryStatusDisplay.setText('已应用实测映射' if self.profile == profile else '验证通过 · 待应用')
        except (ValueError,KeyError,TypeError) as exc:
            u.geometryStatusDisplay.setText(str(exc))
        u.geometryProfileComboBox.setItemText(0,'当前光路 · '+('已应用（修改参数后需核验）' if self.profile else '未应用标定'))

    def apply_profile(self):
        if self.ui.controller.active or self.pending is None:
            raise ValueError('请先完成或载入几何标定。')
        validate_profile(self.pending,self.ui.values().resolve_target_size(),hardware=self.current_hardware())
        self.profile = self.pending
        with QtCore.QSignalBlocker(self.ui.magnificationSourceComboBox):
            self.ui.magnificationSourceComboBox.setCurrentIndex(1)
        self.ui.mapping_confirmed.setChecked(True)
        self.ui.cameraDesignViewComboBox.setCurrentIndex(1)
        self.update_profile_status()
        self.update_camera()
        self.ui.log.appendPlainText('已应用目标坐标 → 相机原始像素映射；未更改传感器 ROI。')

    def load_profile(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self.ui,'载入几何标定','','JSON (*.json)')
        if path:
            self.show_profile(json.loads(Path(path).read_text(encoding='utf-8')))
            self.ui.log.appendPlainText('已载入几何结果，核验后点击应用：'+path)

    def save_profile(self):
        profile = self.pending or self.profile
        if profile:
            path,_ = QtWidgets.QFileDialog.getSaveFileName(self.ui,'保存几何标定','geometry.json','JSON (*.json)')
            if path:
                Path(path).write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')

    @staticmethod
    def pixmap(array):
        array = np.ascontiguousarray(np.uint8(np.clip(array,0,255)))
        h,w = array.shape
        return QtGui.QPixmap.fromImage(QtGui.QImage(array.data,w,h,array.strides[0],QtGui.QImage.Format_Grayscale8).copy())

    def update_target(self):
        u = self.ui
        try:
            config = u.values().resolve_target_size()
            scale=min(1.,1200/max(config.output_width,config.output_height))
            preview_config=replace(config,output_width=max(1,round(config.output_width*scale)),
                                   output_height=max(1,round(config.output_height*scale)))
            if config.target_kind=='file':
                with Image.open(config.target) as image:
                    u.inputImageSizeDisplay.setText(f'{image.width} × {image.height} px')
                    image.thumbnail((1200,1200))
                    u.pixmaps['input'] = self.pixmap(np.asarray(image.convert('L')))
            else:
                u.inputImageSizeDisplay.setText(f'内置目标：{config.output_width} × {config.output_height} px')
                u.pixmaps['input']=self.pixmap(prepare_target(preview_config)*255)
            u.pixmaps['adapted'] = self.pixmap(prepare_target(preview_config)*255)
        except (OSError,ValueError,ZeroDivisionError):
            u.inputImageSizeDisplay.setText('未读取到有效图像')
            u.pixmaps.pop('input',None)
            u.pixmaps.pop('adapted',None)
        u.refresh_previews()

    def select_camera_view(self, checked):
        self.ui.cameraDesignViewComboBox.setCurrentIndex(2 if checked else 1)

    def update_camera(self, *_):
        u = self.ui
        try:
            if self.raw_path is None:
                return
            raw = np.load(self.raw_path, mmap_mode='r',allow_pickle=False)
            config = u.values().resolve_target_size()
            profile = self.pending or self.profile
            if profile:
                # A failed profile may be visualised, but never used for rectification.
                matrix = np.asarray(profile['matrix'])
                info = roi_info(matrix,config,raw.shape)
            else:
                info = roi_info(observation_matrix(config),config,raw.shape)
            if u.cameraDesignViewComboBox.currentIndex()==2:
                candidate = replace(config,geometry_profile=profile) if profile else config
                if profile:
                    validate_profile(profile,candidate,hardware=self.current_hardware(),raw_shape=raw.shape)
                corrected = rectify(raw,candidate)
                u.pixmaps['rectified'] = self.pixmap(corrected/config.camera_white*255)
            signal = raw[...,1] if raw.ndim==3 else raw
            stride = max(1,int(np.ceil(max(signal.shape)/1200)))
            pixmap = self.pixmap(signal[::stride,::stride]/max(config.camera_white,1)*255).toImage().convertToFormat(QtGui.QImage.Format_RGB32)
            if u.showTargetRoiCheckBox.isChecked():
                painter = QtGui.QPainter(pixmap)
                painter.setRenderHint(QtGui.QPainter.Antialiasing)
                painter.setPen(QtGui.QPen(QtGui.QColor('#ff3030'),2))
                painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x/stride,y/stride) for x,y in info['quad']]))
                painter.end()
            u.pixmaps['raw'] = QtGui.QPixmap.fromImage(pixmap)
            u.cameraPreview.setToolTip('最近一次任务采集的原始帧；红框是目标物理矩形的投影。')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            u.pixmaps.pop('rectified',None)
            u.cameraPreview.setToolTip(str(exc))
        u.refresh_previews()
