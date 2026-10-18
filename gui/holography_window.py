"""Designer-editable child UI; intentionally has no training-runtime imports."""
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys

from PyQt5 import QtCore, QtGui, QtWidgets, uic

from ..holography.config import HolographyConfig
from ..utils.paths import TOOLS_ROOT
from .holography_settings import HolographySettings, CONTROLS, STORED_FIELDS


class HolographyWindow(QtWidgets.QDialog):
    def __init__(self, controller):
        super().__init__(controller.window)
        uic.loadUi(str(TOOLS_ROOT/'ui'/'HolographyWindow.ui'), self)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, True)
        self.help_dialog = None
        self.help_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('F1'), self)
        self.help_shortcut.activated.connect(self.show_help)
        self.controller = controller
        self.preferences = QtCore.QSettings('Qing', 'Holography')
        self.pythonPath.setText(self.preferences.value('python', '' if getattr(sys, 'frozen', False) else sys.executable))
        self.workerPath.setText(self.preferences.value('worker', str(TOOLS_ROOT/'holography'/'worker.py')))
        self.outputPath.setText(self.preferences.value('output', str(Path.cwd()/'output'/'holography')))
        self.log.setMaximumBlockCount(500)
        self.pixmaps = {}
        self.seed = 17
        for name in ('pythonPath', 'workerPath', 'target', 'model_path', 'outputPath'):
            getattr(self, name+'Browse').clicked.connect(lambda _, key=name: self.browse(key))
        self.startButton.clicked.connect(lambda: self.guarded(controller.start))
        self.stopButton.clicked.connect(controller.cancel)
        self.applyButton.clicked.connect(lambda: self.guarded(controller.apply_result))
        self.loadResultButton.clicked.connect(lambda: self.guarded(self.load_result))
        self.closeButton.clicked.connect(self.close)
        self.loadConfigButton.clicked.connect(lambda: self.guarded(self.load_config))
        self.saveConfigButton.clicked.connect(lambda: self.guarded(self.save_config))
        self.cameraCenterButton.clicked.connect(lambda: self.guarded(self.use_camera_center))
        self.matchControlsButton.clicked.connect(self.match_control_sizes)
        self.matchTargetButton.clicked.connect(lambda: self.guarded(self.match_target_ratio))
        self.mode.currentIndexChanged.connect(self.update_controls)
        self.operation.currentIndexChanged.connect(self.update_controls)
        for name in ('camera_pitch_um','camera_center_x','camera_center_y','camera_angle_deg',
                     'camera_dark','camera_white'):
            getattr(self,name).valueChanged.connect(lambda _: self.mapping_confirmed.setChecked(False))
        for name in ('camera_flip_x','camera_flip_y'):
            getattr(self,name).toggled.connect(lambda _: self.mapping_confirmed.setChecked(False))
        self.mapping_confirmed.setToolTip('原始相机坐标：X 向右、Y 向下；先水平/垂直翻转，再按图像坐标旋转。有效采样应包含 Binning 和目标平面到传感器的倍率。此处不自动推断光路方向。')
        self.camera_white.setToolTip('输入相机实际饱和码值：8 位常为 255；12 位未左移常为 4095；16 位容器不一定代表 65535。')
        for name in ('grid_width', 'grid_height'):
            getattr(self, name).setToolTip('宽、高分别决定计算点数，乘以 ASM 采样间距得到各方向视野。开始时不足的方向自动增大，保留已足够的手动值；FFT 两个方向均采用 2 倍零填充。')
        self.cameraCenterButton.setToolTip('只填入原始帧几何中心；不代表已完成几何标定，不自动勾选确认。')
        self.source_note = ('入射照明为首片 SLM 有效区内的均匀参考；实物请固定 DMD/光源。'
                            '连接但未参与优化的另一片 SLM 将设为全黑，任务结束恢复。')
        self.descriptionLabel.setText(self.source_note)
        self.descriptionLabel.setWordWrap(True)
        self.settings = HolographySettings(self)
        self.settings.changed()
        self.update_controls()

    def event(self, event):
        if event.type() == QtCore.QEvent.EnterWhatsThisMode:
            QtWidgets.QWhatsThis.leaveWhatsThisMode()
            QtCore.QTimer.singleShot(0, self.show_help)
            return True
        return super().event(event)

    def show_help(self):
        if self.help_dialog is None:
            from .holography_help import HolographyHelp
            self.help_dialog = HolographyHelp(self)
        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()

    def guarded(self, action):
        try:
            action()
        except Exception as exc:
            self.log.appendPlainText(str(exc))
            QtWidgets.QMessageBox.warning(self, '全息操作未完成', str(exc))

    def values(self):
        values = {}
        for field in fields(HolographyConfig):
            if field.name in ('seed','geometry_profile') or field.name in CONTROLS or field.name in STORED_FIELDS:
                continue
            control = getattr(self, field.name)
            if isinstance(control, QtWidgets.QCheckBox):
                value = control.isChecked()
            elif isinstance(control, QtWidgets.QLineEdit):
                value = control.text().strip()
            elif isinstance(control, QtWidgets.QComboBox):
                value = (('slm1','slm2','dual') if field.name == 'mode' else ('generate','feedback','calibrate'))[control.currentIndex()]
            else:
                value = control.value()
            values[field.name] = value
        values.update(self.settings.values())
        return HolographyConfig(**values, seed=self.seed)

    def set_values(self, config):
        self.settings.loading = True
        for key, value in asdict(config).items():
            if key in CONTROLS or key in STORED_FIELDS or key in ('geometry_profile','seed'):
                continue
            control = getattr(self, key, None)
            if isinstance(control, QtWidgets.QCheckBox):
                control.setChecked(value)
            elif isinstance(control, QtWidgets.QLineEdit):
                control.setText(value)
            elif isinstance(control, QtWidgets.QComboBox):
                control.setCurrentIndex((('slm1','slm2','dual') if key == 'mode' else ('generate','feedback','calibrate')).index(value))
            elif control is not None:
                control.setValue(value)
        self.seed = config.seed
        self.settings.set_values(config)
        self.settings.pending = config.geometry_profile or None
        self.settings.changed()
        # Change signals can clear confirmation while restoring other controls.
        self.mapping_confirmed.setChecked(config.mapping_confirmed)
        self.update_controls()

    def match_control_sizes(self):
        for key in self.values().active_slms:
            for axis in ('width','height'):
                getattr(self,key+'_controls_'+axis).setValue(getattr(self,key+'_'+axis).value())

    def match_target_ratio(self):
        config = self.values().resolve_target_size()
        height = int(config.output_width * config.target_height_um / config.target_width_um + .5)
        low, high = self.output_height.minimum(), self.output_height.maximum()
        if not low <= height <= high:
            raise ValueError(f'按目标比例计算的高度为 {height}，超出 {low}～{high}；请调整目标计算宽度后再匹配。')
        self.output_height.setValue(height)

    def browse(self, name):
        field = getattr(self, name)
        if name == 'outputPath':
            path = QtWidgets.QFileDialog.getExistingDirectory(self, '结果目录', field.text())
        else:
            filters = {'target':'Images (*.png *.bmp *.tif *.tiff *.jpg *.jpeg)',
                       'model_path':'Model (model.json)', 'workerPath':'Python (*.py)',
                       'pythonPath':'Python (python.exe python);;All files (*)'}
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择文件', field.text(), filters[name])
        if path:
            field.setText(path)

    def use_camera_center(self):
        packet = self.controller.window.camera_preview_packet
        if packet is None:
            raise ValueError('请先连接相机并等待一帧。')
        # Packet metadata keeps raw sensor dimensions even when preview rotated.
        shape = packet[1].get('raw_shape', packet[0].shape)
        self.camera_center_x.setValue((shape[1]-1)/2)
        self.camera_center_y.setValue((shape[0]-1)/2)
        self.mapping_confirmed.setChecked(False)
        self.log.appendPlainText('已填相机中心；请核对映射方向、有效采样和实际目标位置。')

    def load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '载入全息参数', '', 'JSON (*.json)')
        if path:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            self.set_values(HolographyConfig.from_dict(data.get('config', data)))
            self.mapping_confirmed.setChecked(False)

    def save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '保存全息参数', 'holography.json', 'JSON (*.json)')
        if path:
            Path(path).write_text(json.dumps(asdict(self.values()), ensure_ascii=False, indent=2), encoding='utf-8')

    def load_result(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '载入生成的相位结果', '', 'Result (result.json)')
        if path:
            self.controller.result_path = Path(path)
            self.controller.validate_result()
            self.log.appendPlainText('已载入：'+path)
            self.update_controls()

    def update_controls(self, *_):
        running = self.controller.active
        self.parameterContents.setEnabled(not running)
        for name in ('mode','operation','loadConfigButton','saveConfigButton','loadResultButton'):
            getattr(self, name).setEnabled(not running)
        self.startButton.setEnabled(not running)
        self.stopButton.setEnabled(running)
        self.applyButton.setEnabled(not running and self.controller.result_path is not None)
        if running:
            return
        active = ('slm1','slm2') if self.mode.currentIndex()==2 else (('slm1','slm2')[self.mode.currentIndex()],)
        for key in ('slm1','slm2'):
            for suffix in ('width','height','pitch_um'):
                getattr(self,key+'_'+suffix).setEnabled(key in active)
            for axis in ('width','height'):
                getattr(self,key+'_controls_'+axis).setEnabled(key in active and self.operation.currentIndex()!=2)
        self.matchControlsButton.setEnabled(self.operation.currentIndex()!=2)
        self.slm1_distance_mm.setEnabled(self.mode.currentIndex()==0)
        self.slm2_distance_mm.setEnabled(self.mode.currentIndex()!=0)
        self.inter_slm_mm.setEnabled(self.mode.currentIndex()==2)
        live = self.operation.currentIndex()!=0
        self.group1.setEnabled(live)
        self.mapping_confirmed.setEnabled(live)
        self.cameraCenterButton.setEnabled(live)
        for name in ('target','targetBrowse','steps','feedback_every'):
            getattr(self,name).setEnabled(self.operation.currentIndex()!=2)
        self.samples.setEnabled(self.operation.currentIndex()==2)
        self.epochs.setEnabled(self.operation.currentIndex()==2)
        if hasattr(self,'settings'):
            self.settings.update_controls()

    def preview(self, kind, path):
        if kind == 'camera_raw':
            self.settings.raw_path = path
            self.settings.update_camera()
            return
        pixmap = QtGui.QPixmap(str(path))
        if not pixmap.isNull():
            self.pixmaps[kind] = pixmap
            self.refresh_previews()

    def refresh_previews(self, *_):
        target_key = ('target','input','adapted')[self.targetDesignViewComboBox.currentIndex()]
        if target_key=='target' and 'target' not in self.pixmaps:
            target_key='adapted'
        camera_key = ('camera','raw','rectified')[self.cameraDesignViewComboBox.currentIndex()]
        if camera_key=='camera' and 'camera' not in self.pixmaps:
            camera_key='raw'
        for name,key in (('target',target_key),('simulation','simulation'),('camera',camera_key)):
            label = getattr(self,name+'Preview')
            pixmap = self.pixmaps.get(key)
            if pixmap is None:
                label.setText('等待结果' if key!='rectified' else '暂无有效校正结果（查看悬停提示）')
                continue
            size=label.size()
            aspect=QtCore.Qt.KeepAspectRatio
            if key not in ('input','raw'):
                ratio=self.target_width_um.value()/max(self.target_height_um.value(),.001)
                width=min(size.width(),int(size.height()*ratio))
                size=QtCore.QSize(max(1,width),max(1,int(width/ratio)))
                aspect=QtCore.Qt.IgnoreAspectRatio
            label.setPixmap(pixmap.scaled(size,aspect,QtCore.Qt.SmoothTransformation))

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self,'settings') and not self.controller.active:
            parameters=getattr(self.controller.window,'camera_parameters',{})
            if self.cameraProfileComboBox.currentIndex()==0 and not self.settings.profile and parameters.get('camera',{}).get('pixel_size_um'):
                self.settings.changed('camera_profile')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self,'pixmaps'):
            self.refresh_previews()

    def closeEvent(self, event):
        if self.controller.active:
            self.controller.cancel()
            self.log.appendPlainText('正在停止任务并恢复显示，请稍后关闭。')
            event.ignore()
        else:
            event.accept()
