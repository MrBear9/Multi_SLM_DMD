"""ToupTek selection and capability display; acquisition lives in the worker."""
from functools import partial

from PyQt5 import QtCore, QtWidgets

from ..devices.toupcam_camera import sdk
from ..utils.camera_settings import camera_value
from .configuration import input_path, read_values


class ToupControls(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = w = window
        self.devices = []
        self.geometry = None
        self.acquisition_fps = None
        w.toupcamDeviceComboBox.setItemData(0, '')
        w.toupcamResolutionComboBox.setItemData(0, '')
        w.ccdEnabledCheckBox.toggled.connect(partial(self.select, 'ccd'))
        w.toupcamEnabledCheckBox.toggled.connect(partial(self.select, 'toupcam'))
        w.toupcamDeviceComboBox.currentIndexChanged.connect(self.device_changed)
        w.toupcamEnumerateButton.clicked.connect(lambda: self.enumerate())
        w.toupcamDllPathEditBrowseButton.clicked.connect(lambda: w.browse('toupcamDllPathEdit'))
        w.toupcamDllPathEdit.textEdited.connect(self.invalidate_devices)
        for control in (w.ccdPreviewIntervalSpinBox, w.toupcamPreviewIntervalSpinBox):
            control.valueChanged.connect(self.update_interval)

    def select(self, prefix, checked):
        w = self.window
        if checked:
            other = w.ccdEnabledCheckBox if prefix == 'toupcam' else w.toupcamEnabledCheckBox
            with QtCore.QSignalBlocker(other):
                other.setChecked(False)
        self.update_interval()
        w.update_actions()

    def update_interval(self, *_):
        w = self.window
        w.poll_timer.setInterval(camera_value(read_values(w), 'PreviewIntervalSpinBox', 33))

    def invalidate_devices(self, *_):
        self.devices = []
        w = self.window
        with QtCore.QSignalBlocker(w.toupcamDeviceComboBox):
            w.toupcamDeviceComboBox.clear()
            w.toupcamDeviceComboBox.addItem('未枚举 · 单台相机可自动选择', '')
        self.device_changed()

    def enumerate(self, lines=None):
        w = self.window
        if w.connected or w.camera is not None or w.device_task is not None:
            return
        if w.demo:
            self.show_devices([], ['演示模式：使用合成帧，不枚举真实 ToupTek SDK。'])
            return
        path = input_path(w.toupcamDllPathEdit.text())
        w.statusBar().showMessage('正在枚举 ToupTek 相机…')
        w.start_device_task(lambda cancel: sdk.enumerate_devices(path),
            lambda devices: self.show_devices(devices, lines),
            lambda error: QtWidgets.QMessageBox.warning(w, 'ToupTek 枚举失败', error))

    def show_devices(self, devices, lines=None):
        w = self.window
        self.devices = devices
        selected = w.toupcamDeviceComboBox.currentData()
        with QtCore.QSignalBlocker(w.toupcamDeviceComboBox):
            w.toupcamDeviceComboBox.clear()
            if not devices:
                w.toupcamDeviceComboBox.addItem('未发现相机 · 检查 USB / 电源 / 驱动', '')
            for d in devices:
                w.toupcamDeviceComboBox.addItem(f'{d["friendly_name"] or d["model"]} · {d["id"]}', d['id'])
            index = w.toupcamDeviceComboBox.findData(selected)
            if index >= 0:
                w.toupcamDeviceComboBox.setCurrentIndex(index)
        self.device_changed()
        messages = list(lines or []) + [f'ToupTek：{len(devices)} 个设备']
        for d in devices:
            messages.append(f'{d["friendly_name"]} / 型号：{d["model"]} / ID：{d["id"]}')
        QtWidgets.QMessageBox.information(w, '设备枚举', '\n'.join(messages))

    def device_changed(self, *_):
        w = self.window
        d = next((d for d in self.devices if d['id'] == w.toupcamDeviceComboBox.currentData()), None)
        selected = w.toupcamResolutionComboBox.currentData()
        with QtCore.QSignalBlocker(w.toupcamResolutionComboBox):
            w.toupcamResolutionComboBox.clear()
            w.toupcamResolutionComboBox.addItem('最大分辨率（连接时读取）', '')
            if d:
                for width, height in d['resolutions']:
                    w.toupcamResolutionComboBox.addItem(f'{width} × {height}', f'{width}x{height}')
                index = w.toupcamResolutionComboBox.findData(selected)
                if index >= 0:
                    w.toupcamResolutionComboBox.setCurrentIndex(index)
        if d:
            px, py = d['pixel_size_um']
            mode = '单色' if d['flags'] & sdk.MONO else '彩色（当前不支持）'
            w.toupcamIdentityLabel.setText(f'型号：{d["model"]} · {mode}\n像元 {px:g} × {py:g} μm · 序列号连接后读取')
        else:
            w.toupcamIdentityLabel.setText('型号 / 序列号 / 像元：待连接读取')

    def ready(self, parameters):
        w = self.window
        identity = parameters.get('camera', {})
        self.geometry = parameters.get('geometry')
        if not self.geometry:
            return
        px, py = identity['pixel_size_um']
        w.toupcamIdentityLabel.setText(f'型号：{identity["model"]}\n序列号：{identity["serial"]} · 像元 {px:g} × {py:g} μm')
        self.devices = [identity]
        with QtCore.QSignalBlocker(w.toupcamDeviceComboBox):
            w.toupcamDeviceComboBox.clear()
            w.toupcamDeviceComboBox.addItem(identity['friendly_name'] or identity['model'], identity['id'])
        self.device_changed()
        index = self.geometry['resolution_index']
        if index < len(identity['resolutions']):
            value = '%dx%d' % tuple(identity['resolutions'][index])
            w.toupcamResolutionComboBox.setCurrentIndex(w.toupcamResolutionComboBox.findData(value))
        w.toupcamOutputInfoLabel.setText(f'实际采集 {self.geometry["raw_size"][0]} × {self.geometry["raw_size"][1]} px\n有效位深 {self.geometry["effective_bits"]} bit · 等待第一帧')

    def update_fps(self):
        if self.geometry:
            width, height = self.geometry['raw_size']
            fps = '—' if self.acquisition_fps is None else f'{self.acquisition_fps:.1f}'
            self.window.toupcamOutputInfoLabel.setText(
                f'{width} × {height} px · 有效位深 {self.geometry["effective_bits"]} bit\n'
                f'采集 {fps} FPS · {self.window.ccdActiveAreaLabel.text()}')

    def disconnected(self):
        self.geometry = self.acquisition_fps = None
        self.window.toupcamOutputInfoLabel.setText('未连接 · 实际输出尺寸 / 帧率待读取')
