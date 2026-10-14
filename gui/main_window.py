"""MainWindow: configuration, previews, device lifecycle and sequence playback."""
from __future__ import annotations

from functools import partial
import ipaddress
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from ..devices.zkwx_slm import controller as image_files
from ..devices.zkwx_slm import sdk as zkwx
from ..devices.magicholo_slm import sdk as magic
from ..devices.magicholo_slm import controller as magic_images
from ..devices.dvp2_camera import sdk as dvp
from ..devices.dvp2_camera.controller import camera_identity
from ..utils.paths import TOOLS_ROOT
from ..utils.image_transforms import ImageTransform
from ..utils.dmd_images import DmdImageOptions, prepare_image
from .configuration import (read_values, apply_values, read_configuration,
                            write_configuration, input_path, pil_image_qimage,
                            camera_qimage, set_preview)
from .dmd_backends import DMD_BACKENDS, selected_dmd, create_display
from .device_task import DeviceTask
from .camera_worker import CameraWorker, CaptureRequest
from .calibration_controller import CalibrationController
from .ccd_detail import CcdDetailController
from .toupcam_controls import ToupControls
from .live_inputs import LiveInputs
from ..utils.camera_settings import selected_camera, camera_value

DISPLAY_KEYS = ('dmd', 'slm1', 'slm2')
TRANSFORM_SUFFIXES = ('RotationSpinBox', 'HorizontalFlipCheckBox', 'VerticalFlipCheckBox')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, demo=False, config_path=None, display_factory=create_display,
                 camera_factory=CameraWorker):
        super().__init__()
        uic.loadUi(str(TOOLS_ROOT / 'ui' / 'MainWindow.ui'), self)
        self.ccd_default_ranges = [(w.minimum(), w.maximum()) for w in
                                  (self.ccdExposureSpinBox, self.ccdGainSpinBox)]
        self.ccd_default_title = self.ccdTitleLabel.text()
        self.ccd_default_tooltip = self.ccdTitleLabel.toolTip()
        self.demo = demo
        self.display_factory, self.camera_factory = display_factory, camera_factory
        self.devices = {}
        self.device_task = None
        self.pending_devices = {}
        self.task_success = self.task_failure = None
        self.paths = {key: [] for key in DISPLAY_KEYS}
        self.indices = dict.fromkeys(DISPLAY_KEYS, 0)
        self.images = {}
        self.source_images = {}
        self.camera = None
        self.camera_ready = self.connected = self.disconnecting = self.closing = False
        self.recording = self.recording_pending = False
        self.received_frames = self.saved_frames = 0
        self.preview_fps_started_at = None
        self.preview_fps_frames = 0
        self.settings = None
        self.slm_capture_request = None
        self.camera_parameters = {}
        self.camera_preview_packet = None
        self.calibration = CalibrationController(self)
        self.transforms = {key: ImageTransform() for key in (*DISPLAY_KEYS, 'ccd')}
        self.inputs = LiveInputs(self)
        self.demo_directory = None
        self.timers = {}
        for key in DISPLAY_KEYS:
            timer = QtCore.QTimer(self)
            timer.setTimerType(QtCore.Qt.PreciseTimer)
            timer.timeout.connect(partial(self.guard, partial(self.advance, key)))
            self.timers[key] = timer
            getattr(self, key + 'PlayButton').clicked.connect(partial(self.guard, partial(self.play, key)))
            getattr(self, key + 'PauseButton').clicked.connect(partial(self.guard, partial(self.pause, key)))
            getattr(self, key + 'NextButton').clicked.connect(partial(self.guard, partial(self.step, key, 1)))
            getattr(self, key + 'PreviousButton').clicked.connect(partial(self.guard, partial(self.step, key, -1)))
            getattr(self, key + 'FrameSpinBox').valueChanged.connect(partial(self.select_frame, key))
            getattr(self, key + 'InputPathEdit').editingFinished.connect(partial(self.reload_safely, key))
        self.f4321InputPathEdit.editingFinished.connect(partial(self.reload_safely, 'dmd'))
        for backend in DMD_BACKENDS:
            getattr(self, backend.enabled_field).toggled.connect(partial(self.dmd_backend_changed, backend.name))
        for name in ('dmdPixelModeComboBox', 'dmdInputRangeComboBox'):
            getattr(self, name).currentIndexChanged.connect(self.dmd_conversion_changed)
        self.dmdDitherCheckBox.toggled.connect(self.dmd_conversion_changed)
        for name in ('dmdThresholdSpinBox', 'dmdBlackLevelSpinBox', 'dmdWhiteLevelSpinBox'):
            getattr(self, name).editingFinished.connect(self.dmd_conversion_changed)
        self.dmdUploadButton.clicked.connect(partial(self.guard, self.reupload_dmd))
        for key in (*DISPLAY_KEYS, 'ccd'):
            getattr(self, key + 'ZoomComboBox').currentIndexChanged.connect(partial(self.refresh_preview, key))
            if key != 'ccd':
                getattr(self, key + 'EnabledCheckBox').toggled.connect(self.update_actions)
            getattr(self, key + 'RotationSpinBox').valueChanged.connect(partial(self.transform_changed, key))
            for suffix in ('HorizontalFlipCheckBox', 'VerticalFlipCheckBox'):
                getattr(self, key + suffix).toggled.connect(partial(self.transform_changed, key))
        for key in DISPLAY_KEYS:
            getattr(self, key + 'PreviewModeComboBox').currentIndexChanged.connect(partial(self.refresh_preview, key))
        for name in ('dmdInputPathEdit', 'slm1InputPathEdit', 'slm2InputPathEdit',
                     'slm1SdkPathEdit', 'slm2SdkPathEdit', 'ccdDllPathEdit', 'outputDirectoryEdit',
                     'f4321InputPathEdit', 'f4321DllPathEdit'):
            getattr(self, name + 'BrowseButton').clicked.connect(partial(self.browse, name))
        self.loadConfigurationButton.clicked.connect(self.load_dialog)
        self.saveConfigurationButton.clicked.connect(self.save_dialog)
        self.validateConfigurationButton.clicked.connect(partial(self.guard, self.check_parameters))
        self.enumerateDevicesButton.clicked.connect(partial(self.guard, self.enumerate_devices))
        self.connectDevicesButton.clicked.connect(partial(self.guard, self.toggle_connection))
        self.startAllButton.clicked.connect(partial(self.guard, self.start_all))
        self.stopAllButton.clicked.connect(self.stop_all)
        self.autoCalibrateButton.clicked.connect(self.calibration.open)
        self.calibrationSetupButton.clicked.connect(self.calibration.open)
        for key in ('slm1', 'slm2'):
            getattr(self, key + 'ApplyOffsetButton').clicked.connect(partial(self.guard, partial(self.apply_offset, key)))
        self.synchronizeDmdCcdCheckBox.toggled.connect(self.capture_sync_changed)
        self.slmCaptureSourceComboBox.currentIndexChanged.connect(self.slm_capture_sync_changed)
        self.ccdApplyParametersButton.clicked.connect(partial(self.guard, self.apply_camera_parameters))
        self.ccdCaptureButton.clicked.connect(partial(self.guard, self.capture_manual))
        self.ccdRecordButton.clicked.connect(partial(self.guard, self.toggle_recording))
        self.outputDirectoryEdit.textChanged.connect(lambda text: self.capturePathLabel.setText('保存目录：' + text))
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.poll_camera)
        self.poll_timer.start(self.ccdPreviewIntervalSpinBox.value())
        self.preview_fps_timer = QtCore.QTimer(self)
        self.preview_fps_timer.setInterval(1000)
        self.preview_fps_timer.timeout.connect(self.update_preview_fps)
        self.ccdActiveAreaLabel.setToolTip('最近一个统计周期的平均预览帧率，约每秒更新一次。')
        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.refresh_all_previews)
        self.ccd_detail = CcdDetailController(self)
        self.toupcam = ToupControls(self)
        self.mainTabWidget.currentChanged.connect(lambda _: self.refresh_preview('ccd'))
        if demo:
            self.setup_demo()
        if config_path:
            apply_values(self, read_configuration(config_path))
        self.toupcam.update_interval()
        self.toupcamRoiWidget.setEnabled(self.toupcamRoiCheckBox.isChecked())
        self.sync_transform_values()
        self.update_dmd_labels()
        for key in DISPLAY_KEYS:
            self.reload_safely(key)
        self.update_actions()
        self.statusBar().showMessage('演示模式：不会加载厂商 DLL 或连接真实硬件。' if demo else '就绪：配置参数后连接设备。')

    def guard(self, action, *_):
        try:
            return action()
        except Exception as exc:
            self.stop_all()
            self.statusBar().showMessage(str(exc))
            QtWidgets.QMessageBox.warning(self, '操作未完成', str(exc))
            return None

    def device_enabled(self, key, values=None):
        values = read_values(self) if values is None else values
        if key == 'ccd':
            return selected_camera(values) is not None
        return selected_dmd(values) is not None if key == 'dmd' else values[key + 'EnabledCheckBox']

    def dmd_input_field(self):
        backend = selected_dmd(read_values(self))
        return backend.input_field if backend else 'dmdInputPathEdit'

    def dmd_backend_changed(self, name, checked):
        if checked:
            for backend in DMD_BACKENDS:
                if backend.name != name:
                    control = getattr(self, backend.enabled_field)
                    with QtCore.QSignalBlocker(control):
                        control.setChecked(False)
        if hasattr(self, 'poll_timer'):
            self.update_dmd_labels()
            self.reload_safely('dmd')

    def update_dmd_labels(self):
        backend = selected_dmd(read_values(self))
        self.dmdTitleLabel.setText(backend.title if backend else 'DMD · 未启用')
        is_network = backend is not None and backend.name == 'f4321'
        self.dmdUploadButton.setVisible(is_network)
        self.dmdDeviceInfoLabel.setVisible(is_network)
        self.dmdShowCursorCheckBox.setVisible(not is_network)
        self.dmdOutputHintLabel.setText('缓存 BMP · 原始像素' if is_network else '原始像素 · 居中输出')
        self.update_conversion_controls()

    def update_conversion_controls(self):
        mode = self.dmdPixelModeComboBox.currentText()
        convert = mode != '保持原图'
        binary = mode == '1 位二值'
        self.dmdDitherCheckBox.setEnabled(binary)
        self.dmdThresholdSpinBox.setEnabled(binary and not self.dmdDitherCheckBox.isChecked())
        self.dmdInputRangeComboBox.setEnabled(convert)
        custom = convert and self.dmdInputRangeComboBox.currentText() == '自定义范围'
        self.dmdBlackLevelSpinBox.setEnabled(custom)
        self.dmdWhiteLevelSpinBox.setEnabled(custom)

    def dmd_conversion_changed(self, *_):
        self.update_conversion_controls()
        if hasattr(self, 'poll_timer'):
            self.reload_safely('dmd')

    def network_dmd(self):
        device = self.devices.get('dmd')
        return device if self.f4321EnabledCheckBox.isChecked() and device is not None else None

    def start_device_task(self, action, success, failure):
        if self.device_task is not None:
            raise ValueError('设备正在连接或上传，请等待完成。')
        self.task_success, self.task_failure = success, failure
        self.device_task = DeviceTask(action, self)
        self.device_task.finished.connect(self.finish_device_task)
        self.device_task.start()
        self.update_actions()

    def finish_device_task(self):
        task, self.device_task = self.device_task, None
        success, failure = self.task_success, self.task_failure
        self.task_success = self.task_failure = None
        cancelled = task.isInterruptionRequested() or self.closing
        try:
            if task.error or cancelled:
                failure(task.error or '设备操作已取消。')
            else:
                try:
                    success(task.result)
                except Exception as exc:
                    failure(str(exc))
        finally:
            task.deleteLater()
            self.update_actions()
            if self.closing and self.camera is None and not self.devices and not self.pending_devices:
                self.close()

    def setup_demo(self):
        self.demo_directory = tempfile.TemporaryDirectory(prefix='optical_gui_demo_')
        root = Path(self.demo_directory.name)
        for key, size in [('dmd', 640), ('slm1', 432), ('slm2', 768)]:
            folder = root / key
            folder.mkdir()
            y, x = np.indices((size, size))
            for index in range(3):
                pixels = (((x + index * 16) // 32 + y // 32) % 2 * 128).astype(np.uint8)
                Image.fromarray(pixels).save(folder / f'{index + 1:04d}.png')
            getattr(self, key + 'InputPathEdit').setText(str(folder))
            if key == 'dmd':
                self.f4321InputPathEdit.setText(str(folder))
        self.outputDirectoryEdit.setText('output/gui_demo')
        self.setWindowTitle('MainWindow · 演示模式（未连接真实硬件）')

    def browse(self, name):
        current = getattr(self, name).text()
        if name.endswith('InputPathEdit'):
            menu = QtWidgets.QMenu(self)
            image_action = menu.addAction('选择单张图像')
            directory_action = menu.addAction('选择图像目录')
            action = menu.exec_(QtGui.QCursor.pos())
            if action == image_action:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择图像', current,
                    'Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff)')
            elif action == directory_action:
                path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择图像目录', current)
            else:
                return
        elif name in ('ccdDllPathEdit', 'f4321DllPathEdit', 'toupcamDllPathEdit'):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择设备 DLL', current, 'DLL (*.dll)')
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择目录', current)
        if path:
            getattr(self, name).setText(path)
            if name.endswith('InputPathEdit'):
                self.reload_safely('dmd' if name == 'f4321InputPathEdit' else name.split('Input')[0])

    def load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '载入配置', '', 'JSON (*.json)')
        if path:
            self.guard(partial(self.load_config, Path(path)))

    def load_config(self, path):
        if self.connected or self.camera is not None or self.device_task is not None:
            raise ValueError('请先断开设备，再载入配置。')
        self.stop_all()
        apply_values(self, read_configuration(path))
        self.toupcam.update_interval()
        self.toupcamRoiWidget.setEnabled(self.toupcamRoiCheckBox.isChecked())
        self.sync_transform_values()
        self.update_dmd_labels()
        for key in DISPLAY_KEYS:
            self.reload_safely(key)
        self.statusBar().showMessage(f'已载入配置：{path}')

    def save_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '保存配置', 'optical_config.json', 'JSON (*.json)')
        if path:
            self.guard(lambda: write_configuration(Path(path), read_values(self)))

    def reload_safely(self, key):
        if self.device_task is not None or self.calibration.active:
            return
        if key in self.devices:
            if key == 'dmd' and self.network_dmd() is not None:
                return
            field = self.dmd_input_field() if key == 'dmd' else key + 'InputPathEdit'
            self.inputs.import_safely({key: getattr(self, field).text()})
            return
        try:
            self.load_sequence(key)
        except Exception as exc:
            self.paths[key] = []
            self.images.pop(key, None)
            self.source_images.pop(key, None)
            getattr(self, key + 'PreviewLabel').setText('未载入图像')
            getattr(self, key + 'TotalFramesLabel').setText('/ —')
            self.statusBar().showMessage(f'{key.upper()}：{exc}')
        self.update_actions()

    def load_sequence(self, key):
        self.pause(key)
        field = self.dmd_input_field() if key == 'dmd' else key + 'InputPathEdit'
        path = input_path(getattr(self, field).text())
        paths = image_files.collect_images(path, None, None)
        for p in paths:
            with Image.open(p) as image:
                image.verify()
                if key == 'slm2' and image.mode != 'L':
                    raise ValueError('SLM2 仅支持 8 bit 单通道相位图。')
                if key == 'slm1' and image.mode not in {'1', 'L', 'I', 'I;16'}:
                    raise ValueError('SLM1 需要单通道相位图。')
        self.paths[key] = paths
        self.indices[key] = 0
        spin = getattr(self, key + 'FrameSpinBox')
        with QtCore.QSignalBlocker(spin):
            spin.setRange(1, len(paths))
            spin.setValue(1)
        getattr(self, key + 'TotalFramesLabel').setText(f'/ {len(paths)}')
        self.render_frame(key, 0)

    def validate(self):
        values = read_values(self)
        selected_dmd(values)
        enabled = [key for key in (*DISPLAY_KEYS, 'ccd') if self.device_enabled(key, values)]
        if not enabled:
            raise ValueError('请至少启用一个设备。')
        if values['synchronizeDmdCcdCheckBox']:
            if 'dmd' not in enabled or 'ccd' not in enabled:
                raise ValueError('DMD 同步采集需要同时启用 DMD 和 CCD。')
            if values['dmdSettleSpinBox'] >= values['dmdIntervalSpinBox']:
                raise ValueError('DMD 采集前稳定时间必须小于播放间隔。')
        source = self.slm_capture_source()
        if source:
            if values['synchronizeDmdCcdCheckBox']:
                raise ValueError('DMD 和 SLM 同步采集只能选择一个来源。')
            if source not in enabled or 'ccd' not in enabled:
                raise ValueError(f'{source.upper()} 同步采集需要同时启用该 SLM 和相机。')
        for key in DISPLAY_KEYS:
            if key == 'dmd' and key in enabled and values['f4321EnabledCheckBox']:
                self.load_sequence(key)
        input_path(values['outputDirectoryEdit'])
        if 'dmd' in enabled:
            options = DmdImageOptions.from_settings(values)
            if values['f4321EnabledCheckBox']:
                if options.pixel_mode == 'original':
                    raise ValueError('F4321 输入请选择“8 位灰度”或“1 位二值”。')
                for name in ('f4321LocalIpEdit', 'f4321TargetIpEdit'):
                    ipaddress.IPv4Address(values[name].strip())
                if not self.demo and not input_path(values['f4321DllPathEdit']).is_file():
                    raise FileNotFoundError('F4321 DLL 不存在，请选择 F4321_HS.dll。')
        if not self.demo:
            for key, field, suffix in [('slm1', 'slm1SdkPathEdit', 'SecondDll.dll'),
                                       ('slm2', 'slm2SdkPathEdit', 'HDSLMFunc.dll'),
                                       ('ccd', 'toupcamDllPathEdit' if selected_camera(values) == 'toupcam' else 'ccdDllPathEdit', '')]:
                if key in enabled:
                    sdk_path = input_path(values[field])
                    if suffix:
                        sdk_path /= suffix
                    if not sdk_path.is_file():
                        raise FileNotFoundError(f'{key.upper()} SDK 不存在：{sdk_path}')
        return values

    def check_parameters(self):
        self.validate()
        message = '连接参数检查通过。普通 DMD / SLM 无需输入图像即可连接；图像在导入、播放时另行检查。'
        self.statusBar().showMessage(message)
        QtWidgets.QMessageBox.information(self, '参数检查', message)

    def validate_pairs(self):
        if [p.stem for p in self.paths['slm1']] != [p.stem for p in self.paths['slm2']]:
            raise ValueError('同步播放要求 SLM1 与 SLM2 的文件主干名称及顺序一致。')

    def enumerate_devices(self):
        if self.demo:
            QtWidgets.QMessageBox.information(self, '演示设备', 'DMD、SLM1、SLM2：虚拟显示区域\nCCD：合成图像源\n未访问真实 SDK。')
            return
        lines = ['编号说明：DMD / SLM1 使用 Windows monitor；SLM2 使用 SDK ID。两者按显示区域匹配，不能互换。']
        monitors = []
        if self.f4321EnabledCheckBox.isChecked():
            lines.append(f'F4321：按配置网络地址查询 {self.f4321TargetIpEdit.text()}:{self.f4321TargetPortSpinBox.value()}；点击“连接设备”读取型号、面板尺寸并上传图像。')
        try:
            monitors = zkwx.enumerate_monitors()
            for monitor in monitors:
                suffix = ' [主屏]' if monitor.primary else ''
                lines.append(f'Windows monitor {monitor.index}: {monitor.width}×{monitor.height} @ ({monitor.left}, {monitor.top}){suffix}')
        except Exception as exc:
            lines.append(f'显示器枚举失败：{exc}')
        for enabled, factory, field, method, description in [
            (self.slm2EnabledCheckBox.isChecked(), magic.HDSLM8BitSDK, 'slm2SdkPathEdit', 'list_displays', 'SLM2'),
            (self.ccdEnabledCheckBox.isChecked(), dvp.DvpApi, 'ccdDllPathEdit', 'list_devices', 'CCD'),
        ]:
            if not enabled:
                continue
            api = None
            try:
                api = factory(input_path(getattr(self, field).text()))
                devices = getattr(api, method)()
                lines.append(f'SLM2 SDK：{len(devices)} 个显示目标（含普通显示器）' if description == 'SLM2'
                             else f'{description}：{len(devices)} 个设备')
                for index, device in enumerate(devices):
                    if description == 'CCD':
                        identity = camera_identity(device)
                        lines.append(f'  {index}: {identity["friendly_name"] or "未命名"} / 型号：{identity["model"] or "未知"} / 序列号：{identity["serial"] or "未知"}')
                    else:
                        lines.append(f'  SDK ID {device.index}: {device.name} / {device.width}×{device.height} @ ({device.left}, {device.top})')
                        matches = [m.index for m in monitors if
                                   (m.left, m.top, m.width, m.height) ==
                                   (device.left, device.top, device.width, device.height)]
                        lines.append(f'    对应 Windows monitor {matches[0]}' if len(matches) == 1
                                     else '    Windows monitor 无法唯一匹配，请检查扩展显示和缩放')
            except Exception as exc:
                lines.append(f'{description} 枚举失败：{exc}')
            finally:
                if api is not None:
                    api.close()
        if self.toupcamEnabledCheckBox.isChecked():
            self.toupcam.enumerate(lines)
            return
        QtWidgets.QMessageBox.information(self, '设备枚举', '\n'.join(lines))

    def toggle_connection(self):
        if self.device_task is not None:
            self.device_task.requestInterruption()
            self.statusBar().showMessage('正在取消设备连接或上传，等待 SDK 释放资源。')
        elif self.connected or self.camera is not None or self.devices:
            self.disconnect_devices()
        else:
            self.connect_devices()

    def connect_devices(self):
        if self.device_task is not None:
            raise ValueError('设备操作尚未完成。')
        self.stop_all()
        settings = self.validate()
        devices = {}
        try:
            # Resolve the named SLM2 before selecting SLM1 from the remaining monitors.
            for key in ('dmd', 'slm2', 'slm1'):
                if self.device_enabled(key, settings):
                    paths = self.paths[key] if key == 'dmd' and settings['f4321EnabledCheckBox'] else []
                    if self.display_factory is create_display:
                        occupied = [device.geometry for device in devices.values() if device.geometry is not None]
                        devices[key] = self.display_factory(key, settings, paths, self.demo, occupied=occupied)
                    else:
                        devices[key] = self.display_factory(key, settings, paths, self.demo)
            geometries = [device.geometry for device in devices.values() if device.geometry is not None]
            if len(set(geometries)) != len(geometries):
                raise ValueError('DMD、SLM1、SLM2 不能使用同一个显示区域，请修改设备编号。')
            for key, device in devices.items():
                if key != 'dmd' or not settings['f4321EnabledCheckBox']:
                    device.open()
        except Exception:
            for device in devices.values():
                try:
                    device.close()
                except Exception:
                    pass
            raise
        if settings['f4321EnabledCheckBox']:
            self.pending_devices = devices
            self.dmdStateLabel.setText('● 连接 / 上传中')
            self.dmdDeviceInfoLabel.setText('正在查询面板并准备 BMP…')
            self.statusBar().showMessage('正在连接 F4321 并上传转换后的图像；可取消。')
            self.start_device_task(lambda cancel: devices['dmd'].open(cancel),
                lambda _: self.complete_connection(settings, devices), self.connection_failed)
            return
        self.complete_connection(settings, devices)

    def connection_failed(self, message):
        retained = {}
        for key, device in {**self.pending_devices, **self.devices}.items():
            try:
                device.close()
            except Exception as exc:
                retained[key] = device
                message += f'；{key} 关闭失败：{exc}，请重试断开。'
        self.pending_devices = {}
        self.devices = retained
        self.connected = bool(retained)
        self.dmdStateLabel.setText('● 连接未完成')
        self.dmdDeviceInfoLabel.setText('未就绪，请检查网络、SDK 和图像参数')
        self.statusBar().showMessage(message)
        if not self.closing:
            QtWidgets.QMessageBox.warning(self, 'F4321 连接 / 上传失败', message)

    def complete_connection(self, settings, devices):
        self.pending_devices = {}
        self.settings, self.devices = settings, devices
        self.connected = True
        self.saved_frames = self.received_frames = 0
        self.reset_preview_fps()
        for key, device in devices.items():
            if device.output is not None:
                device.output.closed.connect(self.disconnect_devices)
            getattr(self, key + 'StateLabel').setText('● 演示设备' if self.demo else '● 已连接')
            if device.geometry is not None:
                display = device.display
                identifier = f'SDK ID {display.index} / {display.name}' if key == 'slm2' else f'Windows monitor {display.index}'
                getattr(self, key + 'StateLabel').setToolTip(
                    f'实际选择：{identifier}\n{display.width}×{display.height} @ ({display.left}, {display.top})')
            if key == 'dmd' and settings['f4321EnabledCheckBox']:
                self.render_frame(key, self.indices[key])
        try:
            self.inputs.initialize()
        except Exception:
            self.disconnect_devices()
            raise
        network = self.network_dmd()
        if network is not None:
            self.dmdDeviceInfoLabel.setText(f'缓存就绪 · {len(self.paths["dmd"])} 张')
            self.dmdDeviceInfoLabel.setToolTip(network.info_text)
        try:
            if self.device_enabled('ccd', settings):
                self.camera = self.camera_factory(settings, self.demo, parent=self)
                self.camera.ready.connect(self.on_camera_ready)
                self.camera.parametersChanged.connect(self.on_camera_parameters)
                self.camera.failed.connect(self.on_camera_error)
                self.camera.warning.connect(self.on_camera_warning)
                self.camera.captured.connect(self.on_capture)
                self.camera.captureFinished.connect(self.on_slm_capture_finished)
                self.camera.recordingChanged.connect(self.on_recording)
                self.camera.finished.connect(self.on_camera_finished)
                self.ccdStateLabel.setText('● 连接中')
                self.camera.start()
        except Exception:
            self.disconnect_devices()
            if self.camera is not None and not self.camera.isRunning():
                self.on_camera_finished()
            raise
        self.sessionHintLabel.setText('当前实验：' + settings['sessionNameEdit'])
        self.mainTabWidget.setCurrentIndex(1)
        self.update_actions()

    def on_camera_ready(self, parameters):
        if self.disconnecting:
            return
        self.camera_ready = True
        self.camera_parameters = dict(parameters)
        self.toupcam.ready(parameters)
        self.ccd_detail.start_session(parameters.get('camera', {}), self.settings or {})
        self.reset_preview_fps()
        self.preview_fps_started_at = time.monotonic()
        self.preview_fps_timer.start()
        identity = parameters.get('camera', {})
        if not self.demo:
            self.ccdTitleLabel.setText('相机 · ' + (identity.get('model') or '型号未返回'))
            self.ccdTitleLabel.setToolTip(
                f'实际型号：{identity.get("model") or "未知"}\n'
                f'序列号：{identity.get("serial") or "未知"}\n'
                f'设备名称：{identity.get("friendly_name") or "未命名"}')
        else:
            self.ccdTitleLabel.setToolTip('演示模式：使用合成图像，未连接真实相机。')
        self.on_camera_parameters(parameters)
        self.ccdStateLabel.setText('● 演示图像源' if self.demo else '● 已连接')
        self.update_actions()
        self.calibration.camera_ready()

    def on_camera_parameters(self, parameters):
        self.camera_parameters.update(parameters)
        for field, key, default in [(self.ccdExposureSpinBox, 'exposure_range', self.ccd_default_ranges[0]),
                                    (self.ccdGainSpinBox, 'gain_range', self.ccd_default_ranges[1])]:
            field.setRange(*parameters.get(key, default))
        self.ccdExposureSpinBox.setValue(parameters['exposure'])
        self.ccdGainSpinBox.setValue(parameters['gain'])
        self.ccdAutoExposureCheckBox.setChecked(parameters['auto'])
        self.statusBar().showMessage('已读取 CCD 当前曝光和增益。')

    def on_camera_error(self, message):
        self.disconnect_devices()
        self.statusBar().showMessage('CCD 错误：' + message)
        if not self.closing:
            QtWidgets.QMessageBox.warning(self, 'CCD 连接 / 采集失败', message)

    def on_camera_warning(self, message):
        self.statusBar().showMessage(message)
        self.ccdStateLabel.setText('● ' + message[:50])

    def disconnect_devices(self, *_):
        if self.calibration.active:
            self.calibration.disconnect_after = True
            self.calibration.cancel()
            if self.calibration.task is not None:
                return
        if self.device_task is not None:
            self.device_task.requestInterruption()
            return
        if self.disconnecting:
            return
        self.disconnecting = True
        self.stop_all()
        devices, self.devices = self.devices, {}
        for key, device in devices.items():
            try:
                device.close()
            except Exception as exc:
                self.statusBar().showMessage(f'{key} 关闭失败：{exc}')
                self.devices[key] = device
            getattr(self, key + 'StateLabel').setText('● 关闭失败，请重试' if key in self.devices else '● 未连接')
            if key not in self.devices:
                getattr(self, key + 'StateLabel').setToolTip('')
        self.connected, self.camera_ready = bool(self.devices), False
        if self.camera is not None:
            self.camera.requestInterruption()
        else:
            self.disconnecting = False
        self.update_actions()

    def on_camera_finished(self):
        self.cancel_slm_capture()
        for key in ('slm1', 'slm2'):
            if self.slm_capture_source() == key:
                self.pause(key)
        worker, self.camera = self.camera, None
        if worker is not None:
            worker.deleteLater()
        self.camera_ready = self.recording = self.recording_pending = False
        self.disconnecting = False
        self.ccdStateLabel.setText('● 未连接')
        self.ccd_detail.clear()
        self.toupcam.disconnected()
        self.camera_parameters = {}
        self.camera_preview_packet = None
        for control, limits in zip((self.ccdExposureSpinBox, self.ccdGainSpinBox), self.ccd_default_ranges):
            control.setRange(*limits)
        self.reset_preview_fps()
        self.ccdTitleLabel.setText(self.ccd_default_title)
        self.ccdTitleLabel.setToolTip(self.ccd_default_tooltip)
        self.update_actions()
        if self.closing:
            self.close()

    def group_for(self, key):
        if key in ('slm1', 'slm2') and self.synchronizeSlmsCheckBox.isChecked() and self.slm1EnabledCheckBox.isChecked() and self.slm2EnabledCheckBox.isChecked():
            self.validate_pairs()
            return ('slm1', 'slm2')
        return (key,)

    def render_frame(self, key, index):
        self.publish_frame(key, self.prepare_frame(key, index))

    def prepare_frame(self, key, index, values=None, path=None):
        """Prepare and validate without changing the preview or hardware."""
        values = read_values(self) if values is None else values
        path = self.paths[key][index] if path is None else path
        transform = ImageTransform.from_settings(values, key)
        background = values['slm2BackgroundSpinBox'] if key == 'slm2' else 0
        opened = prepare_image(path, DmdImageOptions.from_settings(values)) if key == 'dmd' else Image.open(path)
        with opened as original:
            with transform.apply_pil(original, fill=background) as transformed:
                source = pil_image_qimage(transformed)
        display = self.devices[key].display if key in self.devices else zkwx.Monitor(0, 0, 0, 1920, 1080, False)
        unknown_panel = key == 'dmd' and values.get('f4321EnabledCheckBox') and key not in self.devices
        if unknown_panel:
            display = zkwx.Monitor(0, 0, 0, max(1920, source.width()), max(1080, source.height()), False)
        offset_x = values[key + 'OffsetXSpinBox'] if key != 'dmd' else 0
        offset_y = values[key + 'OffsetYSpinBox'] if key != 'dmd' else 0
        source_size = (source.width(), source.height())
        if key == 'slm1':
            origin = image_files.canvas_origin(source_size, display, offset_x, offset_y)
        elif key == 'slm2':
            origin = magic_images.canvas_origin(source_size, display, offset_x, offset_y)
        else:
            if source.width() > display.width or source.height() > display.height:
                raise ValueError('DMD 输入超过面板尺寸，不能原尺寸显示。')
            origin = ((display.width - source.width()) // 2, (display.height - source.height()) // 2)
        canvas = QtGui.QImage(display.width, display.height, QtGui.QImage.Format_ARGB32)
        canvas.fill(QtGui.QColor(background, background, background))
        painter = QtGui.QPainter(canvas)
        painter.drawImage(origin[0], origin[1], source)
        painter.end()
        return source, canvas, origin, display, path, transform

    def publish_frame(self, key, prepared):
        source, canvas, origin, display, path, transform = prepared
        self.transforms[key] = transform
        self.source_images[key] = source
        self.images[key] = canvas
        self.refresh_preview(key)
        getattr(self, key + 'FrameInfoLabel').setText('文件：' + path.name)
        getattr(self, key + 'FrameInfoLabel').setToolTip(str(path))
        unknown_panel = key == 'dmd' and self.f4321EnabledCheckBox.isChecked() and key not in self.devices
        getattr(self, key + 'PanelInfoLabel').setText('面板尺寸：连接后读取' if unknown_panel else f'面板 {display.width} × {display.height}')
        getattr(self, key + 'ActiveAreaLabel').setText(f'有效区 {source.width()} × {source.height()}')
        getattr(self, key + 'ActiveAreaLabel').setToolTip(f'有效区左上角：({origin[0]}, {origin[1]}) 面板像素')

    def show_frame(self, key, index, capture=False):
        if key == 'dmd' and capture and self.synchronizeDmdCcdCheckBox.isChecked():
            self.require_capture_sync_ready()
        values = read_values(self)
        group = self.group_for(key)
        source = self.slm_capture_source()
        slm_capture = capture and source in group
        if slm_capture:
            self.require_slm_capture_ready()
            if self.slm_capture_request is not None:
                raise ValueError('请等待当前 SLM 图案采集保存完成。')
        prepared = {}
        for target in group:
            if not self.paths[target]:
                raise ValueError(f'{target.upper()} 未载入图像。')
            prepared[target] = self.prepare_frame(target, index, values)
        for target in group:
            path = self.paths[target][index]
            if target in self.devices:
                self.devices[target].show(path, values)
            self.indices[target] = index
            with QtCore.QSignalBlocker(getattr(self, target + 'FrameSpinBox')):
                getattr(self, target + 'FrameSpinBox').setValue(index + 1)
            self.publish_frame(target, prepared[target])
            self.inputs.tests.pop(target, None)
        if key == 'dmd' and capture and self.synchronizeDmdCcdCheckBox.isChecked():
            self.queue_capture(self.paths[key][index].stem + '.png', automatic=True)
        elif slm_capture:
            self.queue_capture(self.paths[source][index].stem + '.png', automatic=True, source=source)

    def interval(self, key):
        return getattr(self, key + 'IntervalSpinBox').value()

    def play(self, key):
        if self.device_task is not None:
            raise ValueError('设备正在连接或上传。')
        group = self.group_for(key)
        master = group[0]
        if self.connected and any(target not in self.devices for target in group):
            raise ValueError('目标显示设备未连接。')
        if not self.paths[master]:
            raise ValueError('请先载入图像。')
        if master == 'dmd' and self.network_dmd() is not None:
            period_us = self.f4321ExposureSpinBox.value() + self.f4321DarkTimeSpinBox.value()
            if self.interval('dmd') * 1000000 < period_us:
                raise ValueError('F4321 切图间隔不能小于一张图像的曝光与暗场时间之和。')
        if master == 'dmd' and self.synchronizeDmdCcdCheckBox.isChecked():
            self.require_capture_sync_ready()
        if self.slm_capture_source() in group:
            self.require_slm_capture_ready()
            names = [p.stem.casefold() for p in self.paths[self.slm_capture_source()]]
            if len(names) != len(set(names)):
                raise ValueError('SLM 序列存在相同主文件名，无法一一对应保存；请先修改输入文件名。')
        self.pause(master)
        index = self.indices[master]
        if index == len(self.paths[master]) - 1:
            index = 0
        self.show_frame(master, index, capture=True)
        self.timers[master].start(max(1, round(self.interval(master) * 1000)))
        self.statusBar().showMessage(('设备播放：' if self.connected else '本地预览：') + ', '.join(group))
        self.update_actions()

    def advance(self, key):
        if (self.slm_capture_request is not None
                and self.slm_capture_request.metadata['capture_master'] == key):
            return  # Hold the physical pattern until the matching PNG is saved.
        index = self.indices[key] + 1
        if index >= len(self.paths[key]):
            if key == 'dmd' and self.dmdLoopCheckBox.isChecked():
                index = 0
            else:
                self.pause(key)
                if key == 'slm1' and not self.slm1HoldLastFrameCheckBox.isChecked() and 'slm1' in self.devices:
                    self.devices['slm1'].blank()
                return
        self.show_frame(key, index, capture=True)

    def pause(self, key, *_):
        targets = ('slm1', 'slm2') if key in ('slm1', 'slm2') and self.synchronizeSlmsCheckBox.isChecked() else (key,)
        for target in targets:
            if target in self.timers:
                self.timers[target].stop()
        if (self.slm_capture_request is not None
                and self.slm_capture_request.metadata['capture_source'] in targets):
            self.cancel_slm_capture()
        if key == 'dmd' and self.camera is not None:
            self.camera.submit('cancel_dmd_captures')
        # Hold the selected F4321 image by letting its bit planes keep cycling.
        # A board Stop would freeze/end the PWM needed for eight-bit grayscale.
        # stop_all/disconnect explicitly stop board playback instead.
        if hasattr(self, 'poll_timer'):
            self.update_actions()

    def step(self, key, delta):
        self.pause(key)
        index = min(max(self.indices[key] + delta, 0), len(self.paths[key]) - 1)
        self.show_frame(key, index)

    def select_frame(self, key, value):
        if not self.paths[key]:
            return
        def select():
            self.pause(key)
            self.show_frame(key, value - 1)
        self.guard(select)

    def start_all(self):
        if not self.connected:
            raise ValueError('请先连接设备。')
        if self.synchronizeDmdCcdCheckBox.isChecked():
            self.require_capture_sync_ready()
        if self.slm_capture_source():
            self.require_slm_capture_ready()
        started = set()
        # Send phase patterns before the first DMD exposure/capture window.
        for key in ('slm1', 'slm2', 'dmd'):
            if key not in self.devices or not self.paths[key]:
                continue
            if key in started:
                continue
            self.play(key)
            started.update(self.group_for(key))

    def stop_all(self, *_):
        self.cancel_slm_capture()
        self.calibration.cancel()
        for timer in self.timers.values():
            timer.stop()
        if self.device_task is not None:
            self.device_task.requestInterruption()
        network = self.network_dmd()
        if network is not None and self.device_task is None:
            try:
                network.stop()
            except Exception as exc:
                self.statusBar().showMessage(f'F4321 停止失败：{exc}')
        if self.camera is not None:
            self.camera.submit('cancel_captures')
            self.camera.submit('record_stop')
        if hasattr(self, 'poll_timer'):
            self.update_actions()

    def apply_offset(self, key='slm2'):
        if self.slm_capture_running():
            raise ValueError('请先暂停 SLM 同步采集，再修改偏移。')
        if self.paths[key]:
            # Validate the target canvas before sending either paired SLM frame.
            self.render_frame(key, self.indices[key])
            self.show_frame(key, self.indices[key])
            self.statusBar().showMessage(f'{key.upper()} 偏移已应用；切换“完整面板”可查看图案位置。')

    def reupload_dmd(self):
        network = self.network_dmd()
        if network is None:
            raise ValueError('请先连接 F4321 DMD。')
        if self.device_task is not None:
            raise ValueError('正在上传，请等待完成。')
        self.stop_all()
        values = read_values(self)
        previous = self.transforms['dmd']
        index = self.indices['dmd']
        self.dmdStateLabel.setText('● 上传中')
        self.dmdDeviceInfoLabel.setText('正在转换并重新上传…')

        def complete(_):
            try:
                network.show(self.paths['dmd'][index], values)
                self.render_frame('dmd', index)
                for suffix in TRANSFORM_SUFFIXES:
                    self.settings['dmd' + suffix] = values['dmd' + suffix]
                self.dmdStateLabel.setText('● 演示设备' if self.demo else '● 已连接')
                self.dmdDeviceInfoLabel.setText(f'缓存就绪 · {len(self.paths["dmd"])} 张')
                self.statusBar().showMessage('F4321 上传完成，当前帧已显示。')
            except Exception as exc:
                failed(str(exc))

        def failed(message):
            for suffix, value in zip(TRANSFORM_SUFFIXES, (previous.angle, previous.horizontal, previous.vertical)):
                control = getattr(self, 'dmd' + suffix)
                with QtCore.QSignalBlocker(control):
                    control.setValue(value) if suffix == 'RotationSpinBox' else control.setChecked(value)
            network.uploaded = False
            self.dmdStateLabel.setText('● 未就绪')
            self.dmdDeviceInfoLabel.setText('上传未完成 · 请断开后重连')
            self.statusBar().showMessage(message)
            if not self.closing:
                QtWidgets.QMessageBox.warning(self, 'F4321 上传失败', message)
            if self.closing:
                self.disconnect_devices()

        self.start_device_task(lambda cancel: network.reload(values, cancel), complete, failed)

    def sync_transform_values(self):
        values = read_values(self)
        self.transforms = {key: ImageTransform.from_settings(values, key)
                           for key in (*DISPLAY_KEYS, 'ccd')}

    def transform_editable(self, key):
        if key in self.inputs.tests:
            return False
        if self.calibration.active or self.disconnecting or self.device_task is not None:
            return False
        synchronized_capture = self.synchronizeDmdCcdCheckBox.isChecked() and self.timers['dmd'].isActive()
        if synchronized_capture or self.slm_capture_running():
            return False
        if key == 'ccd':
            return not (self.recording or self.recording_pending or
                        (self.camera is not None and not self.camera_ready))
        targets = ('slm1', 'slm2') if key in ('slm1', 'slm2') and self.synchronizeSlmsCheckBox.isChecked() else (key,)
        return not any(self.timers[target].isActive() for target in targets)

    def transform_changed(self, key, *_):
        self.guard(partial(self.apply_transform, key))

    def apply_transform(self, key):
        previous = self.transforms[key]
        values = read_values(self)
        transform = ImageTransform.from_settings(values, key)
        if transform == previous:
            return
        try:
            if not self.transform_editable(key):
                raise ValueError('请先暂停播放或停止录像，再调整图像方向。')
            if key == 'ccd':
                if self.camera is not None:
                    self.camera.submit('transform', transform)
            elif self.paths[key]:
                # Check every frame before applying an orientation to a sequence.
                display = self.devices[key].display if key in self.devices else zkwx.Monitor(0, 0, 0, 1920, 1080, False)
                for path in self.paths[key]:
                    with Image.open(path) as source:
                        size = transform.output_size(source.size)
                    if key == 'slm1':
                        image_files.canvas_origin(size, display, values['slm1OffsetXSpinBox'], values['slm1OffsetYSpinBox'])
                    elif key == 'slm2':
                        magic_images.canvas_origin(size, display, values['slm2OffsetXSpinBox'], values['slm2OffsetYSpinBox'])
                    elif not (self.f4321EnabledCheckBox.isChecked() and key not in self.devices) and (size[0] > display.width or size[1] > display.height):
                        raise ValueError(f'{path.name} 旋转后为 {size[0]}×{size[1]}，超过面板 {display.width}×{display.height}。')
                prepared = self.prepare_frame(key, self.indices[key], values)
                if key == 'dmd' and self.network_dmd() is not None:
                    self.reupload_dmd()
                    return
                if key in self.devices:
                    self.devices[key].show(self.paths[key][self.indices[key]], values)
                self.publish_frame(key, prepared)
            self.transforms[key] = transform
            if self.settings is not None:
                for suffix in TRANSFORM_SUFFIXES:
                    self.settings[key + suffix] = values[key + suffix]
            self.statusBar().showMessage(f'{key.upper()} 图像方向已应用：{transform.angle:g}°，先翻转后旋转。')
        except Exception:
            for suffix, value in zip(TRANSFORM_SUFFIXES, (previous.angle, previous.horizontal, previous.vertical)):
                control = getattr(self, key + suffix)
                with QtCore.QSignalBlocker(control):
                    if suffix == 'RotationSpinBox':
                        control.setValue(value)
                    else:
                        control.setChecked(value)
            raise

    def require_capture_sync_ready(self):
        if not self.device_enabled('dmd') or not self.device_enabled('ccd'):
            raise ValueError('DMD 同步采集需要同时启用 DMD 和 CCD。')
        if not self.connected or 'dmd' not in self.devices or not self.camera_ready or not self.received_frames:
            raise ValueError('DMD 同步采集需要先连接 DMD 和 CCD，并等待相机第一帧。')
        network = self.network_dmd()
        if self.device_task is not None or (network is not None and not network.ready):
            raise ValueError('请等待 F4321 图像上传并确认就绪。')
        if self.dmdSettleSpinBox.value() >= self.interval('dmd'):
            raise ValueError('稳定时间必须小于 DMD 播放间隔。')

    def capture_sync_changed(self, checked):
        if checked:
            with QtCore.QSignalBlocker(self.slmCaptureSourceComboBox):
                self.slmCaptureSourceComboBox.setCurrentIndex(0)
            self.cancel_slm_capture()
        if not checked and self.camera is not None:
            self.camera.submit('cancel_auto_captures')
        self.update_actions()

    def slm_capture_source(self):
        return (None, 'slm1', 'slm2')[self.slmCaptureSourceComboBox.currentIndex()]

    def slm_capture_running(self):
        source = self.slm_capture_source()
        targets = ('slm1', 'slm2') if self.synchronizeSlmsCheckBox.isChecked() else (source,)
        return self.slm_capture_request is not None or bool(source and any(
            self.timers[key].isActive() for key in targets))

    def slm_capture_sync_changed(self, *_):
        self.cancel_slm_capture()
        if self.slm_capture_source():
            self.synchronizeDmdCcdCheckBox.setChecked(False)
        self.update_actions()

    def cancel_slm_capture(self):
        request, self.slm_capture_request = self.slm_capture_request, None
        if request is not None:
            request.cancelled.set()

    def require_slm_capture_ready(self):
        source = self.slm_capture_source()
        if not source or not self.device_enabled(source) or not self.device_enabled('ccd'):
            raise ValueError('SLM 同步采集需要同时启用所选 SLM 和相机。')
        if (not self.connected or source not in self.devices or not self.camera_ready
                or not self.received_frames or self.disconnecting or self.device_task is not None):
            raise ValueError('SLM 同步采集需要先连接所选 SLM 和相机，并等待相机第一帧。')
        if self.synchronizeDmdCcdCheckBox.isChecked():
            raise ValueError('DMD 和 SLM 同步采集只能选择一个来源。')

    def on_slm_capture_finished(self, capture_id, success, message):
        request = self.slm_capture_request
        if request is None or request.metadata['capture_id'] != capture_id:
            return  # Ignore manual captures and late results from cancelled runs.
        self.slm_capture_request = None
        if not success:
            self.pause(request.metadata['capture_master'])
            self.on_camera_warning('SLM 同步采集已停止：' + message)
        self.update_actions()

    def apply_camera_parameters(self):
        if not self.camera_ready:
            raise ValueError('CCD 尚未连接。')
        self.camera.submit('parameters', self.ccdExposureSpinBox.value(), self.ccdGainSpinBox.value(),
                           self.ccdAutoExposureCheckBox.isChecked())

    def queue_capture(self, filename, automatic=False, source='dmd'):
        if not self.camera_ready:
            raise ValueError('CCD 尚未就绪。')
        slm_capture = automatic and source in ('slm1', 'slm2')
        if slm_capture:
            self.require_slm_capture_ready()
        elif automatic:
            if not self.synchronizeDmdCcdCheckBox.isChecked():
                return
            self.require_capture_sync_ready()
        now = time.monotonic()
        settings = self.settings
        after = now + self.dmdSettleSpinBox.value() if automatic else now
        deadline = now + self.interval('dmd') if automatic else now + max(5, camera_value(settings, 'TimeoutSpinBox', 500) / 1000)
        if slm_capture:
            after = now + self.slmCaptureSettleSpinBox.value()
            deadline = after + max(5, camera_value(settings, 'TimeoutSpinBox', 500) / 1000,
                                   3 * self.ccdExposureSpinBox.value() / 1e6 + 1)
        if not automatic and selected_camera(settings) == 'toupcam':
            # Fresh capture may discard an in-flight exposure before the saved one.
            deadline = max(deadline, now + 3*self.ccdExposureSpinBox.value()/1e6 + 1)
        metadata = {'experiment': settings['sessionNameEdit'], 'demo': self.demo, 'automatic': automatic,
                    'display_files': {k: str(self.inputs.current_path(k)) for k in self.devices},
                    'test_patterns': {k: name for k, (name, _) in self.inputs.tests.items() if k in self.devices},
                    'display_transforms': {k: self.transforms[k].as_dict() for k in self.devices},
                    'dmd_conversion': DmdImageOptions.from_settings(settings).as_dict(),
                    'dmd_backend': 'f4321' if settings.get('f4321EnabledCheckBox') else 'desktop'}
        if slm_capture:
            metadata.update(capture_source=source, capture_master=self.group_for(source)[0],
                            capture_id=uuid4().hex,
                            source_file=str(self.paths[source][self.indices[source]]),
                            discard_after_settle=1,
                            settle_seconds=self.slmCaptureSettleSpinBox.value())
        request = CaptureRequest(input_path(settings['outputDirectoryEdit']) / filename, after,
                                 settings['overwriteCheckBox'], settings['saveManifestCheckBox'], metadata, deadline)
        if slm_capture:
            if request.path.resolve() in {path.resolve() for paths in self.paths.values() for path in paths}:
                raise ValueError('采集文件与输入图像路径相同，请更换保存目录，避免覆盖相位图。')
            request.discard_frames = 1
            self.slm_capture_request = request
        self.camera.submit('capture', request)

    def capture_manual(self):
        self.queue_capture('frame_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8] + '.png')

    def toggle_recording(self):
        if not self.camera_ready:
            raise ValueError('CCD 尚未就绪。')
        if self.recording:
            self.camera.submit('record_stop')
        else:
            path = input_path(self.settings['outputDirectoryEdit']) / ('ccd_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8] + '.avi')
            self.camera.submit('record_start', path, 20.0)
        self.recording_pending = True
        self.update_actions()

    def on_recording(self, active, path):
        self.recording, self.recording_pending = active, False
        self.ccdRecordButton.setText('停止录像' if active else '开始录像')
        if path:
            self.statusBar().showMessage(('正在录像：' if active else '录像已保存：') + path)
        self.update_actions()

    def on_capture(self, path):
        self.saved_frames += 1
        self.ccdFramesReceivedLabel.setText(f'已保存 {self.saved_frames} 帧')
        self.statusBar().showMessage('已保存：' + path)

    def reset_preview_fps(self):
        self.preview_fps_timer.stop()
        self.preview_fps_started_at = None
        self.preview_fps_frames = 0
        self.ccdActiveAreaLabel.setText('预览 — FPS')

    def update_preview_fps(self):
        if self.preview_fps_started_at is None:
            return
        now = time.monotonic()
        elapsed = now - self.preview_fps_started_at
        if elapsed <= 0:
            return
        fps = self.preview_fps_frames / elapsed
        self.ccdActiveAreaLabel.setText(f'预览 {fps:.1f} FPS')
        self.ccd_detail.update_status()
        self.toupcam.update_fps()
        if self.camera_preview_packet is not None:
            metadata = self.camera_preview_packet[1]
            if metadata.get('auto'):
                for field, key in [(self.ccdExposureSpinBox, 'exposure'), (self.ccdGainSpinBox, 'gain')]:
                    if not field.hasFocus():
                        with QtCore.QSignalBlocker(field):
                            field.setValue(metadata[key])
        self.preview_fps_started_at = now
        self.preview_fps_frames = 0

    def poll_camera(self):
        if self.camera is None or self.disconnecting:
            return
        if hasattr(self.camera, 'take_latest_with_raw'):
            packet = self.camera.take_latest_with_raw()
        else:
            legacy = self.camera.take_latest()
            packet = None if legacy is None else (*legacy, legacy[0] if
                ImageTransform(**legacy[1].get('image_transform', {})).is_identity else None)
        if packet is None:
            return
        image, metadata, raw = packet
        self.received_frames += 1
        self.camera_preview_packet = (image, metadata)
        self.ccd_detail.receive(raw, metadata)
        self.refresh_preview('ccd')
        self.toupcam.acquisition_fps = metadata.get('acquisition_fps')
        self.preview_fps_frames += 1
        raw_shape = metadata.get('raw_shape', image.shape)
        self.ccdPanelInfoLabel.setText(f'采集 {raw_shape[1]} × {raw_shape[0]}')
        self.ccdPanelInfoLabel.setToolTip(f'变换后图像：{image.shape[1]} × {image.shape[0]}')
        self.ccdFrameInfoLabel.setText(f'帧 {metadata["frame_id"]} · 曝光 {metadata["exposure"]:g} μs · 增益 {metadata["gain"]:g}')
        self.ccdStateLabel.setText('● 演示图像源' if self.demo else '● 正在取帧')
        self.update_actions()

    def refresh_preview(self, key, *_):
        if key == 'ccd' and self.camera_preview_packet is not None:
            if self.mainTabWidget.currentWidget() is not self.previewPage:
                return
            array, metadata = self.camera_preview_packet
            if self.ccdZoomComboBox.currentIndex() == 0:
                # Only the overview is downsampled; captures and detail retain raw pixels.
                step = max(1, max(array.shape[:2]) // max(1, self.ccdPreviewLabel.width()*2))
                array = np.ascontiguousarray(array[::step, ::step])
            self.images['ccd'] = camera_qimage(array)
        if key in self.images:
            image = self.images[key]
            if key in DISPLAY_KEYS and getattr(self, key + 'PreviewModeComboBox').currentIndex() == 0:
                image = self.source_images.get(key, image)
            set_preview(getattr(self, key + 'PreviewLabel'), image,
                        getattr(self, key + 'ZoomComboBox').currentIndex() == 0)

    def refresh_all_previews(self):
        for key in self.images:
            self.refresh_preview(key)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'resize_timer'):
            self.resize_timer.start(50)

    def update_actions(self, *_):
        transferring = self.device_task is not None
        busy = self.connected or self.camera is not None or self.disconnecting or transferring or bool(self.devices)
        running = any(timer.isActive() for timer in self.timers.values())
        waiting = self.connected and self.settings and self.device_enabled('ccd', self.settings) and (not self.camera_ready or not self.received_frames)
        self.configurationScrollContents.setEnabled(not busy)
        self.loadConfigurationButton.setEnabled(not busy)
        self.enumerateDevicesButton.setEnabled(not busy)
        self.validateConfigurationButton.setEnabled(not busy)
        self.connectDevicesButton.setEnabled(not self.disconnecting)
        self.connectDevicesButton.setText('取消连接 / 上传' if transferring else '断开设备' if busy else '连接设备')
        self.connectionStateLabel.setText(('● 演示模式 · ' if self.demo else '● ') +
            ('断开中' if self.disconnecting else '连接 / 上传中' if transferring else '连接中' if waiting else '设备已连接' if self.connected else '设备未连接'))
        sync_capture = self.synchronizeDmdCcdCheckBox.isChecked()
        slm_source = self.slm_capture_source()
        slm_running = self.slm_capture_running()
        network = self.network_dmd()
        network_ready = network is None or network.ready
        paired = self.synchronizeSlmsCheckBox.isChecked() and all(self.device_enabled(k) for k in ('slm1', 'slm2'))
        pair_ready = bool(self.paths['slm1'] and self.paths['slm2']) and (
            [p.stem for p in self.paths['slm1']] == [p.stem for p in self.paths['slm2']])
        playable = any(self.paths[k] and (k == 'dmd' or not paired or pair_ready) for k in self.devices)
        source_ready = (not slm_source or bool(self.paths[slm_source])) and (not sync_capture or bool(self.paths['dmd']))
        self.startAllButton.setEnabled(self.connected and playable and source_ready and not (waiting and (sync_capture or slm_source)) and not self.disconnecting and not transferring and network_ready)
        self.stopAllButton.setEnabled(running or self.recording or self.recording_pending or transferring or (network is not None and network.ready))
        self.synchronizeSlmsCheckBox.setEnabled(not busy and not running)
        self.synchronizeDmdCcdCheckBox.setEnabled(
            self.device_enabled('dmd') and self.device_enabled('ccd')
            and not self.timers['dmd'].isActive() and not slm_running and not waiting and not self.disconnecting and not transferring)
        self.slmCaptureSourceComboBox.setEnabled(not running and not waiting and not self.disconnecting and not transferring)
        self.slmCaptureSettleSpinBox.setEnabled(bool(slm_source) and not running and not self.disconnecting and not transferring)
        for key in DISPLAY_KEYS:
            usable = bool(self.paths[key]) and self.device_enabled(key) and not self.disconnecting and not transferring
            if key in ('slm1', 'slm2') and paired:
                usable = usable and pair_ready
            if busy:
                capture_member = key == slm_source or (slm_source and key in ('slm1', 'slm2') and self.synchronizeSlmsCheckBox.isChecked())
                usable = usable and key in self.devices and not (((key == 'dmd' and sync_capture) or capture_member) and waiting)
            if key == 'dmd':
                usable = usable and network_ready
            for suffix in ('PlayButton', 'PreviousButton', 'NextButton', 'FrameSpinBox'):
                getattr(self, key + suffix).setEnabled(usable)
            getattr(self, key + 'PauseButton').setEnabled((running or (key == 'dmd' and network is not None and network_ready)) and usable)
        for key in ('slm1', 'slm2'):
            getattr(self, key + 'ApplyOffsetButton').setEnabled(bool(self.paths[key]) and key not in self.inputs.tests and not slm_running and not self.disconnecting and not transferring)
        self.ccdApplyParametersButton.setEnabled(self.camera_ready and not slm_running and not self.disconnecting and not transferring)
        self.ccdCaptureButton.setEnabled(self.camera_ready and self.received_frames > 0 and not self.disconnecting and not transferring)
        self.ccdRecordButton.setEnabled(self.camera_ready and self.received_frames > 0 and not self.recording_pending and not self.disconnecting and not transferring)
        self.dmdIntervalSpinBox.setEnabled(not self.timers['dmd'].isActive())
        can_upload = network is not None and (network.demo or (network.controller is not None and network.controller.connected))
        self.dmdUploadButton.setEnabled(can_upload and not transferring and not self.disconnecting and not running)
        for key in (*DISPLAY_KEYS, 'ccd'):
            for suffix in TRANSFORM_SUFFIXES:
                getattr(self, key + suffix).setEnabled(self.transform_editable(key))
        calibrating = self.calibration.active
        self.saveConfigurationButton.setEnabled(not calibrating)
        self.autoCalibrateButton.setText('标定进度…' if calibrating else '自动标定 / 光路对准')
        self.previewScrollContents.setEnabled(not calibrating)
        if calibrating:
            self.connectDevicesButton.setEnabled(self.connected or self.camera is not None)
            self.configurationScrollContents.setEnabled(False)
            for control in (self.startAllButton, self.synchronizeSlmsCheckBox,
                            self.slmCaptureSourceComboBox, self.slmCaptureSettleSpinBox,
                            self.synchronizeDmdCcdCheckBox, self.loadConfigurationButton,
                            self.enumerateDevicesButton, self.validateConfigurationButton):
                control.setEnabled(False)
            self.stopAllButton.setEnabled(True)
        self.inputs.update_actions()

    def closeEvent(self, event):
        if self.calibration.active:
            self.closing = True
            self.calibration.disconnect_after = True
            self.calibration.cancel()
            if self.calibration.task is not None:
                event.ignore()
                return
        if self.device_task is not None:
            self.closing = True
            self.device_task.requestInterruption()
            event.ignore()
            return
        if self.camera is not None:
            self.closing = True
            self.disconnect_devices()
            event.ignore()
            return
        self.disconnect_devices()
        if self.devices:
            self.closing = False
            event.ignore()
            return
        self.poll_timer.stop()
        self.inputs.close()
        self.preview_fps_timer.stop()
        if self.demo_directory is not None:
            self.demo_directory.cleanup()
            self.demo_directory = None
        event.accept()
