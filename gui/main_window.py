"""MainWindow: configuration, previews, device lifecycle and sequence playback."""
from __future__ import annotations

from functools import partial
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
from ..utils.paths import TOOLS_ROOT
from .configuration import (read_values, apply_values, read_configuration,
                            write_configuration, input_path, pil_qimage,
                            camera_qimage, set_preview)
from .displays import DisplayAdapter
from .camera_worker import CameraWorker, CaptureRequest

DISPLAY_KEYS = ('dmd', 'slm1', 'slm2')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, demo=False, config_path=None, display_factory=DisplayAdapter,
                 camera_factory=CameraWorker):
        super().__init__()
        uic.loadUi(str(TOOLS_ROOT / 'ui' / 'MainWindow.ui'), self)
        self.demo = demo
        self.display_factory, self.camera_factory = display_factory, camera_factory
        self.devices = {}
        self.paths = {key: [] for key in DISPLAY_KEYS}
        self.indices = dict.fromkeys(DISPLAY_KEYS, 0)
        self.images = {}
        self.source_images = {}
        self.camera = None
        self.camera_ready = self.connected = self.disconnecting = self.closing = False
        self.recording = self.recording_pending = False
        self.received_frames = self.saved_frames = 0
        self.last_frame_time = None
        self.settings = None
        self.demo_directory = None
        self.timers = {}
        for key in DISPLAY_KEYS:
            timer = QtCore.QTimer(self)
            timer.setTimerType(QtCore.Qt.PreciseTimer)
            timer.timeout.connect(partial(self.guard, partial(self.advance, key)))
            self.timers[key] = timer
            getattr(self, key + 'PlayButton').clicked.connect(partial(self.guard, partial(self.play, key)))
            getattr(self, key + 'PauseButton').clicked.connect(partial(self.pause, key))
            getattr(self, key + 'NextButton').clicked.connect(partial(self.guard, partial(self.step, key, 1)))
            getattr(self, key + 'PreviousButton').clicked.connect(partial(self.guard, partial(self.step, key, -1)))
            getattr(self, key + 'FrameSpinBox').valueChanged.connect(partial(self.select_frame, key))
            getattr(self, key + 'InputPathEdit').editingFinished.connect(partial(self.reload_safely, key))
        for key in (*DISPLAY_KEYS, 'ccd'):
            getattr(self, key + 'ZoomComboBox').currentIndexChanged.connect(partial(self.refresh_preview, key))
            getattr(self, key + 'EnabledCheckBox').toggled.connect(self.update_actions)
        for key in DISPLAY_KEYS:
            getattr(self, key + 'PreviewModeComboBox').currentIndexChanged.connect(partial(self.refresh_preview, key))
        for name in ('dmdInputPathEdit', 'slm1InputPathEdit', 'slm2InputPathEdit',
                     'slm1SdkPathEdit', 'slm2SdkPathEdit', 'ccdDllPathEdit', 'outputDirectoryEdit'):
            getattr(self, name + 'BrowseButton').clicked.connect(partial(self.browse, name))
        self.loadConfigurationButton.clicked.connect(self.load_dialog)
        self.saveConfigurationButton.clicked.connect(self.save_dialog)
        self.validateConfigurationButton.clicked.connect(partial(self.guard, self.check_parameters))
        self.enumerateDevicesButton.clicked.connect(partial(self.guard, self.enumerate_devices))
        self.connectDevicesButton.clicked.connect(partial(self.guard, self.toggle_connection))
        self.startAllButton.clicked.connect(partial(self.guard, self.start_all))
        self.stopAllButton.clicked.connect(self.stop_all)
        for key in ('slm1', 'slm2'):
            getattr(self, key + 'ApplyOffsetButton').clicked.connect(partial(self.guard, partial(self.apply_offset, key)))
        self.synchronizeDmdCcdCheckBox.toggled.connect(self.capture_sync_changed)
        self.ccdApplyParametersButton.clicked.connect(partial(self.guard, self.apply_camera_parameters))
        self.ccdCaptureButton.clicked.connect(partial(self.guard, self.capture_manual))
        self.ccdRecordButton.clicked.connect(partial(self.guard, self.toggle_recording))
        self.outputDirectoryEdit.textChanged.connect(lambda text: self.capturePathLabel.setText('保存目录：' + text))
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.poll_camera)
        self.poll_timer.start(33)
        self.resize_timer = QtCore.QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.refresh_all_previews)
        if demo:
            self.setup_demo()
        if config_path:
            apply_values(self, read_configuration(config_path))
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
        self.outputDirectoryEdit.setText('output/gui_demo')
        self.setWindowTitle('MainWindow · 演示模式（未连接真实硬件）')
        self.subtitleLabel.setText('演示模式 · DMD / SLM 使用示例图案，CCD 为合成画面')

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
        elif name == 'ccdDllPathEdit':
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择相机 DLL', current, 'DLL (*.dll)')
        else:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择目录', current)
        if path:
            getattr(self, name).setText(path)
            if name.endswith('InputPathEdit'):
                self.reload_safely(name.split('Input')[0])

    def load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '载入配置', '', 'JSON (*.json)')
        if path:
            self.guard(partial(self.load_config, Path(path)))

    def load_config(self, path):
        if self.connected or self.camera is not None:
            raise ValueError('请先断开设备，再载入配置。')
        self.stop_all()
        apply_values(self, read_configuration(path))
        for key in DISPLAY_KEYS:
            self.reload_safely(key)
        self.statusBar().showMessage(f'已载入配置：{path}')

    def save_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '保存配置', 'optical_config.json', 'JSON (*.json)')
        if path:
            self.guard(lambda: write_configuration(Path(path), read_values(self)))

    def reload_safely(self, key):
        if self.connected:
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
        path = input_path(getattr(self, key + 'InputPathEdit').text())
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
        enabled = [key for key in (*DISPLAY_KEYS, 'ccd') if values[key + 'EnabledCheckBox']]
        if not enabled:
            raise ValueError('请至少启用一个设备。')
        if values['synchronizeDmdCcdCheckBox']:
            if 'dmd' not in enabled or 'ccd' not in enabled:
                raise ValueError('DMD 同步采集需要同时启用 DMD 和 CCD。')
            if values['dmdSettleSpinBox'] >= values['dmdIntervalSpinBox']:
                raise ValueError('DMD 采集前稳定时间必须小于播放间隔。')
        for key in DISPLAY_KEYS:
            if key in enabled:
                self.load_sequence(key)
        if values['synchronizeSlmsCheckBox'] and 'slm1' in enabled and 'slm2' in enabled:
            self.validate_pairs()
        input_path(values['outputDirectoryEdit'])
        if not self.demo:
            for key, field, suffix in [('slm1', 'slm1SdkPathEdit', 'SecondDll.dll'),
                                       ('slm2', 'slm2SdkPathEdit', 'HDSLMFunc.dll'),
                                       ('ccd', 'ccdDllPathEdit', '')]:
                if key in enabled:
                    sdk_path = input_path(values[field])
                    if suffix:
                        sdk_path /= suffix
                    if not sdk_path.is_file():
                        raise FileNotFoundError(f'{key.upper()} SDK 不存在：{sdk_path}')
        return values

    def check_parameters(self):
        self.validate()
        self.statusBar().showMessage('文件、序列配对与参数检查通过。')
        QtWidgets.QMessageBox.information(self, '参数检查', '参数检查通过，可连接设备。')

    def validate_pairs(self):
        if [p.stem for p in self.paths['slm1']] != [p.stem for p in self.paths['slm2']]:
            raise ValueError('同步播放要求 SLM1 与 SLM2 的文件主干名称及顺序一致。')

    def enumerate_devices(self):
        if self.demo:
            QtWidgets.QMessageBox.information(self, '演示设备', 'DMD、SLM1、SLM2：虚拟显示区域\nCCD：合成图像源\n未访问真实 SDK。')
            return
        lines = []
        try:
            for monitor in zkwx.enumerate_monitors():
                lines.append(f'Windows monitor {monitor.index}: {monitor.width}×{monitor.height} @ ({monitor.left}, {monitor.top})')
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
                lines.append(f'{description}：{len(devices)} 个设备')
                for index, device in enumerate(devices):
                    if description == 'CCD':
                        lines.append(f'  {index}: {dvp.decode_sdk_string(device.FriendlyName)} / {dvp.decode_sdk_string(device.SerialNumber)}')
                    else:
                        lines.append(f'  SDK ID {device.index}: {device.name} / {device.width}×{device.height} @ ({device.left}, {device.top})')
            except Exception as exc:
                lines.append(f'{description} 枚举失败：{exc}')
            finally:
                if api is not None:
                    api.close()
        QtWidgets.QMessageBox.information(self, '设备枚举', '\n'.join(lines))

    def toggle_connection(self):
        if self.connected or self.camera is not None:
            self.disconnect_devices()
        else:
            self.connect_devices()

    def connect_devices(self):
        self.stop_all()
        settings = self.validate()
        devices = {}
        try:
            for key in DISPLAY_KEYS:
                if settings[key + 'EnabledCheckBox']:
                    devices[key] = self.display_factory(key, settings, self.paths[key], self.demo)
            geometries = [device.geometry for device in devices.values()]
            if len(set(geometries)) != len(geometries):
                raise ValueError('DMD、SLM1、SLM2 不能使用同一个显示区域，请修改设备编号。')
            for device in devices.values():
                device.open()
        except Exception:
            for device in devices.values():
                try:
                    device.close()
                except Exception:
                    pass
            raise
        self.settings, self.devices = settings, devices
        self.connected = True
        self.saved_frames = self.received_frames = 0
        self.last_frame_time = None
        for key, device in devices.items():
            if device.output is not None:
                device.output.closed.connect(self.disconnect_devices)
            getattr(self, key + 'StateLabel').setText('● 演示设备' if self.demo else '● 已连接')
        try:
            if settings['ccdEnabledCheckBox']:
                self.camera = self.camera_factory(settings, self.demo, parent=self)
                self.camera.ready.connect(self.on_camera_ready)
                self.camera.parametersChanged.connect(self.on_camera_parameters)
                self.camera.failed.connect(self.on_camera_error)
                self.camera.warning.connect(self.on_camera_warning)
                self.camera.captured.connect(self.on_capture)
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
        self.on_camera_parameters(parameters)
        self.ccdStateLabel.setText('● 演示图像源' if self.demo else '● 已连接')
        self.update_actions()

    def on_camera_parameters(self, parameters):
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
            getattr(self, key + 'StateLabel').setText('● 未连接')
        self.connected = self.camera_ready = False
        if self.camera is not None:
            self.camera.requestInterruption()
        else:
            self.disconnecting = False
        self.update_actions()

    def on_camera_finished(self):
        worker, self.camera = self.camera, None
        if worker is not None:
            worker.deleteLater()
        self.camera_ready = self.recording = self.recording_pending = False
        self.disconnecting = False
        self.ccdStateLabel.setText('● 未连接')
        self.update_actions()
        if self.closing:
            self.close()

    def group_for(self, key):
        if key in ('slm1', 'slm2') and self.synchronizeSlmsCheckBox.isChecked() and self.slm1EnabledCheckBox.isChecked() and self.slm2EnabledCheckBox.isChecked():
            self.validate_pairs()
            return ('slm1', 'slm2')
        return (key,)

    def render_frame(self, key, index):
        path = self.paths[key][index]
        source = pil_qimage(path)
        display = self.devices[key].display if key in self.devices else zkwx.Monitor(0, 0, 0, 1920, 1080, False)
        offset_x = getattr(self, key + 'OffsetXSpinBox').value() if key != 'dmd' else 0
        offset_y = getattr(self, key + 'OffsetYSpinBox').value() if key != 'dmd' else 0
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
        background = self.slm2BackgroundSpinBox.value() if key == 'slm2' else 0
        canvas.fill(QtGui.QColor(background, background, background))
        painter = QtGui.QPainter(canvas)
        painter.drawImage(origin[0], origin[1], source)
        painter.end()
        self.source_images[key] = source
        self.images[key] = canvas
        self.refresh_preview(key)
        getattr(self, key + 'FrameInfoLabel').setText('文件：' + path.name)
        getattr(self, key + 'FrameInfoLabel').setToolTip(str(path))
        getattr(self, key + 'PanelInfoLabel').setText(f'面板 {display.width} × {display.height}')
        getattr(self, key + 'ActiveAreaLabel').setText(f'有效区 {source.width()} × {source.height()}')
        getattr(self, key + 'ActiveAreaLabel').setToolTip(f'有效区左上角：({origin[0]}, {origin[1]}) 面板像素')

    def show_frame(self, key, index, capture=False):
        if key == 'dmd' and capture and self.synchronizeDmdCcdCheckBox.isChecked():
            self.require_capture_sync_ready()
        values = read_values(self)
        group = self.group_for(key)
        for target in group:
            if not self.paths[target]:
                raise ValueError(f'{target.upper()} 未载入图像。')
            path = self.paths[target][index]
            if target in self.devices:
                self.devices[target].show(path, values)
            self.indices[target] = index
            with QtCore.QSignalBlocker(getattr(self, target + 'FrameSpinBox')):
                getattr(self, target + 'FrameSpinBox').setValue(index + 1)
            self.render_frame(target, index)
        if key == 'dmd' and capture and self.synchronizeDmdCcdCheckBox.isChecked():
            self.queue_capture(self.paths[key][index].stem + '.png', automatic=True)

    def interval(self, key):
        return getattr(self, key + 'IntervalSpinBox').value()

    def play(self, key):
        group = self.group_for(key)
        master = group[0]
        if self.connected and any(target not in self.devices for target in group):
            raise ValueError('目标显示设备未连接。')
        if not self.paths[master]:
            raise ValueError('请先载入图像。')
        if master == 'dmd' and self.synchronizeDmdCcdCheckBox.isChecked():
            self.require_capture_sync_ready()
        self.pause(master)
        index = self.indices[master]
        if index == len(self.paths[master]) - 1:
            index = 0
        self.show_frame(master, index, capture=True)
        self.timers[master].start(max(1, round(self.interval(master) * 1000)))
        self.statusBar().showMessage(('设备播放：' if self.connected else '本地预览：') + ', '.join(group))
        self.update_actions()

    def advance(self, key):
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
        if key == 'dmd' and self.camera is not None:
            self.camera.submit('cancel_captures')
        if hasattr(self, 'poll_timer'):
            self.update_actions()

    def step(self, key, delta):
        self.pause(key)
        index = min(max(self.indices[key] + delta, 0), len(self.paths[key]) - 1)
        self.show_frame(key, index)

    def select_frame(self, key, value):
        if not self.paths[key]:
            return
        self.pause(key)
        self.guard(partial(self.show_frame, key, value - 1))

    def start_all(self):
        if not self.connected:
            raise ValueError('请先连接设备。')
        if self.synchronizeDmdCcdCheckBox.isChecked():
            self.require_capture_sync_ready()
        started = set()
        # Send phase patterns before the first DMD exposure/capture window.
        for key in ('slm1', 'slm2', 'dmd'):
            if key not in self.devices:
                continue
            if key in started:
                continue
            self.play(key)
            started.update(self.group_for(key))

    def stop_all(self, *_):
        for timer in self.timers.values():
            timer.stop()
        if self.camera is not None:
            self.camera.submit('cancel_captures')
            self.camera.submit('record_stop')
        if hasattr(self, 'poll_timer'):
            self.update_actions()

    def apply_offset(self, key='slm2'):
        if self.paths[key]:
            # Validate the target canvas before sending either paired SLM frame.
            self.render_frame(key, self.indices[key])
            self.show_frame(key, self.indices[key])
            self.statusBar().showMessage(f'{key.upper()} 偏移已应用；切换“完整面板”可查看图案位置。')

    def require_capture_sync_ready(self):
        if not self.dmdEnabledCheckBox.isChecked() or not self.ccdEnabledCheckBox.isChecked():
            raise ValueError('DMD 同步采集需要同时启用 DMD 和 CCD。')
        if not self.connected or 'dmd' not in self.devices or not self.camera_ready or not self.received_frames:
            raise ValueError('DMD 同步采集需要先连接 DMD 和 CCD，并等待相机第一帧。')
        if self.dmdSettleSpinBox.value() >= self.interval('dmd'):
            raise ValueError('稳定时间必须小于 DMD 播放间隔。')

    def capture_sync_changed(self, checked):
        if not checked and self.camera is not None:
            self.camera.submit('cancel_auto_captures')
        self.update_actions()

    def apply_camera_parameters(self):
        if not self.camera_ready:
            raise ValueError('CCD 尚未连接。')
        self.camera.submit('parameters', self.ccdExposureSpinBox.value(), self.ccdGainSpinBox.value(),
                           self.ccdAutoExposureCheckBox.isChecked())

    def queue_capture(self, filename, automatic=False):
        if not self.camera_ready:
            raise ValueError('CCD 尚未就绪。')
        if automatic:
            if not self.synchronizeDmdCcdCheckBox.isChecked():
                return
            self.require_capture_sync_ready()
        now = time.monotonic()
        settings = self.settings
        after = now + self.dmdSettleSpinBox.value() if automatic else now
        deadline = now + self.interval('dmd') if automatic else now + max(5, self.ccdTimeoutSpinBox.value() / 1000)
        metadata = {'experiment': settings['sessionNameEdit'], 'demo': self.demo, 'automatic': automatic,
                    'display_files': {k: str(self.paths[k][self.indices[k]]) for k in self.devices}}
        request = CaptureRequest(input_path(settings['outputDirectoryEdit']) / filename, after,
                                 settings['overwriteCheckBox'], settings['saveManifestCheckBox'], metadata, deadline)
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

    def poll_camera(self):
        if self.camera is None or self.disconnecting:
            return
        packet = self.camera.take_latest()
        if packet is None:
            return
        image, metadata = packet
        now = metadata['received_at_monotonic']
        fps = 0 if self.last_frame_time is None else 1 / max(0.0001, now - self.last_frame_time)
        self.last_frame_time = now
        self.received_frames += 1
        self.images['ccd'] = camera_qimage(image)
        self.refresh_preview('ccd')
        self.ccdPanelInfoLabel.setText(f'采集 {image.shape[1]} × {image.shape[0]}')
        self.ccdActiveAreaLabel.setText(f'预览 {fps:.1f} FPS')
        self.ccdFrameInfoLabel.setText(f'帧 {metadata["frame_id"]} · 曝光 {metadata["exposure"]:g} μs · 增益 {metadata["gain"]:g}')
        self.ccdStateLabel.setText('● 演示图像源' if self.demo else '● 正在取帧')
        self.update_actions()

    def refresh_preview(self, key, *_):
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
        busy = self.connected or self.camera is not None or self.disconnecting
        running = any(timer.isActive() for timer in self.timers.values())
        waiting = self.connected and self.settings and self.settings['ccdEnabledCheckBox'] and (not self.camera_ready or not self.received_frames)
        self.configurationScrollContents.setEnabled(not busy)
        self.loadConfigurationButton.setEnabled(not busy)
        self.enumerateDevicesButton.setEnabled(not busy)
        self.validateConfigurationButton.setEnabled(not busy)
        self.connectDevicesButton.setEnabled(not self.disconnecting)
        self.connectDevicesButton.setText('断开设备' if busy else '连接设备')
        self.connectionStateLabel.setText(('● 演示模式 · ' if self.demo else '● ') +
            ('断开中' if self.disconnecting else '连接中' if waiting else '设备已连接' if self.connected else '设备未连接'))
        sync_capture = self.synchronizeDmdCcdCheckBox.isChecked()
        self.startAllButton.setEnabled(self.connected and not (waiting and sync_capture) and not self.disconnecting)
        self.stopAllButton.setEnabled(running or self.recording or self.recording_pending)
        self.synchronizeSlmsCheckBox.setEnabled(not busy and not running)
        self.synchronizeDmdCcdCheckBox.setEnabled(
            self.dmdEnabledCheckBox.isChecked() and self.ccdEnabledCheckBox.isChecked()
            and not self.timers['dmd'].isActive() and not waiting and not self.disconnecting)
        for key in DISPLAY_KEYS:
            usable = bool(self.paths[key]) and getattr(self, key + 'EnabledCheckBox').isChecked() and not self.disconnecting
            if busy:
                usable = usable and key in self.devices and not (key == 'dmd' and sync_capture and waiting)
            for suffix in ('PlayButton', 'PreviousButton', 'NextButton', 'FrameSpinBox'):
                getattr(self, key + suffix).setEnabled(usable)
            getattr(self, key + 'PauseButton').setEnabled(running and usable)
        for key in ('slm1', 'slm2'):
            getattr(self, key + 'ApplyOffsetButton').setEnabled(bool(self.paths[key]) and not self.disconnecting)
        self.ccdApplyParametersButton.setEnabled(self.camera_ready and not self.disconnecting)
        self.ccdCaptureButton.setEnabled(self.camera_ready and self.received_frames > 0 and not self.disconnecting)
        self.ccdRecordButton.setEnabled(self.camera_ready and self.received_frames > 0 and not self.recording_pending and not self.disconnecting)
        self.dmdIntervalSpinBox.setEnabled(not self.timers['dmd'].isActive())

    def closeEvent(self, event):
        if self.camera is not None:
            self.closing = True
            self.disconnect_devices()
            event.ignore()
            return
        self.disconnect_devices()
        self.poll_timer.stop()
        if self.demo_directory is not None:
            self.demo_directory.cleanup()
            self.demo_directory = None
        event.accept()
