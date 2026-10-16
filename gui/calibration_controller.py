"""Qt-thread hardware bridge and restoration for automatic calibration."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import tempfile
import threading
import time
from uuid import uuid4

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtWidgets

from ..calibration.acquisition import FrameBatch
from ..calibration.config import CalibrationConfig, write_json
from ..calibration.bundle import load_bundle
from ..calibration.development import check_training, train_bundle
from ..calibration.patterns import save_gray
from ..calibration.runner import CalibrationRunner
from ..utils.image_transforms import ImageTransform
from ..utils.camera_settings import camera_roi
from .configuration import pil_image_qimage, read_values, write_configuration


@dataclass
class Request:
    action: str
    payload: object
    event: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: str = ''


class CalibrationTask(QtCore.QThread):
    requested = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, config, root, settings, training_only, parent, bundle_path=None):
        super().__init__(parent)
        self.config, self.root, self.settings = config, root, settings
        self.training_only = training_only
        self.bundle_path = bundle_path
        self.result = None
        self.error = ''

    def io(self, action, payload):
        request = Request(action, payload)
        self.requested.emit(request)
        deadline = time.monotonic() + self.config.capture_timeout + 10
        while not request.event.wait(.05):
            if self.isInterruptionRequested() and action != 'restore_parameters':
                raise InterruptedError('自动标定已取消。')
            if time.monotonic() > deadline:
                raise TimeoutError(f'硬件操作等待超时：{action}')
        if request.error:
            raise RuntimeError(request.error)
        if action == 'capture':
            batch = request.result
            while not batch.event.wait(.05):
                if self.isInterruptionRequested() or time.monotonic() > batch.deadline:
                    batch.cancelled = True
                    raise InterruptedError('标定采集已取消。') if self.isInterruptionRequested() else TimeoutError('标定取帧超时。')
            image, metadata = batch.result()
            return image, metadata, batch.frames
        return request.result

    def run(self):
        try:
            if self.training_only:
                self.result = train_bundle(self.config, self.root, self.progress.emit, self.isInterruptionRequested)
            else:
                parameters = self.io('parameters', (
                    self.settings['ccdExposureSpinBox'], self.settings['ccdGainSpinBox'], False))
                if parameters.get('auto'):
                    raise RuntimeError('相机未进入固定曝光模式，不能开始标定。')
                runner = CalibrationRunner(self.config, self.root, self.settings, self.io,
                    self.progress.emit, self.isInterruptionRequested, bundle_path=self.bundle_path)
                self.result = runner.run()
        except Exception as exc:
            self.error = str(exc)
            if self.root.exists():
                write_json(self.root / 'task_status.json', {'status': 'cancelled' if self.isInterruptionRequested() else 'failed',
                                                           'reason': self.error})
        finally:
            if not self.training_only:
                try:
                    self.io('restore_parameters', (self.settings['ccdExposureSpinBox'],
                        self.settings['ccdGainSpinBox'], self.settings['ccdAutoExposureCheckBox']))
                except Exception as exc:
                    self.error += '；相机参数恢复未确认：' + str(exc)


class CalibrationController(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.config = CalibrationConfig.defaults()
        self.output = Path.cwd() / 'output' / 'calibration'
        self.dialog = self.task = None
        self.pending = None
        self.disconnect_after = False
        self.directory = None
        self.original = None
        self.bundle_path = None

    @property
    def active(self):
        return self.task is not None or self.pending is not None

    def open(self):
        from .calibration_dialog import CalibrationDialog
        if self.dialog is None:
            self.dialog = CalibrationDialog(self)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def start(self, config, output, training_only, bundle_path=None):
        if self.active:
            raise ValueError('标定任务正在运行。')
        window = self.window
        config.validate()
        # Imported bundles do not import torch or touch the training model.
        # Validate before changing illumination or connecting any hardware.
        if not training_only and bundle_path is not None:
            bundle = load_bundle(bundle_path)
            bundle.check_config(config)
            if any(item['simulation_score'] < config.minimum_score for item in bundle.manifest['patterns']):
                raise ValueError('标定包的模拟分数未达到当前阈值，请在开发环境改善训练后重新导入。')
        else:
            check_training(config)
        self.config, self.output = config, output
        if not training_only:
            if window.demo:
                raise ValueError('demo 相机是棋盘格合成画面，不能用于实物标定；请以真实设备模式运行。')
            if window.f4321EnabledCheckBox.isChecked():
                raise ValueError('当前标定流程适用于普通 DMD，请切换到显示器输出 DMD。')
            if window.device_task is not None or window.disconnecting:
                raise ValueError('请等待设备操作完成。')
            for key in ('dmd', 'slm1', 'slm2', 'ccd'):
                if not window.device_enabled(key):
                    raise ValueError('自动标定需要同时启用普通 DMD、SLM1、SLM2 和相机。')
            if camera_roi(read_values(window)):
                raise ValueError('首轮标定需要完整相机画面，请关闭 ROI 并重新连接。')
            if not window.connected:
                # No demo checkerboards or user's phase files are needed to connect.
                references = output / ('references_' + uuid4().hex[:8])
                for key in ('dmd', 'slm1', 'slm2'):
                    shape = (config.dmd_height, config.dmd_width) if key == 'dmd' else config.slm_shape(key)
                    path = save_gray(references / key / 'reference.png', np.zeros(shape, np.uint8))
                    getattr(window, key + 'InputPathEdit').setText(str(path))
                self.dialog.set_running(True)
                try:
                    window.connect_devices()
                    self.pending = (config, output, training_only, bundle_path)
                except Exception:
                    self.pending = None
                    self.dialog.set_running(False)
                    raise
                window.update_actions()
                return
            if not window.camera_ready:
                raise ValueError('相机尚未就绪。')
            if set(('dmd', 'slm1', 'slm2')) - set(window.devices):
                raise ValueError('三路调制器必须连接成功。')
        self._launch(config, output, training_only, bundle_path)

    def camera_ready(self):
        if self.pending is not None:
            args, self.pending = self.pending, None
            try:
                self._launch(*args)
            except Exception as exc:
                self.dialog.set_running(False)
                self.progress(str(exc))
                QtWidgets.QMessageBox.warning(self.window, '标定参数不匹配', str(exc))

    def _launch(self, config, output, training_only, bundle_path=None):
        window = self.window
        if not training_only:
            parameters = getattr(window, 'camera_parameters', {})
            geometry = parameters.get('geometry')
            if geometry:
                if (geometry['resolution_index'] != 0 or geometry['hardware_binning'] != 1
                        or geometry['software_binning'] != 1):
                    raise ValueError('ToupTek 自动标定暂按原生采样验证：请选择最大分辨率、关闭硬件与软件 Binning。其他模式可正常预览与保存。')
                pitch = parameters['camera'].get('pixel_size_um', [])
                if len(pitch) == 2 and any(abs(config.camera_pitch_um - value) > .001 for value in pitch):
                    raise ValueError(f'标定相机像元为 {config.camera_pitch_um:g} μm，SDK 返回 {pitch} μm；请核对标定参数和相位包。')
        window.stop_all()
        self.original = read_values(window)
        settings = dict(self.original)
        if not training_only:
            settings['slm2BackgroundSpinBox'] = 0
        self.directory = tempfile.TemporaryDirectory(prefix='optical_calibration_')
        root = output / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
        self.task = CalibrationTask(config, root, settings, training_only, self, bundle_path=bundle_path)
        self.task.requested.connect(self.dispatch)
        self.task.progress.connect(self.progress)
        self.task.finished.connect(self.finished)
        self.dialog.set_running(True)
        self.progress('开始。结果目录：' + str(root))
        self.task.start()
        window.update_actions()

    def progress(self, message):
        self.window.statusBar().showMessage(message)
        if self.dialog is not None:
            self.dialog.log.appendPlainText(message)

    def dispatch(self, request):
        window = self.window
        try:
            if self.task is None or (self.task.isInterruptionRequested() and request.action != 'restore_parameters'):
                raise InterruptedError('标定已取消。')
            if request.action == 'display':
                paths, values = request.payload
                prepared = {}
                for key, path in paths.items():
                    device = window.devices[key]
                    transform = ImageTransform.from_settings(values, key)
                    with Image.open(path) as source:
                        with transform.apply_pil(source) as oriented:
                            width, height = device.display.width, device.display.height
                            x = (width - oriented.width) // 2 + (values[key + 'OffsetXSpinBox'] if key != 'dmd' else 0)
                            y = (height - oriented.height) // 2 + (values[key + 'OffsetYSpinBox'] if key != 'dmd' else 0)
                            if x < 0 or y < 0 or x + oriented.width > width or y + oriented.height > height:
                                raise ValueError(f'{key} 标定图案或搜索偏移超出面板，未发送图案。')
                            canvas = Image.new('L', (width, height), 0)
                            canvas.paste(oriented, (x, y))
                            prepared[key] = (canvas, pil_image_qimage(oriented))
                for key, (canvas, preview) in prepared.items():
                    file = Path(self.directory.name) / (uuid4().hex + '.png')
                    canvas.save(file)
                    full_preview = pil_image_qimage(canvas)
                    canvas.close()
                    settings = dict(values)
                    settings.update({key + 'RotationSpinBox': 0,
                        key + 'HorizontalFlipCheckBox': False, key + 'VerticalFlipCheckBox': False,
                        key + 'OffsetXSpinBox': 0, key + 'OffsetYSpinBox': 0,
                        'dmdPixelModeComboBox': '保持原图', 'slm2BackgroundSpinBox': 0})
                    window.devices[key].show(file, settings, calibration=True)
                    window.source_images[key], window.images[key] = preview, full_preview
                    window.refresh_preview(key)
                    getattr(window, key + 'FrameInfoLabel').setText('自动标定：' + Path(paths[key]).name)
                # SDK may retain the most recent path; keep recent files and
                # tolerate locked files until disconnect / final cleanup.
                files = sorted(Path(self.directory.name).glob('*.png'), key=lambda p: p.stat().st_mtime)
                for old in files[:-9]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            elif request.action == 'capture':
                config = request.payload
                now = time.monotonic()
                batch = FrameBatch(now + config.settle_seconds, now + config.capture_timeout + config.settle_seconds,
                                   config.average_frames, config.discard_frames)
                window.camera.submit('calibration_capture', batch)
                request.result = batch
            elif request.action in ('parameters', 'restore_parameters'):
                if window.camera is None or not window.camera.isRunning():
                    raise RuntimeError('相机已停止。')
                window.camera.submit('calibration_parameters', request, *request.payload)
                return  # The acquisition thread acknowledges actual SDK values.
            elif request.action == 'panels':
                request.result = {key: (device.display.width, device.display.height)
                                  for key, device in window.devices.items()}
            else:
                raise ValueError('未知标定硬件操作。')
        except Exception as exc:
            request.error = str(exc)
        request.event.set()

    def cancel(self):
        self.pending = None
        if self.task is not None:
            self.task.requestInterruption()
        elif self.dialog:
            self.dialog.set_running(False)
        if self.window.camera is not None:
            self.window.camera.submit('cancel_calibration')

    def restore(self, values):
        window = self.window
        prepared = {key: window.prepare_frame(key, window.indices[key], values)
                    for key in window.devices if window.paths[key] and key not in window.inputs.tests}
        # Validate all existing input frames as well as the currently displayed one.
        for key in prepared:
            for index in range(len(window.paths[key])):
                window.prepare_frame(key, index, values)
        for key, frame in prepared.items():
            window.devices[key].show(window.paths[key][window.indices[key]], values)
            window.publish_frame(key, frame)
        for key in window.devices:
            if key not in prepared:
                window.inputs.restore_display(key, values)
        for key in ('dmd', 'slm1', 'slm2'):
            for suffix in ('RotationSpinBox', 'HorizontalFlipCheckBox', 'VerticalFlipCheckBox',
                           'OffsetXSpinBox', 'OffsetYSpinBox'):
                control = getattr(window, key + suffix, None)
                if control is None:
                    continue
                with QtCore.QSignalBlocker(control):
                    if isinstance(control, QtWidgets.QCheckBox):
                        control.setChecked(values[key + suffix])
                    else:
                        control.setValue(values[key + suffix])
        window.settings = dict(values)
        with QtCore.QSignalBlocker(window.slm2BackgroundSpinBox):
            window.slm2BackgroundSpinBox.setValue(values['slm2BackgroundSpinBox'])
        window.sync_transform_values()

    def finished(self):
        task = self.task
        if task is None:
            return
        success = bool(task.result) and not task.error and not task.isInterruptionRequested()
        try:
            if not task.training_only:
                values = task.result['applied_settings'] if success else self.original
                try:
                    self.restore(values)
                except Exception as exc:
                    success = False
                    task.error = '应用结果失败：' + str(exc)
                    self.restore(self.original)
                if task.result and not success:
                    task.result.update(status='not_applied', reason=task.error or '已取消')
                    write_json(task.root / 'calibration.json', task.result)
                    write_json(task.root / 'report.json', task.result)
        except Exception as exc:
            success = False
            task.error += '；恢复设备状态失败：' + str(exc)
        finally:
            if task.result and not task.training_only:
                task.result['previous_gui_settings'] = self.original
                if not success:
                    task.result.update(status='not_applied', reason=task.error or '已取消')
                try:
                    write_json(task.root / 'calibration.json', task.result)
                    write_json(task.root / 'report.json', task.result)
                    if success:
                        saved = read_values(self.window)
                        for key in ('ccdExposureSpinBox', 'ccdGainSpinBox', 'ccdAutoExposureCheckBox'):
                            saved[key] = self.original[key]
                        write_configuration(task.root / 'gui_configuration.json', saved)
                except OSError as exc:
                    success = False
                    task.error += '；保存应用结果失败：' + str(exc)
            self.task = None
            task.deleteLater()
            if self.directory is not None:
                try:
                    self.directory.cleanup()
                except OSError:
                    pass
                self.directory = None
            self.dialog.set_running(False)
            self.dialog.progress.setValue(1 if success else 0)
            self.window.update_actions()
        if success:
            self.progress(('相位图训练完成；尚未验证真实硬件。' if task.training_only else
                           '自动标定通过：参数已应用；相机视野外区域未验证。') + '\n' + str(task.root))
        else:
            self.progress('未完成：' + (task.error or '已取消') + '\n记录目录：' + str(task.root))
        if self.disconnect_after or self.window.closing:
            self.disconnect_after = False
            self.window.disconnect_devices()
            if self.window.closing and self.window.camera is None:
                self.window.close()
