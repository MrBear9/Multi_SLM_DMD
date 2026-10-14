"""Camera acquisition, recording and captures owned by one worker thread."""
from __future__ import annotations

import json
import queue
import threading
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore

from ..devices.dvp2_camera import sdk as dvp
from ..devices.dvp2_camera import controller as camera
from ..utils.image_transforms import ImageTransform
from ..utils.camera_settings import selected_camera, camera_value
from ..devices.toupcam_camera.controller import ToupCamera
from .configuration import input_path


class RealCamera:
    def __init__(self, settings):
        self.settings = settings
        self.api = self.session = None
        self.identity = {}

    def open(self):
        s = self.settings
        self.api = dvp.DvpApi(input_path(s['ccdDllPathEdit']))
        devices = self.api.list_devices()
        if s['ccdCameraSpinBox'] >= len(devices):
            raise ValueError('所选 CCD 编号不存在，请先枚举设备。')
        self.identity = camera.camera_identity(devices[s['ccdCameraSpinBox']])
        self.session = dvp.CameraSession(self.api, s['ccdCameraSpinBox'])
        self.session.open()
        handle = self.session.handle.value
        roi = [s[name] for name in ('ccdRoiXSpinBox', 'ccdRoiYSpinBox', 'ccdRoiWidthSpinBox', 'ccdRoiHeightSpinBox')]
        camera.configure_camera(self.api, handle, Namespace(
            load_config=None, save_config=None, target_format=s['ccdFormatComboBox'],
            auto_exposure=False, manual_exposure=False, exposure_us=None, gain=None,
            roi=roi if s['ccdRoiCheckBox'] else None, dialog=False))
        if not s['ccdRoiCheckBox']:
            dvp.check_status(self.api.dll.dvpSetRoiState(handle, False), 'disable ROI')
        self.session.start()
        return self.parameters()

    def parameters(self):
        handle = self.session.handle.value
        return {'exposure': camera.get_exposure(self.api, handle),
                'gain': camera.get_gain(self.api, handle),
                'auto': camera.get_ae_operation(self.api, handle) == dvp.AE_OP_CONTINUOUS,
                'camera': dict(self.identity)}

    def apply(self, exposure, gain, auto):
        handle = self.session.handle.value
        dvp.check_status(self.api.dll.dvpSetAeOperation(handle, dvp.AE_OP_CONTINUOUS if auto else dvp.AE_OP_OFF), 'set auto exposure')
        if not auto:
            dvp.check_status(self.api.dll.dvpSetExposure(handle, exposure), 'set exposure')
            dvp.check_status(self.api.dll.dvpSetAnalogGain(handle, gain), 'set gain')
        return self.parameters()

    def read(self):
        # Short SDK waits make stop/close responsive even with no incoming frames.
        frame, raw = self.session.get_frame(min(200, self.settings['ccdTimeoutSpinBox']))
        return camera.frame_to_image(frame, raw), {'frame_id': int(frame.uFrameID),
            'exposure': float(frame.fExposure), 'gain': float(frame.fAGain),
            'camera': dict(self.identity)}

    def close(self):
        try:
            if self.session is not None:
                self.session.close()
        finally:
            if self.api is not None:
                self.api.close()


class DemoCamera:
    """Explicit demo source; never used as an automatic hardware fallback."""
    def __init__(self, settings):
        self.settings = settings
        self.count = 0
        self.values = {'exposure': 10000.0, 'gain': 1.0, 'auto': False}

    def open(self):
        return dict(self.values)

    def apply(self, exposure, gain, auto):
        self.values = dict(exposure=exposure, gain=gain, auto=auto)
        return dict(self.values)

    def read(self):
        time.sleep(0.03)
        self.count += 1
        y, x = np.indices((480, 640))
        image = (((x + self.count * 3) // 32 + y // 32) % 2 * 160 + 30).astype(np.uint8)
        cv2.putText(image, 'DEMO - NO CAMERA', (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        return image, {'frame_id': self.count, **self.values}

    def close(self):
        pass


@dataclass
class CaptureRequest:
    path: Path
    after: float
    overwrite: bool
    manifest: bool
    metadata: dict
    deadline: float


def save_capture(request, image, metadata):
    """Exclusive creation by default prevents silent loss of existing samples."""
    request.path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('PNG 编码失败。')
    with request.path.open('wb' if request.overwrite else 'xb') as handle:
        handle.write(data.tobytes())
    if request.manifest:
        entry = {**request.metadata, **metadata, 'file': request.path.name,
                 'saved_at': time.time(), 'shape': list(image.shape), 'dtype': str(image.dtype)}
        with (request.path.parent / 'captures.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')


class CameraWorker(QtCore.QThread):
    ready = QtCore.pyqtSignal(dict)
    parametersChanged = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)
    warning = QtCore.pyqtSignal(str)
    captured = QtCore.pyqtSignal(str)
    recordingChanged = QtCore.pyqtSignal(bool, str)

    def __init__(self, settings, demo=False, backend_factory=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.factory = backend_factory or (DemoCamera if demo else
            ToupCamera if selected_camera(settings) == 'toupcam' else RealCamera)
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.latest = None
        self.latest_raw = None
        self.transform = ImageTransform.from_settings(settings, 'ccd')

    def submit(self, action, *args):
        self.commands.put((action, args))

    def take_latest(self):
        with self.lock:
            latest, self.latest = self.latest, None
            self.latest_raw = None
        return latest

    def take_latest_with_raw(self):
        """Drain one matched preview/raw pair; arrays are owned and never reused."""
        with self.lock:
            latest, raw = self.latest, self.latest_raw
            self.latest = self.latest_raw = None
        return None if latest is None else (*latest, raw)

    def run(self):
        backend = None
        recorder = camera.VideoRecorder(20, 'MJPG')
        pending = []
        calibration_batches = []
        recording_path = None
        last_frame_at = last_warning_at = time.monotonic()
        last_video_at = 0.0
        current_image = None
        current_raw = current_metadata = None
        try:
            backend = self.factory(self.settings)
            parameters = backend.open()
            if self.isInterruptionRequested():
                return
            self.ready.emit(parameters)
            while not self.isInterruptionRequested():
                while True:
                    try:
                        action, args = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if action == 'parameters':
                            self.parametersChanged.emit(backend.apply(*args))
                        elif action == 'calibration_parameters':
                            request, exposure, gain, auto = args
                            try:
                                request.result = backend.apply(exposure, gain, auto)
                                self.parametersChanged.emit(request.result)
                            except Exception as exc:
                                request.error = str(exc)
                            finally:
                                request.event.set()
                        elif action == 'calibration_capture':
                            calibration_batches.append(args[0])
                        elif action == 'cancel_calibration':
                            for batch in calibration_batches:
                                batch.fail('标定采集已取消。')
                            calibration_batches.clear()
                        elif action == 'capture':
                            pending.append(args[0])
                        elif action == 'cancel_captures':
                            pending.clear()
                        elif action == 'cancel_auto_captures':
                            pending[:] = [request for request in pending if not request.metadata.get('automatic')]
                        elif action == 'transform':
                            transform = args[0]
                            if not isinstance(transform, ImageTransform):
                                raise TypeError('CCD 图像变换参数无效。')
                            if transform == self.transform:
                                continue
                            if recorder.active:
                                raise RuntimeError('请先停止录像，再调整 CCD 旋转或翻转。')
                            # Prepare before replacing state. A queued record_start
                            # must see the new dimensions even before the next read.
                            transformed = (transform.apply_array(current_raw)
                                           if current_raw is not None else None)
                            self.transform = transform
                            current_image = transformed
                            if current_metadata is not None:
                                current_metadata = {**current_metadata,
                                                    'image_transform': transform.as_dict()}
                                with self.lock:
                                    self.latest = (current_image, current_metadata)
                                    self.latest_raw = current_raw
                            if pending:
                                pending.clear()
                                self.warning.emit('CCD 图像方向已更改，待处理的采集已取消，请重新采集。')
                        elif action == 'record_start':
                            if current_image is None:
                                raise RuntimeError('尚未收到相机帧。')
                            path, fps = args
                            path.parent.mkdir(parents=True, exist_ok=True)
                            # Reserve the filename before OpenCV opens it.
                            with path.open('xb'):
                                pass
                            recorder.fps = fps
                            try:
                                recorder.start(path, current_image)
                            except Exception:
                                path.unlink(missing_ok=True)
                                raise
                            recording_path = path
                            self.recordingChanged.emit(True, str(path))
                        elif action == 'record_stop':
                            recorder.stop()
                            self.recordingChanged.emit(False, str(recording_path or ''))
                            recording_path = None
                    except Exception as exc:
                        self.warning.emit(str(exc))
                        if action.startswith('record_'):
                            self.recordingChanged.emit(recorder.active, str(recording_path or ''))
                now = time.monotonic()
                if hasattr(backend, 'ensure_fresh'):
                    barriers = [request.after for request in (*pending, *calibration_batches)
                                if request.after <= now]
                    if barriers:
                        backend.ensure_fresh(max(barriers))
                read_started = time.monotonic()
                try:
                    raw_image, raw_metadata = backend.read()
                except TimeoutError:
                    now = time.monotonic()
                    for batch in calibration_batches[:]:
                        if now > batch.deadline or batch.cancelled:
                            batch.fail('标定采集超时或已取消。')
                            calibration_batches.remove(batch)
                    if now - last_frame_at >= camera_value(self.settings, 'TimeoutSpinBox', 500) / 1000 and now - last_warning_at >= 2:
                        self.warning.emit('CCD 取帧超时，正在等待下一帧。')
                        last_warning_at = now
                    for request in pending[:]:
                        if now > request.deadline:
                            pending.remove(request)
                            self.warning.emit(f'采集超时，未保存：{request.path.name}')
                    continue
                # Preserve camera pixel values and channel ordering. Preview,
                # PNG captures and AVI recording all consume this same result.
                current_raw = np.array(raw_image, copy=True, order='C')
                image = current_raw if self.transform.is_identity else self.transform.apply_array(current_raw)
                current_image = image
                last_frame_at = time.monotonic()
                metadata = {**raw_metadata,
                            'received_at_monotonic': last_frame_at,
                            'raw_shape': list(current_raw.shape),
                            'image_transform': self.transform.as_dict()}
                current_metadata = metadata
                for batch in calibration_batches[:]:
                    calibration_metadata = {**metadata, 'calibration_raw': True,
                        'image_transform': ImageTransform().as_dict(),
                        'preview_image_transform': self.transform.as_dict(),
                        'read_started_at_monotonic': read_started}
                    batch.offer(current_raw, calibration_metadata, read_started, last_frame_at)
                    if batch.event.is_set():
                        calibration_batches.remove(batch)
                with self.lock:
                    self.latest = (image, metadata)
                    self.latest_raw = current_raw
                if recorder.active and last_frame_at - last_video_at >= 1 / recorder.fps:
                    recorder.write(image)
                    last_video_at = last_frame_at
                for request in pending[:]:
                    if read_started >= request.after:
                        pending.remove(request)
                        try:
                            if last_frame_at > request.deadline:
                                raise TimeoutError(f'采集窗口已过期：{request.path.name}')
                            save_capture(request, image, metadata)
                            self.captured.emit(str(request.path))
                        except Exception as exc:
                            self.warning.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            for batch in calibration_batches:
                batch.fail('相机已关闭。')
            try:
                try:
                    recorder.stop()
                    self.recordingChanged.emit(False, str(recording_path or ''))
                finally:
                    if backend is not None:
                        backend.close()
            except Exception as exc:
                self.warning.emit(f'相机关闭失败：{exc}')
