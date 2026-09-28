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
from .configuration import input_path


class RealCamera:
    def __init__(self, settings):
        self.settings = settings
        self.api = self.session = None

    def open(self):
        s = self.settings
        self.api = dvp.DvpApi(input_path(s['ccdDllPathEdit']))
        devices = self.api.list_devices()
        if s['ccdCameraSpinBox'] >= len(devices):
            raise ValueError('所选 CCD 编号不存在，请先枚举设备。')
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
                'auto': camera.get_ae_operation(self.api, handle) == dvp.AE_OP_CONTINUOUS}

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
            'exposure': float(frame.fExposure), 'gain': float(frame.fAGain)}

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
        self.factory = backend_factory or (DemoCamera if demo else RealCamera)
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.latest = None

    def submit(self, action, *args):
        self.commands.put((action, args))

    def take_latest(self):
        with self.lock:
            latest, self.latest = self.latest, None
        return latest

    def run(self):
        backend = None
        recorder = camera.VideoRecorder(20, 'MJPG')
        pending = []
        recording_path = None
        last_frame_at = last_warning_at = time.monotonic()
        last_video_at = 0.0
        current_image = None
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
                        elif action == 'capture':
                            pending.append(args[0])
                        elif action == 'cancel_captures':
                            pending.clear()
                        elif action == 'cancel_auto_captures':
                            pending[:] = [request for request in pending if not request.metadata.get('automatic')]
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
                read_started = time.monotonic()
                try:
                    image, metadata = backend.read()
                except TimeoutError:
                    now = time.monotonic()
                    if now - last_frame_at >= self.settings['ccdTimeoutSpinBox'] / 1000 and now - last_warning_at >= 2:
                        self.warning.emit('CCD 取帧超时，正在等待下一帧。')
                        last_warning_at = now
                    for request in pending[:]:
                        if now > request.deadline:
                            pending.remove(request)
                            self.warning.emit(f'采集超时，未保存：{request.path.name}')
                    continue
                current_image = image
                last_frame_at = time.monotonic()
                metadata['received_at_monotonic'] = last_frame_at
                with self.lock:
                    self.latest = (image, metadata)
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
            try:
                try:
                    recorder.stop()
                    self.recordingChanged.emit(False, str(recording_path or ''))
                finally:
                    if backend is not None:
                        backend.close()
            except Exception as exc:
                self.warning.emit(f'相机关闭失败：{exc}')
