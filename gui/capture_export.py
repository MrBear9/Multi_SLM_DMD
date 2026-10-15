"""Named PNG exports of fresh camera frames or a pinned detail-preview frame."""
from copy import deepcopy
from pathlib import Path
import time

from PyQt5 import QtCore, QtWidgets

from .camera_worker import CaptureRequest, save_capture
from .configuration import input_path, read_values
from ..utils.image_transforms import ImageTransform


class FrozenFrameWriter(QtCore.QThread):
    def __init__(self, request, packet, parent):
        super().__init__(parent)
        self.request, self.packet = request, packet
        self.error = None

    def run(self):
        try:
            raw, metadata = self.packet
            transform = ImageTransform(**metadata.get('image_transform', {}))
            image = raw if transform.is_identity else transform.apply_array(raw)
            save_capture(self.request, image, metadata)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.packet = None


class CaptureExport(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.dialog_open = False
        self.job = None
        window.ccdSaveAsButton.clicked.connect(lambda: self.save_as(detail=False))
        window.ccdDetailSaveAsButton.clicked.connect(lambda: self.save_as(detail=True))

    def update_actions(self):
        w = self.window
        available = not (self.dialog_open or self.job is not None or w.calibration.active
                         or w.device_task is not None or w.disconnecting or w.closing)
        live = w.camera_ready and w.received_frames > 0
        frozen = w.ccd_detail.frozen and w.ccd_detail.displayed is not None
        w.ccdSaveAsButton.setEnabled(available and live)
        w.ccdDetailSaveAsButton.setEnabled(available and (live or frozen))

    def save_as(self, detail=False):
        w = self.window
        if (self.dialog_open or self.job is not None or w.calibration.active
                or w.device_task is not None or w.disconnecting or w.closing):
            return
        # Keep the frame and its orientation before opening the nested dialog loop.
        # Published raw arrays are owned by their packets and are never reused.
        packet = None
        if detail and w.ccd_detail.frozen and w.ccd_detail.displayed is not None:
            raw, metadata, _ = w.ccd_detail.displayed
            packet = (raw, deepcopy(metadata))
        worker = w.camera
        if packet is None and not (w.camera_ready and worker is not None):
            return
        values = dict(w.settings) if w.settings is not None else read_values(w)
        self.dialog_open = True
        self.update_actions()
        try:
            title = '冻结帧另存为' if packet is not None else '新采集帧另存为'
            name = 'frame_' + time.strftime('%Y%m%d_%H%M%S') + '.png'
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                w, title, str(input_path(values['outputDirectoryEdit']) / name),
                'PNG 图像 (*.png)', options=QtWidgets.QFileDialog.DontConfirmOverwrite)
            if not chosen or w.closing:
                return
            path = Path(chosen)
            if not path.suffix:
                path = path.with_suffix('.png')
            if path.suffix.lower() != '.png':
                raise ValueError('请使用 .png 文件名，以保留采集图像的位深。')
            path = path.resolve()
            if path.is_dir():
                raise ValueError('请选择文件名，不能保存到同名目录。')
            overwrite = path.exists()
            if overwrite and QtWidgets.QMessageBox.question(
                    w, '确认覆盖', f'文件已存在，是否覆盖？\n{path}',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
                return
            if w.closing:
                return
            if packet is None:
                if w.camera is not worker or not w.camera_ready or w.disconnecting:
                    raise ValueError('相机连接已改变，请连接就绪后重新另存。')
                # The freshness barrier is created after the save dialog closes.
                w.queue_capture(path.name, destination=path, overwrite=overwrite)
                w.statusBar().showMessage('正在采集新帧并另存：' + str(path))
            else:
                request = CaptureRequest(path, 0, overwrite, values['saveManifestCheckBox'],
                    {'experiment': values['sessionNameEdit'], 'demo': w.demo,
                     'automatic': False, 'save_source': 'frozen_preview'}, float('inf'))
                self.job = FrozenFrameWriter(request, packet, self)
                self.job.finished.connect(self.finished)
                self.job.start()
                w.statusBar().showMessage('正在另存冻结帧：' + str(path))
        except Exception as exc:
            # Export failures must not stop an experiment or camera recording.
            QtWidgets.QMessageBox.warning(w, '另存失败', str(exc))
        finally:
            self.dialog_open = False
            self.update_actions()

    def finished(self):
        w = self.window
        job, self.job = self.job, None
        if job.error:
            w.statusBar().showMessage('冻结帧另存失败：' + job.error)
            if not w.closing:
                QtWidgets.QMessageBox.warning(w, '另存失败', job.error)
        else:
            w.on_capture(str(job.request.path))
        job.deleteLater()
        self.update_actions()
        if w.closing:
            w.close()
