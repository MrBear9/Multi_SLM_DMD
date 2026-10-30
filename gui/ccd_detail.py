"""CCD detail page, bounded frame caching and raw-coordinate ROI annotations."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ..utils.image_transforms import ImageTransform
from ..utils.camera_settings import camera_roi
from .configuration import camera_qimage, input_path
from .ccd_detail_view import CcdDetailView, FrameNavigator


def validate_roi_document(document, frame_size, camera, capture_roi):
    """Validate completely before changing the current annotation."""
    if not isinstance(document, dict) or document.get('version') != 1 or document.get('kind') != 'ccd_roi':
        raise ValueError('不是支持的 CCD ROI 标注文件。')
    if document.get('coordinates') != 'raw_frame_xywh':
        raise ValueError('标注必须使用原始采集帧的像素坐标。')
    if document.get('frame_size') != list(frame_size):
        raise ValueError('标注的图像尺寸与当前采集帧不同，请使用匹配的图像。')
    stored_camera = document.get('camera', {})
    if not isinstance(stored_camera, dict):
        raise ValueError('标注相机信息格式无效。')
    for key in ('serial', 'model'):
        if camera.get(key) and stored_camera.get(key) and camera[key] != stored_camera[key]:
            raise ValueError('标注来自另一台相机或型号，不能直接套用。')
    if document.get('capture_roi') != capture_roi:
        raise ValueError('标注的相机采集 ROI 与当前设置不同。')
    roi = document.get('roi')
    if not isinstance(roi, dict) or not isinstance(roi.get('name'), str) or not roi['name'].strip():
        raise ValueError('ROI 名称不能为空。')
    values = [roi.get(key) for key in ('x', 'y', 'width', 'height')]
    if any(type(value) is not int for value in values):
        raise ValueError('ROI 坐标及尺寸必须是整数。')
    x, y, width, height = values
    if min(x, y) < 0 or min(width, height) < 1 or x+width > frame_size[0] or y+height > frame_size[1]:
        raise ValueError('ROI 超出当前原始图像范围。')
    return roi['name'].strip(), QtCore.QRect(*values)


class CcdDetailController(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = w = window
        self.latest = self.displayed = None
        self.overview_viewport = None
        self.frozen = False
        self.cursor_point = None
        self.camera = {}
        self.capture_roi = None
        old = w.ccdDetailGraphicsView
        self.view = CcdDetailView(old.parentWidget())
        self.view.setObjectName(old.objectName())
        w.ccdDetailCanvasLayout.replaceWidget(old, self.view)
        old.hide()
        old.deleteLater()
        w.ccdDetailGraphicsView = self.view
        old = w.ccdDetailNavigatorLabel
        self.navigator = FrameNavigator(self.view, old.parentWidget())
        self.navigator.setObjectName(old.objectName())
        w.ccdDetailNavigatorLayout.replaceWidget(old, self.navigator)
        old.hide()
        old.deleteLater()
        w.ccdDetailNavigatorLabel = self.navigator
        self.navigator.centerRequested.connect(self.view.centerOn)
        self.view.viewChanged.connect(self.view_changed)
        self.view.roiChanged.connect(self.roi_changed)
        self.view.editingStarted.connect(self.begin_roi_edit)
        self.view.cursorChanged.connect(self.show_pixel)
        w.ccdDetailFitButton.clicked.connect(self.view.fit_image)
        w.ccdDetailActualSizeButton.clicked.connect(lambda: self.view.set_zoom(1))
        w.ccdDetailZoomInButton.clicked.connect(lambda: self.view.set_zoom(self.view.zoom*1.2))
        w.ccdDetailZoomOutButton.clicked.connect(lambda: self.view.set_zoom(self.view.zoom/1.2))
        w.ccdDetailZoomSpinBox.setKeyboardTracking(False)
        w.ccdDetailZoomSpinBox.setDecimals(1)
        w.ccdDetailZoomSpinBox.setMinimum(.01)
        w.ccdDetailZoomSpinBox.valueChanged.connect(lambda value: self.view.set_zoom(value/100))
        w.ccdDetailPanButton.toggled.connect(lambda checked: self.view.set_draw_mode(False) if checked else None)
        w.ccdDetailRoiDrawButton.toggled.connect(lambda checked: self.view.set_draw_mode(True) if checked else None)
        w.ccdDetailFreezeButton.toggled.connect(self.set_frozen)
        w.ccdDetailBackButton.clicked.connect(lambda: w.mainTabWidget.setCurrentWidget(w.previewPage))
        w.mainTabWidget.currentChanged.connect(self.page_changed)
        w.ccdDetailRoiApplyButton.clicked.connect(lambda: self.run_action(self.apply_roi))
        w.ccdDetailRoiClearButton.clicked.connect(lambda: self.view.set_roi(None))
        w.ccdDetailShowRoiCheckBox.toggled.connect(self.show_roi)
        w.ccdDetailRoiNameEdit.textChanged.connect(self.rename_roi)
        self.fields = [getattr(w, 'ccdDetailRoi'+key+'SpinBox') for key in ('X', 'Y', 'Width', 'Height')]
        for field in self.fields:
            field.setKeyboardTracking(False)
            field.editingFinished.connect(self.fields_edited)
        w.ccdDetailRoiLoadButton.clicked.connect(self.load_dialog)
        w.ccdDetailRoiSaveButton.clicked.connect(self.save_dialog)
        w.ccdDetailCaptureButton.clicked.connect(lambda: w.guard(w.capture_manual))
        w.ccdDetailRecordButton.clicked.connect(lambda: w.guard(w.toggle_recording))
        w.ccdDetailOutputBrowseButton.clicked.connect(self.choose_output_directory)
        w.ccdDetailOutputDirectoryEdit.editingFinished.connect(self.edit_output_directory)
        w.outputDirectoryEdit.textChanged.connect(self.sync_output_directory)
        self.sync_output_directory(w.outputDirectoryEdit.text())
        self.update_capture_actions()
        w.ccdDetailSplitter.setSizes([1000, 310])
        self.screen_handle = None
        w.installEventFilter(self)
        from .ccd_geometry_overlay import CcdGeometryOverlay
        self.geometry_overlay = CcdGeometryOverlay(self)
        self.update_actions()

    def sync_output_directory(self, text):
        field = self.window.ccdDetailOutputDirectoryEdit
        field.setText(text)
        try:
            field.setToolTip(str(input_path(text)) + '\n可直接输入目录，按回车或离开输入框后应用。')
        except ValueError:
            field.setToolTip(text)

    def choose_output_directory(self):
        w = self.window
        if getattr(w, 'automation_active', w.calibration.active) or w.disconnecting or w.device_task is not None:
            return
        text = w.outputDirectoryEdit.text()
        initial = str(input_path(text)) if text.strip() else ''
        path = QtWidgets.QFileDialog.getExistingDirectory(w, '选择 CCD 保存目录', initial)
        if not path:
            return
        self.set_output_directory(path)

    def edit_output_directory(self):
        self.set_output_directory(self.window.ccdDetailOutputDirectoryEdit.text())

    def set_output_directory(self, text):
        w = self.window
        previous = w.outputDirectoryEdit.text()
        if getattr(w, 'automation_active', w.calibration.active) or w.disconnecting or w.device_task is not None:
            self.sync_output_directory(previous)
            return
        text = text.strip()
        if text == previous:
            self.sync_output_directory(previous)
            return
        try:
            directory = input_path(text)
            directory.mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as exc:
            # Restore before the dialog takes focus, to avoid another failed commit.
            self.sync_output_directory(previous)
            QtWidgets.QMessageBox.warning(w, '保存目录无效', f'未更改保存目录：{exc}')
            return
        w.outputDirectoryEdit.setText(text)
        if w.settings is not None:
            # Queued captures already own their paths; an active recorder owns its file.
            w.settings['outputDirectoryEdit'] = text
        w.statusBar().showMessage('保存目录已更新；后续采集和新录像使用此目录。', 6000)

    def update_capture_actions(self):
        w = self.window
        w.ccdDetailCaptureButton.setEnabled(w.ccdCaptureButton.isEnabled())
        w.ccdDetailRecordButton.setEnabled(w.ccdRecordButton.isEnabled())
        w.ccdDetailRecordButton.setText(w.ccdRecordButton.text())
        w.ccdDetailOutputBrowseButton.setEnabled(
            not (getattr(w, 'automation_active', w.calibration.active) or w.disconnecting or w.device_task is not None))
        w.ccdDetailOutputDirectoryEdit.setEnabled(w.ccdDetailOutputBrowseButton.isEnabled())
        suffix = ' · 录像处理中' if w.recording_pending else (' · 正在录像' if w.recording else '')
        w.ccdDetailSaveStatusLabel.setText(f'已保存 {w.saved_frames} 帧' + suffix)

    def eventFilter(self, watched, event):
        # Qt can deliver destruction events after Python has cleared the
        # controller's attributes while collecting a closed window.
        window = getattr(self, 'window', None)
        if window is not None and watched is window and event.type() == QtCore.QEvent.Show:
            handle = window.windowHandle()
            if handle is not None and handle is not self.screen_handle:
                self.screen_handle = handle
                handle.screenChanged.connect(lambda _: self.view.refresh_scale())
        return super().eventFilter(watched, event)

    def run_action(self, action):
        # Annotation failures must not stop device playback or acquisition.
        try:
            action()
        except (ValueError, OSError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(self.window, 'CCD 标注', str(exc))

    def start_session(self, identity, settings):
        self.clear()
        self.camera = dict(identity)
        self.capture_roi = camera_roi(settings)

    def clear(self):
        self.latest = self.displayed = None
        self.overview_viewport = None
        self.frozen = False
        with QtCore.QSignalBlocker(self.window.ccdDetailFreezeButton):
            self.window.ccdDetailFreezeButton.setChecked(False)
        self.window.ccdDetailFreezeButton.setText('冻结画面')
        self.view.clear_image()
        self.geometry_overlay.refresh()
        self.navigator.image = self.view.image_item.pixmap().toImage()
        self.navigator.update()
        self.window.ccdDetailImageSizeLabel.setText('采集尺寸  — × — px')
        self.window.ccdDetailFrameLabel.setText('帧号 —   /   预览 — FPS')
        self.window.ccdDetailStateLabel.setText('● 未连接 · 等待相机画面')
        self.update_actions()

    def receive(self, raw, metadata, preview=None):
        if raw is None:
            return
        # Worker frames own their data and are never mutated after publication.
        # Keep only the newest packet and (while frozen) the displayed packet.
        self.latest = (raw, dict(metadata), preview)
        if not self.frozen and self.window.mainTabWidget.currentWidget() is self.window.ccdDetailPage:
            self.present_latest()

    def present_latest(self):
        if self.latest is None:
            return
        raw, metadata, preview = self.latest
        changed = self.displayed is not None and raw.shape[:2] != self.displayed[0].shape[:2]
        self.displayed = self.latest
        if changed:
            self.view.cancel_edit()
            self.view.set_roi(None)
            self.window.statusBar().showMessage('CCD 帧尺寸已变化，已清除旧 ROI；请重新标注。', 8000)
        transform = ImageTransform(**metadata.get('image_transform', {}))
        orientation_changed = transform != self.view.orientation
        image = preview if preview is not None and transform.is_identity else camera_qimage(raw)
        self.view.set_image(image, transform)
        self.geometry_overlay.frame_changed()
        self.navigator.image = image
        self.navigator.update()
        h, width = raw.shape[:2]
        self.window.ccdDetailImageSizeLabel.setText(f'采集尺寸  {width} × {h} px · {raw.dtype}')
        for field, limit in zip(self.fields, (width-1, h-1, width, h)):
            with QtCore.QSignalBlocker(field):
                field.setMaximum(limit)
        self.show_pixel(None if changed or orientation_changed else self.cursor_point)
        self.update_actions()
        self.update_status()

    def overview_image(self, size):
        if self.latest is None or self.overview_viewport is None:
            return None
        return self.overview_viewport.render(self.latest,size)

    def page_changed(self, index):
        if self.window.mainTabWidget.currentWidget() is self.window.ccdDetailPage:
            if not self.frozen:
                self.present_latest()
            self.update_status()

    def freeze(self):
        if self.displayed is not None:
            self.window.ccdDetailFreezeButton.setChecked(True)

    def begin_roi_edit(self):
        self.freeze()
        self.window.ccdDetailShowRoiCheckBox.setChecked(True)

    def set_frozen(self, frozen):
        if getattr(self, 'selection_locked', False):
            frozen = True
        self.frozen = frozen and self.displayed is not None
        self.window.ccdDetailFreezeButton.setText('恢复实时' if self.frozen else '冻结画面')
        if not self.frozen:
            self.view.cancel_edit()
            self.present_latest()
        self.update_status()
        if hasattr(self.window, 'capture_export'):
            self.window.capture_export.update_actions()

    def update_status(self):
        if self.displayed is None:
            return
        w = self.window
        w.ccdDetailStateLabel.setText('● 已冻结 · 后台采集继续' if self.frozen else
                                    ('● 演示图像源' if w.demo else '● 实时预览'))
        metadata = self.displayed[1]
        fps = '冻结帧' if self.frozen else w.ccdActiveAreaLabel.text()
        w.ccdDetailFrameLabel.setText(f'帧 {metadata.get("frame_id", "—")}   /   {fps}')

    def view_changed(self):
        with QtCore.QSignalBlocker(self.window.ccdDetailZoomSpinBox):
            self.window.ccdDetailZoomSpinBox.setValue(self.view.zoom*100)
        self.navigator.update()
        # Save only the visible viewport geometry, never the frozen image.
        if self.displayed is not None and self.window.mainTabWidget.currentWidget() is self.window.ccdDetailPage:
            from .ccd_viewport_preview import DetailViewport
            self.overview_viewport = DetailViewport.capture(self.view,self.displayed)

    def show_roi(self, checked):
        self.view.show_roi = checked
        self.view.viewport().update()

    def rename_roi(self, name):
        self.view.roi_name = name.strip() or 'ROI'
        self.view.viewport().update()

    def roi_changed(self, rect):
        values = (rect.x(), rect.y(), rect.width(), rect.height()) if rect is not None else (0,0,0,0)
        for field, value in zip(self.fields, values):
            with QtCore.QSignalBlocker(field):
                field.setValue(value)
        self.update_actions()

    def apply_roi(self):
        if self.displayed is None:
            raise ValueError('请先连接相机并等待采集帧。')
        name = self.window.ccdDetailRoiNameEdit.text()
        document = self._document(dict(zip(('x','y','width','height'), [f.value() for f in self.fields]), name=name))
        name, rect = validate_roi_document(document, self.frame_size, self.camera, self.capture_roi)
        self.freeze()
        self.view.set_roi(rect)
        self.window.statusBar().showMessage(f'ROI 已确认：({rect.x()}, {rect.y()})，{rect.width()} × {rect.height()} px', 5000)

    def fields_edited(self):
        if self.displayed is not None and all(field.value() > 0 for field in self.fields[2:]):
            self.run_action(self.apply_roi)

    @property
    def frame_size(self):
        return tuple(self.displayed[0].shape[1::-1])

    def _document(self, roi):
        metadata = self.displayed[1]
        return {'version': 1, 'kind': 'ccd_roi', 'coordinates': 'raw_frame_xywh',
                'frame_size': list(self.frame_size), 'camera': self.camera,
                'capture_roi': self.capture_roi, 'roi': roi,
                'camera_geometry': metadata.get('camera_geometry'),
                'reference_frame': {key: metadata.get(key) for key in
                    ('frame_id', 'exposure', 'gain', 'received_at_monotonic', 'image_transform')},
                'dtype': str(self.displayed[0].dtype),
                'saved_at': datetime.now(timezone.utc).isoformat()}

    def save_roi(self, path):
        if self.displayed is None or self.view.roi is None:
            raise ValueError('请先绘制或设置有效的 ROI。')
        r = self.view.roi
        document = self._document({'name': self.window.ccdDetailRoiNameEdit.text().strip(),
                                  'x': r.x(), 'y': r.y(), 'width': r.width(), 'height': r.height()})
        validate_roi_document(document, self.frame_size, self.camera, self.capture_roi)
        self.freeze()
        data = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
        output = QtCore.QSaveFile(str(path))
        if not output.open(QtCore.QIODevice.WriteOnly):
            raise OSError(output.errorString())
        if output.write(data) != len(data) or not output.commit():
            raise OSError(output.errorString())
        self.window.statusBar().showMessage('已保存 ROI 标注：' + str(path), 6000)

    def load_roi(self, path):
        if self.displayed is None:
            raise ValueError('请先连接相机并等待采集帧，再载入匹配的标注。')
        document = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(document, dict) or document.get('camera_geometry') != self.displayed[1].get('camera_geometry'):
            raise ValueError('标注与当前相机分辨率 / Binning 模式不一致。')
        name, rect = validate_roi_document(document, self.frame_size, self.camera, self.capture_roi)
        self.freeze()
        self.window.ccdDetailRoiNameEdit.setText(name)
        self.view.set_roi(rect)
        self.window.ccdDetailShowRoiCheckBox.setChecked(True)
        self.window.statusBar().showMessage('已载入 ROI 标注：' + str(path), 6000)

    def save_dialog(self):
        self.freeze()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.window, '保存 ROI 标注', 'ccd_roi.json', 'ROI 标注 (*.json)')
        if path:
            self.run_action(lambda: self.save_roi(path if Path(path).suffix else path+'.json'))

    def load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, '载入 ROI 标注', '', 'ROI 标注 (*.json)')
        if path:
            self.run_action(lambda: self.load_roi(path))

    def show_pixel(self, point):
        self.cursor_point = point
        w = self.window
        if point is None or self.displayed is None:
            w.ccdDetailCursorLabel.setText('像素坐标  X —   Y —')
            w.ccdDetailPixelValueLabel.setText('像素值 —')
            return
        x, y = math.floor(point.x()), math.floor(point.y())
        raw = self.displayed[0]
        if not 0 <= x < raw.shape[1] or not 0 <= y < raw.shape[0]:
            return self.show_pixel(None)
        value = raw[y, x]
        w.ccdDetailCursorLabel.setText(f'原始像素坐标  X {x}   Y {y}')
        if np.ndim(value):
            value = 'BGR ' + ', '.join(str(v) for v in value[:3])
        w.ccdDetailPixelValueLabel.setText(f'像素值 {value}')

    def update_actions(self):
        w = self.window
        ready = self.displayed is not None
        for name in ('Fit','ActualSize','ZoomIn','ZoomOut','Pan','RoiDraw','Freeze','RoiApply','RoiLoad'):
            getattr(w, 'ccdDetail'+name+'Button').setEnabled(ready)
        w.ccdDetailZoomSpinBox.setEnabled(ready)
        w.ccdDetailRoiGroupBox.setEnabled(ready)
        w.ccdDetailRoiClearButton.setEnabled(ready and self.view.roi is not None)
        w.ccdDetailRoiSaveButton.setEnabled(ready and self.view.roi is not None)
        if getattr(self, 'selection_locked', False):
            w.ccdDetailFreezeButton.setEnabled(False)
