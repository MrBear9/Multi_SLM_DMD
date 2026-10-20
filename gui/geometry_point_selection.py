"""Non-blocking, raw-frame selection on the existing CCD detail page."""
import json

import numpy as np
from PyQt5 import QtCore, QtWidgets, uic

from ..holography.spot_detection import search_rect
from ..utils.paths import TOOLS_ROOT


class GeometryPointSelection:
    def __init__(self, controller):
        self.controller = controller
        self.window = w = controller.window
        self.pending = None
        self.panel = uic.loadUi(str(TOOLS_ROOT/'ui'/'GeometryPointSelection.ui'))
        w.ccdDetailPageLayout.insertWidget(0, self.panel)
        self.panel.hide()
        self.panel.confirmButton.clicked.connect(lambda: w.ccd_detail.run_action(self.confirm))
        self.panel.recaptureButton.clicked.connect(lambda: w.ccd_detail.run_action(self.recapture))
        self.panel.cancelButton.clicked.connect(controller.cancel)

    def begin(self, message):
        c = self.controller
        if self.pending is not None or c.capture_task is not None or not c.hardware_active or c.config.operation != 'geometry':
            raise ValueError('当前任务不能进入定位框选。')
        path = c.safe_path(message['path'])
        raw = np.load(path, allow_pickle=False)
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if isinstance(metadata, list):
            if not metadata:
                raise ValueError('定位采集图缺少帧元数据。')
            metadata = metadata[-1]
        # Use the actual transaction metadata, never the newest preview frame.
        hardware = c.hardware_signature()
        geometry = metadata.get('camera_geometry')
        if geometry != hardware.get('camera', {}).get('geometry'):
            raise ValueError('定位采集图与当前相机几何设置不一致。')
        if metadata.get('camera') and metadata['camera'] != hardware.get('camera', {}).get('camera'):
            raise ValueError('定位采集图来自不同相机。')
        detail = self.window.ccd_detail
        self.previous = (detail.displayed, detail.frozen, detail.view.roi,
                         self.window.ccdDetailRoiNameEdit.text(),
                         self.window.ccdDetailRoiDrawButton.isChecked())
        self.pending = dict(message)
        self.packet = (raw, metadata, None)
        latest = detail.latest
        detail.selection_locked = True
        detail.latest = self.packet
        detail.present_latest()
        detail.latest = latest
        detail.set_frozen(True)
        with QtCore.QSignalBlocker(self.window.ccdDetailFreezeButton):
            self.window.ccdDetailFreezeButton.setChecked(True)
        detail.view.set_roi(None)
        self.window.ccdDetailRoiNameEdit.setText('当前定位点搜索范围')
        self.window.ccdDetailShowRoiCheckBox.setChecked(True)
        self.window.ccdDetailRoiDrawButton.setChecked(True)
        detail.view.set_draw_mode(True)
        exposure = metadata.get('exposure', self.window.camera_parameters.get('exposure', 150.))
        self.panel.exposure.setValue(exposure)
        reason = message.get('reason')
        self.panel.instruction.setText(
            f'几何标定已暂停 · 定位点 {message["point_index"]+1}/{message["point_count"]}'
            f' · 第 {message["repeat_index"]+1} 次采集 · 已冻结本次定位采集图（按设置平均）。\n'
            + (f'说明：{reason}\n' if reason else '')
            + ('框住中心及数圈完整内环，避开矩形外框；确认后拟合共同中心。可缩放、平移。'
               if c.config.geometry_detector=='ring' else
               '框住当前光斑及完整外围（留出余量），确认后由程序拟合中心；可缩放、平移。'))
        self.panel.show()
        c.dialog.hide()
        self.window.mainTabWidget.setCurrentWidget(self.window.ccdDetailPage)
        detail.view.fit_image()
        self.window.raise_()
        self.window.activateWindow()
        c.log('等待第三页框选当前定位点；相位保持，定位图已冻结。')

    def confirm(self):
        if self.pending is None:
            return
        c = self.controller
        detail = self.window.ccd_detail
        if detail.displayed is not self.packet or not detail.frozen:
            raise ValueError('定位图已改变，请重新采集。')
        rect = detail.view.roi
        if rect is None:
            raise ValueError('请先框选当前定位图案及完整外围。')
        document = detail._document(dict(name='当前定位点', x=rect.x(), y=rect.y(),
                                        width=rect.width(), height=rect.height()))
        document = json.loads(json.dumps(document, ensure_ascii=False))
        search_rect(document, self.packet[0].shape, c.hardware_signature())
        self.reply({'action':'select', 'roi':document})

    def recapture(self):
        if self.pending is None:
            return
        exposure = self.panel.exposure.value()
        self.controller.validate_exposure(exposure)
        self.reply({'action':'recapture', 'exposure_us':exposure})

    def reply(self, response):
        c = self.controller
        if c.stopping or c.process is None or c.process_done:
            self.finish()
            return
        response['request_id'] = self.pending['request_id']
        c.process.write((json.dumps(response, ensure_ascii=False)+'\n').encode('utf-8'))
        self.finish()

    def finish(self):
        if self.pending is None:
            return
        self.pending = None
        self.panel.hide()
        detail = self.window.ccd_detail
        displayed, frozen, roi, name, drawing = self.previous
        detail.selection_locked = False
        if frozen and displayed is not None:
            latest = detail.latest
            detail.latest = displayed
            detail.present_latest()
            detail.latest = latest
        detail.set_frozen(frozen)
        with QtCore.QSignalBlocker(self.window.ccdDetailFreezeButton):
            self.window.ccdDetailFreezeButton.setChecked(detail.frozen)
        # A later resolution change must not restore stale annotation coordinates.
        same_size = (displayed is not None and detail.displayed is not None
                     and displayed[0].shape[:2] == detail.displayed[0].shape[:2])
        detail.view.set_roi(roi if same_size else None)
        self.window.ccdDetailRoiNameEdit.setText(name)
        (self.window.ccdDetailRoiDrawButton if drawing else self.window.ccdDetailPanButton).setChecked(True)
        detail.view.set_draw_mode(drawing)
        detail.update_actions()
        self.packet = self.previous = None
        self.controller.dialog.show()
        self.controller.dialog.raise_()
