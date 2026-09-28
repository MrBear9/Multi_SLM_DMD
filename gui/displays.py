"""Qt DMD output and adapters around the two existing SLM SDKs."""
from __future__ import annotations

from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

from ..devices.zkwx_slm import sdk as zkwx
from ..devices.zkwx_slm import controller as zkwx_images
from ..devices.magicholo_slm import sdk as magic
from ..devices.magicholo_slm import controller as magic_images
from .configuration import input_path, pil_qimage


class DmdOutput(QtWidgets.QWidget):
    closed = QtCore.pyqtSignal()

    def __init__(self, geometry, show_cursor=False):
        super().__init__(None, QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, False)
        self.setWindowTitle('DMD · 原始像素输出')
        self.setGeometry(*geometry)
        self.image = None
        self.set_cursor(show_cursor)
        self.winId()
        screen = next((s for s in QtWidgets.QApplication.screens()
                       if s.geometry() == QtCore.QRect(*geometry)), None)
        if screen is None:
            raise ValueError('Qt 与 Windows 显示器坐标不一致，请检查显示缩放设置。')
        self.windowHandle().setScreen(screen)
        self.showFullScreen()

    def set_cursor(self, visible):
        self.setCursor(QtCore.Qt.ArrowCursor if visible else QtCore.Qt.BlankCursor)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)
        if self.image is not None:
            painter.drawImage((self.width() - self.image.width()) // 2,
                              (self.height() - self.image.height()) // 2, self.image)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()


class DisplayAdapter:
    def __init__(self, key, settings, paths, demo=False):
        self.key, self.demo = key, demo
        self.sdk = self.output = self.display = None
        self.geometry = None
        self.settings = settings
        try:
            if demo:
                size = (1920, 1080)
                self.geometry = ((0, 1920, 3840)[('dmd', 'slm1', 'slm2').index(key)], 0, *size)
                self.display = magic.SDKDisplay(0, 'DEMO', *self.geometry, False) if key == 'slm2' else zkwx.Monitor(0, *self.geometry, False)
            elif key == 'dmd':
                monitors = zkwx.enumerate_monitors()
                index = settings['dmdMonitorSpinBox']
                if index >= len(monitors):
                    raise ValueError(f'DMD 显示器 {index} 不存在。')
                self.display = monitors[index]
            elif key == 'slm1':
                self.display = zkwx_images.select_monitor(zkwx.enumerate_monitors(),
                    None if settings['slm1MonitorSpinBox'] == -1 else settings['slm1MonitorSpinBox'])
                self.sdk = zkwx.ZhongkeTimeoutSDK(input_path(settings['slm1SdkPathEdit']))
            else:
                self.sdk = magic.HDSLM8BitSDK(input_path(settings['slm2SdkPathEdit']))
                self.display = magic_images.select_display(self.sdk.list_displays(),
                    None if settings['slm2DisplaySpinBox'] == -1 else settings['slm2DisplaySpinBox'])
            self.geometry = (self.display.left, self.display.top, self.display.width, self.display.height)
            if key == 'slm2':
                size = magic_images.validate_images(paths, self.display)
                magic_images.canvas_origin(size, self.display, settings['slm2OffsetXSpinBox'], settings['slm2OffsetYSpinBox'])
            elif key == 'slm1':
                size = zkwx_images.validate_images(paths, self.display)
                zkwx_images.canvas_origin(size, self.display,
                    settings.get('slm1OffsetXSpinBox', 0), settings.get('slm1OffsetYSpinBox', 0))
            else:
                for path in paths:
                    with Image.open(path) as image:
                        if image.width > self.display.width or image.height > self.display.height:
                            raise ValueError(f'DMD 图像 {path.name} 超出显示器尺寸，禁止拉伸或裁切。')
        except Exception:
            self.close()
            raise

    def open(self):
        if self.demo:
            return
        if self.key == 'dmd':
            self.output = DmdOutput(self.geometry, self.settings['dmdShowCursorCheckBox'])
        elif self.key == 'slm1':
            self.sdk.open()
        else:
            self.sdk.open(self.display.index)

    def show(self, path, settings):
        if self.key == 'slm2':
            with Image.open(path) as image:
                origin = magic_images.canvas_origin(image.size, self.display,
                    settings['slm2OffsetXSpinBox'], settings['slm2OffsetYSpinBox'])
            data = magic_images.prepare_canvas(path, self.display, origin, settings['slm2BackgroundSpinBox'])
            if not self.demo:
                self.sdk.show_8bit(self.display.index, self.display.width, self.display.height, data)
        elif self.key == 'slm1':
            offset_x = settings.get('slm1OffsetXSpinBox', 0)
            offset_y = settings.get('slm1OffsetYSpinBox', 0)
            with Image.open(path) as image:
                zkwx_images.canvas_origin(image.size, self.display, offset_x, offset_y)
            if not self.demo:
                self.sdk.dll.Timeout_ShowWindow(True)
                self.sdk.show(path, self.display, 16, offset_x, offset_y)
        elif not self.demo:
            image = pil_qimage(path)
            if image.width() > self.output.width() or image.height() > self.output.height():
                raise ValueError('DMD 输入超过输出尺寸。')
            self.output.image = image
            self.output.set_cursor(settings['dmdShowCursorCheckBox'])
            self.output.repaint()

    def blank(self):
        if not self.demo and self.key == 'slm1' and self.sdk is not None:
            self.sdk.dll.Timeout_ShowWindow(False)

    def close(self):
        if self.output is not None:
            output, self.output = self.output, None
            output.blockSignals(True)
            output.close()
            output.deleteLater()
        if self.sdk is not None:
            sdk, self.sdk = self.sdk, None
            sdk.close()
