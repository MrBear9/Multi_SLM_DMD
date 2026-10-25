"""Interactive camera viewport. Scene coordinates are untransformed raw pixels."""
import math

from PyQt5 import QtCore, QtGui, QtWidgets

from ..utils.image_transforms import ImageTransform


def orientation_matrix(orientation):
    return QtGui.QTransform().rotate(orientation.angle).scale(
        -1 if orientation.horizontal else 1, -1 if orientation.vertical else 1)


class CcdDetailView(QtWidgets.QGraphicsView):
    viewChanged = QtCore.pyqtSignal()
    roiChanged = QtCore.pyqtSignal(object)
    editingStarted = QtCore.pyqtSignal()
    cursorChanged = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.image_item = self.scene().addPixmap(QtGui.QPixmap())
        self.image_item.setTransformationMode(QtCore.Qt.FastTransformation)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setMinimumSize(400, 240)
        self.setMouseTracking(True)
        self.setTransformationAnchor(self.NoAnchor)
        self.setResizeAnchor(self.AnchorViewCenter)
        self.zoom = 1.0
        self.fit_mode = False
        self.orientation = ImageTransform()
        self.image_size = QtCore.QSize()
        self.roi = None
        self.roi_ellipse = False
        self.roi_name = 'ROI 01'
        self.show_roi = True
        self.draw_mode = False
        self.action = None
        self.setDragMode(self.ScrollHandDrag)
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(self.viewChanged)

    def set_image(self, image, orientation):
        changed = image.size() != self.image_size or orientation != self.orientation
        center = self.mapToScene(self.viewport().rect().center())
        new_size = image.size() != self.image_size
        self.image_size, self.orientation = image.size(), orientation
        self.image_item.setPixmap(QtGui.QPixmap.fromImage(image))
        self.setSceneRect(QtCore.QRectF(QtCore.QPointF(), QtCore.QSizeF(image.size())))
        if changed:
            if new_size:
                self.set_roi(None)
            self._apply_zoom()
            self.centerOn(self.sceneRect().center() if new_size else center)
        if self.fit_mode:
            self.fit_image()
        self.viewChanged.emit()

    def clear_image(self):
        self.action = None
        self.image_size = QtCore.QSize()
        self.image_item.setPixmap(QtGui.QPixmap())
        self.setSceneRect(QtCore.QRectF())
        self.set_roi(None)
        self.cursorChanged.emit(None)
        self.viewChanged.emit()

    def _apply_zoom(self):
        # Qt views use logical coordinates. 100% means one sensor pixel per
        # physical display pixel, including on Windows high-DPI screens.
        scale = self.zoom / self.viewport().devicePixelRatioF()
        self.setTransform(orientation_matrix(self.orientation).scale(scale, scale))

    def set_zoom(self, zoom, anchor=None):
        self.fit_mode = False
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        anchor = anchor or self.viewport().rect().center()
        before = self.mapToScene(anchor)
        self.zoom = min(32.0, max(.01, float(zoom)))
        self._apply_zoom()
        after = self.mapToScene(anchor)
        self.centerOn(self.mapToScene(self.viewport().rect().center()) + before - after)
        self.viewChanged.emit()

    def fit_image(self):
        self.fit_mode = True
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        if self.image_size.isEmpty():
            return
        extent = orientation_matrix(self.orientation).mapRect(self.sceneRect())
        self.zoom = max(.0001, min((self.viewport().width()-4)/extent.width(),
                                  (self.viewport().height()-4)/extent.height())) * self.viewport().devicePixelRatioF()
        self._apply_zoom()
        self.centerOn(self.sceneRect().center())
        self.viewChanged.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'fit_mode'):
            self.refresh_scale()

    def refresh_scale(self):
        if self.fit_mode:
            self.fit_image()
        else:
            self._apply_zoom()
        self.viewChanged.emit()

    def wheelEvent(self, event):
        if not self.image_size.isEmpty():
            delta = event.angleDelta().y() or event.pixelDelta().y()
            self.set_zoom(self.zoom * 1.2 ** (delta / 120), event.pos())
        event.accept()

    def set_draw_mode(self, enabled):
        self.cancel_edit()
        self.draw_mode = enabled
        self.setDragMode(self.NoDrag if enabled else self.ScrollHandDrag)
        self.viewport().setCursor(QtCore.Qt.CrossCursor if enabled else QtCore.Qt.OpenHandCursor)

    def set_roi(self, rect):
        self.roi = QtCore.QRect(rect) if rect is not None else None
        self.viewport().update()
        self.roiChanged.emit(self.roi)

    def _handles(self):
        r = QtCore.QRectF(self.roi)
        x0, y0, x1, y1 = r.left(), r.top(), r.right(), r.bottom()
        xc, yc = r.center().x(), r.center().y()
        return {'tl': (x0,y0), 't': (xc,y0), 'tr': (x1,y0), 'r': (x1,yc),
                'br': (x1,y1), 'b': (xc,y1), 'bl': (x0,y1), 'l': (x0,yc)}

    def _point(self, pos):
        p = self.mapToScene(pos)
        return QtCore.QPointF(min(self.image_size.width(), max(0, p.x())),
                             min(self.image_size.height(), max(0, p.y())))

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton or not self.draw_mode or self.image_size.isEmpty():
            return super().mousePressEvent(event)
        p = self.mapToScene(event.pos())
        if not self.sceneRect().contains(p):
            return
        self.editingStarted.emit()
        self.original_roi = QtCore.QRect(self.roi) if self.roi is not None else None
        self.start_point = self._point(event.pos())
        self.action = 'new'
        if self.roi is not None and self.show_roi:
            for handle, point in self._handles().items():
                screen = self.mapFromScene(QtCore.QPointF(*point))
                if (screen-event.pos()).manhattanLength() <= 12:
                    self.action = handle
                    break
            else:
                if QtCore.QRectF(self.roi).contains(p):
                    self.action = 'move'
        event.accept()

    def mouseMoveEvent(self, event):
        p = self.mapToScene(event.pos())
        self.cursorChanged.emit(p if self.sceneRect().contains(p) and not self.image_size.isEmpty() else None)
        if self.action is None:
            return super().mouseMoveEvent(event)
        end = self._point(event.pos())
        w, h = self.image_size.width(), self.image_size.height()
        if self.action == 'move':
            r = QtCore.QRect(self.original_roi)
            r.moveTo(min(w-r.width(), max(0, round(r.x()+end.x()-self.start_point.x()))),
                     min(h-r.height(), max(0, round(r.y()+end.y()-self.start_point.y()))))
        else:
            if self.action == 'new':
                a, b = self.start_point, end
            else:
                original = QtCore.QRectF(self.original_roi)
                a, b = original.topLeft(), original.bottomRight()
                if 'l' in self.action: a.setX(end.x())
                if 'r' in self.action: b.setX(end.x())
                if 't' in self.action: a.setY(end.y())
                if 'b' in self.action: b.setY(end.y())
            x0 = min(w-1, max(0, math.floor(min(a.x(), b.x()))))
            y0 = min(h-1, max(0, math.floor(min(a.y(), b.y()))))
            x1 = min(w, max(x0+1, math.ceil(max(a.x(), b.x()))))
            y1 = min(h, max(y0+1, math.ceil(max(a.y(), b.y()))))
            r = QtCore.QRect(x0, y0, x1-x0, y1-y0)
        self.set_roi(r)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.action is not None and event.button() == QtCore.Qt.LeftButton:
            self.mouseMoveEvent(event)
            self.action = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def cancel_edit(self):
        if self.action is not None:
            self.set_roi(self.original_roi)
            self.action = None

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape and self.action is not None:
            self.cancel_edit()
        elif event.key() == QtCore.Qt.Key_Delete:
            self.set_roi(None)
        else:
            super().keyPressEvent(event)

    def leaveEvent(self, event):
        self.cursorChanged.emit(None)
        super().leaveEvent(event)

    def drawForeground(self, painter, rect):
        if self.roi is None or not self.show_roi:
            return
        pen = QtGui.QPen(QtGui.QColor('#ff3b30'), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        if self.roi_ellipse:
            painter.drawEllipse(QtCore.QRectF(self.roi))
        else:
            painter.drawRect(QtCore.QRectF(self.roi))
        scale = math.hypot(self.transform().m11(), self.transform().m12())
        radius = 3.5 / max(scale, 1e-8)
        painter.setBrush(QtGui.QColor('#ff3b30'))
        for x, y in self._handles().values():
            painter.drawRect(QtCore.QRectF(x-radius, y-radius, radius*2, radius*2))
        painter.save()
        painter.resetTransform()
        anchor = self.mapFromScene(QtCore.QPointF(self.roi.x(), self.roi.y()))
        painter.drawText(max(4, anchor.x()), max(18, anchor.y()-8),
                         f'{self.roi_name}  {self.roi.width()} × {self.roi.height()} px')
        painter.restore()


class FrameNavigator(QtWidgets.QWidget):
    centerRequested = QtCore.pyqtSignal(object)

    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view
        self.image = QtGui.QImage()
        self.mapping = QtGui.QTransform()
        self.setFixedHeight(140)
        self.setMinimumWidth(0)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        view.viewChanged.connect(self.update)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor('#101b2b'))
        if self.image.isNull():
            painter.setPen(QtGui.QColor('#8ea4c0'))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, '全图缩略图\n等待相机画面')
            return
        orientation = orientation_matrix(self.view.orientation)
        bounds = orientation.mapRect(QtCore.QRectF(self.image.rect()))
        scale = min((self.width()-8)/bounds.width(), (self.height()-8)/bounds.height())
        self.mapping = QtGui.QTransform().translate(
            (self.width()-bounds.width()*scale)/2-bounds.left()*scale,
            (self.height()-bounds.height()*scale)/2-bounds.top()*scale).scale(scale, scale)
        # Explicit composition: orient raw pixels, then scale/place in thumbnail.
        self.mapping = orientation * self.mapping
        painter.setTransform(self.mapping)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.drawImage(0, 0, self.image)
        visible = self.view.mapToScene(self.view.viewport().rect()).intersected(
            QtGui.QPolygonF(QtCore.QRectF(self.image.rect())))
        pen = QtGui.QPen(QtGui.QColor('#48d9dc'), 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(72, 217, 220, 25))
        painter.drawPolygon(visible)

    def _navigate(self, pos):
        if not self.image.isNull():
            point = self.mapping.inverted()[0].map(QtCore.QPointF(pos))
            point.setX(min(self.image.width(), max(0, point.x())))
            point.setY(min(self.image.height(), max(0, point.y())))
            self.centerRequested.emit(point)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._navigate(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self._navigate(event.pos())
