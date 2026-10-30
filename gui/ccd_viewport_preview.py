"""Render fresh raw camera pixels through a saved detail-page viewport."""
from dataclasses import dataclass
import json
import math

import numpy as np
from PyQt5 import QtCore, QtGui

from .configuration import camera_qimage


def frame_geometry(packet):
    raw, metadata, _ = packet
    identity = metadata.get('camera') or {}
    return json.dumps([list(raw.shape[:2]),identity.get('serial'),identity.get('model'),
                       metadata.get('camera_geometry')],sort_keys=True)


@dataclass
class DetailViewport:
    transform: QtGui.QTransform
    size: QtCore.QSize
    geometry: str

    @classmethod
    def capture(cls, view, packet):
        return cls(QtGui.QTransform(view.viewportTransform()),QtCore.QSize(view.viewport().size()),
                   frame_geometry(packet))

    def render(self, packet, size):
        if self.geometry != frame_geometry(packet) or self.size.isEmpty() or size.isEmpty():
            return None
        raw = packet[0]
        factor = min(size.width()/self.size.width(),size.height()/self.size.height())
        output = QtGui.QImage(max(1,round(self.size.width()*factor)),
                              max(1,round(self.size.height()*factor)),QtGui.QImage.Format_RGB32)
        output.fill(QtGui.QColor('#101b2b'))
        inverse, valid = self.transform.inverted()
        if not valid:
            return None
        bounds = inverse.mapRect(QtCore.QRectF(QtCore.QPointF(),QtCore.QSizeF(self.size)))
        h,w = raw.shape[:2]
        x0,y0 = max(0,math.floor(bounds.left())),max(0,math.floor(bounds.top()))
        x1,y1 = min(w,math.ceil(bounds.right())+1),min(h,math.ceil(bounds.bottom())+1)
        if x1<=x0 or y1<=y0:
            return output
        scale = math.hypot(self.transform.m11(),self.transform.m12())*factor
        stride = max(1,int(1/max(scale,1e-8)))
        crop = np.ascontiguousarray(raw[y0:y1:stride,x0:x1:stride])
        # Same full-frame brightness normalization as page three.
        maximum = float(raw.max()) if raw.dtype != np.uint8 else None
        image = camera_qimage(crop,maximum=maximum)
        painter = QtGui.QPainter(output)
        try:
            # Qt composes left to right: crop -> raw -> detail -> output.
            mapping = (QtGui.QTransform().scale(stride,stride)
                       * QtGui.QTransform.fromTranslate(x0,y0)
                       * self.transform * QtGui.QTransform().scale(factor,factor))
            painter.setTransform(mapping)
            painter.drawImage(0,0,image)
        finally:
            painter.end()
        return output
