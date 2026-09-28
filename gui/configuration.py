"""JSON configuration and image conversion shared by the Qt window."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

from ..utils.paths import resolve_path

SCHEMA_VERSION = 1
EDITORS = (QtWidgets.QLineEdit, QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox,
           QtWidgets.QComboBox, QtWidgets.QCheckBox)


def editors(window):
    return {w.objectName(): w for cls in EDITORS for w in window.findChildren(cls)
            if w.objectName() and not w.objectName().startswith('qt_')
            and not w.objectName().endswith('FrameSpinBox')}


def read_values(window):
    result = {}
    for name, w in editors(window).items():
        if isinstance(w, QtWidgets.QLineEdit):
            result[name] = w.text()
        elif isinstance(w, QtWidgets.QCheckBox):
            result[name] = w.isChecked()
        elif isinstance(w, QtWidgets.QComboBox):
            result[name] = w.currentText()
        else:
            result[name] = w.value()
    return result


def apply_values(window, values):
    """Validate the entire document before changing any widget."""
    if not isinstance(values, dict):
        raise ValueError('配置 values 必须是 JSON 对象。')
    controls = editors(window)
    for name, value in values.items():
        if name not in controls:
            raise ValueError(f'未知配置字段：{name}')
        w = controls[name]
        if isinstance(w, QtWidgets.QLineEdit):
            valid = isinstance(value, str)
        elif isinstance(w, QtWidgets.QCheckBox):
            valid = isinstance(value, bool)
        elif isinstance(w, QtWidgets.QComboBox):
            valid = isinstance(value, str) and w.findText(value) >= 0
        else:
            valid = type(value) in (int, float) and w.minimum() <= value <= w.maximum()
            if isinstance(w, QtWidgets.QSpinBox):
                valid = valid and type(value) is int
        if not valid:
            raise ValueError(f'配置字段 {name} 的值无效：{value!r}')
    for name, value in values.items():
        w = controls[name]
        if isinstance(w, QtWidgets.QLineEdit):
            w.setText(value)
        elif isinstance(w, QtWidgets.QCheckBox):
            w.setChecked(value)
        elif isinstance(w, QtWidgets.QComboBox):
            w.setCurrentText(value)
        else:
            w.setValue(value)


def write_configuration(path: Path, values: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps({'version': SCHEMA_VERSION, 'values': values},
                                        ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_configuration(path: Path):
    document = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict) or document.get('version') != SCHEMA_VERSION:
        raise ValueError('不支持的配置版本。')
    return document.get('values')


def input_path(value: str) -> Path:
    if not value.strip():
        raise ValueError('请填写输入路径。')
    return resolve_path(Path(value.strip()).expanduser())


def pil_qimage(path: Path) -> QtGui.QImage:
    with Image.open(path) as image:
        if image.mode.startswith('I;16'):
            data = np.array(image, dtype=np.uint16, order='C')
            return QtGui.QImage(data.data, data.shape[1], data.shape[0], data.strides[0],
                                QtGui.QImage.Format_Grayscale16).copy()
        image = image.convert('RGBA')
        data = image.tobytes()
        return QtGui.QImage(data, image.width, image.height, image.width * 4,
                            QtGui.QImage.Format_RGBA8888).copy()


def camera_qimage(image: np.ndarray) -> QtGui.QImage:
    # Preview-only conversion. Capture files retain the original frame values.
    if image.dtype != np.uint8:
        maximum = float(image.max()) if image.size else 1.0
        image = np.clip(image.astype(np.float32) * (255.0 / max(maximum, 1)), 0, 255).astype(np.uint8)
    if image.ndim == 2:
        data = np.ascontiguousarray(image)
        fmt = QtGui.QImage.Format_Grayscale8
    else:
        data = np.ascontiguousarray(image[:, :, :3][:, :, ::-1])
        fmt = QtGui.QImage.Format_RGB888
    return QtGui.QImage(data.data, data.shape[1], data.shape[0], data.strides[0], fmt).copy()


def set_preview(label, image, fit=True):
    pixmap = QtGui.QPixmap.fromImage(image)
    if fit:
        pixmap = pixmap.scaled(label.contentsRect().size(), QtCore.Qt.KeepAspectRatio,
                              QtCore.Qt.FastTransformation)
    else:
        # Crop explicitly so a native-sized QPixmap cannot enlarge the layout.
        width, height = min(pixmap.width(), label.width()), min(pixmap.height(), label.height())
        pixmap = pixmap.copy((pixmap.width() - width) // 2, (pixmap.height() - height) // 2,
                             width, height)
    label.setPixmap(pixmap)
