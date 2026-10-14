"""Replace desktop display inputs without reconnecting; explicit test patterns."""
from functools import partial
from pathlib import Path
import tempfile
from uuid import uuid4

from PIL import Image, ImageDraw
from PyQt5 import QtCore, QtWidgets

from .configuration import input_path, read_values
from ..devices.zkwx_slm import controller as images


class LiveInputs:
    def __init__(self, window):
        self.window = window
        self.directory = None
        self.tests = {}
        for key in ('dmd', 'slm1', 'slm2'):
            getattr(window, key + 'ImportImagesButton').clicked.connect(partial(self.choose_inputs, key))
            getattr(window, key + 'TestPatternButton').clicked.connect(partial(self.choose_test, key))

    def field(self, key):
        return self.window.dmd_input_field() if key == 'dmd' else key + 'InputPathEdit'

    def network(self, key):
        return key == 'dmd' and self.window.f4321EnabledCheckBox.isChecked()

    def group(self, key):
        w = self.window
        if key in ('slm1', 'slm2') and w.synchronizeSlmsCheckBox.isChecked() and all(
                w.device_enabled(k) for k in ('slm1', 'slm2')):
            return ('slm1', 'slm2')
        return (key,)

    def check_idle(self):
        w = self.window
        if w.calibration.active or w.disconnecting or w.device_task is not None:
            raise ValueError('请等待标定、连接或上传操作完成。')

    def pause_inputs(self, keys):
        w = self.window
        for key in keys:
            w.pause(key)
        # Changing any optical pattern invalidates an outstanding exposure.
        if w.camera is not None:
            w.camera.submit('cancel_captures')
        if w.slm_capture_running():
            source = w.slm_capture_source()
            w.pause(source)
        if w.synchronizeDmdCcdCheckBox.isChecked():
            w.pause('dmd')

    def choose_inputs(self, key, *_):
        w = self.window
        if self.network(key):
            return
        self.pause_inputs(self.group(key))
        menu = QtWidgets.QMenu(w)
        single = menu.addAction('选择单张图像')
        folder = menu.addAction('选择图像目录')
        action = menu.exec_(getattr(w, key + 'ImportImagesButton').mapToGlobal(QtCore.QPoint(0, 0)))
        if action not in (single, folder):
            return
        selections = {}
        for target in self.group(key):
            current = getattr(w, self.field(target)).text()
            if action == folder:
                path = QtWidgets.QFileDialog.getExistingDirectory(w, target.upper() + '：选择图像目录', current)
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(w, target.upper() + '：选择图像', current,
                    'Images (*.png *.bmp *.jpg *.jpeg *.tif *.tiff)')
            if not path:
                return
            selections[target] = path
        self.import_safely(selections)

    def read_paths(self, key, text, values):
        w = self.window
        paths = images.collect_images(input_path(text), None, None)
        if not paths:
            raise ValueError('未找到可用图像。')
        native_size = None
        for path in paths:
            with Image.open(path) as image:
                image.verify()
                if key in ('slm1', 'slm2'):
                    valid_modes = {'L'} if key == 'slm2' else {'1', 'L', 'I', 'I;16'}
                    if image.mode not in valid_modes:
                        raise ValueError(f'{key.upper()} 相位图格式不正确：{path.name} ({image.mode})')
                    if native_size is not None and image.size != native_size:
                        raise ValueError(f'{key.upper()} 同一路序列的图像尺寸必须一致。')
                    native_size = image.size
            # Check every image against the actual connected panel and offsets.
            w.prepare_frame(key, 0, values, path=path)
        return paths

    def import_safely(self, selections):
        try:
            self.import_inputs(selections)
            return True
        except Exception as exc:
            w = self.window
            for key in selections:
                if w.settings is not None:
                    getattr(w, self.field(key)).setText(w.settings.get(self.field(key), ''))
            w.statusBar().showMessage('导入失败：' + str(exc))
            QtWidgets.QMessageBox.warning(w, '图像导入失败', str(exc))
            w.update_actions()
            return False

    def import_inputs(self, selections):
        self.check_idle()
        w = self.window
        if any(self.network(key) for key in selections):
            raise ValueError('F4321 请使用现有图像上传流程。')
        self.pause_inputs(selections)
        values = read_values(w)
        candidates = {key: self.read_paths(key, text, values) for key, text in selections.items()}
        if any(key in candidates for key in ('slm1', 'slm2')) and len(self.group('slm1')) == 2:
            pair = {key: candidates.get(key, w.paths[key]) for key in ('slm1', 'slm2')}
            if not pair['slm1'] or [p.stem for p in pair['slm1']] != [p.stem for p in pair['slm2']]:
                raise ValueError('双 SLM 新序列的名称和数量不匹配；请用“导入图像…”一起选择两路匹配输入。')
            # If only one path changed, also display frame 1 of its paired device.
            candidates.update(pair)
        prepared = {key: w.prepare_frame(key, 0, values, path=paths[0])
                    for key, paths in candidates.items()}
        attempted = []
        try:
            for key, paths in candidates.items():
                if key in w.devices:
                    attempted.append(key)
                    w.devices[key].show(paths[0], values, calibration=True)
        except Exception as failure:
            restore_errors = []
            for key in attempted:
                try:
                    self.restore_display(key, values)
                except Exception as exc:
                    getattr(w, key + 'StateLabel').setText('● 画面恢复失败，请重新连接')
                    restore_errors.append(f'{key.upper()} 恢复失败：{exc}')
            if restore_errors:
                raise RuntimeError(f'导入显示失败：{failure}；' + '；'.join(restore_errors)) from failure
            raise
        for key, paths in candidates.items():
            w.paths[key], w.indices[key] = paths, 0
            if key in w.devices and key != 'dmd':
                with Image.open(paths[0]) as image:
                    w.devices[key].source_size = image.size
            spin = getattr(w, key + 'FrameSpinBox')
            with QtCore.QSignalBlocker(spin):
                spin.setRange(1, len(paths))
                spin.setValue(1)
            getattr(w, key + 'TotalFramesLabel').setText(f'/ {len(paths)}')
            self.tests.pop(key, None)
            w.publish_frame(key, prepared[key])
        for key, text in selections.items():
            getattr(w, self.field(key)).setText(text)
            if w.settings is not None:
                w.settings[self.field(key)] = text
        w.update_actions()
        w.statusBar().showMessage('已导入并显示第一张图像；未触发采集。')

    def initialize(self):
        w = self.window
        keys = [key for key in w.devices if not self.network(key)]
        for key in keys:
            w.paths[key] = []
            w.indices[key] = 0
            getattr(w, key + 'TotalFramesLabel').setText('/ —')
            self.show_test(key, '全黑', initialize=True)
        handled = set()
        for key in keys:
            group = tuple(k for k in self.group(key) if k in keys)
            if key in handled:
                continue
            handled.update(group)
            selections = {k: getattr(w, self.field(k)).text() for k in group}
            if not any(text.strip() for text in selections.values()):
                continue
            try:
                self.import_inputs(selections)
            except Exception as exc:
                for target in group:
                    label = getattr(w, target + 'FrameInfoLabel')
                    label.setText('输入未加载 · 保持全黑（悬停查看原因）')
                    label.setToolTip(str(exc))

    def choose_test(self, key, *_):
        w = self.window
        menu = QtWidgets.QMenu(w)
        actions = {menu.addAction(name): name for name in ('全黑', '全白', '棋盘格')}
        restore = menu.addAction('恢复当前输入图像')
        restore.setEnabled(bool(w.paths[key]))
        action = menu.exec_(getattr(w, key + 'TestPatternButton').mapToGlobal(QtCore.QPoint(0, 0)))
        if action == restore:
            w.guard(partial(self.restore_input, key))
        elif action in actions:
            w.guard(partial(self.show_test, key, actions[action]))

    def show_test(self, key, name, initialize=False, restoring=False):
        if not restoring:
            self.check_idle()
        w = self.window
        if key not in w.devices or self.network(key):
            raise ValueError('请先连接普通显示器 DMD 或 SLM。')
        self.pause_inputs(self.group(key))
        if self.directory is None:
            self.directory = tempfile.TemporaryDirectory(prefix='optical_test_')
        display = w.devices[key].display
        path = Path(self.directory.name) / f'{key}_{uuid4().hex}.bmp'
        image = Image.new('L', (display.width, display.height), 255 if name == '全白' else 0)
        if name == '棋盘格':
            draw = ImageDraw.Draw(image)
            for y in range(0, image.height, 64):
                for x in range(0, image.width, 64):
                    if (x // 64 + y // 64) % 2:
                        draw.rectangle((x, y, x + 63, y + 63), fill=255)
        image.save(path)
        image.close()
        values = self.test_values(key, read_values(w))
        prepared = w.prepare_frame(key, 0, values, path=path)
        w.devices[key].show(path, values, calibration=True)
        w.publish_frame(key, prepared)
        self.tests[key] = (name, path)
        getattr(w, key + 'FrameInfoLabel').setText(
            '未加载输入图像 · 全屏零灰度' if initialize else f'测试图案：{name} · 未触发采集')
        w.update_actions()

    @staticmethod
    def test_values(key, values):
        values = dict(values)
        values.update({key + 'RotationSpinBox': 0.0, key + 'HorizontalFlipCheckBox': False,
                       key + 'VerticalFlipCheckBox': False, key + 'OffsetXSpinBox': 0,
                       key + 'OffsetYSpinBox': 0, 'slm2BackgroundSpinBox': 0})
        if key == 'dmd':
            values['dmdPixelModeComboBox'] = '保持原图'
        return values

    def current_path(self, key):
        if key in self.tests:
            return self.tests[key][1]
        paths = self.window.paths[key]
        return paths[self.window.indices[key]] if paths else ''

    def restore_display(self, key, values):
        w = self.window
        if key in self.tests or not w.paths[key]:
            self.show_test(key, self.tests.get(key, ('全黑', None))[0], restoring=True)
        else:
            path = w.paths[key][w.indices[key]]
            w.devices[key].show(path, values, calibration=True)
            w.render_frame(key, w.indices[key])

    def restore_input(self, key):
        self.check_idle()
        self.pause_inputs(self.group(key))
        self.window.show_frame(key, self.window.indices[key])
        self.window.update_actions()

    def update_actions(self):
        w = self.window
        available = not (w.disconnecting or w.device_task is not None or w.calibration.active)
        for key in ('dmd', 'slm1', 'slm2'):
            ordinary = not self.network(key)
            getattr(w, key + 'ImportImagesButton').setEnabled(available and ordinary and w.device_enabled(key))
            getattr(w, key + 'TestPatternButton').setEnabled(available and ordinary and key in w.devices)

    def close(self):
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
