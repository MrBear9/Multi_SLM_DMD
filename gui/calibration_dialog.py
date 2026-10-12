"""Calibration setup and progress; the main hardware window stays responsive."""
from dataclasses import replace
from pathlib import Path
import sys

from PyQt5 import QtCore, QtWidgets

from ..calibration.config import CalibrationConfig, MAX_GRID_SIZE, write_json
from ..calibration.bundle import load_bundle, TRAINING_FIELDS

GRID_INPUTS = {'dmd_width', 'dmd_height', 'dmd_pitch_um', 'slm1_grid_size', 'slm2_grid_size',
               'slm1_sampling_um', 'slm2_sampling_um'}


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, controller):
        super().__init__(controller.window)
        self.controller = controller
        self.development = not getattr(sys, 'frozen', False)
        self.running = False
        self.grid_adjustment_error = ''
        self.setWindowTitle('自动标定 · 光路对准')
        self.resize(820, 820)
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            '普通 DMD → 等倍率倒像 4f → SLM1 → 自由传播 → SLM2 → 自由传播 → 相机\n'
            '导入标定包后自动采集、搜索偏移/方向并验证；通过后应用，失败恢复。\n'
            '开发环境可进行 ASM＋SGD 训练。波长、距离需按实物核对；棋盘格只用于 demo。')
        description.setWordWrap(True)
        layout.addWidget(description)
        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(QtWidgets.QLabel('相位来源'))
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem('导入标定包', 'import')
        if self.development:
            self.source_combo.addItem('本地训练（开发环境）', 'development')
            self.source_combo.setCurrentIndex(1)
        source_row.addWidget(self.source_combo)
        self.bundle_path_edit = QtWidgets.QLineEdit()
        self.bundle_path_edit.setReadOnly(True)
        self.bundle_path_edit.setPlaceholderText('选择训练输出的 manifest.json')
        source_row.addWidget(self.bundle_path_edit, 1)
        self.import_button = QtWidgets.QPushButton('导入相位包…')
        self.import_button.clicked.connect(self.import_bundle)
        source_row.addWidget(self.import_button)
        layout.addLayout(source_row)
        self.grid_summary = QtWidgets.QLabel()
        self.grid_summary.setWordWrap(True)
        layout.addWidget(self.grid_summary)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        parameter_layout = QtWidgets.QVBoxLayout(content)
        parameter_layout.setContentsMargins(8, 8, 8, 8)
        parameter_layout.setSpacing(12)
        self.controls = {}
        labels = {}
        config = controller.config
        definitions = [
            ('wavelength_nm', '激光波长 / nm', 100, 2000, 1),
            ('slm1_to_slm2_mm', 'SLM1 → SLM2 / mm', .1, 10000, 2),
            ('slm2_to_camera_mm', 'SLM2 → 相机 / mm', .1, 10000, 2),
            ('dmd_width', 'DMD 有效宽度 / px', 8, 1920, 0),
            ('dmd_height', 'DMD 有效高度 / px', 8, 1080, 0),
            ('dmd_pitch_um', 'DMD 物理像素间距 / μm', .1, 100, 3),
            ('slm1_pitch_um', 'SLM1 物理像素间距 / μm', .1, 100, 3),
            ('slm2_pitch_um', 'SLM2 物理像素间距 / μm', .1, 100, 3),
            ('camera_pitch_um', '相机像素间距 / μm', .1, 100, 3),
            ('slm1_grid_size', 'SLM1 · ASM 网格边长', 32, MAX_GRID_SIZE, 0),
            ('slm1_sampling_um', 'SLM1 · 数值采样 / μm', .1, 100, 3),
            ('slm1_phase_resolution', 'SLM1 · 相位控制点边长', 8, 1920, 0),
            ('slm2_grid_size', 'SLM2 · ASM 网格边长', 32, MAX_GRID_SIZE, 0),
            ('slm2_sampling_um', 'SLM2 · 数值采样 / μm', .1, 100, 3),
            ('slm2_phase_resolution', 'SLM2 · 相位控制点边长', 8, 1920, 0),
            ('steps', '每张相位图 SGD 步数', 1, 20000, 0),
            ('learning_rate', 'SGD 学习率', .0001, 10, 4),
            ('settle_seconds', '切图后稳定时间 / s', .05, 30, 2),
            ('average_frames', '每次平均帧数', 1, 100, 0),
            ('discard_frames', '稳定后丢弃帧数', 1, 100, 0),
            ('capture_timeout', '每批采集超时 / s', 1, 120, 1),
            ('offset_radius', '偏移搜索半径 / 原生 px', 0, 512, 0),
            ('angle_radius', '精细角度步长 / °', 0, 45, 1),
            ('search_rounds', '精细搜索轮数', 1, 6, 0),
            ('minimum_score', '最低匹配分数', .01, .99, 2),
            ('minimum_coverage', '最低目标覆盖率', .1, .99, 2),
        ]
        for key, label, low, high, decimals in definitions:
            control = QtWidgets.QSpinBox() if decimals == 0 else QtWidgets.QDoubleSpinBox()
            if decimals:
                control.setDecimals(decimals)
            control.setRange(low, high)
            # Commit a typed number before resizing: intermediate digits such
            # as '64' while entering '640' must not trigger an auto-growth.
            control.setKeyboardTracking(False)
            control.setValue(getattr(config, key))
            control.setMinimumWidth(100)
            self.controls[key] = control
            labels[key] = label
            if key.endswith('_sampling_um'):
                control.setToolTip('用于 ASM 传播的数值采样。修改后自动扩展不足的网格；器件物理像素间距单独设置。')
            elif key.endswith('_grid_size'):
                control.setToolTip('包含周边计算区域的数值网格。范围不足时自动增大；已足够大的手动值会保留。')
            elif key.endswith('_phase_resolution'):
                control.setToolTip('每边 8～1920 个 SGD 相位控制点，可填 640。训练时重采样到原生有效尺寸；控制点比原生尺寸大时会缩小。')
        self.controls['camera_pitch_um'].setToolTip('用于相机像素与物理尺寸换算。')
        self.controls['angle_radius'].setToolTip('全角度粗搜索完成后，精细搜索使用的初始角度步长。')
        sections = (
            ('光路与器件', (
                ('wavelength_nm', 'dmd_pitch_um'),
                ('slm1_to_slm2_mm', 'slm2_to_camera_mm'),
                ('dmd_width', 'dmd_height'),
                ('slm1_pitch_um', 'slm2_pitch_um'),
            )),
            ('SLM 数值计算与训练', (
                ('slm1_grid_size', 'slm2_grid_size'),
                ('slm1_sampling_um', 'slm2_sampling_um'),
                ('slm1_phase_resolution', 'slm2_phase_resolution'),
                ('steps', 'learning_rate'),
            )),
            ('相机采集与对准搜索', (
                ('camera_pitch_um', 'capture_timeout'),
                ('settle_seconds', 'average_frames'),
                ('discard_frames', 'search_rounds'),
                ('offset_radius', 'angle_radius'),
                ('minimum_score', 'minimum_coverage'),
            )),
        )
        # Each side is a complete label/field pair. Keep matching SLM settings
        # on the same row and keyboard navigation in visual row order.
        previous_control = None
        for title, pairs in sections:
            # Parent groups before linking focus across them; unparented
            # groups are separate windows to QWidget.setTabOrder().
            group = QtWidgets.QGroupBox(title, content)
            grid = QtWidgets.QGridLayout(group)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(6)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(4, 1)
            grid.setColumnMinimumWidth(2, 16)
            for row_index, pair in enumerate(pairs):
                for side, key in enumerate(pair):
                    label = QtWidgets.QLabel(labels[key])
                    label.setBuddy(self.controls[key])
                    grid.addWidget(label, row_index, side * 3)
                    grid.addWidget(self.controls[key], row_index, side * 3 + 1)
                    if previous_control is not None:
                        self.setTabOrder(previous_control, self.controls[key])
                    previous_control = self.controls[key]
            parameter_layout.addWidget(group)
        options = QtWidgets.QHBoxLayout()
        for key, label in (('gray_inverted', '默认线性灰度相位反向'),
                           ('coarse_orientation', '搜索全角度和镜像（30° 步长）')):
            control = QtWidgets.QCheckBox(label)
            control.setChecked(getattr(config, key))
            self.controls[key] = control
            options.addWidget(control, 1)
        parameter_layout.addLayout(options)
        form = QtWidgets.QFormLayout()
        parameter_layout.addLayout(form)
        for key, label in (('slm1_lut', 'SLM1 相位 LUT（256 行 rad，可空）'),
                           ('slm2_lut', 'SLM2 相位 LUT（256 行 rad，可空）')):
            control = QtWidgets.QLineEdit(getattr(config, key))
            self.controls[key] = control
            form.addRow(label, control)
        self.output = QtWidgets.QLineEdit(str(controller.output))
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output)
        browse = QtWidgets.QPushButton('选择…')
        browse.clicked.connect(self.browse)
        output_row.addWidget(browse)
        form.addRow('标定输出目录', output_row)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        row = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton('载入参数')
        self.load_button.clicked.connect(self.load_config)
        row.addWidget(self.load_button)
        self.defaults_button = QtWidgets.QPushButton('恢复默认')
        self.defaults_button.clicked.connect(self.restore_defaults)
        row.addWidget(self.defaults_button)
        self.train_button = QtWidgets.QPushButton('仅训练相位图')
        self.train_button.setVisible(self.development)
        self.train_button.clicked.connect(lambda: self.start(True))
        row.addWidget(self.train_button)
        self.run_button = QtWidgets.QPushButton('开始自动标定')
        self.run_button.clicked.connect(lambda: self.start(False))
        row.addWidget(self.run_button)
        self.stop_button = QtWidgets.QPushButton('停止')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(controller.cancel)
        row.addWidget(self.stop_button)
        layout.addLayout(row)
        # Enter commits a parameter; it must not also launch a file picker or
        # start a calibration through QDialog's implicit default button.
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
        self.scroll = scroll
        self.source_combo.currentIndexChanged.connect(self.refresh_source_controls)
        for key, control in self.controls.items():
            if isinstance(control, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                control.valueChanged.connect(self.adjust_grid_sizes if key in GRID_INPUTS else self.refresh_grid_summary)
        self.refresh_source_controls()

    def values(self, validate=True):
        values = {}
        for key, control in self.controls.items():
            if self.source_combo.currentData() == 'import' and key in TRAINING_FIELDS:
                # Read-only spin boxes may round a more precise imported value.
                # Keep the exact training coordinates from the package.
                values[key] = getattr(self.controller.config, key)
                continue
            if isinstance(control, QtWidgets.QCheckBox):
                values[key] = control.isChecked()
            elif isinstance(control, QtWidgets.QLineEdit):
                values[key] = control.text().strip()
            else:
                values[key] = control.value()
        config = replace(self.controller.config, **values)
        return config.validate() if validate else config

    def refresh_grid_summary(self, *_):
        config = self.values(validate=False)
        hardware, grids, phase = [], [], []
        for key in ('slm1', 'slm2'):
            h, w = config.slm_shape(key)
            hardware.append(f'{key.upper()} {w}×{h}（{getattr(config, key + "_pitch_um"):g} μm）')
            n = config.grid_size_for(key)
            grids.append(f'{key.upper()} {n}×{n} / {config.sampling_um_for(key):g} μm / {config.grid_extent_um(key) / 1000:.3f} mm')
            ph, pw = config.phase_shape(key)
            phase.append(f'{key.upper()} {pw}×{ph} → {w}×{h}')
        summary = ('硬件相位图：' + '；'.join(hardware) + '\n'
                   + 'ASM 网格 / 采样 / 视野：' + '；'.join(grids) + '\n'
                   + '相位控制点 → 导出尺寸：' + '；'.join(phase))
        if self.source_combo.currentData() == 'development':
            summary += '\n网格范围不足时自动增大；已足够大的手动值会保留。'
        try:
            if self.grid_adjustment_error:
                raise ValueError(self.grid_adjustment_error)
            config.validate()
        except ValueError as exc:
            summary += '\n待调整：' + str(exc)
        self.grid_summary.setText(summary)

    def adjust_grid_sizes(self, *_):
        previous_error, self.grid_adjustment_error = self.grid_adjustment_error, ''
        if self.source_combo.currentData() == 'development' and not self.running:
            original = self.values(validate=False)
            try:
                fitted = original.fit_grids()
            except ValueError as exc:
                self.grid_adjustment_error = str(exc)
                if self.grid_adjustment_error != previous_error:
                    self.log.appendPlainText('无法自动扩展网格：' + str(exc))
            else:
                changes = []
                for key in ('slm1', 'slm2'):
                    before, after = original.grid_size_for(key), fitted.grid_size_for(key)
                    if before != after:
                        control = self.controls[key + '_grid_size']
                        with QtCore.QSignalBlocker(control):
                            control.setValue(after)
                        changes.append(f'{key.upper()} {before} → {after}')
                if changes:
                    self.log.appendPlainText('已自动增大 ASM 网格：' + '；'.join(changes))
        self.refresh_grid_summary()

    def apply_config(self, config, *, adjust_grids=True):
        self.controller.config = config
        for key, control in self.controls.items():
            value = getattr(config, key)
            with QtCore.QSignalBlocker(control):
                if isinstance(control, QtWidgets.QCheckBox):
                    control.setChecked(value)
                elif isinstance(control, QtWidgets.QLineEdit):
                    control.setText(value)
                else:
                    control.setValue(value)
        if adjust_grids:
            self.adjust_grid_sizes()
        else:
            self.grid_adjustment_error = ''
            self.refresh_grid_summary()

    def refresh_source_controls(self, *_):
        local = self.source_combo.currentData() == 'development'
        for key in TRAINING_FIELDS:
            if key in self.controls:
                self.controls[key].setEnabled(local)
        self.train_button.setEnabled(local and not self.running)
        self.bundle_path_edit.setEnabled(not local)
        self.adjust_grid_sizes()

    def set_bundle(self, path):
        bundle = load_bundle(path)
        config = replace(self.values(validate=False),
                         **{key: getattr(bundle.config, key) for key in TRAINING_FIELDS})
        self.apply_config(config, adjust_grids=False)
        self.controller.bundle_path = bundle.root / 'manifest.json'
        self.bundle_path_edit.setText(str(self.controller.bundle_path))
        self.source_combo.setCurrentIndex(0)
        self.refresh_source_controls()
        scores = [item['simulation_score'] for item in bundle.manifest['patterns']]
        self.log.appendPlainText(f'已导入 12 组标定相位，模拟分数 {min(scores):.3f}～{max(scores):.3f}。训练参数已锁定；采集与搜索参数仍可调整。')

    def import_bundle(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '导入标定训练包', '', '标定包 (manifest.json)')
        if path:
            try:
                self.set_bundle(path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, '标定包导入失败', str(exc))

    def restore_defaults(self):
        self.controller.bundle_path = None
        self.bundle_path_edit.clear()
        self.apply_config(CalibrationConfig())
        self.log.appendPlainText('已恢复分层默认：SLM1 512×512 / 13 μm；SLM2 1024×1024 / 6.4 μm。改变传播参数后需重新训练。')

    def start(self, training_only):
        try:
            self.adjust_grid_sizes()
            if self.grid_adjustment_error:
                raise ValueError(self.grid_adjustment_error)
            config = self.values()
            bundle_path = None
            if not training_only and self.source_combo.currentData() == 'import':
                bundle_path = self.controller.bundle_path
                if bundle_path is None:
                    raise ValueError('请先导入完整标定包中的 manifest.json。单张相位图缺少自动对准需要的参考信息。')
            if not self.output.text().strip():
                raise ValueError('请选择可写的输出目录。')
            output = Path(self.output.text()).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / 'last_config.json', config.to_dict())
            self.controller.start(config, output, training_only, bundle_path=bundle_path)
        except Exception as exc:
            self.log.appendPlainText(str(exc))
            QtWidgets.QMessageBox.warning(self, '无法开始标定', str(exc))

    def browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '标定输出目录', self.output.text())
        if path:
            self.output.setText(path)

    def load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '标定参数', '', 'JSON (*.json)')
        if not path:
            return
        try:
            local = self.source_combo.currentData() == 'development'
            config = CalibrationConfig.read(path, on_migration=self.log.appendPlainText,
                                            check_grid_coverage=not local)
            if self.source_combo.currentData() == 'import' and self.controller.bundle_path:
                load_bundle(self.controller.bundle_path).check_config(config)
            self.apply_config(config)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, '参数读取失败', str(exc))

    def set_running(self, running):
        self.running = running
        self.scroll.setEnabled(not running)
        for button in (self.run_button, self.train_button, self.load_button, self.defaults_button):
            button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.source_combo.setEnabled(not running)
        self.import_button.setEnabled(not running)
        self.refresh_source_controls()
        self.progress.setRange(0, 0 if running else 1)

    def reject(self):
        if self.controller.active:
            self.controller.cancel()
            self.log.appendPlainText('正在取消；等待当前训练步骤或 SDK 调用结束。')
        else:
            super().reject()

