"""Portable, data-only calibration bundles. No torch or model imports here."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .config import CalibrationConfig, write_json
from .patterns import TARGETS, illumination_mask
from .registration import correlation


# Fixed by the trained model. Acquisition/search settings remain adjustable.
MODEL_FIELDS = (
    'wavelength_nm', 'slm1_to_slm2_mm', 'slm2_to_camera_mm',
    'dmd_width', 'dmd_height', 'dmd_pitch_um', 'slm1_pitch_um', 'slm2_pitch_um',
    'slm1_grid_size', 'slm2_grid_size', 'slm1_sampling_um', 'slm2_sampling_um',
    'asm_padding_factor', 'asm_supersample_factor',
    'slm1_phase_resolution', 'slm2_phase_resolution', 'gray_inverted',
    'dmd_rotation', 'dmd_horizontal_flip', 'target_x_um', 'target_y_um',
    'slm1_lut', 'slm2_lut',
)
TRAINING_FIELDS = MODEL_FIELDS + ('steps', 'learning_rate', 'momentum', 'seed', 'device')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seal_bundle(root, manifest):
    """Record all generated assets so accidental edits cannot mix training runs."""
    root = Path(root)
    manifest['files'] = {path.relative_to(root).as_posix(): file_hash(path)
                         for path in sorted(root.rglob('*'))
                         if path.is_file() and path.name != 'manifest.json'}
    write_json(root / 'manifest.json', manifest)


def asset_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('标定包文件路径必须为包内相对路径。')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'标定包文件缺失或路径越界：{relative}')
    return path


@dataclass
class CalibrationBundle:
    root: Path
    config: CalibrationConfig
    manifest: dict

    def check_config(self, config):
        changed = [key for key in MODEL_FIELDS if getattr(config, key) != getattr(self.config, key)]
        if changed:
            raise ValueError('当前参数与标定包训练参数不一致，请重新导入或重新训练：' + '、'.join(changed))

    def check_illumination(self, measured_config):
        trained, measured = illumination_mask(self.config), illumination_mask(measured_config)
        difference = np.count_nonzero(trained != measured) / max(np.count_nonzero(trained | measured), 1)
        if difference > .005:
            raise ValueError(f'DMD 补偿后的照明口径与训练相差 {difference:.1%}，需按实测方向重新训练。')
        return float(difference)


def load_bundle(path):
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.name != 'manifest.json':
            raise ValueError('请选择标定训练目录中的 manifest.json；单张相位图缺少预测参考和训练参数。')
        root = root.parent
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 2:
        raise ValueError('标定包版本不支持；请用当前分层网格训练脚本重新生成。')
    if manifest.get('status') != 'trained':
        raise ValueError('标定包尚未完成训练。')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('标定包缺少文件校验记录，请重新训练导出。')
    for name, digest in files.items():
        if file_hash(asset_path(root, name)) != digest:
            raise ValueError(f'标定包文件校验失败，可能混用了其他训练结果：{name}')

    def required(name):
        if not isinstance(name, str) or Path(name).as_posix() not in files:
            raise ValueError(f'标定包缺少必要文件的校验记录：{name}')
        return asset_path(root, name)

    values = json.loads(required('config.json').read_text(encoding='utf-8'))
    config = CalibrationConfig(**values).validate()
    if values != manifest.get('config') or config.grid_metadata() != manifest.get('grids'):
        raise ValueError('标定包配置和网格元数据不一致。')
    if manifest.get('output_grid') != 'slm2':
        raise ValueError('标定包输出网格必须是 SLM2。')
    for key in ('slm1', 'slm2'):
        lut = np.load(required(key + '_lut_rad.npy'), allow_pickle=False)
        if lut.shape != (256,) or not np.issubdtype(lut.dtype, np.floating) or not np.isfinite(lut).all():
            raise ValueError(f'{key} 相位 LUT 不合法。')
    entries = manifest.get('patterns')
    if not isinstance(entries, list):
        raise ValueError('标定包缺少图案列表。')
    expected_pairs = {(stage, name) for stage in ('slm1', 'slm2', 'joint') for name in TARGETS}
    found = set()
    for entry in entries:
        pair = entry['stage'], entry['name']
        if pair not in expected_pairs or pair in found:
            raise ValueError('标定包图案重复或名称不支持。')
        found.add(pair)
        active = {'slm1', 'slm2'} if entry['stage'] == 'joint' else {entry['stage']}
        if set(entry['paths']) != active:
            raise ValueError(f'{pair} 的相位图组合不正确。')
        for key, name in entry['paths'].items():
            with Image.open(required(name)) as phase:
                if phase.mode != 'L' or phase.size != config.slm_shape(key)[::-1]:
                    raise ValueError(f'{name} 应为原生尺寸的 8 位灰度相位图。')
                phase.load()
        arrays = []
        for kind in ('expected', 'target'):
            array = np.load(required(entry[kind]), allow_pickle=False)
            if (array.shape != (config.output_grid_size,) * 2
                    or not np.issubdtype(array.dtype, np.floating)
                    or not np.isfinite(array).all() or np.min(array) < 0 or np.max(array) <= 0):
                raise ValueError(f'{pair} 的 {kind} 强度数组不合法。')
            arrays.append(array)
        score = correlation(*arrays)
        if not np.isfinite(entry['simulation_score']) or abs(score - entry['simulation_score']) > 1e-5:
            raise ValueError(f'{pair} 模拟分数与保存的参考不一致。')
    if found != expected_pairs:
        raise ValueError('自动标定需要 SLM1、SLM2、joint 各四组目标的完整训练包（共 12 组）。')
    return CalibrationBundle(root, config, manifest)


def copy_bundle(bundle, destination):
    """Take an audited snapshot of the input bundle for this hardware run."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for relative in bundle.manifest['files']:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset_path(bundle.root, relative), target)
    write_json(destination / 'manifest.json', bundle.manifest)
    return load_bundle(destination)
