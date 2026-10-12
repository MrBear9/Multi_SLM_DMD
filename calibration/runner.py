"""Bounded physical alignment with fixed probes, holdouts and rollback by the UI."""
from dataclasses import replace
from pathlib import Path
import time

import cv2
import numpy as np

from .config import write_json
from .bundle import load_bundle, copy_bundle
from .development import train_bundle
from .patterns import dmd_patterns, save_gray
from .registration import camera_map, centroid, compare_capture, fit_geometry, intensity, reference_coverage
from ..utils.image_transforms import ImageTransform


class CalibrationRunner:
    def __init__(self, config, root, settings, io, progress, cancelled, bundle_path=None):
        self.config, self.root = config.validate(), Path(root)
        self.settings = dict(settings)
        self.io, self.progress, self.cancelled = io, progress, cancelled
        self.counter = 0
        self.panels = {}
        self.bundle_path = bundle_path
        self.report = {'version': 1, 'status': 'running', 'started_at': time.time(),
                       'original_settings': settings, 'evaluations': [],
                       'scope': '普通 DMD、等倍率倒像 4f、两段自由空间 ASM；软件参数对准'}

    def check(self):
        if self.cancelled():
            raise InterruptedError('自动标定已取消。')

    def capture(self, paths, settings, label):
        self.check()
        self.progress(label)
        self.io('display', (paths, settings))
        image, metadata, frames = self.io('capture', self.config)
        self.check()
        folder = self.root / 'captures' / f'{self.counter:05d}'
        folder.mkdir(parents=True)
        for i, frame in enumerate(frames):
            success, encoded = cv2.imencode('.png', frame)
            if not success:
                raise RuntimeError('标定原始帧编码失败。')
            (folder / f'{i:02d}.png').write_bytes(encoded.tobytes())
        write_json(folder / 'capture.json', {'label': label, 'frames': metadata,
            'patterns': {k: str(v) for k, v in paths.items()}, 'settings': settings})
        self.counter += 1
        if image.dtype != np.uint8:
            raise ValueError('当前标定请将相机输出设为 bgr8 或 mono8，避免将 10/12 位容器误判为 16 位饱和阈值。')
        return image

    def evaluate(self, patterns, settings, label, disabled=None):
        results = []
        for item in patterns:
            paths = {'dmd': self.white, **self.flat}
            paths.update({key: self.bundle / path for key, path in item['paths'].items()
                          if key != disabled})
            image = self.capture(paths, settings, f'{label} · {item["stage"]}/{item["name"]}')
            expected = np.load(self.bundle / item['expected'], allow_pickle=False)
            result, reference = compare_capture(image, self.dark_image, expected, self.mapping, self.config)
            result.update(stage=item['stage'], name=item['name'], capture=self.counter - 1,
                          disabled=disabled)
            results.append(result)
            overlay = np.zeros((*reference.shape, 3), np.uint8)
            measured = np.maximum(intensity(image) - intensity(self.dark_image), 0)
            overlay[..., 1] = np.clip(measured / max(float(measured.max()), 1) * 255, 0, 255)
            overlay[..., 2] = np.clip(reference / max(float(reference.max()), 1e-8) * 255, 0, 255)
            ok, encoded = cv2.imencode('.png', overlay)
            if ok:
                (self.root / 'captures' / f'{self.counter - 1:05d}' / 'overlay.png').write_bytes(encoded.tobytes())
        score = float(np.mean([result['score'] for result in results]))
        self.report['evaluations'].append({'label': label, 'score': score, 'results': results,
            'parameters': {key: value for key, value in settings.items()
                           if key.endswith(('RotationSpinBox', 'FlipCheckBox', 'OffsetXSpinBox', 'OffsetYSpinBox'))}})
        write_json(self.root / 'report.json', self.report)
        return score

    def optimize(self, key, patterns):
        config = self.config
        base = dict(self.settings)
        best = dict(base)
        score = self.evaluate(patterns, best, f'{key} 初始状态')
        candidates = []
        if config.coarse_orientation:
            for flip in (False, True):
                for angle in range(0, 360, 30):
                    candidate = dict(base)
                    candidate[key + 'RotationSpinBox'] = ((base[key + 'RotationSpinBox'] + angle + 180) % 360) - 180
                    candidate[key + 'HorizontalFlipCheckBox'] = flip
                    candidate[key + 'VerticalFlipCheckBox'] = False
                    candidates.append(candidate)
        for candidate in candidates:
            if not self.candidate_fits(key, candidate):
                continue
            candidate_score = self.evaluate(patterns, candidate, f'{key} 搜索方向')
            if candidate_score > score + .002:
                best, score = candidate, candidate_score
        anchor_angle = best[key + 'RotationSpinBox']
        for round_index in range(config.search_rounds):
            for suffix, initial_step in (('OffsetXSpinBox', config.offset_radius),
                                         ('OffsetYSpinBox', config.offset_radius),
                                         ('RotationSpinBox', config.angle_radius)):
                step = initial_step / (2 ** round_index)
                if 'Offset' in suffix:
                    step = round(step)
                if not step:
                    continue
                current_best = dict(best)
                for sign in (-1, 1):
                    candidate = dict(current_best)
                    value = current_best[key + suffix] + sign * step
                    if 'Offset' in suffix:
                        if abs(value - base[key + suffix]) > config.offset_radius:
                            continue
                    else:
                        value = round(((value + 180) % 360) - 180, 1)
                        if abs(((value - anchor_angle + 180) % 360) - 180) > config.angle_radius:
                            continue
                    candidate[key + suffix] = value
                    if not self.candidate_fits(key, candidate):
                        continue
                    candidate_score = self.evaluate(patterns, candidate, f'{key} 精调 {round_index + 1}/{config.search_rounds}')
                    if candidate_score > score + .002:
                        best, score = candidate, candidate_score
        self.settings = best
        self.report[key] = {'training_score': score,
            'search_offset_radius_px': config.offset_radius,
            'note': '以固定图案的相机反馈搜索；不将相位屏偏移等同于光斑平移。'}
        if score < config.minimum_score:
            raise ValueError(f'{key} 相机匹配分数 {score:.3f} 未通过；请检查光束覆盖、传播距离、偏振和相位 LUT。')

    def candidate_fits(self, key, values):
        width, height = self.panels.get(key, (1920, 1080))
        w, h = ImageTransform.from_settings(values, key).output_size(self.config.slm_shape(key)[::-1])
        x = (width - w) // 2 + values[key + 'OffsetXSpinBox']
        y = (height - h) // 2 + values[key + 'OffsetYSpinBox']
        return x >= 0 and y >= 0 and x + w <= width and y + h <= height

    def measure_response(self, key, item):
        """Report an observed local response, without assuming rigid translation."""
        step = max(1, min(8, self.config.offset_radius // 2))
        paths = {'dmd': self.white, **self.flat,
                 **{k: self.bundle / p for k, p in item['paths'].items()}}
        columns, confidence = [], []
        for suffix in ('OffsetXSpinBox', 'OffsetYSpinBox'):
            images = []
            for sign in (-1, 1):
                values = dict(self.settings)
                values[key + suffix] += sign * step
                if not self.candidate_fits(key, values):
                    self.report[key]['local_response'] = {'observable_as_translation': False,
                        'note': '偏移响应探测超出面板，跳过。'}
                    return
                image = self.capture(paths, values, f'{key} 测量 {suffix} {sign * step:+d} px 响应')
                images.append(cv2.GaussianBlur(np.maximum(intensity(image) - intensity(self.dark_image), 0), (0, 0), 2))
            shift, response = cv2.phaseCorrelate(images[0], images[1])
            columns.append(np.asarray(shift) / (2 * step))
            confidence.append(float(response))
        matrix = np.column_stack(columns)
        condition = float(np.linalg.cond(matrix))
        observable = np.isfinite(matrix).all() and np.isfinite(condition) and condition < 20 and min(confidence) > .3
        self.report[key]['local_response'] = {
            'matrix_camera_px_per_panel_px': matrix.tolist() if np.isfinite(matrix).all() else None,
            'phase_correlation_response': confidence,
            'condition_number': condition if np.isfinite(condition) else None,
            'observable_as_translation': bool(observable),
            'note': '仅当前探测图案的局部响应；形变/低置信度时不用于反求偏移。'}

    def run(self):
        self.root.mkdir(parents=True, exist_ok=False)
        write_json(self.root / 'config.json', self.config.to_dict())
        try:
            imported = None
            if self.bundle_path is not None:
                imported = load_bundle(self.bundle_path)
                imported.check_config(self.config)
                imported = copy_bundle(imported, self.root / 'training')
                self.report['phase_source'] = {'kind': 'imported', 'path': str(self.bundle_path),
                    'files': imported.manifest['files']}
            else:
                self.report['phase_source'] = {'kind': 'development_training'}
            self.panels = self.io('panels', None) or {}
            self.progress('采集 DMD 参考和方向映射')
            self.dark, self.white, probes = dmd_patterns(self.config, self.root / 'dmd')
            self.flat = {key: save_gray(self.root / key / 'flat.png',
                np.zeros(self.config.slm_shape(key), np.uint8)) for key in ('slm1', 'slm2')}
            # Uniform grey 0 has no seam when placed on the full grey-0 canvas.
            for suffix, value in (('RotationSpinBox', 0), ('HorizontalFlipCheckBox', False),
                                  ('VerticalFlipCheckBox', False)):
                self.settings['dmd' + suffix] = value
            self.dark_image = self.capture({'dmd': self.dark, **self.flat}, self.settings, '采集背景')
            source, destination, missed = [], [], []
            for path, point in probes:
                image = self.capture({'dmd': path, **self.flat}, self.settings, f'DMD 定位 {path.stem}')
                try:
                    location = centroid(image, self.dark_image)
                except ValueError as exc:
                    missed.append({'point': point, 'reason': str(exc)})
                    continue
                source.append(point)
                destination.append(location)
            geometry = fit_geometry(source, destination, self.config.max_registration_error)
            geometry['missed_points'] = missed
            geometry['camera_um_per_dmd_pixel_xy'] = (np.linalg.norm(np.asarray(geometry['matrix'])[:, :2], axis=0)
                                                       * self.config.camera_pitch_um).tolist()
            self.report['dmd'] = geometry
            self.mapping = camera_map(self.config, geometry)
            self.report['grid_to_raw_camera'] = self.mapping.tolist()
            # Canonical compensation of DMD rotation/parity; the physical mapping
            # above remains expressed in native panel axes, before this command.
            self.settings['dmdRotationSpinBox'] = geometry['rotation_deg']
            self.settings['dmdHorizontalFlipCheckBox'] = geometry['horizontal_flip']
            self.settings['dmdRotationSpinBox'] = round(geometry['rotation_deg'], 1)
            aligned_source, aligned_camera = [], []
            for path, point in probes[:9]:
                image = self.capture({'dmd': path, **self.flat}, self.settings, '验证 DMD 方向补偿')
                try:
                    aligned_camera.append(centroid(image, self.dark_image))
                    aligned_source.append(point)
                except ValueError:
                    continue
            aligned = fit_geometry(aligned_source, aligned_camera, self.config.max_registration_error)
            axes = np.asarray(aligned['matrix'])[:, :2]
            if axes[0, 0] <= 0 or axes[1, 1] <= 0 or max(abs(axes[0, 1]), abs(axes[1, 0])) > .15 * min(axes[0, 0], axes[1, 1]):
                raise ValueError('DMD 方向补偿验证未通过；可能存在透视/剪切，当前旋转翻转控制无法充分校正。')
            self.report['dmd']['corrected_geometry'] = aligned
            center = np.array([(self.dark_image.shape[1] - 1) / 2,
                               (self.dark_image.shape[0] - 1) / 2])
            grid_center = np.linalg.solve(self.mapping[:, :2], center - self.mapping[:, 2])
            shift = (grid_center - (self.config.output_grid_size - 1) / 2) * self.config.output_sampling_um
            measured_config = replace(self.config, dmd_rotation=self.settings['dmdRotationSpinBox'],
                dmd_horizontal_flip=geometry['horizontal_flip'], target_x_um=float(shift[0]),
                target_y_um=float(shift[1]))
            request_path = self.root / 'training_request.json'
            write_json(request_path, measured_config.to_dict())
            # DMD orientation changes the illuminated aperture; train with this
            # exact commanded orientation, instead of applying it after training.
            self.io('display', ({'dmd': self.dark, **self.flat}, self.settings))
            self.bundle = self.root / 'training'
            if imported is None:
                self.config = measured_config
                manifest = train_bundle(self.config, self.bundle, self.progress, self.cancelled)
            else:
                self.progress('使用导入相位和固定预测参考，检查实测照明与目标可见范围')
                manifest = imported.manifest
                try:
                    self.report['illumination_difference'] = imported.check_illumination(measured_config)
                    for item in manifest['patterns']:
                        expected = np.load(self.bundle / item['expected'], allow_pickle=False)
                        reference_coverage(expected, self.mapping, self.dark_image.shape, self.config)
                except ValueError as exc:
                    raise ValueError(f'{exc}\n已保存实测训练参数：{request_path}\n在开发环境用此参数重新训练并导入；本轮不应用标定结果。') from exc
            if any(item['simulation_score'] < self.config.minimum_score for item in manifest['patterns']):
                raise ValueError('量化后的模拟图案未通过训练质量阈值；增加训练步数或调整光学模型后重试。')
            for key in ('slm1', 'slm2'):
                patterns = [item for item in manifest['patterns']
                            if item['stage'] == key and item['name'] in ('dots', 'arrow')]
                self.optimize(key, patterns)
                self.measure_response(key, patterns[0])
            # Fixed joint masks are now tested, including a never-fitted target.
            validations = {}
            for stage in ('slm1', 'slm2', 'joint'):
                held = [item for item in manifest['patterns']
                        if item['stage'] == stage and item['name'] in ('corners', 'holdout')]
                validations[stage] = self.evaluate(held, self.settings, f'{stage} 独立验证')
                if any(result['score'] < self.config.minimum_score for result in self.report['evaluations'][-1]['results']):
                    raise ValueError(f'{stage} 至少一张独立验证图未通过；本轮参数不应用。')
            if min(validations.values()) < self.config.minimum_score:
                raise ValueError('独立图案或双 SLM 联合验证未通过；本轮参数不应用。')
            joint = [item for item in manifest['patterns'] if item['stage'] == 'joint' and item['name'] == 'holdout']
            positive = self.evaluate(joint, self.settings, '联合正对照')
            controls = {key: self.evaluate(joint, self.settings, f'{key} 均匀相位反对照', disabled=key)
                        for key in ('slm1', 'slm2')}
            if any(positive - score < self.config.negative_control_margin for score in controls.values()):
                raise ValueError('联合图案无法明确区分两片 SLM 的作用，标定结果不够可信。')
            self.report.update(status='passed', validation=validations, negative_controls=controls,
                applied_settings=self.settings, config=self.config.to_dict(), finished_at=time.time(),
                unverified='相机视野外区域、机械光轴、相位 LUT 本身及其他输入尺寸未在本轮验证。')
            write_json(self.root / 'calibration.json', self.report)
            return self.report
        except Exception as exc:
            self.report.update(status='cancelled' if isinstance(exc, InterruptedError) else 'failed',
                               reason=str(exc), finished_at=time.time())
            raise
        finally:
            write_json(self.root / 'report.json', self.report)
