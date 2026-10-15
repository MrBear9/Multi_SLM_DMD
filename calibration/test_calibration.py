"""Calibration checks with synthetic optics and fake hardware; no vendor DLLs."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
import subprocess
import sys
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from tools.calibration.acquisition import FrameBatch
from tools.calibration.config import CalibrationConfig, write_json
from tools.calibration.bundle import load_bundle, seal_bundle, copy_bundle
from tools.calibration.patterns import save_gray, target_intensity
from tools.calibration.registration import camera_map, centroid, compare_capture, fit_geometry
from tools.calibration.runner import CalibrationRunner
from tools.utils.image_transforms import ImageTransform


def empirical_config(**overrides):
    """Previously saved numerical parameters remain supported unchanged."""
    values = dict(slm1_grid_size=512, slm2_grid_size=1024,
                  slm1_sampling_um=13, slm2_sampling_um=6.4)
    values.update(overrides)
    return CalibrationConfig(**values)


class CalibrationTests(unittest.TestCase):
    def test_raw_batch_rejects_old_duplicate_and_discarded_frames(self):
        batch = FrameBatch(10, 20, 2, 1)
        image = np.full((4, 4), 8, np.uint8)
        batch.offer(image, {'frame_id': 1}, 9, 11)
        batch.offer(image, {'frame_id': 2}, 11, 12)
        batch.offer(image, {'frame_id': 2}, 12, 13)
        self.assertFalse(batch.frames)
        batch.offer(image, {'frame_id': 3}, 13, 14)
        image[:] = 16
        batch.offer(image, {'frame_id': 4}, 14, 15)
        average, metadata = batch.result()
        self.assertTrue(np.all(average == 12))
        self.assertEqual([m['frame_id'] for m in metadata], [3, 4])

    def test_raw_batch_timeout_and_shape_change(self):
        batch = FrameBatch(0, 2, 2, 0)
        batch.offer(np.zeros((4, 4), np.uint8), {}, 0, 1)
        batch.offer(np.zeros((5, 4), np.uint8), {}, 1, 1.5)
        with self.assertRaisesRegex(RuntimeError, '尺寸'):
            batch.result()
        batch = FrameBatch(0, 2, 2, 0)
        batch.offer(np.zeros((4, 4), np.uint8), {}, 0, 3)
        with self.assertRaisesRegex(RuntimeError, '超时'):
            batch.result()

    def test_affine_recovers_reflection_angle_and_rejects_non_imaging(self):
        y, x = np.mgrid[0:4, 0:4]
        source = np.c_[x.ravel(), y.ravel()] * 20
        angle = np.radians(37)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        matrix = 1.2 * rotation @ np.diag([-1, 1])
        destination = source @ matrix.T + [100, 40]
        geometry = fit_geometry(source, destination, .1)
        transform = ImageTransform(geometry['rotation_deg'], geometry['horizontal_flip'])
        theta = np.radians(transform.angle)
        correction = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.diag([-1, 1])
        np.testing.assert_allclose(matrix @ correction, np.eye(2) * 1.2, atol=1e-6)
        destination[1] += 20
        with self.assertRaises(ValueError):
            fit_geometry(source, destination, .1)

    def test_clipped_or_missing_probe_not_used(self):
        image = np.zeros((80, 80), np.uint8)
        cv2.circle(image, (0, 40), 5, 200, -1)
        with self.assertRaises(ValueError):
            centroid(image, np.zeros_like(image))
        with self.assertRaises(ValueError):
            centroid(np.zeros_like(image), np.zeros_like(image))

    def test_model_units_and_coverage(self):
        config = CalibrationConfig()
        self.assertEqual(config.slm_shape('slm1'), (432, 432))
        self.assertEqual(config.slm_shape('slm2'), (768, 768))
        geometry = {'matrix': [[-1, 0, 640], [0, -1, 640]]}
        mapping = camera_map(config, geometry)
        self.assertGreater(mapping[0, 0], 0)
        target = target_intensity(config, 'dots')
        with self.assertRaisesRegex(ValueError, '覆盖率'):
            compare_capture(np.zeros((32, 32), np.uint8), np.zeros((32, 32), np.uint8), target, mapping, config)

    def test_training_exports_native_uint8_and_quantized_simulation(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional torch not installed')
        from tools.calibration.train import train_bundle
        config = CalibrationConfig(dmd_width=32, dmd_height=32,
            slm1_grid_size=48, slm2_grid_size=64, slm1_sampling_um=12, slm2_sampling_um=8,
            slm1_phase_resolution=16, slm2_phase_resolution=24, steps=3, device='cpu',
            slm1_to_slm2_mm=1, slm2_to_camera_mm=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'training'
            result = train_bundle(config, root, stages=('joint',), targets=('dots',))
            self.assertEqual(result['status'], 'trained')
            self.assertEqual(result['grids']['slm1']['asm_shape_hw'], [48, 48])
            self.assertEqual(result['grids']['slm2']['phase_parameter_shape_hw'], [24, 24])
            for key in ('slm1', 'slm2'):
                with Image.open(root / result['patterns'][0]['paths'][key]) as image:
                    self.assertEqual(image.mode, 'L')
                    self.assertEqual(image.size, config.slm_shape(key)[::-1])
            expected = np.load(root / result['patterns'][0]['expected'])
            self.assertEqual(expected.shape, (64, 64))
            self.assertTrue(np.isfinite(expected).all())
            with self.assertRaises(FileExistsError):
                train_bundle(config, root, stages=(), targets=())

    def test_cancelled_training_does_not_publish_trained_bundle(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional torch not installed')
        from tools.calibration.train import train_bundle
        config = CalibrationConfig(dmd_width=32, dmd_height=32,
            slm1_grid_size=64, slm2_grid_size=64, slm1_phase_resolution=8,
            slm2_phase_resolution=8, steps=1, device='cpu')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InterruptedError):
                train_bundle(config, Path(directory) / 'training', cancelled=lambda: True)


class GridTests(unittest.TestCase):
    def test_layer_defaults_and_hardware_sizes_are_distinct(self):
        from models.SLM.physical_defaults import DEFAULT_SLM_LAYER_PROFILES, SLM_PROFILES
        config = CalibrationConfig().validate()
        for index, key in enumerate(('slm1', 'slm2'), 1):
            profile = SLM_PROFILES[DEFAULT_SLM_LAYER_PROFILES[index]]
            self.assertAlmostEqual(config.sampling_um_for(key), profile['hardware_pixel_pitch'] * 1e6)
            self.assertAlmostEqual(getattr(config, key + '_pitch_um'), profile['hardware_pixel_pitch'] * 1e6)
        self.assertEqual((config.slm1_sampling_um, config.slm2_sampling_um), (8, 4.5))
        self.assertEqual((config.slm1_grid_size, config.slm2_grid_size), (704, 1152))
        self.assertEqual(config, CalibrationConfig(slm1_grid_size=432, slm2_grid_size=768).fit_grids())
        self.assertEqual((config.slm1_pitch_um, config.slm2_pitch_um), (8, 4.5))
        self.assertEqual(config.slm_shape('slm1'), (432, 432))
        self.assertEqual(config.slm_shape('slm2'), (768, 768))
        self.assertEqual(config.phase_shape('slm1'), (128, 128))
        self.assertNotEqual(config.slm1_grid_size, config.slm2_grid_size)

    def test_rejects_insufficient_propagation_field_of_view(self):
        config = CalibrationConfig()
        for key in ('slm1_grid_size', 'slm2_grid_size'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, '视野过小'):
                replace(config, **{key: 32}).validate()
        with self.assertRaisesRegex(ValueError, '覆盖 SLM2'):
            replace(config, slm2_grid_size=2048).validate()

    def test_finer_sampling_grows_only_insufficient_grids(self):
        original = CalibrationConfig(slm1_sampling_um=7)
        fitted = original.fit_grids()
        self.assertEqual((fitted.slm1_grid_size, fitted.slm2_grid_size), (768, 1152))
        self.assertEqual(original.slm1_grid_size, 704)
        self.assertEqual(fitted.slm1_sampling_um, 7)
        self.assertEqual(fitted.fit_grids(), fitted)

    def test_old_sampling_parameters_still_expand_when_needed(self):
        original = empirical_config(slm1_sampling_um=11)
        fitted = original.fit_grids()
        self.assertEqual((fitted.slm1_grid_size, fitted.slm2_grid_size), (640, 1024))
        self.assertEqual(original.slm1_grid_size, 512)
        self.assertEqual(fitted.slm1_sampling_um, 11)
        self.assertEqual(fitted.fit_grids(), fitted)
        fitted = empirical_config(slm2_sampling_um=4.5).fit_grids()
        self.assertEqual((fitted.slm1_grid_size, fitted.slm2_grid_size), (512, 1152))

    def test_larger_second_grid_grows_first_grid(self):
        fitted = CalibrationConfig(slm2_grid_size=2048).fit_grids()
        self.assertEqual((fitted.slm1_grid_size, fitted.slm2_grid_size), (1152, 2048))

    def test_sufficient_manual_sizes_are_preserved_without_rounding(self):
        for config in (CalibrationConfig(slm1_grid_size=961, slm2_grid_size=1536),
                       CalibrationConfig(slm1_grid_size=596, slm1_sampling_um=11)):
            with self.subTest(config=config):
                self.assertEqual(config.fit_grids(), config)

    def test_larger_aperture_grows_both_grids(self):
        fitted = CalibrationConfig(dmd_width=1280, dmd_height=720).fit_grids()
        self.assertEqual((fitted.slm1_grid_size, fitted.slm2_grid_size), (1344, 2304))
        fitted.validate()

    def test_impossible_grid_does_not_partially_adjust_parameters(self):
        original = CalibrationConfig(slm1_sampling_um=.1, slm2_sampling_um=4.5)
        with self.assertRaisesRegex(ValueError, 'SLM1.*51840.*4096'):
            original.fit_grids()
        self.assertEqual((original.slm1_grid_size, original.slm2_grid_size), (704, 1152))
        with self.assertRaisesRegex(ValueError, '必须为有限正数'):
            CalibrationConfig(slm1_sampling_um=float('nan')).fit_grids()

    def test_old_configuration_preserves_values_and_notifies_user(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            write_json(path, {'grid_size': 768, 'sampling_um': 8, 'phase_resolution': 64})
            notices = []
            config = CalibrationConfig.read(path, on_migration=notices.append)
            self.assertEqual(len(notices), 1)
            self.assertIn('原值', notices[0])
            for key in ('slm1', 'slm2'):
                self.assertEqual(config.grid_size_for(key), 768)
                self.assertEqual(config.sampling_um_for(key), 8)
                self.assertEqual(config.phase_shape(key), (64, 64))
            self.assertNotIn('sampling_um', config.to_dict())
            config = replace(config, slm1_sampling_um=13, slm2_sampling_um=6.4)
            # Keep enough SLM2 field of view for the default aperture.
            config = replace(config, slm2_grid_size=1024)
            write_json(path, config.to_dict())
            self.assertEqual(CalibrationConfig.read(path), config)
            write_json(path, {'sampling_um': 8, 'slm1_sampling_um': 13})
            with self.assertRaisesRegex(ValueError, '冲突'):
                CalibrationConfig.read(path)

    def test_camera_mapping_preserves_positions_across_output_grid_choices(self):
        geometry = {'matrix': [[-.8, .2, 500], [.1, -.9, 600]]}
        configs = [CalibrationConfig(), replace(CalibrationConfig(), slm1_grid_size=768),
                   replace(CalibrationConfig(), slm1_grid_size=832, slm2_grid_size=2048, slm2_sampling_um=3.2)]
        position_um = np.array([140., -270.])
        mapped = []
        for config in configs:
            config.validate()
            point = position_um / config.output_sampling_um + (config.output_grid_size - 1) / 2
            mapped.append(camera_map(config, geometry) @ np.r_[point, 1])
        np.testing.assert_allclose(mapped, np.tile(mapped[0], (3, 1)), atol=1e-10)
        # Changing only SLM1 does not resize or move the final camera target.
        np.testing.assert_array_equal(target_intensity(configs[0], 'dots'),
                                      target_intensity(configs[1], 'dots'))


class ComplexFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('optional torch not installed')
        cls.torch = torch

    def test_resampling_keeps_complex_values_at_physical_coordinates(self):
        from tools.calibration.resampling import ComplexGridResampler
        torch = self.torch
        source_count, source_pitch = 65, 3.
        count, pitch = 80, 2.
        axis = (torch.arange(source_count) - (source_count - 1) / 2) * source_pitch
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        # Linear complex field has an exact analytic answer under bilinear
        # interpolation; resizing the image to 80 pixels would fail this check.
        source = torch.complex(2 + xx / 100, 1 + yy / 100)[None, None].requires_grad_()
        resampler = ComplexGridResampler(source_count, source_pitch, count, pitch)
        output = resampler(source)
        dest = (torch.arange(count) - (count - 1) / 2) * pitch
        dy, dx = torch.meshgrid(dest, dest, indexing='ij')
        expected = torch.complex(2 + dx / 100, 1 + dy / 100)[None, None]
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
        output.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(source.grad.abs().sum().item(), 0)
        identity = ComplexGridResampler(source_count, source_pitch, source_count, source_pitch)
        self.assertIs(identity(source), source)

    def test_coarser_sampling_rejects_above_nyquist_component(self):
        from tools.calibration.resampling import ComplexGridResampler
        torch = self.torch
        x = torch.arange(64)
        # 24/64 cycles per source pixel exceeds destination Nyquist (1/4).
        carrier = torch.exp(2j * torch.pi * x * 24 / 64)
        field = (1 + carrier[None, :].expand(64, -1))[None, None]
        result = ComplexGridResampler(64, 1, 32, 2)(field)
        torch.testing.assert_close(result, torch.ones_like(result), atol=1e-5, rtol=1e-5)

    def test_each_asm_uses_its_own_sampling_and_both_phase_grids_receive_gradients(self):
        from tools.calibration.model import CalibrationModel
        torch = self.torch
        torch.manual_seed(7)
        config = CalibrationConfig(dmd_width=32, dmd_height=32, slm1_grid_size=48,
            slm2_grid_size=64, slm1_sampling_um=12, slm2_sampling_um=8,
            slm1_phase_resolution=16, slm2_phase_resolution=24,
            slm1_to_slm2_mm=1, slm2_to_camera_mm=1)
        model = CalibrationModel(config)
        parameters = {key: torch.randn(1, 1, *config.phase_shape(key), requires_grad=True)
                      for key in ('slm1', 'slm2')}
        for index, key in enumerate(('slm1', 'slm2')):
            self.assertEqual(model.propagators[index].input_resolution, (config.grid_size_for(key),) * 2)
            self.assertEqual(model.propagators[index].pixel_size, (config.sampling_um_for(key) * 1e-6,) * 2)
        phases = {key: model.native_phase(value, key) for key, value in parameters.items()}
        prediction = model(phases)
        self.assertEqual(tuple(prediction.shape), (1, 1, 64, 64))
        target = torch.from_numpy(target_intensity(config, 'dots'))
        (prediction - target).square().mean().backward()
        for key, parameter in parameters.items():
            self.assertEqual(tuple(phases[key].shape[-2:]), config.slm_shape(key))
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


class FakeOpticalBench:
    def __init__(self, config):
        self.config = config
        self.paths, self.values = {}, {}
        self.transform = np.array([[-2., 0, 95], [0, -2., 95]])
        self.camera_shape = (128, 128)
        self.mapping = camera_map(config, {'matrix': self.transform.tolist()})

    def io(self, action, payload):
        if action == 'panels':
            return {key: (1920, 1080) for key in ('dmd', 'slm1', 'slm2')}
        if action == 'display':
            self.paths, self.values = payload
            return
        if action != 'capture':
            raise AssertionError(action)
        path = self.paths['dmd']
        if path.name.startswith('probe') or path.name == 'dark.png':
            with Image.open(path) as source:
                with ImageTransform.from_settings(self.values, 'dmd').apply_pil(source) as transformed:
                    frame = cv2.warpAffine(np.array(transformed), self.transform, self.camera_shape[::-1])
            frame = (frame.astype(float) * .7 + 5).astype(np.uint8)
        else:
            active = [p for k, p in self.paths.items() if k != 'dmd' and p.name != 'flat.png']
            expected = np.load(active[0].parent / 'expected.npy')
            pattern = cv2.warpAffine(expected, self.mapping, self.camera_shape[::-1])
            if active[0].parent.parent.name == 'joint' and len(active) < 2:
                pattern = np.roll(pattern, 30, axis=1)
            dx = sum(self.values.get(k + 'OffsetXSpinBox', 0) for k in ('slm1', 'slm2') if self.paths[k].name != 'flat.png')
            dy = sum(self.values.get(k + 'OffsetYSpinBox', 0) for k in ('slm1', 'slm2') if self.paths[k].name != 'flat.png')
            pattern = np.roll(np.roll(pattern, dx * 3, axis=1), dy * 3, axis=0)
            frame = np.rint(pattern / max(pattern.max(), 1e-8) * 180 + 5).astype(np.uint8)
        return frame, [{'frame_id': 1}], [frame]

    def train(self, config, root, *args):
        manifest = {'version': 2, 'status': 'trained', 'patterns': [], 'config': config.to_dict(),
                    'grids': config.grid_metadata(), 'output_grid': 'slm2'}
        root.mkdir()
        write_json(root / 'config.json', config.to_dict())
        for key in ('slm1', 'slm2'):
            np.save(root / (key + '_lut_rad.npy'), np.linspace(0, 2 * np.pi, 256, dtype=np.float32))
        for stage in ('slm1', 'slm2', 'joint'):
            for name in ('dots', 'arrow', 'corners', 'holdout'):
                directory = root / stage / name
                directory.mkdir(parents=True)
                expected = target_intensity(config, name)
                np.save(directory / 'expected.npy', expected)
                np.save(directory / 'target.npy', expected)
                keys = ('slm1', 'slm2') if stage == 'joint' else (stage,)
                paths = {key: save_gray(directory / (key + '.png'), np.zeros(config.slm_shape(key), np.uint8)).relative_to(root).as_posix() for key in keys}
                manifest['patterns'].append({'stage': stage, 'name': name, 'paths': paths,
                    'expected': (directory / 'expected.npy').relative_to(root).as_posix(),
                    'target': (directory / 'target.npy').relative_to(root).as_posix(), 'simulation_score': 1.})
        seal_bundle(root, manifest)
        return manifest


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = CalibrationConfig(dmd_width=32, dmd_height=32,
            wavelength_nm=532.125, slm1_grid_size=64, slm2_grid_size=80,
            slm1_sampling_um=8.000123, slm2_sampling_um=6.4,
            slm1_phase_resolution=8, slm2_phase_resolution=16, steps=1,
            coarse_orientation=False, angle_radius=0, search_rounds=1, offset_radius=2)
        self.bench = FakeOpticalBench(self.config)
        self.manifest = self.bench.train(self.config, self.root / 'source')

    def settings(self):
        return {key + suffix: value for key in ('dmd', 'slm1', 'slm2')
                for suffix, value in (('RotationSpinBox', 0), ('HorizontalFlipCheckBox', False),
                    ('VerticalFlipCheckBox', False), ('OffsetXSpinBox', 0), ('OffsetYSpinBox', 0))}

    def test_copy_preserves_package_and_modified_phase_is_rejected(self):
        bundle = load_bundle(self.root / 'source' / 'manifest.json')
        copied = copy_bundle(bundle, self.root / 'copy')
        self.assertEqual(copied.config, self.config)
        self.assertEqual(copied.manifest['files'], bundle.manifest['files'])
        phase = copied.root / copied.manifest['patterns'][0]['paths']['slm1']
        save_gray(phase, np.ones(self.config.slm_shape('slm1'), np.uint8))
        with self.assertRaisesRegex(ValueError, '校验失败'):
            load_bundle(copied.root)

    def test_rejects_bad_arrays_incomplete_pairs_and_path_escape(self):
        for kind in ('nan', 'incomplete', 'escape'):
            with self.subTest(kind=kind):
                destination = self.root / kind
                bundle = copy_bundle(load_bundle(self.root / 'source'), destination)
                manifest = bundle.manifest
                if kind == 'nan':
                    np.save(destination / manifest['patterns'][0]['expected'], np.full((80, 80), np.nan))
                    seal_bundle(destination, manifest)
                    error = '强度数组'
                elif kind == 'incomplete':
                    manifest['patterns'].pop()
                    write_json(destination / 'manifest.json', manifest)
                    error = '12 组'
                else:
                    manifest['files']['../outside.npy'] = 'irrelevant'
                    write_json(destination / 'manifest.json', manifest)
                    error = '越界'
                with self.assertRaisesRegex(ValueError, error):
                    load_bundle(destination)

    def test_model_mismatch_and_aperture_rotation_are_detected(self):
        bundle = load_bundle(self.root / 'source')
        with self.assertRaisesRegex(ValueError, 'wavelength_nm'):
            bundle.check_config(replace(self.config, wavelength_nm=633))
        self.assertEqual(bundle.check_illumination(replace(self.config, dmd_rotation=180)), 0)
        with self.assertRaisesRegex(ValueError, '重新训练'):
            bundle.check_illumination(replace(self.config, dmd_rotation=45))

    def test_imported_pipeline_uses_saved_predictions_without_training(self):
        runner = CalibrationRunner(self.config, self.root / 'run', self.settings(),
            self.bench.io, lambda _: None, lambda: False, bundle_path=self.root / 'source')
        with patch('tools.calibration.runner.train_bundle', side_effect=AssertionError('must not train')):
            result = runner.run()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['phase_source']['kind'], 'imported')
        self.assertTrue((runner.root / 'training_request.json').is_file())
        self.assertEqual(load_bundle(runner.bundle).config, self.config)
        self.assertEqual(result['config']['target_x_um'], 0)

    def test_offscreen_target_exports_request_without_applying_alignment(self):
        config = replace(self.config, target_x_um=150)
        bench = FakeOpticalBench(config)
        bench.train(config, self.root / 'offscreen')
        runner = CalibrationRunner(config, self.root / 'failed_run', self.settings(),
            bench.io, lambda _: None, lambda: False, bundle_path=self.root / 'offscreen')
        with self.assertRaisesRegex(ValueError, 'training_request.json'):
            runner.run()
        self.assertTrue((runner.root / 'training_request.json').is_file())
        self.assertFalse((runner.root / 'calibration.json').exists())
        self.assertEqual(runner.report['status'], 'failed')

    def test_frozen_gui_and_imported_workflow_never_import_torch_or_models(self):
        script = '''
import importlib.abc
import sys
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torchvision', 'models'):
            raise AssertionError('forbidden training dependency: ' + fullname)
sys.meta_path.insert(0, RejectTraining())
from tools.calibration.test_calibration import FakeOpticalBench
from tools.calibration.bundle import load_bundle
from tools.calibration.runner import CalibrationRunner
from tools.calibration.development import check_training
from tools.gui.main_window import MainWindow
from PyQt5 import QtWidgets
from pathlib import Path
from unittest.mock import patch
import time
bundle = load_bundle(sys.argv[1])
app = QtWidgets.QApplication([])
window = MainWindow(demo=True)
sys.frozen = True
window.calibration.open()
dialog = window.calibration.dialog
dialog.set_bundle(bundle.root)
assert dialog.train_button.isHidden()
assert dialog.source_combo.count() == 1
assert not dialog.controls['slm1_sampling_um'].isEnabled()
assert dialog.controls['settle_seconds'].isEnabled()
bundle.check_config(dialog.values())  # Preserve precision beyond the spin-box display.
try:
    check_training(bundle.config)
except RuntimeError as exc:
    assert '打包版' in str(exc)
else:
    raise AssertionError('packaged training was not rejected')
settings = {key + suffix: value for key in ('dmd', 'slm1', 'slm2')
    for suffix, value in (('RotationSpinBox', 0), ('HorizontalFlipCheckBox', False),
        ('VerticalFlipCheckBox', False), ('OffsetXSpinBox', 0), ('OffsetYSpinBox', 0))}
bench = FakeOpticalBench(bundle.config)
runner = CalibrationRunner(bundle.config, Path(sys.argv[2]), settings, bench.io,
    lambda _: None, lambda: False, bundle_path=bundle.root)
assert runner.run()['status'] == 'passed'
def wait_for(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Qt operation timed out')
window.connect_devices()  # Explicit demo adapters; never load a vendor DLL.
wait_for(lambda: window.camera_ready)
window.demo = False
assert all(device.demo for device in window.devices.values())
def finish_fake_run(runner):
    return {'status': 'passed', 'applied_settings': dict(runner.settings)}
with patch.object(CalibrationRunner, 'run', new=finish_fake_run):
    window.calibration.start(dialog.values(), Path(sys.argv[2]) / 'gui', False, bundle_path=bundle.root)
    wait_for(lambda: not window.calibration.active)
assert dialog.progress.value() == 1, dialog.log.toPlainText()
assert 'torch' not in sys.modules and 'models' not in sys.modules
dialog.close()
window.close()
wait_for(lambda: window.camera is None)
print('no-training-dependency workflow passed')
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.root / 'source'),
            str(self.root / 'isolated_run')], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=40, env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'QT_QPA_PLATFORM': 'offscreen'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class PipelineTests(unittest.TestCase):
    def test_complete_simulated_capture_search_validation_and_report(self):
        config = CalibrationConfig(dmd_width=32, dmd_height=32,
            slm1_grid_size=64, slm2_grid_size=64, slm1_sampling_um=8, slm2_sampling_um=8,
            slm1_phase_resolution=8, slm2_phase_resolution=8, steps=1, coarse_orientation=False, angle_radius=0,
            search_rounds=1, offset_radius=2)
        bench = FakeOpticalBench(config)
        settings = {key + suffix: value for key in ('dmd', 'slm1', 'slm2')
                    for suffix, value in (('RotationSpinBox', 0), ('HorizontalFlipCheckBox', False),
                        ('VerticalFlipCheckBox', False), ('OffsetXSpinBox', 0), ('OffsetYSpinBox', 0))}
        with tempfile.TemporaryDirectory() as directory:
            runner = CalibrationRunner(config, Path(directory) / 'run', settings, bench.io, lambda _: None, lambda: False)
            with patch('tools.calibration.runner.train_bundle', side_effect=bench.train):
                result = runner.run()
            self.assertEqual(result['status'], 'passed')
            self.assertLess(abs(abs(result['applied_settings']['dmdRotationSpinBox']) - 180), .1)
            self.assertTrue((runner.root / 'calibration.json').is_file())
            self.assertGreater(len(list((runner.root / 'captures').iterdir())), 40)
            self.assertIn('local_response', result['slm1'])

    def test_no_light_fails_before_training(self):
        config = CalibrationConfig(dmd_width=32, dmd_height=32, slm1_grid_size=64, slm2_grid_size=64)
        def io(action, _):
            if action == 'capture':
                image = np.zeros((64, 64), np.uint8)
                return image, [{}], [image]
        with tempfile.TemporaryDirectory() as directory:
            runner = CalibrationRunner(config, Path(directory) / 'run', {}, io, lambda _: None, lambda: False)
            with patch('tools.calibration.runner.train_bundle') as train:
                with self.assertRaisesRegex(ValueError, '有效 DMD 点不足'):
                    runner.run()
                train.assert_not_called()
            self.assertFalse((runner.root / 'calibration.json').exists())


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_independent_grid_controls_summary_and_defaults(self):
        from tools.gui.main_window import MainWindow
        window = MainWindow(demo=True)
        try:
            window.calibration.open()
            dialog = window.calibration.dialog
            self.assertIn('432×432', dialog.grid_summary.text())
            self.assertIn('768×768', dialog.grid_summary.text())
            dialog.controls['slm1_sampling_um'].setValue(12)
            dialog.controls['slm1_grid_size'].setValue(640)
            dialog.controls['slm2_phase_resolution'].setValue(64)
            config = dialog.values()
            self.assertEqual(config.slm1_grid_size, 640)
            self.assertEqual(config.slm1_sampling_um, 12)
            self.assertEqual(config.slm2_sampling_um, 4.5)
            self.assertEqual(config.slm1_phase_resolution, 128)
            self.assertEqual(config.slm2_phase_resolution, 64)
            dialog.controls['dmd_width'].setValue(320)
            self.assertIn('216×432', dialog.grid_summary.text())
            self.assertIn('384×768', dialog.grid_summary.text())
            dialog.controls['slm2_grid_size'].setValue(2048)
            config = dialog.values()
            self.assertEqual((config.slm1_grid_size, config.slm2_grid_size), (768, 2048))
            self.assertNotIn('待调整', dialog.grid_summary.text())
            dialog.defaults_button.click()
            self.assertEqual(dialog.values(), CalibrationConfig())
            self.assertNotIn('待调整', dialog.grid_summary.text())
        finally:
            window.calibration.dialog.close()
            window.close()

    def test_sampling_auto_growth_and_committed_manual_input(self):
        from PyQt5 import QtCore, QtTest
        from tools.gui.main_window import MainWindow
        window = MainWindow(demo=True)
        try:
            window.calibration.open()
            dialog = window.calibration.dialog
            dialog.apply_config(empirical_config())
            dialog.controls['slm1_sampling_um'].setValue(11)
            grid = dialog.controls['slm1_grid_size']
            self.assertEqual(grid.value(), 640)
            self.assertIn('SLM1 512 → 640', dialog.log.toPlainText())
            # Intermediate digits must not trigger resizing while typing.
            grid.setFocus()
            grid.lineEdit().selectAll()
            QtTest.QTest.keyClicks(grid.lineEdit(), '608')
            self.assertEqual(grid.value(), 640)
            QtTest.QTest.keyClick(grid.lineEdit(), QtCore.Qt.Key_Return)
            self.assertEqual(dialog.values().slm1_grid_size, 608)
            grid.setValue(1000)
            dialog.controls['slm1_sampling_um'].setValue(13)
            self.assertEqual(dialog.values().slm1_grid_size, 1000)
            grid.setValue(32)
            self.assertEqual(dialog.values().slm1_grid_size, 512)
            dialog.controls['dmd_width'].setValue(1280)
            config = dialog.values()
            self.assertEqual((config.slm1_grid_size, config.slm2_grid_size), (832, 1664))
            dialog.controls['dmd_pitch_um'].setValue(6)
            config = dialog.values()
            self.assertEqual((config.slm1_grid_size, config.slm2_grid_size), (960, 1856))
        finally:
            window.calibration.dialog.close()
            window.close()

    def test_auto_growth_limit_explains_error_and_recovers(self):
        from tools.gui.main_window import MainWindow
        window = MainWindow(demo=True)
        try:
            window.calibration.open()
            dialog = window.calibration.dialog
            dialog.apply_config(empirical_config())
            dialog.controls['slm1_sampling_um'].setValue(.1)
            self.assertIn('4096', dialog.grid_summary.text())
            self.assertIn('待调整', dialog.grid_summary.text())
            self.assertEqual(dialog.controls['slm1_grid_size'].value(), 512)
            self.assertEqual(dialog.controls['slm2_grid_size'].value(), 1024)
            dialog.controls['slm1_sampling_um'].setValue(11)
            self.assertFalse(dialog.grid_adjustment_error)
            self.assertNotIn('待调整', dialog.grid_summary.text())
            self.assertEqual(dialog.values().slm1_grid_size, 640)
        finally:
            window.calibration.dialog.close()
            window.close()

    def test_loaded_development_config_and_training_use_adjusted_grids(self):
        from tools.gui.main_window import MainWindow
        window = MainWindow(demo=True)
        try:
            window.calibration.open()
            dialog = window.calibration.dialog
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'input.json'
                write_json(path, empirical_config(slm1_sampling_um=11).to_dict())
                # Reading a saved bundle/config remains strict unless the
                # development UI explicitly requests automatic adjustment.
                with self.assertRaisesRegex(ValueError, '覆盖 SLM2'):
                    CalibrationConfig.read(path)
                with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=(str(path), '')), \
                     patch('PyQt5.QtWidgets.QMessageBox.warning') as warning:
                    dialog.load_config()
                    warning.assert_not_called()
                self.assertEqual(dialog.values().slm1_grid_size, 640)
                dialog.output.setText(directory)
                with patch.object(window.calibration, 'start') as start, \
                     patch('PyQt5.QtWidgets.QMessageBox.warning') as warning:
                    dialog.start(True)
                    warning.assert_not_called()
                    start.assert_called_once()
                saved = CalibrationConfig.read(Path(directory) / 'last_config.json')
                self.assertEqual(saved, start.call_args.args[0])
                self.assertEqual((saved.slm1_grid_size, saved.slm1_sampling_um), (640, 11))
        finally:
            window.calibration.dialog.close()
            window.close()

    def test_640_phase_controls_show_actual_hardware_export_sizes(self):
        from tools.gui.main_window import MainWindow
        window = MainWindow(demo=True)
        try:
            window.calibration.open()
            dialog = window.calibration.dialog
            for key in ('slm1', 'slm2'):
                dialog.controls[key + '_phase_resolution'].setValue(640)
                self.assertEqual(dialog.values().phase_shape(key), (640, 640))
            self.assertIn('SLM1 640×640 → 432×432', dialog.grid_summary.text())
            self.assertIn('SLM2 640×640 → 768×768', dialog.grid_summary.text())
            self.assertEqual(dialog.values().slm1_grid_size, 704)
            self.assertEqual(dialog.values().slm2_grid_size, 1152)
        finally:
            window.calibration.dialog.close()
            window.close()

    def wait_for(self, predicate, seconds=8):
        import time
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Qt operation timed out')

    def controller_case(self, fail):
        from tools.gui.main_window import MainWindow
        from tools.gui.configuration import read_configuration
        from PyQt5 import QtCore
        window = MainWindow(demo=True)
        window.connect_devices()  # Every adapter and the camera are explicit demos.
        self.wait_for(lambda: window.camera_ready)
        window.calibration.open()
        window.demo = False  # Exercise the real workflow gate with fake adapters.
        self.assertTrue(all(device.demo for device in window.devices.values()))
        paths = {key: value[0] for key, value in window.paths.items()}
        def run(runner):
            runner.root.mkdir(parents=True)
            values = dict(runner.settings)
            values['slm1OffsetXSpinBox'] = 4
            runner.io('display', (paths, values))
            image, metadata, frames = runner.io('capture', runner.config)
            self.assertEqual(len(frames), 1)
            self.assertGreater(image.size, 0)
            if fail:
                raise ValueError('simulated validation failure')
            return {'status': 'passed', 'applied_settings': values}
        try:
            with tempfile.TemporaryDirectory() as directory:
                original = window.slm1OffsetXSpinBox.value()
                with patch.object(CalibrationRunner, 'run', new=run):
                    window.calibration.start(CalibrationConfig(settle_seconds=.01, average_frames=1,
                        discard_frames=1), Path(directory), False)
                    root = window.calibration.task.root
                    self.wait_for(lambda: not window.calibration.active)
                self.assertEqual(window.slm1OffsetXSpinBox.value(), original if fail else 4)
                self.assertTrue(window.previewScrollContents.isEnabled())
                if fail:
                    self.assertFalse((root / 'gui_configuration.json').exists())
                    self.assertIn('simulated validation failure', window.calibration.dialog.log.toPlainText())
                else:
                    saved = read_configuration(root / 'gui_configuration.json')
                    self.assertEqual(saved['slm1OffsetXSpinBox'], 4)
                    self.assertIn('参数已应用', window.calibration.dialog.log.toPlainText())
        finally:
            window.calibration.dialog.close()
            window.close()
            self.wait_for(lambda: window.camera is None)

    def test_controller_commits_success_and_saves_reloadable_configuration(self):
        self.controller_case(False)

    def test_controller_restores_original_inputs_on_failure(self):
        self.controller_case(True)

    def test_stop_interrupts_training_and_unlocks_ui(self):
        from tools.gui.main_window import MainWindow
        import time
        window = MainWindow(demo=True)
        window.calibration.open()
        entered = threading.Event()
        def training(config, root, progress, cancelled):
            root.mkdir()
            entered.set()
            while not cancelled():
                time.sleep(.01)
            raise InterruptedError('test cancelled')
        try:
            with tempfile.TemporaryDirectory() as directory:
                with patch('tools.calibration.train.train_bundle', side_effect=training):
                    window.calibration.start(CalibrationConfig(), Path(directory), True)
                    self.wait_for(entered.is_set)
                    self.assertFalse(window.configurationScrollContents.isEnabled())
                    window.stopAllButton.click()
                    self.wait_for(lambda: not window.calibration.active)
                self.assertTrue(window.configurationScrollContents.isEnabled())
                self.assertIn('test cancelled', window.calibration.dialog.log.toPlainText())
        finally:
            window.calibration.dialog.close()
            window.close()

    def test_buttons_and_demo_real_calibration_rejected(self):
        from tools.gui.main_window import MainWindow
        from tools.gui.configuration import read_values
        window = MainWindow(demo=True)
        try:
            before = read_values(window)
            window.calibrationSetupButton.click()
            self.app.processEvents()
            self.assertIsNotNone(window.calibration.dialog)
            self.assertEqual(before, read_values(window))
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, 'demo'):
                    window.calibration.start(CalibrationConfig(), Path(directory), False)
            self.assertFalse(window.calibration.active)
        finally:
            window.calibration.dialog.close()
            window.close()

    def test_raw_batch_ignores_camera_preview_transform(self):
        from tools.gui.camera_worker import CameraWorker
        from tools.utils.image_transforms import ImageTransform
        from types import SimpleNamespace
        import time
        class Camera:
            def __init__(self, settings):
                self.index = 0
            def open(self):
                return {'exposure': 1000., 'gain': 1., 'auto': False}
            def read(self):
                time.sleep(.005)
                self.index += 1
                return np.tile(np.arange(6, dtype=np.uint8), (4, 1)), {'frame_id': self.index}
            def close(self):
                pass
        now = time.monotonic()
        batch = FrameBatch(now, now + 5, 2, 1)
        worker = CameraWorker({'ccdTimeoutSpinBox': 200, 'ccdRotationSpinBox': 90}, backend_factory=Camera)
        worker.submit('calibration_capture', batch)
        worker.start()
        try:
            self.assertTrue(batch.event.wait(5))
            image, metadata = batch.result()
            self.assertEqual(image.shape, (4, 6))
            np.testing.assert_array_equal(image[0], np.arange(6))
            self.assertEqual(metadata[0]['image_transform']['angle'], 0)
            self.assertEqual(metadata[0]['preview_image_transform']['angle'], 90)
        finally:
            worker.requestInterruption()
            self.assertTrue(worker.wait(3000))


if __name__ == '__main__':
    unittest.main()
