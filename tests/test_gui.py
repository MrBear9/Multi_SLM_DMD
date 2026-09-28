"""Headless Qt integration tests; no real SDK is loaded or device opened."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch, MagicMock

try:
    from PyQt5 import QtCore, QtWidgets, QtGui
    import numpy as np
    import cv2
    from tools.gui.main_window import MainWindow
    from tools.gui.configuration import read_values, apply_values, write_configuration, read_configuration
    from tools.gui.camera_worker import CaptureRequest, save_capture, CameraWorker, RealCamera
except ImportError:
    QtWidgets = None


@unittest.skipIf(QtWidgets is None, 'GUI dependencies are optional; install tools/requirements-gui.txt')
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)
        if Path('C:/Windows/Fonts/msyh.ttc').exists():
            QtGui.QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.warnings = []
        self.modal = patch.object(QtWidgets.QMessageBox, 'warning', side_effect=lambda *a: self.warnings.append(a[-1]))
        self.modal.start()
        self.window = MainWindow(demo=True)
        self.window.outputDirectoryEdit.setText(self.temp.name)

    def tearDown(self):
        self.window.close()
        self.wait_until(lambda: self.window.camera is None)
        self.modal.stop()
        self.temp.cleanup()

    def wait_until(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        self.fail('Timed out waiting for asynchronous GUI operation')

    def connect_demo(self):
        self.window.connect_devices()
        self.wait_until(lambda: self.window.camera_ready and self.window.received_frames > 0)

    def test_initial_page_local_previews_and_pair_navigation(self):
        w = self.window
        self.assertEqual(w.mainTabWidget.currentIndex(), 0)
        self.assertEqual(set(w.images), {'dmd', 'slm1', 'slm2'})
        self.assertEqual(w.objectName(), 'MainWindow')
        w.slm2FrameSpinBox.setValue(2)
        self.assertEqual(w.indices['slm1'], 1)
        self.assertEqual(w.indices['slm2'], 1)
        self.assertFalse(w.connected)

    def test_script_and_module_entrypoints_construct_demo_window(self):
        root = Path(__file__).resolve().parents[2]
        hooks = Path(self.temp.name) / 'python_hooks'
        hooks.mkdir()
        # Let each actual CLI entry point construct and show its real MainWindow.
        # Replace only the final event loop so the child process cannot hang.
        (hooks / 'sitecustomize.py').write_text(textwrap.dedent('''\
            from PyQt5 import QtWidgets

            def finish_demo(app):
                windows = [window for window in app.topLevelWidgets()
                           if window.objectName() == 'MainWindow']
                assert len(windows) == 1, 'MainWindow was not constructed'
                window = windows[0]
                assert window.isVisible(), 'MainWindow was not shown'
                assert window.demo and not window.connected, 'Expected isolated demo mode'
                assert set(window.source_images) == {'dmd', 'slm1', 'slm2'}
                assert all((window.images[key].width(), window.images[key].height())
                           == (1920, 1080) for key in ('dmd', 'slm1', 'slm2'))
                window.close()
                app.processEvents()
                print('DEMO_MAIN_WINDOW_READY')
                return 0

            QtWidgets.QApplication.exec_ = finish_demo
            '''), encoding='utf-8')
        environment = os.environ.copy()
        environment['QT_QPA_PLATFORM'] = 'offscreen'
        # Do not put the project on PYTHONPATH: direct-file startup must resolve
        # its own package even when the caller's working directory is elsewhere.
        environment['PYTHONPATH'] = str(hooks)
        cases = [
            ('script', [str(root / 'tools' / 'apps' / 'optical_control.py')], self.temp.name),
            ('module', ['-m', 'tools.apps.optical_control'], str(root)),
        ]
        for mode, command, cwd in cases:
            with self.subTest(entrypoint=mode):
                result = subprocess.run([sys.executable, *command, '--demo'], cwd=cwd,
                                        env=environment, capture_output=True, text=True,
                                        encoding='utf-8', errors='replace', timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('DEMO_MAIN_WINDOW_READY', result.stdout)

    def test_display_previews_keep_panel_and_active_area_sizes_separate(self):
        w = self.window
        for key, active_size in [('dmd', 640), ('slm1', 432), ('slm2', 768)]:
            with self.subTest(device=key):
                image = w.images[key]
                self.assertEqual((image.width(), image.height()), (1920, 1080))
                self.assertIn(str(active_size), getattr(w, key + 'ActiveAreaLabel').text())
                panel_info = getattr(w, key + 'PanelInfoLabel').text()
                self.assertIn('1920', panel_info)
                self.assertIn('1080', panel_info)
                self.assertIn('0001.png', getattr(w, key + 'FrameInfoLabel').text())
                source = QtGui.QImage(str(w.paths[key][0]))
                left, top = (1920 - source.width()) // 2, (1080 - source.height()) // 2
                # Native pixel positions must survive the preview composition.
                for x, y in [(0, 0), (32, 0), (0, 32), (32, 32)]:
                    self.assertEqual(image.pixelColor(left + x, top + y), source.pixelColor(x, y))
                self.assertEqual(image.pixelColor(0, 0), QtGui.QColor(0, 0, 0))

    def assert_preview_aspect(self, key, width, height):
        label = getattr(self.window, key + 'PreviewLabel')
        pixmap = label.pixmap()
        self.assertIsNotNone(pixmap)
        self.assertFalse(pixmap.isNull())
        self.assertLessEqual(pixmap.width(), label.contentsRect().width())
        self.assertLessEqual(pixmap.height(), label.contentsRect().height())
        # Qt rounds a fitted dimension to a pixel; allow at most that rounding.
        self.assertLessEqual(abs(pixmap.width() * height - pixmap.height() * width),
                             max(width, height))

    def test_preview_mode_switches_between_active_area_and_full_panel(self):
        w = self.window
        w.mainTabWidget.setCurrentIndex(1)
        w.show()
        self.app.processEvents()
        for key, active_size in [('dmd', 640), ('slm1', 432), ('slm2', 768)]:
            with self.subTest(device=key):
                mode = getattr(w, key + 'PreviewModeComboBox')
                self.assertEqual(mode.currentText(), '有效区域')
                source = w.source_images[key]
                self.assertEqual((source.width(), source.height()), (active_size, active_size))
                w.refresh_preview(key)
                self.assert_preview_aspect(key, 1, 1)
                mode.setCurrentText('完整面板')
                self.assert_preview_aspect(key, 1920, 1080)
                mode.setCurrentText('有效区域')
                self.assert_preview_aspect(key, 1, 1)
                # A display-only mode change must leave native output untouched.
                self.assertEqual((w.images[key].width(), w.images[key].height()), (1920, 1080))

    def test_non_square_active_images_keep_their_aspect_ratio(self):
        w = self.window
        w.synchronizeSlmsCheckBox.setChecked(False)
        w.mainTabWidget.setCurrentIndex(1)
        w.show()
        self.app.processEvents()
        for width, height in [(300, 120), (120, 300)]:
            path = Path(self.temp.name) / f'input_{width}x{height}.png'
            self.assertTrue(cv2.imwrite(str(path), np.full((height, width), 128, dtype=np.uint8)))
            for key in ('dmd', 'slm1', 'slm2'):
                with self.subTest(device=key, source_size=(width, height)):
                    getattr(w, key + 'InputPathEdit').setText(str(path))
                    w.load_sequence(key)
                    source = w.source_images[key]
                    self.assertEqual((source.width(), source.height()), (width, height))
                    mode = getattr(w, key + 'PreviewModeComboBox')
                    mode.setCurrentText('有效区域')
                    w.refresh_preview(key)
                    self.assert_preview_aspect(key, width, height)
                    mode.setCurrentText('完整面板')
                    self.assert_preview_aspect(key, 1920, 1080)
        self.assertFalse(self.warnings)

    def test_configuration_roundtrip_and_invalid_load_is_atomic(self):
        w = self.window
        settings = read_values(w)
        path = Path(self.temp.name) / 'configuration.json'
        write_configuration(path, settings)
        self.assertEqual(read_configuration(path), settings)
        with self.assertRaises(ValueError):
            apply_values(w, {'sessionNameEdit': 'changed', 'ccdGainSpinBox': -1})
        self.assertEqual(w.sessionNameEdit.text(), settings['sessionNameEdit'])
        apply_values(w, {'synchronizeSlmsCheckBox': False, 'ccdRoiCheckBox': True,
                         'synchronizeDmdCcdCheckBox': True, 'slm1OffsetXSpinBox': 17,
                         'slm1OffsetYSpinBox': -9})
        self.assertTrue(w.slm2IntervalSpinBox.isEnabled())
        self.assertTrue(w.ccdRoiWidget.isEnabled())
        changed = read_values(w)
        self.assertTrue(changed['synchronizeDmdCcdCheckBox'])
        self.assertEqual(changed['slm1OffsetXSpinBox'], 17)
        self.assertEqual(changed['slm1OffsetYSpinBox'], -9)
        write_configuration(path, changed)
        self.assertEqual(read_configuration(path), changed)

    def test_connect_capture_record_stop_and_disconnect(self):
        w = self.window
        self.connect_demo()
        self.assertFalse(w.configurationScrollContents.isEnabled())
        self.assertIn('ccd', w.images)
        w.capture_manual()
        self.wait_until(lambda: w.saved_frames == 1)
        png = next(Path(self.temp.name).glob('*.png'))
        image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        self.assertEqual(image.shape, (480, 640))
        log = json.loads((Path(self.temp.name) / 'captures.jsonl').read_text(encoding='utf-8').splitlines()[0])
        self.assertTrue(log['demo'])
        w.toggle_recording()
        self.wait_until(lambda: w.recording)
        started = time.monotonic()
        self.wait_until(lambda: time.monotonic() - started > .2)
        w.stop_all()
        self.wait_until(lambda: not w.recording)
        avi = next(Path(self.temp.name).glob('*.avi'))
        video = cv2.VideoCapture(str(avi))
        try:
            ok, _ = video.read()
            self.assertTrue(ok)
        finally:
            video.release()
        w.disconnect_devices()
        self.wait_until(lambda: w.camera is None)
        self.assertFalse(w.connected)
        self.assertTrue(w.configurationScrollContents.isEnabled())
        self.assertFalse(self.warnings)

    def test_dmd_automatic_capture_and_stop_cancels_pending(self):
        w = self.window
        w.dmdSettleSpinBox.setValue(.05)
        w.dmdIntervalSpinBox.setValue(.4)
        self.connect_demo()
        self.assertTrue(w.synchronizeDmdCcdCheckBox.isEnabled())
        w.synchronizeDmdCcdCheckBox.setChecked(True)
        w.play('dmd')
        self.assertFalse(w.synchronizeDmdCcdCheckBox.isEnabled())
        self.wait_until(lambda: w.saved_frames >= 1)
        self.assertTrue((Path(self.temp.name) / '0001.png').exists())
        w.stop_all()
        self.assertTrue(w.synchronizeDmdCcdCheckBox.isEnabled())
        count = w.saved_frames
        started = time.monotonic()
        self.wait_until(lambda: time.monotonic() - started > .5)
        self.assertEqual(w.saved_frames, count)
        self.assertFalse(any(t.isActive() for t in w.timers.values()))

    def test_dmd_playback_without_synchronization_does_not_capture(self):
        w = self.window
        self.assertFalse(w.synchronizeDmdCcdCheckBox.isChecked())
        # A capture-only stability delay must not restrict ordinary playback.
        w.dmdSettleSpinBox.setValue(.3)
        w.dmdIntervalSpinBox.setValue(.1)
        self.connect_demo()
        w.play('dmd')
        self.wait_until(lambda: not w.timers['dmd'].isActive())
        self.assertEqual(w.saved_frames, 0)
        self.assertFalse(list(Path(self.temp.name).glob('*.png')))
        w.capture_manual()
        self.wait_until(lambda: w.saved_frames == 1)
        manifest = json.loads((Path(self.temp.name) / 'captures.jsonl').read_text(encoding='utf-8').splitlines()[0])
        self.assertFalse(manifest['automatic'])

    def test_dmd_playback_without_synchronization_does_not_wait_for_camera(self):
        w = self.window
        w.connect_devices()
        # Worker signals remain queued until this thread processes Qt events.
        self.assertFalse(w.camera_ready)
        self.assertEqual(w.received_frames, 0)
        self.assertTrue(w.dmdPlayButton.isEnabled())
        w.play('dmd')
        self.assertTrue(w.timers['dmd'].isActive())
        w.stop_all()

    def test_synchronized_dmd_capture_requires_both_devices_and_valid_timing(self):
        w = self.window
        w.synchronizeDmdCcdCheckBox.setChecked(True)
        for key in ('dmd', 'ccd'):
            with self.subTest(disabled_device=key):
                checkbox = getattr(w, key + 'EnabledCheckBox')
                checkbox.setChecked(False)
                with self.assertRaises(ValueError):
                    w.validate()
                checkbox.setChecked(True)
        with self.assertRaises(ValueError):
            w.play('dmd')
        self.assertFalse(w.timers['dmd'].isActive())
        w.dmdSettleSpinBox.setValue(.3)
        w.dmdIntervalSpinBox.setValue(.3)
        with self.assertRaises(ValueError):
            w.validate()

    def test_synchronized_dmd_capture_waits_for_first_camera_frame(self):
        w = self.window
        w.synchronizeDmdCcdCheckBox.setChecked(True)
        self.connect_demo()
        actual_ready, actual_frames = w.camera_ready, w.received_frames
        try:
            for ready, frames in [(False, 0), (True, 0)]:
                with self.subTest(camera_ready=ready, frames=frames):
                    w.camera_ready, w.received_frames = ready, frames
                    w.update_actions()
                    self.assertFalse(w.dmdPlayButton.isEnabled())
                    with self.assertRaises(ValueError):
                        w.play('dmd')
                    self.assertFalse(w.timers['dmd'].isActive())
                    with self.assertRaises(ValueError):
                        w.start_all()
                    self.assertFalse(any(timer.isActive() for timer in w.timers.values()))
        finally:
            w.camera_ready, w.received_frames = actual_ready, actual_frames
            w.update_actions()

    def test_slm_offsets_move_native_active_area_and_reach_connected_adapter(self):
        w = self.window
        w.synchronizeSlmsCheckBox.setChecked(False)
        self.connect_demo()
        for key in ('slm1', 'slm2'):
            with self.subTest(device=key):
                source = QtGui.QImage(str(w.paths[key][0]))
                getattr(w, key + 'OffsetXSpinBox').setValue(47)
                getattr(w, key + 'OffsetYSpinBox').setValue(-19)
                with patch.object(w.devices[key], 'show', wraps=w.devices[key].show) as show:
                    getattr(w, key + 'ApplyOffsetButton').click()
                show.assert_called_once()
                self.assertEqual(show.call_args.args[1][key + 'OffsetXSpinBox'], 47)
                self.assertEqual(show.call_args.args[1][key + 'OffsetYSpinBox'], -19)
                preview = w.images[key]
                self.assertEqual((preview.width(), preview.height()), (1920, 1080))
                left = (1920 - source.width()) // 2 + 47
                top = (1080 - source.height()) // 2 - 19
                for x, y in [(0, 0), (32, 0), (0, 32), (32, 32)]:
                    self.assertEqual(preview.pixelColor(left + x, top + y), source.pixelColor(x, y))
        self.assertFalse(self.warnings)

    def test_slm_offsets_reject_cropping_without_replacing_preview(self):
        w = self.window
        w.synchronizeSlmsCheckBox.setChecked(False)
        for key in ('slm1', 'slm2'):
            with self.subTest(device=key):
                before = w.images[key].copy()
                source = QtGui.QImage(str(w.paths[key][0]))
                outside = (1920 - source.width()) // 2 + 1
                spin = getattr(w, key + 'OffsetXSpinBox')
                spin.setValue(outside)
                getattr(w, key + 'ApplyOffsetButton').click()
                # A constrained spin box may reject the value at entry; otherwise
                # applying it must fail without installing a cropped preview.
                if spin.value() == outside:
                    self.assertTrue(self.warnings)
                    self.assertEqual(w.images[key], before)
                self.warnings.clear()

    def test_camera_parameter_commands_are_applied_in_worker(self):
        w = self.window
        self.connect_demo()
        w.ccdExposureSpinBox.setValue(1234)
        w.ccdGainSpinBox.setValue(2)
        events = []
        w.camera.parametersChanged.connect(events.append)
        w.apply_camera_parameters()
        self.wait_until(lambda: bool(events))
        self.assertEqual(events[-1]['exposure'], 1234)
        self.assertEqual(events[-1]['gain'], 2)

    def test_capture_preserves_uint16_and_refuses_overwrite(self):
        path = Path(self.temp.name) / 'frame.png'
        request = CaptureRequest(path, 0, False, False, {}, 100)
        image = np.array([[0, 1024], [65535, 42]], dtype=np.uint16)
        save_capture(request, image, {})
        with self.assertRaises(FileExistsError):
            save_capture(request, image * 0, {})
        np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), image)

    def test_connection_failure_rolls_back_created_displays(self):
        created = []
        def factory(key, *args):
            if key == 'slm1':
                raise RuntimeError('Simulated SDK failure')
            device = MagicMock()
            device.geometry = (0, 0, 1920, 1080)
            created.append(device)
            return device
        self.window.display_factory = factory
        with self.assertRaisesRegex(RuntimeError, 'Simulated'):
            self.window.connect_devices()
        created[0].close.assert_called_once()
        self.assertFalse(self.window.connected)
        self.assertIsNone(self.window.camera)

    def test_worker_failure_closes_backend(self):
        backend = MagicMock()
        backend.open.side_effect = RuntimeError('No camera')
        worker = CameraWorker(read_values(self.window), backend_factory=lambda _: backend)
        failures = []
        worker.failed.connect(failures.append)
        worker.start()
        self.wait_until(lambda: not worker.isRunning() and bool(failures))
        worker.wait()
        backend.close.assert_called_once()
        self.assertIn('No camera', failures[0])

    def test_backend_constructor_failure_is_reported(self):
        def fail(_):
            raise RuntimeError('backend constructor failed')
        worker = CameraWorker(read_values(self.window), backend_factory=fail)
        failures = []
        worker.failed.connect(failures.append)
        worker.start()
        self.wait_until(lambda: not worker.isRunning() and bool(failures))
        worker.wait()
        self.assertIn('constructor failed', failures[0])

    def test_expired_capture_is_not_saved(self):
        self.connect_demo()
        warnings = []
        self.window.camera.warning.connect(warnings.append)
        path = Path(self.temp.name) / 'expired.png'
        request = CaptureRequest(path, 0, False, False, {}, time.monotonic() - 1)
        self.window.camera.submit('capture', request)
        self.wait_until(lambda: bool(warnings))
        self.assertFalse(path.exists())

    def test_real_camera_adapter_calls_sdk_and_releases_resources(self):
        import tools.gui.camera_worker as module
        api, session = MagicMock(), MagicMock()
        api.list_devices.return_value = [object()]
        session.handle.value = 7
        for name in ('dvpSetTriggerState', 'dvpSetTargetFormat', 'dvpSetRoiState',
                     'dvpSetAeOperation', 'dvpSetExposure', 'dvpSetAnalogGain',
                     'dvpGetExposure', 'dvpGetAnalogGain', 'dvpGetAeOperation'):
            getattr(api.dll, name).return_value = module.dvp.DVP_STATUS_OK
        frame = module.dvp.DvpFrame()
        frame.iWidth, frame.iHeight = 2, 2
        frame.format, frame.bits = module.dvp.FORMAT_MONO, module.dvp.BITS_8
        session.get_frame.return_value = frame, np.array([0, 1, 128, 255], dtype=np.uint8)
        with patch.object(module.dvp, 'DvpApi', return_value=api), patch.object(module.dvp, 'CameraSession', return_value=session):
            backend = RealCamera(read_values(self.window))
            try:
                backend.open()
                api.dll.dvpSetRoiState.assert_called_once_with(7, False)
                session.start.assert_called_once()
                image, metadata = backend.read()
                np.testing.assert_array_equal(image, [[0, 1], [128, 255]])
                backend.apply(1200, 2, False)
                api.dll.dvpSetExposure.assert_called_once_with(7, 1200)
                api.dll.dvpSetAnalogGain.assert_called_once_with(7, 2)
                session.get_frame.assert_called_once_with(200)
            finally:
                backend.close()
        session.close.assert_called_once()
        api.close.assert_called_once()

    def test_pair_mismatch_rejected_before_connect(self):
        path = Path(self.window.slm2InputPathEdit.text()) / '0003.png'
        path.rename(path.with_name('other.png'))
        with self.assertRaisesRegex(ValueError, '名称'):
            self.window.connect_devices()
        self.assertFalse(self.window.connected)

    def test_close_waits_for_worker_cleanup(self):
        self.connect_demo()
        worker = self.window.camera
        self.window.close()
        self.wait_until(lambda: self.window.camera is None)
        self.assertFalse(worker.isRunning())


if __name__ == '__main__':
    unittest.main()
