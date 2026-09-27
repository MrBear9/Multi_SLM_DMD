"""Regression checks that do not open windows or connect to hardware."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from tools.devices.dvp2_camera import sdk as dvp
from tools.devices.dvp2_camera.controller import frame_to_image
from tools.devices.magicholo_slm import sdk as magic
from tools.devices.magicholo_slm.controller import canvas_origin, prepare_canvas
from tools.devices.zkwx_slm import sdk as zkwx
from tools.devices.zkwx_slm.controller import collect_images
from tools.utils.paths import PROJECT_ROOT, THIRD_PARTY_ROOT, TOOLS_ROOT, resolve_path


class HardwareToolsTests(unittest.TestCase):
    def test_paths_do_not_depend_on_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(resolve_path(Path('tools/input')), TOOLS_ROOT / 'input')
                self.assertEqual(dvp.default_dll_path().parents[5], THIRD_PARTY_ROOT)
                self.assertEqual(zkwx.DEFAULT_SDK_DIR.parent, THIRD_PARTY_ROOT)
                self.assertEqual(magic.DEFAULT_SDK_DIR.parents[2], THIRD_PARTY_ROOT)
            finally:
                os.chdir(original)

    def test_imports_do_not_load_vendor_dlls(self):
        code = (
            "import ctypes; from unittest.mock import Mock; "
            "ctypes.CDLL = ctypes.WinDLL = Mock(side_effect=AssertionError('DLL loaded')); "
            "import tools.devices.dvp2_camera; import tools.devices.magicholo_slm; "
            "import tools.devices.zkwx_slm"
        )
        result = subprocess.run([sys.executable, '-c', code], cwd=PROJECT_ROOT,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))

    def test_alignment_export_dimensions_and_launch_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-m', 'tools.processing.generate_alignment_patterns',
                 '--output', directory, '--target', 'all'],
                cwd=PROJECT_ROOT, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            root = Path(directory)
            for relative, size in [('dmd_camera_pc/checkerboard.png', 640),
                                   ('dual_slm_pc/slm1/checkerboard.png', 432),
                                   ('dual_slm_pc/slm2/checkerboard.png', 768)]:
                with Image.open(root / relative) as image:
                    self.assertEqual(image.size, (size, size))
                    self.assertEqual(image.mode, 'L')
            metadata = json.loads((root / 'alignment_geometry.json').read_text(encoding='utf-8'))
            for command in metadata['launch_examples'].values():
                self.assertTrue(command.startswith('python -m tools.apps.'), command)

    def test_all_cli_help(self):
        modules = [
            'apps.control_dvp2_camera',
            'apps.play_dmd_input_fullscreen',
            'apps.play_dmd_with_dvp2_camera',
            'apps.play_dual_slms',
            'apps.play_magicholo_slm2',
            'apps.play_zkwx_slm1',
            'processing.capture_ccd_video_frames',
            'processing.generate_alignment_patterns',
        ]
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, '-m', 'tools.' + module, '--help'],
                    cwd=PROJECT_ROOT, capture_output=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
                self.assertIn(b'usage:', result.stdout)

    def test_image_selection_preserves_numeric_order_and_start_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('10.png', '2.png', '1.png', 'ignored.txt'):
                (root / name).touch()
            self.assertEqual([p.name for p in collect_images(root, None, None)],
                             ['1.png', '2.png', '10.png'])
            self.assertEqual([p.name for p in collect_images(root, '2', 1)], ['2.png'])

    def test_magic_canvas_keeps_native_pixels(self):
        display = magic.SDKDisplay(0, 'test', 0, 0, 6, 4, False)
        pixels = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'phase.png'
            Image.fromarray(pixels).save(path)
            origin = canvas_origin((2, 2), display, 1, 0)
            self.assertEqual(origin, (3, 1))
            result = np.frombuffer(prepare_canvas(path, display, origin, 0), dtype=np.uint8).reshape(4, 6)
            expected = np.zeros((4, 6), dtype=np.uint8)
            expected[1:3, 3:5] = pixels
            np.testing.assert_array_equal(result, expected)
        with self.assertRaises(ValueError):
            canvas_origin((2, 2), display, 3, 0)

    def test_camera_mono_keeps_intensity_and_owns_buffer(self):
        frame = dvp.DvpFrame()
        frame.iWidth, frame.iHeight = 2, 2
        frame.format, frame.bits = dvp.FORMAT_MONO, dvp.BITS_8
        raw = np.array([0, 17, 128, 255], dtype=np.uint8)
        image = frame_to_image(frame, raw)
        raw[:] = 0
        np.testing.assert_array_equal(image, [[0, 17], [128, 255]])
        with self.assertRaises(dvp.DvpError):
            frame_to_image(frame, np.zeros(1, dtype=np.uint8))

    def test_zkwx_call_disables_stretch_and_releases_resources(self):
        api = zkwx.ZhongkeTimeoutSDK.__new__(zkwx.ZhongkeTimeoutSDK)
        api.dll = MagicMock()
        api.created = False
        api._dll_directory = MagicMock()
        directory_handle = api._dll_directory
        api.open()
        api.show(Path('phase.png'), zkwx.Monitor(1, 1920, 0, 800, 600, False), 16)
        api.dll.Timeout_ShowImageFromFilePath.assert_called_once_with(
            b'phase.png', False, 1920, 0, 800, 600, False, 16,
        )
        api.close()
        api.close()
        api.dll.Timeout_CloseWindow.assert_called_once()
        directory_handle.close.assert_called_once()

    def test_magic_rejects_wrong_frame_size_before_sdk_call(self):
        api = magic.HDSLM8BitSDK.__new__(magic.HDSLM8BitSDK)
        api.dll = MagicMock()
        with self.assertRaises(ValueError):
            api.show_8bit(0, 2, 2, b'123')
        api.dll.SLM_Disp_Data.assert_not_called()


if __name__ == '__main__':
    unittest.main()
