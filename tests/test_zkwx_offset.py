"""SLM1 placement checks without loading the vendor DLL or opening a window."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from tools.devices.zkwx_slm.controller import canvas_origin, prepare_canvas
from tools.devices.zkwx_slm.sdk import Monitor, ZhongkeTimeoutSDK


class ZhongkeOffsetTests(unittest.TestCase):
    def test_offsets_use_full_panel_and_preserve_active_image_bounds(self):
        monitor = Monitor(1, 1920, 0, 1920, 1080, False)
        self.assertEqual(canvas_origin((432, 432), monitor), (744, 324))
        self.assertEqual(canvas_origin((432, 432), monitor, -744, -324), (0, 0))
        self.assertEqual(canvas_origin((432, 432), monitor, 744, 324), (1488, 648))
        for offsets in ((-745, 0), (745, 0), (0, -325), (0, 325)):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                canvas_origin((432, 432), monitor, *offsets)
        with self.assertRaises(ValueError):
            canvas_origin((1920, 1080), monitor, 1, 0)

    def test_odd_margin_placement_has_no_rounding_crop(self):
        monitor = Monitor(1, -1920, -100, 7, 5, False)
        self.assertEqual(canvas_origin((2, 2), monitor, 3, 2), (5, 3))
        with self.assertRaises(ValueError):
            canvas_origin((2, 2), monitor, 4, 2)

    def test_canvas_preserves_grayscale_modes_and_zeros_previous_placement(self):
        monitor = Monitor(1, -1920, 100, 6, 4, False)
        inputs = [
            Image.fromarray(np.array([[False, True], [True, False]], dtype=bool)),
            Image.fromarray(np.array([[1, 17], [128, 255]], dtype=np.uint8)),
            Image.fromarray(np.array([[1, 1024], [32768, 65535]], dtype=np.uint16)),
            Image.fromarray(np.array([[-1, 70000], [123456, 1]], dtype=np.int32)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for image in inputs:
                with self.subTest(mode=image.mode):
                    path = Path(directory) / (image.mode.replace(';', '_') + '.tiff')
                    image.save(path)
                    with prepare_canvas(path, monitor, -2, -1) as first:
                        self.assertEqual(first.mode, image.mode)
                        self.assertEqual(first.size, (6, 4))
                        np.testing.assert_array_equal(np.asarray(first)[:2, :2], np.asarray(image))
                    with prepare_canvas(path, monitor, 2, 1) as second:
                        expected = np.zeros_like(np.asarray(second))
                        expected[2:4, 4:6] = np.asarray(image)
                        np.testing.assert_array_equal(np.asarray(second), expected)

    def test_sdk_keeps_window_on_monitor_and_cleans_temporary_canvas(self):
        api = ZhongkeTimeoutSDK.__new__(ZhongkeTimeoutSDK)
        api.dll = MagicMock()
        api.dll.Timeout_ShowImageFromFilePath.return_value = None  # documented void
        api.created = True
        api._dll_directory = None
        monitor = Monitor(1, -1920, 200, 6, 4, False)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'phase.tiff'
            pixels = np.array([[1, 1024], [32768, 65535]], dtype=np.uint16)
            Image.fromarray(pixels).save(source)
            try:
                api.show(source, monitor, 16, 1, -1)
                args = api.dll.Timeout_ShowImageFromFilePath.call_args.args
                self.assertEqual(args[1:], (False, -1920, 200, 6, 4, False, 16))
                canvas_path = Path(args[0].decode('utf-8'))
                with Image.open(canvas_path) as image:
                    expected = np.zeros((4, 6), dtype=np.uint16)
                    expected[:2, 3:5] = pixels
                    np.testing.assert_array_equal(np.asarray(image), expected)
                # TIFF can retain a memory map after __exit__ closes its fp.
                image.close()
                api.show(source, monitor, 16, -2, 1)
                previous_canvas = canvas_path
                canvas_path = Path(api.dll.Timeout_ShowImageFromFilePath.call_args.args[0].decode('utf-8'))
                self.assertNotEqual(canvas_path, previous_canvas)
                self.assertFalse(previous_canvas.exists())
                with Image.open(canvas_path) as image:
                    expected[:] = 0
                    expected[2:4, :2] = pixels
                    np.testing.assert_array_equal(np.asarray(image), expected)
                image.close()
                api.show(source, monitor, 16)
                self.assertEqual(api.dll.Timeout_ShowImageFromFilePath.call_args.args[0],
                                 str(source).encode('utf-8'))
                self.assertFalse(canvas_path.exists())
            finally:
                api.close()
            self.assertFalse(canvas_path.exists())
            api.dll.Timeout_CloseWindow.assert_called_once()

    def test_sdk_new_frame_paths_keep_current_file_until_next_show_returns(self):
        api = ZhongkeTimeoutSDK.__new__(ZhongkeTimeoutSDK)
        api.dll, api.created, api._dll_directory = MagicMock(), True, None
        monitor = Monitor(1, 1920, 0, 6, 4, False)
        previous = None
        seen = set()

        def display(path_bytes, *_args):
            nonlocal previous
            path = Path(path_bytes.decode('utf-8'))
            self.assertTrue(path.exists())
            self.assertNotIn(path, seen)
            if previous is not None:
                self.assertTrue(previous.exists())
            seen.add(path)
            previous = path

        api.dll.Timeout_ShowImageFromFilePath.side_effect = display
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'phase.bmp'
            Image.new('L', (2, 2), color=100).save(source)
            try:
                for _ in range(api._MAX_CANVAS_FILES * 2):
                    api.show(source, monitor, 16, 1, 0)
                    self.assertEqual(list(Path(api._canvas_directory.name).iterdir()), [previous])
            finally:
                api.close()
            self.assertFalse(previous.exists())

    def test_sdk_defers_locked_canvas_cleanup_and_caps_disk_growth(self):
        api = ZhongkeTimeoutSDK.__new__(ZhongkeTimeoutSDK)
        api.dll, api.created, api._dll_directory = MagicMock(), True, None
        monitor = Monitor(1, 1920, 0, 6, 4, False)
        locked = set()
        unlink = Path.unlink

        def display(path_bytes, *_args):
            locked.add(Path(path_bytes.decode('utf-8')))

        def unlink_unlocked(path, *args, **kwargs):
            if path in locked:
                raise PermissionError('Simulated vendor file lock')
            return unlink(path, *args, **kwargs)

        api.dll.Timeout_ShowImageFromFilePath.side_effect = display
        api.dll.Timeout_CloseWindow.side_effect = locked.clear
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'phase.bmp'
            Image.new('L', (2, 2), color=100).save(source)
            try:
                with patch.object(Path, 'unlink', unlink_unlocked):
                    for _ in range(api._MAX_CANVAS_FILES):
                        api.show(source, monitor, 16, 1, 0)
                    self.assertEqual(len(locked), api._MAX_CANVAS_FILES)
                    with self.assertRaisesRegex(RuntimeError, 'Reconnect SLM1'):
                        api.show(source, monitor, 16, 1, 0)
                    self.assertEqual(api.dll.Timeout_ShowImageFromFilePath.call_count,
                                     api._MAX_CANVAS_FILES)
                    canvas_directory = Path(api._canvas_directory.name)
                    self.assertEqual(len(list(canvas_directory.iterdir())), api._MAX_CANVAS_FILES)
                    released = next(path for path in locked if path != api._active_canvas)
                    locked.remove(released)
                    api.show(source, monitor, 16, 1, 0)
                    self.assertFalse(released.exists())
                    self.assertEqual(len(list(canvas_directory.iterdir())), api._MAX_CANVAS_FILES)
            finally:
                api.close()
            self.assertFalse(canvas_directory.exists())

    def test_invalid_offset_never_reaches_sdk(self):
        api = ZhongkeTimeoutSDK.__new__(ZhongkeTimeoutSDK)
        api.dll = MagicMock()
        monitor = Monitor(1, 1920, 0, 6, 4, False)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'phase.bmp'
            Image.new('L', (2, 2)).save(source)
            with self.assertRaises(ValueError):
                api.show(source, monitor, 16, 3, 0)
        api.dll.Timeout_ShowImageFromFilePath.assert_not_called()

    def test_sdk_binary_and_8bit_canvases_preserve_samples_on_disk(self):
        monitor = Monitor(1, 1920, 0, 6, 4, False)
        for pixels in (np.array([[False, True], [True, False]], dtype=bool),
                       np.array([[0, 17], [128, 255]], dtype=np.uint8)):
            with self.subTest(dtype=pixels.dtype), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'phase.bmp'
                Image.fromarray(pixels).save(source)
                api = ZhongkeTimeoutSDK.__new__(ZhongkeTimeoutSDK)
                api.dll, api.created, api._dll_directory = MagicMock(), False, None
                try:
                    api.show(source, monitor, 16, -2, -1)
                    output = Path(api.dll.Timeout_ShowImageFromFilePath.call_args.args[0].decode('utf-8'))
                    with Image.open(output) as image:
                        expected = np.zeros((4, 6), dtype=pixels.dtype)
                        expected[:2, :2] = pixels
                        np.testing.assert_array_equal(np.asarray(image), expected)
                    image.close()
                finally:
                    api.close()

    @unittest.skipUnless(importlib.util.find_spec('PyQt5'), 'GUI dependencies are optional')
    def test_demo_adapter_validates_offsets_again_when_showing(self):
        from tools.gui.displays import DisplayAdapter

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'phase.bmp'
            Image.new('L', (432, 432), color=127).save(source)
            adapter = DisplayAdapter('slm1', {}, [source], demo=True)
            self.assertEqual(adapter.geometry[2:], (1920, 1080))
            adapter.show(source, {'slm1OffsetXSpinBox': 744, 'slm1OffsetYSpinBox': 324})
            with self.assertRaises(ValueError):
                adapter.show(source, {'slm1OffsetXSpinBox': 745})
            adapter.close()

    def test_sdk_void_binding_matches_vendor_manual(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk_dir = Path(directory)
            (sdk_dir / 'SecondDll.dll').touch()
            dll = MagicMock()
            with patch('tools.devices.zkwx_slm.sdk.ctypes.CDLL', return_value=dll), \
                    patch('tools.devices.zkwx_slm.sdk.os.add_dll_directory', create=True):
                api = ZhongkeTimeoutSDK(sdk_dir)
                try:
                    self.assertIsNone(dll.Timeout_ShowImageFromFilePath.restype)
                finally:
                    api.close()


if __name__ == '__main__':
    unittest.main()
