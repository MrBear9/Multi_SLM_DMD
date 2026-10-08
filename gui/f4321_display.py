"""Stage converted BMP copies and control the F4321 cached frame sequence."""
from pathlib import Path
import tempfile

from PIL import Image

from ..devices.zkwx_slm.sdk import Monitor
from ..utils.dmd_images import DmdImageOptions, prepare_image
from ..utils.image_transforms import ImageTransform
from .configuration import input_path


class F4321DisplayAdapter:
    geometry = None  # A network device does not occupy a Windows desktop screen.
    output = None
    sdk = None

    def __init__(self, settings, paths, demo=False, controller_factory=None):
        self.settings = dict(settings)
        self.paths = list(paths)
        self.demo = demo
        self.controller_factory = controller_factory
        self.controller = None
        self.display = None
        self.uploaded = False
        self.staging = None
        self.signature = None
        self.info_text = ''

    @property
    def ready(self):
        return self.uploaded and (self.demo or (self.controller is not None and self.controller.ready))

    def open(self, cancel=None):
        try:
            if self.demo:
                self.display = Monitor(0, 0, 0, 1920, 1080, False)
                self.info_text = '演示 F4321 · 1920×1080 · 未连接真实硬件'
            else:
                from ..devices.f4321_dmd.controller import F4321Controller
                factory = self.controller_factory or F4321Controller
                s = self.settings
                self.controller = factory(input_path(s['f4321DllPathEdit']), s['f4321DeviceIdSpinBox'])
                info = self.controller.connect(s['f4321LocalIpEdit'], s['f4321LocalPortSpinBox'],
                    s['f4321TargetIpEdit'], s['f4321TargetPortSpinBox'],
                    timeout=s['f4321TimeoutSpinBox'], cancel=cancel)
                self.display = Monitor(0, 0, 0, self.controller.width, self.controller.height, False)
                self.info_text = str(info)
            self.reload(self.settings, cancel)
        except Exception:
            self.close()
            raise

    def reload(self, settings, cancel=None):
        options = DmdImageOptions.from_settings(settings)
        transform = ImageTransform.from_settings(settings, 'dmd')
        if options.pixel_mode not in ('gray8', 'binary1'):
            raise ValueError('F4321 输入请选择“8 位灰度”或“1 位二值”。')
        if self.display is None:
            raise ValueError('F4321 尚未查询到实际面板尺寸。')
        if not self.paths:
            raise ValueError('F4321 输入图像不能为空。')
        if not self.demo and (self.controller is None or not self.controller.connected):
            raise ValueError('F4321 连接未就绪，请断开后重新连接。')
        self.uploaded = False
        source_signature = self._source_signature()
        if self.controller is not None:
            self.controller.stop()
        if self.staging is not None:
            self.staging.cleanup()
        self.staging = tempfile.TemporaryDirectory(prefix='f4321_frames_')
        width, height = self.display.width, self.display.height
        for index, path in enumerate(self.paths):
            if cancel and cancel():
                raise RuntimeError('已取消 F4321 图像准备。')
            with prepare_image(path, options) as converted:
                with transform.apply_pil(converted) as image:
                    if image.width > width or image.height > height:
                        raise ValueError(f'{path.name} 变换后 {image.width}×{image.height} 超过 F4321 实际面板 {width}×{height}。')
                    with Image.new(image.mode, (width, height), 0) as canvas:
                        canvas.paste(image, ((width-image.width)//2, (height-image.height)//2))
                        canvas.save(Path(self.staging.name) / f'{index+1:08d}.bmp', format='BMP')
        if self.controller is not None:
            self.controller.configure(exposure_us=settings['f4321ExposureSpinBox'],
                                      dark_us=settings['f4321DarkTimeSpinBox'])
            self.controller.upload(Path(self.staging.name), 1 if options.pixel_mode == 'binary1' else 8,
                len(self.paths), timeout=settings['f4321TimeoutSpinBox'], cancel=cancel)
        if source_signature != self._source_signature():
            raise ValueError('上传期间源图像发生变化，请重新上传。')
        self.signature = (options, transform, source_signature)
        self.settings = dict(settings)
        self.uploaded = True

    def _source_signature(self):
        return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in self.paths)

    def show(self, path, settings):
        signature = (DmdImageOptions.from_settings(settings),
                     ImageTransform.from_settings(settings, 'dmd'), self._source_signature())
        if not self.ready or signature != self.signature:
            raise ValueError('F4321 缓存与当前输入不一致，请暂停并重新上传。')
        index = self.paths.index(path)
        if self.controller is not None:
            self.controller.show_frame(index)

    def stop(self):
        if self.controller is not None:
            self.controller.stop()

    def close(self):
        self.uploaded = False
        if self.controller is not None:
            # A failed DeInit can leave a vendor reader running. Keep its files
            # and controller alive so the window can retry closing safely.
            self.controller.close()
            self.controller = None
        if self.staging is not None:
            self.staging.cleanup()
            self.staging = None
