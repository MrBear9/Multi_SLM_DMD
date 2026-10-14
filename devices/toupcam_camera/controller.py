"""Monochrome RAW acquisition owned by CameraWorker's thread."""
import ctypes as C
import math
import time

import numpy as np

from ...utils.paths import resolve_path
from ...utils.camera_settings import camera_roi
from pathlib import Path
from . import sdk


class ToupCamera:
    def __init__(self, settings, api_factory=sdk.ToupApi):
        self.settings, self.api_factory = settings, api_factory
        self.api = self.handle = self.buffer = None
        self.started = False
        self.last_barrier = -math.inf
        self.skip_next = False
        self.fps = None
        self.next_status = 0

    def call(self, name, *args):
        return self.api.call(name, self.handle, *args)

    def values(self, name, types, *args):
        return self.api.values(name, self.handle, types, *args)

    def option(self, name, value):
        self.call('put_Option', name, value)

    def open(self):
        s = self.settings
        try:
            self.api = self.api_factory(resolve_path(Path(s['toupcamDllPathEdit'])))
            devices = self.api.enumerate()
            target = s.get('toupcamDeviceComboBox', '')
            if target:
                device = next((d for d in devices if d['id'] == target), None)
                if device is None:
                    raise ValueError('所选 ToupTek 设备 ID 已失效，请重新枚举设备。')
            elif len(devices) == 1:
                device = devices[0]
            else:
                raise ValueError('未找到唯一的 ToupTek 相机，请枚举并选择设备。')
            if not device['flags'] & sdk.MONO:
                raise ValueError('当前 ToupTek 后端只接入单色 RAW 相机；此设备返回了彩色能力，请勿按普通灰度采集。')
            self.handle = self.api.call('Open', device['id'])
            if not self.handle:
                raise RuntimeError('ToupTek 打开失败，请检查占用程序、驱动和 USB 连接。')
            serial = C.create_string_buffer(32)
            self.call('get_SerialNumber', serial)
            self.identity = {**device, 'vendor': 'ToupTek',
                             'serial': serial.value.decode('ascii', errors='replace')}
            resolution = s.get('toupcamResolutionComboBox', '')
            resolutions = ['%dx%d' % tuple(r) for r in device['resolutions']]
            if resolution and resolution not in resolutions:
                raise ValueError('所选分辨率不在当前相机支持列表中，请重新枚举。')
            index = resolutions.index(resolution) if resolution else 0
            self.call('put_eSize', index)
            self.bits = 16 if '16' in s['toupcamFormatComboBox'] else 8
            if self.bits == 16 and self.call('get_MaxBitDepth') <= 8:
                raise ValueError('相机不支持高于 8 位的采集，请选择 8 位输出。')
            self.option(sdk.RAW, 1)
            self.option(sdk.BITDEPTH, int(self.bits == 16))
            self.option(sdk.TRIGGER, 0)
            self.option(sdk.UPSIDE_DOWN, 0)
            self.option(sdk.ROTATE, 0)
            self.call('put_HFlip', 0)
            self.call('put_VFlip', 0)
            self.call('put_RealTime', 1)
            hardware = s.get('toupcamHardwareBinningComboBox', '不合并（1 × 1）')
            factor = 1 if hardware.startswith('不合并') else int(hardware[0])
            self.configure_binning(factor)
            software = s.get('toupcamSoftwareBinningComboBox', '不合并（1 × 1）')
            digital = 1 if software.startswith('不合并') else int(software[0])
            if not 1 <= digital <= 8:
                raise ValueError('软件 Binning 倍数无效。')
            self.option(sdk.BINNING, 1 if digital == 1 else 0x80 | digital)
            gain = s.get('toupcamConversionGainComboBox', '保持相机当前值')
            if gain.startswith(('LCG', 'HCG')):
                if not device['flags'] & sdk.CG:
                    raise ValueError('当前相机未声明支持 LCG/HCG 转换增益。')
                self.option(sdk.CONVERSION_GAIN, int(gain.startswith('HCG')))
            width, height = self.values('get_Size', [C.c_int]*2)
            roi = camera_roi(s)
            if roi:
                x, y, rw, rh = roi
                if min(x,y) < 0 or min(rw,rh) < 8 or any(v % 2 for v in roi) or x+rw > width or y+rh > height:
                    raise ValueError(f'ToupTek ROI 必须为偶数、宽高至少 8，且位于 {width} × {height} 内。')
                self.call('put_Roi', *roi)
            else:
                self.call('put_Roi', 0, 0, 0, 0)  # SDK full-frame reset
            actual_roi = list(self.values('get_Roi', [C.c_uint]*4))
            if roi and actual_roi != roi:
                raise ValueError(f'SDK 返回的 ROI {actual_roi} 与请求 {roi} 不一致。')
            self.size = self.values('get_FinalSize', [C.c_int]*2)
            if min(self.size) <= 0 or self.size[0]*self.size[1] > 200_000_000:
                raise ValueError(f'相机返回无效或过大的帧尺寸：{self.size}')
            fourcc, depth = self.values('get_RawFormat', [C.c_uint]*2)
            if fourcc != int.from_bytes(b'YYYY', 'little'):
                raise ValueError('单色 RAW 格式未返回 YYYY，停止采集以避免错误解释像素。')
            self.depth = min(depth, self.bits)
            if not 1 <= self.depth <= self.bits:
                raise ValueError('相机返回了无效的有效位深。')
            self.exposure_range = self.values('get_ExpTimeRange', [C.c_uint]*3)[:2]
            self.gain_range = tuple(v/100 for v in self.values('get_ExpoAGainRange', [C.c_ushort]*3)[:2])
            # Record all geometry inputs. ROI annotations must not cross modes.
            actual_index, = self.values('get_eSize', [C.c_uint])
            self.geometry = dict(resolution_index=actual_index, hardware_binning=factor,
                                 software_binning=digital, sensor_roi=actual_roi,
                                 raw_size=list(self.size), effective_bits=self.depth)
            self.buffer = np.empty((self.size[1], self.size[0]), dtype=np.uint16 if self.bits == 16 else np.uint8)
            self.call('StartPullModeWithCallback', None, None)
            self.started = True
            return self.parameters()
        except Exception:
            self.close()
            raise

    def configure_binning(self, factor):
        try:
            count = self.call('get_BinningNumber')
        except sdk.SdkError as exc:
            if factor == 1 and exc.code == 0x80004001:
                return
            raise
        if count == 0 and factor == 1:
            return
        available = [self.values('get_BinningValue', [C.c_char_p], i)[0].decode('ascii') for i in range(count)]
        desired = f'{factor}x{factor}'
        if desired not in available:
            raise ValueError(f'当前设备不支持硬件 {desired}，可用值：{available}')
        # Retain current method for 1x1; higher factors explicitly use averaging.
        _, method = self.values('get_Binning', [C.c_char_p]*2)
        self.call('put_Binning', desired.encode(), method if factor == 1 else b'Average')
        actual, _ = self.values('get_Binning', [C.c_char_p]*2)
        if actual.decode('ascii') != desired:
            raise ValueError('硬件 Binning 未按请求生效。')

    def parameters(self):
        return dict(exposure=float(self.values('get_ExpoTime', [C.c_uint])[0]),
                    gain=self.values('get_ExpoAGain', [C.c_ushort])[0]/100,
                    auto=bool(self.values('get_AutoExpoEnable', [C.c_int])[0]),
                    exposure_range=self.exposure_range, gain_range=self.gain_range,
                    camera=dict(self.identity), geometry=dict(self.geometry))

    def apply(self, exposure, gain, auto):
        if not auto and (not math.isfinite(exposure) or not self.exposure_range[0] <= exposure <= self.exposure_range[1]
                         or not math.isfinite(gain) or not self.gain_range[0] <= gain <= self.gain_range[1]):
            raise ValueError('曝光或增益超出当前 ToupTek 相机范围。')
        self.call('put_AutoExpoEnable', int(auto))
        if not auto:
            self.call('put_ExpoTime', round(exposure))
            self.call('put_ExpoAGain', round(gain*100))
        self.ensure_fresh(time.monotonic())
        return self.parameters()

    def ensure_fresh(self, after):
        """Flush SDK/DDR backlog after a pattern settles; discard the in-flight frame."""
        if after > self.last_barrier and time.monotonic() >= after:
            self.option(sdk.FLUSH, 3)
            self.last_barrier, self.skip_next = after, True

    def read(self):
        info = sdk.FrameInfo()
        try:
            self.call('WaitImageV3', min(200, self.settings.get('toupcamTimeoutSpinBox',1000)),
                      C.c_void_p(self.buffer.ctypes.data), 0, self.bits, -1, C.byref(info))
        except sdk.SdkError as exc:
            if exc.code in (0x8000000a, 0x8001011f):
                raise TimeoutError('等待 ToupTek 新帧。') from exc
            raise
        if (info.width, info.height) != self.size:
            raise RuntimeError('相机帧尺寸在采集中变化，请停止并重新连接。')
        if self.skip_next:
            self.skip_next = False
            raise TimeoutError('丢弃切图或参数更改时已在曝光的帧。')
        now = time.monotonic()
        if now >= self.next_status:
            frames, milliseconds, _ = self.values('get_FrameRate', [C.c_uint]*3)
            self.fps = frames*1000/milliseconds if milliseconds else None
            self.cached_parameters = self.parameters()
            self.next_status = now+1
        exposure = info.expotime if info.flag & 4 else self.cached_parameters['exposure']
        gain = info.expogain/100 if info.flag & 8 else self.cached_parameters['gain']
        return self.buffer, dict(frame_id=int(info.seq), exposure=float(exposure), gain=float(gain),
            auto=self.cached_parameters['auto'], camera=dict(self.identity),
            camera_geometry=dict(self.geometry), acquisition_fps=self.fps,
            effective_bits=self.depth, camera_timestamp_us=int(info.timestamp))

    def close(self):
        try:
            if self.handle:
                try:
                    if self.started:
                        self.call('Stop')
                finally:
                    self.call('Close')
                    self.handle = None
                    self.started = False
        finally:
            if self.api is not None and self.handle is None:
                self.api.close()
                self.api = None
            self.buffer = None
