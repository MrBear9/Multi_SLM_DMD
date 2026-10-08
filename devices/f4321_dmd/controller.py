"""Confirmed connection, BMP upload and frame-addressed cached playback."""
from __future__ import annotations

import ipaddress
import os
from pathlib import Path
import re
import time

from PIL import Image

from .sdk import F4321SDK, F4321Error, DeviceInfo


class F4321Controller:
    def __init__(self, dll_path, device_id=1, sdk_factory=F4321SDK):
        self.dll_path, self.device_id, self.sdk_factory = dll_path, device_id, sdk_factory
        self.sdk = None
        self.connected = self.ready = False
        self.info = None
        self.width = self.height = 0
        self.frame_count = self.bit_depth = 0
        self.exposure_us, self.dark_us = 10000.0, 0.0

    @staticmethod
    def _check_cancel(cancel):
        if cancel and cancel():
            raise F4321Error('已取消 F4321 操作。')

    def _wait_text(self, predicate, timeout, cancel, description, baseline_error=''):
        deadline = time.monotonic() + timeout
        latest = ''
        while time.monotonic() < deadline:
            self._check_cancel(cancel)
            error = self.sdk.text('GetLastInfoError')
            if error and error != baseline_error:
                raise F4321Error(error)
            latest = self.sdk.text('GetLastInfoRight')
            if predicate(latest):
                return latest
            time.sleep(0.02)
        raise TimeoutError(f'{description}超时。最新设备反馈：{latest or "无"}')

    def _confirmed_stop(self, timeout, cancel):
        # CMDNoParam updates the SDK's latest-info snapshot synchronously.
        # Wait for the later board acknowledgement, not the send-success text.
        baseline = self.sdk.text('GetLastInfoError')
        self.sdk.command(3)
        self._wait_text(lambda text: 'Rec JX:Play Stop' in text, timeout, cancel,
                        '等待 F4321 停止确认', baseline)

    def connect(self, local_ip, local_port, target_ip, target_port, timeout=15, cancel=None):
        for address in (local_ip, target_ip):
            ipaddress.IPv4Address(address.strip())
        if not all(1 <= port <= 65535 for port in (local_port, target_port)) or timeout <= 0:
            raise ValueError('F4321 端口或超时参数无效。')
        try:
            self.sdk = self.sdk_factory(self.dll_path, self.device_id)
            self.sdk.initialize(local_ip.strip(), local_port, target_ip.strip(), target_port)
            self._confirmed_stop(timeout, cancel)
            self.sdk.command(1)
            deadline = time.monotonic() + timeout
            last_status = '无'
            while time.monotonic() < deadline:
                self._check_cancel(cancel)
                raw = self.sdk.text('GetDeviceInfo')
                if raw:
                    try:
                        self.info = DeviceInfo.parse(raw)
                        break
                    except F4321Error:
                        # The getter formats a default not-ready structure
                        # before the asynchronous Query response arrives.
                        last_status = raw
                time.sleep(0.02)
            else:
                raise TimeoutError(f'F4321 未返回就绪设备信息；UDP 初始化不代表设备在线。最新状态：{last_status}')
            self.width, self.height = self.info.width, self.info.height
            if (self.info.show_width, self.info.show_height) != (self.width, self.height):
                self.sdk.call('SetParamDMDShow', 22, 1, self.height)
                self.sdk.command(1)
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    self._check_cancel(cancel)
                    raw = self.sdk.text('GetDeviceInfo')
                    if raw:
                        info = DeviceInfo.parse(raw)
                        if (info.show_width, info.show_height) == (self.width, self.height):
                            self.info = info
                            break
                    time.sleep(0.02)
                else:
                    raise F4321Error('F4321 未确认恢复完整面板显示区域。')
            self.connected = True
            return self.info
        except Exception:
            self.close()
            raise

    def configure(self, exposure_us=10000, dark_us=0):
        for value in (exposure_us, dark_us):
            if not 0 <= value * 200 <= 0xffffffff:
                raise ValueError('F4321 时间超出 32 位 5 ns 计数范围。')
        if exposure_us <= 0:
            raise ValueError('F4321 曝光时间必须大于零。')
        self.exposure_us, self.dark_us = exposure_us, dark_us

    def upload(self, directory, bit_depth, frame_count, timeout=15, cancel=None):
        if not self.connected:
            raise F4321Error('F4321 连接未就绪，请重新连接。')
        self.ready = False
        try:
            if bit_depth not in (1, 8) or frame_count < 1 or frame_count * bit_depth > 0x7fffffff:
                raise ValueError('F4321 图像位深或缓存长度无效。')
            files = sorted(Path(directory).glob('*.bmp'))
            if len(files) != frame_count:
                raise ValueError('F4321 临时 BMP 数量与上传数量不一致。')
            # The vendor's MinGW readdir uses Windows directory enumeration,
            # without sorting names. os.scandir uses the same FindFirstFile/
            # FindNextFile order on Windows. Require the private staging
            # directory's order to match frame order before handing it over.
            with os.scandir(directory) as entries:
                enumerated = [entry.name for entry in entries if entry.name.endswith('.bmp')]
            if enumerated != [path.name for path in files]:
                raise ValueError('F4321 临时目录的 BMP 枚举顺序与帧号不一致；请将 TEMP 设置到本地 NTFS 磁盘后重试。')
            for path in files:
                self._check_cancel(cancel)
                with Image.open(path) as image:
                    image.load()
                    if image.format != 'BMP' or image.size != (self.width, self.height) or image.mode != ('1' if bit_depth == 1 else 'L'):
                        raise ValueError(f'F4321 BMP 格式、尺寸或位深不匹配：{path.name}')
            expected_bytes = self.width * self.height * bit_depth // 8 * frame_count
            if expected_bytes > 0xffffffff:
                raise ValueError('本次上传超过设备反馈计数范围，请减少图像数量。')
            self._confirmed_stop(timeout, cancel)
            baseline = self.sdk.text('GetLastInfoError')
            self.sdk.call('SetParamPicShow', 21, 1, bit_depth,
                          round(self.exposure_us * 200), round(self.dark_us * 200))
            self.sdk.send_bmp(directory, bit_depth, frame_count)

            def received(text):
                match = re.search(r'Rec JX:Rec Pic Data Num\s*=\s*(-?\d+)', text)
                return match is not None and (int(match[1]) & 0xffffffff) == expected_bytes

            self._wait_text(received, timeout, cancel,
                f'等待 F4321 接收图像数据量 {expected_bytes} 字节确认', baseline)
            self.frame_count, self.bit_depth, self.ready = frame_count, bit_depth, True
        except Exception:
            # SendPic owns asynchronous parsing/sending threads. DeInit must
            # succeed before the caller may remove any staged BMP file.
            self.close()
            raise

    def show_frame(self, index):
        if not self.ready or not 0 <= index < self.frame_count:
            raise F4321Error('F4321 缓存未就绪或帧号无效。')
        self.stop()
        # Eight-bit files occupy eight binary bit planes; keep cycling this
        # one grayscale frame until the application selects the next frame.
        self.sdk.call('Play1', 11, 1 + index * self.bit_depth, self.bit_depth)

    def stop(self):
        if self.sdk is not None and self.sdk.initialized:
            self.sdk.command(3)

    def close(self):
        self.ready = False
        # Even if DeInit fails, this session must never accept another upload.
        # The retained SDK reference exists only to retry resource release.
        self.connected = False
        if self.sdk is not None:
            try:
                self.stop()
            finally:
                self.sdk.close()
                # Only a successful DeInit permits the caller to delete BMPs.
                # Still reset state when Stop failed but DeInit succeeded.
                self.sdk = None
                self.connected = False
                self.info = None
        else:
            self.connected = False
            self.info = None
