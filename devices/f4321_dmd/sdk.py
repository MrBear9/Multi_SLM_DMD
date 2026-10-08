"""Bindings for the supplied F4321_HS header and dynamic-library documents."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import re
import struct
import time


class F4321Error(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceInfo:
    width: int
    height: int
    show_width: int
    show_height: int
    text: str

    def __str__(self):
        return self.text.replace(';', '\n')

    @classmethod
    def parse(cls, text):
        fields = ['DDR1 Initialization status', 'DMD Initialization status', 'Security check']
        ddr_count = re.search(r'Number of DDR\s*:\s*(\d+)', text, re.I)
        if 'DDR2 Initialization status' in text or (ddr_count and int(ddr_count[1]) >= 2):
            fields.append('DDR2 Initialization status')
        for field in fields:
            match = re.search(re.escape(field) + r'\s*:\s*(True|False)', text, re.I)
            if match is None or match[1].lower() != 'true':
                raise F4321Error(f'F4321 状态未就绪：{field}。设备反馈：{text}')
        native = re.search(r'DMD Type\s*:[^;]*?(\d+)\s*[*x×]\s*(\d+)\s*;', text, re.I)
        width = re.search(r'DMD Show Width\s*:\s*(\d+)', text, re.I)
        height = re.search(r'DMD Show Height\s*:\s*(\d+)', text, re.I)
        if not (native and width and height):
            raise F4321Error(f'无法从 F4321 反馈识别面板尺寸：{text}')
        result = cls(int(native[1]), int(native[2]), int(width[1]), int(height[1]), text)
        if min(result.width, result.height, result.show_width, result.show_height) <= 0:
            raise F4321Error('F4321 返回了无效面板尺寸。')
        return result


class F4321SDK:
    PREFIX = 'FL_DLP_HS_F4321_'

    def __init__(self, dll_path: Path, device_id=1):
        if os.name != 'nt' or struct.calcsize('P') != 8:
            raise F4321Error('F4321_HS.dll 需要 Windows 64 位 Python。')
        if device_id not in (1, 2, 3):
            raise ValueError('F4321 设备 ID 仅支持 1、2、3。')
        dll_path = Path(dll_path)
        if not dll_path.is_file():
            raise FileNotFoundError(dll_path)
        self.device_id = device_id
        self.initialized = False
        self._directory = os.add_dll_directory(str(dll_path.resolve().parent))
        self._last_command = 0.0
        self._upload_path = None
        self._module_handle = None
        try:
            self.dll = ctypes.WinDLL(str(dll_path.resolve()))
            self._module_handle = self.dll._handle
            self._bind()
        except Exception:
            self._unload()
            self._directory.close()
            raise

    def _unload(self):
        # DeInit does not clear the vendor's global DeviceInfo cache. Releasing
        # our load reference after its threads exit gives the next connection
        # a freshly initialized DLL instead of stale panel dimensions.
        handle = getattr(self, '_module_handle', None)
        if handle is not None:
            from _ctypes import FreeLibrary
            FreeLibrary(handle)
            self._module_handle = None
            self.dll = None

    def _bind(self):
        I, U, B = ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ctypes.c_ubyte)
        signatures = {
            'Init': [I, ctypes.c_char_p, I, ctypes.c_char_p, I],
            'DeInit': [I], 'CMDNoParam': [I, I],
            'GetDeviceInfo': [I, B, ctypes.POINTER(I)],
            'GetLastInfoRight': [I, B, ctypes.POINTER(I)],
            'GetLastInfoError': [I, B, ctypes.POINTER(I)],
            'SetParamPicShow': [I, I, I, U, U, U],
            'SetParamDMDShow': [I, I, U, U],
            'Play1': [I, I, I, I],
            'SendPic': [I, I, I, U, I, I, B, B, B],
        }
        for name, arguments in signatures.items():
            function = getattr(self.dll, self.PREFIX + name)
            function.argtypes, function.restype = arguments, I

    def call(self, name, *args):
        # Vendor manual requires spacing between commands. Polling reads do not
        # send network commands and do not need this delay.
        if not name.startswith('Get'):
            remaining = 0.002 - (time.monotonic() - self._last_command)
            if remaining > 0:
                time.sleep(remaining)
            self._last_command = time.monotonic()
        code = getattr(self.dll, self.PREFIX + name)(self.device_id, *args)
        if code != 0:
            raise F4321Error(f'F4321 {name} 失败，SDK 返回 {code}。')

    def initialize(self, local_ip, local_port, target_ip, target_port):
        self.call('Init', local_ip.encode('ascii'), local_port, target_ip.encode('ascii'), target_port)
        self.initialized = True

    def text(self, name):
        buffer = (ctypes.c_ubyte * 2048)()
        length = ctypes.c_int(0)
        self.call(name, buffer, ctypes.byref(length))
        if not 0 <= length.value <= len(buffer):
            raise F4321Error(f'F4321 {name} 返回了无效消息长度。')
        raw = bytes(buffer[:length.value]).rstrip(b'\0')
        try:
            message = raw.decode('utf-8')
        except UnicodeDecodeError:
            message = raw.decode('gbk', errors='replace')
        # These are the vendor's initial snapshots, not board replies/errors.
        if re.fullmatch(r'\[(?:Info|ERROE)\]\s*Device\s+\d+:\s*No (?:Info|Error)\s*', message, re.I):
            return ''
        return message

    def command(self, command_id):
        self.call('CMDNoParam', command_id)

    def send_bmp(self, directory, bit_depth, frame_count):
        # Retain the path buffer until DeInit/next completed transfer in case
        # the asynchronous vendor parser retains its pointer.
        self._upload_path = ctypes.create_string_buffer(str(Path(directory).resolve()).encode('utf-8'))
        pointer = ctypes.cast(self._upload_path, ctypes.POINTER(ctypes.c_ubyte))
        self.call('SendPic', 0 if bit_depth == 1 else 3, 1024, frame_count, 0, 1, pointer, None, None)

    def close(self):
        if self.initialized:
            self.call('DeInit')
            self.initialized = False
        self._unload()
        if self._directory is not None:
            self._directory.close()
            self._directory = None
        self._upload_path = None
