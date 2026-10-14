"""Windows ctypes ABI, checked against the supplied SDK 60.32549.

The vendor directory is read-only. Load exactly the DLL selected in the UI.
"""
import ctypes as C
import os
from pathlib import Path

MONO, CG = 0x10, 0x04000000
RAW, BITDEPTH, TRIGGER, BINNING, CONVERSION_GAIN = 4, 6, 11, 23, 25
ROTATE, UPSIDE_DOWN, FLUSH = 24, 35, 61


class SdkError(RuntimeError):
    def __init__(self, operation, result):
        self.code = int(result) & 0xffffffff
        super().__init__(f'ToupTek {operation} 失败（HRESULT 0x{self.code:08X}）。')


class Resolution(C.Structure):
    _fields_ = [('width', C.c_uint), ('height', C.c_uint)]


class Model(C.Structure):
    _fields_ = [('name', C.c_wchar_p), ('flag', C.c_ulonglong),
                ('maxspeed', C.c_uint), ('preview', C.c_uint), ('still', C.c_uint),
                ('maxfanspeed', C.c_uint), ('ioctrol', C.c_uint),
                ('xpixsz', C.c_float), ('ypixsz', C.c_float), ('res', Resolution * 16)]


class Device(C.Structure):
    _fields_ = [('displayname', C.c_wchar * 64), ('id', C.c_wchar * 64),
                ('model', C.POINTER(Model))]


class FrameInfo(C.Structure):
    _fields_ = [('width', C.c_uint), ('height', C.c_uint), ('flag', C.c_uint),
                ('seq', C.c_uint), ('timestamp', C.c_longlong),
                ('shutterseq', C.c_uint), ('expotime', C.c_uint),
                ('expogain', C.c_ushort), ('blacklevel', C.c_ushort)]


class ToupApi:
    def __init__(self, path):
        path = Path(path).resolve()
        if os.name != 'nt':
            raise OSError('当前 ToupTek 封装使用 Windows SDK。')
        if not path.is_file():
            raise FileNotFoundError(f'ToupTek DLL 不存在：{path}')
        self.directory = os.add_dll_directory(str(path.parent))
        try:
            self.dll = C.WinDLL(str(path))
            self._bind()
        except Exception:
            self.directory.close()
            raise

    def _bind(self):
        h, u, i, p = C.c_void_p, C.c_uint, C.c_int, C.POINTER
        signatures = {
            'EnumV2': (u, [p(Device)]), 'Open': (h, [C.c_wchar_p]), 'Close': (None, [h]),
            'StartPullModeWithCallback': (i, [h, h, h]), 'Stop': (i, [h]),
            'WaitImageV3': (i, [h, u, h, i, i, i, p(FrameInfo)]),
            'put_Option': (i, [h, u, i]), 'get_Option': (i, [h, u, p(i)]),
            'put_RealTime': (i, [h, i]), 'put_eSize': (i, [h, u]),
            'get_eSize': (i, [h, p(u)]), 'get_Size': (i, [h, p(i), p(i)]),
            'get_FinalSize': (i, [h, p(i), p(i)]),
            'get_ResolutionRatio': (i, [h, u, p(i), p(i)]),
            'get_RawFormat': (i, [h, p(u), p(u)]), 'get_MaxBitDepth': (i, [h]),
            'get_SerialNumber': (i, [h, C.c_char_p]),
            'get_ExpTimeRange': (i, [h, p(u), p(u), p(u)]),
            'get_ExpoAGainRange': (i, [h, p(C.c_ushort), p(C.c_ushort), p(C.c_ushort)]),
            'get_ExpoTime': (i, [h, p(u)]), 'put_ExpoTime': (i, [h, u]),
            'get_ExpoAGain': (i, [h, p(C.c_ushort)]), 'put_ExpoAGain': (i, [h, C.c_ushort]),
            'get_AutoExpoEnable': (i, [h, p(i)]), 'put_AutoExpoEnable': (i, [h, i]),
            'put_Roi': (i, [h, u, u, u, u]), 'get_Roi': (i, [h, p(u), p(u), p(u), p(u)]),
            'get_FrameRate': (i, [h, p(u), p(u), p(u)]),
            'get_BinningNumber': (i, [h]),
            'get_BinningValue': (i, [h, u, p(C.c_char_p)]),
            'get_BinningMethod': (i, [h, u, p(C.c_char_p)]),
            'get_Binning': (i, [h, p(C.c_char_p), p(C.c_char_p)]),
            'put_Binning': (i, [h, C.c_char_p, C.c_char_p]),
            'put_HFlip': (i, [h, i]), 'put_VFlip': (i, [h, i]),
        }
        for name, (result, args) in signatures.items():
            fn = getattr(self.dll, 'Toupcam_' + name)
            fn.restype, fn.argtypes = result, args

    def call(self, name, *args):
        result = getattr(self.dll, 'Toupcam_' + name)(*args)
        if result is not None and result < 0:
            raise SdkError(name, result)
        return result

    def values(self, name, handle, types, *args):
        values = [t() for t in types]
        self.call(name, handle, *args, *[C.byref(v) for v in values])
        return tuple(v.value for v in values)

    def enumerate(self):
        array = (Device * 128)()
        count = self.call('EnumV2', array)
        if count > len(array):
            raise RuntimeError('SDK 返回的设备数量超出缓冲区。')
        result = []
        for d in array[:count]:
            if not d.model:
                continue
            m = d.model.contents
            if not 1 <= m.preview <= 16:
                raise RuntimeError('SDK 返回了无效的分辨率数量。')
            result.append(dict(id=d.id, friendly_name=d.displayname, model=m.name or '',
                flags=int(m.flag), pixel_size_um=[float(m.xpixsz), float(m.ypixsz)],
                resolutions=[[int(r.width), int(r.height)] for r in m.res[:m.preview]]))
        return result

    def close(self):
        # ctypes owns the DLL handle. No FreeLibrary while SDK threads exist.
        self.directory.close()


def enumerate_devices(path):
    api = ToupApi(path)
    try:
        return api.enumerate()
    finally:
        api.close()
