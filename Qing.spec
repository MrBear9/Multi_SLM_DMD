# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import struct

# PyInstaller supplies SPECPATH as the directory containing this spec file.
tools_dir = Path(SPECPATH).resolve()
icon_file = tools_dir / 'ui' / 'QIC.ico'
if not icon_file.is_file():
    raise FileNotFoundError(f'Application icon not found: {icon_file}')

# The SLM and F4321 SDKs supplied with this project require Windows x64.
if struct.calcsize('P') != 8:
    raise RuntimeError('Build Qing with 64-bit Python: the hardware SDKs require x64.')

# Keep the same paths as MainWindow.ui and utils.paths inside the bundle.
# List runtime files explicitly: SDK samples, debug builds and copies of Windows
# system DLLs in the vendor folders must not be bundled indiscriminately.
sdk_files = {
    'python_VS2015_x64': [
        'SecondDll.dll',
        'opencv_core344.dll',
        'opencv_highgui344.dll',
        'opencv_imgcodecs344.dll',
        'opencv_imgproc344.dll',
        'opencv_videoio344.dll',
        'msvcp140.dll',
        'concrt140.dll',
        'vcruntime140.dll',
    ],
    'HDSLM_SDK_0612/HDSLM_API/Bin': ['HDSLMFunc.dll'],
    'DVP2 SDK CN/library/Visual C++/bin/x64': ['DVPCamera64.dll'],
    'F4321_HS_FG/DLL_Python_2': ['F4321_HS.dll'],
    'toupcamsdk/win/x64': ['toupcam.dll'],
}
hardware_binaries = []
for sdk_directory, filenames in sdk_files.items():
    destination = 'tools/3rdparty/' + sdk_directory
    for filename in filenames:
        hardware_binaries.append((str(tools_dir / '3rdparty' / sdk_directory / filename),
                                  destination))

# HDSLMFunc also needs the VC runtime. Use the supplied release runtime files
# beside this SDK too, so its add_dll_directory() lookup is self-contained.
for filename in ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'):
    hardware_binaries.append((str(tools_dir / '3rdparty' / 'python_VS2015_x64' / filename),
                              'tools/3rdparty/HDSLM_SDK_0612/HDSLM_API/Bin'))

missing_files = [source for source, _ in hardware_binaries if not Path(source).is_file()]
if missing_files:
    raise FileNotFoundError('Missing hardware runtime files:\n' + '\n'.join(missing_files))

# Preserve vendor DLLs when UPX happens to be installed on the build machine.
hardware_upx_exclude = sorted({Path(source).name for source, _ in hardware_binaries})

a = Analysis(
    [str(tools_dir / 'apps' / 'optical_control.py')],
    pathex=[str(tools_dir.parent)],
    binaries=hardware_binaries,
    datas=[
        (str(tools_dir / 'ui' / 'MainWindow.ui'), 'tools/ui'),
        (str(tools_dir / 'ui' / 'HolographyWindow.ui'), 'tools/ui'),
        (str(tools_dir / 'ui' / 'GeometryPointSelection.ui'), 'tools/ui'),
        (str(tools_dir / 'ui' / 'HolographyHelp.ui'), 'tools/ui'),
        (str(tools_dir / 'ui' / 'holography_help.html'), 'tools/ui'),
        (str(tools_dir / 'calibration_config.json'), 'tools'),
        # Runtime Qt window/taskbar icon; EXE icon alone does not set this.
        (str(icon_file), 'tools/ui'),
    ],
    hiddenimports=['tools.gui.main_window'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The packaged GUI imports calibration data; training stays in development.
    excludes=[
        'torch',
        'torchvision',
        'torchaudio',
        'models',
        'yaml',
        'tools.calibration.train',
        'tools.calibration.model',
        'tools.calibration.resampling',
        'tools.holography.worker',
        'tools.holography.optics',
        'tools.holography.initialization',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Qing',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(icon_file)],
    contents_directory='lib',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=hardware_upx_exclude,
    name='Qing',
)
