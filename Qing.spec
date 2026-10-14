# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import struct

# PyInstaller supplies SPECPATH as the directory containing this spec file.
tools_dir = Path(SPECPATH).resolve()
icon_file = tools_dir / 'ui' / 'QIC.ico'
if not icon_file.is_file():
    raise FileNotFoundError(f'Application icon not found: {icon_file}')

a = Analysis(
    [str(tools_dir / 'apps' / 'optical_control.py')],
    pathex=[str(tools_dir.parent)],
    binaries=[(str(tools_dir / '3rdparty' / 'toupcamsdk' / 'win' /
                   ('x64' if struct.calcsize('P') == 8 else 'x86') / 'toupcam.dll'),
               'tools/3rdparty/toupcamsdk/win/' + ('x64' if struct.calcsize('P') == 8 else 'x86'))],
    datas=[
        (str(tools_dir / 'ui' / 'MainWindow.ui'), 'tools/ui'),
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
        'tools.calibration.train',
        'tools.calibration.model',
        'tools.calibration.resampling',
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
    upx_exclude=[],
    name='Qing',
)
