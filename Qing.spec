# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

# PyInstaller supplies SPECPATH as the directory containing this spec file.
tools_dir = Path(SPECPATH).resolve()

a = Analysis(
    [str(tools_dir / 'apps' / 'optical_control.py')],
    pathex=[str(tools_dir.parent)],
    binaries=[],
    datas=[
        (str(tools_dir / 'ui' / 'MainWindow.ui'), 'tools/ui'),
        (str(tools_dir / 'ui' / 'QIC.ico'), 'tools/ui'),
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
    icon=[str(tools_dir / 'ui' / 'QIC.ico')],
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
