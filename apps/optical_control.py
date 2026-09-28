"""Launch the Qt optical bench. Use --demo to work without physical devices."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Direct file execution does not set a package or add the project root.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'tools.apps'


def main(argv=None):
    parser = argparse.ArgumentParser(description='DMD / SLM / CCD Qt control panel')
    parser.add_argument('--demo', action='store_true', help='Explicit synthetic camera and virtual devices; no vendor DLLs.')
    parser.add_argument('--config', type=Path, help='Load a saved JSON configuration.')
    args = parser.parse_args(argv)
    try:
        from PyQt5 import QtCore, QtWidgets
    except ImportError:
        print('需要 PyQt5：请在当前 Python 环境安装 tools/requirements-gui.txt 中的依赖。', file=sys.stderr)
        return 1
    from ..gui.main_window import MainWindow
    # DMD output coordinates and raster pixels must use physical screen pixels.
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_DisableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('Optical Control')
    try:
        window = MainWindow(demo=args.demo, config_path=args.config)
    except Exception as exc:
        print(f'启动失败：{exc}', file=sys.stderr)
        return 1
    window.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
