"""Run a bounded hardware connection or upload without blocking the Qt UI."""
from PyQt5 import QtCore


class DeviceTask(QtCore.QThread):
    def __init__(self, action, parent=None):
        super().__init__(parent)
        self.action = action
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.action(self.isInterruptionRequested)
        except Exception as exc:
            self.error = str(exc)
