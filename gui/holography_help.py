"""Local, searchable help opened by title-bar ? or F1."""
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from ..utils.paths import TOOLS_ROOT


class HolographyHelp(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        uic.loadUi(str(TOOLS_ROOT/'ui'/'HolographyHelp.ui'), self)
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.helpBrowser.setSource(QtCore.QUrl.fromLocalFile(str(TOOLS_ROOT/'ui'/'holography_help.html')))
        self.findButton.clicked.connect(self.find_next)
        self.searchEdit.returnPressed.connect(self.find_next)
        self.contentsButton.clicked.connect(lambda: self.helpBrowser.scrollToAnchor('contents'))
        self.closeButton.clicked.connect(self.close)
        self.search_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Find, self)
        self.search_shortcut.activated.connect(self.searchEdit.setFocus)

    def find_next(self):
        text = self.searchEdit.text().strip()
        if not text:
            return
        found = self.helpBrowser.find(text)
        if not found:
            cursor = self.helpBrowser.textCursor()
            cursor.movePosition(QtGui.QTextCursor.Start)
            self.helpBrowser.setTextCursor(cursor)
            found = self.helpBrowser.find(text)
        self.searchStatus.setText('已找到；继续查找会循环到开头。' if found else '未找到：'+text)
