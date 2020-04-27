#!/usr/bin/env python
import eel
from PyQt5.QtWidgets import QApplication
import wsideidentifier

from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
from PyQt5.QtCore import QUrl, QThread

class EelThread(QThread):
    def __init__(self, parent = None, init='web',url='main.html'):
        QThread.__init__(self, parent)
        self.init = init
        self.url = url

    def run(self):        
        # Note: This is never called directly. It is called by Qt once the
        # thread environment has been set up.
        eel.init(self.init)
        eel.start(self.url, block=True, mode=None)

app=QApplication([]) # create QApplication to enable file dialogs

et=EelThread(init='web',url='app.html')
et.start()

w = QWebEngineView()
w.resize(1100,800)
w.load(QUrl('http://localhost:8000/app.html'))
w.show()

app.exec()

app.exit()