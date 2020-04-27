#!/usr/bin/env python
import sys
import eel
from PyQt5.QtWidgets import QApplication
import wsideidentifier

app=QApplication([]) # create QApplication to enable file dialogs

try:
    eel.init('web')
    eel.start('app.html')
except Exception: #no chrome installed? Try falling  back to Edge (Windows 10)
    try:
        if sys.platform in ['win32', 'win64'] and int(platform.release()) >= 10:
            eel.init('web')
            eel.start('app.html', mode='edge')
        else:
            raise
    except Exception:
            raise Exception

app.exit()