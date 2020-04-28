#!/usr/bin/env python

import platform
import subprocess
import os

opsys = platform.system()

if opsys=='Linux':
	os.system("python -m eel deid-chrome.py web --name deid-chrome-onefile --onefile --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --name deid-qwebengine-onefile --onefile --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm")
	os.system("python -m eel deid-chrome.py web --name deid-chrome-onedir --onedir --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --name deid-qwebengine-onedir --onedir --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm")

elif opsys=='Darwin':
	os.system("python -m eel deid-chrome.py web --name deid-chrome-onefile --onefile --noconsole --icon=icon.icns --add-data '*.md:.' --noconfirm")
	# ***One-file mode with qwebengine does not work with pyinstaller***
	os.system("python -m eel deid-chrome.py web --name deid-chrome-onedir --onedir --noconsole --icon=icon.icns --add-data '*.md:.' --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --name deid-qwebengine-onedir --onedir --noconsole --icon=icon.icns --add-data '*.md:.' --noconfirm")

elif opsys=='Windows':
	os.system("python -m eel deid-chrome.py web --name deid-chrome-onefile --onefile --noconsole --icon=icon.ico --add-data *.md;. --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --name deid-qwebengine-onefile --onefile --noconsole --icon=icon.ico --add-data *.md;. --noconfirm")
	os.system("python -m eel deid-chrome.py web --name deid-chrome-onedir --onedir --noconsole --icon=icon.ico --add-data *.md;. --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --name deid-qwebengine-onedir --onedir --noconsole --icon=icon.ico --add-data *.md;. --noconfirm")

