#!/usr/bin/env python

import platform
import subprocess
import os

opsys = platform.system()

if opsys=='Linux':
	os.system("python -m eel deid-chrome.py web --distpath ./dist/chrome-onefile --name svs-deidentifier --onefile --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --distpath ./dist/qwebengine-onefile --name svs-deidentifier --onefile --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm")
	os.system("python -m eel deid-chrome.py web --distpath ./dist/chrome-onedir --name svs-deidentifier --onedir --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --distpath ./dist/qwebengine-onedir --name svs-deidentifier --onedir --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm")

elif opsys=='Darwin':
	os.system("python -m eel deid-chrome.py web --distpath ./dist/chrome-onefile --name svs-deidentifier --onefile --noconsole --icon=icon.icns --add-data '*.md:.' --noconfirm")
	# ***One-file mode with qwebengine does not work with pyinstaller***
	os.system("python -m eel deid-chrome.py web --distpath ./dist/chrome-onedir --name svs-deidentifier --onedir --noconsole --icon=icon.icns --add-data '*.md:.' --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --distpath ./dist/qwebengine-onedir --name svs-deidentifier --onedir --noconsole --icon=icon.icns --add-data '*.md:.' --noconfirm")

elif opsys=='Windows':
	os.system("python -m eel deid-chrome.py web --distpath ./dist/chrome-onefile --name svs-deidentifier --onefile --noconsole --icon=icon.ico --add-data *.md;. --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --distpath ./dist/qwebengine-onefile --name svs-deidentifier --onefile --noconsole --icon=icon.ico --add-data *.md;. --noconfirm")
	os.system("python -m eel deid-chrome.py web --distpath ./dist/chrome-onedir --name svs-deidentifier --onedir --noconsole --icon=icon.ico --add-data *.md;. --noconfirm")
	os.system("python -m eel deid-qwebengine.py web --distpath ./dist/qwebengine-onedir --name svs-deidentifier --onedir --noconsole --icon=icon.ico --add-data *.md;. --noconfirm")

