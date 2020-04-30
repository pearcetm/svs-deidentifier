# svs-deidentifier

Whole slide images, including Aperio SVS files, can potentially contain patient identifiers on the slide label (and rarely visible on the macro image). This tool allows users to remove these potentially identifying sub-images to enable sharing files safely for research purposes.

## How to install and run the application

The de-identification tool uses an app-mode browser window with HTML5/CSS3/JavaScript for the user interface, with Python3 under the hood. The code can be compiled into stand-alone executable applications and run using Windows, Mac, or Linux. Compiled executables can be downloaded from the ["Releases" tab](https://github.com/pearcetm/svs-deidentifier/releases) of the project GitHub page, under "assets".

Depending on the operating system and user permissions, you may need to take special steps to allow the application to run. For example, on MacOS, running the app will fail the first time, but after doing this, the app can be enabled in Security settings (see [this page](https://support.apple.com/en-us/HT202491) for details).

#### Chrome vs QWebEngine

There are two version of the application:
- Chrome: Uses a Google Chrome window for the user interface (Microsoft Edge is a fallback for Windows 10 users)
- QWebEngine: Entirely stand-alone, with a built-in web browser.

Try them both - the de-identification works exactly the same in both, but the app icons appear differently in the system tray.

## How to use the application

Use the tabs at the top of the user interface to choose which mode you want to use.

- **Copy** mode leaves the original file(s) unchanged, and creates a new copy which does not contain label and macro images. This is most useful when working with whole slide images created for other purposes (e.g. clinical use).
- **Modify** mode changes the original file. If you have already copied the file(s) manually, or only included the label/macro images by mistake while scanning, this is the right mode.

Files to de-identify can be added in two ways: (1) by directly selecting one or more .svs files; or (2) by selecting a .csv file that has a list of .svs filenames (including the path). The second option is most useful when a subset of images need to be copied. Therefore, the csv file must be formatted in a certain way ([see below](#CSV-file-format)).

Notes:
- In **copy** mode, you will be asked to select a location to copy the files into. This allows directly copying onto a removeable hard drive or USB drive to transfer files to end users. Select the base directory you want to start from. If destination file names are prefixed with a path, any necessary folders will be created.
- In **modify** mode, you will require appropriate filesystem permissions in order to alter the files.

#### <a name="CSV-file-format"></a>CSV file format

The file should consist of two columns. The first row contains the column headers, "source" and "destination".
 Each subsequent row should have the path to an existing .svs file in the "source" column and the desired file name (path is optional) in the "destination" column.

 For example:

 ```  
    source,                              destination  
    /path/to/files/slide1.svs,           slide1.svs  
    /path/to/files/subdir/slide2.svs,    case1/h_and_e.svs  
 ``` 
 *Note:  Even in **modify** mode, the file should be formatted with these two columns, but you can leave destination file names blank.*

---

# For Developers

## Application design
The app consists of an HTML/CSS/JS frontend and Python3 backend, which are linked using the [eel library](https://github.com/samuelhwilliams/Eel). The [pyinstaller library](https://github.com/pyinstaller/pyinstaller) is used to compile all dependencies into a single stand-alone executable application file. Pyinstaller is **not a cross-compiler** - for each target platform, the app must be compiled on that platform.

## Build from source

To build the de-identifier yourself, start by cloning the project from GitHub.

**Suggestion:** use `venv` to create a virtual environment with only the necessary python modules

 Python dependencies (other versions may work, but these have been tested and confirmed to work properly): 
 - Eel==0.12.2
 - PyQt5=={version} (**version: 5.14.1 works on Mac but not Ubuntu; 5.14.0 OK on Ubuntu)
 - PyQtWebEngine (Windows and Linux)
 - tinynumpy (needed for tiffparser)
 - commonmark (to automatically include README in the app during the build)
 - pyinstaller (to create application package)

Also requires [Qt](https://www.qt.io/) for cross-platform file dialog and web browser UIs.

#### To install all dependencies at once:
```
    $ pip install Eel==0.12.2 PyQt5=={**version} PyQtWebEngine tinynumpy commonmark pyinstaller
```

### Run as a python script
Using your python terminal/editor of choice, run the `deid-chrome.py` or `deid-qwebengine.py` script.

### Compile a stand-alone application
Using [PyInstaller](https://github.com/pyinstaller/pyinstaller), you can build the application into a distributable package in one of two ways: as a single stand-alone file, or as a directory with a separate executable. These have different advantages and drawbacks. One-file mode is simpler - the isn't an extra directory with lots of files - but the app will take longer to launch. In addition, using the built-in web browser doesn't work on MacOS in one-file mode. One-dir mode launches more quickly but if the directory is modified the app will not work. 

### Build all versions
The individual options below are bundled into a single `build.py` script to simplify the build process. On the command line (from the root directory of the project) run the following:  
`$ python build.py`

### Build a specfic version
You can also select a specific invididual version (one file vs one directory, chrome vs qwebengine) using the options below.

#### One-file mode
In the base directory for the project, run one of the following commands (change the file name as you like):
```
# For Linux
$ python -m eel deid-chrome.py web --name deid-chrome-onefile --onefile --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm
$ python -m eel deid-qwebengine.py web --name deid-qwebengine-onefile --onefile --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm

# For Mac
$ python -m eel deid-chrome.py web --name deid-chrome-onefile --onefile --noconsole --icon=icon.icns --add-data '*.md:.' --noconfirm
***One-file mode with qwebengine does not work with pyinstaller***

# For Windows
$ python -m eel deid-chrome.py web --name deid-chrome-onefile --onefile --noconsole --icon=icon.ico --add-data *.md;. --noconfirm
$ python -m eel deid-qwebengine.py web --name deid-qwebengine-onefile --onefile --noconsole --icon=icon.ico --add-data *.md;. --noconfirm

```

#### One-dir mode
In the base directory for the project, run the following command  (change the file name as you like):
```
# For Linux
$ python -m eel deid-chrome.py web --name deid-chrome-onedir --onedir --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm
$ python -m eel deid-qwebengine.py web --name deid-qwebengine-onedir --onedir --noconsole --icon=icon.png --add-data '*.md:.' --noconfirm

# For Mac
$ python -m eel deid-chrome.py web --name deid-chrome-onedir --onedir --noconsole --icon=icon.icns --add-data '*.md:.' --noconfirm
$ python -m eel deid-qwebengine.py web --name deid-qwebengine-onedir --onedir --noconsole --icon=icon.icns --add-data '*.md:.' --noconfirm

# For Windows
$ python -m eel deid-chrome.py web --name deid-chrome-onedir --onedir --noconsole --icon=icon.ico --add-data *.md;. --noconfirm
$ python -m eel deid-qwebengine.py web --name deid-qwebengine-onedir --onedir --noconsole --icon=icon.ico --add-data *.md;. --noconfirm

```


## Operating system notes
- Windows:
 - Eel doesn't work with python 3.8 - 3.6.8 works
 - use `pip install PyQtWebEngine`
- Mac: QWebEngine does not work in one-file mode; one-dir mode works fine

## Contribute to the project
Please leave comments/questions/issues on the GitHub page, and submit pull requests with bug fixes and new features!




