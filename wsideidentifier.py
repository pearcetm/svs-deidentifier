#!/usr/bin/env python

# for mac: //instructions from https://opensource.com/article/19/5/python-3-default-mac
# //UNINSTALL HOMEBREW VERSIONS OF PYTHON FIRST!!
# $ brew install pyenv //manages python environments to avoid messy system environment
# $ pyenv install <version> //whatever version you want - 3.8.2 as of 3/27/2020
# $ pyenv global <version> //(optional) set version to be your default
# $ echo -e 'eval "$(pyenv init -)"' >> ~/.bash_profile //restart terminal to apply changes


# Run from within a virtual environment (venv) to allow defined versions of all external modules 
# $ python3 -m venv venv //creates venv folder within current directory
# $ source venv/bin/activate //activates virtual environment
# - all dependencies can be installed without conflicts here
# - dependency list:
# -- pip install Eel==0.12.2
# -- pip install PyQt5==5.14.1
# -- pip install tinynumpy (needed for tiffparser)
# -- pip install pyinstaller (to create application package)

# To install all dependencies at once:
# $ pip install Eel==0.12.2 PyQt5==5.14.1 tinynumpy pyinstaller


# os: for opening files
import os
# shutil: for copying files
import shutil
# struct: to pack/unpack data while writing binary files
import struct
# re: for regular expression parsing of file paths
import re
# copy: copy object rather than creating a reference
import copy
#threading.Thread: for async file copying
import threading
from threading import Thread
#csv: for reading csv file with filenames/destinations
import csv
#sys, platform: for checking which operating system is in use
import sys
import platform

# REQUIRED EXTERNAL LIBRARIES

# eel: for the html/css/javascript GUI
# $ pip install Eel==0.12.2
import eel

# tiffparser: stripped down version of tifffile
# located in root project folder alongside this file.
# removes need for numpy, which reduces application size significantly
# requires tinynumpy instead
# $ pip install tinynumpy
import tiffparser

# Parse Markdown to create options interface
import commonmark

from bottle import route


# pyqt5: python binding to Qt GUI framework, needed for file/directory picker dialog
# $ pip install PyQt5==5.14.1
# also requires qt5 to be installed
# - $ brew install qt5
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow
from PyQt5.QtCore import QDir, Qt, QUrl, QThread
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
from PyQt5.QtCore import QObject, pyqtSignal


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

class ThreadsafeCaller(QObject):
    get_signal = pyqtSignal(QObject,dict)
    # set_signal = pyqtSignal(dict)
    
    def __init__(self):
        super(ThreadsafeCaller, self).__init__()
        self.threadid=int(QThread.currentThreadId())
        self.multithreading=False
        self.returnvalue=None # read-only from self; written by other
        self.waiting=False # read-only from self; written by other; atomic

    def _make(self):
        other = ThreadsafeCaller()

        # if constructed from a different thread, use signal-slot to make threadsafe
        if self.threadid != other.threadid:
            other.multithreading=True
            other.get_signal.connect(self._call)
            
        return other

    def _call(self,other,d):
        # unpack and call the function; package returned value into dict to send by signal
        output={'result': d['f'](*d['args'], **d['kwargs'])}
        other.returnvalue=output
        other.waiting=False
        return output

    def _wait(self):
        while self.waiting==True:
            QThread.msleep(20)


    def _callother(self,func,*args,**kwargs):
        # package arguments into dict to send via signal
        d={'f':func,'args':args,'kwargs':kwargs}
        if self.multithreading:
            self.waiting=True # set flag before triggering signal-slot
            self.get_signal.emit(self,d)
            self._wait()
            return self.returnvalue['result'] #unpack result from dict
        else:
            return self._call(self,d)['result'] #unpack result from dict

    def call(self,func,*args,**kwargs): 
        other=self._make()
        return other._callother(func,*args,**kwargs)


@route('/settings')
def settings():
    with open('settings.md','r') as fp:
        return commonmark.commonmark(fp.read())

@route('/readme')
def readme():
    with open('README.md','r') as fp, open('help.md','r') as hp:
        d = {
            'readme':commonmark.commonmark(fp.read()),
            'help':commonmark.commonmark(hp.read())
        }
        return d

# Read/modify TIFF files (as in the SVS files) using tiffparser library (stripped down tifffile lib)

# delete_associated_image will remove a label or macro image from an SVS file
def delete_associated_image(slide_path, image_type):
    # THIS WILL ONLY WORK FOR STRIPED IMAGES CURRENTLY, NOT TILED

    allowed_image_types=['label','macro'];
    if image_type not in allowed_image_types:
        raise Exception('Invalid image type requested for deletion')

    fp = open(slide_path, 'r+b')
    t = tiffparser.TiffFile(fp)

    filtered_pages = [page for page in t.pages if image_type in page.description]
    num_results = len(filtered_pages)
    if num_results > 1:
        raise Exception(f'Invalid SVS format: duplicate associated {image_type} images found')
    if num_results == 0:
        #No image of this type in the WSI file; no need to delete it
        return

    # At this point, exactly 1 image has been identified to remove
    page = filtered_pages[0]

    # get the list of IFDs for the various pages
    offsetformat = t.tiff.ifdoffsetformat
    offsetsize = t.tiff.ifdoffsetsize
    tagnoformat = t.tiff.tagnoformat
    tagnosize = t.tiff.tagnosize
    tagsize = t.tiff.tagsize
    unpack = struct.unpack

    # start by saving this page's IFD offset
    ifds = [{'this': p.offset} for p in t.pages]
    # now add the next page's location and offset to that pointer
    for p in ifds:
        # move to the start of this page
        fp.seek(p['this'])
        # read the number of tags in this page
        (num_tags,) = unpack(tagnoformat, fp.read(tagnosize))

        # move forward past the tag defintions
        fp.seek(num_tags*tagsize, 1)
        # add the current location as the offset to the IFD of the next page
        p['next_ifd_offset'] = fp.tell()
        # read and save the value of the offset to the next page
        (p['next_ifd_value'],) = unpack(offsetformat, fp.read(offsetsize))

    # filter out the entry corresponding to the desired page to remove
    pageifd = [i for i in ifds if i['this'] == page.offset][0]
    # find the page pointing to this one in the IFD list
    previfd = [i for i in ifds if i['next_ifd_value'] == page.offset]
    # check for errors
    if(len(previfd) == 0):
        raise Exception('No page points to this one')
        return
    else:
        previfd = previfd[0]

    # get the strip offsets and byte counts
    offsets = page.tags['StripOffsets'].value
    bytecounts = page.tags['StripByteCounts'].value

    # iterate over the strips and erase the data
    # print('Deleting pixel data from image strips')
    for (o, b) in zip(offsets, bytecounts):
        fp.seek(o)
        fp.write(b'\0'*b)

    # iterate over all tags and erase values if necessary
    # print('Deleting tag values')
    for key, tag in page.tags.items():
        fp.seek(tag.valueoffset)
        fp.write(b'\0'*tag.count)

    offsetsize = t.tiff.ifdoffsetsize
    offsetformat = t.tiff.ifdoffsetformat
    pagebytes = (pageifd['next_ifd_offset']-pageifd['this'])+offsetsize

    # next, zero out the data in this page's header
    # print('Deleting page header')
    fp.seek(pageifd['this'])
    fp.write(b'\0'*pagebytes)

    # finally, point the previous page's IFD to this one's IFD instead
    # this will make it not show up the next time the file is opened
    fp.seek(previfd['next_ifd_offset'])
    fp.write(struct.pack(offsetformat, pageifd['next_ifd_value']))

    fp.close()


def get_csv_message():
    return 'CSV files must contain exactly two columns named "source" and "destination" in order to be processed.'


def parse_csv(file, filelist, invalid):
    try:
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            fields = ['source','destination']
            header=[h.strip().lower() for h in reader.fieldnames]
            if len(header)!=2 or any([h not in fields for h in header]):
                raise ValueError(get_csv_message())
            fieldmap={f: list(filter(lambda l: l.strip().lower()==f, reader.fieldnames))[0] for f in fields }

            for row in reader:
                src=row[fieldmap['source']]
                dest=row[fieldmap['destination']]
                file_format=detect_format(src)
                if file_format=='Aperio':
                    filelist.append({
                            'file':src,
                            'format':file_format,
                            'destination':dest
                        })
                else:
                    invalid.append({'file':src}) #not a valid Aperio file

    except ValueError as e:
        # Add this file to the list of invalid files (it wasn't a valid CSV file)
        invalid.append({'file':file})

    finally:
        return


def detect_format(filename):
    retval = None
    # print('Detecting format for ',filename)
    try:
        with open(filename, 'rb') as fp: 
            t = tiffparser.TiffFile(fp)
            description = t.pages[0].description
            # print('Description for ',filename,': ',description.replace('|','\n'))
            if description.startswith('Aperio'):
                retval = 'Aperio'
    except Exception as e:
        print('Exception in detect_format:',e)
    finally:
        return retval


def add_description(f):
    # print(f'Adding image description to {f}\n')
    try:
        f['filesize'] = os.stat(f['file']).st_size
        with open(f['file'], 'rb') as fp:
            t = tiffparser.TiffFile(fp)
            desc=re.split(';|\||\r\n?',t.pages[0].description)
            a={}
            for idx, item in enumerate(desc):
                if item.startswith('Aperio'):
                    a[item.strip()] = desc[idx+1]
                elif re.match('^[A-Za-z\s]+\s=',item):
                    parts=item.split(' = ')
                    a[parts[0]]=parts[1]
            
            f['description'] = a
    except Exception as e:
        print('Exception in add_description:', e)
    finally:
        return f

def get_filename(f):
    return f.split(os.path.sep)[-1]

def parse_files(files):
    fileinfo = [{'file': f, 'format': detect_format(f), 'destination': get_filename(f)} for f in files]
    # print('File info:',fileinfo)
    aperio=[f for f in fileinfo if f['format']=='Aperio']
    invalid=[]

    others = [parse_csv(f['file'], aperio, invalid) for f in fileinfo if f['format'] != 'Aperio']

    filelist = {'aperio':[add_description(f) for f in aperio],'invalid':invalid}

    # print('File list:', filelist)
    return filelist

def inplace_info(f):
    with open(f['file'],'r+b') as fp:
        f['writable']=fp.writable()
        t = tiffparser.TiffFile(fp)
        label = [page for page in t.pages if 'label' in page.description]
        macro = [page for page in t.pages if 'macro' in page.description]
        f['has_label']=len(label)>0
        f['has_macro']=len(macro)>0

def parse_inplace_files(files):
    fs=parse_files(files);
    [inplace_info(f) for f in fs['aperio']]
    # [print(f'{f}----\n') for f in fs['aperio']]
    filelist={'aperio':[f for f in fs['aperio'] if f['writable']==True],
              'readonly':[f for f in fs['aperio'] if f['writable']==False],
              'invalid':fs['invalid']}
    # output = {'aperio':} // check if aperio files are writable here and add that to the output
    return filelist

# File Dialog methods
# use Filebrowser objects for thread safety via signal-slot mechanisms

# filedialog = Filebrowser()
tsc = ThreadsafeCaller()

@eel.expose
def get_files(dlgtype='native',path=''):
    # c = tsc.make()
    # print(f'get_files requested in {int(QThread.currentThreadId())}')
    result=tsc.call(get_files_, dlgtype=dlgtype, path=path)
    if result['absolutePath']!=False:
        eel.set_follow_source(result['absolutePath'])
    return parse_files(result['files'])


@eel.expose
def get_dir(dlgtype='native',path=''):
    # c = tsc.make()
    # print(f'get_dir requested in {int(QThread.currentThreadId())}')
    result=tsc.call(get_dir_, dlgtype=dlgtype, path=path)
    if result['absolutePath']!=False:
        eel.set_follow_dest(result['absolutePath'])

    if result['directory'] != '':
        total, used, free = shutil.disk_usage(result['directory'])
        result['total']=total
        result['free']=free
        result['writable']=os.access(result['directory'], os.W_OK)

    return result

@eel.expose
def get_inplace_files(dlgtype='native',path=''):
    # c = tsc.make()
    # print(f'get_files requested in {int(QThread.currentThreadId())}')
    result=tsc.call(get_files_, dlgtype=dlgtype, path=path)
    if result['absolutePath']!=False:
        eel.set_follow_source(result['absolutePath'])
    return parse_inplace_files(result['files'])


@eel.expose
def get_inplace_dir(dlgtype='native',path=''):
    # c = tsc.make()
    # print(f'get_dir requested in {int(QThread.currentThreadId())}')
    result=tsc.call(get_dir_, dlgtype=dlgtype, path=path)
    if result['absolutePath']!=False:
        eel.set_follow_dest(result['absolutePath'])

    print('directory chosen:',result['directory'])
    files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(result['directory'])
                              for f in filenames if os.path.splitext(f)[1].lower() == '.svs']
    # if result['directory'] != '':
    #     total, used, free = shutil.disk_usage(result['directory'])
    #     result['total']=total
    #     result['free']=free
    #     result['writable']=os.access(result['directory'], os.W_OK)

    return parse_inplace_files(files)

@eel.expose
def test_file_dialog(dlgtype,path=''):
    return tsc.call(get_files_, dlgtype=dlgtype, path=path)

@eel.expose
def get_config_path(dlgtype='native',path=''):
    return tsc.call(get_dir_, dlgtype, path=path)

def get_files_(dlgtype='native', path=''):
    if path is None:
        path=''
    # print(f'get_files processed in {int(QThread.currentThreadId())}')
    dialog = QFileDialog(None)
    dialog.setFileMode(QFileDialog.ExistingFiles)
    dialog.setViewMode(QFileDialog.Detail)
    if dlgtype=='qt': # default to native unless qt is explicitly requested
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setNameFilters(['Aperio SVS or CSV (*.svs *.csv)'])
    if len(path)>0 and QDir(path).exists():
        dialog.setDirectory(path)

    files = []
    absolutepath=False
    if dialog.exec() == QFileDialog.Accepted:
        dlg_out = dialog.selectedFiles()
        files = dlg_out
        absolutepath=dialog.directory().absolutePath()

    output = {
        'files':files,
        'absolutePath':absolutepath
    }
    return output

def get_dir_(dlgtype='native', path='path'):
    if path is None:
        path=''
    # print(f'get_dir processed in {int(QThread.currentThreadId())}')
    dialog = QFileDialog(None)
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    if dlgtype=='qt': # default to native unless qt is explicitly requested
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setViewMode(QFileDialog.Detail)
    if len(path)>0 and QDir(path).exists():
        dialog.setDirectory(path)

    directory = ''
    absolutepath=False
    if dialog.exec() == QFileDialog.Accepted:
        dlg_out = dialog.selectedFiles()
        directory = dlg_out[0]
        absolutepath=dialog.directory().absolutePath()
    
    output = {
        'directory':directory,
        'absolutePath':absolutepath,
    }

    return output


@eel.expose
def do_copy_and_strip(files):
    copyop = CopyOp([{'source':f['source'],
                      'dest':None,
                      'id':f['id'],
                      'filesize':os.stat(f['source']).st_size,
                      'done':False,
                      'renamed':False,
                      'failed':False,
                      'failure_message':''} for f in files])

    threading.Thread(target=track_copy_progress, args=[copyop]).start()
    for index, f in enumerate(files):
        threading.Thread(target=copy_and_strip, args=[f, copyop, index]).start()
    
    return 'OK'

@eel.expose
def do_strip_in_place(file):
    try:
        print(f'Deidentifying {file}...')
        delete_associated_image(file,'label')
        delete_associated_image(file,'macro')
        print ("Stripped", file)
    except Exception as e:
        print(f'Exception deidentifying {file}: {e}')
        return 'There was a problem deleting associated images from this file'
    
    return 'ok'

@eel.expose
def check_free_space(directory):
    total, used, free = shutil.disk_usage(directory)
    return free

# Threading-related methods

# CopyOp: thread-safe file info to share data between copy and progress threads
class CopyOp(object):
    def __init__(self, start = []):
        self.lock = threading.Lock()
        self.value = start
        self.original = start
    def update(self, index, val):
        self.lock.acquire()
        try:
            for key, value in val.items():
                self.value[index][key]=value
        finally:
            self.lock.release()
    def read(self):
        self.lock.acquire()
        cp = copy.deepcopy(self.value)
        self.lock.release()
        return cp

def file_progress(b):
    progress = 0 # default value
    dest_set = b['dest']!=None
    # check both of these to make sure the new file has been created before trying to query current size
    if dest_set and os.path.isfile(b['dest']):
        progress = os.stat(b['dest']).st_size
    return progress

# def track_copy_progress: update the GUI with progress of copy operations
def track_copy_progress(copyop):
    # Start with the original file structure, in case it has already updated by the time this thread executes
    o = copyop.original
    while(any([f['done']==False for f in o])):
        try:
            n=copyop.read()
            d=[{'id':b['id'],'dest':b['dest'],'renamed':b['renamed']} 
                            for a, b in zip(o,n) if a['dest']!=b['dest'] ]
            p=[{'id':b['id'],'progress':file_progress(b)} for b in n if b['done']==False ]
            f = [b for a,b in zip(o,n) if a['done']!=b['done'] ]
            
            #send updates to javascript/GUI
            eel.update_progress({'destinations':d,'progress':p,'finalized':f})
            #copy new values to old to track what needs updating still
            o = n
        except Exception as e:
            print('Exception in track_copy_progress:',e)
        #rate limit this progress reporting
        eel.sleep(0.03)
        # ii+=1
    print('Finished tracking progress')

# copy_and_strip: single file copy/deidentify operation. 
#   to be done in a thread for concurrent I/O using CopyOp object for progress updates 
def copy_and_strip(file, copyop, index):
    # clean the paths of improper file separators for the OS 
    oldname=os.path.sep.join(re.split('[\\\/]', file['source']))
    newname=os.path.sep.join(re.split('[\\\/]', file['dest']))

    # remove the filename leaving just the path
    dest_path = os.path.sep.join(newname.split(os.path.sep)[:-1])
    try:
        # create the destination directory if necessary
        os.makedirs(dest_path, exist_ok=True)
        filename, file_extension = os.path.splitext(newname)
        # if filename.endswith('failme'):
            # raise ValueError('Cannot copy this file')
        # now the directory exists; check if the file already exists
        if not os.path.exists(newname):  # folder exists, file does not
            copyop.update(index, {'dest':newname})
            shutil.copyfile(oldname, newname)
        else:  # folder exists, file exists as well
            ii = 1
            # filename, file_extension = os.path.splitext(newname)
            while True:
                test_newname = f'{filename}({str(ii)}){file_extension}'
                if not os.path.exists(test_newname):
                    newname = test_newname
                    copyop.update(index, {'dest':newname, 'renamed':True})
                    shutil.copyfile(oldname, newname)
                    break 
                ii += 1
    
        print('Deidentifying...')
        delete_associated_image(newname,'label')
        delete_associated_image(newname,'macro')
        print ("Copied", oldname, "as", newname)
    except Exception as e:
        try:
            os.remove(newname)
        except FileNotFoundError:
            pass
        finally:
            copyop.update(index, {'failed':True,'failure_message':f'{e}'})
            print(f"Deidentification of {oldname} -> {newname} failed; removed copy of WSI file.\nException: {e}\n")
    finally:
        copyop.update(index, {'done':True})
    return


app=QApplication([]) # create QApplication to enable file dialogs

useChrome = True
if useChrome:
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
        except:
            # pass
            et=EelThread(init='web',url='app.html')
            et.start()

            w = QWebEngineView()
            w.resize(1100,800)
            w.load(QUrl('http://localhost:8000/app.html'))
            w.show()

            app.exec()
else:
    et=EelThread(init='web',url='app.html')
    et.start()

    w = QWebEngineView()
    w.resize(1100,800)
    w.load(QUrl('http://localhost:8000/app.html'))
    w.show()

    app.exec()

app.exit() # quit the QApplication





