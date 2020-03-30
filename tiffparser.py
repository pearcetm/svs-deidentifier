# tiffparser.py
# All modifications: Copyright (c) 2020, Thomas Pearce
# All rights reserved

# Modified from, with copyright info and license intact:
# tifffile.py

# Copyright (c) 2008-2020, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read TIFF(r) file (meta)data.


Data can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, SGI,
NIHImage, ImageJ, MicroManager, FluoView, ScanImage, SEQ, GEL, SVS, SCN, SIS,
ZIF, QPTIFF, NDPI, and GeoTIFF files.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.2.16


References
----------
1.  TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
    https://www.adobe.io/open/standards/TIFF.html
2.  TIFF File Format FAQ. https://www.awaresystems.be/imaging/tiff/faq.html
3.  MetaMorph Stack (STK) Image File Format.
    http://mdc.custhelp.com/app/answers/detail/a_id/18862
4.  Image File Format Description LSM 5/7 Release 6.0 (ZEN 2010).
    Carl Zeiss MicroImaging GmbH. BioSciences. May 10, 2011
5.  The OME-TIFF format.
    https://docs.openmicroscopy.org/ome-model/5.6.4/ome-tiff/
6.  UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
    http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf
7.  Micro-Manager File Formats.
    https://micro-manager.org/wiki/Micro-Manager_File_Formats
8.  Tags for TIFF and Related Specifications. Digital Preservation.
    https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
9.  ScanImage BigTiff Specification - ScanImage 2016.
    http://scanimage.vidriotechnologies.com/display/SI2016/
    ScanImage+BigTiff+Specification
10. CIPA DC-008-2016: Exchangeable image file format for digital still cameras:
    Exif Version 2.31.
    http://www.cipa.jp/std/documents/e/DC-008-Translation-2016-E.pdf
11. ZIF, the Zoomable Image File format. http://zif.photo/
12. GeoTIFF File Format https://gdal.org/drivers/raster/gtiff.html


"""

__version__ = '0.1'

import os
import sys
import pathlib
import struct
import enum
import datetime

# use tinynumpy instead to minimize build size
import tinynumpy as numpy

# import numpy

itemsizes={'<B': 1,
 '>B': 1,
 '<b1': 1,
 '>b1': 1,
 'B': 1,
 'b1': 1,
 'bool': 1,
 '<b': 1,
 '>b': 1,
 '<i1': 1,
 '>i1': 1,
 'b': 1,
 'i1': 1,
 'int8': 1,
 '<u1': 1,
 '>u1': 1,
 'u1': 1,
 'uint8': 1,
 '<h': 2,
 '>h': 2,
 '<i2': 2,
 '>i2': 2,
 'h': 2,
 'i2': 2,
 'int16': 2,
 '<H': 2,
 '>H': 2,
 '<u2': 2,
 '>u2': 2,
 'H': 2,
 'u2': 2,
 'uint16': 2,
 '<i': 4,
 '>i': 4,
 '<i4': 4,
 '>i4': 4,
 'i': 4,
 'i4': 4,
 'int32': 4,
 '<I': 4,
 '>I': 4,
 '<u4': 4,
 '>u4': 4,
 'I': 4,
 'u4': 4,
 'uint32': 4,
 '<q': 8,
 '>q': 8,
 '<i8': 8,
 '>i8': 8,
 'q': 8,
 'i8': 8,
 'int64': 8,
 '<Q': 8,
 '>Q': 8,
 '<u8': 8,
 '>u8': 8,
 'Q': 8,
 'u8': 8,
 'uint64': 8,
 '<f': 4,
 '>f': 4,
 '<f4': 4,
 '>f4': 4,
 'f': 4,
 'f4': 4,
 'float32': 4,
 '<d': 8,
 '>d': 8,
 '<f8': 8,
 '>f8': 8,
 'd': 8,
 'f8': 8,
 'float64': 8}

class dtype(object):
  def __init__(self,s):
    self.itemsize=itemsizes[s];

def getdtype(d):
  return dtype(d)

# Hack: replace minimum functionality of numpy.dtype, which is only used for looking up bytes per object
numpy.dtype = getdtype
numpy.integer = False


class lazyattr:
  """Attribute whose value is computed on first access."""

  # TODO: help() doesn't work
  __slots__ = ('func',)

  def __init__(self, func):
      self.func = func
      # self.__name__ = func.__name__
      # self.__doc__ = func.__doc__
      # self.lock = threading.RLock()

  def __get__(self, instance, owner):
      # with self.lock:
      if instance is None:
          return self
      try:
          value = self.func(instance)
      except AttributeError as exc:
          raise RuntimeError(exc)
      if value is NotImplemented:
          return getattr(super(owner, instance), self.func.__name__)
      setattr(instance, self.func.__name__, value)
      return value

class TiffFileError(Exception):
    """Exception to indicate invalid TIFF structure."""

class TiffParserError(Exception):
  """Exception to indicate parser functionality exceeded"""

class TiffFile:
    """Read image and metadata from TIFF file.

    TiffFile instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    TiffFile instances are not thread-safe.

    Attributes
    ----------
    pages : TiffPages
        Sequence of TIFF pages in file.
    series : list of TiffPageSeries
        Sequences of closely related TIFF pages. These are computed
        from OME, LSM, ImageJ, etc. metadata or based on similarity
        of page properties such as shape, dtype, and compression.
    is_flag : bool
        If True, file is of a certain format.
        Flags are: bigtiff, uniform, shaped, ome, imagej, stk, lsm, fluoview,
        nih, vista, micromanager, metaseries, mdgel, mediacy, tvips, fei,
        sem, scn, svs, scanimage, andor, epics, ndpi, pilatus, qpi.

    All attributes are read-only.

    """

    def __init__(self, arg, name=None, offset=None, size=None, multifile=True,
                 _useframes=None, **kwargs):
        """Initialize instance from file.

        Parameters
        ----------
        arg : str or open file
            Name of file or open file object.
            The file objects are closed in TiffFile.close().
        name : str
            Optional name of file in case 'arg' is a file handle.
        offset : int
            Optional start position of embedded file. By default, this is
            the current file position.
        size : int
            Optional size of embedded file. By default, this is the number
            of bytes from the 'offset' to the end of the file.
        multifile : bool
            If True (default), series may include pages from multiple files.
            Currently applies to OME-TIFF only.
        kwargs : bool
            'is_ome': If False, disable processing of OME-XML metadata.

        """
        if kwargs:
            for key in ('movie', 'fastij', 'multifile_close'):
                if key in kwargs:
                    del kwargs[key]
                    log_warning(f'TiffFile: the {key!r} argument is ignored')
            if 'pages' in kwargs:
                raise TypeError(
                    "the TiffFile 'pages' argument is no longer supported.\n\n"
                    "Use TiffFile.asarray(key=[...]) to read image data "
                    "from specific pages.\n")

            for key, value in kwargs.items():
                if key[:3] == 'is_' and key[3:] in TIFF.FILE_FLAGS:
                    if value is not None and not value:
                        setattr(self, key, bool(value))
                else:
                    raise TypeError(f'unexpected keyword argument: {key}')

        fh = FileHandle(arg, mode='rb', name=name, offset=offset, size=size)
        self._fh = fh
        self._multifile = bool(multifile)
        self._files = {fh.name: self}  # cache of TiffFiles
        self._decoders = {}  # cache of TiffPage.decode functions
        try:
            fh.seek(0)
            header = fh.read(4)
            try:
                byteorder = {b'II': '<', b'MM': '>', b'EP': '<'}[header[:2]]
            except KeyError:
                raise TiffFileError('not a TIFF file')

            version = struct.unpack(byteorder + 'H', header[2:4])[0]
            if version == 43:
                # BigTiff
                offsetsize, zero = struct.unpack(byteorder + 'HH', fh.read(4))
                if zero != 0 or offsetsize != 8:
                    raise TiffFileError('invalid BigTIFF file')
                if byteorder == '>':
                    self.tiff = TIFF.BIG_BE
                else:
                    self.tiff = TIFF.BIG_LE
            elif version == 42:
                # Classic TIFF
                if byteorder == '>':
                    self.tiff = TIFF.CLASSIC_BE
                elif kwargs.get('is_ndpi', False):
                    # NDPI uses 64 bit IFD offsets
                    # TODO: fix offsets in NDPI tags if file size > 4 GB
                    self.tiff = TIFF.NDPI_LE
                else:
                    self.tiff = TIFF.CLASSIC_LE
            else:
                raise TiffFileError('invalid TIFF file')

            # file handle is at offset to offset to first page
            self.pages = TiffPages(self)

            if self.is_lsm and (
                self.filehandle.size >= 2**32
                or self.pages[0].compression != 1
                or self.pages[1].compression != 1
            ):
                self._lsm_load_pages()
            elif self.is_scanimage and (
                not self.is_bigtiff and self.filehandle.size >= 2**31
            ):
                self.pages._load_virtual_frames()
            elif _useframes:
                self.pages.useframes = True

        except Exception:
            fh.close()
            raise

    @property
    def byteorder(self):
        return self.tiff.byteorder

    @property
    def is_bigtiff(self):
        return self.tiff.version == 43

    @property
    def filehandle(self):
        """Return file handle."""
        return self._fh

    @property
    def filename(self):
        """Return name of file handle."""
        return self._fh.name

    @lazyattr
    def fstat(self):
        """Return status of file handle as stat_result object."""
        try:
            return os.fstat(self._fh.fileno())
        except Exception:  # io.UnsupportedOperation
            return None

    def close(self):
        """Close open file handle(s)."""
        for tif in self._files.values():
            tif.filehandle.close()
        self._files = {}

    def asarray(self, key=None, series=None, out=None, maxworkers=None):
      raise TiffParserError('TiffParser does not support reading image data') 
        

    @lazyattr
    def series(self):
        """Return related pages as TiffPageSeries.

        Side effect: after calling this function, TiffFile.pages might contain
        TiffPage and TiffFrame instances.

        """
        if not self.pages:
            return []

        useframes = self.pages.useframes
        keyframe = self.pages.keyframe.index
        series = []
        for name in (
            'lsm',
            'ome',
            'imagej',
            'shaped',
            'fluoview',
            'sis',
            'uniform',
            'mdgel',
        ):
            if getattr(self, 'is_' + name, False):
                series = getattr(self, '_series_' + name)()
                break
        self.pages.useframes = useframes
        self.pages.keyframe = keyframe
        if not series:
            series = self._series_generic()

        # remove empty series, e.g. in MD Gel files
        # series = [s for s in series if product(s.shape) > 0]

        for i, s in enumerate(series):
            s.index = i
        return series

    def _series_generic(self):
        """Return image series in file.

        A series is a sequence of TiffPages with the same hash.

        """
        pages = self.pages
        pages._clear(False)
        pages.useframes = False
        if pages.cache:
            pages._load()

        result = []
        keys = []
        series = {}
        for page in pages:
            if not page.shape:  # or product(page.shape) == 0:
                continue
            key = page.hash
            if key in series:
                series[key].append(page)
            else:
                keys.append(key)
                series[key] = [page]

        for key in keys:
            pages = series[key]
            page = pages[0]
            shape = page.shape
            axes = page.axes
            if len(pages) > 1:
                shape = (len(pages),) + shape
                axes = 'I' + axes
            result.append(
                TiffPageSeries(pages, shape, page.dtype, axes, kind='Generic')
            )

        self.is_uniform = len(result) == 1
        return result

    def _series_uniform(self):
        """Return all images in file as single series."""
        page = self.pages[0]
        shape = page.shape
        axes = page.axes
        dtype = page.dtype
        validate = not (page.is_scanimage or page.is_nih)
        pages = self.pages._getlist(validate=validate)
        lenpages = len(pages)
        if lenpages > 1:
            shape = (lenpages,) + shape
            axes = 'I' + axes
        if page.is_scanimage:
            kind = 'ScanImage'
        elif page.is_nih:
            kind = 'NIHImage'
        else:
            kind = 'Uniform'
        return [TiffPageSeries(pages, shape, dtype, axes, kind=kind)]

    def _series_shaped(self):
        """Return image series in "shaped" file."""
        pages = self.pages
        pages.useframes = True
        lenpages = len(pages)

        def append(series, pages, axes, shape, reshape, name, truncated):
            # append TiffPageSeries to series
            page = pages[0]
            if not axes:
                shape = page.shape
                axes = page.axes
                if len(pages) > 1:
                    shape = (len(pages),) + shape
                    axes = 'Q' + axes
            size = product(shape)
            resize = product(reshape)
            if page.is_contiguous and resize > size and resize % size == 0:
                if truncated is None:
                    truncated = True
                axes = 'Q' + axes
                shape = (resize // size,) + shape
            try:
                axes = reshape_axes(axes, shape, reshape)
                shape = reshape
            except ValueError as exc:
                log_warning(
                    f'Shaped series: {exc.__class__.__name__}: {exc}'
                )
            series.append(
                TiffPageSeries(pages, shape, page.dtype, axes,
                               name=name, kind='Shaped', truncated=truncated)
            )

        keyframe = axes = shape = reshape = name = None
        series = []
        index = 0
        while True:
            if index >= lenpages:
                break
            # new keyframe; start of new series
            pages.keyframe = index
            keyframe = pages.keyframe
            if not keyframe.is_shaped:
                log_warning(
                    'Shaped series: invalid metadata or corrupted file')
                return None
            # read metadata
            axes = None
            shape = None
            metadata = json_description_metadata(keyframe.is_shaped)
            name = metadata.get('name', '')
            reshape = metadata['shape']
            truncated = metadata.get('truncated', None)
            if 'axes' in metadata:
                axes = metadata['axes']
                if len(axes) == len(reshape):
                    shape = reshape
                else:
                    axes = ''
                    log_warning('Shaped series: axes do not match shape')
            # skip pages if possible
            spages = [keyframe]
            size = product(reshape)
            if size > 0:
                npages, mod = divmod(size, product(keyframe.shape))
            else:
                npages = 1
                mod = 0
            if mod:
                log_warning(
                    'Shaped series: series shape does not match page shape')
                return None
            if 1 < npages <= lenpages - index:
                size *= keyframe._dtype.itemsize
                if truncated:
                    npages = 1
                elif (
                    keyframe.is_final
                    and keyframe.offset + size < pages[index + 1].offset
                ):
                    truncated = False
                else:
                    # need to read all pages for series
                    truncated = False
                    for j in range(index + 1, index + npages):
                        page = pages[j]
                        page.keyframe = keyframe
                        spages.append(page)
            append(series, spages, axes, shape, reshape, name, truncated)
            index += npages

        self.is_uniform = len(series) == 1

        return series

    def _series_imagej(self):
        """Return image series in ImageJ file."""
        # ImageJ's dimension order is always TZCYXS
        # TODO: fix loading of color, composite, or palette images
        pages = self.pages
        pages.useframes = True
        pages.keyframe = 0
        page = pages[0]
        ij = self.imagej_metadata

        def is_virtual():
            # ImageJ virtual hyperstacks store all image metadata in the first
            # page and image data are stored contiguously before the second
            # page, if any
            if not page.is_final:
                return False
            images = ij.get('images', 0)
            if images <= 1:
                return False
            offset, count = page.is_contiguous
            if (
                count != product(page.shape) * page.bitspersample // 8
                or offset + count * images > self.filehandle.size
            ):
                raise ValueError()
            # check that next page is stored after data
            if len(pages) > 1 and offset + count * images > pages[1].offset:
                return False
            return True

        try:
            isvirtual = is_virtual()
        except ValueError:
            log_warning('ImageJ series: invalid metadata or corrupted file')
            return None
        if isvirtual:
            # no need to read other pages
            pages = [page]
        else:
            pages = pages[:]

        images = ij.get('images', len(pages))
        frames = ij.get('frames', 1)
        slices = ij.get('slices', 1)
        channels = ij.get('channels', 1)
        mode = ij.get('mode', None)

        shape = []
        axes = []
        if frames > 1:
            shape.append(frames)
            axes.append('T')
        if slices > 1:
            shape.append(slices)
            axes.append('Z')
        if channels > 1 and (page.photometric != 2 or mode != 'composite'):
            shape.append(channels)
            axes.append('C')

        remain = images // (product(shape) if shape else 1)
        if remain > 1:
            shape.append(remain)
            axes.append('I')

        if page.axes[0] == 'S' and 'C' in axes:
            # planar storage, S == C, saved by Bio-Formats
            shape.extend(page.shape[1:])
            axes.extend(page.axes[1:])
        elif page.axes[0] == 'I':
            # contiguous multiple images
            shape.extend(page.shape[1:])
            axes.extend(page.axes[1:])
        elif page.axes[:2] == 'SI':
            # color-mapped contiguous multiple images
            shape = page.shape[0:1] + tuple(shape) + page.shape[2:]
            axes = list(page.axes[0]) + axes + list(page.axes[2:])
        else:
            shape.extend(page.shape)
            axes.extend(page.axes)

        truncated = (
            isvirtual
            and len(self.pages) == 1
            and page.is_contiguous[1] != (
                product(shape) * page.bitspersample // 8)
        )

        self.is_uniform = True

        return [
            TiffPageSeries(pages, shape, page.dtype, axes,
                           kind='ImageJ', truncated=truncated)
        ]

    def _series_fluoview(self):
        """Return image series in FluoView file."""
        pages = self.pages._getlist(validate=False)

        mm = self.fluoview_metadata
        mmhd = list(reversed(mm['Dimensions']))
        axes = ''.join(TIFF.MM_DIMENSIONS.get(i[0].upper(), 'Q')
                       for i in mmhd if i[1] > 1)
        shape = tuple(int(i[1]) for i in mmhd if i[1] > 1)
        self.is_uniform = True
        return [
            TiffPageSeries(pages, shape, pages[0].dtype, axes,
                           name=mm['ImageName'], kind='FluoView')
        ]

    def _series_mdgel(self):
        """Return image series in MD Gel file."""
        # only a single page, scaled according to metadata in second page
        self.pages.useframes = False
        self.pages.keyframe = 0
        md = self.mdgel_metadata
        if md['FileTag'] in (2, 128):
            dtype = numpy.dtype('float32')
            scale = md['ScalePixel']
            scale = scale[0] / scale[1]  # rational
            if md['FileTag'] == 2:
                # squary root data format
                def transform(a):
                    return a.astype('float32')**2 * scale
            else:
                def transform(a):
                    return a.astype('float32') * scale
        else:
            transform = None
        page = self.pages[0]
        self.is_uniform = False
        return [
            TiffPageSeries([page], page.shape, dtype, page.axes,
                           transform=transform, kind='MDGel')
        ]

    def _series_sis(self):
        """Return image series in Olympus SIS file."""
        pages = self.pages._getlist(validate=False)
        page = pages[0]
        lenpages = len(pages)
        md = self.sis_metadata
        if 'shape' in md and 'axes' in md:
            shape = md['shape'] + page.shape
            axes = md['axes'] + page.axes
        elif lenpages == 1:
            shape = page.shape
            axes = page.axes
        else:
            shape = (lenpages,) + page.shape
            axes = 'I' + page.axes
        self.is_uniform = True
        return [
            TiffPageSeries(pages, shape, page.dtype, axes, kind='SIS')
        ]

    def _series_ome(self):
        """Return image series in OME-TIFF file(s)."""
        from xml.etree import cElementTree as etree  # delayed import
        omexml = self.pages[0].description
        try:
            root = etree.fromstring(omexml)
        except etree.ParseError as exc:
            # TODO: test badly encoded OME-XML
            log_warning(f'OME series: {exc.__class__.__name__}: {exc}')
            try:
                omexml = omexml.decode(errors='ignore').encode()
                root = etree.fromstring(omexml)
            except Exception:
                return None

        self.pages.cache = True
        self.pages.useframes = True
        self.pages.keyframe = 0
        self.pages._load(keyframe=None)

        root_uuid = root.attrib.get('UUID', None)
        self._files = {root_uuid: self}
        dirname = self._fh.dirname
        modulo = {}
        series = []
        for element in root:
            if element.tag.endswith('BinaryOnly'):
                # TODO: load OME-XML from master or companion file
                log_warning('OME series: not an ome-tiff master file')
                break
            if element.tag.endswith('StructuredAnnotations'):
                for annot in element:
                    if not annot.attrib.get('Namespace',
                                            '').endswith('modulo'):
                        continue
                    for value in annot:
                        for modul in value:
                            for along in modul:
                                if not along.tag[:-1].endswith('Along'):
                                    continue
                                axis = along.tag[-1]
                                newaxis = along.attrib.get('Type', 'other')
                                newaxis = TIFF.AXES_LABELS[newaxis]
                                if 'Start' in along.attrib:
                                    step = float(along.attrib.get('Step', 1))
                                    start = float(along.attrib['Start'])
                                    stop = float(along.attrib['End']) + step
                                    labels = numpy.arange(start, stop, step)
                                else:
                                    labels = [
                                        label.text
                                        for label in along
                                        if label.tag.endswith('Label')
                                    ]
                                modulo[axis] = (newaxis, labels)

            if not element.tag.endswith('Image'):
                continue

            attr = element.attrib
            name = attr.get('Name', None)

            for pixels in element:
                if not pixels.tag.endswith('Pixels'):
                    continue
                attr = pixels.attrib
                # dtype = attr.get('PixelType', None)
                axes = ''.join(reversed(attr['DimensionOrder']))
                shape = idxshape = [int(attr['Size' + ax]) for ax in axes]
                size = product(shape[:-2])
                ifds = None
                spp = 1  # samples per pixel
                for data in pixels:
                    if data.tag.endswith('Channel'):
                        attr = data.attrib
                        if ifds is None:
                            spp = int(attr.get('SamplesPerPixel', spp))
                            ifds = [None] * (size // spp)
                            if spp > 1:
                                # correct channel dimension for spp
                                idxshape = [
                                    shape[i] // spp if ax == 'C' else shape[i]
                                    for i, ax in enumerate(axes)]
                        elif int(attr.get('SamplesPerPixel', 1)) != spp:
                            raise ValueError('OME series: cannot handle '
                                             'differing SamplesPerPixel')
                        continue
                    if ifds is None:
                        ifds = [None] * (size // spp)
                    if not data.tag.endswith('TiffData'):
                        continue
                    attr = data.attrib
                    ifd = int(attr.get('IFD', 0))
                    num = int(attr.get('NumPlanes', 1 if 'IFD' in attr else 0))
                    num = int(attr.get('PlaneCount', num))
                    idx = [int(attr.get('First' + ax, 0)) for ax in axes[:-2]]
                    try:
                      raise TiffParserError('series_ome is not supported by TiffParser')
                        # idx = numpy.ravel_multi_index(idx, idxshape[:-2])
                    except ValueError:
                        # ImageJ produces invalid ome-xml when cropping
                        log_warning('OME series: invalid TiffData index')
                        continue
                    for uuid in data:
                        if not uuid.tag.endswith('UUID'):
                            continue
                        if root_uuid is None and uuid.text is not None:
                            # no global UUID, use this file
                            root_uuid = uuid.text
                            self._files[root_uuid] = self._files[None]
                        elif uuid.text not in self._files:
                            if not self._multifile:
                                # abort reading multifile OME series
                                # and fall back to generic series
                                return []
                            fname = uuid.attrib['FileName']
                            try:
                                tif = TiffFile(os.path.join(dirname, fname))
                                tif.pages.cache = True
                                tif.pages.useframes = True
                                tif.pages.keyframe = 0
                                tif.pages._load(keyframe=None)
                            except (OSError, FileNotFoundError, ValueError):
                                log_warning(
                                    f'OME series: failed to read {fname!r}')
                                break
                            self._files[uuid.text] = tif
                            tif.close()
                        pages = self._files[uuid.text].pages
                        try:
                            for i in range(num if num else len(pages)):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            log_warning('OME series: index out of range')
                        # only process first UUID
                        break
                    else:
                        pages = self.pages
                        try:
                            for i in range(num if num else
                                           min(len(pages), len(ifds))):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            log_warning('OME series: index out of range')

                if all(i is None for i in ifds):
                    # skip images without data
                    continue

                # find a keyframe
                keyframe = None
                for i in ifds:
                    # try find a TiffPage
                    if i and i == i.keyframe:
                        keyframe = i
                        break
                if keyframe is None:
                    # reload a TiffPage from file
                    for i, keyframe in enumerate(ifds):
                        if keyframe:
                            keyframe.parent.pages.keyframe = keyframe.index
                            keyframe = keyframe.parent.pages[keyframe.index]
                            ifds[i] = keyframe
                            break

                # move channel axis to match PlanarConfiguration storage
                # TODO: is this a bug or a inconsistency in the OME spec?
                if spp > 1:
                    if keyframe.planarconfig == 1 and axes[-1] != 'C':
                        i = axes.index('C')
                        axes = axes[:i] + axes[i + 1:] + axes[i: i + 1]
                        shape = shape[:i] + shape[i + 1:] + shape[i: i + 1]

                # FIXME: this implementation assumes the last dimensions are
                # stored in TIFF pages. Apparently that is not always the case.
                # For now, verify that shapes of keyframe and series match
                # If not, skip series.
                if keyframe.shape != tuple(shape[-len(keyframe.shape):]):
                    log_warning(
                        'OME series: incompatible page shape %s; expected %s',
                        keyframe.shape,
                        tuple(shape[-len(keyframe.shape):])
                    )
                    del ifds
                    continue

                # set a keyframe on all IFDs
                for i in ifds:
                    if i is not None:
                        try:
                            i.keyframe = keyframe
                        except RuntimeError as exc:
                            log_warning(f'OME series: {exc}')

                series.append(
                    TiffPageSeries(ifds, shape, keyframe.dtype, axes,
                                   parent=self, name=name, kind='OME')
                )
                del ifds

        for serie in series:
            shape = list(serie.shape)
            for axis, (newaxis, labels) in modulo.items():
                i = serie.axes.index(axis)
                size = len(labels)
                if shape[i] == size:
                    serie.axes = serie.axes.replace(axis, newaxis, 1)
                else:
                    shape[i] //= size
                    shape.insert(i + 1, size)
                    serie.axes = serie.axes.replace(axis, axis + newaxis, 1)
            serie.shape = tuple(shape)

        # squeeze dimensions
        for serie in series:
            serie.shape, serie.axes = squeeze_axes(serie.shape, serie.axes)
        self.is_uniform = len(series) == 1
        return series

    def _series_lsm(self):
        """Return main and thumbnail series in LSM file."""
        lsmi = self.lsm_metadata
        axes = TIFF.CZ_LSMINFO_SCANTYPE[lsmi['ScanType']]
        if self.pages[0].photometric == 2:  # RGB; more than one channel
            axes = axes.replace('C', '').replace('XY', 'XYC')
        if lsmi.get('DimensionP', 0) > 1:
            axes += 'P'
        if lsmi.get('DimensionM', 0) > 1:
            axes += 'M'
        axes = axes[::-1]
        shape = tuple(int(lsmi[TIFF.CZ_LSMINFO_DIMENSIONS[i]]) for i in axes)
        name = lsmi.get('Name', '')
        pages = self.pages._getlist(slice(0, None, 2), validate=False)
        dtype = pages[0].dtype
        series = [
            TiffPageSeries(pages, shape, dtype, axes, name=name, kind='LSM')
        ]

        if self.pages[1].is_reduced:
            pages = self.pages._getlist(slice(1, None, 2), validate=False)
            dtype = pages[0].dtype
            cp = 1
            i = 0
            while cp < len(pages) and i < len(shape) - 2:
                cp *= shape[i]
                i += 1
            shape = shape[:i] + pages[0].shape
            axes = axes[:i] + 'CYX'
            series.append(
                TiffPageSeries(pages, shape, dtype, axes, name=name,
                               kind='LSMreduced')
            )

        self.is_uniform = False
        return series

    def _lsm_load_pages(self):
        """Load and fix all pages from LSM file."""
        # cache all pages to preserve corrected values
        pages = self.pages
        pages.cache = True
        pages.useframes = True
        # use first and second page as keyframes
        pages.keyframe = 1
        pages.keyframe = 0
        # load remaining pages as frames
        pages._load(keyframe=None)
        # fix offsets and bytecounts first
        # TODO: fix multiple conversions between lists and tuples
        self._lsm_fix_strip_offsets()
        self._lsm_fix_strip_bytecounts()
        # assign keyframes for data and thumbnail series
        keyframe = pages[0]
        for page in pages[::2]:
            page.keyframe = keyframe
        keyframe = pages[1]
        for page in pages[1::2]:
            page.keyframe = keyframe

    def _lsm_fix_strip_offsets(self):
        """Unwrap strip offsets for LSM files greater than 4 GB.

        Each series and position require separate unwrapping (undocumented).

        """
        if self.filehandle.size < 2**32:
            return

        pages = self.pages
        npages = len(pages)
        series = self.series[0]
        axes = series.axes

        # find positions
        positions = 1
        for i in 0, 1:
            if series.axes[i] in 'PM':
                positions *= series.shape[i]

        # make time axis first
        if positions > 1:
            ntimes = 0
            for i in 1, 2:
                if axes[i] == 'T':
                    ntimes = series.shape[i]
                    break
            if ntimes:
                div, mod = divmod(npages, 2 * positions * ntimes)
                if mod != 0:
                    raise RuntimeError('mod != 0')
                shape = (positions, ntimes, div, 2)
                indices = numpy.arange(product(shape)).reshape(shape)
                raise TiffParserError('_lsm_fix_strip_offsets is not supported by TiffParser')
                # indices = numpy.moveaxis(indices, 1, 0)
        else:
            indices = numpy.arange(npages).reshape(-1, 2)

        # images of reduced page might be stored first
        if pages[0]._offsetscounts[0][0] > pages[1]._offsetscounts[0][0]:
            indices = indices[..., ::-1]

        # unwrap offsets
        wrap = 0
        previousoffset = 0
        for i in indices.flat:
            page = pages[int(i)]
            dataoffsets = []
            for currentoffset in page._offsetscounts[0]:
                if currentoffset < previousoffset:
                    wrap += 2**32
                dataoffsets.append(currentoffset + wrap)
                previousoffset = currentoffset
            page._offsetscounts = tuple(dataoffsets), page._offsetscounts[1]

    def _lsm_fix_strip_bytecounts(self):
        """Set databytecounts to size of compressed data.

        The StripByteCounts tag in LSM files contains the number of bytes
        for the uncompressed data.

        """
        pages = self.pages
        if pages[0].compression == 1:
            return
        # sort pages by first strip offset
        pages = sorted(pages, key=lambda p: p._offsetscounts[0][0])
        npages = len(pages) - 1
        for i, page in enumerate(pages):
            if page.index % 2:
                continue
            offsets, bytecounts = page._offsetscounts
            if i < npages:
                lastoffset = pages[i + 1]._offsetscounts[0][0]
            else:
                # LZW compressed strips might be longer than uncompressed
                lastoffset = min(offsets[-1] + 2 * bytecounts[-1],
                                 self._fh.size)
            bytecounts = list(bytecounts)
            for j in range(len(bytecounts) - 1):
                bytecounts[j] = offsets[j + 1] - offsets[j]
            bytecounts[-1] = lastoffset - offsets[-1]
            page._offsetscounts = offsets, tuple(bytecounts)

    def __getattr__(self, name):
        """Return 'is_flag' attributes from first page."""
        if name[3:] in TIFF.FILE_FLAGS:
            if not self.pages:
                return False
            value = bool(getattr(self.pages[0], name))
            setattr(self, name, value)
            return value
        raise AttributeError(
            f'{self.__class__.__name__!r} object has no attribute {name!r}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __str__(self, detail=0, width=79):
        """Return string containing information about TiffFile.

        The detail parameter specifies the level of detail returned:

        0: file only.
        1: all series, first page of series and its tags.
        2: large tag values and file metadata.
        3: all pages.

        """
        info = [
            "TiffFile '{}'",
            format_size(self._fh.size),
            ''
            if byteorder_isnative(self.byteorder)
            else {'<': 'little-endian',
                  '>': 'big-endian'}[self.byteorder]
        ]
        if self.is_bigtiff:
            info.append('BigTiff')
        info.append(' '.join(f.lower() for f in self.flags))
        if len(self.pages) > 1:
            info.append(f'{len(self.pages)} Pages')
        if len(self.series) > 1:
            info.append(f'{len(self.series)} Series')
        if len(self._files) > 1:
            info.append(f'{len(self._files)} Files')
        info = '  '.join(info)
        info = info.replace('    ', '  ').replace('   ', '  ')
        info = info.format(
            snipstr(self._fh.name, max(12, width + 2 - len(info))))
        if detail <= 0:
            return info
        info = [info]
        info.append('\n'.join(str(s) for s in self.series))
        if detail >= 3:
            info.extend(
                    TiffPage.__str__(p, detail=detail, width=width)
                    for p in self.pages
                    if p is not None
            )
        elif self.series:
            info.extend(
                    TiffPage.__str__(s.pages[0], detail=detail, width=width)
                    for s in self.series
                    if s.pages[0] is not None
            )
        elif self.pages and self.pages[0]:
            info.append(
                TiffPage.__str__(self.pages[0], detail=detail, width=width)
            )
        if detail >= 2:
            for name in sorted(self.flags):
                if hasattr(self, name + '_metadata'):
                    m = getattr(self, name + '_metadata')
                    if m:
                        info.append(
                            '{}_METADATA\n{}'.format(
                                name.upper(),
                                pformat(m, width=width, height=detail * 12)
                            )
                        )
        return '\n\n'.join(info).replace('\n\n\n', '\n\n')

    @lazyattr
    def flags(self):
        """Return set of file flags."""
        return {
            name.lower()
            for name in sorted(TIFF.FILE_FLAGS)
            if getattr(self, 'is_' + name)
        }

    @lazyattr
    def is_mdgel(self):
        """File has MD Gel format."""
        # TODO: this likely reads the second page from file
        try:
            ismdgel = self.pages[0].is_mdgel or self.pages[1].is_mdgel
            if ismdgel:
                self.is_uniform = False
            return ismdgel
        except IndexError:
            return False

    @lazyattr
    def is_uniform(self):
        """Return if file contains a uniform series of pages."""
        # the hashes of IFDs 0, 7, and -1 are the same
        pages = self.pages
        page = pages[0]
        if page.is_scanimage or page.is_nih:
            return True
        try:
            useframes = pages.useframes
            pages.useframes = False
            h = page.hash
            for i in (1, 7, -1):
                if pages[i].aspage().hash != h:
                    return False
        except IndexError:
            return False
        finally:
            pages.useframes = useframes
        return True

    @property
    def is_appendable(self):
        """Return if pages can be appended to file without corrupting."""
        # TODO: check other formats
        return not (
            self.is_lsm
            or self.is_stk
            or self.is_imagej
            or self.is_fluoview
            or self.is_micromanager
        )

    @lazyattr
    def shaped_metadata(self):
        """Return tifffile metadata from JSON descriptions as dicts."""
        if not self.is_shaped:
            return None
        return tuple(
            json_description_metadata(s.pages[0].is_shaped)
            for s in self.series
            if s.kind.lower() == 'shaped'
        )

    @property
    def ome_metadata(self):
        """Return OME XML."""
        if not self.is_ome:
            return None
        # return xml2dict(self.pages[0].description)['OME']
        return self.pages[0].description

    @property
    def lsm_metadata(self):
        """Return LSM metadata from CZ_LSMINFO tag as dict."""
        if not self.is_lsm:
            return None
        return self.pages[0].tags[34412].value  # CZ_LSMINFO

    @lazyattr
    def stk_metadata(self):
        """Return STK metadata from UIC tags as dict."""
        if not self.is_stk:
            return None
        page = self.pages[0]
        result = {}
        result['NumberPlanes'] = page.tags[33629].count  # UIC2tag
        if page.description:
            result['PlaneDescriptions'] = page.description.split('\0')
            # result['plane_descriptions'] = stk_description_metadata(
            #    page.image_description)
        tag = page.tags.get(33628)  # UIC1tag
        if tag is not None:
            result.update(tag.value)
        tag = page.tags.get(33630)  # UIC3tag
        if tag is not None:
            result.update(tag.value)  # wavelengths
        tag = page.tags.get(33631)  # UIC4tag
        if tag is not None:
            result.update(tag.value)  # override UIC1 tags
        uic2tag = page.tags[33629].value
        result['ZDistance'] = uic2tag['ZDistance']
        result['TimeCreated'] = uic2tag['TimeCreated']
        result['TimeModified'] = uic2tag['TimeModified']
        try:
            result['DatetimeCreated'] = numpy.array(
                [julian_datetime(*dt) for dt in
                 zip(uic2tag['DateCreated'], uic2tag['TimeCreated'])],
                dtype='datetime64[ns]')
            result['DatetimeModified'] = numpy.array(
                [julian_datetime(*dt) for dt in
                 zip(uic2tag['DateModified'], uic2tag['TimeModified'])],
                dtype='datetime64[ns]')
        except ValueError as exc:
            log_warning(f'STK metadata: {exc.__class__.__name__}: {exc}')
        return result

    @lazyattr
    def imagej_metadata(self):
        """Return consolidated ImageJ metadata as dict."""
        if not self.is_imagej:
            return None
        page = self.pages[0]
        result = imagej_description_metadata(page.is_imagej)
        tag = page.tags.get(50839)  # IJMetadata
        if tag is not None:
            try:
                result.update(tag.value)
            except Exception:
                pass
        return result

    @lazyattr
    def fluoview_metadata(self):
        """Return consolidated FluoView metadata as dict."""
        if not self.is_fluoview:
            return None
        result = {}
        page = self.pages[0]
        result.update(page.tags[34361].value)  # MM_Header
        # TODO: read stamps from all pages
        result['Stamp'] = page.tags[34362].value  # MM_Stamp
        # skip parsing image description; not reliable
        # try:
        #     t = fluoview_description_metadata(page.image_description)
        #     if t is not None:
        #         result['ImageDescription'] = t
        # except Exception as exc:
        #     log_warning('FluoView metadata: '
        #                 f'failed to parse image description ({exc})'))
        return result

    @lazyattr
    def nih_metadata(self):
        """Return NIH Image metadata from NIHImageHeader tag as dict."""
        if not self.is_nih:
            return None
        return self.pages[0].tags[43314].value  # NIHImageHeader

    @lazyattr
    def fei_metadata(self):
        """Return FEI metadata from SFEG or HELIOS tags as dict."""
        if not self.is_fei:
            return None
        tags = self.pages[0].tags
        tag = tags.get(34680)  # FEI_SFEG
        if tag is not None:
            return tag.value
        tag = tags.get(34682)  # FEI_HELIOS
        if tag is not None:
            return tag.value
        return None

    @property
    def sem_metadata(self):
        """Return SEM metadata from CZ_SEM tag as dict."""
        if not self.is_sem:
            return None
        return self.pages[0].tags[34118].value

    @lazyattr
    def sis_metadata(self):
        """Return Olympus SIS metadata from SIS and INI tags as dict."""
        if not self.is_sis:
            return None
        tags = self.pages[0].tags
        result = {}
        try:
            result.update(tags[33471].value)  # OlympusINI
        except Exception:
            pass
        try:
            result.update(tags[33560].value)  # OlympusSIS
        except Exception:
            pass
        return result

    @lazyattr
    def mdgel_metadata(self):
        """Return consolidated metadata from MD GEL tags as dict."""
        for page in self.pages[:2]:
            if 33445 in page.tags:  # MDFileTag
                tags = page.tags
                break
        else:
            return None
        result = {}
        for code in range(33445, 33453):
            if code not in tags:
                continue
            name = TIFF.TAGS[code]
            result[name[2:]] = tags[code].value
        return result

    @property
    def andor_metadata(self):
        """Return Andor tags as dict."""
        return self.pages[0].andor_tags

    @property
    def epics_metadata(self):
        """Return EPICS areaDetector tags as dict."""
        return self.pages[0].epics_tags

    @property
    def tvips_metadata(self):
        """Return TVIPS tag as dict."""
        if not self.is_tvips:
            return None
        return self.pages[0].tags[37706].value

    @lazyattr
    def metaseries_metadata(self):
        """Return MetaSeries metadata from image description as dict."""
        if not self.is_metaseries:
            return None
        return metaseries_description_metadata(self.pages[0].description)

    @lazyattr
    def pilatus_metadata(self):
        """Return Pilatus metadata from image description as dict."""
        if not self.is_pilatus:
            return None
        return pilatus_description_metadata(self.pages[0].description)

    @lazyattr
    def micromanager_metadata(self):
        """Return consolidated MicroManager metadata as dict."""
        if not self.is_micromanager:
            return None
        # from file header
        result = read_micromanager_metadata(self._fh)
        # from MicroManagerMetadata tag
        result.update(self.pages[0].tags[51123].value)
        return result

    @lazyattr
    def scanimage_metadata(self):
        """Return ScanImage non-varying frame and ROI metadata as dict."""
        if not self.is_scanimage:
            return None
        result = {}
        try:
            framedata, roidata = read_scanimage_metadata(self._fh)
            result['FrameData'] = framedata
            result.update(roidata)
        except ValueError:
            pass
        # TODO: scanimage_artist_metadata
        try:
            result['Description'] = scanimage_description_metadata(
                self.pages[0].description)
        except Exception as exc:
            log_warning(f'ScanImage metadata: {exc.__class__.__name__}: {exc}')
        return result

    @property
    def geotiff_metadata(self):
        """Return GeoTIFF metadata from first page as dict."""
        if not self.is_geotiff:
            return None
        return self.pages[0].geotiff_tags


class TiffPages:
    """Sequence of TIFF image file directories (IFD chain).

    Instances of TiffPages have a state (cache, keyframe, etc.) and are not
    thread-safe.

    """

    def __init__(self, parent):
        """Initialize instance and read first TiffPage from file.

        If parent is a TiffFile, the file position must be at an offset to an
        offset to a TiffPage. If parent is a TiffPage, page offsets are read
        from the SubIFDs tag.

        """
        self.parent = None
        self.pages = []  # cache of TiffPages, TiffFrames, or their offsets
        self._indexed = False  # True if offsets to all pages were read
        self._cached = False  # True if all pages were read into cache
        self._tiffpage = TiffPage  # class used for reading pages
        self._keyframe = None  # page that is currently used as keyframe
        self._cache = False  # do not cache frames or pages (if not keyframe)
        self._nextpageoffset = None

        if isinstance(parent, TiffFile):
            # read offset to first page from current file position
            self.parent = parent
            fh = parent.filehandle
            self._nextpageoffset = fh.tell()
            offset = struct.unpack(parent.tiff.ifdoffsetformat,
                                   fh.read(parent.tiff.ifdoffsetsize))[0]
        elif 330 in parent.tags:
            # use offsets from SubIFDs tag
            self.parent = parent.parent
            fh = self.parent.filehandle
            offsets = parent.tags[330].value
            offset = offsets[0]
        else:
            self._indexed = True
            return

        if offset == 0:
            log_warning('TiffPages: file contains no pages')
            self._indexed = True
            return
        if offset >= fh.size:
            log_warning(f'TiffPages: invalid page offset {offset!r}')
            self._indexed = True
            return

        # read and cache first page
        fh.seek(offset)
        page = TiffPage(self.parent, index=0)
        self.pages.append(page)
        self._keyframe = page
        if self._nextpageoffset is None:
            # offsets from SubIFDs tag
            self.pages.extend(offsets[1:])
            self._indexed = True
            self._cached = True

    @property
    def cache(self):
        """Return if pages/frames are currently being cached."""
        return self._cache

    @cache.setter
    def cache(self, value):
        """Enable or disable caching of pages/frames. Clear cache if False."""
        value = bool(value)
        if self._cache and not value:
            self._clear()
        self._cache = value

    @property
    def useframes(self):
        """Return if currently using TiffFrame (True) or TiffPage (False)."""
        return self._tiffpage == TiffFrame and TiffFrame is not TiffPage

    @useframes.setter
    def useframes(self, value):
        """Set to use TiffFrame (True) or TiffPage (False)."""
        self._tiffpage = TiffFrame if value else TiffPage

    @property
    def keyframe(self):
        """Return current keyframe."""
        return self._keyframe

    @keyframe.setter
    def keyframe(self, index):
        """Set current keyframe. Load TiffPage from file if necessary."""
        index = int(index)
        if index < 0:
            index %= len(self)
        if self._keyframe.index == index:
            return
        if index == 0:
            self._keyframe = self.pages[0]
            return
        if self._indexed or index < len(self.pages):
            page = self.pages[index]
            if isinstance(page, TiffPage):
                self._keyframe = page
                return
            if isinstance(page, TiffFrame):
                # remove existing TiffFrame
                self.pages[index] = page.offset
        # load TiffPage from file
        tiffpage = self._tiffpage
        self._tiffpage = TiffPage
        try:
            self._keyframe = self._getitem(index)
        finally:
            self._tiffpage = tiffpage
        # always cache keyframes
        self.pages[index] = self._keyframe

    @property
    def next_page_offset(self):
        """Return offset where offset to a new page can be stored."""
        if not self._indexed:
            self._seek(-1)
        return self._nextpageoffset

    def _load(self, keyframe=True):
        """Read all remaining pages from file."""
        if self._cached:
            return
        pages = self.pages
        if not pages:
            return
        if not self._indexed:
            self._seek(-1)
        if not self._cache:
            return
        fh = self.parent.filehandle
        if keyframe is not None:
            keyframe = self._keyframe
        for i, page in enumerate(pages):
            if isinstance(page, (int, numpy.integer)):
                fh.seek(page)
                page = self._tiffpage(self.parent, index=i, keyframe=keyframe)
                pages[i] = page
        self._cached = True

    def _load_virtual_frames(self):
        """Calculate virtual TiffFrames."""
        pages = self.pages
        try:
            if len(pages) > 1:
                raise ValueError('pages already loaded')
            page = pages[0]
            bytecounts = page._offsetscounts[1]
            if len(bytecounts) != 1:
                raise ValueError('data not contiguous')
            self._seek(4)
            delta = pages[2] - pages[1]
            if pages[3] - pages[2] != delta or pages[4] - pages[3] != delta:
                raise ValueError('page offsets not equidistant')
            page1 = self._getitem(1, validate=page.hash)
            offsetoffset = page1._offsetscounts[0][0] - page1.offset
            if offsetoffset < 0 or offsetoffset > delta:
                raise ValueError('page offsets not equidistant')
            pages = [page, page1]
            filesize = self.parent.filehandle.size - delta
            for index, offset in enumerate(range(page1.offset + delta,
                                                 filesize, delta)):
                offsets = [offset + offsetoffset]
                offset = offset if offset < 2**31 else None
                pages.append(
                    TiffFrame(
                        parent=page.parent,
                        index=index + 2,
                        offset=None,
                        offsets=offsets,
                        bytecounts=bytecounts,
                        keyframe=page
                    )
                )
            self.pages = pages
            self._cache = True
            self._cached = True
            self._indexed = True
        except Exception as exc:
            log_warning(f'TiffPages: failed to load virtual frames: {exc}')

    def _clear(self, fully=True):
        """Delete all but first page from cache. Set keyframe to first page."""
        pages = self.pages
        if not pages:
            return
        self._keyframe = pages[0]
        if fully:
            # delete all but first TiffPage/TiffFrame
            for i, page in enumerate(pages[1:]):
                if not isinstance(page, int) and page.offset is not None:
                    pages[i + 1] = page.offset
        elif TiffFrame is not TiffPage:
            # delete only TiffFrames
            for i, page in enumerate(pages):
                if isinstance(page, TiffFrame) and page.offset is not None:
                    pages[i] = page.offset
        self._cached = False

    def _seek(self, index, maxpages=None):
        """Seek file to offset of page specified by index."""
        pages = self.pages
        lenpages = len(pages)
        if lenpages == 0:
            raise IndexError('index out of range')

        fh = self.parent.filehandle
        if fh.closed:
            raise ValueError('seek of closed file')

        if self._indexed or 0 <= index < lenpages:
            page = pages[index]
            offset = page if isinstance(page, int) else page.offset
            fh.seek(offset)
            return

        tiff = self.parent.tiff
        offsetformat = tiff.ifdoffsetformat
        offsetsize = tiff.ifdoffsetsize
        tagnoformat = tiff.tagnoformat
        tagnosize = tiff.tagnosize
        tagsize = tiff.tagsize
        unpack = struct.unpack

        page = pages[-1]
        offset = page if isinstance(page, int) else page.offset

        if maxpages is None:
            maxpages = 2**22
        while lenpages < maxpages:
            # read offsets to pages from file until index is reached
            fh.seek(offset)
            # skip tags
            try:
                tagno = unpack(tagnoformat, fh.read(tagnosize))[0]
                if tagno > 4096:
                    raise TiffFileError(f'suspicious number of tags {tagno!r}')
            except Exception:
                log_warning(
                    'TiffPages: corrupted tag list of page '
                    f'{lenpages} @ {offset}',
                )
                del pages[-1]
                lenpages -= 1
                self._indexed = True
                break
            self._nextpageoffset = offset + tagnosize + tagno * tagsize
            fh.seek(self._nextpageoffset)

            # read offset to next page
            offset = unpack(offsetformat, fh.read(offsetsize))[0]
            if offset == 0:
                self._indexed = True
                break
            if offset >= fh.size:
                log_warning(f'TiffPages: invalid page offset {offset!r}')
                self._indexed = True
                break

            pages.append(offset)
            lenpages += 1
            if 0 <= index < lenpages:
                break

            # detect some circular references
            if lenpages == 100:
                for p in pages[:-1]:
                    if offset == (p if isinstance(p, int) else p.offset):
                        raise TiffFileError('invalid circular IFD reference')

        if index >= lenpages:
            raise IndexError('index out of range')

        page = pages[index]
        fh.seek(page if isinstance(page, int) else page.offset)

    def _getlist(self, key=None, useframes=True, validate=True):
        """Return specified pages as list of TiffPages or TiffFrames.

        The first item is a TiffPage, and is used as a keyframe for
        following TiffFrames.

        """
        getitem = self._getitem
        _useframes = self.useframes

        if key is None:
            key = iter(range(len(self)))
        elif isinstance(key, Iterable):
            key = iter(key)
        elif isinstance(key, slice):
            start, stop, _ = key.indices(2**31 - 1)
            if not self._indexed and max(stop, start) > len(self.pages):
                self._seek(-1)
            key = iter(range(*key.indices(len(self.pages))))
        elif isinstance(key, (int, numpy.integer)):
            # return single TiffPage
            self.useframes = False
            if key == 0:
                return [self.pages[key]]
            try:
                return [getitem(key)]
            finally:
                self.useframes = _useframes
        else:
            raise TypeError('key must be an integer, slice, or iterable')

        # use first page as keyframe
        keyframe = self._keyframe
        self.keyframe = next(key)
        if validate:
            validate = self._keyframe.hash
        if useframes:
            self.useframes = True
        try:
            pages = [getitem(i, validate) for i in key]
            pages.insert(0, self._keyframe)
        finally:
            # restore state
            self._keyframe = keyframe
            if useframes:
                self.useframes = _useframes

        return pages

    def _getitem(self, key, validate=False):
        """Return specified page from cache or file."""
        key = int(key)
        pages = self.pages

        if key < 0:
            key %= len(self)
        elif self._indexed and key >= len(pages):
            raise IndexError(f'index {key} out of range({len(pages)})')

        if key < len(pages):
            page = pages[key]
            if self._cache:
                if not isinstance(page, (int, numpy.integer)):
                    if validate and validate != page.hash:
                        raise RuntimeError('page hash mismatch')
                    return page
            elif isinstance(page, (TiffPage, self._tiffpage)):
                if validate and validate != page.hash:
                    raise RuntimeError('page hash mismatch')
                return page

        self._seek(key)
        page = self._tiffpage(self.parent, index=key, keyframe=self._keyframe)
        if validate and validate != page.hash:
            raise RuntimeError('page hash mismatch')
        if self._cache:
            pages[key] = page
        return page

    def __getitem__(self, key):
        """Return specified page(s)."""
        pages = self.pages
        getitem = self._getitem

        if isinstance(key, (int, numpy.integer)):
            if key == 0:
                return pages[key]
            return getitem(key)

        if isinstance(key, slice):
            start, stop, _ = key.indices(2**31 - 1)
            if not self._indexed and max(stop, start) > len(pages):
                self._seek(-1)
            return [getitem(i) for i in range(*key.indices(len(pages)))]

        if isinstance(key, Iterable):
            return [getitem(k) for k in key]

        raise TypeError('key must be an integer, slice, or iterable')

    def __iter__(self):
        """Return iterator over all pages."""
        i = 0
        while True:
            try:
                yield self._getitem(i)
                i += 1
            except IndexError:
                break
        if self._cache:
            self._cached = True

    def __bool__(self):
        """Return True if file contains any pages."""
        return len(self.pages) > 0

    def __len__(self):
        """Return number of pages in file."""
        if not self._indexed:
            self._seek(-1)
        return len(self.pages)


class TiffPage:
    """TIFF image file directory (IFD).

    Attributes
    ----------
    index : int
        Index of the page in file.
    dtype : numpy.dtype or None
        Data type (native byte order) of the image in IFD.
    shape : tuple of int
        Dimensions of the image in IFD, as returned by asarray.
    axes : str
        Axes label codes for each dimension in shape:
        'X' width,
        'Y' height,
        'S' sample,
        'I' image series|page|plane,
        'Z' depth,
        'C' color|em-wavelength|channel,
        'E' ex-wavelength|lambda,
        'T' time,
        'R' region|tile,
        'A' angle,
        'P' phase,
        'H' lifetime,
        'L' exposure,
        'V' event,
        'Q' unknown,
        '_' missing
    tags : TiffTags
        Multidict like interface to tags in IFD.
    colormap : numpy.ndarray
        Color look up table, if exists.
    shaped : tuple of int
        Normalized 6 dimensional shape of the image in IFD:
        0 : number planes (stk), images (ij), or 1.
        1 : separate samplesperpixel or 1.
        2 : imagedepth Z (sgi) or 1.
        3 : imagelength Y.
        4 : imagewidth X.
        5 : contig samplesperpixel or 1.

    All attributes are read-only.

    """
    # default properties; will be updated from tags
    subfiletype = 0
    imagewidth = 0
    imagelength = 0
    imagedepth = 1
    tilewidth = 0
    tilelength = 0
    tiledepth = 1
    bitspersample = 1
    samplesperpixel = 1
    sampleformat = 1
    rowsperstrip = 2**32 - 1
    compression = 1
    planarconfig = 1
    fillorder = 1
    photometric = 0
    predictor = 1
    extrasamples = 1
    colormap = None
    software = ''
    description = ''
    description1 = ''
    nodata = 0

    def __init__(self, parent, index, keyframe=None):
        """Initialize instance from file.

        The file handle position must be at offset to a valid IFD.

        """
        self.parent = parent
        self.index = index
        self.shape = ()
        self.shaped = ()
        self.dtype = None
        self._dtype = None
        self.axes = ''
        self.tags = tags = TiffTags()
        self.dataoffsets = ()
        self.databytecounts = ()

        tiff = parent.tiff

        # read TIFF IFD structure and its tags from file
        fh = parent.filehandle
        self.offset = fh.tell()  # offset to this IFD
        try:
            tagno = struct.unpack(
                tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
            if tagno > 4096:
                raise TiffFileError(
                    f'TiffPage {self.index}: suspicious number of tags'
                )
        except Exception:
            raise TiffFileError(
                f'TiffPage {self.index}: '
                f'corrupted tag list at offset {self.offset}'
            )

        tagoffset = self.offset + tiff.tagnosize  # fh.tell()
        tagsize = tiff.tagsize
        tagindex = -tagsize

        data = fh.read(tagsize * tagno)

        for _ in range(tagno):
            tagindex += tagsize
            try:
                tag = TiffTag(parent, data[tagindex: tagindex + tagsize],
                              tagoffset + tagindex)
            except TiffFileError as exc:
                log_warning(
                    f'TiffPage {self.index}: {exc.__class__.__name__}: {exc}'
                )
                continue
            tags.add(tag)

        if not tags:
            return  # found in FIBICS

        for code, name in TIFF.TAG_ATTRIBUTES.items():
            tag = tags.get(code)
            if tag is not None:
                if code in (270, 305) and not isinstance(tag.value, str):
                    # wrong string type for software or description
                    continue
                setattr(self, name, tag.value)

        tag = tags.get(270, index=1)
        if tag:
            self.description1 = tag.value

        tag = tags.get(255)  # SubfileType
        if tag and self.subfiletype == 0:
            if tag.value == 2:
                self.subfiletype = 0b1  # reduced image
            elif tag.value == 3:
                self.subfiletype = 0b10  # multi-page

        # consolidate private tags; remove them from self.tags
        # if self.is_andor:
        #     self.andor_tags
        # elif self.is_epics:
        #     self.epics_tags
        # elif self.is_ndpi:
        #     self.ndpi_tags
        # if self.is_sis and 34853 in tags:
        #     # TODO: can't change tag.name
        #     tags[34853].name = 'OlympusSIS2'

        if self.is_lsm or (self.index and self.parent.is_lsm):
            # correct non standard LSM bitspersample tags
            tags[258]._fix_lsm_bitspersample(self)
            if self.compression == 1 and self.predictor != 1:
                # work around bug in LSM510 software
                self.predictor = 1

        elif self.is_vista or (self.index and self.parent.is_vista):
            # ISS Vista writes wrong ImageDepth tag
            self.imagedepth = 1

        elif self.is_stk:
            tag = tags.get(33628)  # UIC1tag
            if tag is not None and not tag.value:
                # read UIC1tag now that plane count is known
                fh.seek(tag.valueoffset)
                tag.value = read_uic1tag(
                    fh,
                    tiff.byteorder,
                    tag.dtype,
                    tag.count,
                    None,
                    tags[33629].count  # UIC2tag
                )

        if 50839 in tags:
            # decode IJMetadata tag
            try:
                tags[50839].value = imagej_metadata(
                    tags[50839].value,
                    tags[50838].value,  # IJMetadataByteCounts
                    tiff.byteorder)
            except Exception as exc:
                log_warning(
                    f'TiffPage {self.index}: {exc.__class__.__name__}: {exc}'
                )

        # BitsPerSample
        tag = tags.get(258)
        if tag is not None:
            if tag.count == 1:
                self.bitspersample = tag.value
            else:
                # LSM might list more items than samplesperpixel
                value = tag.value[:self.samplesperpixel]
                if any(v - value[0] for v in value):
                    self.bitspersample = value
                else:
                    self.bitspersample = value[0]

        # SampleFormat
        tag = tags.get(339)
        if tag is not None:
            if tag.count == 1:
                self.sampleformat = tag.value
            else:
                value = tag.value[:self.samplesperpixel]
                if any(v - value[0] for v in value):
                    self.sampleformat = value
                else:
                    self.sampleformat = value[0]

        if 322 in tags:  # TileWidth
            self.rowsperstrip = None
        elif 257 in tags:  # ImageLength
            if 278 not in tags or tags[278].count > 1:  # RowsPerStrip
                self.rowsperstrip = self.imagelength
            self.rowsperstrip = min(self.rowsperstrip, self.imagelength)
            # self.stripsperimage = int(math.floor(
            #    float(self.imagelength + self.rowsperstrip - 1) /
            #    self.rowsperstrip))

        # determine dtype
        dtype = self.sampleformat, self.bitspersample
        dtype = TIFF.SAMPLE_DTYPES.get(dtype, None)
        if dtype is not None:
            dtype = numpy.dtype(dtype)
        self.dtype = self._dtype = dtype

        # determine shape of data
        imagelength = self.imagelength
        imagewidth = self.imagewidth
        imagedepth = self.imagedepth
        samplesperpixel = self.samplesperpixel

        if self.is_stk:
            if imagedepth != 1:
                raise ValueError('STK imagedepth must be 1')
            tag = tags[33629]  # UIC2tag
            uictag = tag.value
            planes = tag.count
            if self.planarconfig == 1:
                self.shaped = (
                    planes,
                    1,
                    1,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                )
                if samplesperpixel == 1:
                    self.shape = (planes, imagelength, imagewidth)
                    self.axes = 'YX'
                else:
                    self.shape = (
                        planes,
                        imagelength,
                        imagewidth,
                        samplesperpixel,
                    )
                    self.axes = 'YXS'
            else:
                self.shaped = (
                    planes,
                    samplesperpixel,
                    1,
                    imagelength,
                    imagewidth,
                    1,
                )
                if samplesperpixel == 1:
                    self.shape = (planes, imagelength, imagewidth)
                    self.axes = 'YX'
                else:
                    self.shape = (
                        planes,
                        samplesperpixel,
                        imagelength,
                        imagewidth,
                    )
                    self.axes = 'SYX'
            # detect type of series
            if planes == 1:
                self.shape = self.shape[1:]
            elif numpy.all(uictag['ZDistance'] != 0):
                self.axes = 'Z' + self.axes
            elif numpy.all(numpy.diff(uictag['TimeCreated']) != 0):
                self.axes = 'T' + self.axes
            else:
                self.axes = 'I' + self.axes
        elif self.photometric == 2 or samplesperpixel > 1:  # PHOTOMETRIC.RGB
            if self.planarconfig == 1:
                self.shaped = (
                    1,
                    1,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                )
                if imagedepth == 1:
                    self.shape = (imagelength, imagewidth, samplesperpixel)
                    self.axes = 'YXS'
                else:
                    self.shape = (
                        imagedepth,
                        imagelength,
                        imagewidth,
                        samplesperpixel,
                    )
                    self.axes = 'ZYXS'
            else:
                self.shaped = (
                    1,
                    samplesperpixel,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    1,
                )
                if imagedepth == 1:
                    self.shape = (samplesperpixel, imagelength, imagewidth)
                    self.axes = 'SYX'
                else:
                    self.shape = (
                        samplesperpixel,
                        imagedepth,
                        imagelength,
                        imagewidth,
                    )
                    self.axes = 'SZYX'
        else:
            self.shaped = (1, 1, imagedepth, imagelength, imagewidth, 1)
            if imagedepth == 1:
                self.shape = (imagelength, imagewidth)
                self.axes = 'YX'
            else:
                self.shape = (imagedepth, imagelength, imagewidth)
                self.axes = 'ZYX'

        # dataoffsets and databytecounts
        if 324 in tags:  # TileOffsets
            self.dataoffsets = tags[324].value
        elif 273 in tags:  # StripOffsets
            self.dataoffsets = tags[273].value
        if 325 in tags:  # TileByteCounts
            self.databytecounts = tags[325].value
        elif 279 in tags:  # StripByteCounts
            self.databytecounts = tags[279].value
        else:
            self.databytecounts = (
                product(self.shape) * (self.bitspersample // 8),)
            if self.compression != 1:
                log_warning(
                    f'TiffPage {self.index}: ByteCounts tag is missing'
                )

        tag = tags.get(42113)  # GDAL_NODATA
        if tag is not None:
            try:
                pytype = type(dtype.type(0).item())
                self.nodata = pytype(tag.value)
            except Exception:
                pass

    @lazyattr
    def decode(self):
      raise TiffParserError('decode is not supported by TiffParser')
    

    def asarray(self, out=None, squeeze=True, lock=None, reopen=True,
                maxsize=None, maxworkers=None):
      raise TiffParserError('asarray is not supported by TiffParser')


    def asrgb(self, uint8=False, alpha=None, colormap=None,
              dmin=None, dmax=None, **kwargs):
      raise TiffParserError('Decoding is not supported by TiffParser')
    

    def _gettags(self, codes=None, lock=None):
        """Return list of (code, TiffTag)."""
        return [(tag.code, tag) for tag in self.tags
                if codes is None or tag.code in codes]

    def aspage(self):
        """Return self."""
        return self

    @property
    def keyframe(self):
        """Return keyframe, self."""
        return self

    @keyframe.setter
    def keyframe(self, index):
        """Set keyframe, NOP."""
        return

    @lazyattr
    def pages(self):
        """Return sequence of sub-pages (SubIFDs)."""
        if 330 not in self.tags:
            return ()
        return TiffPages(self)

    @lazyattr
    def hash(self):
        """Return checksum to identify pages in same series.

        Pages with the same hash can use the same decode function.

        """
        return hash(
            self.shaped + (
                self.parent.byteorder,
                self.tilewidth,
                self.tilelength,
                self.tiledepth,
                self.sampleformat,
                self.bitspersample,
                self.rowsperstrip,
                self.fillorder,
                self.predictor,
                self.extrasamples,
                self.photometric,
                self.planarconfig,
                self.compression,
            ))

    @lazyattr
    def maxworkers(self):
        """Return maximum number of threads for decoding strips ot tiles."""
        if len(self._offsetscounts[0]) < 4:
            return 1
        if self.compression != 1 or self.fillorder != 1 or self.predictor != 1:
            if imagecodecs is not None:
                return min(TIFF.MAXWORKERS, len(self._offsetscounts[0]))
        return 2  # optimum for large number of uncompressed tiles

    @lazyattr
    def _offsetscounts(self):
        """Return simplified offsets and bytecounts."""
        if self.is_contiguous:
            offset, bytecount = self.is_contiguous
            return ((offset,), (bytecount,))
        return self.dataoffsets, self.databytecounts

    @lazyattr
    def is_contiguous(self):
        """Return offset and size of contiguous data, else None.

        Excludes prediction and fill_order.

        """
        if self.compression != 1 or self.bitspersample not in (8, 16, 32, 64):
            return None
        if 322 in self.tags:  # TileWidth
            if (
                self.imagewidth != self.tilewidth
                or self.imagelength % self.tilelength
                or self.tilewidth % 16
                or self.tilelength % 16
            ):
                return None
            if (
                32997 in self.tags and  # ImageDepth
                32998 in self.tags and  # TileDepth
                (
                    self.imagelength != self.tilelength or
                    self.imagedepth % self.tiledepth
                )
            ):
                return None
        offsets = self.dataoffsets
        bytecounts = self.databytecounts
        if len(offsets) == 1:
            return offsets[0], bytecounts[0]
        if self.is_stk or self.is_lsm:
            return offsets[0], sum(bytecounts)
        if all(
            bytecounts[i] != 0 and offsets[i] + bytecounts[i] == offsets[i + 1]
            for i in range(len(offsets) - 1)
        ):
            return offsets[0], sum(bytecounts)
        return None

    @lazyattr
    def is_final(self):
        """Return if page's image data are stored in final form.

        Excludes byte-swapping.

        """
        return (
            self.is_contiguous
            and self.fillorder == 1
            and self.predictor == 1
            and not self.is_subsampled
        )

    @lazyattr
    def is_memmappable(self):
        """Return if page's image data in file can be memory-mapped."""
        return (
            self.parent.filehandle.is_file
            and self.is_final
            # and (self.bitspersample == 8 or self.parent.isnative)
            # aligned?
            and self.is_contiguous[0] % self.dtype.itemsize == 0
        )

    def __str__(self, detail=0, width=79):
        """Return string containing information about TiffPage."""
        if self.keyframe != self:
            return TiffFrame.__str__(self, detail, width)
        attr = ''
        for name in ('memmappable', 'final', 'contiguous'):
            attr = getattr(self, 'is_' + name)
            if attr:
                attr = name.upper()
                break

        def tostr(name, skip=1):
            obj = getattr(self, name)
            try:
                value = getattr(obj, 'name')
            except AttributeError:
                return ''
            if obj != skip:
                return value
            return ''

        info = '  '.join(
            s.lower()
            for s in (
                'x'.join(str(i) for i in self.shape),
                '{}{}'.format(
                    TIFF.SAMPLEFORMAT(self.sampleformat).name,
                    self.bitspersample,
                ),
                ' '.join(
                    i
                    for i in (
                        TIFF.PHOTOMETRIC(self.photometric).name,
                        'REDUCED' if self.is_reduced else '',
                        'MASK' if self.is_mask else '',
                        'TILED' if self.is_tiled else '',
                        tostr('compression'),
                        tostr('planarconfig'),
                        tostr('predictor'),
                        tostr('fillorder'),
                    )
                    + tuple(f.upper() for f in self.flags)
                    + (attr,)
                    if i
                ),
            )
            if s
        )
        info = f'TiffPage {self.index} @{self.offset}  {info}'
        if detail <= 0:
            return info
        info = [info, self.tags.__str__(detail+1, width=width)]
        if detail > 1:
            for name in ('ndpi',):
                name = name + '_tags'
                attr = getattr(self, name, False)
                if attr:
                    info.append(f'{name.upper()}\n{pformat(attr)}')
        if detail > 3:
            try:
                info.append('DATA\n{}'.format(
                    pformat(self.asarray(), width=width, height=detail * 8)
                ))
            except Exception:
                pass
        return '\n\n'.join(info)

    @lazyattr
    def flags(self):
        """Return set of flags."""
        return {
            name.lower()
            for name in sorted(TIFF.FILE_FLAGS)
            if getattr(self, 'is_' + name)
        }

    @property
    def ndim(self):
        """Return number of array dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Return number of elements in array."""
        return product(self.shape)

    @lazyattr
    def andor_tags(self):
        """Return consolidated metadata from Andor tags as dict."""
        if not self.is_andor:
            return None
        result = {'Id': self.tags[4864].value}  # AndorId
        for tag in self.tags:  # list(self.tags.values()):
            code = tag.code
            if not 4864 < code < 5031:
                continue
            name = tag.name
            name = name[5:] if len(name) > 5 else name
            result[name] = tag.value
            # del self.tags[code]
        return result

    @lazyattr
    def epics_tags(self):
        """Return consolidated metadata from EPICS areaDetector tags as dict.

        """
        if not self.is_epics:
            return None
        result = {}
        for tag in self.tags:  # list(self.tags.values()):
            code = tag.code
            if not 65000 <= code < 65500:
                continue
            value = tag.value
            if code == 65000:
                result['timeStamp'] = datetime.datetime.fromtimestamp(
                    float(value))
            elif code == 65001:
                result['uniqueID'] = int(value)
            elif code == 65002:
                result['epicsTSSec'] = int(value)
            elif code == 65003:
                result['epicsTSNsec'] = int(value)
            else:
                key, value = value.split(':', 1)
                result[key] = astype(value)
            # del self.tags[code]
        return result

    @lazyattr
    def ndpi_tags(self):
        """Return consolidated metadata from Hamamatsu NDPI as dict."""
        if not self.is_ndpi:
            return None
        tags = self.tags
        result = {}
        for name in ('Make', 'Model', 'Software'):
            result[name] = tags[name].value
        for code, name in TIFF.NDPI_TAGS.items():
            if code in tags:
                result[name] = tags[code].value
                # del tags[code]
        return result

    @lazyattr
    def geotiff_tags(self):
        """Return consolidated metadata from GeoTIFF tags as dict."""
        if not self.is_geotiff:
            return None
        tags = self.tags

        gkd = tags[34735].value  # GeoKeyDirectoryTag
        if gkd[0] != 1:
            log_warning('GeoTIFF tags: invalid GeoKeyDirectoryTag')
            return {}

        result = {
            'KeyDirectoryVersion': gkd[0],
            'KeyRevision': gkd[1],
            'KeyRevisionMinor': gkd[2],
            # 'NumberOfKeys': gkd[3],
        }
        # deltags = ['GeoKeyDirectoryTag']
        geokeys = TIFF.GEO_KEYS
        geocodes = TIFF.GEO_CODES
        for index in range(gkd[3]):
            try:
                keyid, tagid, count, offset = gkd[4 + index * 4: index * 4 + 8]
            except Exception as exc:
                log_warning(f'GeoTIFF tags: {exc}')
                continue
            keyid = geokeys.get(keyid, keyid)
            if tagid == 0:
                value = offset
            else:
                try:
                    value = tags[tagid].value[offset: offset + count]
                except KeyError:
                    log_warning(f'GeoTIFF tags: {tagid} not found')
                    continue
                if tagid == 34737 and count > 1 and value[-1] == '|':
                    value = value[:-1]
                value = value if count > 1 else value[0]
            if keyid in geocodes:
                try:
                    value = geocodes[keyid](value)
                except Exception:
                    pass
            result[keyid] = value

        tag = tags.get(33920)  # IntergraphMatrixTag
        if tag is not None:
            value = numpy.array(tag.value)
            if len(value) == 16:
                value = value.reshape((4, 4)).tolist()
            result['IntergraphMatrix'] = value

        tag = tags.get(33550)  # ModelPixelScaleTag
        if tag is not None:
            result['ModelPixelScale'] = numpy.array(tag.value).tolist()

        tag = tags.get(33922)  # ModelTiepointTag
        if tag is not None:
            value = numpy.array(tag.value).reshape((-1, 6)).squeeze().tolist()
            result['ModelTiepoint'] = value

        tag = tags.get(34264)  # ModelTransformationTag
        if tag is not None:
            value = numpy.array(tag.value).reshape((4, 4)).tolist()
            result['ModelTransformation'] = value

        # if 33550 in tags and 33922 in tags:
        #     sx, sy, sz = tags[33550].value  # ModelPixelScaleTag
        #     tiepoints = tags[33922].value  # ModelTiepointTag
        #     transforms = []
        #     for tp in range(0, len(tiepoints), 6):
        #         i, j, k, x, y, z = tiepoints[tp:tp+6]
        #         transforms.append([
        #             [sx, 0.0, 0.0, x - i * sx],
        #             [0.0, -sy, 0.0, y + j * sy],
        #             [0.0, 0.0, sz, z - k * sz],
        #             [0.0, 0.0, 0.0, 1.0]])
        #     if len(tiepoints) == 6:
        #         transforms = transforms[0]
        #     result['ModelTransformation'] = transforms

        tag = tags.get(50844)  # RPCCoefficientTag
        if tag is not None:
            rpcc = tag.value
            result['RPCCoefficient'] = {
                'ERR_BIAS': rpcc[0],
                'ERR_RAND': rpcc[1],
                'LINE_OFF': rpcc[2],
                'SAMP_OFF': rpcc[3],
                'LAT_OFF': rpcc[4],
                'LONG_OFF': rpcc[5],
                'HEIGHT_OFF': rpcc[6],
                'LINE_SCALE': rpcc[7],
                'SAMP_SCALE': rpcc[8],
                'LAT_SCALE': rpcc[9],
                'LONG_SCALE': rpcc[10],
                'HEIGHT_SCALE': rpcc[11],
                'LINE_NUM_COEFF': rpcc[12:33],
                'LINE_DEN_COEFF ': rpcc[33:53],
                'SAMP_NUM_COEFF': rpcc[53:73],
                'SAMP_DEN_COEFF': rpcc[73:],
            }
        return result

    @property
    def is_reduced(self):
        """Page is reduced image of another image."""
        return self.subfiletype & 0b1

    @property
    def is_multipage(self):
        """Page is part of multi-page image."""
        return self.subfiletype & 0b10

    @property
    def is_mask(self):
        """Page is transparency mask for another image."""
        return self.subfiletype & 0b100

    @property
    def is_mrc(self):
        """Page is part of Mixed Raster Content."""
        return self.subfiletype & 0b1000

    @property
    def is_tiled(self):
        """Page contains tiled image."""
        return 322 in self.tags  # TileWidth

    @property
    def is_subsampled(self):
        """Page contains chroma subsampled image."""
        tag = self.tags.get(530)  # YCbCrSubSampling
        if tag is not None:
            return tag.value != (1, 1)
        return (
            self.compression == 7
            and self.planarconfig == 1
            and self.photometric in (2, 6)
        )

    @lazyattr
    def is_imagej(self):
        """Return ImageJ description if exists, else None."""
        for description in (self.description, self.description1):
            if not description:
                return None
            if description[:7] == 'ImageJ=':
                return description
        return None

    @lazyattr
    def is_shaped(self):
        """Return description containing array shape if exists, else None."""
        for description in (self.description, self.description1):
            if not description:
                return None
            if description[:1] == '{' and '"shape":' in description:
                return description
            if description[:6] == 'shape=':
                return description
        return None

    @property
    def is_mdgel(self):
        """Page contains MDFileTag tag."""
        return 33445 in self.tags  # MDFileTag

    @property
    def is_mediacy(self):
        """Page contains Media Cybernetics Id tag."""
        tag = self.tags.get(50288)  # MC_Id
        return tag is not None and tag.value[:7] == b'MC TIFF'

    @property
    def is_stk(self):
        """Page contains UIC2Tag tag."""
        return 33629 in self.tags

    @property
    def is_lsm(self):
        """Page contains CZ_LSMINFO tag."""
        return 34412 in self.tags

    @property
    def is_fluoview(self):
        """Page contains FluoView MM_STAMP tag."""
        return 34362 in self.tags

    @property
    def is_nih(self):
        """Page contains NIHImageHeader tag."""
        return 43314 in self.tags

    @property
    def is_sgi(self):
        """Page contains SGI ImageDepth and TileDepth tags."""
        return 32998 in self.tags and 32997 in self.tags

    @property
    def is_vista(self):
        """Software tag is 'ISS Vista'."""
        return self.software == 'ISS Vista'

    @property
    def is_metaseries(self):
        """Page contains MDS MetaSeries metadata in ImageDescription tag."""
        if self.index > 1 or self.software != 'MetaSeries':
            return False
        d = self.description
        return d.startswith('<MetaData>') and d.endswith('</MetaData>')

    @property
    def is_ome(self):
        """Page contains OME-XML in ImageDescription tag."""
        if self.index > 1 or not self.description:
            return False
        d = self.description
        return d[:13] == '<?xml version' and d[-4:] == 'OME>'

    @property
    def is_scn(self):
        """Page contains Leica SCN XML in ImageDescription tag."""
        if self.index > 1 or not self.description:
            return False
        d = self.description
        return d[:14] == '<?xml version=' and d[-6:] == '</scn>'

    @property
    def is_micromanager(self):
        """Page contains MicroManagerMetadata tag."""
        return 51123 in self.tags

    @property
    def is_andor(self):
        """Page contains Andor Technology tags 4864-5030."""
        return 4864 in self.tags

    @property
    def is_pilatus(self):
        """Page contains Pilatus tags."""
        return self.software[:8] == 'TVX TIFF' and self.description[:2] == '# '

    @property
    def is_epics(self):
        """Page contains EPICS areaDetector tags."""
        return (
            self.description == 'EPICS areaDetector'
            or self.software == 'EPICS areaDetector'
        )

    @property
    def is_tvips(self):
        """Page contains TVIPS metadata."""
        return 37706 in self.tags

    @property
    def is_fei(self):
        """Page contains FEI_SFEG or FEI_HELIOS tags."""
        return 34680 in self.tags or 34682 in self.tags

    @property
    def is_sem(self):
        """Page contains CZ_SEM tag."""
        return 34118 in self.tags

    @property
    def is_svs(self):
        """Page contains Aperio metadata."""
        return self.description[:20] == 'Aperio Image Library'

    @property
    def is_scanimage(self):
        """Page contains ScanImage metadata."""
        return (
            self.description[:12] == 'state.config'
            or self.software[:22] == 'SI.LINE_FORMAT_VERSION'
            or 'scanimage.SI' in self.description[-256:]
        )

    @property
    def is_qpi(self):
        """Page contains PerkinElmer tissue images metadata."""
        # The ImageDescription tag contains XML with a top-level
        # <PerkinElmer-QPI-ImageDescription> element
        return self.software[:15] == 'PerkinElmer-QPI'

    @property
    def is_geotiff(self):
        """Page contains GeoTIFF metadata."""
        return 34735 in self.tags  # GeoKeyDirectoryTag

    @property
    def is_sis(self):
        """Page contains Olympus SIS metadata."""
        return 33560 in self.tags or 33471 in self.tags

    @lazyattr  # must not be property; tag 65420 is later removed
    def is_ndpi(self):
        """Page contains NDPI metadata."""
        return 65420 in self.tags and 271 in self.tags


class TiffFrame:
    """Lightweight TIFF image file directory (IFD).

    Only a limited number of tag values are read from file, e.g. StripOffsets,
    and StripByteCounts. Other tag values are assumed to be identical with a
    specified TiffPage instance, the keyframe.

    TiffFrame is intended to reduce resource usage and speed up reading image
    data from file, not for introspection of metadata.

    """

    __slots__ = ('index', 'parent', 'offset', '_offsetscounts', '_keyframe')

    is_mdgel = False
    pages = None
    # tags = {}

    def __init__(self, parent, index, offset=None, keyframe=None,
                 offsets=None, bytecounts=None):
        """Initialize TiffFrame from file or values.

        The file handle position must be at the offset to a valid IFD.

        """
        self._keyframe = None
        self.parent = parent
        self.index = index
        self.offset = offset

        if offsets is not None:
            # initialize "virtual frame" from offsets and bytecounts
            self._offsetscounts = offsets, bytecounts
            self._keyframe = keyframe
            return

        if offset is None:
            self.offset = parent.filehandle.tell()
        else:
            parent.filehandle.seek(offset)

        if keyframe is None:
            tags = {273, 279, 324, 325}
        elif keyframe.is_contiguous:
            tags = {256, 273, 324}
        else:
            tags = {256, 273, 279, 324, 325}

        dataoffsets = databytecounts = []

        for code, tag in self._gettags(tags):
            if code == 273 or code == 324:
                dataoffsets = tag.value
            elif code == 279 or code == 325:
                databytecounts = tag.value
            elif code == 256 and keyframe.imagewidth != tag.value:
                raise RuntimeError(
                    f'TiffFrame {self.index} incompatible keyframe')

        if not dataoffsets:
            log_warning(f'TiffFrame {self.index}: missing required tags')

        self._offsetscounts = dataoffsets, databytecounts

        if keyframe is not None:
            self.keyframe = keyframe

    def _gettags(self, codes=None, lock=None):
        """Return list of (code, TiffTag) from file."""
        fh = self.parent.filehandle
        tiff = self.parent.tiff
        unpack = struct.unpack
        lock = NullContext() if lock is None else lock
        tags = []

        with lock:
            fh.seek(self.offset)
            try:
                tagno = unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
                if tagno > 4096:
                    raise TiffFileError(
                        f'TiffFrame {self.index}: suspicious number of tags'
                    )
            except Exception:
                raise TiffFileError(
                    f'TiffFrame {self.index}: '
                    f'corrupted page list at offset {self.offset}'
                )

            tagoffset = self.offset + tiff.tagnosize  # fh.tell()
            tagsize = tiff.tagsize
            tagindex = -tagsize
            codeformat = tiff.tagformat1[:2]
            tagbytes = fh.read(tagsize * tagno)

            for _ in range(tagno):
                tagindex += tagsize
                code = unpack(codeformat, tagbytes[tagindex: tagindex + 2])[0]
                if codes and code not in codes:
                    continue
                try:
                    tag = TiffTag(self.parent,
                                  tagbytes[tagindex: tagindex + tagsize],
                                  tagoffset + tagindex)
                except TiffFileError as exc:
                    log_warning(
                        f'TiffFrame {self.index}: '
                        f'{exc.__class__.__name__}: {exc}'
                    )
                    continue
                tags.append((code, tag))

        return tags

    def aspage(self):
        """Return TiffPage from file."""
        if self.offset is None:
            raise ValueError(
                f'TiffFrame {self.index}: cannot return virtual frame as page'
            )
        self.parent.filehandle.seek(self.offset)
        return TiffPage(self.parent, index=self.index)

    def asarray(self, *args, **kwargs):
        """Read image data from file and return as numpy array."""
        if self._keyframe is None:
            raise RuntimeError(f'TiffFrame {self.index}: keyframe not set')
        return TiffPage.asarray(self, *args, **kwargs)

    def asrgb(self, *args, **kwargs):
        """Read image data from file and return RGB image as numpy array."""
        if self._keyframe is None:
            raise RuntimeError(f'TiffFrame {self.index}: keyframe not set')
        return TiffPage.asrgb(self, *args, **kwargs)

    @property
    def keyframe(self):
        """Return keyframe."""
        return self._keyframe

    @keyframe.setter
    def keyframe(self, keyframe):
        """Set keyframe."""
        if self._keyframe == keyframe:
            return
        if self._keyframe is not None:
            raise RuntimeError(
                f'TiffFrame {self.index}: cannot reset keyframe')
        if len(self._offsetscounts[0]) != len(keyframe.dataoffsets):
            raise RuntimeError(
                f'TiffFrame {self.index}: incompatible keyframe')
        if keyframe.is_tiled:
            pass
        if keyframe.is_contiguous:
            self._offsetscounts = (
                (self._offsetscounts[0][0], ),
                (keyframe.is_contiguous[1], ),
            )
        self._keyframe = keyframe

    @property
    def is_contiguous(self):
        """Return offset and size of contiguous data, else None."""
        if self._keyframe is None:
            raise RuntimeError(f'TiffFrame {self.index}: keyframe not set')
        if self._keyframe.is_contiguous:
            return self._offsetscounts[0][0], self._keyframe.is_contiguous[1]
        return None

    @property
    def is_memmappable(self):
        """Return if page's image data in file can be memory-mapped."""
        if self._keyframe is None:
            raise RuntimeError(f'TiffFrame {self.index}: keyframe not set')
        return self._keyframe.is_memmappable

    @property
    def hash(self):
        """Return checksum to identify pages in same series."""
        if self._keyframe is None:
            raise RuntimeError(f'TiffFrame {self.index}: keyframe not set')
        return self._keyframe.hash

    def __getattr__(self, name):
        """Return attribute from keyframe."""
        if name in TIFF.FRAME_ATTRS:
            return getattr(self._keyframe, name)
        # this error could be raised because an AttributeError was
        # raised inside a @property function
        raise AttributeError(
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
        )

    def __str__(self, detail=0, width=79):
        """Return string containing information about TiffFrame."""
        if self._keyframe is None:
            info = ''
            kf = None
        else:
            info = '  '.join(s for s in ('x'.join(str(i) for i in self.shape),
                                         str(self.dtype)))
            kf = TiffPage.__str__(self._keyframe, width=width - 11)
        if detail > 3:
            of, bc = self._offsetscounts
            of = pformat(of, width=width - 9, height=detail - 3)
            bc = pformat(bc, width=width - 13, height=detail - 3)
            info = f'\n Keyframe {kf}\n Offsets {of}\n Bytecounts {bc}'
        return f'TiffFrame {self.index} @{self.offset}  {info}'


class TiffTag:
    """TIFF tag structure.

    Attributes
    ----------
    name : string
        Name of tag, TIFF.TAGS[code].
    code : int
        Decimal code of tag.
    dtype : str
        Datatype of tag data. One of TIFF DATA_FORMATS.
    count : int
        Number of values.
    value : various types
        Tag data as Python object.
    offset : int
        Location of tag structure in file.
    valueoffset : int
        Location of value in file.

    All attributes are read-only.

    """

    __slots__ = ('code', 'count', 'dtype', 'value', 'offset', 'valueoffset')

    def __init__(self, parent, tagheader, tagoffset):
        """Initialize instance from tag header."""
        fh = parent.filehandle
        tiff = parent.tiff
        byteorder = tiff.byteorder
        offsetsize = tiff.offsetsize
        unpack = struct.unpack

        self.offset = tagoffset
        self.valueoffset = tagoffset + offsetsize + 4
        code, type_ = unpack(tiff.tagformat1, tagheader[:4])
        count, value = unpack(tiff.tagformat2, tagheader[4:])

        try:
            dtype = TIFF.DATA_FORMATS[type_]
        except KeyError:
            raise TiffFileError(f'unknown tag data type {type_!r}')

        fmt = '{}{}{}'.format(byteorder, count * int(dtype[0]), dtype[1])
        size = struct.calcsize(fmt)
        if size > offsetsize or code in TIFF.TAG_READERS:
            self.valueoffset = offset = unpack(tiff.offsetformat, value)[0]
            if offset < 8 or offset > fh.size - size:
                raise TiffFileError('invalid tag value offset')
            # if offset % 2:
            #     log_warning('TiffTag: value does not begin on word boundary')
            fh.seek(offset)
            if code in TIFF.TAG_READERS:
                readfunc = TIFF.TAG_READERS[code]
                value = readfunc(fh, byteorder, dtype, count, offsetsize)
            elif type_ == 7 or (count > 1 and dtype[-1] == 'B'):
                value = read_bytes(fh, byteorder, dtype, count, offsetsize)
            # elif code in TIFF.TAGS or dtype[-1] == 's':
            else:
                value = unpack(fmt, fh.read(size))
            # else:
            #     value = read_numpy(fh, byteorder, dtype, count, offsetsize)
        elif dtype[-1] == 'B' or type_ == 7:
            value = value[:size]
        else:
            value = unpack(fmt, value[:size])

        process = (
            code not in TIFF.TAG_READERS
            and code not in TIFF.TAG_TUPLE
            and type_ != 7
        )
        if process and dtype[-1] == 's' and isinstance(value[0], bytes):
            # TIFF ASCII fields can contain multiple strings,
            #   each terminated with a NUL
            value = value[0]
            try:
                value = bytes2str(stripascii(value).strip())
            except UnicodeDecodeError:
                log_warning(
                    f'TiffTag {code}: coercing invalid ASCII to bytes'
                )
                dtype = '1B'
        else:
            if code in TIFF.TAG_ENUM:
                t = TIFF.TAG_ENUM[code]
                try:
                    value = tuple(t(v) for v in value)
                except ValueError as exc:
                    log_warning(f'TiffTag {code}: {exc}')
            if process:
                if len(value) == 1:
                    value = value[0]

        self.code = code
        self.dtype = dtype
        self.count = count
        self.value = value

    @property
    def name(self):
        """Return name of tag from TIFF.TAGS registry."""
        return TIFF.TAGS.get(self.code, str(self.code))

    def _fix_lsm_bitspersample(self, parent):
        """Correct LSM bitspersample tag.

        Old LSM writers may use a separate region for two 16-bit values,
        although they fit into the tag value element of the tag.

        """
        if self.code != 258 or self.count != 2:
            return
        # TODO: test this case; need example file
        log_warning(f'TiffTag {self.code}: correcting LSM bitspersample tag')
        value = struct.pack('<HH', *self.value)
        self.valueoffset = struct.unpack('<I', value)[0]
        parent.filehandle.seek(self.valueoffset)
        self.value = struct.unpack('<HH', parent.filehandle.read(4))

    def __str__(self, detail=0, width=79):
        """Return string containing information about TiffTag."""
        height = 1 if detail <= 0 else 8 * detail
        tcode = '{}{}'.format(self.count * int(self.dtype[0]), self.dtype[1])
        name = '|'.join(TIFF.TAGS.getall(self.code, ()))
        if name:
            name = f'{self.code} {name} @{self.offset}'
        else:
            name = f'{self.code} @{self.offset}'
        line = f'TiffTag {name} {tcode} @{self.valueoffset}  '
        line = line[:width]
        try:
            if self.count == 1:
                value = enumstr(self.value)
            else:
                value = pformat(tuple(enumstr(v) for v in self.value))
        except Exception:
            value = pformat(self.value, width=width, height=height)
        if detail <= 0:
            line += value
            line = line[:width]
        else:
            line += '\n' + value
        return line


class TiffTags:
    """Multidict like interface to TiffTag instances in TiffPage.

    Differences to a regular dict:

    * values are instances of TiffTag.
    * keys are TiffTag.code (int).
    * multiple values can be stored per key.
    * can be indexed with TiffTag.name (str), although slower than by key.
    * iter() returns values instead of keys.
    * values() and items() contain all values sorted by offset stored in file.
    * len() returns the number of all values.
    * get() takes an optional index argument.
    * some functions are not implemented, e.g. update, setdefault, pop.

    """

    __slots__ = ('_dict', '_list')

    def __init__(self):
        """Initialize empty instance."""
        self._dict = {}
        self._list = [self._dict]

    def add(self, tag):
        """Add a tag."""
        code = tag.code
        for d in self._list:
            if code not in d:
                d[code] = tag
                break
        else:
            self._list.append({code: tag})

    def keys(self):
        """Return new view of all codes."""
        return self._dict.keys()

    def values(self):
        """Return all tags in order they are stored in file."""
        tags = (t for d in self._list for t in d.values())
        return sorted(tags, key=lambda t: t.offset)

    def items(self):
        """Return all (code, tag) pairs in order tags are stored in file."""
        items = (i for d in self._list for i in d.items())
        return sorted(items, key=lambda i: i[1].offset)

    def get(self, key, default=None, index=None):
        """Return tag of code or name if exists, else default."""
        if index is None:
            if key in self._dict:
                return self._dict[key]
            if not isinstance(key, str):
                return default
            index = 0
        try:
            tags = self._list[index]
        except IndexError:
            return default
        if key in tags:
            return tags[key]
        if not isinstance(key, str):
            return default
        for tag in tags.values():
            if tag.name == key:
                return tag
        return default

    def getall(self, key, default=None):
        """Return list of all tags of code or name if exists, else default."""
        result = []
        for tags in self._list:
            if key in tags:
                result.append(tags[key])
            else:
                break
        if result:
            return result
        if not isinstance(key, str):
            return default
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    result.append(tag)
                    break
            if not result:
                break
        return result if result else default

    def __getitem__(self, key):
        """Return first tag of code or name. Raise KeyError if not found."""
        if key in self._dict:
            return self._dict[key]
        if not isinstance(key, str):
            raise KeyError(key)
        for tag in self._dict.values():
            if tag.name == key:
                return tag
        raise KeyError(key)

    def __setitem__(self, code, tag):
        """Add a tag."""
        self.add(tag)

    def __delitem__(self, key):
        """Delete all tags of code or name."""
        found = False
        for tags in self._list:
            if key in tags:
                found = True
                del tags[key]
            else:
                break
        if found:
            return None
        if not isinstance(key, str):
            raise KeyError(key)
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    del tags[tag.code]
                    found = True
                    break
            else:
                break
        if not found:
            raise KeyError(key)
        return None

    def __contains__(self, item):
        """Return if tag is in map."""
        if item in self._dict:
            return True
        if not isinstance(item, str):
            return False
        for tag in self._dict.values():
            if tag.name == item:
                return True
        return False

    def __iter__(self):
        """Return iterator over all tags."""
        return iter(self.values())

    def __len__(self):
        """Return number of tags."""
        size = 0
        for d in self._list:
            size += len(d)
        return size

    def __str__(self, detail=0, width=79):
        """Return string with information about TiffTags."""
        info = []
        tlines = []
        vlines = []
        for tag in self:
            value = tag.__str__(width=width+1)
            tlines.append(value[:width].strip())
            if detail > 0 and len(value) > width:
                if detail < 2 and tag.code in (273, 279, 324, 325):
                    value = pformat(tag.value, width=width, height=detail * 4)
                else:
                    value = pformat(tag.value, width=width, height=detail * 12)
                vlines.append(f'{tag.name}\n{value}')
        info.append('\n'.join(tlines))
        if detail > 0 and vlines:
            info.append('\n')
            info.append('\n\n'.join(vlines))
        return '\n'.join(info)


class TiffPageSeries:
    """Series of TIFF pages with compatible shape and data type.

    Attributes
    ----------
    pages : list of TiffPage
        Sequence of TiffPages in series.
    dtype : numpy.dtype
        Data type (native byte order) of the image array in series.
    shape : tuple
        Dimensions of the image array in series.
    axes : str
        Labels of axes in shape. See TiffPage.axes.
    offset : int or None
        Position of image data in file if memory-mappable, else None.

    """

    def __init__(self, pages, shape, dtype, axes, parent=None, name=None,
                 transform=None, kind=None, truncated=False):
        """Initialize instance."""
        self.index = 0
        self._pages = pages  # might contain only first of contiguous pages
        self.shape = tuple(shape)
        self.axes = ''.join(axes)
        self.dtype = numpy.dtype(dtype)
        self.kind = kind if kind else ''
        self.name = name if name else ''
        self.transform = transform
        if parent:
            self.parent = parent
        elif pages:
            self.parent = pages[0].parent
        else:
            self.parent = None
        if not truncated and len(pages) == 1:
            s = product(pages[0].shape)
            if s > 0:
                self._len = int(product(self.shape) // s)
            else:
                self._len = len(pages)
        else:
            self._len = len(pages)

    def asarray(self, **kwargs):
        """Return image data from series of TIFF pages as numpy array."""
        if self.parent:
            result = self.parent.asarray(series=self, **kwargs)
            if self.transform is not None:
                result = self.transform(result)
            return result
        return None

    @lazyattr
    def offset(self):
        """Return offset to series data in file, if any."""
        if not self._pages:
            return None

        pos = 0
        for page in self._pages:
            if page is None:
                return None
            if not page.is_final:
                return None
            if not pos:
                pos = page.is_contiguous[0] + page.is_contiguous[1]
                continue
            if pos != page.is_contiguous[0]:
                return None
            pos += page.is_contiguous[1]

        page = self._pages[0]
        offset = page.is_contiguous[0]
        if (page.is_imagej or page.is_shaped) and len(self._pages) == 1:
            # truncated files
            return offset
        if pos == offset + product(self.shape) * self.dtype.itemsize:
            return offset
        return None

    @property
    def ndim(self):
        """Return number of array dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Return number of elements in array."""
        return int(product(self.shape))

    @property
    def pages(self):
        """Return sequence of all pages in series."""
        # a workaround to keep the old interface working
        return self

    def _getitem(self, key):
        """Return specified page of series from cache or file."""
        key = int(key)
        if key < 0:
            key %= self._len
        if len(self._pages) == 1 and 0 < key < self._len:
            index = self._pages[0].index
            return self.parent.pages._getitem(index + key)
        return self._pages[key]

    def __getitem__(self, key):
        """Return specified page(s)."""
        getitem = self._getitem
        if isinstance(key, (int, numpy.integer)):
            return getitem(key)
        if isinstance(key, slice):
            return [getitem(i) for i in range(*key.indices(self._len))]
        if isinstance(key, Iterable):
            return [getitem(k) for k in key]
        raise TypeError('key must be an integer, slice, or iterable')

    def __iter__(self):
        """Return iterator over pages in series."""
        if len(self._pages) == self._len:
            yield from self._pages
        else:
            pages = self.parent.pages
            index = self._pages[0].index
            for i in range(self._len):
                yield pages[index + i]

    def __len__(self):
        """Return number of pages in series."""
        return self._len

    def __str__(self):
        """Return string with information about TiffPageSeries."""
        s = '  '.join(
            s
            for s in (
                snipstr(f'{self.name!r}', 20) if self.name else '',
                'x'.join(str(i) for i in self.shape),
                str(self.dtype),
                self.axes,
                self.kind,
                f'{len(self.pages)} Pages',
                (f'Offset={self.offset}') if self.offset else '')
            if s
        )
        return f'TiffPageSeries {self.index}  {s}'


class FileSequence:
    """Series of files containing array data of compatible shape and data type.

    Attributes
    ----------
    files : list
        List of file names.
    shape : tuple
        Shape of file series. Excludes shape of individual arrays.
    axes : str
        Labels of axes in shape.

    """

    _patterns = {
        'axes': r"""
            # matches Olympus OIF and Leica TIFF series
            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))
            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
            _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
            """
    }

    def __init__(self, fromfile, files, container=None, sort=None,
                 pattern=None, axesorder=None):
        """Initialize instance from multiple files.

        Parameters
        ----------
        fromfile : function or class
            Array read function or class with asarray function returning numpy
            array from single file.
        files : str, pathlib.Path, or sequence thereof
            Glob filename pattern or sequence of file names. Default: \\*.
            Binary streams are not supported.
        container : str or container instance
            Name or open instance of ZIP file in which files are stored.
        sort : function
            Sort function used to sort file names when 'files' is a pattern.
            The default (None) is natural_sorted. Use sort=False to disable
            sorting.
        pattern : str
            Regular expression pattern that matches axes and sequence indices
            in file names. By default (None), no pattern matching is performed.
            Axes can be specified by matching groups preceding the index groups
            in the file name, be provided as group names for the index groups,
            or be omitted. The predefined 'axes' pattern matches Olympus OIF
            and Leica TIFF series.
        axesorder : sequence of int
            Indices of axes in pattern.

        """
        if files is None:
            files = '*'
        if sort is None:
            sort = natural_sorted
        self._container = container
        if container:
            import fnmatch
            if isinstance(container, str):
                import zipfile
                self._container = zipfile.ZipFile(container)
            elif not hasattr(self._container, 'open'):
                raise ValueError('invalid container')
            if isinstance(files, str):
                files = fnmatch.filter(self._container.namelist(), files)
                if sort:
                    files = sort(files)
        else:
            if isinstance(files, pathlib.Path):
                files = str(files)
            if isinstance(files, str):
                files = glob.glob(files)
                if sort:
                    files = sort(files)
            if not files:
                raise ValueError('no files found')

        files = list(files)
        if not files:
            raise ValueError('no files found')
        if isinstance(files[0], pathlib.Path):
            files = [str(pathlib.Path(f)) for f in files]
        elif not isinstance(files[0], str):
            raise ValueError('not a file name')

        if hasattr(fromfile, 'asarray'):
            # redefine fromfile to use asarray from fromfile class
            if not callable(fromfile.asarray):
                raise ValueError('invalid fromfile function')
            _fromfile0 = fromfile

            def fromfile(fname, **kwargs):
                with _fromfile0(fname) as handle:
                    return handle.asarray(**kwargs)

        elif not callable(fromfile):
            raise ValueError('invalid fromfile function')

        if container:
            # redefine fromfile to read from container
            _fromfile1 = fromfile

            def fromfile(fname, **kwargs):
                with self._container.open(fname) as handle1:
                    with io.BytesIO(handle1.read()) as handle2:
                        return _fromfile1(handle2, **kwargs)

        axes = 'I'
        shape = (len(files),)
        indices = tuple((i,) for i in range(len(files)))
        startindex = (0,)

        pattern = self._patterns.get(pattern, pattern)
        if pattern:
            try:
                axes, shape, indices, startindex = parse_filenames(
                    files, pattern, axesorder)
            except ValueError as exc:
                log_warning(
                    f'FileSequence: failed to parse file names ({exc})')

        if product(shape) != len(files):
            log_warning(
                'FileSequence: files are missing. Missing data are zeroed')

        self.fromfile = fromfile
        self.files = files
        self.pattern = pattern
        self.axes = axes.upper()
        self.shape = shape
        self._indices = indices
        self._startindex = startindex

    def __str__(self):
        """Return string with information about file FileSequence."""
        file = str(self._container) if self._container else self.files[0]
        file = os.path.split(file)[-1]
        return '\n '.join((
            self.__class__.__name__,
            file,
            f'files: {len(self.files)}',
            'shape: {}'.format(', '.join(str(i) for i in self.shape)),
            f'axes: {self.axes}',
        ))

    def __len__(self):
        return len(self.files)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._container:
            self._container.close()
        self._container = None

    def asarray(self, file=None, ioworkers=1, out=None, **kwargs):
        """Read image data from files and return as numpy array.

        Raise IndexError or ValueError if array shapes do not match.

        Parameters
        ----------
        file : int or None
            Index or name of single file to read.
        ioworkers : int or None
            Maximum number of threads to execute the array read function
            asynchronously. Default: 1.
            If None, default to the number of processors multiplied by 5.
            Using threads can significantly improve runtime when
            reading many small files from a network share.
        out : numpy.ndarray, str, or file-like object
            Buffer where image data will be saved.
            If None (default), a new array will be created.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If 'memmap', create a memory-mapped array in a temporary file.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        kwargs : dict
            Additional parameters passed to the array read function.

        """
        if file is not None:
            if isinstance(file, int):
                return self.fromfile(self.files[file], **kwargs)
            return self.fromfile(file, **kwargs)

        im = self.fromfile(self.files[0], **kwargs)
        shape = self.shape + im.shape
        result = create_output(out, shape, dtype=im.dtype)
        result = result.reshape(-1, *im.shape)

        def func(index, fname):
            """Read single image from file into result."""
            index = [i - j for i, j in zip(index, self._startindex)]
            index = numpy.ravel_multi_index(index, self.shape)
            im = self.fromfile(fname, **kwargs)
            result[index] = im

        if len(self.files) < 3:
            ioworkers = 1
        elif ioworkers is None or ioworkers < 1:
            import multiprocessing
            ioworkers = max(multiprocessing.cpu_count() * 5, 1)

        if ioworkers < 2:
            for index, fname in zip(self._indices, self.files):
                func(index, fname)
        else:
            func(self._indices[0], self.files[0])
            with ThreadPoolExecutor(ioworkers) as executor:
                executor.map(func, self._indices[1:], self.files[1:])

        result.shape = shape
        return result


# class TiffSequence(FileSequence):
#     """Series of TIFF files."""

#     def __init__(self, files=None, container=None, sort=None, pattern=None,
#                  imread=imread):
#         """Initialize instance from multiple TIFF files."""
#         super().__init__(
#             imread, '*.tif' if files is None else files,
#             container=container, sort=sort, pattern=pattern)


class FileHandle:
    """Binary file handle.

    A limited, special purpose file handle that can:

    * handle embedded files (for CZI within CZI files)
    * re-open closed files (for multi-file formats, such as OME-TIFF)
    * read and write numpy arrays and records from file like objects

    Only 'rb' and 'wb' modes are supported. Concurrently reading and writing
    of the same stream is untested.

    When initialized from another file handle, do not use it unless this
    FileHandle is closed.

    Attributes
    ----------
    name : str
        Name of the file.
    path : str
        Absolute path to file.
    size : int
        Size of file in bytes.
    is_file : bool
        If True, file has a filno and can be memory-mapped.

    All attributes are read-only.

    """

    __slots__ = (
        '_fh', '_file', '_mode', '_name', '_dir', '_lock', '_offset',
        '_size', '_close', 'is_file'
    )

    def __init__(self, file, mode='rb', name=None, offset=None, size=None):
        """Initialize file handle from file name or another file handle.

        Parameters
        ----------
        file : str, pathlib.Path, binary stream, or FileHandle
            File name or seekable binary stream, such as an open file
            or BytesIO.
        mode : str
            File open mode in case 'file' is a file name. Must be 'rb' or 'wb'.
        name : str
            Optional name of file in case 'file' is a binary stream.
        offset : int
            Optional start position of embedded file. By default, this is
            the current file position.
        size : int
            Optional size of embedded file. By default, this is the number
            of bytes from the 'offset' to the end of the file.

        """
        self._file = file
        self._fh = None
        self._mode = mode
        self._name = name
        self._dir = ''
        self._offset = offset
        self._size = size
        self._close = True
        self.is_file = False
        self._lock = NullContext()
        self.open()

    def open(self):
        """Open or re-open file."""
        if self._fh:
            return  # file is open

        if isinstance(self._file, pathlib.Path):
            self._file = str(self._file)
        if isinstance(self._file, str):
            # file name
            self._file = os.path.realpath(self._file)
            self._dir, self._name = os.path.split(self._file)
            self._fh = open(self._file, self._mode)
            self._close = True
            if self._offset is None:
                self._offset = 0
        elif isinstance(self._file, FileHandle):
            # FileHandle
            self._fh = self._file._fh
            if self._offset is None:
                self._offset = 0
            self._offset += self._file._offset
            self._close = False
            if not self._name:
                if self._offset:
                    name, ext = os.path.splitext(self._file._name)
                    self._name = f'{name}@{self._offset}{ext}'
                else:
                    self._name = self._file._name
            if self._mode and self._mode != self._file._mode:
                raise ValueError('FileHandle has wrong mode')
            self._mode = self._file._mode
            self._dir = self._file._dir
        elif hasattr(self._file, 'seek'):
            # binary stream: open file, BytesIO
            try:
                self._file.tell()
            except Exception:
                raise ValueError('binary stream is not seekable')
            self._fh = self._file
            if self._offset is None:
                self._offset = self._file.tell()
            self._close = False
            if not self._name:
                try:
                    self._dir, self._name = os.path.split(self._fh.name)
                except AttributeError:
                    self._name = 'Unnamed binary stream'
            try:
                self._mode = self._fh.mode
            except AttributeError:
                pass
        else:
            raise ValueError('the first parameter must be a file name, '
                             'seekable binary stream, or FileHandle')

        if self._offset:
            self._fh.seek(self._offset)

        if self._size is None:
            pos = self._fh.tell()
            self._fh.seek(self._offset, 2)
            self._size = self._fh.tell()
            self._fh.seek(pos)

        try:
            self._fh.fileno()
            self.is_file = True
        except Exception:
            self.is_file = False

    def read(self, size=-1):
        """Read 'size' bytes from file, or until EOF is reached."""
        if size < 0 and self._offset:
            size = self._size
        return self._fh.read(size)

    def readinto(self, b):
        """Read up to len(b) bytes into b, and return number of bytes read."""
        return self._fh.readinto(b)

    def write(self, bytestring):
        """Write bytes to file."""
        return self._fh.write(bytestring)

    def flush(self):
        """Flush write buffers if applicable."""
        return self._fh.flush()

    def memmap_array(self, dtype, shape, offset=0, mode='r', order='C'):
      raise TiffParserError('memmap_array is not supported by TiffParser')
        

    def read_array(self, dtype, count=-1, out=None):
      raise TiffParserError('read_array is not supported by TiffParser')
      
    def read_segments(self, offsets, bytecounts, indices=None, sort=True,
                      lock=None, buffersize=None):
      raise TiffParserError('read_segments is not supported by TiffParser')
      

    def read_record(self, dtype, shape=1, byteorder=None):
        """Return numpy record from file."""
        raise TiffParserError('read_record is not supported by TiffParser')
        # rec = numpy.rec
        # try:
        #     record = rec.fromfile(self._fh, dtype, shape, byteorder=byteorder)
        # except Exception:
        #     dtype = numpy.dtype(dtype)
        #     if shape is None:
        #         shape = self._size // dtype.itemsize
        #     size = product(sequence(shape)) * dtype.itemsize
        #     data = self._fh.read(size)
        #     record = rec.fromstring(data, dtype, shape, byteorder=byteorder)
        # return record[0] if shape == 1 else record

    def write_empty(self, size):
        """Append size bytes to file. Position must be at end of file."""
        if size < 1:
            return
        self._fh.seek(size - 1, 1)
        self._fh.write(b'\x00')

    def write_array(self, data):
        """Write numpy array to binary file."""
        try:
            data.tofile(self._fh)
        except Exception:
            # BytesIO
            self._fh.write(data.tostring())

    def tell(self):
        """Return file's current position."""
        return self._fh.tell() - self._offset

    def seek(self, offset, whence=0):
        """Set file's current position."""
        if self._offset:
            if whence == 0:
                self._fh.seek(self._offset + offset, whence)
                return
            if whence == 2 and self._size > 0:
                self._fh.seek(self._offset + self._size + offset, 0)
                return
        self._fh.seek(offset, whence)

    def close(self):
        """Close file."""
        if self._close and self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getattr__(self, name):
        """Return attribute from underlying file object."""
        if self._offset:
            warnings.warn(
                f'FileHandle: {name!r} not implemented for embedded files',
                UserWarning
            )
        return getattr(self._fh, name)

    @property
    def name(self):
        return self._name

    @property
    def dirname(self):
        return self._dir

    @property
    def path(self):
        return os.path.join(self._dir, self._name)

    @property
    def size(self):
        return self._size

    @property
    def closed(self):
        return self._fh is None

    @property
    def lock(self):
        return self._lock

    @lock.setter
    def lock(self, value):
        self._lock = threading.RLock() if value else NullContext()


class NullContext:
    """Null context manager.

    >>> with NullContext():
    ...     pass

    """

    __slots = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class OpenFileCache:
    """Keep files open."""

    __slots__ = ('files', 'past', 'lock', 'size')

    def __init__(self, size, lock=None):
        """Initialize open file cache."""
        self.past = []  # FIFO of opened files
        self.files = {}  # refcounts of opened files
        self.lock = NullContext() if lock is None else lock
        self.size = int(size)

    def open(self, filehandle):
        """Re-open file if necessary."""
        with self.lock:
            if filehandle in self.files:
                self.files[filehandle] += 1
            elif filehandle.closed:
                filehandle.open()
                self.files[filehandle] = 1
                self.past.append(filehandle)

    def close(self, filehandle):
        """Close openend file if no longer used."""
        with self.lock:
            if filehandle in self.files:
                self.files[filehandle] -= 1
                # trim the file cache
                index = 0
                size = len(self.past)
                while size > self.size and index < size:
                    filehandle = self.past[index]
                    if self.files[filehandle] == 0:
                        filehandle.close()
                        del self.files[filehandle]
                        del self.past[index]
                        size -= 1
                    else:
                        index += 1

    def clear(self):
        """Close all opened files if not in use."""
        with self.lock:
            for filehandle, refcount in list(self.files.items()):
                if refcount == 0:
                    filehandle.close()
                    del self.files[filehandle]
                    del self.past[self.past.index(filehandle)]




class LazyConst:
    """Class whose attributes are computed on first access from its methods."""

    def __init__(self, cls):
        self._cls = cls
        self.__doc__ = getattr(cls, '__doc__')

    def __getattr__(self, name):
        func = getattr(self._cls, name)
        if not callable(func):
            return func
        value = func()
        setattr(self, name, value)
        return value


@LazyConst
class TIFF:
    """Namespace for module constants."""

    def CLASSIC_LE():
        class ClassicTiffLe:
            __slots__ = ()
            version = 42
            byteorder = '<'
            offsetsize = 4
            offsetformat = '<I'
            ifdoffsetsize = 4
            ifdoffsetformat = '<I'
            tagnosize = 2
            tagnoformat = '<H'
            tagsize = 12
            tagformat1 = '<HH'
            tagformat2 = '<I4s'

        return ClassicTiffLe

    def CLASSIC_BE():
        class ClassicTiffBe:
            __slots__ = ()
            version = 42
            byteorder = '>'
            offsetsize = 4
            offsetformat = '>I'
            ifdoffsetsize = 4
            ifdoffsetformat = '>I'
            tagnosize = 2
            tagnoformat = '>H'
            tagsize = 12
            tagformat1 = '>HH'
            tagformat2 = '>I4s'

        return ClassicTiffBe

    def BIG_LE():
        class BigTiffLe:
            __slots__ = ()
            version = 43
            byteorder = '<'
            offsetsize = 8
            offsetformat = '<Q'
            ifdoffsetsize = 8
            ifdoffsetformat = '<Q'
            tagnosize = 8
            tagnoformat = '<Q'
            tagsize = 20
            tagformat1 = '<HH'
            tagformat2 = '<Q8s'

        return BigTiffLe

    def BIG_BE():
        class BigTiffBe:
            __slots__ = ()
            version = 43
            byteorder = '>'
            offsetsize = 8
            offsetformat = '>Q'
            ifdoffsetsize = 8
            ifdoffsetformat = '>Q'
            tagnosize = 8
            tagnoformat = '>Q'
            tagsize = 20
            tagformat1 = '>HH'
            tagformat2 = '>Q8s'

        return BigTiffBe

    def NDPI_LE():
        class NdpiTiffLe:
            __slots__ = ()
            version = 42
            byteorder = '<'
            offsetsize = 4
            offsetformat = '<I'
            ifdoffsetsize = 8  # NDPI uses 8 bytes IFD offsets
            ifdoffsetformat = '<Q'
            tagnosize = 2
            tagnoformat = '<H'
            tagsize = 12
            tagformat1 = '<HH'
            tagformat2 = '<I4s'

        return NdpiTiffLe

    def TAGS():
        # TIFF tag codes and names from TIFF6, TIFF/EP, EXIF, and other specs
        class TiffTagRegistry:
            """Registry of TIFF tag codes and names.

            The registry allows to look up tag codes and names by indexing
            with names and codes respectively.
            One tag code may be registered with several names,
            e.g. 34853 is used for GPSTag or OlympusSIS2.
            Different tag codes may be registered with the same name,
            e.g. 37387 and 41483 are both named FlashEnergy.

            """

            def __init__(self, arg):
                self._dict = {}
                self._list = [self._dict]
                self.update(arg)

            def update(self, arg):
                """Add codes and names from sequence or dict to registry."""
                if isinstance(arg, dict):
                    arg = arg.items()
                for code, name in arg:
                    self.add(code, name)

            def add(self, code, name):
                """Add code and name to registry."""
                for d in self._list:
                    if code in d and d[code] == name:
                        break
                    if code not in d and name not in d:
                        d[code] = name
                        d[name] = code
                        break
                else:
                    self._list.append({code: name, name: code})

            def items(self):
                """Return all registry items as (code, name)."""
                items = (i for d in self._list for i in d.items()
                         if isinstance(i[0], int))
                return sorted(items, key=lambda i: i[0])

            def get(self, key, default=None):
                """Return first code/name if exists, else default."""
                for d in self._list:
                    if key in d:
                        return d[key]
                return default

            def getall(self, key, default=None):
                """Return list of all codes/names if exists, else default."""
                result = [d[key] for d in self._list if key in d]
                return result if result else default

            def __getitem__(self, key):
                """Return first code/name. Raise KeyError if not found."""
                for d in self._list:
                    if key in d:
                        return d[key]
                raise KeyError(key)

            def __delitem__(self, key):
                """Delete all tags of code or name."""
                found = False
                for d in self._list:
                    if key in d:
                        found = True
                        value = d[key]
                        del d[key]
                        del d[value]
                if not found:
                    raise KeyError(key)

            def __contains__(self, item):
                """Return if code or name is in registry."""
                for d in self._list:
                    if item in d:
                        return True
                return False

            def __iter__(self):
                """Return iterator over all items in registry."""
                return iter(self.items())

            def __len__(self):
                """Return number of registered tags."""
                size = 0
                for d in self._list:
                    size += len(d)
                return size // 2

            def __str__(self):
                """Return string with information about TiffTags."""
                return 'TiffTagRegistry(((\n  {}\n))'.format(
                    ',\n  '.join(f'({code}, {name!r})'
                                 for code, name in self.items()))

        return TiffTagRegistry((
            (11, 'ProcessingSoftware'),
            (254, 'NewSubfileType'),
            (255, 'SubfileType'),
            (256, 'ImageWidth'),
            (257, 'ImageLength'),
            (258, 'BitsPerSample'),
            (259, 'Compression'),
            (262, 'PhotometricInterpretation'),
            (263, 'Thresholding'),
            (264, 'CellWidth'),
            (265, 'CellLength'),
            (266, 'FillOrder'),
            (269, 'DocumentName'),
            (270, 'ImageDescription'),
            (271, 'Make'),
            (272, 'Model'),
            (273, 'StripOffsets'),
            (274, 'Orientation'),
            (277, 'SamplesPerPixel'),
            (278, 'RowsPerStrip'),
            (279, 'StripByteCounts'),
            (280, 'MinSampleValue'),
            (281, 'MaxSampleValue'),
            (282, 'XResolution'),
            (283, 'YResolution'),
            (284, 'PlanarConfiguration'),
            (285, 'PageName'),
            (286, 'XPosition'),
            (287, 'YPosition'),
            (288, 'FreeOffsets'),
            (289, 'FreeByteCounts'),
            (290, 'GrayResponseUnit'),
            (291, 'GrayResponseCurve'),
            (292, 'T4Options'),
            (293, 'T6Options'),
            (296, 'ResolutionUnit'),
            (297, 'PageNumber'),
            (300, 'ColorResponseUnit'),
            (301, 'TransferFunction'),
            (305, 'Software'),
            (306, 'DateTime'),
            (315, 'Artist'),
            (316, 'HostComputer'),
            (317, 'Predictor'),
            (318, 'WhitePoint'),
            (319, 'PrimaryChromaticities'),
            (320, 'ColorMap'),
            (321, 'HalftoneHints'),
            (322, 'TileWidth'),
            (323, 'TileLength'),
            (324, 'TileOffsets'),
            (325, 'TileByteCounts'),
            (326, 'BadFaxLines'),
            (327, 'CleanFaxData'),
            (328, 'ConsecutiveBadFaxLines'),
            (330, 'SubIFDs'),
            (332, 'InkSet'),
            (333, 'InkNames'),
            (334, 'NumberOfInks'),
            (336, 'DotRange'),
            (337, 'TargetPrinter'),
            (338, 'ExtraSamples'),
            (339, 'SampleFormat'),
            (340, 'SMinSampleValue'),
            (341, 'SMaxSampleValue'),
            (342, 'TransferRange'),
            (343, 'ClipPath'),
            (344, 'XClipPathUnits'),
            (345, 'YClipPathUnits'),
            (346, 'Indexed'),
            (347, 'JPEGTables'),
            (351, 'OPIProxy'),
            (400, 'GlobalParametersIFD'),
            (401, 'ProfileType'),
            (402, 'FaxProfile'),
            (403, 'CodingMethods'),
            (404, 'VersionYear'),
            (405, 'ModeNumber'),
            (433, 'Decode'),
            (434, 'DefaultImageColor'),
            (435, 'T82Options'),
            (437, 'JPEGTables'),  # 347
            (512, 'JPEGProc'),
            (513, 'JPEGInterchangeFormat'),
            (514, 'JPEGInterchangeFormatLength'),
            (515, 'JPEGRestartInterval'),
            (517, 'JPEGLosslessPredictors'),
            (518, 'JPEGPointTransforms'),
            (519, 'JPEGQTables'),
            (520, 'JPEGDCTables'),
            (521, 'JPEGACTables'),
            (529, 'YCbCrCoefficients'),
            (530, 'YCbCrSubSampling'),
            (531, 'YCbCrPositioning'),
            (532, 'ReferenceBlackWhite'),
            (559, 'StripRowCounts'),
            (700, 'XMP'),  # XMLPacket
            (769, 'GDIGamma'),  # GDI+
            (770, 'ICCProfileDescriptor'),  # GDI+
            (771, 'SRGBRenderingIntent'),  # GDI+
            (800, 'ImageTitle'),  # GDI+
            (999, 'USPTO_Miscellaneous'),
            (4864, 'AndorId'),  # TODO, Andor Technology 4864 - 5030
            (4869, 'AndorTemperature'),
            (4876, 'AndorExposureTime'),
            (4878, 'AndorKineticCycleTime'),
            (4879, 'AndorAccumulations'),
            (4881, 'AndorAcquisitionCycleTime'),
            (4882, 'AndorReadoutTime'),
            (4884, 'AndorPhotonCounting'),
            (4885, 'AndorEmDacLevel'),
            (4890, 'AndorFrames'),
            (4896, 'AndorHorizontalFlip'),
            (4897, 'AndorVerticalFlip'),
            (4898, 'AndorClockwise'),
            (4899, 'AndorCounterClockwise'),
            (4904, 'AndorVerticalClockVoltage'),
            (4905, 'AndorVerticalShiftSpeed'),
            (4907, 'AndorPreAmpSetting'),
            (4908, 'AndorCameraSerial'),
            (4911, 'AndorActualTemperature'),
            (4912, 'AndorBaselineClamp'),
            (4913, 'AndorPrescans'),
            (4914, 'AndorModel'),
            (4915, 'AndorChipSizeX'),
            (4916, 'AndorChipSizeY'),
            (4944, 'AndorBaselineOffset'),
            (4966, 'AndorSoftwareVersion'),
            (18246, 'Rating'),
            (18247, 'XP_DIP_XML'),
            (18248, 'StitchInfo'),
            (18249, 'RatingPercent'),
            (20481, 'ResolutionXUnit'),  # GDI+
            (20482, 'ResolutionYUnit'),  # GDI+
            (20483, 'ResolutionXLengthUnit'),  # GDI+
            (20484, 'ResolutionYLengthUnit'),  # GDI+
            (20485, 'PrintFlags'),  # GDI+
            (20486, 'PrintFlagsVersion'),  # GDI+
            (20487, 'PrintFlagsCrop'),  # GDI+
            (20488, 'PrintFlagsBleedWidth'),  # GDI+
            (20489, 'PrintFlagsBleedWidthScale'),  # GDI+
            (20490, 'HalftoneLPI'),  # GDI+
            (20491, 'HalftoneLPIUnit'),  # GDI+
            (20492, 'HalftoneDegree'),  # GDI+
            (20493, 'HalftoneShape'),  # GDI+
            (20494, 'HalftoneMisc'),  # GDI+
            (20495, 'HalftoneScreen'),  # GDI+
            (20496, 'JPEGQuality'),  # GDI+
            (20497, 'GridSize'),  # GDI+
            (20498, 'ThumbnailFormat'),  # GDI+
            (20499, 'ThumbnailWidth'),  # GDI+
            (20500, 'ThumbnailHeight'),  # GDI+
            (20501, 'ThumbnailColorDepth'),  # GDI+
            (20502, 'ThumbnailPlanes'),  # GDI+
            (20503, 'ThumbnailRawBytes'),  # GDI+
            (20504, 'ThumbnailSize'),  # GDI+
            (20505, 'ThumbnailCompressedSize'),  # GDI+
            (20506, 'ColorTransferFunction'),  # GDI+
            (20507, 'ThumbnailData'),
            (20512, 'ThumbnailImageWidth'),  # GDI+
            (20513, 'ThumbnailImageHeight'),  # GDI+
            (20514, 'ThumbnailBitsPerSample'),  # GDI+
            (20515, 'ThumbnailCompression'),
            (20516, 'ThumbnailPhotometricInterp'),  # GDI+
            (20517, 'ThumbnailImageDescription'),  # GDI+
            (20518, 'ThumbnailEquipMake'),  # GDI+
            (20519, 'ThumbnailEquipModel'),  # GDI+
            (20520, 'ThumbnailStripOffsets'),  # GDI+
            (20521, 'ThumbnailOrientation'),  # GDI+
            (20522, 'ThumbnailSamplesPerPixel'),  # GDI+
            (20523, 'ThumbnailRowsPerStrip'),  # GDI+
            (20524, 'ThumbnailStripBytesCount'),  # GDI+
            (20525, 'ThumbnailResolutionX'),
            (20526, 'ThumbnailResolutionY'),
            (20527, 'ThumbnailPlanarConfig'),  # GDI+
            (20528, 'ThumbnailResolutionUnit'),
            (20529, 'ThumbnailTransferFunction'),
            (20530, 'ThumbnailSoftwareUsed'),  # GDI+
            (20531, 'ThumbnailDateTime'),  # GDI+
            (20532, 'ThumbnailArtist'),  # GDI+
            (20533, 'ThumbnailWhitePoint'),  # GDI+
            (20534, 'ThumbnailPrimaryChromaticities'),  # GDI+
            (20535, 'ThumbnailYCbCrCoefficients'),  # GDI+
            (20536, 'ThumbnailYCbCrSubsampling'),  # GDI+
            (20537, 'ThumbnailYCbCrPositioning'),
            (20538, 'ThumbnailRefBlackWhite'),  # GDI+
            (20539, 'ThumbnailCopyRight'),  # GDI+
            (20545, 'InteroperabilityIndex'),
            (20546, 'InteroperabilityVersion'),
            (20624, 'LuminanceTable'),
            (20625, 'ChrominanceTable'),
            (20736, 'FrameDelay'),  # GDI+
            (20737, 'LoopCount'),  # GDI+
            (20738, 'GlobalPalette'),  # GDI+
            (20739, 'IndexBackground'),  # GDI+
            (20740, 'IndexTransparent'),  # GDI+
            (20752, 'PixelUnit'),  # GDI+
            (20753, 'PixelPerUnitX'),  # GDI+
            (20754, 'PixelPerUnitY'),  # GDI+
            (20755, 'PaletteHistogram'),  # GDI+
            (28672, 'SonyRawFileType'),  # Sony ARW
            (28722, 'VignettingCorrParams'),  # Sony ARW
            (28725, 'ChromaticAberrationCorrParams'),  # Sony ARW
            (28727, 'DistortionCorrParams'),  # Sony ARW
            # Private tags >= 32768
            (32781, 'ImageID'),
            (32931, 'WangTag1'),
            (32932, 'WangAnnotation'),
            (32933, 'WangTag3'),
            (32934, 'WangTag4'),
            (32953, 'ImageReferencePoints'),
            (32954, 'RegionXformTackPoint'),
            (32955, 'WarpQuadrilateral'),
            (32956, 'AffineTransformMat'),
            (32995, 'Matteing'),
            (32996, 'DataType'),  # use SampleFormat
            (32997, 'ImageDepth'),
            (32998, 'TileDepth'),
            (33300, 'ImageFullWidth'),
            (33301, 'ImageFullLength'),
            (33302, 'TextureFormat'),
            (33303, 'TextureWrapModes'),
            (33304, 'FieldOfViewCotangent'),
            (33305, 'MatrixWorldToScreen'),
            (33306, 'MatrixWorldToCamera'),
            (33405, 'Model2'),
            (33421, 'CFARepeatPatternDim'),
            (33422, 'CFAPattern'),
            (33423, 'BatteryLevel'),
            (33424, 'KodakIFD'),
            (33434, 'ExposureTime'),
            (33437, 'FNumber'),
            (33432, 'Copyright'),
            (33445, 'MDFileTag'),
            (33446, 'MDScalePixel'),
            (33447, 'MDColorTable'),
            (33448, 'MDLabName'),
            (33449, 'MDSampleInfo'),
            (33450, 'MDPrepDate'),
            (33451, 'MDPrepTime'),
            (33452, 'MDFileUnits'),
            (33471, 'OlympusINI'),
            (33550, 'ModelPixelScaleTag'),
            (33560, 'OlympusSIS'),  # see also 33471 and 34853
            (33589, 'AdventScale'),
            (33590, 'AdventRevision'),
            (33628, 'UIC1tag'),  # Metamorph  Universal Imaging Corp STK
            (33629, 'UIC2tag'),
            (33630, 'UIC3tag'),
            (33631, 'UIC4tag'),
            (33723, 'IPTCNAA'),
            (33858, 'ExtendedTagsOffset'),  # DEFF points IFD with private tags
            (33918, 'IntergraphPacketData'),  # INGRPacketDataTag
            (33919, 'IntergraphFlagRegisters'),  # INGRFlagRegisters
            (33920, 'IntergraphMatrixTag'),  # IrasBTransformationMatrix
            (33921, 'INGRReserved'),
            (33922, 'ModelTiepointTag'),
            (33923, 'LeicaMagic'),
            (34016, 'Site'),  # 34016..34032 ANSI IT8 TIFF/IT
            (34017, 'ColorSequence'),
            (34018, 'IT8Header'),
            (34019, 'RasterPadding'),
            (34020, 'BitsPerRunLength'),
            (34021, 'BitsPerExtendedRunLength'),
            (34022, 'ColorTable'),
            (34023, 'ImageColorIndicator'),
            (34024, 'BackgroundColorIndicator'),
            (34025, 'ImageColorValue'),
            (34026, 'BackgroundColorValue'),
            (34027, 'PixelIntensityRange'),
            (34028, 'TransparencyIndicator'),
            (34029, 'ColorCharacterization'),
            (34030, 'HCUsage'),
            (34031, 'TrapIndicator'),
            (34032, 'CMYKEquivalent'),
            (34118, 'CZ_SEM'),  # Zeiss SEM
            (34152, 'AFCP_IPTC'),
            (34232, 'PixelMagicJBIGOptions'),  # EXIF, also TI FrameCount
            (34263, 'JPLCartoIFD'),
            (34122, 'IPLAB'),  # number of images
            (34264, 'ModelTransformationTag'),
            (34306, 'WB_GRGBLevels'),  # Leaf MOS
            (34310, 'LeafData'),
            (34361, 'MM_Header'),
            (34362, 'MM_Stamp'),
            (34363, 'MM_Unknown'),
            (34377, 'ImageResources'),  # Photoshop
            (34386, 'MM_UserBlock'),
            (34412, 'CZ_LSMINFO'),
            (34665, 'ExifTag'),
            (34675, 'InterColorProfile'),  # ICCProfile
            (34680, 'FEI_SFEG'),  #
            (34682, 'FEI_HELIOS'),  #
            (34683, 'FEI_TITAN'),  #
            (34687, 'FXExtensions'),
            (34688, 'MultiProfiles'),
            (34689, 'SharedData'),
            (34690, 'T88Options'),
            (34710, 'MarCCD'),  # offset to MarCCD header
            (34732, 'ImageLayer'),
            (34735, 'GeoKeyDirectoryTag'),
            (34736, 'GeoDoubleParamsTag'),
            (34737, 'GeoAsciiParamsTag'),
            (34750, 'JBIGOptions'),
            (34821, 'PIXTIFF'),  # ? Pixel Translations Inc
            (34850, 'ExposureProgram'),
            (34852, 'SpectralSensitivity'),
            (34853, 'GPSTag'),  # GPSIFD  also OlympusSIS2
            (34853, 'OlympusSIS2'),
            (34855, 'ISOSpeedRatings'),
            (34856, 'OECF'),
            (34857, 'Interlace'),
            (34858, 'TimeZoneOffset'),
            (34859, 'SelfTimerMode'),
            (34864, 'SensitivityType'),
            (34865, 'StandardOutputSensitivity'),
            (34866, 'RecommendedExposureIndex'),
            (34867, 'ISOSpeed'),
            (34868, 'ISOSpeedLatitudeyyy'),
            (34869, 'ISOSpeedLatitudezzz'),
            (34908, 'HylaFAXFaxRecvParams'),
            (34909, 'HylaFAXFaxSubAddress'),
            (34910, 'HylaFAXFaxRecvTime'),
            (34911, 'FaxDcs'),
            (34929, 'FedexEDR'),
            (34954, 'LeafSubIFD'),
            (34959, 'Aphelion1'),
            (34960, 'Aphelion2'),
            (34961, 'AphelionInternal'),  # ADCIS
            (36864, 'ExifVersion'),
            (36867, 'DateTimeOriginal'),
            (36868, 'DateTimeDigitized'),
            (36873, 'GooglePlusUploadCode'),
            (36880, 'OffsetTime'),
            (36881, 'OffsetTimeOriginal'),
            (36882, 'OffsetTimeDigitized'),
            # TODO, Pilatus/CHESS/TV6 36864..37120 conflicting with Exif tags
            (36864, 'TVX_Unknown'),
            (36865, 'TVX_NumExposure'),
            (36866, 'TVX_NumBackground'),
            (36867, 'TVX_ExposureTime'),
            (36868, 'TVX_BackgroundTime'),
            (36870, 'TVX_Unknown'),
            (36873, 'TVX_SubBpp'),
            (36874, 'TVX_SubWide'),
            (36875, 'TVX_SubHigh'),
            (36876, 'TVX_BlackLevel'),
            (36877, 'TVX_DarkCurrent'),
            (36878, 'TVX_ReadNoise'),
            (36879, 'TVX_DarkCurrentNoise'),
            (36880, 'TVX_BeamMonitor'),
            (37120, 'TVX_UserVariables'),  # A/D values
            (37121, 'ComponentsConfiguration'),
            (37122, 'CompressedBitsPerPixel'),
            (37377, 'ShutterSpeedValue'),
            (37378, 'ApertureValue'),
            (37379, 'BrightnessValue'),
            (37380, 'ExposureBiasValue'),
            (37381, 'MaxApertureValue'),
            (37382, 'SubjectDistance'),
            (37383, 'MeteringMode'),
            (37384, 'LightSource'),
            (37385, 'Flash'),
            (37386, 'FocalLength'),
            (37387, 'FlashEnergy'),  # 37387
            (37388, 'SpatialFrequencyResponse'),  # 37388
            (37389, 'Noise'),
            (37390, 'FocalPlaneXResolution'),
            (37391, 'FocalPlaneYResolution'),
            (37392, 'FocalPlaneResolutionUnit'),
            (37393, 'ImageNumber'),
            (37394, 'SecurityClassification'),
            (37395, 'ImageHistory'),
            (37396, 'SubjectLocation'),
            (37397, 'ExposureIndex'),
            (37398, 'TIFFEPStandardID'),
            (37399, 'SensingMethod'),
            (37434, 'CIP3DataFile'),
            (37435, 'CIP3Sheet'),
            (37436, 'CIP3Side'),
            (37439, 'StoNits'),
            (37500, 'MakerNote'),
            (37510, 'UserComment'),
            (37520, 'SubsecTime'),
            (37521, 'SubsecTimeOriginal'),
            (37522, 'SubsecTimeDigitized'),
            (37679, 'MODIText'),  # Microsoft Office Document Imaging
            (37680, 'MODIOLEPropertySetStorage'),
            (37681, 'MODIPositioning'),
            (37706, 'TVIPS'),  # offset to TemData structure
            (37707, 'TVIPS1'),
            (37708, 'TVIPS2'),  # same TemData structure as undefined
            (37724, 'ImageSourceData'),  # Photoshop
            (37888, 'Temperature'),
            (37889, 'Humidity'),
            (37890, 'Pressure'),
            (37891, 'WaterDepth'),
            (37892, 'Acceleration'),
            (37893, 'CameraElevationAngle'),
            (40000, 'XPos'),   # Janelia
            (40001, 'YPos'),
            (40002, 'ZPos'),
            (40001, 'MC_IpWinScal'),  # Media Cybernetics
            (40001, 'RecipName'),  # MS FAX
            (40002, 'RecipNumber'),
            (40003, 'SenderName'),
            (40004, 'Routing'),
            (40005, 'CallerId'),
            (40006, 'TSID'),
            (40007, 'CSID'),
            (40008, 'FaxTime'),
            (40100, 'MC_IdOld'),
            (40106, 'MC_Unknown'),
            (40965, 'InteroperabilityTag'),  # InteropOffset
            (40091, 'XPTitle'),
            (40092, 'XPComment'),
            (40093, 'XPAuthor'),
            (40094, 'XPKeywords'),
            (40095, 'XPSubject'),
            (40960, 'FlashpixVersion'),
            (40961, 'ColorSpace'),
            (40962, 'PixelXDimension'),
            (40963, 'PixelYDimension'),
            (40964, 'RelatedSoundFile'),
            (40976, 'SamsungRawPointersOffset'),
            (40977, 'SamsungRawPointersLength'),
            (41217, 'SamsungRawByteOrder'),
            (41218, 'SamsungRawUnknown'),
            (41483, 'FlashEnergy'),
            (41484, 'SpatialFrequencyResponse'),
            (41485, 'Noise'),  # 37389
            (41486, 'FocalPlaneXResolution'),  # 37390
            (41487, 'FocalPlaneYResolution'),  # 37391
            (41488, 'FocalPlaneResolutionUnit'),  # 37392
            (41489, 'ImageNumber'),  # 37393
            (41490, 'SecurityClassification'),  # 37394
            (41491, 'ImageHistory'),  # 37395
            (41492, 'SubjectLocation'),  # 37395
            (41493, 'ExposureIndex '),  # 37397
            (41494, 'TIFF-EPStandardID'),
            (41495, 'SensingMethod'),  # 37399
            (41728, 'FileSource'),
            (41729, 'SceneType'),
            (41730, 'CFAPattern'),  # 33422
            (41985, 'CustomRendered'),
            (41986, 'ExposureMode'),
            (41987, 'WhiteBalance'),
            (41988, 'DigitalZoomRatio'),
            (41989, 'FocalLengthIn35mmFilm'),
            (41990, 'SceneCaptureType'),
            (41991, 'GainControl'),
            (41992, 'Contrast'),
            (41993, 'Saturation'),
            (41994, 'Sharpness'),
            (41995, 'DeviceSettingDescription'),
            (41996, 'SubjectDistanceRange'),
            (42016, 'ImageUniqueID'),
            (42032, 'CameraOwnerName'),
            (42033, 'BodySerialNumber'),
            (42034, 'LensSpecification'),
            (42035, 'LensMake'),
            (42036, 'LensModel'),
            (42037, 'LensSerialNumber'),
            (42112, 'GDAL_METADATA'),
            (42113, 'GDAL_NODATA'),
            (42240, 'Gamma'),
            (43314, 'NIHImageHeader'),
            (44992, 'ExpandSoftware'),
            (44993, 'ExpandLens'),
            (44994, 'ExpandFilm'),
            (44995, 'ExpandFilterLens'),
            (44996, 'ExpandScanner'),
            (44997, 'ExpandFlashLamp'),
            (48129, 'PixelFormat'),  # HDP and WDP
            (48130, 'Transformation'),
            (48131, 'Uncompressed'),
            (48132, 'ImageType'),
            (48256, 'ImageWidth'),  # 256
            (48257, 'ImageHeight'),
            (48258, 'WidthResolution'),
            (48259, 'HeightResolution'),
            (48320, 'ImageOffset'),
            (48321, 'ImageByteCount'),
            (48322, 'AlphaOffset'),
            (48323, 'AlphaByteCount'),
            (48324, 'ImageDataDiscard'),
            (48325, 'AlphaDataDiscard'),
            (50003, 'KodakAPP3'),
            (50215, 'OceScanjobDescription'),
            (50216, 'OceApplicationSelector'),
            (50217, 'OceIdentificationNumber'),
            (50218, 'OceImageLogicCharacteristics'),
            (50255, 'Annotations'),
            (50288, 'MC_Id'),  # Media Cybernetics
            (50289, 'MC_XYPosition'),
            (50290, 'MC_ZPosition'),
            (50291, 'MC_XYCalibration'),
            (50292, 'MC_LensCharacteristics'),
            (50293, 'MC_ChannelName'),
            (50294, 'MC_ExcitationWavelength'),
            (50295, 'MC_TimeStamp'),
            (50296, 'MC_FrameProperties'),
            (50341, 'PrintImageMatching'),
            (50495, 'PCO_RAW'),  # TODO, PCO CamWare
            (50547, 'OriginalFileName'),
            (50560, 'USPTO_OriginalContentType'),  # US Patent Office
            (50561, 'USPTO_RotationCode'),
            (50648, 'CR2Unknown1'),
            (50649, 'CR2Unknown2'),
            (50656, 'CR2CFAPattern'),
            (50674, 'LercParameters'),  # ESGI 50674 .. 50677
            (50706, 'DNGVersion'),  # DNG 50706 .. 51112
            (50707, 'DNGBackwardVersion'),
            (50708, 'UniqueCameraModel'),
            (50709, 'LocalizedCameraModel'),
            (50710, 'CFAPlaneColor'),
            (50711, 'CFALayout'),
            (50712, 'LinearizationTable'),
            (50713, 'BlackLevelRepeatDim'),
            (50714, 'BlackLevel'),
            (50715, 'BlackLevelDeltaH'),
            (50716, 'BlackLevelDeltaV'),
            (50717, 'WhiteLevel'),
            (50718, 'DefaultScale'),
            (50719, 'DefaultCropOrigin'),
            (50720, 'DefaultCropSize'),
            (50721, 'ColorMatrix1'),
            (50722, 'ColorMatrix2'),
            (50723, 'CameraCalibration1'),
            (50724, 'CameraCalibration2'),
            (50725, 'ReductionMatrix1'),
            (50726, 'ReductionMatrix2'),
            (50727, 'AnalogBalance'),
            (50728, 'AsShotNeutral'),
            (50729, 'AsShotWhiteXY'),
            (50730, 'BaselineExposure'),
            (50731, 'BaselineNoise'),
            (50732, 'BaselineSharpness'),
            (50733, 'BayerGreenSplit'),
            (50734, 'LinearResponseLimit'),
            (50735, 'CameraSerialNumber'),
            (50736, 'LensInfo'),
            (50737, 'ChromaBlurRadius'),
            (50738, 'AntiAliasStrength'),
            (50739, 'ShadowScale'),
            (50740, 'DNGPrivateData'),
            (50741, 'MakerNoteSafety'),
            (50752, 'RawImageSegmentation'),
            (50778, 'CalibrationIlluminant1'),
            (50779, 'CalibrationIlluminant2'),
            (50780, 'BestQualityScale'),
            (50781, 'RawDataUniqueID'),
            (50784, 'AliasLayerMetadata'),
            (50827, 'OriginalRawFileName'),
            (50828, 'OriginalRawFileData'),
            (50829, 'ActiveArea'),
            (50830, 'MaskedAreas'),
            (50831, 'AsShotICCProfile'),
            (50832, 'AsShotPreProfileMatrix'),
            (50833, 'CurrentICCProfile'),
            (50834, 'CurrentPreProfileMatrix'),
            (50838, 'IJMetadataByteCounts'),
            (50839, 'IJMetadata'),
            (50844, 'RPCCoefficientTag'),
            (50879, 'ColorimetricReference'),
            (50885, 'SRawType'),
            (50898, 'PanasonicTitle'),
            (50899, 'PanasonicTitle2'),
            (50908, 'RSID'),  # DGIWG
            (50909, 'GEO_METADATA'),  # DGIWG XML
            (50931, 'CameraCalibrationSignature'),
            (50932, 'ProfileCalibrationSignature'),
            (50933, 'ProfileIFD'),
            (50934, 'AsShotProfileName'),
            (50935, 'NoiseReductionApplied'),
            (50936, 'ProfileName'),
            (50937, 'ProfileHueSatMapDims'),
            (50938, 'ProfileHueSatMapData1'),
            (50939, 'ProfileHueSatMapData2'),
            (50940, 'ProfileToneCurve'),
            (50941, 'ProfileEmbedPolicy'),
            (50942, 'ProfileCopyright'),
            (50964, 'ForwardMatrix1'),
            (50965, 'ForwardMatrix2'),
            (50966, 'PreviewApplicationName'),
            (50967, 'PreviewApplicationVersion'),
            (50968, 'PreviewSettingsName'),
            (50969, 'PreviewSettingsDigest'),
            (50970, 'PreviewColorSpace'),
            (50971, 'PreviewDateTime'),
            (50972, 'RawImageDigest'),
            (50973, 'OriginalRawFileDigest'),
            (50974, 'SubTileBlockSize'),
            (50975, 'RowInterleaveFactor'),
            (50981, 'ProfileLookTableDims'),
            (50982, 'ProfileLookTableData'),
            (51008, 'OpcodeList1'),
            (51009, 'OpcodeList2'),
            (51022, 'OpcodeList3'),
            (51023, 'FibicsXML'),  #
            (51041, 'NoiseProfile'),
            (51043, 'TimeCodes'),
            (51044, 'FrameRate'),
            (51058, 'TStop'),
            (51081, 'ReelName'),
            (51089, 'OriginalDefaultFinalSize'),
            (51090, 'OriginalBestQualitySize'),
            (51091, 'OriginalDefaultCropSize'),
            (51105, 'CameraLabel'),
            (51107, 'ProfileHueSatMapEncoding'),
            (51108, 'ProfileLookTableEncoding'),
            (51109, 'BaselineExposureOffset'),
            (51110, 'DefaultBlackRender'),
            (51111, 'NewRawImageDigest'),
            (51112, 'RawToPreviewGain'),
            (51125, 'DefaultUserCrop'),
            (51123, 'MicroManagerMetadata'),
            (51159, 'ZIFmetadata'),  # Objective Pathology Services
            (51160, 'ZIFannotations'),  # Objective Pathology Services
            (59932, 'Padding'),
            (59933, 'OffsetSchema'),
            # Reusable Tags 65000-65535
            # (65000,  Dimap_Document XML
            # (65000-65112,  Photoshop Camera RAW EXIF tags
            # (65000, 'OwnerName'),
            # (65001, 'SerialNumber'),
            # (65002, 'Lens'),
            # (65024, 'KDC_IFD'),
            # (65100, 'RawFile'),
            # (65101, 'Converter'),
            # (65102, 'WhiteBalance'),
            # (65105, 'Exposure'),
            # (65106, 'Shadows'),
            # (65107, 'Brightness'),
            # (65108, 'Contrast'),
            # (65109, 'Saturation'),
            # (65110, 'Sharpness'),
            # (65111, 'Smoothness'),
            # (65112, 'MoireFilter'),
            (65200, 'FlexXML'),
        ))

    def TAG_READERS():
        # map tag codes to import functions
        return {
            320: read_colormap,
            # 700: read_bytes,  # read_utf8,
            # 34377: read_bytes,
            33723: read_bytes,
            # 34675: read_bytes,
            33628: read_uic1tag,  # Universal Imaging Corp STK
            33629: read_uic2tag,
            33630: read_uic3tag,
            33631: read_uic4tag,
            34118: read_cz_sem,  # Carl Zeiss SEM
            34361: read_mm_header,  # Olympus FluoView
            34362: read_mm_stamp,
            34363: read_numpy,  # MM_Unknown
            34386: read_numpy,  # MM_UserBlock
            34412: read_cz_lsminfo,  # Carl Zeiss LSM
            34680: read_fei_metadata,  # S-FEG
            34682: read_fei_metadata,  # Helios NanoLab
            37706: read_tvips_header,  # TVIPS EMMENU
            37724: read_bytes,  # ImageSourceData
            33923: read_bytes,  # read_leica_magic
            43314: read_nih_image_header,
            # 40001: read_bytes,
            40100: read_bytes,
            50288: read_bytes,
            50296: read_bytes,
            50839: read_bytes,
            51123: read_json,
            33471: read_sis_ini,
            33560: read_sis,
            34665: read_exif_ifd,
            34853: read_gps_ifd,  # conflicts with OlympusSIS
            40965: read_interoperability_ifd,
        }

    def TAG_TUPLE():
        # tags whose values must be stored as tuples
        return frozenset((273, 279, 324, 325, 330, 530, 531, 34736))

    def TAG_ATTRIBUTES():
        # map tag codes to TiffPage attribute names
        return {
            254: 'subfiletype',
            256: 'imagewidth',
            257: 'imagelength',
            258: 'bitspersample',
            259: 'compression',
            262: 'photometric',
            266: 'fillorder',
            270: 'description',
            277: 'samplesperpixel',
            278: 'rowsperstrip',
            284: 'planarconfig',
            305: 'software',
            320: 'colormap',
            317: 'predictor',
            322: 'tilewidth',
            323: 'tilelength',
            338: 'extrasamples',
            339: 'sampleformat',
            32997: 'imagedepth',
            32998: 'tiledepth',
        }

    def TAG_ENUM():
        # map tag codes to Enums
        return {
            254: TIFF.FILETYPE,
            255: TIFF.OFILETYPE,
            259: TIFF.COMPRESSION,
            262: TIFF.PHOTOMETRIC,
            263: TIFF.THRESHHOLD,
            266: TIFF.FILLORDER,
            274: TIFF.ORIENTATION,
            284: TIFF.PLANARCONFIG,
            290: TIFF.GRAYRESPONSEUNIT,
            # 292: TIFF.GROUP3OPT,
            # 293: TIFF.GROUP4OPT,
            296: TIFF.RESUNIT,
            300: TIFF.COLORRESPONSEUNIT,
            317: TIFF.PREDICTOR,
            338: TIFF.EXTRASAMPLE,
            339: TIFF.SAMPLEFORMAT,
            # 512: TIFF.JPEGPROC,
            # 531: TIFF.YCBCRPOSITION,
        }

    def FILETYPE():
        class FILETYPE(enum.IntFlag):
            UNDEFINED = 0
            REDUCEDIMAGE = 1
            PAGE = 2
            MASK = 4
            UNKNOWN = 8  # found in AperioSVS

        return FILETYPE

    def OFILETYPE():
        class OFILETYPE(enum.IntEnum):
            UNDEFINED = 0
            IMAGE = 1
            REDUCEDIMAGE = 2
            PAGE = 3

        return OFILETYPE

    def COMPRESSION():
        class COMPRESSION(enum.IntEnum):
            NONE = 1  # Uncompressed
            CCITTRLE = 2  # CCITT 1D
            CCITT_T4 = 3  # 'T4/Group 3 Fax',
            CCITT_T6 = 4  # 'T6/Group 4 Fax',
            LZW = 5
            OJPEG = 6  # old-style JPEG
            JPEG = 7
            ADOBE_DEFLATE = 8
            JBIG_BW = 9
            JBIG_COLOR = 10
            JPEG_99 = 99
            KODAK_262 = 262
            NEXT = 32766
            SONY_ARW = 32767
            PACKED_RAW = 32769
            SAMSUNG_SRW = 32770
            CCIRLEW = 32771
            SAMSUNG_SRW2 = 32772
            PACKBITS = 32773
            THUNDERSCAN = 32809
            IT8CTPAD = 32895
            IT8LW = 32896
            IT8MP = 32897
            IT8BL = 32898
            PIXARFILM = 32908
            PIXARLOG = 32909
            DEFLATE = 32946
            DCS = 32947
            APERIO_JP2000_YCBC = 33003  # Leica Aperio
            APERIO_JP2000_RGB = 33005  # Leica Aperio
            JBIG = 34661
            SGILOG = 34676
            SGILOG24 = 34677
            JPEG2000 = 34712
            NIKON_NEF = 34713
            JBIG2 = 34715
            MDI_BINARY = 34718  # Microsoft Document Imaging
            MDI_PROGRESSIVE = 34719  # Microsoft Document Imaging
            MDI_VECTOR = 34720  # Microsoft Document Imaging
            LERC = 34887  # ESRI Lerc
            JPEG_LOSSY = 34892
            LZMA = 34925
            ZSTD_DEPRECATED = 34926
            WEBP_DEPRECATED = 34927
            PNG = 34933  # Objective Pathology Services
            JPEGXR = 34934  # Objective Pathology Services
            ZSTD = 50000
            WEBP = 50001
            PIXTIFF = 50013
            KODAK_DCR = 65000
            PENTAX_PEF = 65535

            def __bool__(self):
                return self != 1

        return COMPRESSION

    def PHOTOMETRIC():
        class PHOTOMETRIC(enum.IntEnum):
            MINISWHITE = 0
            MINISBLACK = 1
            RGB = 2
            PALETTE = 3
            MASK = 4
            SEPARATED = 5  # CMYK
            YCBCR = 6
            CIELAB = 8
            ICCLAB = 9
            ITULAB = 10
            CFA = 32803  # Color Filter Array
            LOGL = 32844
            LOGLUV = 32845
            LINEAR_RAW = 34892

        return PHOTOMETRIC

    def THRESHHOLD():
        class THRESHHOLD(enum.IntEnum):
            BILEVEL = 1
            HALFTONE = 2
            ERRORDIFFUSE = 3

        return THRESHHOLD

    def FILLORDER():
        class FILLORDER(enum.IntEnum):
            MSB2LSB = 1
            LSB2MSB = 2

        return FILLORDER

    def ORIENTATION():
        class ORIENTATION(enum.IntEnum):
            TOPLEFT = 1
            TOPRIGHT = 2
            BOTRIGHT = 3
            BOTLEFT = 4
            LEFTTOP = 5
            RIGHTTOP = 6
            RIGHTBOT = 7
            LEFTBOT = 8

        return ORIENTATION

    def PLANARCONFIG():
        class PLANARCONFIG(enum.IntEnum):
            CONTIG = 1
            SEPARATE = 2

        return PLANARCONFIG

    def GRAYRESPONSEUNIT():
        class GRAYRESPONSEUNIT(enum.IntEnum):
            _10S = 1
            _100S = 2
            _1000S = 3
            _10000S = 4
            _100000S = 5

        return GRAYRESPONSEUNIT

    def GROUP4OPT():
        class GROUP4OPT(enum.IntEnum):
            UNCOMPRESSED = 2

        return GROUP4OPT

    def RESUNIT():
        class RESUNIT(enum.IntEnum):
            NONE = 1
            INCH = 2
            CENTIMETER = 3

            def __bool__(self):
                return self != 1

        return RESUNIT

    def COLORRESPONSEUNIT():
        class COLORRESPONSEUNIT(enum.IntEnum):
            _10S = 1
            _100S = 2
            _1000S = 3
            _10000S = 4
            _100000S = 5

        return COLORRESPONSEUNIT

    def PREDICTOR():
        class PREDICTOR(enum.IntEnum):
            NONE = 1
            HORIZONTAL = 2
            FLOATINGPOINT = 3

            def __bool__(self):
                return self != 1

        return PREDICTOR

    def EXTRASAMPLE():
        class EXTRASAMPLE(enum.IntEnum):
            UNSPECIFIED = 0
            ASSOCALPHA = 1
            UNASSALPHA = 2

        return EXTRASAMPLE

    def SAMPLEFORMAT():
        class SAMPLEFORMAT(enum.IntEnum):
            UINT = 1
            INT = 2
            IEEEFP = 3
            VOID = 4
            COMPLEXINT = 5
            COMPLEXIEEEFP = 6

        return SAMPLEFORMAT

    def DATATYPES():
        class DATATYPES(enum.IntEnum):
            NOTYPE = 0
            BYTE = 1
            ASCII = 2
            SHORT = 3
            LONG = 4
            RATIONAL = 5
            SBYTE = 6
            UNDEFINED = 7
            SSHORT = 8
            SLONG = 9
            SRATIONAL = 10
            FLOAT = 11
            DOUBLE = 12
            IFD = 13
            UNICODE = 14
            COMPLEX = 15
            LONG8 = 16
            SLONG8 = 17
            IFD8 = 18

        return DATATYPES

    def DATA_FORMATS():
        # map TIFF DATATYPES to Python struct formats
        return {
            1: '1B',   # BYTE 8-bit unsigned integer.
            2: '1s',   # ASCII 8-bit byte that contains a 7-bit ASCII code;
                       #   the last byte must be NULL (binary zero).
            3: '1H',   # SHORT 16-bit (2-byte) unsigned integer
            4: '1I',   # LONG 32-bit (4-byte) unsigned integer.
            5: '2I',   # RATIONAL Two LONGs: the first represents the numerator
                       #   of a fraction; the second, the denominator.
            6: '1b',   # SBYTE An 8-bit signed (twos-complement) integer.
            7: '1B',   # UNDEFINED An 8-bit byte that may contain anything,
                       #   depending on the definition of the field.
            8: '1h',   # SSHORT A 16-bit (2-byte) signed (twos-complement)
                       #   integer.
            9: '1i',   # SLONG A 32-bit (4-byte) signed (twos-complement)
                       #   integer.
            10: '2i',  # SRATIONAL Two SLONGs: the first represents the
                       #   numerator of a fraction, the second the denominator.
            11: '1f',  # FLOAT Single precision (4-byte) IEEE format.
            12: '1d',  # DOUBLE Double precision (8-byte) IEEE format.
            13: '1I',  # IFD unsigned 4 byte IFD offset.
            # 14: '',  # UNICODE
            # 15: '',  # COMPLEX
            16: '1Q',  # LONG8 unsigned 8 byte integer (BigTiff)
            17: '1q',  # SLONG8 signed 8 byte integer (BigTiff)
            18: '1Q',  # IFD8 unsigned 8 byte IFD offset (BigTiff)
        }

    def DATA_DTYPES():
        # map numpy dtypes to TIFF DATATYPES
        return {
            'B': 1,
            's': 2,
            'H': 3,
            'I': 4,
            '2I': 5,
            'b': 6,
            'h': 8,
            'i': 9,
            '2i': 10,
            'f': 11,
            'd': 12,
            'Q': 16,
            'q': 17,
        }

    def SAMPLE_DTYPES():
        # map SampleFormat and BitsPerSample to numpy dtype
        return {
            # UINT
            (1, 1): '?',  # bitmap
            (1, 2): 'B',
            (1, 3): 'B',
            (1, 4): 'B',
            (1, 5): 'B',
            (1, 6): 'B',
            (1, 7): 'B',
            (1, 8): 'B',
            (1, 9): 'H',
            (1, 10): 'H',
            (1, 11): 'H',
            (1, 12): 'H',
            (1, 13): 'H',
            (1, 14): 'H',
            (1, 15): 'H',
            (1, 16): 'H',
            (1, 17): 'I',
            (1, 18): 'I',
            (1, 19): 'I',
            (1, 20): 'I',
            (1, 21): 'I',
            (1, 22): 'I',
            (1, 23): 'I',
            (1, 24): 'I',
            (1, 25): 'I',
            (1, 26): 'I',
            (1, 27): 'I',
            (1, 28): 'I',
            (1, 29): 'I',
            (1, 30): 'I',
            (1, 31): 'I',
            (1, 32): 'I',
            (1, 64): 'Q',
            # VOID : treat as UINT
            (4, 1): '?',  # bitmap
            (4, 2): 'B',
            (4, 3): 'B',
            (4, 4): 'B',
            (4, 5): 'B',
            (4, 6): 'B',
            (4, 7): 'B',
            (4, 8): 'B',
            (4, 9): 'H',
            (4, 10): 'H',
            (4, 11): 'H',
            (4, 12): 'H',
            (4, 13): 'H',
            (4, 14): 'H',
            (4, 15): 'H',
            (4, 16): 'H',
            (4, 17): 'I',
            (4, 18): 'I',
            (4, 19): 'I',
            (4, 20): 'I',
            (4, 21): 'I',
            (4, 22): 'I',
            (4, 23): 'I',
            (4, 24): 'I',
            (4, 25): 'I',
            (4, 26): 'I',
            (4, 27): 'I',
            (4, 28): 'I',
            (4, 29): 'I',
            (4, 30): 'I',
            (4, 31): 'I',
            (4, 32): 'I',
            (4, 64): 'Q',
            # INT
            (2, 8): 'b',
            (2, 16): 'h',
            (2, 32): 'i',
            (2, 64): 'q',
            # IEEEFP : 24 bit not supported by numpy
            (3, 16): 'e',
            # (3, 24): '',  #
            (3, 32): 'f',
            (3, 64): 'd',
            # COMPLEXIEEEFP
            (6, 64): 'F',
            (6, 128): 'D',
            # RGB565
            (1, (5, 6, 5)): 'B',
            # COMPLEXINT : not supported by numpy
        }

    def PREDICTORS():
        # map PREDICTOR to predictor encode functions
        if imagecodecs is None:
            return {
                None: identityfunc,
                1: identityfunc,
                2: delta_encode,
            }
        return {
            None: imagecodecs.none_encode,
            1: imagecodecs.none_encode,
            2: imagecodecs.delta_encode,
            3: imagecodecs.floatpred_encode,
        }

    def UNPREDICTORS():
        # map PREDICTOR to predictor decode functions
        if imagecodecs is None:
            return {
                None: identityfunc,
                1: identityfunc,
                2: delta_decode,
            }
        return {
            None: imagecodecs.none_decode,
            1: imagecodecs.none_decode,
            2: imagecodecs.delta_decode,
            3: imagecodecs.floatpred_decode,
        }

    def COMPESSORS():
        # map COMPRESSION to compress functions
        if imagecodecs is None:
            return {
                None: identityfunc,
                1: identityfunc,
                8: zlib_encode,
                32946: zlib_encode,
                34925: lzma_encode,
            }
        return {
            None: imagecodecs.none_encode,
            1: imagecodecs.none_encode,
            7: imagecodecs.jpeg_encode,
            8: imagecodecs.zlib_encode,
            32946: imagecodecs.zlib_encode,
            32773: imagecodecs.packbits_encode,
            34712: imagecodecs.jpeg2k_encode,
            34925: imagecodecs.lzma_encode,
            34933: imagecodecs.png_encode,
            34934: imagecodecs.jpegxr_encode,
            50000: imagecodecs.zstd_encode,
            50001: imagecodecs.webp_encode,
        }

    def DECOMPESSORS():
        # map COMPRESSION to decompress functions
        if imagecodecs is None:
            return {
                None: identityfunc,
                1: identityfunc,
                8: zlib_decode,
                32946: zlib_decode,
                34925: lzma_decode,
            }
        return {
            None: imagecodecs.none_decode,
            1: imagecodecs.none_decode,
            5: imagecodecs.lzw_decode,
            6: imagecodecs.jpeg_decode,
            7: imagecodecs.jpeg_decode,
            8: imagecodecs.zlib_decode,
            32946: imagecodecs.zlib_decode,
            32773: imagecodecs.packbits_decode,
            # 34892: imagecodecs.jpeg_decode,  # DNG lossy
            34925: imagecodecs.lzma_decode,
            34926: imagecodecs.zstd_decode,  # deprecated
            34927: imagecodecs.webp_decode,  # deprecated
            33003: imagecodecs.jpeg2k_decode,
            33005: imagecodecs.jpeg2k_decode,
            34712: imagecodecs.jpeg2k_decode,
            34933: imagecodecs.png_decode,
            34934: imagecodecs.jpegxr_decode,
            50000: imagecodecs.zstd_decode,
            50001: imagecodecs.webp_decode,
        }

    def FRAME_ATTRS():
        # attributes that a TiffFrame shares with its keyframe
        return {
            'shape',
            'ndim',
            'size',
            'dtype',
            'axes',
            'is_final',
            'decode',
        }

    def FILE_FLAGS():
        # TiffFile and TiffPage 'is_\*' attributes
        exclude = {
            'reduced',
            'mask',
            'final',
            'memmappable',
            'contiguous',
            'tiled',
            'subsampled',
        }
        return {
            a[3:]
            for a in dir(TiffPage)
            if a[:3] == 'is_' and a[3:] not in exclude
        }

    def FILE_EXTENSIONS():
        # TIFF file extensions
        return (
            'tif', 'tiff', 'ome.tif', 'lsm', 'stk', 'qpi', 'pcoraw', 'qptiff',
            'gel', 'seq', 'svs', 'zif', 'ndpi', 'bif', 'tf8', 'tf2', 'btf',
        )

    def FILEOPEN_FILTER():
        # string for use in Windows File Open box
        return [
            (f'{ext.upper()} files', f'*.{ext}')
            for ext in TIFF.FILE_EXTENSIONS
        ] + [('allfiles', '*')]

    def AXES_LABELS():
        # TODO: is there a standard for character axes labels?
        axes = {
            'X': 'width',
            'Y': 'height',
            'Z': 'depth',
            'S': 'sample',  # rgb(a)
            'I': 'series',  # general sequence, plane, page, IFD
            'T': 'time',
            'C': 'channel',  # color, emission wavelength
            'A': 'angle',
            'P': 'phase',  # formerly F    # P is Position in LSM!
            'R': 'tile',  # region, point, mosaic
            'H': 'lifetime',  # histogram
            'E': 'lambda',  # excitation wavelength
            'L': 'exposure',  # lux
            'V': 'event',
            'Q': 'other',
            'M': 'mosaic',  # LSM 6
        }
        axes.update({v: k for k, v in axes.items()})
        return axes

    def NDPI_TAGS():
        # 65420 - 65458  Private Hamamatsu NDPI tags
        tags = {code: str(code) for code in range(65420, 65459)}
        tags.update({
            65420: 'FileFormat',
            65421: 'Magnification',  # SourceLens
            65422: 'XOffsetFromSlideCentre',
            65423: 'YOffsetFromSlideCentre',
            65424: 'ZOffsetFromSlideCentre',  # FocalPlane
            65426: 'McuStartsLowBytes',
            65427: 'UserLabel',  # Reference
            65428: 'AuthCode',  # ?
            65432: 'McuStartsHighBytes',
            65442: 'ScannerSerialNumber',
            65449: 'Comments',  # PropertyMap
            65447: 'BlankLanes',
            65434: 'Fluorescence',
        })
        return tags

    def EXIF_TAGS():
        # 65000 - 65112  Photoshop Camera RAW EXIF tags
        tags = {
            65000: 'OwnerName',
            65001: 'SerialNumber',
            65002: 'Lens',
            65100: 'RawFile',
            65101: 'Converter',
            65102: 'WhiteBalance',
            65105: 'Exposure',
            65106: 'Shadows',
            65107: 'Brightness',
            65108: 'Contrast',
            65109: 'Saturation',
            65110: 'Sharpness',
            65111: 'Smoothness',
            65112: 'MoireFilter',
        }
        tags.update(reversed(TIFF.TAGS.items()))  # TODO: rework this
        return tags

    def GPS_TAGS():
        return {
            0: 'GPSVersionID',
            1: 'GPSLatitudeRef',
            2: 'GPSLatitude',
            3: 'GPSLongitudeRef',
            4: 'GPSLongitude',
            5: 'GPSAltitudeRef',
            6: 'GPSAltitude',
            7: 'GPSTimeStamp',
            8: 'GPSSatellites',
            9: 'GPSStatus',
            10: 'GPSMeasureMode',
            11: 'GPSDOP',
            12: 'GPSSpeedRef',
            13: 'GPSSpeed',
            14: 'GPSTrackRef',
            15: 'GPSTrack',
            16: 'GPSImgDirectionRef',
            17: 'GPSImgDirection',
            18: 'GPSMapDatum',
            19: 'GPSDestLatitudeRef',
            20: 'GPSDestLatitude',
            21: 'GPSDestLongitudeRef',
            22: 'GPSDestLongitude',
            23: 'GPSDestBearingRef',
            24: 'GPSDestBearing',
            25: 'GPSDestDistanceRef',
            26: 'GPSDestDistance',
            27: 'GPSProcessingMethod',
            28: 'GPSAreaInformation',
            29: 'GPSDateStamp',
            30: 'GPSDifferential',
            31: 'GPSHPositioningError',
        }

    def IOP_TAGS():
        return {
            1: 'InteroperabilityIndex',
            2: 'InteroperabilityVersion',
            4096: 'RelatedImageFileFormat',
            4097: 'RelatedImageWidth',
            4098: 'RelatedImageLength',
        }

    def GEO_KEYS():
        return {
            1024: 'GTModelTypeGeoKey',
            1025: 'GTRasterTypeGeoKey',
            1026: 'GTCitationGeoKey',
            2048: 'GeographicTypeGeoKey',
            2049: 'GeogCitationGeoKey',
            2050: 'GeogGeodeticDatumGeoKey',
            2051: 'GeogPrimeMeridianGeoKey',
            2052: 'GeogLinearUnitsGeoKey',
            2053: 'GeogLinearUnitSizeGeoKey',
            2054: 'GeogAngularUnitsGeoKey',
            2055: 'GeogAngularUnitsSizeGeoKey',
            2056: 'GeogEllipsoidGeoKey',
            2057: 'GeogSemiMajorAxisGeoKey',
            2058: 'GeogSemiMinorAxisGeoKey',
            2059: 'GeogInvFlatteningGeoKey',
            2060: 'GeogAzimuthUnitsGeoKey',
            2061: 'GeogPrimeMeridianLongGeoKey',
            2062: 'GeogTOWGS84GeoKey',
            3059: 'ProjLinearUnitsInterpCorrectGeoKey',  # GDAL
            3072: 'ProjectedCSTypeGeoKey',
            3073: 'PCSCitationGeoKey',
            3074: 'ProjectionGeoKey',
            3075: 'ProjCoordTransGeoKey',
            3076: 'ProjLinearUnitsGeoKey',
            3077: 'ProjLinearUnitSizeGeoKey',
            3078: 'ProjStdParallel1GeoKey',
            3079: 'ProjStdParallel2GeoKey',
            3080: 'ProjNatOriginLongGeoKey',
            3081: 'ProjNatOriginLatGeoKey',
            3082: 'ProjFalseEastingGeoKey',
            3083: 'ProjFalseNorthingGeoKey',
            3084: 'ProjFalseOriginLongGeoKey',
            3085: 'ProjFalseOriginLatGeoKey',
            3086: 'ProjFalseOriginEastingGeoKey',
            3087: 'ProjFalseOriginNorthingGeoKey',
            3088: 'ProjCenterLongGeoKey',
            3089: 'ProjCenterLatGeoKey',
            3090: 'ProjCenterEastingGeoKey',
            3091: 'ProjFalseOriginNorthingGeoKey',
            3092: 'ProjScaleAtNatOriginGeoKey',
            3093: 'ProjScaleAtCenterGeoKey',
            3094: 'ProjAzimuthAngleGeoKey',
            3095: 'ProjStraightVertPoleLongGeoKey',
            3096: 'ProjRectifiedGridAngleGeoKey',
            4096: 'VerticalCSTypeGeoKey',
            4097: 'VerticalCitationGeoKey',
            4098: 'VerticalDatumGeoKey',
            4099: 'VerticalUnitsGeoKey',
        }

    def GEO_CODES():
        try:
            from .tifffile_geodb import GEO_CODES  # delayed import
        except ImportError:
            try:
                from tifffile_geodb import GEO_CODES  # delayed import
            except ImportError:
                GEO_CODES = {}
        return GEO_CODES

    def CZ_LSMINFO():
        return [
            ('MagicNumber', 'u4'),
            ('StructureSize', 'i4'),
            ('DimensionX', 'i4'),
            ('DimensionY', 'i4'),
            ('DimensionZ', 'i4'),
            ('DimensionChannels', 'i4'),
            ('DimensionTime', 'i4'),
            ('DataType', 'i4'),  # DATATYPES
            ('ThumbnailX', 'i4'),
            ('ThumbnailY', 'i4'),
            ('VoxelSizeX', 'f8'),
            ('VoxelSizeY', 'f8'),
            ('VoxelSizeZ', 'f8'),
            ('OriginX', 'f8'),
            ('OriginY', 'f8'),
            ('OriginZ', 'f8'),
            ('ScanType', 'u2'),
            ('SpectralScan', 'u2'),
            ('TypeOfData', 'u4'),  # TYPEOFDATA
            ('OffsetVectorOverlay', 'u4'),
            ('OffsetInputLut', 'u4'),
            ('OffsetOutputLut', 'u4'),
            ('OffsetChannelColors', 'u4'),
            ('TimeIntervall', 'f8'),
            ('OffsetChannelDataTypes', 'u4'),
            ('OffsetScanInformation', 'u4'),  # SCANINFO
            ('OffsetKsData', 'u4'),
            ('OffsetTimeStamps', 'u4'),
            ('OffsetEventList', 'u4'),
            ('OffsetRoi', 'u4'),
            ('OffsetBleachRoi', 'u4'),
            ('OffsetNextRecording', 'u4'),
            # LSM 2.0 ends here
            ('DisplayAspectX', 'f8'),
            ('DisplayAspectY', 'f8'),
            ('DisplayAspectZ', 'f8'),
            ('DisplayAspectTime', 'f8'),
            ('OffsetMeanOfRoisOverlay', 'u4'),
            ('OffsetTopoIsolineOverlay', 'u4'),
            ('OffsetTopoProfileOverlay', 'u4'),
            ('OffsetLinescanOverlay', 'u4'),
            ('ToolbarFlags', 'u4'),
            ('OffsetChannelWavelength', 'u4'),
            ('OffsetChannelFactors', 'u4'),
            ('ObjectiveSphereCorrection', 'f8'),
            ('OffsetUnmixParameters', 'u4'),
            # LSM 3.2, 4.0 end here
            ('OffsetAcquisitionParameters', 'u4'),
            ('OffsetCharacteristics', 'u4'),
            ('OffsetPalette', 'u4'),
            ('TimeDifferenceX', 'f8'),
            ('TimeDifferenceY', 'f8'),
            ('TimeDifferenceZ', 'f8'),
            ('InternalUse1', 'u4'),
            ('DimensionP', 'i4'),
            ('DimensionM', 'i4'),
            ('DimensionsReserved', '16i4'),
            ('OffsetTilePositions', 'u4'),
            ('', '9u4'),  # Reserved
            ('OffsetPositions', 'u4'),
            # ('', '21u4'),  # must be 0
        ]

    def CZ_LSMINFO_READERS():
        # import functions for CZ_LSMINFO sub-records
        # TODO: read more CZ_LSMINFO sub-records
        return {
            'ScanInformation': read_lsm_scaninfo,
            'TimeStamps': read_lsm_timestamps,
            'EventList': read_lsm_eventlist,
            'ChannelColors': read_lsm_channelcolors,
            'Positions': read_lsm_floatpairs,
            'TilePositions': read_lsm_floatpairs,
            'VectorOverlay': None,
            'InputLut': None,
            'OutputLut': None,
            'TimeIntervall': None,
            'ChannelDataTypes': None,
            'KsData': None,
            'Roi': None,
            'BleachRoi': None,
            'NextRecording': None,
            'MeanOfRoisOverlay': None,
            'TopoIsolineOverlay': None,
            'TopoProfileOverlay': None,
            'ChannelWavelength': None,
            'SphereCorrection': None,
            'ChannelFactors': None,
            'UnmixParameters': None,
            'AcquisitionParameters': None,
            'Characteristics': None,
        }

    def CZ_LSMINFO_SCANTYPE():
        # map CZ_LSMINFO.ScanType to dimension order
        return {
            0: 'XYZCT',  # 'Stack' normal x-y-z-scan
            1: 'XYZCT',  # 'Z-Scan' x-z-plane Y=1
            2: 'XYZCT',  # 'Line'
            3: 'XYTCZ',  # 'Time Series Plane' time series x-y  XYCTZ ? Z=1
            4: 'XYZTC',  # 'Time Series z-Scan' time series x-z
            5: 'XYTCZ',  # 'Time Series Mean-of-ROIs'
            6: 'XYZTC',  # 'Time Series Stack' time series x-y-z
            7: 'XYCTZ',  # Spline Scan
            8: 'XYCZT',  # Spline Plane x-z
            9: 'XYTCZ',  # Time Series Spline Plane x-z
            10: 'XYZCT',  # 'Time Series Point' point mode
        }

    def CZ_LSMINFO_DIMENSIONS():
        # map dimension codes to CZ_LSMINFO attribute
        return {
            'X': 'DimensionX',
            'Y': 'DimensionY',
            'Z': 'DimensionZ',
            'C': 'DimensionChannels',
            'T': 'DimensionTime',
            'P': 'DimensionP',
            'M': 'DimensionM',
        }

    def CZ_LSMINFO_DATATYPES():
        # description of CZ_LSMINFO.DataType
        return {
            0: 'varying data types',
            1: '8 bit unsigned integer',
            2: '12 bit unsigned integer',
            5: '32 bit float',
        }

    def CZ_LSMINFO_TYPEOFDATA():
        # description of CZ_LSMINFO.TypeOfData
        return {
            0: 'Original scan data',
            1: 'Calculated data',
            2: '3D reconstruction',
            3: 'Topography height map',
        }

    def CZ_LSMINFO_SCANINFO_ARRAYS():
        return {
            0x20000000: 'Tracks',
            0x30000000: 'Lasers',
            0x60000000: 'DetectionChannels',
            0x80000000: 'IlluminationChannels',
            0xA0000000: 'BeamSplitters',
            0xC0000000: 'DataChannels',
            0x11000000: 'Timers',
            0x13000000: 'Markers',
        }

    def CZ_LSMINFO_SCANINFO_STRUCTS():
        return {
            # 0x10000000: 'Recording',
            0x40000000: 'Track',
            0x50000000: 'Laser',
            0x70000000: 'DetectionChannel',
            0x90000000: 'IlluminationChannel',
            0xB0000000: 'BeamSplitter',
            0xD0000000: 'DataChannel',
            0x12000000: 'Timer',
            0x14000000: 'Marker',
        }

    def CZ_LSMINFO_SCANINFO_ATTRIBUTES():
        return {
            # Recording
            0x10000001: 'Name',
            0x10000002: 'Description',
            0x10000003: 'Notes',
            0x10000004: 'Objective',
            0x10000005: 'ProcessingSummary',
            0x10000006: 'SpecialScanMode',
            0x10000007: 'ScanType',
            0x10000008: 'ScanMode',
            0x10000009: 'NumberOfStacks',
            0x1000000A: 'LinesPerPlane',
            0x1000000B: 'SamplesPerLine',
            0x1000000C: 'PlanesPerVolume',
            0x1000000D: 'ImagesWidth',
            0x1000000E: 'ImagesHeight',
            0x1000000F: 'ImagesNumberPlanes',
            0x10000010: 'ImagesNumberStacks',
            0x10000011: 'ImagesNumberChannels',
            0x10000012: 'LinscanXySize',
            0x10000013: 'ScanDirection',
            0x10000014: 'TimeSeries',
            0x10000015: 'OriginalScanData',
            0x10000016: 'ZoomX',
            0x10000017: 'ZoomY',
            0x10000018: 'ZoomZ',
            0x10000019: 'Sample0X',
            0x1000001A: 'Sample0Y',
            0x1000001B: 'Sample0Z',
            0x1000001C: 'SampleSpacing',
            0x1000001D: 'LineSpacing',
            0x1000001E: 'PlaneSpacing',
            0x1000001F: 'PlaneWidth',
            0x10000020: 'PlaneHeight',
            0x10000021: 'VolumeDepth',
            0x10000023: 'Nutation',
            0x10000034: 'Rotation',
            0x10000035: 'Precession',
            0x10000036: 'Sample0time',
            0x10000037: 'StartScanTriggerIn',
            0x10000038: 'StartScanTriggerOut',
            0x10000039: 'StartScanEvent',
            0x10000040: 'StartScanTime',
            0x10000041: 'StopScanTriggerIn',
            0x10000042: 'StopScanTriggerOut',
            0x10000043: 'StopScanEvent',
            0x10000044: 'StopScanTime',
            0x10000045: 'UseRois',
            0x10000046: 'UseReducedMemoryRois',
            0x10000047: 'User',
            0x10000048: 'UseBcCorrection',
            0x10000049: 'PositionBcCorrection1',
            0x10000050: 'PositionBcCorrection2',
            0x10000051: 'InterpolationY',
            0x10000052: 'CameraBinning',
            0x10000053: 'CameraSupersampling',
            0x10000054: 'CameraFrameWidth',
            0x10000055: 'CameraFrameHeight',
            0x10000056: 'CameraOffsetX',
            0x10000057: 'CameraOffsetY',
            0x10000059: 'RtBinning',
            0x1000005A: 'RtFrameWidth',
            0x1000005B: 'RtFrameHeight',
            0x1000005C: 'RtRegionWidth',
            0x1000005D: 'RtRegionHeight',
            0x1000005E: 'RtOffsetX',
            0x1000005F: 'RtOffsetY',
            0x10000060: 'RtZoom',
            0x10000061: 'RtLinePeriod',
            0x10000062: 'Prescan',
            0x10000063: 'ScanDirectionZ',
            # Track
            0x40000001: 'MultiplexType',  # 0 After Line; 1 After Frame
            0x40000002: 'MultiplexOrder',
            0x40000003: 'SamplingMode',  # 0 Sample; 1 Line Avg; 2 Frame Avg
            0x40000004: 'SamplingMethod',  # 1 Mean; 2 Sum
            0x40000005: 'SamplingNumber',
            0x40000006: 'Acquire',
            0x40000007: 'SampleObservationTime',
            0x4000000B: 'TimeBetweenStacks',
            0x4000000C: 'Name',
            0x4000000D: 'Collimator1Name',
            0x4000000E: 'Collimator1Position',
            0x4000000F: 'Collimator2Name',
            0x40000010: 'Collimator2Position',
            0x40000011: 'IsBleachTrack',
            0x40000012: 'IsBleachAfterScanNumber',
            0x40000013: 'BleachScanNumber',
            0x40000014: 'TriggerIn',
            0x40000015: 'TriggerOut',
            0x40000016: 'IsRatioTrack',
            0x40000017: 'BleachCount',
            0x40000018: 'SpiCenterWavelength',
            0x40000019: 'PixelTime',
            0x40000021: 'CondensorFrontlens',
            0x40000023: 'FieldStopValue',
            0x40000024: 'IdCondensorAperture',
            0x40000025: 'CondensorAperture',
            0x40000026: 'IdCondensorRevolver',
            0x40000027: 'CondensorFilter',
            0x40000028: 'IdTransmissionFilter1',
            0x40000029: 'IdTransmission1',
            0x40000030: 'IdTransmissionFilter2',
            0x40000031: 'IdTransmission2',
            0x40000032: 'RepeatBleach',
            0x40000033: 'EnableSpotBleachPos',
            0x40000034: 'SpotBleachPosx',
            0x40000035: 'SpotBleachPosy',
            0x40000036: 'SpotBleachPosz',
            0x40000037: 'IdTubelens',
            0x40000038: 'IdTubelensPosition',
            0x40000039: 'TransmittedLight',
            0x4000003A: 'ReflectedLight',
            0x4000003B: 'SimultanGrabAndBleach',
            0x4000003C: 'BleachPixelTime',
            # Laser
            0x50000001: 'Name',
            0x50000002: 'Acquire',
            0x50000003: 'Power',
            # DetectionChannel
            0x70000001: 'IntegrationMode',
            0x70000002: 'SpecialMode',
            0x70000003: 'DetectorGainFirst',
            0x70000004: 'DetectorGainLast',
            0x70000005: 'AmplifierGainFirst',
            0x70000006: 'AmplifierGainLast',
            0x70000007: 'AmplifierOffsFirst',
            0x70000008: 'AmplifierOffsLast',
            0x70000009: 'PinholeDiameter',
            0x7000000A: 'CountingTrigger',
            0x7000000B: 'Acquire',
            0x7000000C: 'PointDetectorName',
            0x7000000D: 'AmplifierName',
            0x7000000E: 'PinholeName',
            0x7000000F: 'FilterSetName',
            0x70000010: 'FilterName',
            0x70000013: 'IntegratorName',
            0x70000014: 'ChannelName',
            0x70000015: 'DetectorGainBc1',
            0x70000016: 'DetectorGainBc2',
            0x70000017: 'AmplifierGainBc1',
            0x70000018: 'AmplifierGainBc2',
            0x70000019: 'AmplifierOffsetBc1',
            0x70000020: 'AmplifierOffsetBc2',
            0x70000021: 'SpectralScanChannels',
            0x70000022: 'SpiWavelengthStart',
            0x70000023: 'SpiWavelengthStop',
            0x70000026: 'DyeName',
            0x70000027: 'DyeFolder',
            # IlluminationChannel
            0x90000001: 'Name',
            0x90000002: 'Power',
            0x90000003: 'Wavelength',
            0x90000004: 'Aquire',
            0x90000005: 'DetchannelName',
            0x90000006: 'PowerBc1',
            0x90000007: 'PowerBc2',
            # BeamSplitter
            0xB0000001: 'FilterSet',
            0xB0000002: 'Filter',
            0xB0000003: 'Name',
            # DataChannel
            0xD0000001: 'Name',
            0xD0000003: 'Acquire',
            0xD0000004: 'Color',
            0xD0000005: 'SampleType',
            0xD0000006: 'BitsPerSample',
            0xD0000007: 'RatioType',
            0xD0000008: 'RatioTrack1',
            0xD0000009: 'RatioTrack2',
            0xD000000A: 'RatioChannel1',
            0xD000000B: 'RatioChannel2',
            0xD000000C: 'RatioConst1',
            0xD000000D: 'RatioConst2',
            0xD000000E: 'RatioConst3',
            0xD000000F: 'RatioConst4',
            0xD0000010: 'RatioConst5',
            0xD0000011: 'RatioConst6',
            0xD0000012: 'RatioFirstImages1',
            0xD0000013: 'RatioFirstImages2',
            0xD0000014: 'DyeName',
            0xD0000015: 'DyeFolder',
            0xD0000016: 'Spectrum',
            0xD0000017: 'Acquire',
            # Timer
            0x12000001: 'Name',
            0x12000002: 'Description',
            0x12000003: 'Interval',
            0x12000004: 'TriggerIn',
            0x12000005: 'TriggerOut',
            0x12000006: 'ActivationTime',
            0x12000007: 'ActivationNumber',
            # Marker
            0x14000001: 'Name',
            0x14000002: 'Description',
            0x14000003: 'TriggerIn',
            0x14000004: 'TriggerOut',
        }

    def NIH_IMAGE_HEADER():
        return [
            ('FileID', 'a8'),
            ('nLines', 'i2'),
            ('PixelsPerLine', 'i2'),
            ('Version', 'i2'),
            ('OldLutMode', 'i2'),
            ('OldnColors', 'i2'),
            ('Colors', 'u1', (3, 32)),
            ('OldColorStart', 'i2'),
            ('ColorWidth', 'i2'),
            ('ExtraColors', 'u2', (6, 3)),
            ('nExtraColors', 'i2'),
            ('ForegroundIndex', 'i2'),
            ('BackgroundIndex', 'i2'),
            ('XScale', 'f8'),
            ('Unused2', 'i2'),
            ('Unused3', 'i2'),
            ('UnitsID', 'i2'),  # NIH_UNITS_TYPE
            ('p1', [('x', 'i2'), ('y', 'i2')]),
            ('p2', [('x', 'i2'), ('y', 'i2')]),
            ('CurveFitType', 'i2'),  # NIH_CURVEFIT_TYPE
            ('nCoefficients', 'i2'),
            ('Coeff', 'f8', 6),
            ('UMsize', 'u1'),
            ('UM', 'a15'),
            ('UnusedBoolean', 'u1'),
            ('BinaryPic', 'b1'),
            ('SliceStart', 'i2'),
            ('SliceEnd', 'i2'),
            ('ScaleMagnification', 'f4'),
            ('nSlices', 'i2'),
            ('SliceSpacing', 'f4'),
            ('CurrentSlice', 'i2'),
            ('FrameInterval', 'f4'),
            ('PixelAspectRatio', 'f4'),
            ('ColorStart', 'i2'),
            ('ColorEnd', 'i2'),
            ('nColors', 'i2'),
            ('Fill1', '3u2'),
            ('Fill2', '3u2'),
            ('Table', 'u1'),  # NIH_COLORTABLE_TYPE
            ('LutMode', 'u1'),  # NIH_LUTMODE_TYPE
            ('InvertedTable', 'b1'),
            ('ZeroClip', 'b1'),
            ('XUnitSize', 'u1'),
            ('XUnit', 'a11'),
            ('StackType', 'i2'),  # NIH_STACKTYPE_TYPE
            # ('UnusedBytes', 'u1', 200)
        ]

    def NIH_COLORTABLE_TYPE():
        return (
            'CustomTable',
            'AppleDefault',
            'Pseudo20',
            'Pseudo32',
            'Rainbow',
            'Fire1',
            'Fire2',
            'Ice',
            'Grays',
            'Spectrum',
        )

    def NIH_LUTMODE_TYPE():
        return (
            'PseudoColor',
            'OldAppleDefault',
            'OldSpectrum',
            'GrayScale',
            'ColorLut',
            'CustomGrayscale',
        )

    def NIH_CURVEFIT_TYPE():
        return (
            'StraightLine',
            'Poly2',
            'Poly3',
            'Poly4',
            'Poly5',
            'ExpoFit',
            'PowerFit',
            'LogFit',
            'RodbardFit',
            'SpareFit1',
            'Uncalibrated',
            'UncalibratedOD',
        )

    def NIH_UNITS_TYPE():
        return (
            'Nanometers',
            'Micrometers',
            'Millimeters',
            'Centimeters',
            'Meters',
            'Kilometers',
            'Inches',
            'Feet',
            'Miles',
            'Pixels',
            'OtherUnits',
        )

    def TVIPS_HEADER_V1():
        # TVIPS TemData structure from EMMENU Help file
        return [
            ('Version', 'i4'),
            ('CommentV1', 'a80'),
            ('HighTension', 'i4'),
            ('SphericalAberration', 'i4'),
            ('IlluminationAperture', 'i4'),
            ('Magnification', 'i4'),
            ('PostMagnification', 'i4'),
            ('FocalLength', 'i4'),
            ('Defocus', 'i4'),
            ('Astigmatism', 'i4'),
            ('AstigmatismDirection', 'i4'),
            ('BiprismVoltage', 'i4'),
            ('SpecimenTiltAngle', 'i4'),
            ('SpecimenTiltDirection', 'i4'),
            ('IlluminationTiltDirection', 'i4'),
            ('IlluminationTiltAngle', 'i4'),
            ('ImageMode', 'i4'),
            ('EnergySpread', 'i4'),
            ('ChromaticAberration', 'i4'),
            ('ShutterType', 'i4'),
            ('DefocusSpread', 'i4'),
            ('CcdNumber', 'i4'),
            ('CcdSize', 'i4'),
            ('OffsetXV1', 'i4'),
            ('OffsetYV1', 'i4'),
            ('PhysicalPixelSize', 'i4'),
            ('Binning', 'i4'),
            ('ReadoutSpeed', 'i4'),
            ('GainV1', 'i4'),
            ('SensitivityV1', 'i4'),
            ('ExposureTimeV1', 'i4'),
            ('FlatCorrected', 'i4'),
            ('DeadPxCorrected', 'i4'),
            ('ImageMean', 'i4'),
            ('ImageStd', 'i4'),
            ('DisplacementX', 'i4'),
            ('DisplacementY', 'i4'),
            ('DateV1', 'i4'),
            ('TimeV1', 'i4'),
            ('ImageMin', 'i4'),
            ('ImageMax', 'i4'),
            ('ImageStatisticsQuality', 'i4'),
        ]

    def TVIPS_HEADER_V2():
        return [
            ('ImageName', 'V160'),  # utf16
            ('ImageFolder', 'V160'),
            ('ImageSizeX', 'i4'),
            ('ImageSizeY', 'i4'),
            ('ImageSizeZ', 'i4'),
            ('ImageSizeE', 'i4'),
            ('ImageDataType', 'i4'),
            ('Date', 'i4'),
            ('Time', 'i4'),
            ('Comment', 'V1024'),
            ('ImageHistory', 'V1024'),
            ('Scaling', '16f4'),
            ('ImageStatistics', '16c16'),
            ('ImageType', 'i4'),
            ('ImageDisplaType', 'i4'),
            ('PixelSizeX', 'f4'),  # distance between two px in x, [nm]
            ('PixelSizeY', 'f4'),  # distance between two px in y, [nm]
            ('ImageDistanceZ', 'f4'),
            ('ImageDistanceE', 'f4'),
            ('ImageMisc', '32f4'),
            ('TemType', 'V160'),
            ('TemHighTension', 'f4'),
            ('TemAberrations', '32f4'),
            ('TemEnergy', '32f4'),
            ('TemMode', 'i4'),
            ('TemMagnification', 'f4'),
            ('TemMagnificationCorrection', 'f4'),
            ('PostMagnification', 'f4'),
            ('TemStageType', 'i4'),
            ('TemStagePosition', '5f4'),  # x, y, z, a, b
            ('TemImageShift', '2f4'),
            ('TemBeamShift', '2f4'),
            ('TemBeamTilt', '2f4'),
            ('TilingParameters', '7f4'),  # 0: tiling? 1:x 2:y 3: max x
                                          # 4: max y 5: overlap x 6: overlap y
            ('TemIllumination', '3f4'),  # 0: spotsize 1: intensity
            ('TemShutter', 'i4'),
            ('TemMisc', '32f4'),
            ('CameraType', 'V160'),
            ('PhysicalPixelSizeX', 'f4'),
            ('PhysicalPixelSizeY', 'f4'),
            ('OffsetX', 'i4'),
            ('OffsetY', 'i4'),
            ('BinningX', 'i4'),
            ('BinningY', 'i4'),
            ('ExposureTime', 'f4'),
            ('Gain', 'f4'),
            ('ReadoutRate', 'f4'),
            ('FlatfieldDescription', 'V160'),
            ('Sensitivity', 'f4'),
            ('Dose', 'f4'),
            ('CamMisc', '32f4'),
            ('FeiMicroscopeInformation', 'V1024'),
            ('FeiSpecimenInformation', 'V1024'),
            ('Magic', 'u4'),
        ]

    def MM_HEADER():
        # Olympus FluoView MM_Header
        MM_DIMENSION = [
            ('Name', 'a16'),
            ('Size', 'i4'),
            ('Origin', 'f8'),
            ('Resolution', 'f8'),
            ('Unit', 'a64'),
        ]
        return [
            ('HeaderFlag', 'i2'),
            ('ImageType', 'u1'),
            ('ImageName', 'a257'),
            ('OffsetData', 'u4'),
            ('PaletteSize', 'i4'),
            ('OffsetPalette0', 'u4'),
            ('OffsetPalette1', 'u4'),
            ('CommentSize', 'i4'),
            ('OffsetComment', 'u4'),
            ('Dimensions', MM_DIMENSION, 10),
            ('OffsetPosition', 'u4'),
            ('MapType', 'i2'),
            ('MapMin', 'f8'),
            ('MapMax', 'f8'),
            ('MinValue', 'f8'),
            ('MaxValue', 'f8'),
            ('OffsetMap', 'u4'),
            ('Gamma', 'f8'),
            ('Offset', 'f8'),
            ('GrayChannel', MM_DIMENSION),
            ('OffsetThumbnail', 'u4'),
            ('VoiceField', 'i4'),
            ('OffsetVoiceField', 'u4'),
        ]

    def MM_DIMENSIONS():
        # map FluoView MM_Header.Dimensions to axes characters
        return {
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'T': 'T',
            'CH': 'C',
            'WAVELENGTH': 'C',
            'TIME': 'T',
            'XY': 'R',
            'EVENT': 'V',
            'EXPOSURE': 'L',
        }

    def UIC_TAGS():
        # map Universal Imaging Corporation MetaMorph internal tag ids to
        # name and type
        from fractions import Fraction  # delayed import
        return [
            ('AutoScale', int),
            ('MinScale', int),
            ('MaxScale', int),
            ('SpatialCalibration', int),
            ('XCalibration', Fraction),
            ('YCalibration', Fraction),
            ('CalibrationUnits', str),
            ('Name', str),
            ('ThreshState', int),
            ('ThreshStateRed', int),
            ('tagid_10', None),  # undefined
            ('ThreshStateGreen', int),
            ('ThreshStateBlue', int),
            ('ThreshStateLo', int),
            ('ThreshStateHi', int),
            ('Zoom', int),
            ('CreateTime', julian_datetime),
            ('LastSavedTime', julian_datetime),
            ('currentBuffer', int),
            ('grayFit', None),
            ('grayPointCount', None),
            ('grayX', Fraction),
            ('grayY', Fraction),
            ('grayMin', Fraction),
            ('grayMax', Fraction),
            ('grayUnitName', str),
            ('StandardLUT', int),
            ('wavelength', int),
            ('StagePosition', '(%i,2,2)u4'),  # N xy positions as fract
            ('CameraChipOffset', '(%i,2,2)u4'),  # N xy offsets as fract
            ('OverlayMask', None),
            ('OverlayCompress', None),
            ('Overlay', None),
            ('SpecialOverlayMask', None),
            ('SpecialOverlayCompress', None),
            ('SpecialOverlay', None),
            ('ImageProperty', read_uic_image_property),
            ('StageLabel', '%ip'),  # N str
            ('AutoScaleLoInfo', Fraction),
            ('AutoScaleHiInfo', Fraction),
            ('AbsoluteZ', '(%i,2)u4'),  # N fractions
            ('AbsoluteZValid', '(%i,)u4'),  # N long
            ('Gamma', 'I'),  # 'I' uses offset
            ('GammaRed', 'I'),
            ('GammaGreen', 'I'),
            ('GammaBlue', 'I'),
            ('CameraBin', '2I'),
            ('NewLUT', int),
            ('ImagePropertyEx', None),
            ('PlaneProperty', int),
            ('UserLutTable', '(256,3)u1'),
            ('RedAutoScaleInfo', int),
            ('RedAutoScaleLoInfo', Fraction),
            ('RedAutoScaleHiInfo', Fraction),
            ('RedMinScaleInfo', int),
            ('RedMaxScaleInfo', int),
            ('GreenAutoScaleInfo', int),
            ('GreenAutoScaleLoInfo', Fraction),
            ('GreenAutoScaleHiInfo', Fraction),
            ('GreenMinScaleInfo', int),
            ('GreenMaxScaleInfo', int),
            ('BlueAutoScaleInfo', int),
            ('BlueAutoScaleLoInfo', Fraction),
            ('BlueAutoScaleHiInfo', Fraction),
            ('BlueMinScaleInfo', int),
            ('BlueMaxScaleInfo', int),
            # ('OverlayPlaneColor', read_uic_overlay_plane_color),
        ]

    def PILATUS_HEADER():
        # PILATUS CBF Header Specification, Version 1.4
        # map key to [value_indices], type
        return {
            'Detector': ([slice(1, None)], str),
            'Pixel_size': ([1, 4], float),
            'Silicon': ([3], float),
            'Exposure_time': ([1], float),
            'Exposure_period': ([1], float),
            'Tau': ([1], float),
            'Count_cutoff': ([1], int),
            'Threshold_setting': ([1], float),
            'Gain_setting': ([1, 2], str),
            'N_excluded_pixels': ([1], int),
            'Excluded_pixels': ([1], str),
            'Flat_field': ([1], str),
            'Trim_file': ([1], str),
            'Image_path': ([1], str),
            # optional
            'Wavelength': ([1], float),
            'Energy_range': ([1, 2], float),
            'Detector_distance': ([1], float),
            'Detector_Voffset': ([1], float),
            'Beam_xy': ([1, 2], float),
            'Flux': ([1], str),
            'Filter_transmission': ([1], float),
            'Start_angle': ([1], float),
            'Angle_increment': ([1], float),
            'Detector_2theta': ([1], float),
            'Polarization': ([1], float),
            'Alpha': ([1], float),
            'Kappa': ([1], float),
            'Phi': ([1], float),
            'Phi_increment': ([1], float),
            'Chi': ([1], float),
            'Chi_increment': ([1], float),
            'Oscillation_axis': ([slice(1, None)], str),
            'N_oscillations': ([1], int),
            'Start_position': ([1], float),
            'Position_increment': ([1], float),
            'Shutter_time': ([1], float),
            'Omega': ([1], float),
            'Omega_increment': ([1], float),
        }

    def ALLOCATIONGRANULARITY():
        # alignment for writing contiguous data to TIFF
        import mmap  # delayed import
        return mmap.ALLOCATIONGRANULARITY

    def MAXWORKERS():
        # half of CPU cores
        import multiprocessing  # delayed import
        return max(multiprocessing.cpu_count() // 2, 1)


def read_tags(fh, byteorder, offsetsize, tagnames, customtags=None,
              maxifds=None):
    """Read tags from chain of IFDs and return as list of dicts.

    The file handle position must be at a valid IFD header.

    """
    if offsetsize == 4:
        offsetformat = byteorder + 'I'
        tagnosize = 2
        tagnoformat = byteorder + 'H'
        tagsize = 12
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'I4s'
    elif offsetsize == 8:
        offsetformat = byteorder + 'Q'
        tagnosize = 8
        tagnoformat = byteorder + 'Q'
        tagsize = 20
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'Q8s'
    else:
        raise ValueError('invalid offset size')

    if customtags is None:
        customtags = {}
    if maxifds is None:
        maxifds = 2**32

    result = []
    unpack = struct.unpack
    offset = fh.tell()
    while len(result) < maxifds:
        # loop over IFDs
        try:
            tagno = unpack(tagnoformat, fh.read(tagnosize))[0]
            if tagno > 4096:
                raise TiffFileError('suspicious number of tags')
        except Exception:
            log_warning(f'read_tags: corrupted tag list at offset {offset}')
            break

        tags = {}
        data = fh.read(tagsize * tagno)
        pos = fh.tell()
        index = 0
        for _ in range(tagno):
            code, type_ = unpack(tagformat1, data[index:index + 4])
            count, value = unpack(tagformat2, data[index + 4: index + tagsize])
            index += tagsize
            name = tagnames.get(code, str(code))
            try:
                dtype = TIFF.DATA_FORMATS[type_]
            except KeyError:
                raise TiffFileError(f'unknown tag data type {type_}')

            fmt = '{}{}{}'.format(byteorder, count * int(dtype[0]), dtype[1])
            size = struct.calcsize(fmt)
            if size > offsetsize or code in customtags:
                offset = unpack(offsetformat, value)[0]
                if offset < 8 or offset > fh.size - size:
                    raise TiffFileError(f'invalid tag value offset {offset}')
                fh.seek(offset)
                if code in customtags:
                    readfunc = customtags[code][1]
                    value = readfunc(fh, byteorder, dtype, count, offsetsize)
                elif type_ == 7 or (count > 1 and dtype[-1] == 'B'):
                    value = read_bytes(fh, byteorder, dtype, count, offsetsize)
                elif code in tagnames or dtype[-1] == 's':
                    value = unpack(fmt, fh.read(size))
                else:
                    value = read_numpy(fh, byteorder, dtype, count, offsetsize)
            elif dtype[-1] == 'B' or type_ == 7:
                value = value[:size]
            else:
                value = unpack(fmt, value[:size])

            if code not in customtags and code not in TIFF.TAG_TUPLE:
                if len(value) == 1:
                    value = value[0]
            if type_ != 7 and dtype[-1] == 's' and isinstance(value, bytes):
                # TIFF ASCII fields can contain multiple strings,
                #   each terminated with a NUL
                try:
                    value = bytes2str(stripascii(value).strip())
                except UnicodeDecodeError:
                    log_warning(
                        'read_tags: coercing invalid ASCII to bytes '
                        f'(tag {code})'
                    )
            tags[name] = value

        result.append(tags)
        # read offset to next page
        fh.seek(pos)
        offset = unpack(offsetformat, fh.read(offsetsize))[0]
        if offset == 0:
            break
        if offset >= fh.size:
            log_warning(f'read_tags: invalid page offset ({offset})')
            break
        fh.seek(offset)

    if result and maxifds == 1:
        result = result[0]
    return result


def read_exif_ifd(fh, byteorder, dtype, count, offsetsize):
    """Read EXIF tags from file and return as dict."""
    exif = read_tags(fh, byteorder, offsetsize, TIFF.EXIF_TAGS, maxifds=1)
    for name in ('ExifVersion', 'FlashpixVersion'):
        try:
            exif[name] = bytes2str(exif[name])
        except Exception:
            pass
    if 'UserComment' in exif:
        idcode = exif['UserComment'][:8]
        try:
            if idcode == b'ASCII\x00\x00\x00':
                exif['UserComment'] = bytes2str(exif['UserComment'][8:])
            elif idcode == b'UNICODE\x00':
                exif['UserComment'] = exif['UserComment'][8:].decode('utf-16')
        except Exception:
            pass
    return exif


def read_gps_ifd(fh, byteorder, dtype, count, offsetsize):
    """Read GPS tags from file and return as dict."""
    return read_tags(fh, byteorder, offsetsize, TIFF.GPS_TAGS, maxifds=1)


def read_interoperability_ifd(fh, byteorder, dtype, count, offsetsize):
    """Read Interoperability tags from file and return as dict."""
    tag_names = {1: 'InteroperabilityIndex'}
    return read_tags(fh, byteorder, offsetsize, tag_names, maxifds=1)


def read_bytes(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as bytes."""
    dtype = 'B' if dtype[-1] == 's' else byteorder + dtype[-1]
    count *= numpy.dtype(dtype).itemsize
    data = fh.read(count)
    if len(data) != count:
        log_warning(
            f'read_bytes: failed to read all bytes ({len(data)} < {count})'
        )
    return data


def read_utf8(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as unicode string."""
    return fh.read(count).decode()


def read_numpy(fh, byteorder, dtype, count, offsetsize):
    """Read tag data from file and return as numpy array."""
    dtype = 'b' if dtype[-1] == 's' else byteorder + dtype[-1]
    return fh.read_array(dtype, count)


def read_colormap(fh, byteorder, dtype, count, offsetsize):
    """Read ColorMap data from file and return as numpy array."""
    cmap = fh.read_array(byteorder + dtype[-1], count)
    cmap.shape = (3, -1)
    return cmap


def read_json(fh, byteorder, dtype, count, offsetsize):
    """Read JSON tag data from file and return as object."""
    data = fh.read(count)
    try:
        return json.loads(stripnull(data).decode())
    except ValueError:
        log_warning('read_json: invalid JSON')


def read_mm_header(fh, byteorder, dtype, count, offsetsize):
    """Read FluoView mm_header tag from file and return as dict."""
    mmh = fh.read_record(TIFF.MM_HEADER, byteorder=byteorder)
    mmh = recarray2dict(mmh)
    mmh['Dimensions'] = [
        (bytes2str(d[0]).strip(), d[1], d[2], d[3], bytes2str(d[4]).strip())
        for d in mmh['Dimensions']]
    d = mmh['GrayChannel']
    mmh['GrayChannel'] = (
        bytes2str(d[0]).strip(), d[1], d[2], d[3], bytes2str(d[4]).strip())
    return mmh


def read_mm_stamp(fh, byteorder, dtype, count, offsetsize):
    """Read FluoView mm_stamp tag from file and return as numpy.ndarray."""
    return fh.read_array(byteorder + 'f8', 8)


def read_uic1tag(fh, byteorder, dtype, count, offsetsize, planecount=None):
    """Read MetaMorph STK UIC1Tag from file and return as dict.

    Return empty dictionary if planecount is unknown.

    """
    if dtype not in ('2I', '1I') or byteorder != '<':
        raise ValueError('invalid UIC1Tag')
    result = {}
    if dtype == '2I':
        # pre MetaMorph 2.5 (not tested)
        values = fh.read_array('<u4', 2 * count).reshape(count, 2)
        result = {'ZDistance': values[:, 0] / values[:, 1]}
    elif planecount:
        for _ in range(count):
            tagid = struct.unpack('<I', fh.read(4))[0]
            if tagid in (28, 29, 37, 40, 41):
                # silently skip unexpected tags
                fh.read(4)
                continue
            name, value = read_uic_tag(fh, tagid, planecount, offset=True)
            result[name] = value
    return result


def read_uic2tag(fh, byteorder, dtype, planecount, offsetsize):
    """Read MetaMorph STK UIC2Tag from file and return as dict."""
    if dtype != '2I' or byteorder != '<':
        raise ValueError('invalid UIC2Tag')
    values = fh.read_array('<u4', 6 * planecount).reshape(planecount, 6)
    return {
        'ZDistance': values[:, 0] / values[:, 1],
        'DateCreated': values[:, 2],  # julian days
        'TimeCreated': values[:, 3],  # milliseconds
        'DateModified': values[:, 4],  # julian days
        'TimeModified': values[:, 5],  # milliseconds
    }


def read_uic3tag(fh, byteorder, dtype, planecount, offsetsize):
    """Read MetaMorph STK UIC3Tag from file and return as dict."""
    if dtype != '2I' or byteorder != '<':
        raise ValueError('invalid UIC3Tag')
    values = fh.read_array('<u4', 2 * planecount).reshape(planecount, 2)
    return {'Wavelengths': values[:, 0] / values[:, 1]}


def read_uic4tag(fh, byteorder, dtype, planecount, offsetsize):
    """Read MetaMorph STK UIC4Tag from file and return as dict."""
    if dtype != '1I' or byteorder != '<':
        raise ValueError('invalid UIC4Tag')
    result = {}
    while True:
        tagid = struct.unpack('<H', fh.read(2))[0]
        if tagid == 0:
            break
        name, value = read_uic_tag(fh, tagid, planecount, offset=False)
        result[name] = value
    return result


def read_uic_tag(fh, tagid, planecount, offset):
    """Read a single UIC tag value from file and return tag name and value.

    UIC1Tags use an offset.

    """

    def read_int(count=1):
        value = struct.unpack(f'<{count}I', fh.read(4 * count))
        return value[0] if count == 1 else value

    try:
        name, dtype = TIFF.UIC_TAGS[tagid]
    except IndexError:
        # unknown tag
        return f'_TagId{tagid}', read_int()

    Fraction = TIFF.UIC_TAGS[4][1]

    if offset:
        pos = fh.tell()
        if dtype not in (int, None):
            off = read_int()
            if off < 8:
                if dtype is str:
                    return name, ''
                log_warning(
                    f'read_uic_tag: invalid offset for tag {name!r} @{off}'
                )
                return name, off
            fh.seek(off)

    if dtype is None:
        # skip
        name = '_' + name
        value = read_int()
    elif dtype is int:
        # int
        value = read_int()
    elif dtype is Fraction:
        # fraction
        value = read_int(2)
        value = value[0] / value[1]
    elif dtype is julian_datetime:
        # datetime
        value = julian_datetime(*read_int(2))
    elif dtype is read_uic_image_property:
        # ImagePropertyEx
        value = read_uic_image_property(fh)
    elif dtype is str:
        # pascal string
        size = read_int()
        if 0 <= size < 2**10:
            value = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
            value = bytes2str(stripnull(value))
        elif offset:
            value = ''
            log_warning(f'read_uic_tag: corrupt string in tag {name!r}')
        else:
            raise ValueError(f'read_uic_tag: invalid string size {size}')
    elif dtype == '%ip':
        # sequence of pascal strings
        value = []
        for _ in range(planecount):
            size = read_int()
            if 0 <= size < 2**10:
                string = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
                string = bytes2str(stripnull(string))
                value.append(string)
            elif offset:
                log_warning(f'read_uic_tag: corrupt string in tag {name!r}')
            else:
                raise ValueError(f'read_uic_tag: invalid string size: {size}')
    else:
        # struct or numpy type
        dtype = '<' + dtype
        if '%i' in dtype:
            dtype = dtype % planecount
        if '(' in dtype:
            # numpy type
            value = fh.read_array(dtype, 1)[0]
            if value.shape[-1] == 2:
                # assume fractions
                value = value[..., 0] / value[..., 1]
        else:
            # struct format
            value = struct.unpack(dtype, fh.read(struct.calcsize(dtype)))
            if len(value) == 1:
                value = value[0]

    if offset:
        fh.seek(pos + 4)

    return name, value


def read_uic_image_property(fh):
    """Read UIC ImagePropertyEx tag from file and return as dict."""
    # TODO: test this
    size = struct.unpack('B', fh.read(1))[0]
    name = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
    flags, prop = struct.unpack('<IB', fh.read(5))
    if prop == 1:
        value = struct.unpack('II', fh.read(8))
        value = value[0] / value[1]
    else:
        size = struct.unpack('B', fh.read(1))[0]
        value = struct.unpack(f'{size}s', fh.read(size))[0]
    return dict(name=name, flags=flags, value=value)


def read_cz_lsminfo(fh, byteorder, dtype, count, offsetsize):
    """Read CZ_LSMINFO tag from file and return as dict."""
    if byteorder != '<':
        raise ValueError('invalid CZ_LSMINFO structure')
    magic_number, structure_size = struct.unpack('<II', fh.read(8))
    if magic_number not in (50350412, 67127628):
        raise ValueError('invalid CZ_LSMINFO structure')
    fh.seek(-8, 1)

    if structure_size < numpy.dtype(TIFF.CZ_LSMINFO).itemsize:
        # adjust structure according to structure_size
        lsminfo = []
        size = 0
        for name, dtype in TIFF.CZ_LSMINFO:
            size += numpy.dtype(dtype).itemsize
            if size > structure_size:
                break
            lsminfo.append((name, dtype))
    else:
        lsminfo = TIFF.CZ_LSMINFO

    lsminfo = fh.read_record(lsminfo, byteorder=byteorder)
    lsminfo = recarray2dict(lsminfo)

    # read LSM info subrecords at offsets
    for name, reader in TIFF.CZ_LSMINFO_READERS.items():
        if reader is None:
            continue
        offset = lsminfo.get('Offset' + name, 0)
        if offset < 8:
            continue
        fh.seek(offset)
        try:
            lsminfo[name] = reader(fh)
        except ValueError:
            pass
    return lsminfo


def read_lsm_floatpairs(fh):
    """Read LSM sequence of float pairs from file and return as list."""
    size = struct.unpack('<i', fh.read(4))[0]
    return fh.read_array('<2f8', count=size)


def read_lsm_positions(fh):
    """Read LSM positions from file and return as list."""
    size = struct.unpack('<I', fh.read(4))[0]
    return fh.read_array('<2f8', count=size)


def read_lsm_timestamps(fh):
    """Read LSM time stamps from file and return as list."""
    size, count = struct.unpack('<ii', fh.read(8))
    if size != (8 + 8 * count):
        log_warning('read_lsm_timestamps: invalid LSM TimeStamps block')
        return []
    # return struct.unpack(f'<{count}d', fh.read(8 * count))
    return fh.read_array('<f8', count=count)


def read_lsm_eventlist(fh):
    """Read LSM events from file and return as list of (time, type, text)."""
    count = struct.unpack('<II', fh.read(8))[1]
    events = []
    while count > 0:
        esize, etime, etype = struct.unpack('<IdI', fh.read(16))
        etext = bytes2str(stripnull(fh.read(esize - 16)))
        events.append((etime, etype, etext))
        count -= 1
    return events


def read_lsm_channelcolors(fh):
    """Read LSM ChannelColors structure from file and return as dict."""
    result = {'Mono': False, 'Colors': [], 'ColorNames': []}
    pos = fh.tell()
    (size, ncolors, nnames,
     coffset, noffset, mono) = struct.unpack('<IIIIII', fh.read(24))
    if ncolors != nnames:
        log_warning(
            'read_lsm_channelcolors: invalid LSM ChannelColors structure')
        return result
    result['Mono'] = bool(mono)
    # Colors
    fh.seek(pos + coffset)
    colors = fh.read_array('uint8', count=ncolors * 4).reshape((ncolors, 4))
    result['Colors'] = colors.tolist()
    # ColorNames
    fh.seek(pos + noffset)
    buffer = fh.read(size - noffset)
    names = []
    while len(buffer) > 4:
        size = struct.unpack('<I', buffer[:4])[0]
        names.append(bytes2str(buffer[4:3 + size]))
        buffer = buffer[4 + size:]
    result['ColorNames'] = names
    return result


def read_lsm_scaninfo(fh):
    """Read LSM ScanInfo structure from file and return as dict."""
    block = {}
    blocks = [block]
    unpack = struct.unpack
    if struct.unpack('<I', fh.read(4))[0] != 0x10000000:
        # not a Recording sub block
        log_warning('read_lsm_scaninfo: invalid LSM ScanInfo structure')
        return block
    fh.read(8)
    while True:
        entry, dtype, size = unpack('<III', fh.read(12))
        if dtype == 2:
            # ascii
            value = bytes2str(stripnull(fh.read(size)))
        elif dtype == 4:
            # long
            value = unpack('<i', fh.read(4))[0]
        elif dtype == 5:
            # rational
            value = unpack('<d', fh.read(8))[0]
        else:
            value = 0
        if entry in TIFF.CZ_LSMINFO_SCANINFO_ARRAYS:
            blocks.append(block)
            name = TIFF.CZ_LSMINFO_SCANINFO_ARRAYS[entry]
            newobj = []
            block[name] = newobj
            block = newobj
        elif entry in TIFF.CZ_LSMINFO_SCANINFO_STRUCTS:
            blocks.append(block)
            newobj = {}
            block.append(newobj)
            block = newobj
        elif entry in TIFF.CZ_LSMINFO_SCANINFO_ATTRIBUTES:
            name = TIFF.CZ_LSMINFO_SCANINFO_ATTRIBUTES[entry]
            block[name] = value
        elif entry == 0xFFFFFFFF:
            # end sub block
            block = blocks.pop()
        else:
            # unknown entry
            block[f'Entry0x{entry:x}'] = value
        if not blocks:
            break
    return block


def read_sis(fh, byteorder, dtype, count, offsetsize):
    """Read OlympusSIS structure and return as dict.

    No specification is avaliable. Only few fields are known.

    """
    result = {}

    (magic, _, minute, hour, day, month, year, _, name, tagcount
     ) = struct.unpack('<4s6shhhhh6s32sh', fh.read(60))

    if magic != b'SIS0':
        raise ValueError('invalid OlympusSIS structure')

    result['name'] = bytes2str(stripnull(name))
    try:
        result['datetime'] = datetime.datetime(1900 + year, month + 1, day,
                                               hour, minute)
    except ValueError:
        pass

    data = fh.read(8 * tagcount)
    for i in range(0, tagcount * 8, 8):
        tagtype, count, offset = struct.unpack('<hhI', data[i: i + 8])
        fh.seek(offset)
        if tagtype == 1:
            # general data
            (_, lenexp, xcal, ycal, _, mag, _, camname, pictype,
             ) = struct.unpack('<10shdd8sd2s34s32s', fh.read(112))  # 220
            m = math.pow(10, lenexp)
            result['pixelsizex'] = xcal * m
            result['pixelsizey'] = ycal * m
            result['magnification'] = mag
            result['cameraname'] = bytes2str(stripnull(camname))
            result['picturetype'] = bytes2str(stripnull(pictype))
        elif tagtype == 10:
            # channel data
            continue
            # TODO: does not seem to work?
            # (length, _, exptime, emv, _, camname, _, mictype,
            #  ) = struct.unpack('<h22sId4s32s48s32s', fh.read(152))  # 720
            # result['exposuretime'] = exptime
            # result['emvoltage'] = emv
            # result['cameraname2'] = bytes2str(stripnull(camname))
            # result['microscopename'] = bytes2str(stripnull(mictype))

    return result


def read_sis_ini(fh, byteorder, dtype, count, offsetsize):
    """Read OlympusSIS INI string and return as dict."""
    inistr = fh.read(count)
    inistr = bytes2str(stripnull(inistr))
    try:
        return olympusini_metadata(inistr)
    except Exception as exc:
        log_warning(f'olympusini_metadata: {exc.__class__.__name__}: {exc}')
        return {}


def read_tvips_header(fh, byteorder, dtype, count, offsetsize):
    """Read TVIPS EM-MENU headers and return as dict."""
    result = {}
    header = fh.read_record(TIFF.TVIPS_HEADER_V1, byteorder=byteorder)
    for name, typestr in TIFF.TVIPS_HEADER_V1:
        result[name] = header[name].tolist()
    if header['Version'] == 2:
        header = fh.read_record(TIFF.TVIPS_HEADER_V2, byteorder=byteorder)
        if header['Magic'] != int(0xAAAAAAAA):
            log_warning('read_tvips_header: invalid TVIPS v2 magic number')
            return {}
        # decode utf16 strings
        for name, typestr in TIFF.TVIPS_HEADER_V2:
            if typestr.startswith('V'):
                s = header[name].tostring().decode('utf-16', errors='ignore')
                result[name] = stripnull(s, null='\0')
            else:
                result[name] = header[name].tolist()
        # convert nm to m
        for axis in 'XY':
            header['PhysicalPixelSize' + axis] /= 1e9
            header['PixelSize' + axis] /= 1e9
    elif header.version != 1:
        log_warning('read_tvips_header: unknown TVIPS header version')
        return {}
    return result


def read_fei_metadata(fh, byteorder, dtype, count, offsetsize):
    """Read FEI SFEG/HELIOS headers and return as dict."""
    result = {}
    section = {}
    data = bytes2str(stripnull(fh.read(count)))
    for line in data.splitlines():
        line = line.strip()
        if line.startswith('['):
            section = {}
            result[line[1:-1]] = section
            continue
        try:
            key, value = line.split('=')
        except ValueError:
            continue
        section[key] = astype(value)
    return result


def read_cz_sem(fh, byteorder, dtype, count, offsetsize):
    """Read Zeiss SEM tag and return as dict.

    See https://sourceforge.net/p/gwyddion/mailman/message/29275000/ for
    unnamed values.

    """
    result = {'': ()}
    key = None
    data = bytes2str(stripnull(fh.read(count)))
    for line in data.splitlines():
        if line.isupper():
            key = line.lower()
        elif key:
            try:
                name, value = line.split('=')
            except ValueError:
                try:
                    name, value = line.split(':', 1)
                except Exception:
                    continue
            value = value.strip()
            unit = ''
            try:
                v, u = value.split()
                number = astype(v, (int, float))
                if number != v:
                    value = number
                    unit = u
            except Exception:
                number = astype(value, (int, float))
                if number != value:
                    value = number
                if value in ('No', 'Off'):
                    value = False
                elif value in ('Yes', 'On'):
                    value = True
            result[key] = (name.strip(), value)
            if unit:
                result[key] += (unit,)
            key = None
        else:
            result[''] += (astype(line, (int, float)),)
    return result


def read_nih_image_header(fh, byteorder, dtype, count, offsetsize):
    """Read NIH_IMAGE_HEADER tag from file and return as dict."""
    a = fh.read_record(TIFF.NIH_IMAGE_HEADER, byteorder=byteorder)
    a = a.newbyteorder(byteorder)
    a = recarray2dict(a)
    a['XUnit'] = a['XUnit'][:a['XUnitSize']]
    a['UM'] = a['UM'][:a['UMsize']]
    return a


def read_scanimage_metadata(fh):
    """Read ScanImage BigTIFF v3 static and ROI metadata from open file.

    Return non-varying frame data as dict and ROI group data as JSON.

    The settings can be used to read image data and metadata without parsing
    the TIFF file.

    Raise ValueError if file does not contain valid ScanImage v3 metadata.

    """
    fh.seek(0)
    try:
        byteorder, version = struct.unpack('<2sH', fh.read(4))
        if byteorder != b'II' or version != 43:
            raise Exception
        fh.seek(16)
        magic, version, size0, size1 = struct.unpack('<IIII', fh.read(16))
        if magic != 117637889 or version != 3:
            raise Exception
    except Exception:
        raise ValueError('not a ScanImage BigTIFF v3 file')

    frame_data = matlabstr2py(bytes2str(fh.read(size0)[:-1]))
    roi_data = read_json(fh, '<', None, size1, None) if size1 > 1 else {}
    return frame_data, roi_data


def read_micromanager_metadata(fh):
    """Read MicroManager non-TIFF settings from open file and return as dict.

    The settings can be used to read image data without parsing the TIFF file.

    Raise ValueError if the file does not contain valid MicroManager metadata.

    """
    fh.seek(0)
    try:
        byteorder = {b'II': '<', b'MM': '>'}[fh.read(2)]
    except IndexError:
        raise ValueError('not a MicroManager TIFF file')

    result = {}
    fh.seek(8)
    (
        index_header,
        index_offset,
        display_header,
        display_offset,
        comments_header,
        comments_offset,
        summary_header,
        summary_length
    ) = struct.unpack(byteorder + 'IIIIIIII', fh.read(32))

    if summary_header != 2355492:
        raise ValueError('invalid MicroManager summary header')
    result['Summary'] = read_json(fh, byteorder, None, summary_length, None)

    if index_header != 54773648:
        raise ValueError('invalid MicroManager index header')
    fh.seek(index_offset)
    header, count = struct.unpack(byteorder + 'II', fh.read(8))
    if header != 3453623:
        raise ValueError('invalid MicroManager index header')
    data = struct.unpack(byteorder + 'IIIII' * count, fh.read(20 * count))
    result['IndexMap'] = {
        'Channel': data[::5],
        'Slice': data[1::5],
        'Frame': data[2::5],
        'Position': data[3::5],
        'Offset': data[4::5],
    }

    if display_header != 483765892:
        raise ValueError('invalid MicroManager display header')
    fh.seek(display_offset)
    header, count = struct.unpack(byteorder + 'II', fh.read(8))
    if header != 347834724:
        raise ValueError('invalid MicroManager display header')
    result['DisplaySettings'] = read_json(fh, byteorder, None, count, None)

    if comments_header != 99384722:
        raise ValueError('invalid MicroManager comments header')
    fh.seek(comments_offset)
    header, count = struct.unpack(byteorder + 'II', fh.read(8))
    if header != 84720485:
        raise ValueError('invalid MicroManager comments header')
    result['Comments'] = read_json(fh, byteorder, None, count, None)

    return result


def read_metaseries_catalog(fh):
    """Read MetaSeries non-TIFF hint catalog from file.

    Raise ValueError if the file does not contain a valid hint catalog.

    """
    # TODO: implement read_metaseries_catalog
    raise NotImplementedError()


def imagej_metadata_tag(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to TiffWriter.save() as extratags.

    The metadata dict may contain the following keys and values:

        Info : str
            Human-readable information as string.
        Labels : sequence of str
            Human-readable labels for each channel.
        Ranges : sequence of doubles
            Lower and upper values for each channel.
        LUTs : sequence of (3, 256) uint8 ndarrays
            Color palettes for each channel.
        Plot : bytes
            Undocumented ImageJ internal format.
        ROI: bytes
            Undocumented ImageJ internal region of interest format.
        Overlays : bytes
            Undocumented ImageJ internal format.

    """
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def _string(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def _doubles(data, byteorder):
        return struct.pack(byteorder + ('d' * len(data)), *data)

    def _ndarray(data, byteorder):
        return data.tobytes()

    def _bytes(data, byteorder):
        return data

    metadata_types = (
        ('Info', b'info', _string),
        ('Labels', b'labl', _string),
        ('Ranges', b'rang', _doubles),
        ('LUTs', b'luts', _ndarray),
        ('Plot', b'plot', _bytes),
        ('ROI', b'roi ', _bytes),
        ('Overlays', b'over', _bytes),
    )

    for key, mtype, func in metadata_types:
        if key.lower() in metadata:
            key = key.lower()
        elif key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if isinstance(values, list):
            count = len(values)
        else:
            values = [values]
            count = 1
        header.append(mtype + struct.pack(byteorder + 'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    if not body:
        return ()
    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder + ('I' * len(bytecounts)), *bytecounts)
    return (
        (50839, 'B', len(data), data, True),
        (50838, 'I', len(bytecounts) // 4, bytecounts, True)
    )


def imagej_metadata(data, bytecounts, byteorder):
    """Return IJMetadata tag value as dict.

    The 'Info' string can have multiple formats, e.g. OIF or ScanImage,
    that might be parsed into dicts using the matlabstr2py or
    oiffile.SettingsFile functions.
    'ROI' and 'Overlays' are returned as bytes, which can be parsed with the
    ImagejRoi.frombytes() function of the roifile package.

    """

    def _string(data, byteorder):
        return data.decode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def _doubles(data, byteorder):
        return struct.unpack(byteorder + ('d' * (len(data) // 8)), data)

    def _lut(data, byteorder):
      raise TiffParserError('_lut is not supported by TiffParser')
        # return numpy.frombuffer(data, 'uint8').reshape(-1, 256)

    def _bytes(data, byteorder):
        return data

    # big-endian
    metadata_types = {
        b'info': ('Info', _string),
        b'labl': ('Labels', _string),
        b'rang': ('Ranges', _doubles),
        b'luts': ('LUTs', _lut),
        b'plot': ('Plot', _bytes),
        b'roi ': ('ROI', _bytes),
        b'over': ('Overlays', _bytes),
    }
    # little-endian
    metadata_types.update({k[::-1]: v for k, v in metadata_types.items()})

    if not bytecounts:
        raise ValueError('no ImageJ metadata')

    if not data[:4] in (b'IJIJ', b'JIJI'):
        raise ValueError('invalid ImageJ metadata')

    header_size = bytecounts[0]
    if header_size < 12 or header_size > 804:
        raise ValueError('invalid ImageJ metadata header size')

    ntypes = (header_size - 4) // 8
    header = struct.unpack(byteorder + '4sI' * ntypes, data[4: 4 + ntypes * 8])
    pos = 4 + ntypes * 8
    counter = 0
    result = {}
    for mtype, count in zip(header[::2], header[1::2]):
        values = []
        name, func = metadata_types.get(mtype, (bytes2str(mtype), read_bytes))
        for _ in range(count):
            counter += 1
            pos1 = pos + bytecounts[counter]
            values.append(func(data[pos:pos1], byteorder))
            pos = pos1
        result[name.strip()] = values[0] if count == 1 else values
    return result


def imagej_description_metadata(description):
    """Return metatata from ImageJ image description as dict.

    Raise ValueError if not a valid ImageJ description.

    >>> description = 'ImageJ=1.11a\\nimages=510\\nhyperstack=true\\n'
    >>> imagej_description_metadata(description)  # doctest: +SKIP
    {'ImageJ': '1.11a', 'images': 510, 'hyperstack': True}

    """

    def _bool(val):
        return {'true': True, 'false': False}[val.lower()]

    result = {}
    for line in description.splitlines():
        try:
            key, val = line.split('=')
        except Exception:
            continue
        key = key.strip()
        val = val.strip()
        for dtype in (int, float, _bool):
            try:
                val = dtype(val)
                break
            except Exception:
                pass
        result[key] = val

    if 'ImageJ' not in result:
        raise ValueError('not a ImageJ image description')
    return result


def imagej_description(shape, rgb=None, colormaped=False, version=None,
                       hyperstack=None, mode=None, loop=None, **kwargs):
    """Return ImageJ image description from data shape.

    ImageJ can handle up to 6 dimensions in order TZCYXS.

    >>> imagej_description((51, 5, 2, 196, 171))  # doctest: +SKIP
    ImageJ=1.11a
    images=510
    channels=2
    slices=5
    frames=51
    hyperstack=true
    mode=grayscale
    loop=false

    """
    if colormaped:
        raise NotImplementedError('ImageJ colormapping not supported')
    if version is None:
        version = '1.11a'
    shape = imagej_shape(shape, rgb=rgb)
    rgb = shape[-1] in (3, 4)

    result = [f'ImageJ={version}']
    append = []
    result.append(f'images={product(shape[:-3])}')
    if hyperstack is None:
        hyperstack = True
        append.append('hyperstack=true')
    else:
        append.append(f'hyperstack={bool(hyperstack)}')
    if shape[2] > 1:
        result.append(f'channels={shape[2]}')
    if mode is None and not rgb:
        mode = 'grayscale'
    if hyperstack and mode:
        append.append(f'mode={mode}')
    if shape[1] > 1:
        result.append(f'slices={shape[1]}')
    if shape[0] > 1:
        result.append(f'frames={shape[0]}')
        if loop is None:
            append.append('loop=false')
    if loop is not None:
        append.append(f'loop={bool(loop)}'.lower())
    for key, value in kwargs.items():
        append.append(f'{key.lower()}={value}')

    return '\n'.join(result + append + [''])


def imagej_shape(shape, rgb=None):
    """Return shape normalized to 6D ImageJ hyperstack TZCYXS.

    Raise ValueError if not a valid ImageJ hyperstack shape.

    >>> imagej_shape((2, 3, 4, 5, 3), False)
    (2, 3, 4, 5, 3, 1)

    """
    shape = tuple(int(i) for i in shape)
    ndim = len(shape)
    if 1 > ndim > 6:
        raise ValueError('invalid ImageJ hyperstack: not 2 to 6 dimensional')
    if rgb is None:
        rgb = shape[-1] in (3, 4) and ndim > 2
    if rgb and shape[-1] not in (3, 4):
        raise ValueError('invalid ImageJ hyperstack: not a RGB image')
    if not rgb and ndim == 6 and shape[-1] != 1:
        raise ValueError('invalid ImageJ hyperstack: not a non-RGB image')
    if rgb or shape[-1] == 1:
        return (1, ) * (6 - ndim) + shape
    return (1, ) * (5 - ndim) + shape + (1,)


def json_description(shape, **metadata):
    """Return JSON image description from data shape and other metadata.

    Return UTF-8 encoded JSON.

    >>> json_description((256, 256, 3), axes='YXS')  # doctest: +SKIP
    b'{"shape": [256, 256, 3], "axes": "YXS"}'

    """
    metadata.update(shape=shape)
    return json.dumps(metadata)  # .encode()


def json_description_metadata(description):
    """Return metatata from JSON formated image description as dict.

    Raise ValuError if description is of unknown format.

    >>> description = '{"shape": [256, 256, 3], "axes": "YXS"}'
    >>> json_description_metadata(description)  # doctest: +SKIP
    {'shape': [256, 256, 3], 'axes': 'YXS'}
    >>> json_description_metadata('shape=(256, 256, 3)')
    {'shape': (256, 256, 3)}

    """
    if description[:6] == 'shape=':
        # old-style 'shaped' description; not JSON
        shape = tuple(int(i) for i in description[7:-1].split(','))
        return dict(shape=shape)
    if description[:1] == '{' and description[-1:] == '}':
        # JSON description
        return json.loads(description)
    raise ValueError('invalid JSON image description', description)


def fluoview_description_metadata(description, ignoresections=None):
    """Return metatata from FluoView image description as dict.

    The FluoView image description format is unspecified. Expect failures.

    >>> descr = ('[Intensity Mapping]\\nMap Ch0: Range=00000 to 02047\\n'
    ...          '[Intensity Mapping End]')
    >>> fluoview_description_metadata(descr)
    {'Intensity Mapping': {'Map Ch0: Range': '00000 to 02047'}}

    """
    if not description.startswith('['):
        raise ValueError('invalid FluoView image description')
    if ignoresections is None:
        ignoresections = {'Region Info (Fields)', 'Protocol Description'}

    result = {}
    sections = [result]
    comment = False
    for line in description.splitlines():
        if not comment:
            line = line.strip()
        if not line:
            continue
        if line[0] == '[':
            if line[-5:] == ' End]':
                # close section
                del sections[-1]
                section = sections[-1]
                name = line[1:-5]
                if comment:
                    section[name] = '\n'.join(section[name])
                if name[:4] == 'LUT ':
                    a = numpy.array(section[name], dtype='uint8')
                    a.shape = -1, 3
                    section[name] = a
                continue
            # new section
            comment = False
            name = line[1:-1]
            if name[:4] == 'LUT ':
                section = []
            elif name in ignoresections:
                section = []
                comment = True
            else:
                section = {}
            sections.append(section)
            result[name] = section
            continue
        # add entry
        if comment:
            section.append(line)
            continue
        line = line.split('=', 1)
        if len(line) == 1:
            section[line[0].strip()] = None
            continue
        key, value = line
        if key[:4] == 'RGB ':
            section.extend(int(rgb) for rgb in value.split())
        else:
            section[key.strip()] = astype(value.strip())
    return result


def pilatus_description_metadata(description):
    """Return metatata from Pilatus image description as dict.

    Return metadata from Pilatus pixel array detectors by Dectris, created
    by camserver or TVX software.

    >>> pilatus_description_metadata('# Pixel_size 172e-6 m x 172e-6 m')
    {'Pixel_size': (0.000172, 0.000172)}

    """
    result = {}
    if not description.startswith('# '):
        return result
    for c in '#:=,()':
        description = description.replace(c, ' ')
    for line in description.split('\n'):
        if line[:2] != '  ':
            continue
        line = line.split()
        name = line[0]
        if line[0] not in TIFF.PILATUS_HEADER:
            try:
                result['DateTime'] = datetime.datetime.strptime(
                    ' '.join(line), '%Y-%m-%dT%H %M %S.%f')
            except Exception:
                result[name] = ' '.join(line[1:])
            continue
        indices, dtype = TIFF.PILATUS_HEADER[line[0]]
        if isinstance(indices[0], slice):
            # assumes one slice
            values = line[indices[0]]
        else:
            values = [line[i] for i in indices]
        if dtype is float and values[0] == 'not':
            values = ['NaN']
        values = tuple(dtype(v) for v in values)
        if dtype == str:
            values = ' '.join(values)
        elif len(values) == 1:
            values = values[0]
        result[name] = values
    return result


def svs_description_metadata(description):
    """Return metatata from Aperio image description as dict.

    The Aperio image description format is unspecified. Expect failures.

    >>> svs_description_metadata('Aperio Image Library v1.0')
    {'Aperio Image Library': 'v1.0'}

    """
    if not description.startswith('Aperio Image Library '):
        raise ValueError('invalid Aperio image description')
    result = {}
    lines = description.split('\n')
    key, value = lines[0].strip().rsplit(None, 1)  # 'Aperio Image Library'
    result[key.strip()] = value.strip()
    if len(lines) == 1:
        return result
    items = lines[1].split('|')
    result[''] = items[0].strip()  # TODO: parse this?
    for item in items[1:]:
        key, value = item.split(' = ')
        result[key.strip()] = astype(value.strip())
    return result


def stk_description_metadata(description):
    """Return metadata from MetaMorph image description as list of dict.

    The MetaMorph image description format is unspecified. Expect failures.

    """
    description = description.strip()
    if not description:
        return []
    try:
        description = bytes2str(description)
    except UnicodeDecodeError as exc:
        log_warning(
            f'stk_description_metadata: {exc.__class__.__name__}: {exc}'
        )
        return []
    result = []
    for plane in description.split('\x00'):
        d = {}
        for line in plane.split('\r\n'):
            line = line.split(':', 1)
            if len(line) > 1:
                name, value = line
                d[name.strip()] = astype(value.strip())
            else:
                value = line[0].strip()
                if value:
                    if '' in d:
                        d[''].append(value)
                    else:
                        d[''] = [value]
        result.append(d)
    return result


def metaseries_description_metadata(description):
    """Return metatata from MetaSeries image description as dict."""
    if not description.startswith('<MetaData>'):
        raise ValueError('invalid MetaSeries image description')

    from xml.etree import cElementTree as etree  # delayed import

    root = etree.fromstring(description)
    types = {
        'float': float,
        'int': int,
        'bool': lambda x: asbool(x, 'on', 'off'),
    }

    def parse(root, result):
        # recursive
        for child in root:
            attrib = child.attrib
            if not attrib:
                result[child.tag] = parse(child, {})
                continue
            if 'id' in attrib:
                i = attrib['id']
                t = attrib['type']
                v = attrib['value']
                if t in types:
                    result[i] = types[t](v)
                else:
                    result[i] = v
        return result

    adict = parse(root, {})
    if 'Description' in adict:
        adict['Description'] = adict['Description'].replace('&#13;&#10;', '\n')
    return adict


def scanimage_description_metadata(description):
    """Return metatata from ScanImage image description as dict."""
    return matlabstr2py(description)


def scanimage_artist_metadata(artist):
    """Return metatata from ScanImage artist tag as dict."""
    try:
        return json.loads(artist)
    except ValueError as exc:
        log_warning(
            f'scanimage_artist_metadata: {exc.__class__.__name__}: {exc}'
        )


def olympusini_metadata(inistr):
    """Return OlympusSIS metadata from INI string.

    No documentation is available.

    """

    def keyindex(key):
        # split key into name and index
        index = 0
        i = len(key.rstrip('0123456789'))
        if i < len(key):
            index = int(key[i:]) - 1
            key = key[:i]
        return key, index

    result = {}
    bands = []
    zpos = None
    tpos = None
    for line in inistr.splitlines():
        line = line.strip()
        if line == '' or line[0] == ';':
            continue
        if line[0] == '[' and line[-1] == ']':
            section_name = line[1:-1]
            result[section_name] = section = {}
            if section_name == 'Dimension':
                result['axes'] = axes = []
                result['shape'] = shape = []
            elif section_name == 'ASD':
                result[section_name] = []
            elif section_name == 'Z':
                if 'Dimension' in result:
                    result[section_name]['ZPos'] = zpos = []
            elif section_name == 'Time':
                if 'Dimension' in result:
                    result[section_name]['TimePos'] = tpos = []
            elif section_name == 'Band':
                nbands = result['Dimension']['Band']
                bands = [{'LUT': []} for i in range(nbands)]
                result[section_name] = bands
                iband = 0
        else:
            key, value = line.split('=')
            if value.strip() == '':
                value = None
            elif ',' in value:
                value = tuple(astype(v) for v in value.split(','))
            else:
                value = astype(value)

            if section_name == 'Dimension':
                section[key] = value
                axes.append(key)
                shape.append(value)
            elif section_name == 'ASD':
                if key == 'Count':
                    result['ASD'] = [{}] * value
                else:
                    key, index = keyindex(key)
                    result['ASD'][index][key] = value
            elif section_name == 'Band':
                if key[:3] == 'LUT':
                    lut = bands[iband]['LUT']
                    value = struct.pack('<I', value)
                    lut.append(
                        [ord(value[0:1]), ord(value[1:2]), ord(value[2:3])])
                else:
                    key, iband = keyindex(key)
                    bands[iband][key] = value
            elif key[:4] == 'ZPos' and zpos is not None:
                zpos.append(value)
            elif key[:7] == 'TimePos' and tpos is not None:
                tpos.append(value)
            else:
                section[key] = value

    if 'axes' in result:
        sisaxes = {'Band': 'C'}
        axes = []
        shape = []
        for i, x in zip(result['shape'], result['axes']):
            if i > 1:
                axes.append(sisaxes.get(x, x[0].upper()))
                shape.append(i)
        result['axes'] = ''.join(axes)
        result['shape'] = tuple(shape)
    try:
        result['Z']['ZPos'] = numpy.array(
            result['Z']['ZPos'][:result['Dimension']['Z']], 'float64')
    except Exception:
        pass
    try:
        result['Time']['TimePos'] = numpy.array(
            result['Time']['TimePos'][:result['Dimension']['Time']], 'int32')
    except Exception:
        pass
    for band in bands:
        band['LUT'] = numpy.array(band['LUT'], 'uint8')
    return result


def unpack_rgb(data, dtype=None, bitspersample=None, rescale=True):
  raise TiffParserError('unpack_rgb is not supported by TiffParser')


def apply_colormap(image, colormap, contig=True):
  raise TiffParserError('apply_colormap is not supported by TiffParser')


def parse_filenames(files, pattern, axesorder=None):
  raise TiffParserError('parse_filenames is not supported by TiffParser')

def reorient(image, orientation):
  raise TiffParserError('reorient is not supported by TiffParser')


def repeat_nd(a, repeats):
  raise TiffParserError('repeat_nd is not supported by TiffParser')



def reshape_nd(data_or_shape, ndim):
  raise TiffParserError('reshape_nd is not supported by TiffParser')
  


def squeeze_axes(shape, axes, skip=None):
  raise TiffParserError('squeeze_axes is not supported by TiffParser')
  


def transpose_axes(image, axes, asaxes=None):
  raise TiffParserError('transpose_axes is not supported by TiffParser')
  


def reshape_axes(axes, shape, newshape, unknown=None):
  raise TiffParserError('reshape_axes is not supported by TiffParser')
  


def stack_pages(pages, out=None, maxworkers=None, **kwargs):
  raise TiffParserError('stack_pages is not supported by TiffParser')
  

def create_output(out, shape, dtype, mode='w+', suffix=None):
  raise TiffParserError('create_output is not supported by TiffParser')


def matlabstr2py(string):
  raise TiffParserError('matlabstr2py is not supported by TiffParser')


def stripnull(string, null=b'\x00'):
    """Return string truncated at first null character.

    Clean NULL terminated C strings. For unicode strings use null='\\0'.

    >>> stripnull(b'string\\x00')
    b'string'
    >>> stripnull('string\\x00', null='\\0')
    'string'

    """
    i = string.find(null)
    return string if (i < 0) else string[:i]


def stripascii(string):
    """Return string truncated at last byte that is 7-bit ASCII.

    Clean NULL separated and terminated TIFF strings.

    >>> stripascii(b'string\\x00string\\n\\x01\\x00')
    b'string\\x00string\\n'
    >>> stripascii(b'\\x00')
    b''

    """
    # TODO: pythonize this
    i = len(string)
    while i:
        i -= 1
        if 8 < string[i] < 127:
            break
    else:
        i = -1
    return string[: i + 1]


def asbool(value, true=(b'true', 'true'), false=(b'false', 'false')):
    """Return string as bool if possible, else raise TypeError.

    >>> asbool(b' False ')
    False

    """
    value = value.strip().lower()
    if value in true:  # might raise UnicodeWarning/BytesWarning
        return True
    if value in false:
        return False
    raise TypeError()


def astype(value, types=None):
    """Return argument as one of types if possible.

    >>> astype('42')
    42
    >>> astype('3.14')
    3.14
    >>> astype('True')
    True
    >>> astype(b'Neee-Wom')
    'Neee-Wom'

    """
    if types is None:
        types = int, float, asbool, bytes2str
    for typ in types:
        try:
            return typ(value)
        except (ValueError, AttributeError, TypeError, UnicodeEncodeError):
            pass
    return value


def format_size(size, threshold=1536):
    """Return file size as string from byte size.

    >>> format_size(1234)
    '1234 B'
    >>> format_size(12345678901)
    '11.50 GiB'

    """
    if size < threshold:
        return f'{size} B'
    for unit in ('KiB', 'MiB', 'GiB', 'TiB', 'PiB'):
        size /= 1024.0
        if size < threshold:
            return f'{size:.2f} {unit}'
    return 'ginormous'


def identityfunc(arg, *args, **kwargs):
    """Single argument identity function.

    >>> identityfunc('arg')
    'arg'

    """
    return arg


def nullfunc(*args, **kwargs):
    """Null function.

    >>> nullfunc('arg', kwarg='kwarg')

    """
    return


def sequence(value):
    """Return tuple containing value if value is not a tuple or list.

    >>> sequence(1)
    (1,)
    >>> sequence([1])
    [1]
    >>> sequence('ab')
    ('ab',)

    """
    return value if isinstance(value, (tuple, list)) else (value,)


def product(iterable):
    """Return product of sequence of numbers.

    Equivalent of functools.reduce(operator.mul, iterable, 1).
    Multiplying numpy integers might overflow.

    >>> product([2**8, 2**30])
    274877906944
    >>> product([])
    1

    """
    prod = 1
    for i in iterable:
        prod *= i
    return prod


def natural_sorted(iterable):
    """Return human sorted list of strings.

    E.g. for sorting file names.

    >>> natural_sorted(['f1', 'f2', 'f10'])
    ['f1', 'f2', 'f10']

    """

    def sortkey(x):
        return [(int(c) if c.isdigit() else c) for c in re.split(numbers, x)]

    numbers = re.compile(r'(\d+)')
    return sorted(iterable, key=sortkey)


def excel_datetime(timestamp, epoch=None):
    """Return datetime object from timestamp in Excel serial format.

    Convert LSM time stamps.

    >>> excel_datetime(40237.029999999795)
    datetime.datetime(2010, 2, 28, 0, 43, 11, 999982)

    """
    if epoch is None:
        epoch = datetime.datetime.fromordinal(693594)
    return epoch + datetime.timedelta(timestamp)


def julian_datetime(julianday, milisecond=0):
    """Return datetime from days since 1/1/4713 BC and ms since midnight.

    Convert Julian dates according to MetaMorph.

    >>> julian_datetime(2451576, 54362783)
    datetime.datetime(2000, 2, 2, 15, 6, 2, 783)

    """
    if julianday <= 1721423:
        # no datetime before year 1
        return None

    a = julianday + 1
    if a > 2299160:
        alpha = math.trunc((a - 1867216.25) / 36524.25)
        a += 1 + alpha - alpha // 4
    b = a + (1524 if a > 1721423 else 1158)
    c = math.trunc((b - 122.1) / 365.25)
    d = math.trunc(365.25 * c)
    e = math.trunc((b - d) / 30.6001)

    day = b - d - math.trunc(30.6001 * e)
    month = e - (1 if e < 13.5 else 13)
    year = c - (4716 if month > 2.5 else 4715)

    hour, milisecond = divmod(milisecond, 1000 * 60 * 60)
    minute, milisecond = divmod(milisecond, 1000 * 60)
    second, milisecond = divmod(milisecond, 1000)

    return datetime.datetime(year, month, day,
                             hour, minute, second, milisecond)


def byteorder_isnative(byteorder):
    """Return if byteorder matches the system's byteorder.

    >>> byteorder_isnative('=')
    True

    """
    if byteorder in ('=', sys.byteorder):
        return True
    keys = {'big': '>', 'little': '<'}
    return keys.get(byteorder, byteorder) == keys[sys.byteorder]


def recarray2dict(recarray):
    """Return numpy.recarray as dict."""
    # TODO: subarrays
    result = {}
    for descr, value in zip(recarray.dtype.descr, recarray):
        name, dtype = descr[:2]
        if dtype[1] == 'S':
            value = bytes2str(stripnull(value))
        elif value.ndim < 2:
            value = value.tolist()
        result[name] = value
    return result


def xml2dict(xml, sanitize=True, prefix=None):
    """Return XML as dict.

    >>> xml2dict('<?xml version="1.0" ?><root attr="name"><key>1</key></root>')
    {'root': {'key': 1, 'attr': 'name'}}
    >>> xml2dict('<level1><level2>3.5322</level2></level1>')
    {'level1': {'level2': 3.5322}}

    """
    from xml.etree import cElementTree as etree  # delayed import

    at = tx = ''
    if prefix:
        at, tx = prefix

    def astype(value):
        # return string value as int, float, bool, or unchanged
        if not isinstance(value, (str, bytes)):
            return value
        for t in (int, float, asbool):
            try:
                return t(value)
            except Exception:
                pass
        return value

    def etree2dict(t):
        # adapted from https://stackoverflow.com/a/10077069/453463
        key = t.tag
        if sanitize:
            key = key.rsplit('}', 1)[-1]
        d = {key: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = collections.defaultdict(list)
            for dc in map(etree2dict, children):
                for k, v in dc.items():
                    dd[k].append(astype(v))
            d = {key: {k: astype(v[0]) if len(v) == 1 else astype(v)
                       for k, v in dd.items()}}
        if t.attrib:
            d[key].update((at + k, astype(v)) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[key][tx + 'value'] = astype(text)
            else:
                d[key] = astype(text)
        return d

    return etree2dict(etree.fromstring(xml))


def hexdump(bytestr, width=75, height=24, snipat=-2, modulo=2, ellipsis=None):
    """Return hexdump representation of bytes.

    >>> hexdump(binascii.unhexlify('49492a00080000000e00fe0004000100'))
    '49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 II*.............'

    """
    size = len(bytestr)
    if size < 1 or width < 2 or height < 1:
        return ''
    if height == 1:
        addr = b''
        bytesperline = min(modulo * (((width - len(addr)) // 4) // modulo),
                           size)
        if bytesperline < 1:
            return ''
        nlines = 1
    else:
        addr = b'%%0%ix: ' % len(b'%x' % size)
        bytesperline = min(modulo * (((width - len(addr % 1)) // 4) // modulo),
                           size)
        if bytesperline < 1:
            return ''
        width = 3 * bytesperline + len(addr % 1)
        nlines = (size - 1) // bytesperline + 1

    if snipat is None or snipat == 1:
        snipat = height
    elif 0 < abs(snipat) < 1:
        snipat = int(math.floor(height * snipat))
    if snipat < 0:
        snipat += height

    if height == 1 or nlines == 1:
        blocks = [(0, bytestr[:bytesperline])]
        addr = b''
        height = 1
        width = 3 * bytesperline
    elif height is None or nlines <= height:
        blocks = [(0, bytestr)]
    elif snipat <= 0:
        start = bytesperline * (nlines - height)
        blocks = [(start, bytestr[start:])]  # (start, None)
    elif snipat >= height or height < 3:
        end = bytesperline * height
        blocks = [(0, bytestr[:end])]  # (end, None)
    else:
        end1 = bytesperline * snipat
        end2 = bytesperline * (height - snipat - 1)
        blocks = [
            (0, bytestr[:end1]),
            (size - end1 - end2, None),
            (size - end2, bytestr[size - end2:]),
        ]

    ellipsis = b'...' if ellipsis is None else ellipsis.encode('cp1252')
    result = []
    for start, bytestr in blocks:
        if bytestr is None:
            result.append(ellipsis)  # 'skip %i bytes' % start)
            continue
        hexstr = binascii.hexlify(bytestr)
        strstr = re.sub(br'[^\x20-\x7f]', b'.', bytestr)
        for i in range(0, len(bytestr), bytesperline):
            h = hexstr[2 * i: 2 * i + bytesperline * 2]
            r = (addr % (i + start)) if height > 1 else addr
            r += b' '.join(h[i: i + 2] for i in range(0, 2 * bytesperline, 2))
            r += b' ' * (width - len(r))
            r += strstr[i: i + bytesperline]
            result.append(r)
    result = b'\n'.join(result)
    result = result.decode('ascii')
    return result


def isprintable(string):
    """Return if all characters in string are printable.

    >>> isprintable('abc')
    True
    >>> isprintable(b'\01')
    False

    """
    string = string.strip()
    if not string:
        return True
    try:
        return string.isprintable()
    except Exception:
        pass
    try:
        return string.decode().isprintable()
    except Exception:
        pass


def clean_whitespace(string, compact=False):
    """Return string with compressed whitespace."""
    for a, b in (
        ('\r\n', '\n'),
        ('\r', '\n'),
        ('\n\n', '\n'),
        ('\t', ' '),
        ('  ', ' ')
    ):
        string = string.replace(a, b)
    if compact:
        for a, b in (
            ('\n', ' '),
            ('[ ', '['),
            ('  ', ' '),
            ('  ', ' '),
            ('  ', ' ')
        ):
            string = string.replace(a, b)
    return string.strip()


def pformat_xml(xml):
    """Return pretty formatted XML."""
    try:
        from lxml import etree  # delayed import

        if not isinstance(xml, bytes):
            xml = xml.encode()
        xml = etree.parse(io.BytesIO(xml))
        xml = etree.tostring(xml, pretty_print=True, xml_declaration=True,
                             encoding=xml.docinfo.encoding)
        xml = bytes2str(xml)
    except Exception:
        if isinstance(xml, bytes):
            xml = bytes2str(xml)
        xml = xml.replace('><', '>\n<')
    return xml.replace('  ', ' ').replace('\t', ' ')


def pformat(arg, width=79, height=24, compact=True):
  raise TiffParserError('pformat is not supported by TiffParser')
  """Return pretty formatted representation of object as string.

  # Whitespace might be altered.

  # """
  # if height is None or height < 1:
  #     height = 1024
  # if width is None or width < 1:
  #     width = 256

  # npopt = numpy.get_printoptions()
  # numpy.set_printoptions(threshold=100, linewidth=width)

  # if isinstance(arg, (str, bytes)):
  #     if arg[:5].lower() in ('<?xml', b'<?xml'):
  #         if isinstance(arg, bytes):
  #             arg = bytes2str(arg)
  #         if height == 1:
  #             arg = arg[: 4 * width]
  #         else:
  #             arg = pformat_xml(arg)
  #     elif isinstance(arg, bytes):
  #         if isprintable(arg):
  #             arg = bytes2str(arg)
  #             arg = clean_whitespace(arg)
  #         else:
  #             numpy.set_printoptions(**npopt)
  #             return hexdump(arg, width=width, height=height, modulo=1)
  #     arg = arg.rstrip()
  # elif isinstance(arg, numpy.record):
  #     arg = arg.pprint()
  # else:
  #     import pprint  # delayed import

  #     arg = pprint.pformat(arg, width=width, compact=compact)

  # numpy.set_printoptions(**npopt)

  # if height == 1:
  #     arg = clean_whitespace(arg, compact=True)
  #     return arg[:width]

  # argl = list(arg.splitlines())
  # if len(argl) > height:
  #     arg = '\n'.join(argl[:height // 2] + ['...'] + argl[-height // 2:])
  # return arg


def snipstr(string, width=79, snipat=None, ellipsis=None):
    """Return string cut to specified length.

    >>> snipstr('abcdefghijklmnop', 8)
    'abc...op'

    """
    if snipat is None:
        snipat = 0.5
    if ellipsis is None:
        if isinstance(string, bytes):
            ellipsis = b'...'
        else:
            ellipsis = '\u2026'
    esize = len(ellipsis)

    splitlines = string.splitlines()
    # TODO: finish and test multiline snip

    result = []
    for line in splitlines:
        if line is None:
            result.append(ellipsis)
            continue
        linelen = len(line)
        if linelen <= width:
            result.append(string)
            continue

        split = snipat
        if split is None or split == 1:
            split = linelen
        elif 0 < abs(split) < 1:
            split = int(math.floor(linelen * split))
        if split < 0:
            split += linelen
            if split < 0:
                split = 0

        if esize == 0 or width < esize + 1:
            if split <= 0:
                result.append(string[-width:])
            else:
                result.append(string[:width])
        elif split <= 0:
            result.append(ellipsis + string[esize - width:])
        elif split >= linelen or width < esize + 4:
            result.append(string[:width - esize] + ellipsis)
        else:
            splitlen = linelen - width + esize
            end1 = split - splitlen // 2
            end2 = end1 + splitlen
            result.append(string[:end1] + ellipsis + string[end2:])

    if isinstance(string, bytes):
        return b'\n'.join(result)
    return '\n'.join(result)


def enumstr(enum):
    """Return short string representation of Enum instance."""
    name = enum.name
    if name is None:
        name = str(enum)
    return name


def enumarg(enum, arg):
    """Return enum member from its name or value.

    >>> enumarg(TIFF.PHOTOMETRIC, 2)
    <PHOTOMETRIC.RGB: 2>
    >>> enumarg(TIFF.PHOTOMETRIC, 'RGB')
    <PHOTOMETRIC.RGB: 2>

    """
    try:
        return enum(arg)
    except Exception:
        try:
            return enum[arg.upper()]
        except Exception:
            raise ValueError(f'invalid argument {arg}')


def parse_kwargs(kwargs, *keys, **keyvalues):
    """Return dict with keys from keys|keyvals and values from kwargs|keyvals.

    Existing keys are deleted from kwargs.

    >>> kwargs = {'one': 1, 'two': 2, 'four': 4}
    >>> kwargs2 = parse_kwargs(kwargs, 'two', 'three', four=None, five=5)
    >>> kwargs == {'one': 1}
    True
    >>> kwargs2 == {'two': 2, 'four': 4, 'five': 5}
    True

    """
    result = {}
    for key in keys:
        if key in kwargs:
            result[key] = kwargs[key]
            del kwargs[key]
    for key, value in keyvalues.items():
        if key in kwargs:
            result[key] = kwargs[key]
            del kwargs[key]
        else:
            result[key] = value
    return result


def update_kwargs(kwargs, **keyvalues):
    """Update dict with keys and values if keys do not already exist.

    >>> kwargs = {'one': 1, }
    >>> update_kwargs(kwargs, one=None, two=2)
    >>> kwargs == {'one': 1, 'two': 2}
    True

    """
    for key, value in keyvalues.items():
        if key not in kwargs:
            kwargs[key] = value


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging
    logging.getLogger(__name__).warning(msg, *args, **kwargs)


def validate_jhove(filename, jhove=None, ignore=None):
    """Validate TIFF file using jhove -m TIFF-hul.

    Raise ValueError if jhove outputs an error message unless the message
    contains one of the strings in 'ignore'.

    JHOVE does not support bigtiff or more than 50 IFDs.

    See `JHOVE TIFF-hul Module <http://jhove.sourceforge.net/tiff-hul.html>`_

    """
    import subprocess
    if ignore is None:
        ignore = ['More than 50 IFDs']
    if jhove is None:
        jhove = 'jhove'
    out = subprocess.check_output([jhove, filename, '-m', 'TIFF-hul'])
    if b'ErrorMessage: ' in out:
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(b'ErrorMessage: '):
                error = line[14:].decode()
                for i in ignore:
                    if i in error:
                        break
                else:
                    raise ValueError(error)
                break


def lsm2bin(lsmfile, binfile=None, tile=None, verbose=True):
  raise TiffParserError('lsm2bin is not supported by TiffParser')
  


def imshow(data, photometric=None, planarconfig=None, bitspersample=None,
           nodata=0, interpolation=None, cmap=None, vmin=None, vmax=None,
           figure=None, title=None, dpi=96, subplot=None, maxdim=None,
           **kwargs):
  raise TiffParserError('imshow is not supported by TiffParser')


def _app_show():
  raise TiffParserError('_app_show is not supported by TiffParser')
  


def askopenfilename(**kwargs):
  raise TiffParserError('askopenfilename is not supported by TiffParser')




def bytes2str(b, encoding=None, errors='strict'):
    """Return unicode string from encoded bytes."""
    if encoding is not None:
        return b.decode(encoding, errors)
    try:
        return b.decode('utf-8', errors)
    except UnicodeDecodeError:
        return b.decode('cp1252', errors)


def bytestr(s, encoding='cp1252'):
    """Return bytes from unicode string, else pass through."""
    return s.encode(encoding) if isinstance(s, str) else s

