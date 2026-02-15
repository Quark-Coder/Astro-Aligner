# AstroAligner v1.3

<img width="1920" height="1020" alt="python_BOf3SODMUT" src="https://github.com/user-attachments/assets/d1fb5e46-3766-4b2b-abda-138317ead715" />

AstroAligner is a manual alignment tool for astronomical images designed for situations where automatic star detection fails.
It supports RAW, FITS and standard image formats and allows precise 3 point or comet based alignment with visual magnification and fine drag control.

---

## What’s New in v1.3

* Full PyQt6 interface redesign
* Background threaded image loading with smart caching
* Support for RAW formats via rawpy
* FITS metadata handling via astropy
* EXIF observation date extraction
* Magnifier overlay for subpixel precision
* Shift key fine adjustment while dragging points
* Display resolution reduction mode 1x, 2x, 4x, 6x
* Monochrome preview mode
* Persistent per directory reference image
* Automatic alignment point autosave in JSON
* Euclidean and Affine transform modes
* Comet only alignment mode
* Observation date written to FITS header
* Preserved zoom and pan between frames

---

## Supported Formats

* RAW: .nef, .cr2, .cr3, .crw, .arw, .raf, .dng, .orf, .rw2, .pef, .x3f
* FITS: .fit, .fits
* Images: .png, .jpg, .jpeg, .tif, .tiff

---

## Features

### Manual Alignment Modes

* 3 Point mode
  Select 3 non collinear stars on each image.

* Comet Only mode
  Designed for comet sequences. Use coma and tail structure as reference.

### Transform Methods

* Euclidean
  Rotation + translation.

* Affine
  Rotation + translation + scale.

### Precision Tools

* Crosshair markers with indexed labels
* Enlarged center circle for better targeting
* Live magnifier overlay near cursor
* Shift modifier for ultra fine dragging
* Visual alignment status indicators

### Display Controls

* Stretch factor slider
* Black point slider
* Monochrome preview toggle
* Reduce display resolution 1x to 6x for performance

### Performance

* Background threaded loader
* Priority queue based image preloading
* Smart memory cache with automatic trimming
* Raw data cache for fast re rendering

### Metadata Handling

* Reads DATE-OBS, DATE, Observation Date from FITS
* Reads EXIF DateTimeOriginal for RAW and JPEG
* Fallback to filesystem timestamp
* Writes observation date to aligned FITS output

### Persistence

* Alignment points stored per directory in JSON
* Reference frame stored per working directory
* Automatic restore when reopening directory

---

## Installation

### Option 1 - Run with Python

#### Requirements

* Python 3.9 or newer

Install dependencies:

```bash
pip install numpy opencv-python pillow rawpy astropy exifread PyQt6
```

Run:

```bash
python aa_pyqt_v1.3.py
```

---

### Option 2 - Build Standalone EXE

Install PyInstaller:

```bash
pip install pyinstaller
```

Build:

```bash
build.cmd
```

Executable will appear in the dist folder.

---

## Usage

1. Launch the application.
2. Click "Load working directory".
3. Select a folder containing RAW, FITS or image files.
4. Navigate using:

   * Left and Right arrow keys
5. Select alignment type:

   * 3 Point
   * Comet Only
6. Place up to 3 points per image.
7. Use Shift while dragging for fine adjustment.
8. Mark one image as Reference.
9. Choose transform method:

   * Euclidean
   * Affine
10. Click Align.
11. Select output directory.

Aligned files will be saved as FITS with preserved observation metadata.

---

## Controls

Mouse:

* Left click - place or select alignment point
* Left drag - move alignment point
* Middle drag - pan image
* Mouse wheel - zoom

Keyboard:

* Left Arrow - previous image
* Right Arrow - next image
* Shift - fine adjustment while dragging

---

## Output

* 16 bit FITS files
* Observation date written to header
* Transform applied relative to selected reference frame
* Files saved in selected output directory

---

## Internal Files

Alignment data is stored in JSON format per directory.
Reference frame index is remembered per working directory session.

---

AstroAligner v1.3 focuses on manual precision alignment for difficult datasets including comets, low SNR frames and star trailing sequences.

