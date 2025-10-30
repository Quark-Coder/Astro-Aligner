# AstroAligner v1.0
<img width="1537" height="915" alt="python_PXwkMmm1gq" src="https://github.com/user-attachments/assets/29123f7e-026b-4a57-8456-5282f3c933f0" />
<img width="1537" height="904" alt="python_NYe0ii0skj" src="https://github.com/user-attachments/assets/d28328df-3816-4f5f-a5ef-be5debada058" />

A manual alignment tool for astronomical images when automatic star detection methods fail.

https://github.com/user-attachments/assets/7024b787-57e1-40ca-a014-eaa552b8a007

AstroAligner allows you to manually select 3 non-collinear points or comet coma and tail on each image to align a sequence of astrophotos relative to a chosen reference frame. It produces aligned FITS files ready for stacking or further astrophotometric processing.

---

## Features

* Manual 3-point alignment / comet alignment
* Euclidean or similarity transformation modes
* Visual crosshair markers for precise point placement
* Mouse-based panning and zooming
* Automatic image stretch for faint targets
* Saves aligned images as 16-bit FITS with metadata
* Auto-saving and loading of alignment points

---

## Installation

You can run AstroAligner either as a Python script or as a standalone executable.

### Option 1 - Run with Python

#### Requirements

* Python 3.9 or newer
* Install dependencies:

```bash
pip install numpy pillow rawpy astropy opencv-python exifread
```

#### Run

```bash
python AstroAligner_v1.0.py
```
Or just double click.

### Option 2 - Build or Use the Standalone EXE

If you have a precompiled version:

* Download `AstroAligner_v1.0.exe`
* Run it directly (no installation required)

If you want to build it yourself:

```bash
pip install pyinstaller
build.cmd
```

The generated executable will appear in the `dist` folder.

---

## Usage

1. **Launch** the application.
2. Click **"Load DNG Directory"** and select the folder containing your `.dng` files.
3. Navigate between images using **Left/Right arrow keys** or **Previous/Next** buttons.
4. On each image, click to mark **3 non-collinear points** (bright stars work best).

   * Indicators show progress (red/yellow/green).
5. Select one image as **Reference** using the checkbox (*requires 3 valid points*).
6. If you do not see three points on one of the images, the program will skip the image if no points are placed.
7. Choose the transform mode: **Euclidean** (no scale) or **Similarity** (with scale).
8. Click **"Align and Save FITS"** and choose an output directory.
9. Aligned images will be saved as `aligned_00001.fit`, `aligned_00002.fit`, etc. Can be automatically opened in Siril as sequence `aligned.seq`

---

alignment_points.txt - user-defined points for each image. 

transformed_points.txt - points after geometric alignment.






