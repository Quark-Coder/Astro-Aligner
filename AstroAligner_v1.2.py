import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
from PIL import Image, ImageTk
import rawpy
from astropy.io import fits
import cv2
import gc
import math
import threading
import re
import exifread
from datetime import datetime
import sys  # For icon path

def is_raw_file(filename):
    """Check if file is a RAW format."""
    return filename.lower().endswith(('.dng', '.cr2', '.cr3'))

def autostretch_image(img, a=20.0, black_percent=0.0):
    if len(img.shape) == 2:
        ch = img.astype(np.float64) / 255.0
    else:
        ch = np.mean(img.astype(np.float64) / 255.0, axis=2)
    # Apply black clip if >0
    if black_percent > 0.0:
        black_level = np.percentile(ch, black_percent)
        ch = np.clip((ch - black_level) / (1.0 - black_level), 0.0, 1.0)
    stretched_ch = np.arcsinh(a * ch) / np.arcsinh(a)
    gamma = 0.8
    stretched_ch = np.power(np.clip(stretched_ch, 0, 1), gamma)
    stretched = np.clip(stretched_ch * 255, 0, 255).astype(np.uint8)
    stretched_rgb = np.stack([stretched] * 3, axis=2)
    return stretched_rgb

def compute_transform(src_points, dst_points, mode="euclidean"):
    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)
    if len(src) < 3 or len(dst) < 3:
        raise ValueError("It takes at least 3 points to calculate the transformation.")
    if mode == "similarity":
        M, inliers = cv2.estimateAffinePartial2D(src.reshape((-1, 1, 2)), dst.reshape((-1, 1, 2)), method=cv2.LMEDS)
        if M is None:
            raise ValueError("Similarity transformation could not be evaluated. Check the points for collinearity.")
        scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        print(f"Similarity mode: {np.sum(inliers)} inliers, calculated scale={scale:.4f}")
        return M, scale
    elif mode == "euclidean":
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_c = src - src_mean
        dst_c = dst - dst_mean
        H = src_c.T @ dst_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = dst_mean - R @ src_mean
        M = np.zeros((2, 3), dtype=np.float32)
        M[:, :2] = R
        M[:, 2] = t
        scale = 1.0
        return M, scale
    else:
        raise ValueError(f"Unknown conversion mode: {mode}")

def check_collinearity(points):
    if len(points) != 3:
        return False
    p1, p2, p3 = points
    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    return area > 10.0

def comet_to_triplet(ann):
    if not ann or "center" not in ann or "tail_dir" not in ann:
        return None
    cx, cy = ann["center"]
    tx, ty = ann["tail_dir"]
    vec = np.array([tx - cx, ty - cy], dtype=np.float32)
    length = np.linalg.norm(vec)
    if length < 1e-3:
        # No/zero tail: default direction (positive X for consistency)
        unit_vec = np.array([1.0, 0.0], dtype=np.float32)
    else:
        unit_vec = vec / length  # Normalize: pure direction, ignore length
    # Perpendicular unit vector (90° CCW, for stable rotation)
    perp_unit = np.array([-unit_vec[1], unit_vec[0]], dtype=np.float32)
    # Fixed unit scale (1.0): ignores radius and tail length completely
    fixed_length = 1.0  # Arbitrary small value; makes triplet scale-invariant
    tail_vec = unit_vec * fixed_length
    perp_vec = perp_unit * fixed_length
    p1 = np.array([cx, cy], dtype=np.float32)  # Center: translation anchor
    p2 = p1 + tail_vec  # Tail direction: rotation only
    p3 = p1 + perp_vec  # Perp: ensures rigidity, no scale/shear
    return np.array([p1, p2, p3])

def extract_description_dates(raw):
    if not hasattr(raw, 'image_description') or not raw.image_description:
        return {}
    desc = raw.image_description
    created = None
    date_obs = None
    subsec = ''
    subsec_pattern = r'Sub Sec Time Original\s*:\s*(\d+)'
    subsec_match = re.search(subsec_pattern, desc, re.IGNORECASE)
    if subsec_match:
        subsec_digits = subsec_match.group(1).ljust(6, '0')[:6]
        subsec = '.' + subsec_digits
    date_pattern = r'(Date/Time Original|Create Date|Observation Date)\s*:\s*(\d{4}[:-]\d{2}[:-]\d{2}[:\sT]\d{2}:\d{2}:\d{2})'
    matches = re.findall(date_pattern, desc, re.IGNORECASE)
    for label, date_str in matches:
        if ' ' in date_str:
            date_part, time_part = date_str.split(' ', 1)
        elif 'T' in date_str:
            date_part, time_part = date_str.split('T')
        else:
            date_part = date_str[:10]
            time_part = date_str[10:]
        date_part = date_part.replace(':', '-')
        full_date = date_part + 'T' + time_part.strip()
        if subsec and '.'.join(full_date.split('T')[1].split('.')[:1]) != full_date.split('T')[1]:
            full_date += subsec
        full_date = full_date.strip()
        label_lower = label.lower()
        if 'date/time original' in label_lower or 'create' in label_lower:
            created = full_date
        elif 'observation' in label_lower:
            date_obs = full_date
    dates = {}
    if created:
        dates['CREATED'] = created
    if date_obs:
        dates['DATE-OBS'] = date_obs
    elif created:
        dates['DATE-OBS'] = created
    if dates:
        return dates
    return {}

def extract_date_from_exif(file_path):
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, stop_tag='EXIF DateTimeOriginal')
        if 'EXIF DateTimeOriginal' in tags:
            date_time_str = str(tags['EXIF DateTimeOriginal']).strip()
            dt = datetime.strptime(date_time_str, '%Y:%m:%d %H:%M:%S')
            fits_date = dt.strftime('%Y-%m-%dT%H:%M:%S')
            return fits_date
        else:
            print(f"DateTimeOriginal not found in EXIF for {os.path.basename(file_path)}")
            return None
    except Exception as e:
        print(f"EXIF extraction failed for {os.path.basename(file_path)}: {e}")
        return None

def extract_raw_date(raw, file_path):
    # Prioritize LibRaw timestamp (from EXIF DateTimeOriginal)
    try:
        if hasattr(raw.other, 'timestamp') and raw.other.timestamp > 0:
            dt = datetime.fromtimestamp(raw.other.timestamp)
            return dt.strftime('%Y-%m-%dT%H:%M:%S')
    except:
        pass  # Fall back if timestamp not available or invalid
    # Fall back to EXIF
    exif_date = extract_date_from_exif(file_path)
    if exif_date:
        return exif_date
    # Fall back to description parsing (mainly for DNG)
    dates = extract_description_dates(raw)
    if 'DATE-OBS' in dates:
        return dates['DATE-OBS']
    return None

def transform_points(M, points):
    if M is None:
        return points
    points_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(points_np, M)
    return transformed.reshape(-1, 2).tolist()

def is_valid_comet_ann(ann):
    """Validate comet annotation for reference or display."""
    if not ann or "center" not in ann or "tail_dir" not in ann:
        return False
    cx, cy = ann["center"]
    r = ann.get("radius", 0.0)
    tx, ty = ann["tail_dir"]
    vec_len = math.hypot(tx - cx, ty - cy)
    return r > 0 and vec_len > 10.0

class AstroAligner:
    def __init__(self):
        print("AstroAligner v1.8 - Fixed comet loading and reference restore")
        self.root = tk.Tk()
        self.root.title("Image Aligner for RAW/PNG/JPG Files (Stars & Comet)")
        # Icon loading: fix for direct Python vs PyInstaller
        try:
            if getattr(sys, 'frozen', False):
                # PyInstaller bundle: icon in extracted temp dir
                icon_path = os.path.join(sys._MEIPASS, 'icon.ico')
            else:  # Direct run: icon in script directory
                icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')
            self.root.iconbitmap(icon_path)
        except (tk.TclError, FileNotFoundError) as e:
            print(f"Failed to load icon: {e}. Using default Tkinter icon.")
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.raw_files = []
        self.display_pil_images = []
        self.points_list = []
        self.comet_ann = []
        self.current_idx = 0
        self.ref_idx = None
        self.zoom_factor = 1.0
        self.image_item = None
        self.is_panning = False
        self.dragged_point_idx = None
        self.current_scaled_w = 0.0
        self.current_scaled_h = 0.0
        self.current_img_w = 0.0
        self.current_img_h = 0.0
        self.last_shown_idx = -1
        self.initial_fit_done = False
        self.preload_active = False
        self.input_dir = None
        self.raw_cache = None
        self.display_scales = []  # Per-image scales for RAW/JPEG fix
        self.transform_mode = tk.StringVar(value="euclidean")
        self.align_mode = tk.StringVar(value="stars")
        self.offset_x = 0.0
        self.offset_y = 0.0
        # Panning state
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_offset_x = 0.0
        self.pan_start_offset_y = 0.0
        # Stretch settings
        self.stretch_factor = 20.0
        self.black_clip = 0.0
        self.stretch_update_timer = None  # For debouncing full reload
        # Comet-specific
        self._comet_state = "idle"
        self.dragged_comet_part = None
        self.temp_comet_center = None
        self.temp_comet_radius = 0.0
        self.temp_tail_end = None
        # Indicators
        self.samples_subframe = None
        self.indicators = []
        self.setup_ui()
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        max_w = min(1400, int(screen_w * 0.8))
        max_h = min(800, int(screen_h * 0.8))
        win_w = max(1000, max_w)
        win_h = max(520, max_h)
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.focus_set()

    def reset_data(self):
        self.raw_files.clear()
        self.display_pil_images.clear()
        self.points_list.clear()
        self.comet_ann.clear()
        self.current_idx = 0
        self.ref_idx = None
        self.zoom_factor = 1.0
        self.last_shown_idx = -1
        self.initial_fit_done = False
        self.is_panning = False
        self.dragged_point_idx = None
        self.current_scaled_w = 0.0
        self.current_scaled_h = 0.0
        self.current_img_w = 0.0
        self.current_img_h = 0.0
        self.preload_active = False
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_offset_x = 0.0
        self.pan_start_offset_y = 0.0
        self.raw_cache = None
        self.display_scales = []  # Reset scales
        self._comet_state = "idle"
        self.dragged_comet_part = None
        self.temp_comet_center = None
        self.temp_comet_radius = 0.0
        self.temp_tail_end = None
        if hasattr(self, 'canvas'):
            self.canvas.delete("all")
        if hasattr(self, 'ref_label'):
            self.ref_label.config(text="None selected")
        if hasattr(self, 'set_ref_var'):
            self.set_ref_var.set(False)
        # Clear indicators but don't destroy subframe - will recreate on load
        if hasattr(self, 'samples_subframe'):
            for widget in self.samples_subframe.winfo_children():
                widget.destroy()
            self.indicators.clear()
        # Cancel any pending stretch update
        if self.stretch_update_timer is not None:
            self.root.after_cancel(self.stretch_update_timer)
            self.stretch_update_timer = None

    def setup_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_container, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=1)
        self.status_label = ttk.Label(main_container, text="Ready")
        self.status_label.pack(pady=1)
        self.title_label = ttk.Label(main_container, text="", font=("Arial", 12, "bold"))
        self.title_label.pack(pady=1)
        # Fixed control panel (no scrolling)
        self.control_frame = ttk.Frame(main_container, width=260, relief="ridge")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=1)
        self.control_frame.pack_propagate(False)
        # Load all control elements into fixed frame
        load_btn = ttk.Button(self.control_frame, text="Load Directory", command=self.load_directory)
        load_btn.pack(pady=1)
        nav_frame = ttk.Frame(self.control_frame)
        nav_frame.pack(pady=1)
        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=1)
        self.current_label = ttk.Label(nav_frame, text="Image 0/0")
        self.current_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=1)
        # Align Mode (compact)
        ttk.Label(self.control_frame, text="Align Mode:", font=("Arial", 9, "bold")).pack(pady=(2,0))
        self.align_combo = ttk.Combobox(self.control_frame, textvariable=self.align_mode, values=["stars", "comet"], state="readonly", width=10)
        self.align_combo.pack(pady=1)
        self.align_mode.trace("w", self.on_align_mode_change)
        # Transform Mode
        ttk.Label(self.control_frame, text="Transform Mode:", font=("Arial", 9, "bold")).pack(pady=(2,0))
        self.transform_combo = ttk.Combobox(self.control_frame, textvariable=self.transform_mode, values=["euclidean", "similarity"], state="readonly", width=10)
        self.transform_combo.pack(pady=1)
        self.transform_combo.bind("<<ComboboxSelected>>", lambda e: print(f"Transform mode: {self.transform_mode.get()}"))
        # Stretch Settings (more compact)
        ttk.Separator(self.control_frame, orient="horizontal").pack(pady=1, fill=tk.X)
        ttk.Label(self.control_frame, text="Stretch Settings:", font=("Arial", 9, "bold")).pack(pady=(1,0))
        # Stretch Factor
        stretch_frame = ttk.Frame(self.control_frame)
        stretch_frame.pack(pady=0, fill=tk.X)
        ttk.Label(stretch_frame, text="Stretch Factor:").pack(anchor=tk.W)
        self.stretch_var = tk.DoubleVar(value=self.stretch_factor)
        self.stretch_slider = ttk.Scale(stretch_frame, from_=1.0, to=50.0, variable=self.stretch_var, orient=tk.HORIZONTAL, length=120, command=self.on_stretch_change)
        self.stretch_slider.pack(pady=0, fill=tk.X)
        self.stretch_label = ttk.Label(stretch_frame, text=f"{self.stretch_factor:.1f}")
        self.stretch_label.pack(anchor=tk.W)
        # Black Clip %
        black_frame = ttk.Frame(self.control_frame)
        black_frame.pack(pady=0, fill=tk.X)
        ttk.Label(black_frame, text="Black Clip %:").pack(anchor=tk.W)
        self.black_var = tk.DoubleVar(value=self.black_clip)
        self.black_slider = ttk.Scale(black_frame, from_=0.0, to=20.0, variable=self.black_var, orient=tk.HORIZONTAL, length=120, command=self.on_stretch_change)
        self.black_slider.pack(pady=0, fill=tk.X)
        self.black_label = ttk.Label(black_frame, text=f"{self.black_clip:.1f}")
        self.black_label.pack(anchor=tk.W)
        # Samples (dynamic, compact)
        ttk.Label(self.control_frame, text="Samples:", font=("Arial", 9, "bold")).pack(pady=(1,0))
        self.samples_subframe = ttk.Frame(self.control_frame)
        self.samples_subframe.pack(pady=0)
        clear_btn = ttk.Button(self.control_frame, text="Clear Annotation", command=self.clear_current_points)
        clear_btn.pack(pady=1)
        ttk.Separator(self.control_frame, orient="horizontal").pack(pady=1, fill=tk.X)
        ttk.Label(self.control_frame, text="Reference Image:", font=("Arial", 9, "bold")).pack(pady=(1,0))
        self.ref_label = ttk.Label(self.control_frame, text="None selected", wraplength=250)
        self.ref_label.pack(pady=0)
        self.set_ref_var = tk.BooleanVar(value=False)
        self.ref_checkbox = ttk.Checkbutton(self.control_frame, text="Set current as reference (requires 3 points)",
                                            variable=self.set_ref_var, command=self.on_set_ref)
        self.ref_checkbox.pack(pady=0)
        ttk.Separator(self.control_frame, orient="horizontal").pack(pady=1, fill=tk.X)
        align_btn = ttk.Button(self.control_frame, text="Align and Save FITS", command=self.align_and_save)
        align_btn.pack(pady=2)
        # Canvas frame (right side, with improved scrollbar span)
        self.canvas_frame = ttk.Frame(main_container)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        cv_scrollbar_v = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        cv_scrollbar_h = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        # No yscroll/xscroll commands initially - will set dynamically if needed
        cv_scrollbar_v.config(command=self.canvas.yview)
        cv_scrollbar_h.config(command=self.canvas.xview)
        self.canvas.config(yscrollcommand=cv_scrollbar_v.set, xscrollcommand=cv_scrollbar_h.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        cv_scrollbar_v.grid(row=0, column=1, sticky="ns")
        cv_scrollbar_h.grid(row=1, column=0, columnspan=2, sticky="ew")  # Span both columns for full width
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(1, weight=0)
        self.canvas_frame.grid_rowconfigure(1, weight=0)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_point_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_point_release)
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Enter>", lambda e: self.canvas.focus_set())
        # Initial UI for stars mode
        self.update_samples_ui()

    def _update_scrollregion(self):
        """Dynamically update scrollregion to cover canvas + image position (allows free panning)."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return
        img_right = self.offset_x + self.current_scaled_w
        img_bottom = self.offset_y + self.current_scaled_h
        min_x = min(0.0, self.offset_x)
        min_y = min(0.0, self.offset_y)
        max_x = max(float(canvas_w), img_right)
        max_y = max(float(canvas_h), img_bottom)
        self.canvas.config(scrollregion=(min_x, min_y, max_x, max_y))

    def _update_display_position(self, do_center=False):
        """Update image position and scrollregion."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return
        if do_center:
            self.offset_x = max(0.0, (canvas_w - self.current_scaled_w) / 2.0)
            self.offset_y = max(0.0, (canvas_h - self.current_scaled_h) / 2.0)
        if self.image_item:
            self.canvas.coords(self.image_item, self.offset_x, self.offset_y)
        self._update_scrollregion()

    def on_stretch_change(self, *args):
        """Update values and labels immediately (no heavy ops for smooth dragging)."""
        self.stretch_factor = self.stretch_var.get()
        self.black_clip = self.black_var.get()
        self.stretch_label.config(text=f"{self.stretch_factor:.1f}")
        self.black_label.config(text=f"{self.black_clip:.1f}")
        print(f"Stretch dragging: a={self.stretch_factor:.1f}, black_clip={self.black_clip:.1f}%")
        # Cancel pending full update
        if self.stretch_update_timer is not None:
            self.root.after_cancel(self.stretch_update_timer)
            self.stretch_update_timer = None
        # Debounced: Schedule full reload of loaded images after 500ms of no change
        self.stretch_update_timer = self.root.after(500, self.apply_stretch_settings)

    def apply_stretch_settings(self):
        """Reload only already loaded display images with new stretch settings (debounced, selective)."""
        self.stretch_update_timer = None
        if not self.raw_files:
            return
        total = len(self.raw_files)
        # Only update already loaded images (not None) to avoid full preload
        loaded_indices = [i for i in range(total) if self.display_pil_images[i] is not None]
        if not loaded_indices:
            self.update_status("No images loaded yet.")
            return
        self.update_status(f"Updating {len(loaded_indices)} loaded images with new stretch settings...")
        self.progress_var.set(0)
        for j, i in enumerate(loaded_indices):
            self.load_display_image(i, force_reload=True)
            self.progress_var.set((j + 1) / len(loaded_indices) * 100)
            self.root.update_idletasks()
        self.progress_var.set(100)
        # Refresh current image display
        if self.current_idx < total and self.display_pil_images[self.current_idx] is not None:
            self.show_current_image()
        self.update_status(f"Stretch settings applied to {len(loaded_indices)} images. (New params will apply to unloaded on load.)")

    def on_align_mode_change(self, *args):
        mode = self.align_mode.get()
        if mode == "comet":
            self.transform_mode.set("euclidean")  # Force rigid for comet (no scale)
            print("Comet mode: switched to euclidean (translation + rotation only)")
        self.update_samples_ui()  # Dynamic update indicators
        req_text = "complete comet annotation (Coma + Tail)" if mode == "comet" else "3 points"
        self.ref_checkbox.config(text=f"Set current as reference (requires {req_text})")
        # Redraw if images loaded
        if self.raw_files and self.current_idx < len(self.raw_files):
            self.redraw_points()
            self.update_indicators()

    def update_samples_ui(self):
        mode = self.align_mode.get()
        # Clear existing
        for widget in self.samples_subframe.winfo_children():
            widget.destroy()
        self.indicators.clear()
        # Create new based on mode (compact: smaller frames)
        if mode == "stars":
            for i in range(3):
                ind_frame = tk.Frame(self.samples_subframe, width=15, height=15, bg="red")
                ind_frame.pack(pady=0)
                ind_label = ttk.Label(self.samples_subframe, text=f"P{i+1}")
                ind_label.pack()
                self.indicators.append({'frame': ind_frame, 'label': ind_label, 'index': i})
        else:  # comet
            # Coma indicator
            coma_frame = tk.Frame(self.samples_subframe, width=15, height=15, bg="red")
            coma_frame.pack(pady=0)
            coma_label = ttk.Label(self.samples_subframe, text="Coma")
            coma_label.pack()
            self.indicators.append({'frame': coma_frame, 'label': coma_label, 'part': 'coma'})
            # Tail indicator
            tail_frame = tk.Frame(self.samples_subframe, width=15, height=15, bg="red")
            tail_frame.pack(pady=0)
            tail_label = ttk.Label(self.samples_subframe, text="Tail")
            tail_label.pack()
            self.indicators.append({'frame': tail_frame, 'label': tail_label, 'part': 'tail'})
        # Update colors immediately
        self.update_indicators()
        print(f"Updated samples UI for {mode} mode")

    def update_indicators(self):
        mode = self.align_mode.get()
        if self.current_idx >= len(self.raw_files):
            return
        if mode == "stars":
            points = self.points_list[self.current_idx]
            is_valid = len(points) == 3 and check_collinearity(points)
            for i in range(min(3, len(self.indicators))):
                ind = self.indicators[i]
                if len(points) > i:
                    ind['frame'].config(bg="green" if is_valid else "yellow")
                else:
                    ind['frame'].config(bg="red")
        else:  # comet
            ann = self.comet_ann[self.current_idx]
            if ann is None:
                for ind in self.indicators:
                    ind['frame'].config(bg="red")
                return
            cx, cy = ann["center"]
            r = ann.get("radius", 0.0)
            if r <= 0:
                r = 20.0  # Fallback for old files
                ann["radius"] = r
            tx, ty = ann["tail_dir"]
            vec_len = math.hypot(tx - cx, ty - cy)
            # Coma: green if center and radius >0
            coma_ind = next((ind for ind in self.indicators if ind.get('part') == 'coma'), None)
            if coma_ind:
                if r > 0.0:
                    coma_ind['frame'].config(bg="green")
                else:
                    coma_ind['frame'].config(bg="red")
            # Tail: green >10px, yellow 1-10px, red <=1px
            tail_ind = next((ind for ind in self.indicators if ind.get('part') == 'tail'), None)
            if tail_ind:
                if vec_len > 10.0:
                    tail_ind['frame'].config(bg="green")
                elif vec_len > 1.0:
                    tail_ind['frame'].config(bg="yellow")
                else:
                    tail_ind['frame'].config(bg="red")

    def clear_current_points(self):
        mode = self.align_mode.get()
        if mode == "stars":
            self.points_list[self.current_idx] = []
            self.save_points_to_file(self.input_dir)
        else:
            self.comet_ann[self.current_idx] = None
            self.save_comet_annotations(self.input_dir)
        self.redraw_points()
        self.update_indicators()
        print(f"Cleared {mode} annotation for image {self.current_idx}")
        if self.ref_idx == self.current_idx:
            self.set_ref_var.set(False)
            self.ref_label.config(text="None selected")
            self.ref_idx = None

    def on_set_ref(self):
        mode = self.align_mode.get()
        if mode == "stars":
            points = self.points_list[self.current_idx]
            if len(points) != 3 or not check_collinearity(points):
                messagebox.showwarning("Warning", "Current image must have exactly 3 non-collinear points to set as reference.")
                self.set_ref_var.set(False)
                return
        else:  # comet
            ann = self.comet_ann[self.current_idx]
            if ann is None or not is_valid_comet_ann(ann):
                messagebox.showwarning("Warning", "Current image must have valid comet annotation (radius >0, tail >10px) to set as reference.")
                self.set_ref_var.set(False)
                return
        if self.set_ref_var.get():
            self.ref_idx = self.current_idx
            base_name = os.path.basename(self.raw_files[self.current_idx]) if self.raw_files else f"Image {self.current_idx}"
            self.ref_label.config(text=f"Selected: {base_name} (index {self.current_idx})")
            print(f"Set reference to image {self.current_idx} ({mode} mode)")
            if mode == "stars":
                self.save_points_to_file(self.input_dir)
            else:
                self.save_comet_annotations(self.input_dir)
        else:
            self.ref_idx = None
            self.ref_label.config(text="None selected")
            print("Reference deselected")
            if mode == "stars":
                self.save_points_to_file(self.input_dir)
            else:
                self.save_comet_annotations(self.input_dir)

    def schedule_preload(self):
        if self.preload_active:
            return
        if not self.raw_files or self.current_idx >= len(self.raw_files) - 1:
            return
        next_start = self.current_idx + 1
        next_end = min(next_start + 10, len(self.raw_files))
        unloaded = [i for i in range(next_start, next_end) if i < len(self.display_pil_images) and self.display_pil_images[i] is None]
        if unloaded:
            self.preload_active = True
            threading.Thread(target=self._preload_batch, args=(unloaded,), daemon=True).start()

    def _preload_batch(self, indices):
        for idx in indices:
            if idx < len(self.display_pil_images) and self.display_pil_images[idx] is None:
                self.load_display_image(idx)
        self.root.after(0, lambda: setattr(self, 'preload_active', False))

    def load_display_image(self, idx, force_reload=False):
        if idx >= len(self.raw_files):
            return
        file_path = self.raw_files[idx]
        # Skip if already loaded and not forcing reload
        if not force_reload and self.display_pil_images[idx] is not None:
            return
        try:
            filename = os.path.basename(file_path).lower()
            if is_raw_file(filename):
                with rawpy.imread(file_path) as raw:
                    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=8, bright=10.0,
                                          half_size=True, use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB)
                    print(f"Loaded display for {os.path.basename(file_path)}: min={np.min(rgb)}, max={np.max(rgb)}")
                    # Set scale for RAW: half_size=True means display is half original
                    self.display_scales[idx] = 0.5
            else:
                img = Image.open(file_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                rgb = np.array(img)
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb / np.iinfo(rgb.dtype).max * 255, 0, 255).astype(np.uint8)
                # Set scale for non-RAW: full size
                self.display_scales[idx] = 1.0
            stretched = autostretch_image(rgb, self.stretch_factor, self.black_clip)
            self.display_pil_images[idx] = Image.fromarray(stretched)
        except Exception as e:
            print(f"Error loading display {file_path}: {e}")
            self.display_pil_images[idx] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8) + 128)
            # Default to 1.0 if error
            self.display_scales[idx] = 1.0

    def load_raw_image(self, filepath):
        filename = os.path.basename(filepath).lower()
        try:
            if is_raw_file(filename):
                with rawpy.imread(filepath) as raw:
                    rgb = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        output_bps=16,
                        use_camera_wb=True,
                        half_size=False
                    )
                    mono = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    return mono.astype(np.float32)
            else:
                # Load color via PIL for flexibility (handles 8/16-bit PNG/JPG)
                img = Image.open(filepath).convert('RGB')
                arr = np.array(img)
                if arr.dtype == np.uint8:
                    # Upscale 8-bit to 16-bit range
                    color_16 = (arr.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
                else:
                    # Assume uint16
                    color_16 = arr.astype(np.uint16)
                mono = cv2.cvtColor(color_16, cv2.COLOR_RGB2GRAY)
                return mono.astype(np.float32)
        except Exception as e:
            print(f"Error loading RAW for alignment: {os.path.basename(filepath)}, {e}")
            return None

    def _preload_raws(self):
        if self.raw_cache is None or len(self.raw_cache) != len(self.raw_files):
            return
        for i in range(len(self.raw_files)):
            if self.raw_cache[i] is None:
                self.raw_cache[i] = self.load_raw_image(self.raw_files[i])
        self.root.after(0, lambda: self.update_status("Raw data preloaded."))

    def load_points_from_file(self, directory):
        if not directory:
            return 0, 0
        points_file = os.path.join(directory, "alignment_points.txt")
        if not os.path.exists(points_file):
            print("Points file not found: alignment_points.txt")
            return 0, 0
        loaded_images = 0
        total_points = 0
        current_basename = None
        current_points = []
        print(f"Loading points from {points_file}")
        with open(points_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_basename and len(current_points) == 3 and check_collinearity(current_points):
                        for idx, file_path in enumerate(self.raw_files):
                            basename_match = os.path.splitext(os.path.basename(file_path))[0]
                            if basename_match == current_basename:
                                self.points_list[idx] = current_points[:3]
                                loaded_images += 1
                                total_points += 3
                                print(f"Loaded points for {current_basename} at index {idx}")
                                break
                    current_basename = None
                    current_points = []
                    continue
                if current_basename is None:
                    current_basename = line
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            if len(current_points) < 3:
                                current_points.append([x, y])
                        except ValueError:
                            pass
        # Handle last entry
        if current_basename and len(current_points) == 3 and check_collinearity(current_points):
            for idx, file_path in enumerate(self.raw_files):
                basename_match = os.path.splitext(os.path.basename(file_path))[0]
                if basename_match == current_basename:
                    self.points_list[idx] = current_points
                    loaded_images += 1
                    total_points += 3
                    print(f"Loaded points for {current_basename} at index {idx}")
                    break
        print(f"Loaded {loaded_images} images with {total_points} points total")
        return loaded_images, total_points

    def load_comet_from_file(self, directory, try_transformed=False):
        if not directory:
            return 0
        loaded = 0
        files_to_try = ["comet_annotations.txt"]
        if try_transformed:
            files_to_try.append("transformed_comet_annotations.txt")
        for filename in files_to_try:
            comet_file = os.path.join(directory, filename)
            if not os.path.exists(comet_file):
                print(f"Comet file not found: {filename}")
                continue
            print(f"Loading comet annotations from {comet_file}")
            with open(comet_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 6:  # basename cx cy r tx ty
                        basename = parts[0]
                        try:
                            cx = float(parts[1])
                            cy = float(parts[2])
                            r = max(float(parts[3]), 0.0)  # Ensure r >=0
                            tx = float(parts[4])
                            ty = float(parts[5])
                            vec_len = math.hypot(tx - cx, ty - cy)
                            if vec_len < 1.0:
                                print(f"Warning: Invalid tail length {vec_len} for {basename}, skipping")
                                continue
                            for idx, file_path in enumerate(self.raw_files):
                                basename_match = os.path.splitext(os.path.basename(file_path))[0]
                                if basename_match == basename or (filename.endswith("transformed") and basename_match + "_aligned" == basename):
                                    self.comet_ann[idx] = {"center": (cx, cy), "radius": r, "tail_dir": (tx, ty)}
                                    loaded += 1
                                    print(f"Loaded comet for {basename} at index {idx}: r={r:.1f}, tail_len={vec_len:.1f}")
                                    break
                        except (ValueError, IndexError) as e:
                            print(f"Invalid line in comet file {filename}: {line} ({e})")
                            pass
            if loaded > 0:
                break  # Stop if loaded from first file
        print(f"Loaded {loaded} comet annotations total")
        return loaded

    def save_points_to_file(self, out_dir=None, transformed_points=None):
        if out_dir is None:
            out_dir = self.input_dir
        if not out_dir:
            return
        points_file = os.path.join(out_dir, "alignment_points.txt")
        with open(points_file, 'w') as f:
            for idx, points in enumerate(self.points_list):
                if len(points) != 3 or not check_collinearity(points):
                    continue
                basename = os.path.splitext(os.path.basename(self.raw_files[idx]))[0]
                f.write(f"{basename}\n")
                for pt in points:
                    f.write(f"{pt[0]:.2f} {pt[1]:.2f}\n")
                f.write("\n")
        print(f"Original points saved to {points_file}")
        if transformed_points is not None:
            trans_file = os.path.join(out_dir, "transformed_points.txt")
            with open(trans_file, 'w') as f:
                for idx in transformed_points:
                    if transformed_points[idx] is None or len(transformed_points[idx]) != 3:
                        continue
                    basename = os.path.splitext(os.path.basename(self.raw_files[idx]))[0] + "_aligned"
                    f.write(f"{basename}\n")
                    for pt in transformed_points[idx]:
                        f.write(f"{pt[0]:.2f} {pt[1]:.2f}\n")
                    f.write("\n")
            print(f"Transformed points saved to {trans_file}")

    def save_comet_annotations(self, out_dir=None, ann_dict=None):
        if out_dir is None:
            out_dir = self.input_dir
        if not out_dir:
            return
        if ann_dict is None:
            ann_dict = self.comet_ann
        if all(a is None for a in ann_dict) if isinstance(ann_dict, (list, tuple)) else all(v is None for v in ann_dict.values() if isinstance(ann_dict, dict)):
            return  # Skip if no annotations
        is_dict = isinstance(ann_dict, dict)
        idxs = sorted(ann_dict.keys()) if is_dict else range(len(ann_dict))
        filename = "comet_annotations.txt" if out_dir == self.input_dir else "transformed_comet_annotations.txt"
        file_path = os.path.join(out_dir, filename)
        with open(file_path, 'w') as f:
            for idx in idxs:
                ann = ann_dict.get(idx) if is_dict else ann_dict[idx]
                if ann is None:
                    continue
                basename = os.path.splitext(os.path.basename(self.raw_files[idx]))[0]
                if filename.endswith("transformed"):
                    basename += "_aligned"
                cx, cy = ann["center"]
                r = ann.get("radius", 20.0)
                tx, ty = ann["tail_dir"]
                f.write(f"{basename} {cx:.2f} {cy:.2f} {r:.2f} {tx:.2f} {ty:.2f}\n")
        print(f"Comet annotations saved to {file_path}")

    def load_directory(self):
        directory = filedialog.askdirectory(title="Select Directory (RAW/PNG/JPG)")
        if not directory:
            self.update_status("Load cancelled.")
            return
        self.reset_data()
        self.input_dir = directory
        self.progress_var.set(0)
        self.update_status("Scanning directory...")
        self.root.update_idletasks()
        extensions = ('.dng', '.cr2', '.cr3', '.png', '.jpg', '.jpeg')
        raw_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)]
        raw_files.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0].lower())
        if not raw_files:
            messagebox.showerror("Error", "No supported files (DNG, CR2, CR3, PNG, JPG, JPEG) found in directory.")
            self.update_status("Ready")
            return
        self.finalize_loading(raw_files)

    def finalize_loading(self, raw_files):
        total_files = len(raw_files)
        self.raw_files = raw_files
        self.display_pil_images = [None] * total_files
        self.points_list = [[] for _ in raw_files]
        self.comet_ann = [None] * total_files
        self.current_idx = 0
        self.ref_idx = None  # Will restore below if valid
        self.ref_label.config(text="None selected")
        self.set_ref_var.set(False)
        self.raw_cache = [None] * total_files
        self.display_scales = [1.0] * total_files  # Default to 1.0, updated per image on load
        # Load existing annotations (stars first, then comet)
        loaded_points, total_points = self.load_points_from_file(self.input_dir)
        loaded_comet = self.load_comet_from_file(self.input_dir, try_transformed=True)  # Try transformed if main empty
        load_msg = f"Loaded {loaded_points} stars images with {total_points} points, {loaded_comet} comet annotations."
        print(load_msg)
        # Restore reference if possible (check current or first valid)
        mode = self.align_mode.get()
        if mode == "stars":
            for idx in range(total_files):
                if len(self.points_list[idx]) == 3 and check_collinearity(self.points_list[idx]):
                    self.ref_idx = idx
                    self.set_ref_var.set(True)
                    base_name = os.path.basename(self.raw_files[idx])
                    self.ref_label.config(text=f"Restored: {base_name} (index {idx})")
                    print(f"Restored stars reference: {idx}")
                    break
        else:  # comet
            for idx in range(total_files):
                if self.comet_ann[idx] is not None and is_valid_comet_ann(self.comet_ann[idx]):
                    self.ref_idx = idx
                    self.set_ref_var.set(True)
                    base_name = os.path.basename(self.raw_files[idx])
                    self.ref_label.config(text=f"Restored: {base_name} (index {idx})")
                    print(f"Restored comet reference: {idx}")
                    break
        # Recreate indicators after loading (since reset destroyed them)
        self.update_samples_ui()
        # Update indicators to reflect loaded data
        self.update_indicators()
        # Force redraw for current image if annotations loaded
        if self.raw_files:
            self.redraw_points()
        self.update_status(f"Pre-loading first 10 display images...")
        num_preload = min(10, total_files)
        if num_preload > 0:
            self.progress_var.set(0)
            for i in range(num_preload):
                self.load_display_image(i)
                self.progress_var.set(((i + 1) / num_preload) * 100)
                self.root.update_idletasks()
            self.progress_var.set(100)
        self.update_status("Pre-loading raw data in background...")
        threading.Thread(target=self._preload_raws, daemon=True).start()
        self.show_current_image()
        self.current_label.config(text=f"Image {self.current_idx + 1}/{total_files}")
        self.update_status(f"Ready. {load_msg}")
        self.root.after(1000, self.schedule_preload)

    def update_status(self, text):
        self.status_label.config(text=text)

    def show_current_image(self):
        if not self.display_pil_images or self.current_idx >= len(self.display_pil_images):
            self.title_label.config(text="No image loaded")
            self.update_status("Image not loaded")
            return
        if self.display_pil_images[self.current_idx] is None:
            self.update_status(f"Loading image {self.current_idx}...")
            self.load_display_image(self.current_idx)
            if self.display_pil_images[self.current_idx] is None:
                self.title_label.config(text="Failed to load image")
                self.update_status("Image load failed")
                return
        self.root.update_idletasks()
        self.canvas.update_idletasks()
        img = self.display_pil_images[self.current_idx]
        self.title_label.config(text=os.path.basename(self.raw_files[self.current_idx]))
        self.current_label.config(text=f"Image {self.current_idx + 1}/{len(self.raw_files)}")
        self.update_status("Ready")
        self.canvas.delete("all")
        self.image_item = None
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            self.root.after(100, self.show_current_image)
            return
        img_w, img_h = img.size
        self.current_img_w = float(img_w)
        self.current_img_h = float(img_h)
        is_new_image = self.current_idx != self.last_shown_idx
        self.last_shown_idx = self.current_idx
        do_center = False
        if is_new_image:
            if not self.initial_fit_done:
                fit_scale = min((canvas_w * 0.9 / img_w), (canvas_h * 0.9 / img_h))
                self.zoom_factor = fit_scale
                self.initial_fit_done = True
                do_center = True
                print(f"Initial auto-fit: scale={self.zoom_factor:.3f}")
        scaled_width = int(img_w * self.zoom_factor)
        scaled_height = int(img_h * self.zoom_factor)
        if scaled_width > 0 and scaled_height > 0:
            scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(scaled_img)
            self.image_item = self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor=tk.NW, tags="image")
            self.current_scaled_w = float(scaled_width)
            self.current_scaled_h = float(scaled_height)
            self._update_display_position(do_center=do_center)
            self.redraw_points()
            self.update_indicators()
        self.root.after(0, self.schedule_preload)

    def update_scaled_image(self):
        if not self.display_pil_images or self.current_idx >= len(self.display_pil_images):
            return
        if self.display_pil_images[self.current_idx] is None:
            return
        img = self.display_pil_images[self.current_idx]
        img_w, img_h = img.size
        scaled_width = int(img_w * self.zoom_factor)
        scaled_height = int(img_h * self.zoom_factor)
        if scaled_width > 0 and scaled_height > 0:
            scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(scaled_img)
            if self.image_item:
                self.canvas.itemconfig(self.image_item, image=self.photo)
            self.current_scaled_w = float(scaled_width)
            self.current_scaled_h = float(scaled_height)
            self._update_scrollregion()
            self.redraw_points()

    def on_canvas_resize(self, event):
        if event.widget != self.canvas or event.width < 1 or event.height < 1:
            return
        self.update_scaled_image()
        # Re-center if image smaller than new canvas size (optional, for better UX)
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if self.current_scaled_w < canvas_w or self.current_scaled_h < canvas_h:
            self._update_display_position(do_center=True)

    def on_mousewheel(self, event):
        self.zoom_image(event)

    def zoom_image(self, event):
        if self.image_item is None or self.current_scaled_w <= 0:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return
        old_zoom = self.zoom_factor
        mouse_canvas_x = event.x
        mouse_canvas_y = event.y
        # Image-relative position before zoom
        mouse_img_x = (mouse_canvas_x - self.offset_x) / old_zoom
        mouse_img_y = (mouse_canvas_y - self.offset_y) / old_zoom
        # Apply zoom factor
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        new_zoom = old_zoom * factor
        self.zoom_factor = max(0.1, min(10.0, new_zoom))
        # Update scaled image
        self.update_scaled_image()
        # Adjust offset to keep mouse_img point under cursor
        new_offset_x = mouse_canvas_x - mouse_img_x * self.zoom_factor
        new_offset_y = mouse_canvas_y - mouse_img_y * self.zoom_factor
        self.offset_x = new_offset_x
        self.offset_y = new_offset_y
        if self.image_item:
            self.canvas.coords(self.image_item, self.offset_x, self.offset_y)
        self._update_scrollregion()
        self.redraw_points()

    def start_pan(self, event):
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.pan_start_offset_x = self.offset_x
        self.pan_start_offset_y = self.offset_y

    def pan(self, event):
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.offset_x = self.pan_start_offset_x + dx
            self.offset_y = self.pan_start_offset_y + dy
            if self.image_item:
                self.canvas.coords(self.image_item, self.offset_x, self.offset_y)
            self._update_scrollregion()
            self.redraw_points()  # Redraw points during pan for smooth feedback

    def stop_pan(self, event):
        self.is_panning = False

    def redraw_points(self):
        self.canvas.delete("point")
        if not self.image_item or self.current_idx >= len(self.raw_files):
            return
        mode = self.align_mode.get()
        current_scale = self.display_scales[self.current_idx] if self.current_idx < len(self.display_scales) else 1.0
        if mode == "stars":
            points = self.points_list[self.current_idx]
            if len(points) == 0:
                return
            arm_length = 8.0
            text_offset = 12.0
            for i, pt in enumerate(points):
                if len(pt) == 2:
                    display_pt_x = pt[0] * current_scale
                    display_pt_y = pt[1] * current_scale
                    content_x = self.offset_x + display_pt_x * self.zoom_factor
                    content_y = self.offset_y + display_pt_y * self.zoom_factor
                    self.canvas.create_line(content_x - arm_length, content_y, content_x + arm_length, content_y, fill="lime", width=1, tags="point")
                    self.canvas.create_line(content_x, content_y - arm_length, content_x, content_y + arm_length, fill="lime", width=1, tags="point")
                    self.canvas.create_text(content_x + text_offset, content_y, text=f"P{i+1}", fill="lime", anchor=tk.W, tags="point", font=("Arial", 10, "bold"))
            if len(points) == 3:
                self.save_points_to_file(self.input_dir)
        else:  # comet
            ann = self.comet_ann[self.current_idx]
            if ann:
                cx, cy = ann["center"]
                r = ann.get("radius", 20.0)  # Fallback
                ann["radius"] = r
                tx, ty = ann["tail_dir"]
                display_cx = cx * current_scale
                display_cy = cy * current_scale
                display_tx = tx * current_scale
                display_ty = ty * current_scale
                display_r = r * current_scale
                content_cx = self.offset_x + display_cx * self.zoom_factor
                content_cy = self.offset_y + display_cy * self.zoom_factor
                content_tx = self.offset_x + display_tx * self.zoom_factor
                content_ty = self.offset_y + display_ty * self.zoom_factor
                # Circle for coma
                self.canvas.create_oval(content_cx - display_r * self.zoom_factor, content_cy - display_r * self.zoom_factor,
                                        content_cx + display_r * self.zoom_factor, content_cy + display_r * self.zoom_factor,
                                        outline="green", width=2, tags="point")
                # Line for tail
                self.canvas.create_line(content_cx, content_cy, content_tx, content_ty, fill="blue", width=3, tags="point")
                # Cross at center
                arm = 8 * self.zoom_factor
                self.canvas.create_line(content_cx - arm, content_cy, content_cx + arm, content_cy, fill="lime", width=1, tags="point")
                self.canvas.create_line(content_cx, content_cy - arm, content_cx, content_cy + arm, fill="lime", width=1, tags="point")
                # Labels
                self.canvas.create_text(content_cx + 12 * self.zoom_factor, content_cy, text="C", fill="lime", anchor=tk.W, tags="point", font=("Arial", 10, "bold"))
                self.canvas.create_text(content_tx + 12 * self.zoom_factor, content_ty, text="T", fill="blue", anchor=tk.W, tags="point", font=("Arial", 10, "bold"))
                self.save_comet_annotations(self.input_dir)

    def on_click(self, event):
        mode = self.align_mode.get()
        if self.current_idx >= len(self.raw_files) or self.display_pil_images[self.current_idx] is None or self.current_scaled_w <= 0:
            return
        img_w = self.current_img_w
        img_h = self.current_img_h
        if img_w <= 0 or img_h <= 0:
            return
        mouse_content_x = event.x  # Fixed view, so canvasx = x
        mouse_content_y = event.y
        rel_x = mouse_content_x - self.offset_x
        rel_y = mouse_content_y - self.offset_y
        if not (0 <= rel_x <= self.current_scaled_w and 0 <= rel_y <= self.current_scaled_h):
            return  # Click outside image - ignore
        display_orig_x = rel_x / self.zoom_factor
        display_orig_y = rel_y / self.zoom_factor
        display_orig_x = max(0.0, min(display_orig_x, img_w - 1))
        display_orig_y = max(0.0, min(display_orig_y, img_h - 1))
        current_scale = self.display_scales[self.current_idx] if self.current_idx < len(self.display_scales) else 1.0
        full_orig_x = display_orig_x / current_scale
        full_orig_y = display_orig_y / current_scale
        if mode == "stars":
            points = self.points_list[self.current_idx]
            if len(points) < 3:
                points.append([full_orig_x, full_orig_y])
                self.redraw_points()
                self.update_indicators()
                if len(points) == 3 and self.ref_idx == self.current_idx and not self.set_ref_var.get():
                    if check_collinearity(points):
                        self.set_ref_var.set(True)
                        self.on_set_ref()
                    else:
                        messagebox.showwarning("Warning", "Points are collinear—spread them out for stable alignment.")
                return
            else:
                display_points = [[p[0] * current_scale, p[1] * current_scale] for p in points]
                distances = [math.hypot(display_orig_x - dp[0], display_orig_y - dp[1]) for dp in display_points]
                min_dist = min(distances)
                if min_dist < 15.0 / self.zoom_factor:
                    self.dragged_point_idx = distances.index(min_dist)
                else:
                    self.dragged_point_idx = None
                return
        else:  # mode == "comet"
            ann = self.comet_ann[self.current_idx]
            if ann is None:
                # Initial click: start with center
                self._comet_state = "circle_drag"
                self.temp_comet_center = [full_orig_x, full_orig_y]
                self.temp_comet_radius = 0.0
                self.comet_ann[self.current_idx] = {"center": (full_orig_x, full_orig_y), "radius": 0.0, "tail_dir": (full_orig_x, full_orig_y)}
                self.dragged_comet_part = "center"
                self.redraw_points()
                self.update_indicators()
                return
            cx, cy = ann["center"]
            r = ann["radius"]
            tx, ty = ann["tail_dir"]
            vec_len = math.hypot(tx - cx, ty - cy)
            has_tail = vec_len > 1.0
            cx_display = cx * current_scale
            cy_display = cy * current_scale
            r_display = r * current_scale
            tx_display = tx * current_scale
            ty_display = ty * current_scale
            dist_to_center_display = math.hypot(display_orig_x - cx_display, display_orig_y - cy_display)
            inside_circle = dist_to_center_display <= r_display * 1.5
            dist_to_tail_end_display = math.hypot(display_orig_x - tx_display, display_orig_y - ty_display)
            near_tail_end = dist_to_tail_end_display < 30.0
            if inside_circle:
                self._comet_state = "drag_center"
                self.dragged_comet_part = "center"
                self.temp_comet_center = [cx, cy]
                if has_tail:
                    self.temp_tail_end = [tx, ty]
                self.redraw_points()
                return
            elif has_tail and near_tail_end:
                self._comet_state = "tail_drag"
                self.dragged_comet_part = "tail"
                self.temp_tail_end = [tx, ty]
                self.redraw_points()
                return
            elif not has_tail or vec_len < 10.0:
                if dist_to_center_display > r_display:
                    self._comet_state = "tail_drag"
                    self.dragged_comet_part = "tail"
                    self.temp_tail_end = [full_orig_x, full_orig_y]
                    ann["tail_dir"] = (full_orig_x, full_orig_y)
                    self.redraw_points()
                    return
                else:
                    self._comet_state = "drag_center"
                    self.dragged_comet_part = "center"
                    self.temp_comet_center = [cx, cy]
                    self.temp_tail_end = [tx, ty]
                    self.redraw_points()
                    return
            else:
                return

    def on_point_drag(self, event):
        mode = self.align_mode.get()
        if mode == "stars":
            if self.dragged_point_idx is None or self.current_idx >= len(self.raw_files) or self.display_pil_images[self.current_idx] is None or self.current_scaled_w <= 0:
                return
            img_w = self.current_img_w
            img_h = self.current_img_h
            if img_w <= 0 or img_h <= 0:
                return
            mouse_content_x = event.x
            mouse_content_y = event.y
            display_new_x = (mouse_content_x - self.offset_x) / self.zoom_factor
            display_new_y = (mouse_content_y - self.offset_y) / self.zoom_factor
            display_new_x = max(0.0, min(display_new_x, img_w - 1))
            display_new_y = max(0.0, min(display_new_y, img_h - 1))
            current_scale = self.display_scales[self.current_idx] if self.current_idx < len(self.display_scales) else 1.0
            full_new_x = display_new_x / current_scale
            full_new_y = display_new_y / current_scale
            points = self.points_list[self.current_idx]
            points[self.dragged_point_idx] = [full_new_x, full_new_y]
            self.redraw_points()
            return
        # Comet drag (updated for fixed view)
        if self._comet_state not in ["circle_drag", "drag_center", "tail_drag"] or self.dragged_comet_part is None:
            return
        img_w = self.current_img_w
        img_h = self.current_img_h
        if img_w <= 0 or img_h <= 0:
            return
        mouse_content_x = event.x
        mouse_content_y = event.y
        display_new_x = (mouse_content_x - self.offset_x) / self.zoom_factor
        display_new_y = (mouse_content_y - self.offset_y) / self.zoom_factor
        display_new_x = max(0.0, min(display_new_x, img_w - 1))
        display_new_y = max(0.0, min(display_new_y, img_h - 1))
        current_scale = self.display_scales[self.current_idx] if self.current_idx < len(self.display_scales) else 1.0
        full_new_x = display_new_x / current_scale
        full_new_y = display_new_y / current_scale
        ann = self.comet_ann[self.current_idx]
        if self.dragged_comet_part == "center":
            if self._comet_state == "circle_drag":
                dr_x = full_new_x - self.temp_comet_center[0]
                dr_y = full_new_y - self.temp_comet_center[1]
                self.temp_comet_radius = max(5.0, math.hypot(dr_x, dr_y))
                ann["radius"] = self.temp_comet_radius
            else:
                dx = full_new_x - ann["center"][0]
                dy = full_new_y - ann["center"][1]
                self.temp_comet_center = [full_new_x, full_new_y]
                new_tx = ann["tail_dir"][0] + dx
                new_ty = ann["tail_dir"][1] + dy
                ann["center"] = (full_new_x, full_new_y)
                ann["tail_dir"] = (new_tx, new_ty)
            self.redraw_points()
            self.update_indicators()
        elif self.dragged_comet_part == "tail":
            self.temp_tail_end = [full_new_x, full_new_y]
            ann["tail_dir"] = (full_new_x, full_new_y)
            self.redraw_points()
            self.update_indicators()

    def on_point_release(self, event):
        mode = self.align_mode.get()
        if mode == "stars":
            if self.dragged_point_idx is not None:
                self.dragged_point_idx = None
                self.redraw_points()
                self.update_indicators()
                if len(self.points_list[self.current_idx]) == 3:
                    self.save_points_to_file(self.input_dir)
                return
        # Comet release
        if self._comet_state in ["circle_drag", "drag_center", "tail_drag"]:
            ann = self.comet_ann[self.current_idx]
            if ann is not None:
                if "radius" in ann and ann["radius"] < 5.0:
                    ann["radius"] = 5.0
            self._comet_state = "idle"
            self.dragged_comet_part = None
            self.temp_comet_center = None
            self.temp_comet_radius = 0.0
            self.temp_tail_end = None
            self.redraw_points()
            self.update_indicators()
            self.save_comet_annotations(self.input_dir)

    def prev_image(self):
        if self.current_idx > 0:
            mode = self.align_mode.get()
            if mode == "stars":
                self.save_points_to_file(self.input_dir)
            else:
                self.save_comet_annotations(self.input_dir)
            self.current_idx -= 1
            self.show_current_image()

    def next_image(self):
        if len(self.raw_files) > 0 and self.current_idx < len(self.raw_files) - 1:
            mode = self.align_mode.get()
            if mode == "stars":
                self.save_points_to_file(self.input_dir)
            else:
                self.save_comet_annotations(self.input_dir)
            self.current_idx += 1
            self.show_current_image()

    def align_and_save(self):
        mode = self.align_mode.get()
        if not self.raw_files:
            messagebox.showerror("Error", "No images loaded.")
            return
        if mode == "comet":
            if self.ref_idx is None:
                messagebox.showerror("Error", "No reference image selected (needs comet annotation).")
                return
            ref_ann = self.comet_ann[self.ref_idx]
            if ref_ann is None or not is_valid_comet_ann(ref_ann):
                messagebox.showerror("Error", "Reference must have valid comet annotation (center + tail).")
                return
            ref_triplet = comet_to_triplet(ref_ann)
            if ref_triplet is None or len(ref_triplet) != 3:
                messagebox.showerror("Error", "Invalid reference comet: check center and tail direction.")
                return
            self.transform_mode.set("euclidean")
            print("Comet alignment: using rigid transform (center + direction only)")
        else:
            if self.ref_idx is None:
                messagebox.showerror("Error", "No reference image selected.")
                return
            ref_points = np.array(self.points_list[self.ref_idx], dtype=np.float32)
            if len(ref_points) != 3 or not check_collinearity(ref_points):
                messagebox.showerror("Error", "Reference must have 3 non-collinear points.")
                return
        outdir = filedialog.askdirectory(title="Select Output Directory for Aligned FITS")
        if not outdir:
            self.update_status("Save cancelled.")
            return
        os.makedirs(outdir, exist_ok=True)
        self.update_status("Starting alignment process...")
        self.progress_var.set(0)
        self.root.update_idletasks()
        ref_filepath = self.raw_files[self.ref_idx]
        filename = os.path.basename(ref_filepath).lower()
        try:
            if is_raw_file(filename):
                with rawpy.imread(ref_filepath) as raw:
                    ref_color = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1, 1))
                    ref_date_obs = extract_raw_date(raw, ref_filepath)
            else:
                img = Image.open(ref_filepath).convert('RGB')
                arr = np.array(img)
                if arr.dtype == np.uint8:
                    ref_color = (arr.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
                else:
                    ref_color = arr.astype(np.uint16)
                ref_date_obs = extract_date_from_exif(ref_filepath)
            ref_mono = cv2.cvtColor(ref_color, cv2.COLOR_RGB2GRAY)
            print(f"Loaded reference: {os.path.basename(ref_filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reference: {e}")
            self.update_status("Ready.")
            return
        ref_size = (ref_mono.shape[1], ref_mono.shape[0])
        transformed_ann = {} if mode == "comet" else {i: None for i in range(len(self.raw_files))}
        processed_count = 0
        skipped_count = 0
        total_files = len(self.raw_files)
        for i, filepath in enumerate(self.raw_files):
            if mode == "comet":
                src_ann = self.comet_ann[i]
                if src_ann is None:
                    print(f"Skipping image {i}: no comet annotation.")
                    skipped_count += 1
                    self.progress_var.set((i + 1) / total_files * 100)
                    self.root.update_idletasks()
                    continue
                src_triplet = comet_to_triplet(src_ann)
                if src_triplet is None or len(src_triplet) != 3:
                    print(f"Skipping image {i}: invalid comet triplet.")
                    skipped_count += 1
                    self.progress_var.set((i + 1) / total_files * 100)
                    self.root.update_idletasks()
                    continue
                src_points, dst_points = src_triplet, ref_triplet
            else:
                src_points = np.array(self.points_list[i], dtype=np.float32)
                if len(src_points) != 3 or not check_collinearity(src_points):
                    print(f"Skipping image {i}: invalid points.")
                    skipped_count += 1
                    self.progress_var.set((i + 1) / total_files * 100)
                    self.root.update_idletasks()
                    continue
                dst_points = ref_points
            src_filename = os.path.basename(filepath).lower()
            try:
                if is_raw_file(src_filename):
                    with rawpy.imread(filepath) as raw:
                        src_color = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1, 1))
                        src_date_obs = extract_raw_date(raw, filepath)
                else:
                    img = Image.open(filepath).convert('RGB')
                    arr = np.array(img)
                    if arr.dtype == np.uint8:
                        src_color = (arr.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
                    else:
                        src_color = arr.astype(np.uint16)
                    src_date_obs = extract_date_from_exif(filepath)
                src_mono = cv2.cvtColor(src_color, cv2.COLOR_RGB2GRAY).astype(np.float32)
            except Exception as e:
                print(f"Skipping image {i}: load error {e}")
                skipped_count += 1
                self.progress_var.set((i + 1) / total_files * 100)
                self.root.update_idletasks()
                continue
            if i == self.ref_idx:
                M = np.eye(2, 3, dtype=np.float32)
            else:
                try:
                    M, _ = compute_transform(src_points, dst_points, mode=self.transform_mode.get())
                    if M is None:
                        raise ValueError("Transform failed.")
                except Exception as e:
                    print(f"Skipping image {i}: transform error {e}")
                    skipped_count += 1
                    self.progress_var.set((i + 1) / total_files * 100)
                    self.root.update_idletasks()
                    continue
            aligned_color = cv2.warpAffine(src_color, M, ref_size,
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0))
            fits_data = np.moveaxis(aligned_color, 2, 0).astype(np.float32)
            filename = os.path.join(outdir, f"aligned_{i+1:05d}.fit")
            try:
                header = fits.Header()
                if src_date_obs:
                    header['DATE-OBS'] = src_date_obs
                    header['CREATED'] = src_date_obs
                else:
                    print(f"No DATE-OBS for image {i}.")
                hdu = fits.PrimaryHDU(data=fits_data, header=header)
                hdu.writeto(filename, overwrite=True)
                processed_count += 1
                print(f"Saved: {filename}")
                if mode == "comet":
                    trans_center = transform_points(M, [[src_ann["center"][0], src_ann["center"][1]]])[0]
                    trans_tail = transform_points(M, [[src_ann["tail_dir"][0], src_ann["tail_dir"][1]]])[0]
                    transformed_ann[i] = {"center": tuple(trans_center), "radius": src_ann["radius"], "tail_dir": tuple(trans_tail)}
                else:
                    transformed_ann[i] = transform_points(M, self.points_list[i])
            except Exception as e:
                print(f"Save error for {filename}: {e}")
                skipped_count += 1
            self.progress_var.set((i + 1) / total_files * 100)
            self.root.update_idletasks()
            del src_color, src_mono, aligned_color, fits_data
            gc.collect()
        if mode == "stars":
            self.save_points_to_file(self.input_dir)
            self.save_points_to_file(outdir, transformed_ann)
        else:
            self.save_comet_annotations(self.input_dir)
            self.save_comet_annotations(outdir, transformed_ann)
        del ref_color, ref_mono
        gc.collect()
        message = f"Alignment complete!\nProcessed: {processed_count} images.\nSkipped: {skipped_count} images."
        messagebox.showinfo("Success", message)
        self.update_status("Ready.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AstroAligner()
    app.run()
