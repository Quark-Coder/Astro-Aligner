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

def autostretch_image(img):
    if len(img.shape) == 2:
        ch = img.astype(np.float64) / 255.0
    else:
        ch = np.mean(img.astype(np.float64) / 255.0, axis=2)  
    a = 20.0
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

def extract_dng_dates(raw):
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

def extract_date_from_exif(dng_file_path):
    try:
        with open(dng_file_path, 'rb') as f:
            tags = exifread.process_file(f, stop_tag='EXIF DateTimeOriginal')
            if 'EXIF DateTimeOriginal' in tags:
                date_time_str = str(tags['EXIF DateTimeOriginal']).strip()
                dt = datetime.strptime(date_time_str, '%Y:%m:%d %H:%M:%S')
                fits_date = dt.strftime('%Y-%m-%dT%H:%M:%S')
                return fits_date
            else:
                print(f"DateTimeOriginal not found in EXIF for {os.path.basename(dng_file_path)}")
                return None
    except Exception as e:
        print(f"EXIF extraction failed for {os.path.basename(dng_file_path)}: {e}")
        return None

def transform_points(M, points):
    if M is None:
        return points
    points_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(points_np, M)
    return transformed.reshape(-1, 2).tolist()

class AstroAligner:
    def __init__(self):
        print("Created by Quark")
        self.root = tk.Tk()
        self.root.title("3 point Image Aligner")
        try:
            icon_path = os.path.join(sys._MEIPASS, 'icon.ico')
            self.root.iconbitmap(icon_path)
        except tk.TclError as e:
            print(f"Failed to load icon: {e}. Using default Tkinter icon.")
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.dng_files = []
        self.display_pil_images = []  
        self.points_list = []
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
        self.display_scale = 1.0  
        self.transform_mode = tk.StringVar(value="euclidean")  
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.setup_ui()
        self.root.update_idletasks()  
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        max_w = min(1400, int(screen_w * 0.8))
        max_h = min(900, int(screen_h * 0.8))
        win_w = max(1000, max_w)  
        win_h = max(700, max_h)  
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.focus_set()

    def reset_data(self):
        self.dng_files.clear()
        self.display_pil_images.clear()
        self.points_list.clear()
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
        self.raw_cache = None
        self.display_scale = 1.0  
        if hasattr(self, 'canvas'):
            self.canvas.delete("all")
        if hasattr(self, 'ref_label'):
            self.ref_label.config(text="None selected")
        if hasattr(self, 'set_ref_var'):
            self.set_ref_var.set(False)

    def setup_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_container, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.status_label = ttk.Label(main_container, text="Ready")
        self.status_label.pack(pady=2)
        self.title_label = ttk.Label(main_container, text="", font=("Arial", 12, "bold"))
        self.title_label.pack(pady=5)
        control_panel = ttk.Frame(main_container, width=200, relief="ridge")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        load_btn = ttk.Button(control_panel, text="Load DNG Directory", command=self.load_directory)
        load_btn.pack(pady=10)
        nav_frame = ttk.Frame(control_panel)
        nav_frame.pack(pady=10)
        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        self.current_label = ttk.Label(nav_frame, text="Image 0/0")
        self.current_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_panel, text="Transform Mode:", font=("Arial", 10, "bold")).pack(pady=(10,0))
        self.transform_combo = ttk.Combobox(control_panel, textvariable=self.transform_mode, values=["euclidean", "similarity"], state="readonly", width=12)
        self.transform_combo.pack(pady=5)
        self.transform_combo.bind("<<ComboboxSelected>>", lambda e: print(f"Transform mode: {self.transform_mode.get()}"))
        ttk.Label(control_panel, text="Samples:", font=("Arial", 10, "bold")).pack(pady=(10,5))
        self.indicators = []
        for i in range(3):
            ind_frame = tk.Frame(control_panel, width=20, height=20, bg="red")
            ind_frame.pack(pady=2)
            ttk.Label(control_panel, text=f"Sample {i+1}").pack()
            self.indicators.append(ind_frame)
        clear_btn = ttk.Button(control_panel, text="Clear Points", command=self.clear_current_points)
        clear_btn.pack(pady=5)
        ttk.Separator(control_panel, orient="horizontal").pack(pady=10, fill=tk.X)
        ttk.Label(control_panel, text="Reference Image:", font=("Arial", 10, "bold")).pack(pady=(10,5))
        self.ref_label = ttk.Label(control_panel, text="None selected")
        self.ref_label.pack(pady=5)
        self.set_ref_var = tk.BooleanVar(value=False)
        self.ref_checkbox = ttk.Checkbutton(control_panel, text="Set current as reference (requires 3 points)",
                                            variable=self.set_ref_var, command=self.on_set_ref)
        self.ref_checkbox.pack(pady=5)
        ttk.Separator(control_panel, orient="horizontal").pack(pady=10, fill=tk.X)
        align_btn = ttk.Button(control_panel, text="Align and Save FITS", command=self.align_and_save)
        align_btn.pack(pady=20)
        self.center_btn = ttk.Button(control_panel, text="Center Image", command=self.center_image)
        self.center_btn.pack(pady=5)
        self.canvas_frame = ttk.Frame(main_container)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        v_scrollbar.config(command=self.canvas.yview)
        h_scrollbar.config(command=self.canvas.xview)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_point_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_point_release)
        self.canvas.bind("<Button-3>", self.start_pan)  
        self.canvas.bind("<B3-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-3>", self.stop_pan)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-2>", lambda e: self.canvas.focus_set())

    def clear_current_points(self):
        if self.current_idx < len(self.points_list):
            self.points_list[self.current_idx] = []
            self.redraw_points()
            self.update_indicators()
            print(f"Cleared points for image {self.current_idx}")
            if self.ref_idx == self.current_idx:
                self.set_ref_var.set(False)
                self.ref_label.config(text="None selected")
                self.ref_idx = None
        self.save_points_to_file(self.input_dir)

    def center_image(self):
        if self.current_idx >= len(self.display_pil_images) or self.display_pil_images[self.current_idx] is None:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return
        self.update_scaled_image()
        self._apply_layout_after_scale(self.current_scaled_w, self.current_scaled_h, do_center=True)
        if self.current_scaled_w > canvas_w:
            self.canvas.xview_moveto(0.5)
            self.canvas.yview_moveto(0.5)
        else:
            self.canvas.xview_moveto(0.0)
            self.canvas.yview_moveto(0.0)
        self.redraw_points()

    def on_set_ref(self):
        if self.set_ref_var.get():
            if len(self.points_list[self.current_idx]) != 3 or not check_collinearity(self.points_list[self.current_idx]):
                messagebox.showwarning("Warning", "Current image must have exactly 3 non-collinear points to set as reference.")
                self.set_ref_var.set(False)
                return
            self.ref_idx = self.current_idx
            base_name = os.path.basename(self.dng_files[self.current_idx]) if self.dng_files else f"Image {self.current_idx}"
            self.ref_label.config(text=f"Selected: {base_name} (index {self.current_idx})")
            print(f"Set reference to image {self.current_idx}")
            self.save_points_to_file(self.input_dir)
        else:
            self.ref_idx = None
            self.ref_label.config(text="None selected")
            print("Reference deselected")
            self.save_points_to_file(self.input_dir)

    def schedule_preload(self):
        if self.preload_active:
            return
        if not self.dng_files or self.current_idx >= len(self.dng_files) - 1:
            return
        next_start = self.current_idx + 1
        next_end = min(next_start + 10, len(self.dng_files))
        unloaded = [i for i in range(next_start, next_end) if i < len(self.display_pil_images) and self.display_pil_images[i] is None]
        if unloaded:
            self.preload_active = True
            threading.Thread(target=self._preload_batch, args=(unloaded,), daemon=True).start()

    def _preload_batch(self, indices):
        for idx in indices:
            if idx < len(self.display_pil_images) and self.display_pil_images[idx] is None:
                self.load_display_image(idx)
        self.root.after(0, lambda: setattr(self, 'preload_active', False))

    def load_display_image(self, idx):
        if idx >= len(self.dng_files):
            return
        file_path = self.dng_files[idx]
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=8, bright=10.0,
                                      half_size=True, use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB)
                print(f"Loaded display for {os.path.basename(file_path)}: min={np.min(rgb)}, max={np.max(rgb)}")
                stretched = autostretch_image(rgb)
                self.display_pil_images[idx] = Image.fromarray(stretched)
        except Exception as e:
            print(f"Error loading display {file_path}: {e}")
            self.display_pil_images[idx] = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8) + 128)

    def load_raw_image(self, filepath):
        try:
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
        except Exception as e:
            print(f"Error loading RAW for alignment: {os.path.basename(filepath)}, {e}")
            return None

    def _preload_raws(self):
        if self.raw_cache is None or len(self.raw_cache) != len(self.dng_files):
            return
        for i in range(len(self.dng_files)):
            if self.raw_cache[i] is None:
                self.raw_cache[i] = self.load_raw_image(self.dng_files[i])
        self.root.after(0, lambda: self.update_status("Raw data preloaded."))

    def load_points_from_file(self, directory):
        if not directory:
            return 0, 0
        points_file = os.path.join(directory, "alignment_points.txt")
        if not os.path.exists(points_file):
            return 0, 0  
        loaded_images = 0
        total_points = 0
        current_basename = None
        current_points = []
        with open(points_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_basename and len(current_points) == 3 and check_collinearity(current_points):  
                        for idx, file_path in enumerate(self.dng_files):
                            if os.path.splitext(os.path.basename(file_path))[0] == current_basename:
                                self.points_list[idx] = current_points[:3]  
                                loaded_images += 1
                                total_points += 3
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
        if current_basename and len(current_points) == 3 and check_collinearity(current_points):
            for idx, file_path in enumerate(self.dng_files):
                if os.path.splitext(os.path.basename(file_path))[0] == current_basename:
                    self.points_list[idx] = current_points
                    loaded_images += 1
                    total_points += 3
                    break
        return loaded_images, total_points

    def save_points_to_file(self, out_dir=None, transformed_points=None):
        if self.input_dir:
            points_file = os.path.join(self.input_dir, "alignment_points.txt")
            with open(points_file, 'w') as f:
                for idx, points in enumerate(self.points_list):
                    if len(points) != 3 or not check_collinearity(points):  
                        continue
                    basename = os.path.splitext(os.path.basename(self.dng_files[idx]))[0]
                    f.write(f"{basename}\n")
                    for pt in points:
                        f.write(f"{pt[0]:.2f} {pt[1]:.2f}\n")
                    f.write("\n")  
            print(f"Original points (including ref) saved to {points_file}")
        if out_dir and transformed_points is not None:
            points_file = os.path.join(out_dir, "transformed_points.txt")
            with open(points_file, 'w') as f:
                for idx in transformed_points:
                    if transformed_points[idx] is None or len(transformed_points[idx]) != 3:
                        continue
                    basename = os.path.splitext(os.path.basename(self.dng_files[idx]))[0] + "_aligned"
                    f.write(f"{basename}\n")
                    for pt in transformed_points[idx]:
                        f.write(f"{pt[0]:.2f} {pt[1]:.2f}\n")
                    f.write("\n")  
            print(f"Transformed points saved to {points_file}")

    def load_directory(self):
        directory = filedialog.askdirectory(title="Select DNG Files Directory")
        if not directory:
            self.update_status("Load cancelled.")
            return
        self.reset_data()
        self.input_dir = directory  
        self.progress_var.set(0)
        self.update_status("Scanning directory...")
        self.root.update_idletasks()
        dng_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.dng')]
        dng_files.sort(key=lambda x: os.path.basename(x).lower())
        if not dng_files:
            messagebox.showerror("Error", "No DNG files found in directory.")
            self.update_status("Ready")
            return
        self.finalize_loading(dng_files)

    def finalize_loading(self, dng_files):
        total_files = len(dng_files)
        self.dng_files = dng_files
        self.display_pil_images = [None] * total_files
        self.points_list = [[] for _ in dng_files]
        self.current_idx = 0
        self.ref_idx = None
        self.ref_label.config(text="None selected")
        self.set_ref_var.set(False)
        self.raw_cache = [None] * total_files  
        self.display_scale = 0.5  
        loaded_images, total_points = self.load_points_from_file(self.input_dir)
        load_msg = f"Loaded {loaded_images} images with {total_points} points from file." if loaded_images > 0 else "No points file found."
        print(load_msg)
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

    def _apply_layout_after_scale(self, scaled_w, scaled_h, do_center=False):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if do_center:
            self.offset_x = max((canvas_w - scaled_w) / 2.0, 0.0)
            self.offset_y = max((canvas_h - scaled_h) / 2.0, 0.0)
            if self.image_item is not None:
                self.canvas.coords(self.image_item, self.offset_x, self.offset_y)
        total_w = max(scaled_w, canvas_w)
        total_h = max(scaled_h, canvas_h)
        self.canvas.config(scrollregion=(0, 0, total_w, total_h))

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
        self.title_label.config(text=os.path.basename(self.dng_files[self.current_idx]))
        self.current_label.config(text=f"Image {self.current_idx + 1}/{len(self.dng_files)}")
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
        if is_new_image:
            if not self.initial_fit_done:
                fit_scale = min((canvas_w * 0.9 / img_w), (canvas_h * 0.9 / img_h))
                self.zoom_factor = fit_scale
                self.initial_fit_done = True
                print(f"Initial auto-fit: scale={self.zoom_factor:.3f}")
        scaled_width = int(img_w * self.zoom_factor)
        scaled_height = int(img_h * self.zoom_factor)
        if scaled_width > 0 and scaled_height > 0:
            scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(scaled_img)
            self.image_item = self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor=tk.NW, tags="image")
        self.current_scaled_w = float(scaled_width)
        self.current_scaled_h = float(scaled_height)
        do_center = is_new_image and not self.initial_fit_done
        self._apply_layout_after_scale(scaled_width, scaled_height, do_center=do_center)
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
            else:
                self.image_item = self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor=tk.NW, tags="image")
        self.current_scaled_w = float(scaled_width)
        self.current_scaled_h = float(scaled_height)
        self._apply_layout_after_scale(scaled_width, scaled_height, do_center=False)
        self.redraw_points()

    def on_canvas_resize(self, event):
        if event.widget != self.canvas or event.width < 1 or event.height < 1:
            return
        self.update_scaled_image()

    def on_mousewheel(self, event):
        self.zoom_image(event)

    def zoom_image(self, event):
        if self.image_item is None or self.current_scaled_w <= 0:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        mouse_canvas_x = event.x
        mouse_canvas_y = event.y
        mouse_content_x = self.canvas.canvasx(mouse_canvas_x)
        mouse_content_y = self.canvas.canvasy(mouse_canvas_y)
        mouse_img_x = (mouse_content_x - self.offset_x) / self.zoom_factor
        mouse_img_y = (mouse_content_y - self.offset_y) / self.zoom_factor
        old_zoom = self.zoom_factor
        factor = 1.2 if event.delta > 0 else 1 / 1.2
        new_zoom = old_zoom * factor
        self.zoom_factor = max(0.1, min(10.0, new_zoom))
        self.update_scaled_image()
        scaled_w = self.current_scaled_w
        scaled_h = self.current_scaled_h
        total_w = max(scaled_w, canvas_w)
        total_h = max(scaled_h, canvas_h)
        new_mouse_content_x = self.offset_x + mouse_img_x * self.zoom_factor
        new_mouse_content_y = self.offset_y + mouse_img_y * self.zoom_factor
        new_visible_left = new_mouse_content_x - mouse_canvas_x
        new_visible_top = new_mouse_content_y - mouse_canvas_y
        x_scrollable = total_w - canvas_w
        y_scrollable = total_h - canvas_h
        if x_scrollable > 0:
            x_frac = max(0.0, min(1.0, new_visible_left / x_scrollable))
            self.canvas.xview_moveto(x_frac)
        if y_scrollable > 0:
            y_frac = max(0.0, min(1.0, new_visible_top / y_scrollable))
            self.canvas.yview_moveto(y_frac)

    def start_pan(self, event):
        self.is_panning = True
        self.canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        if self.is_panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def stop_pan(self, event):
        if self.is_panning:
            self.is_panning = False

    def redraw_points(self):
        self.canvas.delete("point")
        if not self.image_item or self.current_idx >= len(self.points_list):
            return
        points = self.points_list[self.current_idx]  
        if len(points) == 0:
            return
        arm_length = 8.0
        text_offset = 12.0
        for i, pt in enumerate(points):
            if len(pt) == 2:
                display_pt_x = pt[0] * self.display_scale
                display_pt_y = pt[1] * self.display_scale
                content_x = self.offset_x + display_pt_x * self.zoom_factor
                content_y = self.offset_y + display_pt_y * self.zoom_factor
                self.canvas.create_line(content_x - arm_length, content_y, content_x + arm_length, content_y, fill="lime", width=1, tags="point")
                self.canvas.create_line(content_x, content_y - arm_length, content_x, content_y + arm_length, fill="lime", width=1, tags="point")
                self.canvas.create_text(content_x + text_offset, content_y, text=f"P{i+1}", fill="lime", anchor=tk.W, tags="point", font=("Arial", 10, "bold"))
        self.save_points_to_file(self.input_dir)

    def update_indicators(self):
        if self.current_idx >= len(self.points_list):
            return
        points = self.points_list[self.current_idx]
        is_valid = len(points) == 3 and check_collinearity(points)
        for i in range(3):
            if len(points) > i:
                self.indicators[i].config(bg="green" if is_valid else "yellow")  
            else:
                self.indicators[i].config(bg="red")

    def on_click(self, event):
        if self.current_idx >= len(self.dng_files) or self.display_pil_images[self.current_idx] is None or self.current_scaled_w <= 0:
            return
        img_w = self.current_img_w  
        img_h = self.current_img_h
        if img_w <= 0 or img_h <= 0:
            return
        mouse_content_x = self.canvas.canvasx(event.x)
        mouse_content_y = self.canvas.canvasy(event.y)
        rel_x = mouse_content_x - self.offset_x
        rel_y = mouse_content_y - self.offset_y
        if 0 <= rel_x <= self.current_scaled_w and 0 <= rel_y <= self.current_scaled_h:
            display_orig_x = rel_x / self.zoom_factor
            display_orig_y = rel_y / self.zoom_factor
            display_orig_x = max(0.0, min(display_orig_x, img_w - 1))
            display_orig_y = max(0.0, min(display_orig_y, img_h - 1))
            full_orig_x = display_orig_x / self.display_scale  
            full_orig_y = display_orig_y / self.display_scale
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
            else:
                display_points = [[p[0] * self.display_scale, p[1] * self.display_scale] for p in points]
                distances = [math.sqrt((display_orig_x - dp[0])**2 + (display_orig_y - dp[1])**2) for dp in display_points]
                min_dist = min(distances)
                if min_dist < 15.0 / self.zoom_factor:  
                    self.dragged_point_idx = distances.index(min_dist)
                else:
                    self.dragged_point_idx = None
        else:
            self.dragged_point_idx = None

    def on_point_drag(self, event):
        if self.dragged_point_idx is None or self.current_idx >= len(self.dng_files) or self.display_pil_images[self.current_idx] is None or self.current_scaled_w <= 0:
            return
        img_w = self.current_img_w  
        img_h = self.current_img_h
        if img_w <= 0 or img_h <= 0:
            return
        mouse_content_x = self.canvas.canvasx(event.x)
        mouse_content_y = self.canvas.canvasy(event.y)
        display_new_x = (mouse_content_x - self.offset_x) / self.zoom_factor
        display_new_y = (mouse_content_y - self.offset_y) / self.zoom_factor
        display_new_x = max(0.0, min(display_new_x, img_w - 1))
        display_new_y = max(0.0, min(display_new_y, img_h - 1))
        full_new_x = display_new_x / self.display_scale
        full_new_y = display_new_y / self.display_scale
        points = self.points_list[self.current_idx]
        points[self.dragged_point_idx] = [full_new_x, full_new_y]
        self.redraw_points()  

    def on_point_release(self, event):
        if self.dragged_point_idx is not None:
            self.dragged_point_idx = None

    def prev_image(self):
        if self.current_idx > 0:
            self.save_points_to_file(self.input_dir)
            self.current_idx -= 1
            self.show_current_image()

    def next_image(self):
        if len(self.dng_files) > 0 and self.current_idx < len(self.dng_files) - 1:
            self.save_points_to_file(self.input_dir)
            self.current_idx += 1
            self.show_current_image()

    def align_and_save(self):
        if not self.dng_files:
            messagebox.showerror("Error", "No images loaded.")
            return
        if self.ref_idx is None:
            messagebox.showerror("Error", "No reference image selected.")
            return
        ref_filepath = self.dng_files[self.ref_idx]
        ref_points = np.array(self.points_list[self.ref_idx], dtype=np.float32)
        if len(ref_points) != 3 or not check_collinearity(ref_points):
            messagebox.showerror("Error", "Reference image must have exactly 3 non-collinear points.")
            return
        outdir = filedialog.askdirectory(title="Select Output Directory for Aligned FITS")
        if not outdir:
            self.update_status("Save cancelled.")
            return
        os.makedirs(outdir, exist_ok=True)
        self.update_status("Starting alignment process...")
        self.progress_var.set(0)
        self.root.update_idletasks()
        try:
            with rawpy.imread(ref_filepath) as raw:
                ref_color = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1, 1))
                ref_dates = extract_dng_dates(raw)  
                ref_mono = cv2.cvtColor(ref_color, cv2.COLOR_RGB2GRAY)
                print(f"Loaded reference image: {os.path.basename(ref_filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reference image: {e}")
            self.update_status("Ready.")
            return
        ref_size = (ref_mono.shape[1], ref_mono.shape[0])  
        transformed_points = {i: None for i in range(len(self.dng_files))}
        processed_count = 0
        skipped_count = 0
        total_files = len(self.dng_files)
        for i, filepath in enumerate(self.dng_files):
            src_points = np.array(self.points_list[i], dtype=np.float32)
            if len(src_points) != 3 or not check_collinearity(src_points):
                print(f"Skipping image {i} due to invalid or collinear points.")
                skipped_count += 1
                self.progress_var.set((i + 1) / total_files * 100)
                self.root.update_idletasks()
                continue
            try:
                with rawpy.imread(filepath) as raw:
                    src_color = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1, 1))
                    src_dates = extract_dng_dates(raw)  
                    src_mono = cv2.cvtColor(src_color, cv2.COLOR_RGB2GRAY)
            except Exception as e:
                print(f"Skipping image {i} due to load failure: {e}")
                skipped_count += 1
                self.progress_var.set((i + 1) / total_files * 100)
                self.root.update_idletasks()
                continue
            if i == self.ref_idx:
                M = np.eye(2, 3, dtype=np.float32)
            else:
                try:
                    M, _ = compute_transform(src_points, ref_points, mode=self.transform_mode.get())
                    if M is None: raise ValueError("Transform computation failed.")
                except Exception as e:
                    print(f"Skipping image {i} due to transform computation error: {e}")
                    skipped_count += 1
                    self.progress_var.set((i + 1) / total_files * 100)
                    self.root.update_idletasks()
                    continue
            transformed_pts = transform_points(M, self.points_list[i])
            transformed_points[i] = transformed_pts  
            aligned_color = cv2.warpAffine(src_color, M, ref_size,
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0))
            fits_data = np.moveaxis(aligned_color, 2, 0).astype(np.float32)
            filename = os.path.join(outdir, f"aligned_{i+1:05d}.fit")
            try:
                header = fits.Header()
                exif_date_obs = extract_date_from_exif(filepath)
                if exif_date_obs:
                    header['DATE-OBS'] = exif_date_obs
                if src_dates:
                    if src_dates.get('CREATED') and 'CREATED' not in header:
                        header['CREATED'] = src_dates['CREATED']
                    if not exif_date_obs and src_dates.get('DATE-OBS'):
                        header['DATE-OBS'] = src_dates['DATE-OBS']
                if 'DATE-OBS' not in header:
                    print(f"No DATE-OBS found for image {i}; header incomplete.")
                hdu = fits.PrimaryHDU(data=fits_data, header=header)
                hdu.writeto(filename, overwrite=True)
                processed_count += 1
                print(f"Successfully saved: {filename}")
            except Exception as e:
                print(f"Failed to save FITS file {filename}: {e}")
                skipped_count += 1
            self.progress_var.set((i + 1) / total_files * 100)
            self.root.update_idletasks()
            del src_color, src_mono, aligned_color, fits_data
            gc.collect()
        self.save_points_to_file(self.input_dir)
        self.save_points_to_file(outdir, transformed_points)
        del ref_color, ref_mono
        gc.collect()
        message = f"Alignment complete!\n\nProcessed and saved: {processed_count} images.\nSkipped: {skipped_count} images.\nPoints files saved (originals & transformed)."
        messagebox.showinfo("Success", message)
        self.update_status("Ready.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AstroAligner()
    app.run()
