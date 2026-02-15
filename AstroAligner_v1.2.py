import os
import sys
import json
import threading
from queue import PriorityQueue
from datetime import datetime

from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                            QLabel, QFileDialog, QProgressBar, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                            QMessageBox, QGraphicsLineItem, QGraphicsSimpleTextItem, QSlider, QCheckBox, QComboBox,
                            QGraphicsEllipseItem)
from PyQt6.QtGui import (QPixmap, QImage, QCursor, QPen, QFont, QPainter, QPainterPath, QColor)
from PyQt6.QtCore import (Qt, QRectF, QPoint, QPointF, QSettings, QTimer, pyqtSignal, QObject)

import numpy as np
import cv2

try:
    from astropy.io import fits
except ImportError:
    fits = None

try:
    import rawpy
except ImportError:
    rawpy = None
    
try:
    import exifread
except ImportError:
    exifread = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
FITS_EXT = {".fits", ".fit"}
RAW_EXTS = {".nef", ".cr2", ".cr3", ".crw", ".arw", ".raf", ".dng", ".mrw", ".orf", ".rw2", ".pef", ".x3f"}

ALL_EXTS = IMAGE_EXTS | FITS_EXT | RAW_EXTS


class LoaderSignals(QObject):
    pixmap_ready = pyqtSignal(int, object)


class ImageLoaderThread(threading.Thread):
    
    def __init__(self, queue, signals, viewer):
        super().__init__(daemon=True)
        self.queue = queue
        self.signals = signals
        self.viewer = viewer
        self.running = True

    def run(self):
        while self.running:
            try:
                priority, task_id, idx, path, stretch, black = self.queue.get(timeout=0.2)
                
                if idx is None:
                    break
                
                arr = self.viewer.load_image_data(path)
                if arr is None:
                    self.queue.task_done()
                    continue
                
                self.viewer.raw_cache[idx] = arr
                
                pixmap = self.viewer._array_to_pixmap_with_params(arr, stretch, black)
                if pixmap is not None:
                    self.signals.pixmap_ready.emit(idx, pixmap)
                
                self.queue.task_done()
            except:
                pass

    def stop(self):
        self.running = False


class MagnifierOverlay(QWidget):

    def __init__(self, viewer, parent=None):
        super().__init__(parent or viewer.viewport())
        self.viewer = viewer
        self.radius = 110
        self.zoom = 2.9
        self.scene_pos = None

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(self.diameter, self.diameter)

    @property
    def diameter(self) -> int:
        return int(self.radius * 2)

    def set_center_scene_pos(self, scene_pos: QPointF):
        self.scene_pos = scene_pos
        self.update()

    def paintEvent(self, event):
        if self.scene_pos is None:
            return

        scene = self.viewer._scene
        if scene is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

        src_radius = self.radius / self.zoom
        sx = self.scene_pos.x() - src_radius
        sy = self.scene_pos.y() - src_radius
        sw = src_radius * 2
        sh = src_radius * 2
        source_rect = QRectF(sx, sy, sw, sh)
        if source_rect.isEmpty():
            return

        target_rect = QRectF(0, 0, self.diameter, self.diameter)

        path = QPainterPath()
        path.addEllipse(target_rect)
        painter.setClipPath(path)

        scene.render(painter, target_rect, source_rect)

        painter.setClipping(False)

        pen = QPen(QColor(0, 255, 0, 180), 2)
        painter.setPen(pen)
        painter.drawEllipse(target_rect.adjusted(0.5, 0.5, -0.5, -0.5))


class PhotoViewer(QGraphicsView):
    pointClicked = pyqtSignal(QPointF)
    pointDrag = pyqtSignal(QPointF, QPoint)
    pointRelease = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._photo.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setFrameStyle(0)

        self._empty = True
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._last_pos = QPoint()
        self.setCursor(Qt.CursorShape.ArrowCursor)

        self.fit_scale = 1.0
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.magnifier = MagnifierOverlay(self, self.viewport())
        self.magnifier.hide()

    def hasPhoto(self):
        return not self._empty

    def get_relative_zoom(self):
        return self.transform().m11() / self.fit_scale if self.fit_scale > 0 else 1.0

    def get_relative_pan(self):
        scene_rect = self._scene.sceneRect()
        if scene_rect.isEmpty() or scene_rect.width() == 0 or scene_rect.height() == 0:
            return 0.5, 0.5

        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()

        rel_x = hbar.value() / max(1, hbar.maximum()) if hbar.maximum() > 0 else 0.5
        rel_y = vbar.value() / max(1, vbar.maximum()) if vbar.maximum() > 0 else 0.5
        return rel_x, rel_y

    def setPhoto(self, pixmap=None, rel_zoom=1.0, rel_x=0.5, rel_y=0.5):
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)

            rect = pixmap.rect()
            w, h = rect.width(), rect.height()

            pad_x, pad_y = w * 0.1, h * 0.1
            padded_rect = QRectF(rect).adjusted(-pad_x, -pad_y, pad_x, pad_y)
            self._scene.setSceneRect(padded_rect)

            view_w = self.viewport().width()
            view_h = self.viewport().height()
            new_fit_scale = min(view_w / w, view_h / h) if w > 0 and h > 0 else 1.0
            self.fit_scale = new_fit_scale

            self.resetTransform()
            target_scale = new_fit_scale * rel_zoom
            self.scale(target_scale, target_scale)

            QApplication.processEvents()

            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            new_hbar_max = hbar.maximum()
            new_vbar_max = vbar.maximum()
            target_h_scroll = int(rel_x * max(0, new_hbar_max))
            target_v_scroll = int(rel_y * max(0, new_vbar_max))
            hbar.setValue(target_h_scroll)
            vbar.setValue(target_v_scroll)

            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self._empty = True
            self._photo.setPixmap(QPixmap())
            self.resetTransform()
            self.fit_scale = 1.0
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        if not self.hasPhoto():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.hasPhoto():
            vp_pos = event.position().toPoint()
            self._update_magnifier(vp_pos)

            scene_pos = self.mapToScene(vp_pos)
            self.pointClicked.emit(scene_pos)
            event.accept()
            return

        elif event.button() == Qt.MouseButton.MiddleButton and self.hasPhoto():
            self._last_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            event.buttons() == Qt.MouseButton.MiddleButton
            and self.hasPhoto()
            and not self._last_pos.isNull()
        ):
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            delta_x = self._last_pos.x() - event.position().toPoint().x()
            delta_y = self._last_pos.y() - event.position().toPoint().y()
            hbar.setValue(hbar.value() + delta_x)
            vbar.setValue(vbar.value() + delta_y)
            self._last_pos = event.position().toPoint()
            event.accept()
            return

        elif event.buttons() == Qt.MouseButton.LeftButton and self.hasPhoto():
            vp_pos = event.position().toPoint()
            scene_pos = self.mapToScene(vp_pos)
            self.pointDrag.emit(scene_pos, vp_pos)
            event.accept()
            return

        else:
            self.magnifier.hide()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.hasPhoto():
            self.magnifier.hide()
            scene_pos = self.mapToScene(event.position().toPoint())
            self.pointRelease.emit(scene_pos)
            event.accept()
            return

        elif event.button() == Qt.MouseButton.MiddleButton:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._last_pos = QPoint()
            event.accept()
            return

        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.magnifier.hide()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            event.ignore()
            return
        super().keyPressEvent(event)

    def _update_magnifier(self, viewport_pos: QPoint):
        if not self.hasPhoto():
            self.magnifier.hide()
            return

        scene_pos = self.mapToScene(viewport_pos)
        self.magnifier.set_center_scene_pos(scene_pos)

        d = self.magnifier.diameter
        offset = 20
        x = viewport_pos.x() + offset
        y = viewport_pos.y() + offset

        max_x = self.viewport().width() - d
        max_y = self.viewport().height() - d
        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))

        self.magnifier.move(x, y)
        self.magnifier.show()
        self.magnifier.update()


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = QSettings("AstroAligner", "ImageViewer")
        self.setWindowTitle("AstroAligner")
        self.resize(1200, 800)

        self.last_dir = self.settings.value("last_directory", os.getcwd())
        self.current_index = 0
        self.file_paths = []

        self.cache = {}
        self.raw_cache = {}
        self.max_cache_size = 20

        self.stretch_factor = 1.0
        self.black_level = 0.0
        
        self.monochrome = True
        
        self.reduce_factor = 1.0

        self.reference_index = None
        self.align_method = "rotate_shift"

        self.align_file = None

        self.alignment_store = {}
        self.reference_store = {}
        
        self.load_alignment_store()
        self.load_reference_store()

        print(f"DEBUG: Alignment file init: {self.align_file}")
        print(f"DEBUG: Reference store init (in-memory): {self.reference_store}")

        self.align_points_items = []
        self.points_count = 0
        self.drag_point_index = None
        self.dragging = False

        self.drag_start_scene = None
        self.drag_start_img_pos = None
        self.drag_slow_factor_fine = 0.05
        self.drag_slow_factor_normal = 1.0

        self.cross_size = 80.0

        self.loader_signals = LoaderSignals()
        self.loader_signals.pixmap_ready.connect(self.on_pixmap_ready)
        self.loader_queue = PriorityQueue()
        self.loader_thread = ImageLoaderThread(self.loader_queue, self.loader_signals, self)
        self.loader_thread.start()
        
        self.queued_indices = set()
        self.task_counter = 0

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        top_layout.addWidget(self.progress_bar)

        main_layout.addWidget(top_panel)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.load_button = QPushButton("Load working directory")
        self.load_button.clicked.connect(self.load_directory)
        left_layout.addWidget(self.load_button)

        # НОВОЕ: Align type (3 Point / Comet only)
        self.align_type = "3point"  # значение по умолчанию

        align_type_label = QLabel("Align type")
        left_layout.addWidget(align_type_label)

        self.align_type_combo = QComboBox()
        self.align_type_combo.addItem("3 Point", userData="3point")
        self.align_type_combo.addItem("Comet only", userData="comet")
        self.align_type_combo.setCurrentIndex(0)
        self.align_type_combo.currentIndexChanged.connect(self.on_align_type_changed)
        left_layout.addWidget(self.align_type_combo)

        self.current_label = QLabel("No image loaded\n0/0")
        self.current_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.current_label)

        self.align_indicators = []
        for text in ["Align point 1", "Align point 2", "Align point 3"]:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            circle = QLabel()
            circle.setFixedSize(14, 14)
            circle.setStyleSheet(
                "background-color: red;"
                "border-radius: 7px;"
                "border: 1px solid #550000;"
            )

            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            row_layout.addWidget(circle)
            row_layout.addWidget(label)
            left_layout.addWidget(row, 0, Qt.AlignmentFlag.AlignHCenter)

            self.align_indicators.append(circle)

        self.clear_points_button = QPushButton("Clear points")
        self.clear_points_button.clicked.connect(self.on_clear_points_clicked)
        left_layout.addWidget(self.clear_points_button)

        stretch_label = QLabel("Stretch factor")
        left_layout.addWidget(stretch_label)

        self.stretch_slider = QSlider(Qt.Orientation.Horizontal)
        self.stretch_slider.setRange(10, 500)
        self.stretch_slider.setValue(100)
        self.stretch_slider.sliderMoved.connect(self.on_stretch_slider_changed)
        left_layout.addWidget(self.stretch_slider)

        black_label = QLabel("Black points")
        left_layout.addWidget(black_label)
        self.black_slider = QSlider(Qt.Orientation.Horizontal)
        self.black_slider.setRange(0, 100)
        self.black_slider.setValue(0)
        self.black_slider.sliderMoved.connect(self.on_black_slider_changed)
        left_layout.addWidget(self.black_slider)
        
        self.monochrome_checkbox = QCheckBox("Monochrome")
        self.monochrome_checkbox.setChecked(True)
        self.monochrome_checkbox.stateChanged.connect(self.on_monochrome_changed)
        left_layout.addWidget(self.monochrome_checkbox)

        self.apply_global_button = QPushButton("Apply global")
        self.apply_global_button.clicked.connect(self.on_apply_global_clicked)
        left_layout.addWidget(self.apply_global_button)
        
        reduce_label = QLabel("Reduce display resolution")
        left_layout.addWidget(reduce_label)
        self.reduce_combo = QComboBox()
        self.reduce_combo.addItem("1x")
        self.reduce_combo.addItem("2x")
        self.reduce_combo.addItem("4x")
        self.reduce_combo.addItem("6x")
        self.reduce_combo.setCurrentIndex(0)
        self.reduce_combo.currentTextChanged.connect(self.on_reduce_resolution_changed)
        left_layout.addWidget(self.reduce_combo)

        self.reference_checkbox = QCheckBox("Reference image")
        self.reference_checkbox.stateChanged.connect(self.on_reference_checkbox_changed)
        left_layout.addWidget(self.reference_checkbox)

        align_label = QLabel("Align method")
        left_layout.addWidget(align_label)
        self.align_method_combo = QComboBox()
        self.align_method_combo.addItem("Euclidean", userData="rotate_shift")
        self.align_method_combo.addItem(
            "Affine", userData="rotate_shift_scale"
        )
        self.align_method_combo.currentIndexChanged.connect(
            self.on_align_method_changed
        )
        left_layout.addWidget(self.align_method_combo)

        self.align_button = QPushButton("Align")
        self.align_button.clicked.connect(self.on_align_clicked)
        left_layout.addWidget(self.align_button)

        left_layout.addStretch()

        content_layout.addWidget(left_panel, 1)

        self.viewer = PhotoViewer(self)
        self.viewer.pointClicked.connect(self.on_point_clicked)
        self.viewer.pointDrag.connect(self.on_point_drag)
        self.viewer.pointRelease.connect(self.on_point_release)
        content_layout.addWidget(self.viewer, 3)

        self.current_dir = None

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2,
        )

        self.viewer.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        self.update_align_indicators()

    def on_pixmap_ready(self, idx: int, pixmap):
        """Обработчик готового pixmap'а из потока."""
        self.cache[idx] = pixmap
        
        if idx == self.current_index:
            if self.viewer.hasPhoto():
                rel_zoom = self.viewer.get_relative_zoom()
                rel_x, rel_y = self.viewer.get_relative_pan()
            else:
                rel_zoom = 1.0
                rel_x = rel_y = 0.5
            
            self.viewer.setPhoto(pixmap, rel_zoom=rel_zoom, rel_x=rel_x, rel_y=rel_y)

    def get_display_scale(self) -> float:
        """
        Масштаб уменьшения для отображения.
        1.0 - без уменьшения, 2.0 - 2x меньше по каждой оси и т.п.
        """
        factor = getattr(self, "reduce_factor", 1.0)
        if factor is None or factor <= 0:
            factor = 1.0
        return float(factor)
        
    def get_observation_date_for_file(self, img_path: str):
        """
        Вернуть дату наблюдения для файла img_path в формате 'YYYY-MM-DDTHH:MM:SS'.

        Приоритет:
        1) Для FITS — берём DATE-OBS / Observation Date / DATE из заголовка.
        2) Для RAW/JPEG/PNG/TIFF и т.п. — EXIF DateTimeOriginal.
        3) Если EXIF нет — время файла (mtime) из файловой системы.
        """
        ext = os.path.splitext(img_path)[1].lower()

        # 1. FITS: попробовать вытащить уже существующую дату из заголовка
        if ext in FITS_EXT and fits is not None:
            try:
                hdr = fits.getheader(img_path)
                for key in ("Observation Date", "DATE-OBS", "DATE"):
                    val = hdr.get(key)
                    if val:
                        # здесь предполагаем, что в заголовке уже подходящий формат
                        return str(val)
            except Exception as e:
                print(f"Не удалось прочитать FITS header {img_path}: {e}")

        # 2. EXIF DateTimeOriginal для RAW / JPEG / др.
        if exifread is not None:
            try:
                with open(img_path, "rb") as f:
                    tags = exifread.process_file(
                        f,
                        stop_tag="EXIF DateTimeOriginal",
                        details=False,
                    )
                dt_tag = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
                if dt_tag:
                    # ожидаемый формат: 'YYYY:MM:DD HH:MM:SS'
                    s = str(dt_tag).strip()
                    try:
                        dt_obj = datetime.strptime(s, "%Y:%m:%d %H:%M:%S")  # EXIF‑формат[web:30]
                        return dt_obj.strftime("%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        # На крайний случай: грубое преобразование, если формат чуть другой
                        # первые два ':' меняем на '-', пробел на 'T'
                        return s.replace(" ", "T").replace(":", "-", 2)
            except Exception as e:
                print(f"Не удалось прочитать EXIF из {img_path}: {e}")

        # 3. Fallback — время файла из ОС
        try:
            ts = os.path.getmtime(img_path)
            dt_obj = datetime.fromtimestamp(ts)
            return dt_obj.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as e:
            print(f"Не удалось получить время файла {img_path}: {e}")
            return None

    def img_to_display(self, p: QPointF) -> QPointF:
        """
        Перевод координат из полного разрешения (array) в координаты pixmap.
        """
        s = self.get_display_scale()
        return QPointF(p.x() / s, p.y() / s)

    def display_to_img(self, p: QPointF) -> QPointF:
        """
        Перевод координат из координат pixmap в полное разрешение (array).
        """
        s = self.get_display_scale()
        return QPointF(p.x() * s, p.y() * s)
        
    def queue_image_load(self, idx: int, window_size: int = 3):
        if not self.file_paths or idx < 0 or idx >= len(self.file_paths):
            return
        
        path = self.file_paths[idx]
        
        if idx in self.cache and idx in self.raw_cache:
            return
        
        if idx in self.queued_indices:
            return
        
        priority = abs(idx - self.current_index)
        
        self.task_counter += 1
        task_id = self.task_counter
        
        self.loader_queue.put((
            priority,
            task_id,
            idx,
            path,
            self.stretch_factor,
            self.black_level
        ))
        
        self.queued_indices.add(idx)

    def refresh_queue_for_position(self, window_size: int = 3):
        for offset in range(-window_size, window_size + 1):
            idx = self.current_index + offset
            if 0 <= idx < len(self.file_paths):
                self.queue_image_load(idx, window_size)
        
        if len(self.cache) > self.max_cache_size:
            distances = [(abs(idx - self.current_index), idx) for idx in list(self.cache.keys())]
            distances.sort()
            to_keep = [idx for _, idx in distances[: self.max_cache_size]]
            to_remove = set(self.cache.keys()) - set(to_keep)
            for idx in to_remove:
                self.cache.pop(idx, None)
                self.raw_cache.pop(idx, None)

    def load_reference_store(self):
        self.reference_store = {}

    def save_current_reference(self):
        if self.current_dir is None or not self.file_paths:
            return

        if self.reference_index is not None and 0 <= self.reference_index < len(self.file_paths):
            self.reference_store[self.current_dir] = {
                "index": self.reference_index,
                "file_path": self.file_paths[self.reference_index]
            }
        else:
            self.reference_store.pop(self.current_dir, None)

    def load_current_reference(self):
        if self.current_dir is None:
            self.reference_index = None
            return

        if self.current_dir in self.reference_store:
            ref_data = self.reference_store[self.current_dir]
            idx = ref_data.get("index")
            if isinstance(idx, int) and 0 <= idx < len(self.file_paths):
                self.reference_index = idx
            else:
                self.reference_index = None
        else:
            self.reference_index = None

    def on_reference_checkbox_changed(self, state: int):
   
        if not self.file_paths:
            self.reference_index = None
            self.update_reference_checkbox_text()
            return

        if state == Qt.CheckState.Checked.value or state == 2:
            self.reference_index = self.current_index
        else:
            if self.reference_index == self.current_index:
                self.reference_index = None

        self.save_current_reference()
        
        self.update_reference_checkbox_text()
        self.update_reference_ui_state()

    def sync_reference_checkbox(self):
        self.reference_checkbox.blockSignals(True)
        is_current_reference = (self.reference_index == self.current_index)
        self.reference_checkbox.setChecked(is_current_reference)
        self.reference_checkbox.blockSignals(False)

    def update_reference_checkbox_text(self):
        if self.reference_index is not None and 0 <= self.reference_index < len(self.file_paths):
            text = f"Reference image (#{self.reference_index + 1})"
        else:
            text = "Reference image"
        
        self.reference_checkbox.setText(text)

    def update_reference_ui_state(self):
        if self.reference_index is not None and 0 <= self.reference_index < len(self.file_paths):
            ref_name = os.path.basename(self.file_paths[self.reference_index])
            self.status_label.setText(f"Reference: {ref_name}")
        else:
            if self.file_paths:
                self.status_label.setText("Ready")
            else:
                self.status_label.setText("Ready")
                
    def on_align_type_changed(self, index: int):
        data = self.align_type_combo.itemData(index)
        if data in ("3point", "comet"):
            self.align_type = data
        else:
            self.align_type = "3point"

    def on_align_method_changed(self, index: int):
        data = self.align_method_combo.itemData(index)
        if data in ("rotate_shift", "rotate_shift_scale"):
            self.align_method = data
        else:
            self.align_method = "rotate_shift"

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left and self.current_index > 0:
            self.previous_image()
            event.accept()
            return
        elif (
            event.key() == Qt.Key.Key_Right
            and self.current_index < len(self.file_paths) - 1
        ):
            self.next_image()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.save_current_image_alignment()
        self.save_current_reference()
        self.loader_queue.put((999, 0, None, "", 0, 0))
        self.loader_thread.stop()
        super().closeEvent(event)

    def load_alignment_store(self):
        if not self.align_file or not os.path.exists(self.align_file):
            self.alignment_store = {}
            return
        try:
            with open(self.align_file, "r", encoding="utf-8") as f:
                data = f.read().strip()
                if not data:
                    self.alignment_store = {}
                else:
                    self.alignment_store = json.loads(data)
        except Exception:
            self.alignment_store = {}

    def save_alignment_store(self):
        if not self.align_file:
            return
        try:
            with open(self.align_file, "w", encoding="utf-8") as f:
                json.dump(self.alignment_store, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Не удалось сохранить alignment_points.txt: {e}")

    def get_image_key(self, path: str) -> str:
        if self.current_dir:
            try:
                return os.path.relpath(path, self.current_dir)
            except Exception:
                pass
        return path

    def save_current_image_alignment(self):
        if not self.file_paths:
            return
        if self.current_dir is None:
            return

        dir_key = self.current_dir
        img_path = self.file_paths[self.current_index]
        img_key = self.get_image_key(img_path)

        if dir_key not in self.alignment_store:
            self.alignment_store[dir_key] = {}

        points_list = []
        for item in self.align_points_items[: self.points_count]:
            img_pos = item["img_pos"]
            points_list.append({"x": img_pos.x(), "y": img_pos.y()})

        if points_list:
            self.alignment_store[dir_key][img_key] = points_list
        else:
            if img_key in self.alignment_store[dir_key]:
                del self.alignment_store[dir_key][img_key]
            if not self.alignment_store[dir_key]:
                del self.alignment_store[dir_key]

        self.save_alignment_store()

    def on_clear_points_clicked(self):
        self.clear_align_markers()

        if not self.file_paths or self.current_dir is None:
            return

        dir_key = self.current_dir
        img_path = self.file_paths[self.current_index]
        img_key = self.get_image_key(img_path)

        if dir_key in self.alignment_store and img_key in self.alignment_store[dir_key]:
            del self.alignment_store[dir_key][img_key]
            if not self.alignment_store[dir_key]:
                del self.alignment_store[dir_key]
            self.save_alignment_store()

        self.status_label.setText("Align points cleared for current image")

    def load_image_alignment(self, img_path: str):
        self.clear_align_markers()
        if self.current_dir is None:
            return

        dir_key = self.current_dir
        if dir_key not in self.alignment_store:
            return

        img_key = self.get_image_key(img_path)
        points_data = self.alignment_store[dir_key].get(img_key)
        if not points_data:
            return

        for pd in points_data[:3]:
            img_pos = QPointF(float(pd["x"]), float(pd["y"]))
            self.add_point_from_image_coords(img_pos)

        self.update_align_indicators()

    def clear_align_markers(self):
        scene = self.viewer._scene
        for item in self.align_points_items:
            for key in (
                "h_outer_left", "h_outer_right", "h_inner",
                "v_outer_top", "v_outer_bottom", "v_inner",
                "circle", "text"
            ):
                obj = item.get(key)
                if obj is not None:
                    scene.removeItem(obj)
        self.align_points_items.clear()
        self.points_count = 0
        self.drag_point_index = None
        self.dragging = False
        self.drag_start_scene = None
        self.drag_start_img_pos = None
        self.update_align_indicators()



    def on_point_clicked(self, scene_pos: QPointF):
        if not self.viewer.hasPhoto():
            return

        # Координаты в системе pixmap
        disp_pos = self.viewer._photo.mapFromScene(scene_pos)
        # Переводим в координаты полного массива
        img_pos = self.display_to_img(disp_pos)

        if self.points_count < 3:
            self.add_point_from_image_coords(img_pos)
            self.drag_point_index = self.points_count - 1
            self.dragging = True
            self.drag_start_scene = scene_pos
            self.drag_start_img_pos = img_pos   # полные координаты
            return

        idx = self.find_nearest_point(scene_pos, max_dist=20.0)
        if idx is not None:
            self.drag_point_index = idx
            self.dragging = True
            self.drag_start_scene = scene_pos
            self.drag_start_img_pos = self.align_points_items[idx]["img_pos"]

    def on_point_drag(self, scene_pos: QPointF, viewport_pos: QPoint):
        if not self.viewer.hasPhoto():
            return

        # Если просто двигаем курсор без активного drag точки — лупа по курсору
        if not self.dragging or self.drag_point_index is None:
            center_scene = scene_pos
            self._update_magnifier_from_viewer(center_scene, viewport_pos)
            return

        if self.drag_start_scene is None or self.drag_start_img_pos is None:
            return

        mods = QApplication.keyboardModifiers()
        fine = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        k = self.drag_slow_factor_fine if fine else self.drag_slow_factor_normal

        idx = self.drag_point_index
        if 0 <= idx < len(self.align_points_items):
            dx_scene = (scene_pos.x() - self.drag_start_scene.x()) * k
            dy_scene = (scene_pos.y() - self.drag_start_scene.y()) * k

            # Стартовая позиция в display-координатах (pixmap)
            start_disp = self.img_to_display(self.drag_start_img_pos)
            p0_scene = self.viewer._photo.mapToScene(start_disp)

            # Новая позиция в сцене
            p1_scene = QPointF(p0_scene.x() + dx_scene, p0_scene.y() + dy_scene)

            # Назад: scene -> display -> img
            new_disp_pos = self.viewer._photo.mapFromScene(p1_scene)
            new_img_pos = self.display_to_img(new_disp_pos)

            self.align_points_items[idx]["img_pos"] = new_img_pos
            self._update_point_graphics(idx)

            self.drag_start_scene = scene_pos
            self.drag_start_img_pos = new_img_pos

            # ВСЕГДА центрируем лупу по координатам точки
            center_scene = self.viewer._photo.mapToScene(self.img_to_display(new_img_pos))
            self._update_magnifier_from_viewer(center_scene, viewport_pos)


    def on_point_release(self, scene_pos: QPointF):
        self.dragging = False
        self.drag_point_index = None
        self.drag_start_scene = None
        self.drag_start_img_pos = None

    def add_point_from_image_coords(self, img_pos: QPointF):
        # img_pos - координаты в полном разрешении (array), а не pixmap
        if self.points_count >= 3:
            return
        if not self.viewer.hasPhoto():
            return

        scene = self.viewer._scene

        size = self.cross_size          # общий размер перекрестия (конец лучей)
        half = size / 2.0

        circle_diam = 12.0              # диаметр кружка (в 2 раза больше, чем было 6)
        circle_r = circle_diam / 2.0

        index = self.points_count

        # Переводим координаты из полного масштаба в координаты pixmap
        disp_pos = self.img_to_display(img_pos)
        scene_pos = self.viewer._photo.mapToScene(disp_pos)
        cx, cy = scene_pos.x(), scene_pos.y()

        # Внешние части перекрестия — яркие линии
        outer_pen = QPen(QColor(0, 255, 0, 230), 2.0)
        outer_pen.setCosmetic(True)

        # Внутренние части, проходящие внутри круга — полупрозрачные
        inner_pen = QPen(QColor(0, 255, 0, 120), 2.0)
        inner_pen.setCosmetic(True)

        # Горизонтальный луч:
        #  - левая внешняя часть
        #  - правая внешняя часть
        #  - внутренняя часть внутри круга
        h_outer_left = QGraphicsLineItem(cx - half, cy, cx - circle_r, cy)
        h_outer_right = QGraphicsLineItem(cx + circle_r, cy, cx + half, cy)
        h_inner = QGraphicsLineItem(cx - circle_r, cy, cx + circle_r, cy)

        # Вертикальный луч:
        #  - верхняя внешняя часть
        #  - нижняя внешняя часть
        #  - внутренняя часть внутри круга
        v_outer_top = QGraphicsLineItem(cx, cy - half, cx, cy - circle_r)
        v_outer_bottom = QGraphicsLineItem(cx, cy + circle_r, cx, cy + half)
        v_inner = QGraphicsLineItem(cx, cy - circle_r, cx, cy + circle_r)

        for item in (h_outer_left, h_outer_right, v_outer_top, v_outer_bottom):
            item.setPen(outer_pen)
            item.setZValue(1000)  # поверх изображения

        for item in (h_inner, v_inner):
            item.setPen(inner_pen)
            item.setZValue(1000)

        # Круг вокруг центра — больше чем раньше, без заливки
        circle_pen = QPen(QColor(0, 255, 0, 200), 2.0)
        circle_pen.setCosmetic(True)
        circle = QGraphicsEllipseItem(
            cx - circle_r, cy - circle_r, circle_diam, circle_diam
        )
        circle.setPen(circle_pen)
        circle.setBrush(QColor(0, 0, 0, 0))  # прозрачная заливка
        circle.setZValue(999)

        # Номер точки — крупнее и немного дальше от центра
        text_item = QGraphicsSimpleTextItem(str(index + 1))
        text_item.setBrush(QColor(0, 255, 0, 240))
        text_font = QFont()
        text_font.setPointSize(30)  # было 14
        text_item.setFont(text_font)
        text_item.setPos(cx + half + 6, cy - half - 4)
        text_item.setZValue(1001)

        # Добавляем в сцену
        scene.addItem(h_outer_left)
        scene.addItem(h_outer_right)
        scene.addItem(h_inner)
        scene.addItem(v_outer_top)
        scene.addItem(v_outer_bottom)
        scene.addItem(v_inner)
        scene.addItem(circle)
        scene.addItem(text_item)

        # Сохраняем все элементы в align_points_items
        self.align_points_items.append({
            "img_pos": img_pos,        # полное разрешение
            "h_outer_left": h_outer_left,
            "h_outer_right": h_outer_right,
            "h_inner": h_inner,
            "v_outer_top": v_outer_top,
            "v_outer_bottom": v_outer_bottom,
            "v_inner": v_inner,
            "circle": circle,
            "text": text_item,
            "circle_diam": circle_diam,
        })
        self.points_count += 1
        self.update_align_indicators()


    def _update_point_graphics(self, index: int):
        if not (0 <= index < len(self.align_points_items)):
            return
        if not self.viewer.hasPhoto():
            return

        item = self.align_points_items[index]
        img_pos = item["img_pos"]  # полное разрешение

        size = self.cross_size
        half = size / 2.0

        circle_diam = item.get("circle_diam", 12.0)
        circle_r = circle_diam / 2.0

        # Перевод в координаты pixmap -> scene
        disp_pos = self.img_to_display(img_pos)
        scene_pos = self.viewer._photo.mapToScene(disp_pos)
        cx, cy = scene_pos.x(), scene_pos.y()

        # Обновляем линии
        item["h_outer_left"].setLine(cx - half, cy, cx - circle_r, cy)
        item["h_outer_right"].setLine(cx + circle_r, cy, cx + half, cy)
        item["h_inner"].setLine(cx - circle_r, cy, cx + circle_r, cy)

        item["v_outer_top"].setLine(cx, cy - half, cx, cy - circle_r)
        item["v_outer_bottom"].setLine(cx, cy + circle_r, cx, cy + half)
        item["v_inner"].setLine(cx, cy - circle_r, cx, cy + circle_r)

        # Обновляем круг
        item["circle"].setRect(cx - circle_r, cy - circle_r, circle_diam, circle_diam)

        # Обновляем позицию числа
        item["text"].setPos(cx + half + 6, cy - half - 4)



    def find_nearest_point(self, scene_pos: QPointF, max_dist: float = 20.0):
        if not self.align_points_items:
            return None
        if not self.viewer.hasPhoto():
            return None

        max_d2 = max_dist * max_dist
        best_idx = None
        best_d2 = max_d2

        for i, item in enumerate(self.align_points_items):
            img_pos = item["img_pos"]                  # полные координаты
            disp_pos = self.img_to_display(img_pos)    # в координаты pixmap
            p_scene = self.viewer._photo.mapToScene(disp_pos)

            dx = p_scene.x() - scene_pos.x()
            dy = p_scene.y() - scene_pos.y()
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best_idx = i

        return best_idx

        

    def update_align_indicators(self):
        if not hasattr(self, "align_indicators") or len(self.align_indicators) < 3:
            return

        for i in range(3):
            if self.points_count == 0:
                self.set_align_point_state(i, "red")
            else:
                if i < self.points_count:
                    self.set_align_point_state(i, "green")
                else:
                    if self.points_count < 3:
                        self.set_align_point_state(i, "yellow")
                    else:
                        self.set_align_point_state(i, "green")

    def set_align_point_state(self, index: int, state: str):
        if not (0 <= index < len(self.align_indicators)):
            return

        colors = {
            "red": "red",
            "yellow": "yellow",
            "green": "lime",
        }
        color = colors.get(state, "red")

        circle = self.align_indicators[index]
        circle.setStyleSheet(
            f"background-color: {color};"
            "border-radius: 7px;"
            "border: 1px solid #555555;"
        )

    def _update_magnifier_from_viewer(self, center_scene: QPointF, viewport_pos: QPoint):
        if not self.viewer.hasPhoto():
            self.viewer.magnifier.hide()
            return

        self.viewer.magnifier.set_center_scene_pos(center_scene)

        d = self.viewer.magnifier.diameter
        offset = 20
        x = viewport_pos.x() + offset
        y = viewport_pos.y() + offset

        max_x = self.viewer.viewport().width() - d
        max_y = self.viewer.viewport().height() - d
        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))

        self.viewer.magnifier.move(x, y)
        self.viewer.magnifier.show()
        self.viewer.magnifier.update()

    def load_directory(self):
        self.status_label.setText("Selecting directory...")
        QApplication.processEvents()

        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите директорию с изображениями",
            self.last_dir,
        )

        if dir_path:
            self.save_current_image_alignment()
            self.save_current_reference()

            self.current_dir = dir_path

            self.align_file = os.path.join(self.current_dir, "alignment_points.txt")
            self.load_alignment_store()
            print(f"DEBUG: Alignment file: {self.align_file}")

            self.file_paths = []
            self.current_index = 0
            self.cache.clear()
            self.raw_cache.clear()
            self.queued_indices.clear()
            self.clear_align_markers()

            self.settings.setValue("last_directory", dir_path)
            self.last_dir = dir_path

            self.status_label.setText("Scanning directory...")
            QApplication.processEvents()

            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in ALL_EXTS:
                        full_path = os.path.join(root, file)
                        self.file_paths.append(full_path)

            total = len(self.file_paths)
            if total > 0:
                self.load_current_reference()
                self.update_label(total)
                self.preload_initial()
            else:
                self.current_label.setText("No image loaded\n0/0")
                self.status_label.setText("No supported files found")

    def preload_initial(self):
        self.status_label.setText("Ready")
        self.load_current_image(preserve=False)
        self.update_reference_ui_state()
        self.update_reference_checkbox_text()
        self.sync_reference_checkbox()
        
        self.refresh_queue_for_position(window_size=3)

    def update_label(self, total):
        if total > 0:
            filename = os.path.basename(self.file_paths[self.current_index])
            self.current_label.setText(
                f"{filename}\n{self.current_index + 1}/{total}"
            )
        else:
            self.current_label.setText("No image loaded\n0/0")

    def previous_image(self):
        if self.current_index > 0:
            self.save_current_image_alignment()
            self.current_index -= 1
            self.update_label(len(self.file_paths))
            self.load_current_image(preserve=True)
            self.update_reference_ui_state()
            self.update_reference_checkbox_text()
            self.sync_reference_checkbox()
            self.refresh_queue_for_position(window_size=3)

    def next_image(self):
        total = len(self.file_paths)
        if self.current_index < total - 1:
            self.save_current_image_alignment()
            self.current_index += 1
            self.update_label(total)
            self.load_current_image(preserve=True)
            self.update_reference_ui_state()
            self.update_reference_checkbox_text()
            self.sync_reference_checkbox()
            self.refresh_queue_for_position(window_size=3)

    def load_current_image(self, preserve=True):
        if not self.file_paths:
            return

        idx = self.current_index
        path = self.file_paths[idx]

        arr = self.raw_cache.get(idx)
        if arr is None:
            arr = self.load_image_data(path)
            if arr is None:
                QMessageBox.warning(
                    self,
                    "Ошибка",
                    f"Не удалось загрузить {os.path.basename(path)}.",
                )
                self.raw_cache.pop(idx, None)
                self.cache.pop(idx, None)
                return
            self.raw_cache[idx] = arr

        pixmap = self.array_to_pixmap(arr)
        if pixmap is None:
            return

        if preserve and self.viewer.hasPhoto():
            rel_zoom = self.viewer.get_relative_zoom()
            rel_x, rel_y = self.viewer.get_relative_pan()
        else:
            rel_zoom = 1.0
            rel_x = rel_y = 0.5

        self.viewer.setPhoto(
            pixmap,
            rel_zoom=rel_zoom,
            rel_x=rel_x,
            rel_y=rel_y,
        )
        self.cache[idx] = pixmap

        self.load_image_alignment(path)

    def on_stretch_slider_changed(self, value: int):
        self.stretch_factor = value / 100.0
        self.update_current_pixmap_only()
        
    def on_monochrome_changed(self, state: int):
        self.monochrome = (state == Qt.CheckState.Checked.value or state == 2)

        if not self.file_paths:
            return

        self.cache.clear()
        self.queued_indices.clear()

        self.update_current_pixmap_only()

        self.refresh_queue_for_position(window_size=3)

    def on_black_slider_changed(self, value: int):
        self.black_level = value / 100.0
        self.update_current_pixmap_only()
        
    def on_reduce_resolution_changed(self, text: str):
        # text вида "1x", "2x", "4x", "6x"
        try:
            factor = int(text.replace("x", "").strip())
        except ValueError:
            factor = 1
        if factor < 1:
            factor = 1

        new_factor = float(factor)
        if new_factor <= 0:
            new_factor = 1.0

        self.reduce_factor = new_factor

        if not self.file_paths:
            return

        # Кэш готовых pixmap зависит от масштаба — очищаем
        self.cache.clear()
        self.queued_indices.clear()

        # Перестроить текущий pixmap с новым уменьшающим фактором
        self.update_current_pixmap_only()

        # Перезаполнить очередь подгрузки окрестности
        self.refresh_queue_for_position(window_size=3)

        # Перерисовать все точки с учётом нового масштаба
        for i in range(len(self.align_points_items)):
            self._update_point_graphics(i)

    def update_current_pixmap_only(self):
        if not self.file_paths:
            return

        idx = self.current_index
        path = self.file_paths[idx]

        arr = self.raw_cache.get(idx)
        if arr is None:
            arr = self.load_image_data(path)
            if arr is None:
                return
            self.raw_cache[idx] = arr

        pixmap = self.array_to_pixmap(arr)
        if pixmap is None:
            return

        if self.viewer.hasPhoto():
            rel_zoom = self.viewer.get_relative_zoom()
            rel_x, rel_y = self.viewer.get_relative_pan()
        else:
            rel_zoom = 1.0
            rel_x = rel_y = 0.5

        self.viewer.setPhoto(
            pixmap,
            rel_zoom=rel_zoom,
            rel_x=rel_x,
            rel_y=rel_y,
        )
        self.cache[idx] = pixmap

    def on_apply_global_clicked(self):
        if not self.file_paths:
            return

        self.apply_global_button.setEnabled(False)
        self.status_label.setText("Queuing all images...")
        QApplication.processEvents()

        self.queued_indices.clear()
        self.cache.clear()
        
        for i in range(len(self.file_paths)):
            path = self.file_paths[i]
            self.task_counter += 1
            self.loader_queue.put((
                abs(i - self.current_index),
                self.task_counter,
                i,
                path,
                self.stretch_factor,
                self.black_level
            ))
            self.queued_indices.add(i)

        self.status_label.setText("Global apply queued")
        QApplication.processEvents()
        QTimer.singleShot(2000, lambda: self.status_label.setText("Ready"))
        self.apply_global_button.setEnabled(True)

    def _compute_rigid_transform(self, src_points: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
        src_mean = src_points.mean(axis=0)
        ref_mean = ref_points.mean(axis=0)

        X = src_points - src_mean
        Y = ref_points - ref_mean

        H = X.T @ Y  

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T  

        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        t = ref_mean - R @ src_mean  

        M = np.zeros((2, 3), dtype=np.float32)
        M[:, :2] = R.astype(np.float32)
        M[:, 2] = t.astype(np.float32)
        return M

    def on_align_clicked(self):
        
        if self.reference_index is None:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите Reference image!")
            return
        
        ref_dir_key = self.current_dir
        ref_img_path = self.file_paths[self.reference_index]
        ref_img_key = self.get_image_key(ref_img_path)
        
        if ref_dir_key not in self.alignment_store or ref_img_key not in self.alignment_store[ref_dir_key]:
            QMessageBox.warning(self, "Ошибка", "Reference image должно иметь 3 align points!")
            return
        
        ref_points_data = self.alignment_store[ref_dir_key][ref_img_key]
        if len(ref_points_data) != 3:
            QMessageBox.warning(self, "Ошибка", "Reference image должно иметь ровно 3 align points!")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения выровненных изображений")
        if not output_dir:
            return
        
        self.align_button.setEnabled(False)
        self.status_label.setText("Выравнивание изображений...")
        self.progress_bar.setVisible(True)
        QApplication.processEvents()
        
        try:
            ref_arr = self.load_image_data(ref_img_path)
            if ref_arr is None:
                raise Exception(f"Не удалось загрузить reference: {ref_img_path}")
            
            if ref_arr.ndim == 3:
                ref_h, ref_w, ref_c = ref_arr.shape
            else:
                ref_h, ref_w = ref_arr.shape
                ref_c = 1

            ref_points = np.array([
                [float(p["x"]), float(p["y"])] for p in ref_points_data
            ], dtype=np.float32)
            
            aligned_count = 0
            skipped_count = 0
            total_images = len(self.file_paths)
            
            self.progress_bar.setMaximum(total_images)
            
            for idx, img_path in enumerate(self.file_paths):
                self.progress_bar.setValue(idx)
                QApplication.processEvents()
                
                img_key = self.get_image_key(img_path)
                
                if ref_dir_key not in self.alignment_store:
                    skipped_count += 1
                    continue
                
                if img_key not in self.alignment_store[ref_dir_key]:
                    skipped_count += 1
                    continue
                
                points_data = self.alignment_store[ref_dir_key][img_key]
                
                if not isinstance(points_data, list):
                    skipped_count += 1
                    continue
                
                if len(points_data) == 0:
                    skipped_count += 1
                    continue
                
                if len(points_data) != 3:
                    skipped_count += 1
                    continue
                
                try:
                    src_points_array = np.array([
                        [float(p["x"]), float(p["y"])] for p in points_data
                    ], dtype=np.float32)
                    
                    if src_points_array.size == 0 or np.isnan(src_points_array).any():
                        skipped_count += 1
                        continue
                        
                except (ValueError, KeyError, TypeError) as e:
                    skipped_count += 1
                    continue
      
                img_arr = self.load_image_data(img_path)
                if img_arr is None:
                    skipped_count += 1
                    continue
                
                if img_arr.ndim == 3:
                    img_h, img_w, img_c = img_arr.shape
                else:
                    img_h, img_w = img_arr.shape
                    img_c = 1

                src_points = src_points_array.astype(np.float32)

                try:
                    if self.align_method == "rotate_shift_scale":
                        M = cv2.getAffineTransform(src_points, ref_points)
                    else:
                        M = self._compute_rigid_transform(src_points, ref_points)
                except cv2.error as e:
                    skipped_count += 1
                    continue
                
                try:
                    output_size = (ref_w, ref_h)
                    
                    if img_arr.ndim == 2:
                        warped = cv2.warpAffine(
                            img_arr,
                            M,
                            output_size,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                    else:
                        warped = cv2.warpAffine(
                            img_arr,
                            M,
                            output_size,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0)
                        )    
                    
                    if warped.ndim == 3:
                        warped_h, warped_w, warped_c = warped.shape
                    else:
                        warped_h, warped_w = warped.shape
                    
                    if warped_h != ref_h or warped_w != ref_w:
                        print(f"ERROR: Warped size mismatch! Expected ({ref_h}, {ref_w}), got ({warped_h}, {warped_w})")
                        skipped_count += 1
                        continue
                    
                    output_filename = f"aligned_{aligned_count + 1:05d}.fit"
                    output_path = os.path.join(output_dir, output_filename)

                    warped_clipped = np.clip(warped, 0, 1)
                    output_data = (warped_clipped * 65535).astype(np.uint16)

                    if output_data.ndim == 3 and output_data.shape[2] == 3:
                        output_data = np.transpose(output_data, (2, 0, 1))  # (3, height, width)

                    # Создаём HDU
                    hdu = fits.PrimaryHDU(output_data)

                    # НОВОЕ: получить дату наблюдения исходного файла и записать в заголовок FITS
                    obs_date = self.get_observation_date_for_file(img_path)
                    if obs_date:
                        try:
                            # Стандартный ключ FITS
                            hdu.header["DATE-OBS"] = (obs_date, "Observation date")  # [web:35]
                            # Дополнительно — длинный ключ 'Observation Date' (создаст HIERARCH‑ключ)[web:41]
                            hdu.header["Observation Date"] = obs_date
                        except Exception as e:
                            print(f"Не удалось записать Observation Date в {output_path}: {e}")

                    # Сохраняем файл
                    hdu.writeto(output_path, overwrite=True)
                    aligned_count += 1
                    self.status_label.setText(f"Выравнено {aligned_count} изображений (пропущено {skipped_count})...")
                    QApplication.processEvents()
                
                except Exception as e:
                    print(f"ERROR: Failed to warp {img_key}: {e}")
                    import traceback
                    traceback.print_exc()
                    skipped_count += 1
                    continue
            
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"Готово! Выровнено {aligned_count}, пропущено {skipped_count}")
            QMessageBox.information(
                self, 
                "Успех", 
                f"✓ Выровнено {aligned_count} изображений\n✗ Пропущено {skipped_count} (без 3 points)\nСохранено в: {output_dir}"
            )
        
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_label.setText("Ready")
            QMessageBox.critical(self, "Ошибка", f"Ошибка выравнивания: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.align_button.setEnabled(True)
            QTimer.singleShot(2000, lambda: self.status_label.setText("Ready"))

    def load_image_data(self, path):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in IMAGE_EXTS:
                qimg = QImage(path)
                if qimg.isNull():
                    return None

                if qimg.isGrayscale():
                    qimg = qimg.convertToFormat(QImage.Format.Format_Grayscale8)
                    w = qimg.width()
                    h = qimg.height()
                    bpl = qimg.bytesPerLine()
                    ptr = qimg.bits()
                    ptr.setsize(h * bpl)
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
                    arr = arr[:, :w]
                    arr_f = arr.astype(np.float32) / 255.0
                    return arr_f
                else:
                    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
                    w = qimg.width()
                    h = qimg.height()
                    bpl = qimg.bytesPerLine()
                    ptr = qimg.bits()
                    ptr.setsize(h * bpl)
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
                    arr = arr.reshape((h, w, 3))
                    arr_f = arr.astype(np.float32) / 255.0
                    return arr_f

            elif ext in FITS_EXT and fits is not None:
                with fits.open(path) as hdul:
                    data = hdul[0].data
                    if data is None:
                        return None
                    data = np.squeeze(data)

                    if data.ndim == 3 and data.shape[0] == 3:
                        data = np.transpose(data, (1, 2, 0))

                    data = np.array(data, dtype=np.float32)
                    dmin = float(np.min(data))
                    dmax = float(np.max(data))
                    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
                        return None

                    arr_f = (data - dmin) / (dmax - dmin)
                    arr_f = np.clip(arr_f, 0.0, 1.0)
                    return arr_f

            elif ext in RAW_EXTS and rawpy is not None:
                with rawpy.imread(path) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        output_bps=8,
                    )
                rgb = np.asarray(rgb, dtype=np.uint8)
                arr_f = rgb.astype(np.float32) / 255.0
                return arr_f

            else:
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    return None
                qimg = pixmap.toImage()
                if qimg.isGrayscale():
                    qimg = qimg.convertToFormat(QImage.Format.Format_Grayscale8)
                    w = qimg.width()
                    h = qimg.height()
                    bpl = qimg.bytesPerLine()
                    ptr = qimg.bits()
                    ptr.setsize(h * bpl)
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
                    arr = arr[:, :w]
                    arr_f = arr.astype(np.float32) / 255.0
                    return arr_f
                else:
                    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
                    w = qimg.width()
                    h = qimg.height()
                    bpl = qimg.bytesPerLine()
                    ptr = qimg.bits()
                    ptr.setsize(h * bpl)
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
                    arr = arr.reshape((h, w, 3))
                    arr_f = arr.astype(np.float32) / 255.0
                    return arr_f

        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")
            return None

    def _array_to_pixmap_with_params(self, arr, stretch, black):
        if arr is None:
            return None

        img = (arr - black) * stretch
        img = np.clip(img, 0.0, 1.0)

        factor = getattr(self, "reduce_factor", 1.0)
        if factor is None:
            factor = 1.0
        if factor > 1.0:
            h, w = img.shape[:2]
            new_w = max(1, int(w / factor))
            new_h = max(1, int(h / factor))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if getattr(self, "monochrome", False) and img.ndim == 3 and img.shape[2] == 3:
            img = img[:, :, 1]

        img_u8 = (img * 255.0 + 0.5).astype(np.uint8)

        if img_u8.ndim == 2:
            h, w = img_u8.shape
            qimg = QImage(
                img_u8.tobytes(),
                w,
                h,
                w,
                QImage.Format.Format_Grayscale8,
            )
            return QPixmap.fromImage(qimg)

        elif img_u8.ndim == 3 and img_u8.shape[2] == 3:
            h, w, c = img_u8.shape
            bytes_per_line = w * 3
            qimg = QImage(
                img_u8.tobytes(),
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )
            return QPixmap.fromImage(qimg)

        return None

    def array_to_pixmap(self, arr):
        return self._array_to_pixmap_with_params(arr, self.stretch_factor, self.black_level)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.showMaximized() 
    sys.exit(app.exec())
