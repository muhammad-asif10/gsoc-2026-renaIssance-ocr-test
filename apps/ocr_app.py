# scripts/professional_ocr_app_with_zoom.py
"""
Professional Historical Document OCR with Zoom
Image-Focused Layout for 1366x768 - Interactive Zoom
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import json
from pathlib import Path
import warnings
import logging
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
import easyocr
import time
import queue

warnings.filterwarnings('ignore')

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ocr_app.log')]
)
logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    PROJECTION = "projection"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


@dataclass
class DetectionResult:
    lines: List[Tuple[int, int]]
    projection: np.ndarray
    method: DetectionMethod
    parameters: dict
    timestamp: datetime
    processing_time: float


@dataclass
class OCRResult:
    line_number: int
    text: str
    confidence: float
    processing_time: float


class AdvancedLineDetector:
    """Line detection engine"""
    
    def __init__(self):
        self.history = deque(maxlen=10)
    
    def preprocess_historical(self, img: np.ndarray, clahe_clip: float = 3.0) -> np.ndarray:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
            illumination = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            corrected = cv2.divide(enhanced, illumination, scale=255)
            
            return corrected
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def detect_lines_projection(self, img: np.ndarray, 
                               gap_threshold: float = 0.015, 
                               min_height: int = 12) -> Tuple[List, np.ndarray]:
        try:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            horizontal_sum = np.sum(binary, axis=1)
            max_sum = np.max(horizontal_sum)
            
            if max_sum == 0:
                return [], horizontal_sum
            
            threshold = gap_threshold * max_sum
            text_rows = np.where(horizontal_sum > threshold)[0]
            
            if len(text_rows) == 0:
                return [], horizontal_sum
            
            lines = []
            y_start = text_rows[0]
            y_prev = text_rows[0]
            
            for y in text_rows[1:]:
                if y != y_prev + 1:
                    line_height = y_prev - y_start + 1
                    if line_height >= min_height:
                        lines.append((y_start, y_prev))
                    y_start = y
                y_prev = y
            
            line_height = y_prev - y_start + 1
            if line_height >= min_height:
                lines.append((y_start, y_prev))
            
            return lines, horizontal_sum
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return [], np.array([])
    
    def detect_lines_adaptive(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            morph = cv2.erode(img, kernel, iterations=1)
            morph = cv2.dilate(morph, kernel, iterations=1)
            
            binary = cv2.adaptiveThreshold(morph, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            horizontal_sum = np.sum(binary, axis=1)
            max_sum = np.max(horizontal_sum)
            
            if max_sum == 0:
                return [], horizontal_sum
            
            threshold = 0.01 * max_sum
            text_rows = np.where(horizontal_sum > threshold)[0]
            
            if len(text_rows) == 0:
                return [], horizontal_sum
            
            lines = []
            y_start = text_rows[0]
            y_prev = text_rows[0]
            
            for y in text_rows[1:]:
                if y != y_prev + 1:
                    if y_prev - y_start >= 10:
                        lines.append((y_start, y_prev))
                    y_start = y
                y_prev = y
            
            if y_prev - y_start >= 10:
                lines.append((y_start, y_prev))
            
            return lines, horizontal_sum
        except Exception as e:
            logger.error(f"Adaptive error: {e}")
            return [], np.array([])
    
    def detect_lines_hybrid(self, img: np.ndarray, 
                           gap_threshold: float = 0.015,
                           min_height: int = 12) -> Tuple[List, np.ndarray]:
        try:
            proj_lines, proj_sum = self.detect_lines_projection(img, gap_threshold, min_height)
            adapt_lines, adapt_sum = self.detect_lines_adaptive(img)
            
            all_lines = proj_lines + adapt_lines
            unique_lines = []
            
            for y1, y2 in sorted(all_lines):
                skip = False
                for uy1, uy2 in unique_lines:
                    if abs(y1 - uy1) < 5 and abs(y2 - uy2) < 5:
                        skip = True
                        break
                if not skip:
                    unique_lines.append((y1, y2))
            
            return unique_lines, proj_sum
        except Exception as e:
            logger.error(f"Hybrid error: {e}")
            return [], np.array([])


class OCRProcessor:
    """OCR processing engine"""
    
    def __init__(self):
        self.reader = None
        self.cancel_flag = False
    
    def initialize(self):
        if self.reader is None:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)
                logger.info("OCR reader initialized")
            except Exception as e:
                logger.error(f"OCR init error: {e}")
                raise
    
    def process_line(self, line_image: np.ndarray, conf_threshold: float = 0.5) -> Optional[OCRResult]:
        try:
            if self.reader is None:
                self.initialize()
            
            if self.cancel_flag:
                return None
            
            start_time = time.time()
            
            if len(line_image.shape) == 2:
                line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_GRAY2RGB)
            else:
                line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
            
            results = self.reader.readtext(line_image_rgb, detail=1)
            
            if not results:
                return None
            
            texts = [d[1] for d in results if d[2] >= conf_threshold]
            combined_text = ' '.join(texts)
            avg_confidence = np.mean([d[2] for d in results])
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                line_number=0,
                text=combined_text,
                confidence=avg_confidence,
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None


class ZoomableImageCanvas:
    """Zoomable matplotlib canvas"""
    
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.press = None
        self.xpress = None
        self.ypress = None
        self.initial_xlim = None
        self.initial_ylim = None
        self.zoom_level = 1.0
        
        self.cid_scroll = fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
    
    def on_scroll(self, event):
        """Handle mouse wheel zoom"""
        if event.inaxes != self.ax:
            return
        
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        if event.button == 'up':
            scale_factor = 0.7  # Zoom in
            self.zoom_level *= 1.3
        elif event.button == 'down':
            scale_factor = 1.3  # Zoom out
            self.zoom_level *= 0.8
        else:
            return
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.fig.canvas.draw_idle()
    
    def on_press(self, event):
        """Start panning"""
        if event.inaxes != self.ax:
            return
        self.press = True
        self.xpress = event.xdata
        self.ypress = event.ydata
    
    def on_release(self, event):
        """End panning"""
        self.press = None
        self.xpress = None
        self.ypress = None
    
    def on_motion(self, event):
        """Handle panning"""
        if not self.press or self.xpress is None or self.ypress is None:
            return
        if event.inaxes != self.ax:
            return
        
        dx = event.xdata - self.xpress
        dy = event.ydata - self.ypress
        
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        new_xlim = [cur_xlim[0] - dx, cur_xlim[1] - dx]
        new_ylim = [cur_ylim[0] - dy, cur_ylim[1] - dy]
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        self.fig.canvas.draw_idle()
    
    def reset_zoom(self):
        """Reset to original zoom"""
        if self.initial_xlim:
            self.ax.set_xlim(self.initial_xlim)
            self.ax.set_ylim(self.initial_ylim)
            self.zoom_level = 1.0
            self.fig.canvas.draw_idle()
    
    def set_initial_limits(self):
        """Store initial limits"""
        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()


class ImageFocusedOCRApp:
    """Image-focused UI with zoom for 1366x768"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Document OCR - Image Focused with Zoom")
        self.root.geometry("1366x768")
        self.root.resizable(False, False)
        
        # Colors
        self.bg = "#f5f5f5"
        self.card_bg = "#ffffff"
        self.text_dark = "#2c3e50"
        self.text_light = "#7f8c8d"
        self.accent = "#3498db"
        self.success = "#27ae60"
        self.border = "#ecf0f1"
        
        # State
        self.detector = AdvancedLineDetector()
        self.ocr_processor = OCRProcessor()
        self.current_image = None
        self.current_preprocessed = None
        self.current_result: Optional[DetectionResult] = None
        self.ocr_results: List[OCRResult] = []
        self.result_queue = queue.Queue()
        self.zoom_handler = None
        
        self.ui_elements = {}
        
        self.setup_styles()
        self.setup_ui()
        self.check_results()
        
        logger.info("App initialized - Image Focused with Zoom")
    
    def setup_styles(self):
        """Setup styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=self.bg)
        style.configure('TLabel', background=self.bg, foreground=self.text_dark)
        style.configure('TButton', font=('Segoe UI', 8), padding=4)
        style.configure('Header.TLabel', font=('Segoe UI', 10, 'bold'), foreground=self.accent)
        style.configure('Info.TLabel', font=('Segoe UI', 8), foreground=self.text_light)
    
    def setup_ui(self):
        """Setup main UI"""
        self.root.configure(bg=self.bg)
        
        # ============ TOP TOOLBAR ============
        top = tk.Frame(self.root, bg=self.card_bg, height=60)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)
        
        # Title
        tk.Label(top, text="📄 Document OCR", font=('Segoe UI', 13, 'bold'),
                bg=self.card_bg, fg=self.accent).pack(side=tk.LEFT, padx=15, pady=8)
        
        # Quick buttons
        btn_frame = tk.Frame(top, bg=self.card_bg)
        btn_frame.pack(side=tk.LEFT, padx=5, pady=6)
        
        tk.Button(btn_frame, text="📁 Load", command=self.load_image, bg=self.accent,
                 fg='white', font=('Segoe UI', 8, 'bold'), padx=10, pady=4,
                 relief=tk.FLAT, cursor='hand2', bd=0).pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_frame, text="🔍 Detect", command=self.run_detection, bg=self.success,
                 fg='white', font=('Segoe UI', 8, 'bold'), padx=10, pady=4,
                 relief=tk.FLAT, cursor='hand2', bd=0).pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_frame, text="📝 OCR", command=self.run_ocr, bg='#e67e22',
                 fg='white', font=('Segoe UI', 8, 'bold'), padx=10, pady=4,
                 relief=tk.FLAT, cursor='hand2', bd=0).pack(side=tk.LEFT, padx=2)
        
        # Status
        status_frame = tk.Frame(top, bg=self.border)
        status_frame.pack(side=tk.RIGHT, padx=10, pady=6, fill=tk.X, expand=True)
        
        self.ui_elements['status'] = tk.Label(status_frame, text="Ready",
                                             bg=self.border, fg=self.text_dark,
                                             font=('Segoe UI', 8), padx=8, pady=4)
        self.ui_elements['status'].pack(fill=tk.BOTH, expand=True)
        
        # ============ MAIN LAYOUT ============
        main = tk.Frame(self.root, bg=self.bg)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # LEFT: Large Image Display (70%)
        left_paned = tk.PanedWindow(main, orient=tk.VERTICAL, bg=self.card_bg,
                                    sashwidth=2, sashrelief=tk.RIDGE)
        left_paned.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        self.setup_visualization_panel(left_paned)
        self.setup_ocr_results_panel(left_paned)
        
        # RIGHT: Controls (30%)
        right = tk.Frame(main, bg=self.card_bg, width=300)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=0)
        right.pack_propagate(False)
        
        self.setup_right_panel(right)
    
    def setup_visualization_panel(self, parent):
        """Large visualization area with zoom controls"""
        # Header with zoom buttons
        header = tk.Frame(parent, bg=self.card_bg, height=40)
        header.pack(fill=tk.X, padx=10, pady=(10, 5))
        header.pack_propagate(False)
        
        # Left: Title
        title_frame = tk.Frame(header, bg=self.card_bg)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(title_frame, text="Image & Detection", bg=self.card_bg,
                fg=self.accent, font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=5)
        
        # Right: Zoom buttons
        zoom_frame = tk.Frame(header, bg=self.card_bg)
        zoom_frame.pack(side=tk.RIGHT)
        
        tk.Label(zoom_frame, text="Zoom:", bg=self.card_bg, fg=self.text_dark,
                font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(zoom_frame, text="🔍+", command=self.zoom_in, bg=self.accent,
                 fg='white', font=('Segoe UI', 8, 'bold'), padx=6, pady=2,
                 relief=tk.FLAT, cursor='hand2', bd=0).pack(side=tk.LEFT, padx=2)
        
        tk.Button(zoom_frame, text="🔍-", command=self.zoom_out, bg=self.accent,
                 fg='white', font=('Segoe UI', 8, 'bold'), padx=6, pady=2,
                 relief=tk.FLAT, cursor='hand2', bd=0).pack(side=tk.LEFT, padx=2)
        
        tk.Button(zoom_frame, text="⟲ Reset", command=self.reset_zoom, bg='#95a5a6',
                 fg='white', font=('Segoe UI', 8, 'bold'), padx=6, pady=2,
                 relief=tk.FLAT, cursor='hand2', bd=0).pack(side=tk.LEFT, padx=2)
        
        self.ui_elements['zoom_label'] = tk.Label(zoom_frame, text="100%", 
                                                 bg=self.card_bg, fg=self.text_light,
                                                 font=('Segoe UI', 8, 'bold'), width=4)
        self.ui_elements['zoom_label'].pack(side=tk.LEFT, padx=(10, 0))
        
        # Canvas
        fig_frame = tk.Frame(parent, bg=self.card_bg)
        parent.add(fig_frame, height=400)
        
        fig = Figure(figsize=(9, 4.5), dpi=92, tight_layout=False)
        fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.08, hspace=0.3, wspace=0.25)
        
        self.ui_elements['fig'] = fig
        
        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.ui_elements['canvas'] = canvas
    
    def setup_ocr_results_panel(self, parent):
        """OCR results area"""
        # Header
        header = tk.Frame(parent, bg=self.card_bg, height=30)
        header.pack(fill=tk.X, padx=10, pady=(5, 5))
        header.pack_propagate(False)
        
        tk.Label(header, text="Transcriptions", bg=self.card_bg,
                fg=self.success, font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=5)
        
        # Text area
        results_frame = tk.Frame(parent, bg=self.card_bg)
        parent.add(results_frame, height=150)
        
        # Notebook
        notebook = ttk.Notebook(results_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Full text tab
        text_frame = tk.Frame(notebook, bg=self.card_bg)
        notebook.add(text_frame, text="Full Text")
        
        self.ui_elements['ocr_text'] = tk.Text(text_frame, height=8, font=('Arial', 9),
                                              bg=self.border, fg=self.text_dark,
                                              wrap=tk.WORD, relief=tk.FLAT, bd=0)
        self.ui_elements['ocr_text'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Line-by-line tab
        lines_frame = tk.Frame(notebook, bg=self.card_bg)
        notebook.add(lines_frame, text="Line-by-Line")
        
        self.ui_elements['lines_text'] = tk.Text(lines_frame, height=8, font=('Courier New', 8),
                                                bg=self.border, fg=self.text_dark,
                                                wrap=tk.WORD, relief=tk.FLAT, bd=0)
        self.ui_elements['lines_text'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_right_panel(self, parent):
        """Setup right control panel"""
        # Scrollable container
        canvas = tk.Canvas(parent, bg=self.card_bg, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=self.card_bg)
        
        scroll_frame.bind("<Configure>",
                         lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # ============ DETECTION SECTION ============
        self._section(scroll_frame, "Line Detection")
        
        # Method
        meth_frame = tk.Frame(scroll_frame, bg=self.card_bg)
        meth_frame.pack(fill=tk.X, padx=12, pady=8)
        
        tk.Label(meth_frame, text="Method", bg=self.card_bg, fg=self.text_dark,
                font=('Segoe UI', 8, 'bold')).pack(anchor='w', pady=(0, 4))
        
        method_var = tk.StringVar(value="projection")
        self.ui_elements['method_var'] = method_var
        
        for method in ["projection", "adaptive", "hybrid"]:
            tk.Radiobutton(meth_frame, text=method.capitalize(), variable=method_var,
                          value=method, bg=self.card_bg, fg=self.text_dark,
                          command=self.on_method_change, font=('Segoe UI', 7),
                          activebackground=self.card_bg).pack(anchor='w', pady=1)
        
        # Sliders - Compact
        self._compact_slider(scroll_frame, "Gap", 0.001, 0.1, 0.015,
                           "gap_slider", "gap_val", self.on_gap_change)
        
        self._compact_slider(scroll_frame, "Height", 5, 50, 12,
                           "height_slider", "height_val", self.on_height_change)
        
        self._compact_slider(scroll_frame, "CLAHE", 1.0, 5.0, 3.0,
                           "clahe_slider", "clahe_val", self.on_clahe_change)
        
        # ============ OCR SECTION ============
        self._section(scroll_frame, "OCR Settings")
        
        self._compact_slider(scroll_frame, "Confidence", 0.1, 0.95, 0.5,
                           "conf_slider", "conf_val", self.on_conf_change)
        
        # ============ STATISTICS ============
        self._section(scroll_frame, "Statistics")
        
        stats_box = tk.Frame(scroll_frame, bg=self.card_bg)
        stats_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        
        self.ui_elements['stats'] = tk.Text(stats_box, height=8, width=35,
                                           font=('Courier New', 7), bg=self.border,
                                           fg=self.text_dark, relief=tk.FLAT, bd=0, wrap=tk.WORD)
        self.ui_elements['stats'].pack(fill=tk.BOTH, expand=True)
        self.ui_elements['stats'].insert('1.0', "Load image")
        self.ui_elements['stats'].config(state=tk.DISABLED)
        
        # ============ ACTION BUTTONS ============
        btn_box = tk.Frame(scroll_frame, bg=self.card_bg)
        btn_box.pack(fill=tk.X, padx=12, pady=12)
        
        tk.Button(btn_box, text="💾 Save", command=self.save_results,
                 bg=self.accent, fg='white', font=('Segoe UI', 7, 'bold'),
                 padx=6, pady=4, relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        
        tk.Button(btn_box, text="📥 Export", command=self.export_lines,
                 bg=self.accent, fg='white', font=('Segoe UI', 7, 'bold'),
                 padx=6, pady=4, relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        
        tk.Button(btn_box, text="🔄 Reset", command=self.reset_parameters,
                 bg='#95a5a6', fg='white', font=('Segoe UI', 7, 'bold'),
                 padx=6, pady=4, relief=tk.FLAT, bd=0).pack(fill=tk.X, pady=2)
        
        # ============ PROGRESS ============
        prog_box = tk.Frame(scroll_frame, bg=self.card_bg)
        prog_box.pack(fill=tk.X, padx=12, pady=8)
        
        self.ui_elements['progress'] = ttk.Progressbar(prog_box, mode='determinate', maximum=100)
        self.ui_elements['progress'].pack(fill=tk.X)
        
        self.ui_elements['progress_label'] = tk.Label(prog_box, text="0%",
                                                      bg=self.card_bg, fg=self.text_light,
                                                      font=('Segoe UI', 7))
        self.ui_elements['progress_label'].pack(anchor='e', pady=(2, 0))
    
    def _section(self, parent, title):
        """Create section header"""
        header = tk.Frame(parent, bg=self.accent, height=28)
        header.pack(fill=tk.X, padx=12, pady=(10, 2), side=tk.TOP)
        header.pack_propagate(False)
        
        tk.Label(header, text=title, bg=self.accent, fg='white',
                font=('Segoe UI', 8, 'bold'), padx=8).pack(fill=tk.X, pady=6)
    
    def _compact_slider(self, parent, label, from_, to, default, slider_key, val_key, cmd):
        """Create compact slider"""
        frame = tk.Frame(parent, bg=self.card_bg)
        frame.pack(fill=tk.X, padx=12, pady=6)
        
        # Label row
        label_frame = tk.Frame(frame, bg=self.card_bg)
        label_frame.pack(fill=tk.X, pady=(0, 3))
        
        tk.Label(label_frame, text=label, bg=self.card_bg, fg=self.text_dark,
                font=('Segoe UI', 8)).pack(side=tk.LEFT)
        
        val_label = tk.Label(label_frame, text=f"{default:.2f}", bg=self.border,
                           fg=self.accent, font=('Courier New', 7), width=6, relief=tk.FLAT)
        val_label.pack(side=tk.RIGHT)
        
        self.ui_elements[val_key] = val_label
        
        # Slider
        slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, command=cmd)
        slider.set(default)
        slider.pack(fill=tk.X)
        
        self.ui_elements[slider_key] = slider
    
    # ============ ZOOM FUNCTIONS ============
    
    def zoom_in(self):
        """Zoom in"""
        try:
            if self.zoom_handler:
                # Simulate scroll event
                cur_xlim = self.zoom_handler.ax.get_xlim()
                cur_ylim = self.zoom_handler.ax.get_ylim()
                
                xdata = (cur_xlim[0] + cur_xlim[1]) / 2
                ydata = (cur_ylim[0] + cur_ylim[1]) / 2
                
                scale_factor = 0.7
                self.zoom_handler.zoom_level *= 1.3
                
                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                
                relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
                rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
                
                self.zoom_handler.ax.set_xlim([xdata - new_width * (1 - relx), 
                                               xdata + new_width * relx])
                self.zoom_handler.ax.set_ylim([ydata - new_height * (1 - rely), 
                                               ydata + new_height * rely])
                
                self.ui_elements['fig'].canvas.draw_idle()
                self.update_zoom_label()
        except Exception as e:
            logger.error(f"Zoom in: {e}")
    
    def zoom_out(self):
        """Zoom out"""
        try:
            if self.zoom_handler:
                cur_xlim = self.zoom_handler.ax.get_xlim()
                cur_ylim = self.zoom_handler.ax.get_ylim()
                
                xdata = (cur_xlim[0] + cur_xlim[1]) / 2
                ydata = (cur_ylim[0] + cur_ylim[1]) / 2
                
                scale_factor = 1.3
                self.zoom_handler.zoom_level *= 0.8
                
                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                
                relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
                rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
                
                self.zoom_handler.ax.set_xlim([xdata - new_width * (1 - relx), 
                                               xdata + new_width * relx])
                self.zoom_handler.ax.set_ylim([ydata - new_height * (1 - rely), 
                                               ydata + new_height * rely])
                
                self.ui_elements['fig'].canvas.draw_idle()
                self.update_zoom_label()
        except Exception as e:
            logger.error(f"Zoom out: {e}")
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        try:
            if self.zoom_handler:
                self.zoom_handler.reset_zoom()
                self.update_zoom_label()
        except Exception as e:
            logger.error(f"Reset zoom: {e}")
    
    def update_zoom_label(self):
        """Update zoom percentage label"""
        try:
            if self.zoom_handler:
                zoom_pct = int(self.zoom_handler.zoom_level * 100)
                self.ui_elements['zoom_label'].config(text=f"{zoom_pct}%")
        except:
            pass
    
    # ============ CALLBACKS ============
    
    def on_gap_change(self, val):
        try:
            v = float(val)
            self.ui_elements['gap_val'].config(text=f"{v:.3f}")
            if self.current_preprocessed is not None:
                self.update_detection()
        except:
            pass
    
    def on_height_change(self, val):
        try:
            v = int(float(val))
            self.ui_elements['height_val'].config(text=str(v))
            if self.current_preprocessed is not None:
                self.update_detection()
        except:
            pass
    
    def on_clahe_change(self, val):
        try:
            v = float(val)
            self.ui_elements['clahe_val'].config(text=f"{v:.1f}")
            if self.current_image is not None:
                clahe = float(self.ui_elements['clahe_slider'].get())
                self.current_preprocessed = self.detector.preprocess_historical(img, clahe)
                self.update_detection()
        except:
            pass
    
    def on_conf_change(self, val):
        try:
            v = float(val)
            self.ui_elements['conf_val'].config(text=f"{v:.2f}")
        except:
            pass
    
    def on_method_change(self):
        try:
            if self.current_preprocessed is not None:
                self.update_detection()
        except:
            pass
    
    def update_detection(self):
        """Update detection"""
        try:
            if self.current_preprocessed is None:
                return
            
            method = self.ui_elements['method_var'].get()
            gap = float(self.ui_elements['gap_slider'].get())
            height = int(float(self.ui_elements['height_slider'].get()))
            
            start = time.time()
            
            if method == "projection":
                lines, proj = self.detector.detect_lines_projection(
                    self.current_preprocessed, gap, height)
            elif method == "adaptive":
                lines, proj = self.detector.detect_lines_adaptive(self.current_preprocessed)
            else:
                lines, proj = self.detector.detect_lines_hybrid(
                    self.current_preprocessed, gap, height)
            
            proc_time = time.time() - start
            
            self.current_result = DetectionResult(
                lines=lines,
                projection=proj,
                method=DetectionMethod(method),
                parameters={'gap': gap, 'height': height},
                timestamp=datetime.now(),
                processing_time=proc_time
            )
            
            self.update_visualization()
            self.update_statistics()
            
        except Exception as e:
            logger.error(f"Detection: {e}")
    
    def update_visualization(self):
        """Update large visualization"""
        try:
            if self.current_result is None or self.current_image is None:
                return
            
            fig = self.ui_elements['fig']
            fig.clear()
            
            # LARGE IMAGE with lines (Left)
            ax1 = fig.add_subplot(1, 2, 1)
            img = self.current_image.copy()
            
            for i, (y1, y2) in enumerate(self.current_result.lines):
                y1, y2 = int(y1), int(y2)
                cv2.rectangle(img, (0, y1), (img.shape[1], y2), (0, 255, 0), 3)
                cv2.putText(img, str(i+1), (8, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 0, 255), 2)
            
            ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"Detected Lines: {len(self.current_result.lines)}", 
                         fontweight='bold', fontsize=11)
            ax1.axis('on')
            ax1.tick_params(labelsize=8)
            
            # PROJECTION (Right)
            ax2 = fig.add_subplot(1, 2, 2)
            proj = np.array(self.current_result.projection, dtype=np.float32)
            ax2.plot(proj, range(len(proj)), linewidth=2, color=self.accent)
            ax2.fill_betweenx(range(len(proj)), 0, proj, alpha=0.3, color=self.accent)
            ax2.set_title(f"Projection ({self.current_result.method.value})", 
                         fontweight='bold', fontsize=11)
            ax2.set_xlabel("Pixel Count", fontsize=9)
            ax2.set_ylabel("Row", fontsize=9)
            ax2.grid(True, alpha=0.2, linewidth=0.5)
            ax2.invert_yaxis()
            ax2.tick_params(labelsize=8)
            
            # Draw line boundaries
            for y1, y2 in self.current_result.lines:
                ax2.axhline(y=float(y1), color='green', linestyle='--', alpha=0.3, linewidth=0.8)
            
            fig.tight_layout()
            self.ui_elements['canvas'].draw()
            
            # Setup zoom handler on the image subplot
            self.zoom_handler = ZoomableImageCanvas(fig, ax1)
            self.zoom_handler.set_initial_limits()
            self.update_zoom_label()
            
        except Exception as e:
            logger.error(f"Visualization: {e}")
    
    def update_statistics(self):
        """Update statistics"""
        try:
            text = self.ui_elements['stats']
            text.config(state=tk.NORMAL)
            text.delete('1.0', tk.END)
            
            if self.current_result is None:
                text.insert('1.0', "Load image to start")
                text.config(state=tk.DISABLED)
                return
            
            lines = self.current_result.lines
            stats = f"""METHOD
{self.current_result.method.value}

PROCESSING
{self.current_result.processing_time:.3f}s

DETECTED
{len(lines)} lines

PARAMETERS
Gap: {self.current_result.parameters['gap']:.4f}
Height: {self.current_result.parameters['height']}px
"""
            
            if len(lines) > 0:
                heights = np.array([y2-y1 for y1, y2 in lines])
                stats += f"""
ANALYSIS
Avg: {np.mean(heights):.0f}px
Min: {np.min(heights):.0f}px
Max: {np.max(heights):.0f}px
StdDev: {np.std(heights):.1f}px

FIRST 3
"""
                for i, (y1, y2) in enumerate(lines[:3]):
                    h = y2 - y1
                    stats += f"L{i+1}: {int(y1)}-{int(y2)}({int(h)})\n"
            
            text.insert('1.0', stats)
            text.config(state=tk.DISABLED)
            
        except Exception as e:
            logger.error(f"Stats: {e}")
    
    def load_image(self):
        """Load image"""
        try:
            path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.tiff *.bmp"), ("All", "*.*")]
            )
            
            if not path:
                return
            
            img = cv2.imread(path)
            if img is None:
                messagebox.showerror("Error", "Cannot load image")
                return
            
            self.current_image = img
            
            clahe = float(self.ui_elements['clahe_slider'].get())
            self.current_preprocessed = self.detector.preprocess_historical(img, clahe)
            
            self.current_result = None
            self.ocr_results = []
            self.clear_results()
            
            filename = Path(path).name
            self.update_status(f"✓ {filename}")
            self.update_detection()
            
        except Exception as e:
            logger.error(f"Load: {e}")
            messagebox.showerror("Error", str(e))
    
    def run_detection(self):
        """Run detection"""
        try:
            if self.current_preprocessed is None:
                messagebox.showwarning("Warning", "Load image first")
                return
            
            self.update_status("🔍 Detecting lines...")
            self.update_detection()
            self.update_status("✓ Detection complete")
            
        except Exception as e:
            logger.error(f"Detect: {e}")
            messagebox.showerror("Error", str(e))
    
    def run_ocr(self):
        """Run OCR"""
        try:
            if not self.current_result:
                messagebox.showwarning("Warning", "Detect lines first")
                return
            
            self.update_status("📝 Running OCR...")
            self.ui_elements['progress'].config(maximum=len(self.current_result.lines), value=0)
            
            thread = threading.Thread(target=self._ocr_thread, daemon=True)
            thread.start()
            
        except Exception as e:
            logger.error(f"OCR: {e}")
            messagebox.showerror("Error", str(e))
    
    def _ocr_thread(self):
        """OCR thread"""
        try:
            self.ocr_processor.initialize()
            conf = float(self.ui_elements['conf_slider'].get())
            self.ocr_results = []
            
            for i, (y1, y2) in enumerate(self.current_result.lines):
                y1, y2 = int(y1), int(y2)
                line_img = self.current_preprocessed[y1:y2, :]
                
                if line_img.shape[0] == 0:
                    continue
                
                result = self.ocr_processor.process_line(line_img, conf)
                
                if result:
                    result.line_number = i + 1
                    self.ocr_results.append(result)
                
                self.ui_elements['progress']['value'] = i + 1
                pct = int((i+1)/len(self.current_result.lines)*100)
                self.ui_elements['progress_label'].config(text=f"{pct}%")
                self.root.update_idletasks()
            
            self.result_queue.put(('ocr_complete', self.ocr_results))
            
        except Exception as e:
            logger.error(f"OCR thread: {e}")
            self.result_queue.put(('error', str(e)))
    
    def check_results(self):
        """Check result queue"""
        try:
            while True:
                rtype, data = self.result_queue.get_nowait()
                
                if rtype == 'ocr_complete':
                    self.display_ocr(data)
                    self.update_status("✓ OCR complete")
                elif rtype == 'error':
                    messagebox.showerror("Error", data)
                    self.update_status("✗ Error")
        
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_results)
    
    def display_ocr(self, results):
        """Display OCR results"""
        try:
            full = '\n'.join([r.text for r in results])
            self.ui_elements['ocr_text'].delete('1.0', tk.END)
            self.ui_elements['ocr_text'].insert('1.0', full)
            
            lines = ""
            for r in results:
                lines += f"L{r.line_number}: {r.text}\n[{r.confidence:.0%}]\n"
            
            self.ui_elements['lines_text'].delete('1.0', tk.END)
            self.ui_elements['lines_text'].insert('1.0', lines)
            
        except Exception as e:
            logger.error(f"Display OCR: {e}")
    
    def update_status(self, msg):
        """Update status"""
        self.ui_elements['status'].config(text=msg)
        self.root.update_idletasks()
    
    def clear_results(self):
        """Clear results"""
        self.ui_elements['ocr_text'].delete('1.0', tk.END)
        self.ui_elements['lines_text'].delete('1.0', tk.END)
        fig = self.ui_elements['fig']
        fig.clear()
        self.ui_elements['canvas'].draw()
    
    def save_results(self):
        """Save results"""
        try:
            if self.current_result is None:
                messagebox.showwarning("Warning", "No results")
                return
            
            folder = filedialog.askdirectory(title="Save Location")
            if not folder:
                return
            
            Path(folder).mkdir(parents=True, exist_ok=True)
            
            data = {
                "method": self.current_result.method.value,
                "lines": len(self.current_result.lines),
                "details": [{"line": i+1, "y_start": int(y1), "y_end": int(y2)}
                           for i, (y1, y2) in enumerate(self.current_result.lines)]
            }
            
            with open(Path(folder) / "detection.json", 'w') as f:
                json.dump(data, f, indent=2)
            
            if self.ocr_results:
                with open(Path(folder) / "text.txt", 'w', encoding='utf-8') as f:
                    f.write('\n'.join([r.text for r in self.ocr_results]))
            
            messagebox.showinfo("Success", "Saved")
            
        except Exception as e:
            logger.error(f"Save: {e}")
            messagebox.showerror("Error", str(e))
    
    def export_lines(self):
        """Export lines"""
        try:
            if not self.current_result:
                messagebox.showwarning("Warning", "No lines")
                return
            
            folder = filedialog.askdirectory(title="Export Location")
            if not folder:
                return
            
            Path(folder).mkdir(parents=True, exist_ok=True)
            count = 0
            
            for i, (y1, y2) in enumerate(self.current_result.lines, 1):
                y1, y2 = int(y1), int(y2)
                line_img = self.current_preprocessed[y1:y2, :]
                
                if line_img.shape[0] == 0:
                    continue
                
                h_target = 64
                ratio = h_target / line_img.shape[0]
                w_new = int(line_img.shape[1] * ratio)
                line_img = cv2.resize(line_img, (w_new, h_target))
                
                if line_img.shape[1] < 1024:
                    line_img = cv2.copyMakeBorder(line_img, 0, 0, 0,
                                                 1024-line_img.shape[1],
                                                 cv2.BORDER_CONSTANT, value=255)
                else:
                    line_img = line_img[:, :1024]
                
                cv2.imwrite(str(Path(folder) / f"line_{i:04d}.png"), line_img)
                count += 1
            
            messagebox.showinfo("Success", f"Exported {count} lines")
            
        except Exception as e:
            logger.error(f"Export: {e}")
            messagebox.showerror("Error", str(e))
    
    def reset_parameters(self):
        """Reset parameters"""
        self.ui_elements['gap_slider'].set(0.015)
        self.ui_elements['height_slider'].set(12)
        self.ui_elements['clahe_slider'].set(3.0)
        self.ui_elements['conf_slider'].set(0.5)


def main():
    try:
        root = tk.Tk()
        app = ImageFocusedOCRApp(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()