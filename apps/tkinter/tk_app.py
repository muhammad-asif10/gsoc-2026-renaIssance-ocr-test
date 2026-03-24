# scripts/gui_line_detector_advanced.py

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Slider as MplSlider
import threading
import json
from pathlib import Path
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum
import logging
from datetime import datetime
from collections import deque

warnings.filterwarnings('ignore')

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('line_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    """Line detection methods"""
    PROJECTION = "projection"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


@dataclass
class DetectionResult:
    """Container for detection results"""
    lines: List[Tuple[int, int]]
    projection: np.ndarray
    method: DetectionMethod
    parameters: dict
    timestamp: datetime
    processing_time: float


class AdvancedLineDetector:
    """Advanced line detection with multiple algorithms"""
    
    def __init__(self):
        self.cache = {}
        self.history = deque(maxlen=10)
    
    def preprocess_historical(self, img: np.ndarray, 
                            bilateral_d: int = 9,
                            bilateral_sigma_color: float = 75,
                            bilateral_sigma_space: float = 75,
                            clahe_clip: float = 3.0) -> np.ndarray:
        """Advanced preprocessing with configurable parameters"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Bilateral filtering
            denoised = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
            
            # CLAHE contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Morphological lighting correction
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
        """Projection-based line detection"""
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
            logger.error(f"Line detection error: {e}")
            return [], np.array([])
    
    def detect_lines_adaptive(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        """Adaptive thresholding for poor quality"""
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
            logger.error(f"Adaptive detection error: {e}")
            return [], np.array([])
    
    def detect_lines_hybrid(self, img: np.ndarray, 
                           gap_threshold: float = 0.015,
                           min_height: int = 12) -> Tuple[List, np.ndarray]:
        """Hybrid method combining multiple approaches"""
        try:
            proj_lines, proj_sum = self.detect_lines_projection(img, gap_threshold, min_height)
            adapt_lines, adapt_sum = self.detect_lines_adaptive(img)
            
            # Merge results (prefer longer lines)
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
            logger.error(f"Hybrid detection error: {e}")
            return [], np.array([])


class AdvancedLineDetectorGUI:
    """Advanced GUI with professional features"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Historical Document Line Detector")
        self.root.geometry("1800x1100")
        self.root.minsize(1400, 900)
        
        self.detector = AdvancedLineDetector()
        self.current_image = None
        self.current_preprocessed = None
        self.current_result: Optional[DetectionResult] = None
        self.detection_history = deque(maxlen=10)
        
        # UI Elements
        self.ui_elements = {}
        
        # Setup
        self.setup_ui()
        logger.info("GUI initialized successfully")
    
    def setup_ui(self):
        """Create advanced UI layout"""
        try:
            # Style configuration
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('Header.TLabel', font=("Arial", 14, "bold"), foreground="#1f77b4")
            style.configure('Subheader.TLabel', font=("Arial", 10, "bold"))
            style.configure('Info.TLabel', font=("Courier", 8))
            
            # ============ TOP MENU BAR ============
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # File menu
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Open Image", command=self.load_image, accelerator="Ctrl+O")
            file_menu.add_command(label="Save Results", command=self.save_results, accelerator="Ctrl+S")
            file_menu.add_command(label="Export Lines", command=self.export_lines, accelerator="Ctrl+E")
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self.root.quit)
            
            # Edit menu
            edit_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Edit", menu=edit_menu)
            edit_menu.add_command(label="Reset Parameters", command=self.reset_parameters)
            edit_menu.add_command(label="Clear History", command=self.clear_history)
            
            # Help menu
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="About", command=self.show_about)
            help_menu.add_command(label="Documentation", command=self.show_docs)
            
            # ============ TOP PANEL: FILE & STATUS ============
            top_panel = ttk.Frame(self.root)
            top_panel.pack(fill=tk.X, padx=10, pady=10)
            
            ttk.Button(top_panel, text="📁 Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
            
            self.ui_elements['status_label'] = ttk.Label(top_panel, text="Ready", relief=tk.SUNKEN)
            self.ui_elements['status_label'].pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            self.ui_elements['info_label'] = ttk.Label(top_panel, text="", relief=tk.SUNKEN)
            self.ui_elements['info_label'].pack(side=tk.RIGHT, padx=5)
            
            # ============ MAIN CONTAINER ============
            main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
            main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # ============ LEFT PANEL: CONTROLS ============
            left_panel = ttk.Frame(main_container)
            main_container.add(left_panel, weight=0)
            
            self.setup_left_panel(left_panel)
            
            # ============ RIGHT PANEL: VISUALIZATION ============
            right_panel = ttk.Frame(main_container)
            main_container.add(right_panel, weight=1)
            
            self.setup_right_panel(right_panel)
            
            # ============ STATUS BAR ============
            status_frame = ttk.Frame(self.root)
            status_frame.pack(fill=tk.X, padx=10, pady=5)
            
            self.ui_elements['progress'] = ttk.Progressbar(status_frame, mode='determinate')
            self.ui_elements['progress'].pack(fill=tk.X)
            
            logger.info("UI setup complete")
        except Exception as e:
            logger.error(f"UI setup error: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_left_panel(self, parent):
        """Setup left control panel"""
        try:
            # Title
            ttk.Label(parent, text="⚙️ PARAMETERS", style='Header.TLabel').pack(anchor=tk.W, pady=(10, 5))
            
            # Detection method
            ttk.Label(parent, text="Detection Method", style='Subheader.TLabel').pack(anchor=tk.W, pady=(10, 0))
            
            method_var = tk.StringVar(value="projection")
            self.ui_elements['method_var'] = method_var
            
            for method in ["projection", "adaptive", "hybrid"]:
                ttk.Radiobutton(parent, text=method.capitalize(), variable=method_var, 
                              value=method, command=self.update_detection).pack(anchor=tk.W)
            
            ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
            
            # Gap threshold
            ttk.Label(parent, text="Gap Threshold", style='Subheader.TLabel').pack(anchor=tk.W)
            ttk.Label(parent, text="Lower = More Lines", style='Info.TLabel').pack(anchor=tk.W)
            
            gap_frame = ttk.Frame(parent)
            gap_frame.pack(fill=tk.X, pady=5)
            
            self.ui_elements['gap_slider'] = ttk.Scale(gap_frame, from_=0.001, to=0.1, 
                                                      orient=tk.HORIZONTAL, command=self.on_param_change)
            self.ui_elements['gap_slider'].set(0.015)
            self.ui_elements['gap_slider'].pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            self.ui_elements['gap_label'] = ttk.Label(gap_frame, text="0.0150", width=8, style='Info.TLabel')
            self.ui_elements['gap_label'].pack(side=tk.LEFT, padx=5)
            
            # Min height
            ttk.Label(parent, text="Min Line Height (px)", style='Subheader.TLabel').pack(anchor=tk.W, pady=(10, 0))
            ttk.Label(parent, text="Higher = Fewer Lines", style='Info.TLabel').pack(anchor=tk.W)
            
            height_frame = ttk.Frame(parent)
            height_frame.pack(fill=tk.X, pady=5)
            
            self.ui_elements['height_slider'] = ttk.Scale(height_frame, from_=5, to=50, 
                                                         orient=tk.HORIZONTAL, command=self.on_param_change)
            self.ui_elements['height_slider'].set(12)
            self.ui_elements['height_slider'].pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            self.ui_elements['height_label'] = ttk.Label(height_frame, text="12", width=8, style='Info.TLabel')
            self.ui_elements['height_label'].pack(side=tk.LEFT, padx=5)
            
            # Preprocessing controls (Advanced)
            ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
            ttk.Label(parent, text="🔧 ADVANCED", style='Header.TLabel').pack(anchor=tk.W, pady=(10, 5))
            
            ttk.Label(parent, text="CLAHE Clip Limit", style='Subheader.TLabel').pack(anchor=tk.W)
            
            clahe_frame = ttk.Frame(parent)
            clahe_frame.pack(fill=tk.X, pady=5)
            
            self.ui_elements['clahe_slider'] = ttk.Scale(clahe_frame, from_=1.0, to=5.0, 
                                                        orient=tk.HORIZONTAL, command=self.on_param_change)
            self.ui_elements['clahe_slider'].set(3.0)
            self.ui_elements['clahe_slider'].pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            self.ui_elements['clahe_label'] = ttk.Label(clahe_frame, text="3.0", width=4, style='Info.TLabel')
            self.ui_elements['clahe_label'].pack(side=tk.LEFT, padx=5)
            
            # Separator
            ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
            
            # Statistics
            ttk.Label(parent, text="📊 STATISTICS", style='Header.TLabel').pack(anchor=tk.W, pady=(10, 5))
            
            stats_frame = ttk.Frame(parent)
            stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            scrollbar = ttk.Scrollbar(stats_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.ui_elements['stats_text'] = tk.Text(stats_frame, height=20, width=32, 
                                                    font=("Courier", 8), yscrollcommand=scrollbar.set, 
                                                    wrap=tk.WORD, relief=tk.SUNKEN)
            self.ui_elements['stats_text'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=self.ui_elements['stats_text'].yview)
            
            # Buttons
            button_frame = ttk.Frame(parent)
            button_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(button_frame, text="💾 Save", command=self.save_results).pack(fill=tk.X, pady=2)
            ttk.Button(button_frame, text="📥 Export", command=self.export_lines).pack(fill=tk.X, pady=2)
            ttk.Button(button_frame, text="🔄 Reset", command=self.reset_parameters).pack(fill=tk.X, pady=2)
            
        except Exception as e:
            logger.error(f"Left panel setup error: {e}")
    
    def setup_right_panel(self, parent):
        """Setup right visualization panel"""
        try:
            # Create figure
            self.ui_elements['fig'] = Figure(figsize=(12, 8), dpi=100, tight_layout=False)
            self.ui_elements['fig'].subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, 
                                                   hspace=0.3, wspace=0.25)
            
            # Canvas
            canvas_widget = FigureCanvasTkAgg(self.ui_elements['fig'], master=parent)
            canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.ui_elements['canvas'] = canvas_widget
        except Exception as e:
            logger.error(f"Right panel setup error: {e}")
    
    def load_image(self):
        """Load image with validation"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.tiff *.bmp"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
            
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            filename = Path(file_path).name
            self.ui_elements['status_label'].config(text=f"✅ Loaded: {filename}")
            
            # Preprocess
            clahe_clip = float(self.ui_elements['clahe_slider'].get())
            self.current_preprocessed = self.detector.preprocess_historical(
                self.current_image, clahe_clip=clahe_clip
            )
            
            # Initial detection
            self.update_detection()
            logger.info(f"Loaded image: {filename}")
            
        except Exception as e:
            logger.error(f"Image loading error: {e}")
            messagebox.showerror("Error", str(e))
    
    def on_param_change(self, value):
        """Handle parameter changes"""
        try:
            gap_val = float(self.ui_elements['gap_slider'].get())
            self.ui_elements['gap_label'].config(text=f"{gap_val:.4f}")
            
            height_val = int(float(self.ui_elements['height_slider'].get()))
            self.ui_elements['height_label'].config(text=str(height_val))
            
            clahe_val = float(self.ui_elements['clahe_slider'].get())
            self.ui_elements['clahe_label'].config(text=f"{clahe_val:.1f}")
            
            if self.current_preprocessed is not None:
                self.update_detection()
        except Exception as e:
            logger.error(f"Parameter change error: {e}")
    
    def update_detection(self):
        """Update detection with selected method"""
        try:
            if self.current_preprocessed is None:
                return
            
            method = self.ui_elements['method_var'].get()
            gap_threshold = float(self.ui_elements['gap_slider'].get())
            min_height = int(float(self.ui_elements['height_slider'].get()))
            
            start_time = datetime.now()
            
            # Select method
            if method == "projection":
                lines, projection = self.detector.detect_lines_projection(
                    self.current_preprocessed, gap_threshold, min_height
                )
            elif method == "adaptive":
                lines, projection = self.detector.detect_lines_adaptive(self.current_preprocessed)
            else:  # hybrid
                lines, projection = self.detector.detect_lines_hybrid(
                    self.current_preprocessed, gap_threshold, min_height
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store result
            self.current_result = DetectionResult(
                lines=lines,
                projection=projection,
                method=DetectionMethod(method),
                parameters={
                    'gap_threshold': gap_threshold,
                    'min_height': min_height
                },
                timestamp=datetime.now(),
                processing_time=processing_time
            )
            
            # Update visualization
            thread = threading.Thread(target=self.visualize_results, daemon=True)
            thread.start()
            
            # Update statistics
            self.update_statistics()
            
            # Update history
            self.detection_history.append(self.current_result)
        
        except Exception as e:
            logger.error(f"Detection error: {e}")
    
    def visualize_results(self):
        """Visualize detection results"""
        try:
            if self.current_result is None or self.current_image is None:
                return
            
            fig = self.ui_elements['fig']
            fig.clear()
            
            # LEFT: Image with lines
            ax1 = fig.add_subplot(1, 2, 1)
            img_copy = self.current_image.copy()
            
            for idx, (y1, y2) in enumerate(self.current_result.lines):
                y1, y2 = int(y1), int(y2)
                cv2.rectangle(img_copy, (0, y1), (img_copy.shape[1], y2), (0, 255, 0), 2)
                cv2.putText(img_copy, str(idx + 1), (10, (y1 + y2) // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            ax1.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"Detected Lines: {len(self.current_result.lines)}", fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # RIGHT: Projection graph
            ax2 = fig.add_subplot(1, 2, 2)
            projection = np.array(self.current_result.projection, dtype=np.float32)
            
            ax2.plot(projection, range(len(projection)), linewidth=2, color='blue')
            ax2.set_title(f"Horizontal Projection ({self.current_result.method.value})", 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel("Pixel Count", fontsize=10)
            ax2.set_ylabel("Row Number", fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()
            
            # Draw line boundaries
            for y1, y2 in self.current_result.lines:
                ax2.axhline(y=float(y1), color='g', linestyle='--', alpha=0.5, linewidth=0.8)
                ax2.axhline(y=float(y2), color='g', linestyle='--', alpha=0.5, linewidth=0.8)
            
            fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, hspace=0.3, wspace=0.25)
            
            self.ui_elements['canvas'].draw()
            self.root.update()
        
        except Exception as e:
            logger.error(f"Visualization error: {e}")
    
    def update_statistics(self):
        """Update statistics display"""
        try:
            text_widget = self.ui_elements['stats_text']
            text_widget.config(state=tk.NORMAL)
            text_widget.delete("1.0", tk.END)
            
            if self.current_result is None:
                text_widget.insert("1.0", "No detection results")
                text_widget.config(state=tk.DISABLED)
                return
            
            stats = f"""📋 DETECTION INFO
{'─'*30}
Method:         {self.current_result.method.value}
Time:           {self.current_result.processing_time:.3f}s
Timestamp:      {self.current_result.timestamp.strftime('%H:%M:%S')}

📊 PARAMETERS
{'─'*30}
Gap Threshold:  {self.current_result.parameters['gap_threshold']:.4f}
Min Height:     {self.current_result.parameters['min_height']} px

📈 RESULTS
{'─'*30}
Total Lines:    {len(self.current_result.lines)}
"""
            
            if len(self.current_result.lines) > 0:
                heights = np.array([y2 - y1 for y1, y2 in self.current_result.lines], dtype=np.float32)
                
                stats += f"""Line Heights:
  Average:    {np.mean(heights):.1f} px
  Min:        {np.min(heights):.0f} px
  Max:        {np.max(heights):.0f} px
  Std Dev:    {np.std(heights):.1f} px

📍 FIRST 10 LINES
{'─'*30}"""
                
                for idx, (y1, y2) in enumerate(self.current_result.lines[:10], 1):
                    height = y2 - y1
                    stats += f"\nL{idx:2d}: {int(y1):4d}-{int(y2):4d} (h={int(height):3d})"
                
                if len(self.current_result.lines) > 10:
                    stats += f"\n\n+{len(self.current_result.lines) - 10} more"
            
            stats += f"\n\n📚 HISTORY\n{'─'*30}\n{len(self.detection_history)} detections"
            
            text_widget.insert("1.0", stats)
            text_widget.config(state=tk.DISABLED)
        
        except Exception as e:
            logger.error(f"Statistics update error: {e}")
    
    def save_results(self):
        """Save results as JSON with metadata"""
        try:
            if self.current_result is None:
                messagebox.showwarning("Warning", "No detection results to save")
                return
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
            
            if not file_path:
                return
            
            results = {
                "timestamp": self.current_result.timestamp.isoformat(),
                "method": self.current_result.method.value,
                "processing_time_seconds": self.current_result.processing_time,
                "parameters": self.current_result.parameters,
                "total_lines": len(self.current_result.lines),
                "lines": [
                    {
                        "line_number": idx + 1,
                        "y_start": int(y1),
                        "y_end": int(y2),
                        "height": int(y2 - y1)
                    }
                    for idx, (y1, y2) in enumerate(self.current_result.lines)
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            messagebox.showinfo("Success", f"Results saved to {Path(file_path).name}")
            logger.info(f"Results saved: {file_path}")
        
        except Exception as e:
            logger.error(f"Save error: {e}")
            messagebox.showerror("Error", str(e))
    
    def export_lines(self):
        """Export cropped line images with advanced options"""
        try:
            if self.current_result is None or not self.current_result.lines:
                messagebox.showwarning("Warning", "No lines to export")
                return
            
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if not output_dir:
                return
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            for idx, (y1, y2) in enumerate(self.current_result.lines, 1):
                y1, y2 = int(y1), int(y2)
                
                line_img = self.current_preprocessed[y1:y2, :]
                h, w = line_img.shape[:2]
                
                if h == 0:
                    continue
                
                # Normalize
                target_h = 64
                ratio = target_h / h
                new_w = int(w * ratio)
                line_img_resized = cv2.resize(line_img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
                
                # Pad
                if line_img_resized.shape[1] < 1024:
                    pad = 1024 - line_img_resized.shape[1]
                    line_img_resized = cv2.copyMakeBorder(
                        line_img_resized, 0, 0, 0, pad,
                        cv2.BORDER_CONSTANT, value=255
                    )
                else:
                    line_img_resized = line_img_resized[:, :1024]
                
                output_file = Path(output_dir) / f"line_{idx:04d}.png"
                cv2.imwrite(str(output_file), line_img_resized)
            
            messagebox.showinfo("Success", f"Exported {len(self.current_result.lines)} lines")
            logger.info(f"Exported {len(self.current_result.lines)} lines to {output_dir}")
        
        except Exception as e:
            logger.error(f"Export error: {e}")
            messagebox.showerror("Error", str(e))
    
    def reset_parameters(self):
        """Reset parameters to defaults"""
        self.ui_elements['gap_slider'].set(0.015)
        self.ui_elements['height_slider'].set(12)
        self.ui_elements['clahe_slider'].set(3.0)
        logger.info("Parameters reset to defaults")
    
    def clear_history(self):
        """Clear detection history"""
        self.detection_history.clear()
        logger.info("Detection history cleared")
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About",
            "Advanced Historical Document Line Detector\n"
            "v2.0\n\n"
            "Professional line detection tool for historical documents\n"
            "with multiple detection methods and advanced preprocessing")
    
    def show_docs(self):
        """Show documentation"""
        messagebox.showinfo("Documentation",
            "DETECTION METHODS:\n"
            "• Projection: Fast, good for clear documents\n"
            "• Adaptive: Better for poor quality\n"
            "• Hybrid: Combines both methods\n\n"
            "PARAMETERS:\n"
            "• Gap Threshold: Sensitivity to line gaps\n"
            "• Min Height: Minimum pixels per line\n"
            "• CLAHE Clip: Contrast enhancement strength")


def main():
    try:
        root = tk.Tk()
        app = AdvancedLineDetectorGUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    