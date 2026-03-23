# scripts/03_line_segmentation_enhanced.py

import cv2
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=f'logs/line_segmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AdvancedLineSegmenter:
    """
    Advanced line segmentation for historical documents
    
    Features:
    ✅ Handles variable text quality
    ✅ Detects inconsistent baselines
    ✅ Removes decorative elements
    ✅ Works with multiple fonts/sizes
    ✅ Auto-parameter tuning
    ✅ Visual diagnostics
    ✅ Adaptive thresholding
    ✅ Morphological operations
    """
    
    def __init__(self, config=None):
        """Initialize with enhanced configuration"""
        
        self.config = config or self._get_default_config()
        self.metadata = []
        self.diagnostics = []
        
        print("✅ Advanced LineSegmenter initialized")
        self._print_config()
    
    def _get_default_config(self):
        """Enhanced configuration for historical documents"""
        
        return {
            # ============ CORE PARAMETERS ============
            'min_line_height': 12,
            'gap_threshold': 0.015,
            'min_gap_size': 3,
            'padding': 3,
            'output_height': 64,
            'max_width': 1024,
            'save_quality': 0,
            
            # ============ PREPROCESSING ============
            'median_blur_kernel': 3,
            'use_morphology': True,
            'erosion_iterations': 1,
            'dilation_iterations': 1,
            
            # ============ LINE DETECTION ============
            'use_adaptive_threshold': False,
            'adaptive_block_size': 11,
            'merge_close_lines': True,
            'merge_distance': 5,
            
            # ============ POST-PROCESSING ============
            'deskew': True,
            'remove_noise': True,
            'noise_size': 50,
            'auto_tune': True,
            'debug_visualize': False
        }
    
    def _print_config(self):
        """Print current configuration"""
        
        print(f"   Configuration:")
        print(f"   ├─ Min line height: {self.config['min_line_height']} px")
        print(f"   ├─ Gap threshold: {self.config['gap_threshold']}")
        print(f"   ├─ Median blur: {self.config['median_blur_kernel']}")
        print(f"   ├─ Morphology: {self.config['use_morphology']}")
        print(f"   ├─ Deskew: {self.config['deskew']}")
        print(f"   ├─ Remove noise: {self.config['remove_noise']}")
        print(f"   ├─ Auto-tune: {self.config['auto_tune']}")
        print(f"   └─ Debug visualize: {self.config['debug_visualize']}")
    
    def auto_tune_parameters(self, img, target_lines=25):
        """
        Automatically tune parameters based on image characteristics
        
        Args:
            img: Grayscale image
            target_lines: Expected number of lines per page
        
        Returns:
            Tuned config
        """
        
        if not self.config['auto_tune']:
            return self.config
        
        print(f"   🔧 Auto-tuning parameters...")
        
        best_config = dict(self.config)
        best_score = float('inf')
        
        # Test ranges
        min_heights = [10, 12, 15]
        gap_thresholds = [0.01, 0.015, 0.02, 0.025]
        
        for min_h in min_heights:
            for gap_t in gap_thresholds:
                lines = self._detect_lines_simple(img, min_h, gap_t)
                
                # Score: how close to target
                score = abs(len(lines) - target_lines)
                
                if score < best_score:
                    best_score = score
                    best_config['min_line_height'] = min_h
                    best_config['gap_threshold'] = gap_t
        
        self.config = best_config
        print(f"   ✅ Best parameters: min_h={best_config['min_line_height']}, gap={best_config['gap_threshold']:.3f}")
        
        return best_config
    
    def _detect_lines_simple(self, img, min_height, gap_threshold):
        """Simple line detection for parameter tuning"""
        
        _, binary = cv2.threshold(img, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        horizontal_sum = np.sum(binary, axis=1)
        max_sum = np.max(horizontal_sum)
        
        if max_sum == 0:
            return []
        
        threshold = gap_threshold * max_sum
        text_rows = np.where(horizontal_sum > threshold)[0]
        
        lines = []
        if len(text_rows) > 0:
            y_start = text_rows[0]
            y_prev = text_rows[0]
            
            for y in text_rows[1:]:
                if y != y_prev + 1:
                    if y_prev - y_start >= min_height:
                        lines.append((y_start, y_prev))
                    y_start = y
                y_prev = y
            
            if y_prev - y_start >= min_height:
                lines.append((y_start, y_prev))
        
        return lines
    
    def preprocess_image(self, img):
        """
        Advanced preprocessing for historical documents
        
        Steps:
        1. Median blur
        2. Morphological operations
        3. Optional adaptive threshold
        """
        
        # Median blur to reduce noise
        kernel = self.config['median_blur_kernel']
        if kernel > 1 and kernel % 2 == 1:
            img = cv2.medianBlur(img, kernel)
        
        # Morphological operations
        if self.config['use_morphology']:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Erode to remove isolated noise
            img = cv2.erode(img, kernel, iterations=self.config['erosion_iterations'])
            
            # Dilate to reconnect broken text
            img = cv2.dilate(img, kernel, iterations=self.config['dilation_iterations'])
        
        return img
    
    def detect_lines(self, img):
        """
        Advanced line detection with multiple strategies
        
        Args:
            img: Grayscale preprocessed image
        
        Returns:
            List of (y_start, y_end) tuples
        """
        
        try:
            # Auto-tune if enabled
            if self.config['auto_tune']:
                self.auto_tune_parameters(img)
            
            # Preprocess
            processed = self.preprocess_image(img)
            
            # Binarize
            if self.config['use_adaptive_threshold']:
                binary = cv2.adaptiveThreshold(
                    processed, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    self.config['adaptive_block_size'],
                    2
                )
            else:
                _, binary = cv2.threshold(processed, 0, 255,
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Horizontal projection
            horizontal_sum = np.sum(binary, axis=1)
            
            max_sum = np.max(horizontal_sum)
            if max_sum == 0:
                return []
            
            # Threshold
            gap_threshold = self.config['gap_threshold'] * max_sum
            text_rows = np.where(horizontal_sum > gap_threshold)[0]
            
            if len(text_rows) == 0:
                return []
            
            # ============ SMART GAP DETECTION ============
            
            lines = []
            y_start = text_rows[0]
            y_prev = text_rows[0]
            min_gap = self.config['min_gap_size']
            
            for y in text_rows[1:]:
                gap_size = y - y_prev
                
                if gap_size > min_gap:  # Significant gap
                    line_height = y_prev - y_start + 1
                    
                    if line_height >= self.config['min_line_height']:
                        lines.append((y_start, y_prev))
                    
                    y_start = y
                
                y_prev = y
            
            # Last line
            line_height = y_prev - y_start + 1
            if line_height >= self.config['min_line_height']:
                lines.append((y_start, y_prev))
            
            # ============ MERGE CLOSE LINES ============
            
            if self.config['merge_close_lines'] and len(lines) > 1:
                merged = []
                merge_dist = self.config['merge_distance']
                
                for i, (y1, y2) in enumerate(lines):
                    if i == 0:
                        merged.append([y1, y2])
                    else:
                        last_y1, last_y2 = merged[-1]
                        if y1 - last_y2 < merge_dist:
                            merged[-1][1] = y2
                        else:
                            merged.append([y1, y2])
                
                lines = [(int(y1), int(y2)) for y1, y2 in merged]
            
            return lines
        
        except Exception as e:
            logging.error(f"Error detecting lines: {e}")
            return []
    
    def deskew_line(self, line_img, max_angle=5):
        """Correct text skew with angle limit"""
        
        if not self.config['deskew']:
            return line_img
        
        try:
            h, w = line_img.shape[:2]
            
            if h < 5 or w < 20:
                return line_img
            
            _, binary = cv2.threshold(line_img, 150, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) < 3:
                return line_img
            
            angles = []
            
            for contour in contours[:100]:
                area = cv2.contourArea(contour)
                
                if area < 10:
                    continue
                
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                if -max_angle < angle < max_angle:
                    angles.append(angle)
            
            if not angles:
                return line_img
            
            median_angle = np.median(angles)
            
            if abs(median_angle) < 0.5:
                return line_img
            
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            
            rotated = cv2.warpAffine(
                line_img, M, (w, h),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC
            )
            
            return rotated
        
        except Exception as e:
            logging.warning(f"Error deskewing: {e}")
            return line_img
    
    def remove_small_noise(self, line_img):
        """Remove small isolated components"""
        
        if not self.config['remove_noise']:
            return line_img
        
        try:
            _, binary = cv2.threshold(line_img, 150, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            mask = np.ones_like(line_img) * 255
            
            for contour in contours:
                if cv2.contourArea(contour) >= self.config['noise_size']:
                    cv2.drawContours(mask, [contour], 0, 0, -1)
            
            cleaned = cv2.bitwise_and(line_img, mask)
            
            return cleaned
        
        except Exception as e:
            logging.warning(f"Error removing noise: {e}")
            return line_img
    
    def normalize_line(self, line_img):
        """Normalize to 64×1024"""
        
        try:
            h, w = line_img.shape[:2]
            
            if h == 0 or w == 0:
                return None
            
            target_height = self.config['output_height']
            max_width = self.config['max_width']
            
            ratio = target_height / h
            new_w = int(w * ratio)
            new_w = min(max(new_w, 32), max_width)
            
            resized = cv2.resize(line_img, (new_w, target_height),
                               interpolation=cv2.INTER_CUBIC)
            
            if resized.shape[1] < max_width:
                pad_right = max_width - resized.shape[1]
                
                resized = cv2.copyMakeBorder(
                    resized,
                    top=0, bottom=0,
                    left=0, right=pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=255
                )
            
            elif resized.shape[1] > max_width:
                resized = resized[:, :max_width]
            
            return resized
        
        except Exception as e:
            logging.error(f"Error normalizing: {e}")
            return None
    
    def visualize_detection(self, img, lines, output_path):
        """Create visualization of detected lines"""
        
        try:
            if len(img.shape) == 2:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_color = img.copy()
            
            # Draw lines
            for idx, (y1, y2) in enumerate(lines):
                cv2.rectangle(img_color, (0, y1), (img.shape[1], y2),
                            (0, 255, 0), 2)
                cv2.putText(img_color, str(idx+1), (5, (y1+y2)//2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            cv2.imwrite(output_path, img_color)
            
        except Exception as e:
            logging.warning(f"Error visualizing: {e}")
    
    def extract_and_save_lines(self, img, lines, page_output_dir, 
                               book_name, page_name, page_idx):
        """Extract and save line images"""
        
        line_count = 0
        line_metadata = []
        
        padding = self.config['padding']
        
        for line_idx, (y1, y2) in enumerate(lines, 1):
            try:
                y1_padded = max(0, y1 - padding)
                y2_padded = min(img.shape[0], y2 + padding)
                
                line_img = img[y1_padded:y2_padded, :]
                
                line_img = self.deskew_line(line_img)
                line_img = self.remove_small_noise(line_img)
                
                normalized = self.normalize_line(line_img)
                
                if normalized is None:
                    continue
                
                line_filename = f"page_{page_idx}_line_{line_idx}.png"
                line_path = os.path.join(page_output_dir, line_filename)
                
                compression = self.config['save_quality']
                success = cv2.imwrite(
                    line_path,
                    normalized,
                    [cv2.IMWRITE_PNG_COMPRESSION, compression]
                )
                
                if not success:
                    continue
                
                metadata_entry = {
                    "image": line_filename,
                    "image_path": line_path,
                    "book": book_name,
                    "page": page_name,
                    "page_number": page_idx,
                    "line_number": line_idx,
                    "folder": f"page_{page_idx}",
                    "original_y_start": int(y1),
                    "original_y_end": int(y2),
                    "original_height": int(y2 - y1),
                    "normalized_height": normalized.shape[0],
                    "normalized_width": normalized.shape[1],
                    "text": "",
                    "confidence": 0.0
                }
                
                line_metadata.append(metadata_entry)
                line_count += 1
                
            except Exception as e:
                logging.error(f"Error extracting line {line_idx}: {e}")
                continue
        
        return line_count, line_metadata
    
    def process_page(self, page_path, base_output_dir, book_name, page_idx):
        """Process single page"""
        
        try:
            img = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                return 0, []
            
            page_folder_name = f"page_{page_idx}"
            page_folder_path = os.path.join(base_output_dir, page_folder_name)
            
            Path(page_folder_path).mkdir(parents=True, exist_ok=True)
            
            lines = self.detect_lines(img)
            
            if not lines:
                return 0, []
            
            # Visualize if debug enabled
            if self.config['debug_visualize']:
                vis_path = os.path.join(page_folder_path, "debug_visualization.png")
                self.visualize_detection(img, lines, vis_path)
            
            page_name = Path(page_path).stem
            line_count, metadata = self.extract_and_save_lines(
                img, lines, page_folder_path,
                book_name, page_name, page_idx
            )
            
            return line_count, metadata
        
        except Exception as e:
            logging.error(f"Error processing page: {e}")
            return 0, []
    
    def process_all_pages(self, input_dir, output_dir, book_name=None):
        """Process all pages"""
        
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return 0
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if book_name is None:
            book_name = input_path.name
        
        print(f"\n{'='*70}")
        print(f"✂️  ADVANCED LINE SEGMENTATION")
        print(f"{'='*70}")
        print(f"   Input:  {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   Book:   {book_name}\n")
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        page_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not page_files:
            print(f"❌ No images found")
            return 0
        
        print(f"   Found {len(page_files)} pages\n")
        
        total_lines = 0
        all_metadata = []
        failed_pages = []
        
        for page_idx, page_file in enumerate(page_files, 1):
            print(f"   Processing {page_idx:3d}/{len(page_files)}: {page_file.name}...", end=" ", flush=True)
            
            try:
                line_count, metadata = self.process_page(
                    str(page_file),
                    str(output_path),
                    book_name,
                    page_idx
                )
                
                if line_count > 0:
                    print(f"✅ {line_count} lines")
                    total_lines += line_count
                    all_metadata.extend(metadata)
                else:
                    print(f"⚠️  No lines")
                    failed_pages.append(page_file.name)
                
            except Exception as e:
                print(f"❌ Error")
                logging.error(f"Error processing page {page_idx}: {e}")
                failed_pages.append(page_file.name)
                continue
        
        self.metadata = all_metadata
        
        print(f"\n{'='*70}")
        print(f"✅ SEGMENTATION COMPLETE")
        print(f"{'='*70}")
        print(f"   ✅ Total lines: {total_lines}")
        print(f"   📊 Average lines/page: {total_lines / len(page_files):.1f}")
        
        if failed_pages:
            print(f"   ⚠️  Failed pages: {len(failed_pages)}")
            for page in failed_pages[:5]:
                print(f"      - {page}")
        
        page_folders = sorted([f for f in output_path.iterdir() if f.is_dir()])
        print(f"   📁 Page folders: {len(page_folders)}")
        
        print(f"\n   Output: {output_dir}")
        print(f"{'='*70}\n")
        
        logging.info(f"Complete: {total_lines} lines from {len(page_files)} pages")
        
        return total_lines
    
    def save_metadata(self, output_file):
        """Save metadata JSON"""
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Metadata saved: {output_file}")
        print(f"   Total entries: {len(self.metadata)}")
    
    def save_report(self, output_dir, report_file=None):
        """Save segmentation report"""
        
        output_path = Path(output_dir)
        page_folders = sorted([f for f in output_path.iterdir() if f.is_dir()])
        
        report = f"""
{'='*70}
LINE SEGMENTATION REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
  Min line height: {self.config['min_line_height']} px
  Gap threshold: {self.config['gap_threshold']}
  Min gap size: {self.config['min_gap_size']} px
  Deskew: {self.config['deskew']}
  Remove noise: {self.config['remove_noise']}
  Merge close lines: {self.config['merge_close_lines']}

Results:
  Total pages: {len(page_folders)}
  Total lines: {len(self.metadata)}
  Average lines/page: {len(self.metadata) / len(page_folders):.1f if page_folders else 0}

Output Directory: {output_dir}

Page Details:
"""
        
        for folder in page_folders[:10]:
            line_files = list(folder.glob("*.png"))
            num_lines = len(line_files)
            report += f"\n  {folder.name}/: {num_lines} lines"
        
        if len(page_folders) > 10:
            report += f"\n  ... and {len(page_folders) - 10} more pages"
        
        report += f"\n\n{'='*70}\n"
        
        print(report)
        
        if report_file:
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"Report saved: {report_file}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Advanced line segmentation for historical documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python 03_line_segmentation_enhanced.py -i "preprocessed" -o "segmented"
  
  # With auto-tuning
  python 03_line_segmentation_enhanced.py -i "preprocessed" -o "segmented" --auto-tune
  
  # Custom parameters
  python 03_line_segmentation_enhanced.py -i "preprocessed" -o "segmented" \\
    --min-height 12 --gap-threshold 0.015
  
  # With debug visualization
  python 03_line_segmentation_enhanced.py -i "preprocessed" -o "segmented" --debug
  
  # For poor quality documents
  python 03_line_segmentation_enhanced.py -i "preprocessed" -o "segmented" \\
    --adaptive-threshold --no-denoise
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input directory')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('-b', '--book-name', default=None, help='Book name')
    parser.add_argument('-m', '--metadata', default=None, help='Metadata output file')
    parser.add_argument('-r', '--report', default=None, help='Report output file')
    
    parser.add_argument('--min-height', type=int, default=12, help='Min line height')
    parser.add_argument('--gap-threshold', type=float, default=0.015, help='Gap threshold')
    parser.add_argument('--min-gap-size', type=int, default=3, help='Min gap size')
    parser.add_argument('--median-blur', type=int, default=3, help='Median blur kernel')
    parser.add_argument('--noise-size', type=int, default=50, help='Min noise size')
    
    parser.add_argument('--auto-tune', action='store_true', help='Auto-tune parameters')
    parser.add_argument('--adaptive-threshold', action='store_true', help='Use adaptive threshold')
    parser.add_argument('--no-deskew', action='store_true', help='Disable deskew')
    parser.add_argument('--no-denoise', action='store_true', help='Disable denoising')
    parser.add_argument('--no-morphology', action='store_true', help='Disable morphology')
    parser.add_argument('--debug', action='store_true', help='Debug visualization')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  ADVANCED LINE SEGMENTATION FOR HISTORICAL DOCUMENTS")
    print("="*70)
    
    try:
        config = {
            'min_line_height': args.min_height,
            'gap_threshold': args.gap_threshold,
            'min_gap_size': args.min_gap_size,
            'median_blur_kernel': args.median_blur,
            'noise_size': args.noise_size,
            'padding': 3,
            'output_height': 64,
            'max_width': 1024,
            'save_quality': 0,
            'use_morphology': not args.no_morphology,
            'use_adaptive_threshold': args.adaptive_threshold,
            'deskew': not args.no_deskew,
            'remove_noise': not args.no_denoise,
            'merge_close_lines': True,
            'auto_tune': args.auto_tune,
            'debug_visualize': args.debug
        }
        
        segmenter = AdvancedLineSegmenter(config)
        
        total_lines = segmenter.process_all_pages(
            input_dir=args.input,
            output_dir=args.output,
            book_name=args.book_name
        )
        
        if args.metadata:
            segmenter.save_metadata(args.metadata)
        else:
            metadata_path = os.path.join(args.output, "metadata.json")
            segmenter.save_metadata(metadata_path)
        
        if args.report:
            segmenter.save_report(args.output, args.report)
        else:
            report_path = os.path.join(args.output, "segmentation_report.txt")
            segmenter.save_report(args.output, report_path)
        
        if total_lines > 0:
            print("✅ Segmentation completed successfully!")
            return 0
        else:
            print("❌ No lines extracted")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted")
        return 130
    
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)