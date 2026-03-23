# scripts/process_book_lines.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import os
from datetime import datetime

class BookLineProcessor:
    """
    Process entire book:
    1. Load all pages
    2. Detect lines on each page
    3. Save cropped lines in organized folders
    4. Generate metadata
    """
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.metadata = []
        self.stats = {
            'total_pages': 0,
            'total_lines': 0,
            'pages_processed': 0,
            'pages_failed': 0
        }
    
    def _get_default_config(self):
        """Default configuration"""
        return {
            'gap_threshold': 0.015,
            'min_line_height': 12,
            'output_line_height': 64,
            'output_max_width': 1024,
            'bilateral_kernel': 9,
            'clahe_clip': 3.0,
            'deskew': True,
            'save_visualization': False  # Save debug images
        }
    
    def load_image(self, image_path):
        """Load single image"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        return img
    
    def preprocess_historical(self, img):
        """Preprocess for historical documents"""
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise using bilateral filter
        denoised = cv2.bilateralFilter(gray, self.config['bilateral_kernel'], 75, 75)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.config['clahe_clip'], tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Fix uneven lighting
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
        illumination = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        corrected = cv2.divide(enhanced, illumination, scale=255)
        
        return corrected
    
    def detect_lines(self, img):
        """Detect lines using horizontal projection"""
        
        # Binarize
        _, binary = cv2.threshold(img, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection
        horizontal_sum = np.sum(binary, axis=1)
        
        # Normalize
        max_sum = np.max(horizontal_sum)
        if max_sum == 0:
            return [], horizontal_sum
        
        # Threshold
        threshold = self.config['gap_threshold'] * max_sum
        text_rows = np.where(horizontal_sum > threshold)[0]
        
        if len(text_rows) == 0:
            return [], horizontal_sum
        
        # Find continuous text regions (lines)
        lines = []
        y_start = text_rows[0]
        y_prev = text_rows[0]
        
        for y in text_rows[1:]:
            if y != y_prev + 1:  # Gap found
                line_height = y_prev - y_start + 1
                
                if line_height >= self.config['min_line_height']:
                    lines.append((y_start, y_prev))
                
                y_start = y
            
            y_prev = y
        
        # Don't forget last line
        line_height = y_prev - y_start + 1
        if line_height >= self.config['min_line_height']:
            lines.append((y_start, y_prev))
        
        return lines, horizontal_sum
    
    def deskew_line(self, line_img):
        """Correct line skew"""
        
        if not self.config['deskew']:
            return line_img
        
        try:
            h, w = line_img.shape[:2]
            
            if h < 5 or w < 20:
                return line_img
            
            _, binary = cv2.threshold(line_img, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) < 3:
                return line_img
            
            angles = []
            for contour in contours[:100]:
                if cv2.contourArea(contour) < 10:
                    continue
                
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                
                if -10 < angle < 10:
                    angles.append(angle)
            
            if not angles:
                return line_img
            
            median_angle = np.median(angles)
            
            if abs(median_angle) < 0.5:
                return line_img
            
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(line_img, M, (w, h), 
                                    borderMode=cv2.BORDER_REPLICATE, 
                                    flags=cv2.INTER_CUBIC)
            
            return rotated
        except:
            return line_img
    
    def normalize_line(self, line_img):
        """Normalize line to 64×1024"""
        
        try:
            h, w = line_img.shape[:2]
            
            if h == 0 or w == 0:
                return None
            
            target_h = self.config['output_line_height']
            max_w = self.config['output_max_width']
            
            # Resize maintaining aspect ratio
            ratio = target_h / h
            new_w = int(w * ratio)
            new_w = min(max(new_w, 32), max_w)
            
            resized = cv2.resize(line_img, (new_w, target_h),
                               interpolation=cv2.INTER_CUBIC)
            
            # Pad to fixed width
            if resized.shape[1] < max_w:
                pad = max_w - resized.shape[1]
                resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad,
                                           cv2.BORDER_CONSTANT, value=255)
            else:
                resized = resized[:, :max_w]
            
            return resized
        except:
            return None
    
    def extract_and_save_lines(self, img, lines, page_output_dir, book_name, page_num):
        """Extract and save line images"""
        
        line_count = 0
        page_metadata = []
        
        for line_idx, (y1, y2) in enumerate(lines, 1):
            try:
                # Extract line
                line_img = img[y1:y2, :]
                
                # Deskew
                line_img = self.deskew_line(line_img)
                
                # Normalize
                normalized = self.normalize_line(line_img)
                
                if normalized is None:
                    continue
                
                # Generate filename
                line_filename = f"page_{page_num:04d}_line_{line_idx:04d}.png"
                line_path = os.path.join(page_output_dir, line_filename)
                
                # Save
                cv2.imwrite(line_path, normalized, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
                # Metadata
                page_metadata.append({
                    "line_filename": line_filename,
                    "page_number": page_num,
                    "line_number": line_idx,
                    "y_start": int(y1),
                    "y_end": int(y2),
                    "original_height": int(y2 - y1),
                    "normalized_height": normalized.shape[0],
                    "normalized_width": normalized.shape[1],
                    "text": "",
                    "confidence": 0.0
                })
                
                line_count += 1
                
            except Exception as e:
                continue
        
        return line_count, page_metadata
    
    def visualize_page(self, original_img, lines, horizontal_sum, page_num, output_dir):
        """Create visualization of detected lines"""
        
        try:
            img_copy = original_img.copy()
            
            # Draw rectangles
            for idx, (y1, y2) in enumerate(lines):
                cv2.rectangle(img_copy, (0, y1), (img_copy.shape[1], y2), 
                             (0, 255, 0), 2)
                cv2.putText(img_copy, str(idx + 1), (10, (y1 + y2) // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Create figure
            fig = plt.figure(figsize=(16, 6))
            
            # Left: Image with lines
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"Page {page_num} - Detected Lines: {len(lines)}", fontsize=12)
            ax1.axis('off')
            
            # Right: Projection
            ax2 = plt.subplot(1, 2, 2)
            ax2.plot(horizontal_sum, range(len(horizontal_sum)), linewidth=1, color='blue')
            ax2.set_title(f"Page {page_num} - Horizontal Projection", fontsize=12)
            ax2.set_xlabel("Pixel Count")
            ax2.set_ylabel("Row Number")
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()
            
            # Draw line boundaries
            for y1, y2 in lines:
                ax2.axhline(y=y1, color='g', linestyle='--', alpha=0.5, linewidth=0.8)
                ax2.axhline(y=y2, color='g', linestyle='--', alpha=0.5, linewidth=0.8)
            
            plt.tight_layout()
            
            # Save figure
            vis_path = os.path.join(output_dir, f"page_{page_num:04d}_visualization.png")
            plt.savefig(vis_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return vis_path
        except:
            return None
    
    def process_page(self, image_path, output_base_dir, book_name, page_num, 
                    save_visualization=False):
        """Process single page"""
        
        try:
            # Load image
            img = self.load_image(image_path)
            if img is None:
                print(f"   ⚠️  Page {page_num}: Failed to load")
                self.stats['pages_failed'] += 1
                return 0, []
            
            # Preprocess
            preprocessed = self.preprocess_historical(img)
            
            # Detect lines
            lines, horizontal_sum = self.detect_lines(preprocessed)
            
            if not lines:
                print(f"   ⚠️  Page {page_num}: No lines detected")
                self.stats['pages_failed'] += 1
                return 0, []
            
            # Create page-specific folder
            page_folder_name = f"page_{page_num:04d}"
            page_folder_path = os.path.join(output_base_dir, page_folder_name)
            Path(page_folder_path).mkdir(parents=True, exist_ok=True)
            
            # Extract and save lines
            line_count, page_metadata = self.extract_and_save_lines(
                preprocessed, lines, page_folder_path, book_name, page_num
            )
            
            # Visualize
            if save_visualization:
                self.visualize_page(img, lines, horizontal_sum, page_num, page_folder_path)
            
            print(f"   ✅ Page {page_num}: {line_count} lines extracted")
            self.stats['pages_processed'] += 1
            
            return line_count, page_metadata
            
        except Exception as e:
            print(f"   ❌ Page {page_num}: Error - {str(e)}")
            self.stats['pages_failed'] += 1
            return 0, []
    
    def process_book(self, input_dir, output_dir, book_name=None, 
                    save_visualization=False):
        """Process entire book"""
        
        # Setup
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return 0
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if book_name is None:
            book_name = input_path.name
        
        # Find images
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        image_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            print(f"❌ No images found in {input_dir}")
            return 0
        
        # Header
        print("\n" + "="*70)
        print("📖 BOOK LINE PROCESSING")
        print("="*70)
        print(f"Book: {book_name}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Total pages: {len(image_files)}")
        print(f"{'='*70}\n")
        
        self.stats['total_pages'] = len(image_files)
        
        # Process each page
        total_lines = 0
        all_metadata = []
        
        for page_idx, image_file in enumerate(image_files, 1):
            print(f"Processing {page_idx:3d}/{len(image_files)}: {image_file.name}...", 
                 end=" ", flush=True)
            
            line_count, page_metadata = self.process_page(
                str(image_file),
                str(output_path),
                book_name,
                page_idx,
                save_visualization=save_visualization
            )
            
            total_lines += line_count
            all_metadata.extend(page_metadata)
        
        self.metadata = all_metadata
        self.stats['total_lines'] = total_lines
        
        # Summary
        print(f"\n{'='*70}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"   Book: {book_name}")
        print(f"   Pages processed: {self.stats['pages_processed']}")
        print(f"   Pages failed: {self.stats['pages_failed']}")
        print(f"   Total lines extracted: {total_lines}")
        
        if self.stats['pages_processed'] > 0:
            avg_lines = total_lines / self.stats['pages_processed']
            print(f"   Average lines per page: {avg_lines:.1f}")
        
        print(f"   Output directory: {output_dir}")
        print(f"{'='*70}\n")
        
        return total_lines
    
    def save_metadata(self, output_file):
        """Save metadata as JSON"""
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_dict = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "stats": self.stats,
            "lines": self.metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Metadata saved: {output_file}")
    
    def save_summary_report(self, output_file):
        """Save summary report"""
        
        report = f"""
{'='*70}
BOOK LINE PROCESSING REPORT
{'='*70}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
  Gap threshold: {self.config['gap_threshold']}
  Min line height: {self.config['min_line_height']} px
  Output line height: {self.config['output_line_height']} px
  Output max width: {self.config['output_max_width']} px
  Deskew: {self.config['deskew']}

Processing Statistics:
  Total pages: {self.stats['total_pages']}
  Pages processed: {self.stats['pages_processed']}
  Pages failed: {self.stats['pages_failed']}
  Total lines extracted: {self.stats['total_lines']}
  
  Average lines per page: {self.stats['total_lines'] / self.stats['pages_processed']:.1f if self.stats['pages_processed'] > 0 else 0}

Page-by-page breakdown:
"""
        
        # Group metadata by page
        pages = {}
        for line in self.metadata:
            page_num = line['page_number']
            if page_num not in pages:
                pages[page_num] = 0
            pages[page_num] += 1
        
        for page_num in sorted(pages.keys()):
            report += f"  Page {page_num:04d}: {pages[page_num]} lines\n"
        
        report += f"\n{'='*70}\n"
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"📊 Report saved: {output_file}")
        print(report)


def main():
    parser = argparse.ArgumentParser(
        description='Process entire book - detect lines and save as separate folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic - process all images in folder
  python process_book_lines.py -i "book_pages/" -o "output/"
  
  # With book name
  python process_book_lines.py -i "book_pages/" -o "output/" -b "MyBook"
  
  # With custom parameters
  python process_book_lines.py -i "book_pages/" -o "output/" \\
    --gap-threshold 0.015 --min-height 12
  
  # Save visualizations for debugging
  python process_book_lines.py -i "book_pages/" -o "output/" --save-viz
  
  # With metadata and report
  python process_book_lines.py -i "book_pages/" -o "output/" \\
    -m metadata.json -r report.txt
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input directory with book pages')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('-b', '--book-name', default=None, help='Book name')
    parser.add_argument('-m', '--metadata', default=None, help='Metadata JSON file')
    parser.add_argument('-r', '--report', default=None, help='Summary report file')
    parser.add_argument('--gap-threshold', type=float, default=0.015, help='Gap threshold')
    parser.add_argument('--min-height', type=int, default=12, help='Min line height')
    parser.add_argument('--save-viz', action='store_true', help='Save visualizations')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  BOOK LINE PROCESSING SYSTEM")
    print("="*70)
    
    # Create config
    config = {
        'gap_threshold': args.gap_threshold,
        'min_line_height': args.min_height,
        'output_line_height': 64,
        'output_max_width': 1024,
        'bilateral_kernel': 9,
        'clahe_clip': 3.0,
        'deskew': True,
        'save_visualization': args.save_viz
    }
    
    # Process
    processor = BookLineProcessor(config)
    
    total_lines = processor.process_book(
        args.input,
        args.output,
        book_name=args.book_name,
        save_visualization=args.save_viz
    )
    
    # Save metadata
    if args.metadata:
        processor.save_metadata(args.metadata)
    else:
        metadata_path = os.path.join(args.output, "metadata.json")
        processor.save_metadata(metadata_path)
    
    # Save report
    if args.report:
        processor.save_summary_report(args.report)
    else:
        report_path = os.path.join(args.output, "processing_report.txt")
        processor.save_summary_report(report_path)
    
    if total_lines > 0:
        print("✅ Book processing completed successfully!")
        return 0
    else:
        print("❌ No lines were extracted")
        return 1


if __name__ == "__main__":
    exit_code = main()
    os.sys.exit(exit_code)