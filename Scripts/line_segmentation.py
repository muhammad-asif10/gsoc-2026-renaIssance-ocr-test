# scripts/03_line_segmentation.py

import cv2
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=f'logs/line_segmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LineSegmenter:
    """
    Segment preprocessed document images into individual text lines
    Organized by page number in folders
    
    Output Structure:
    output_dir/
    ├── page_1/
    │   ├── page_1_line_1.png
    │   ├── page_1_line_2.png
    │   └── ...
    ├── page_2/
    │   ├── page_2_line_1.png
    │   ├── page_2_line_2.png
    │   └── ...
    └── ...
    """
    
    def __init__(self, config=None):
        """
        Initialize line segmenter with configuration
        
        Args:
            config: Dictionary with segmentation parameters
        """
        
        # Default configuration
        self.config = config or {
            'min_line_height': 15,           # Minimum pixels for valid line
            'gap_threshold': 0.02,           # Threshold for detecting gaps
            'padding': 2,                    # Pixels to pad around lines
            'output_height': 64,             # Standard output height
            'max_width': 1024,               # Maximum output width
            'save_quality': 0,               # PNG compression (0 = no compression)
            'deskew': True,                  # Enable skew correction
            'remove_noise': True,            # Remove small noise
            'noise_size': 100                # Min pixels to keep
        }
        
        self.metadata = []
        
        print("✅ LineSegmenter initialized")
        print(f"   Min line height: {self.config['min_line_height']} px")
        print(f"   Output height: {self.config['output_height']} px")
        print(f"   Max width: {self.config['max_width']} px")
    
    def detect_lines(self, img):
        """
        Detect horizontal text lines using vertical projection
        
        Method:
        1. Binarize image for line detection
        2. Calculate vertical projection (sum pixels per row)
        3. Find gaps between text (empty rows)
        4. Group consecutive non-empty rows into lines
        
        Args:
            img: Grayscale image (preprocessed)
        
        Returns:
            List of (y_start, y_end) tuples for detected lines
        """
        
        try:
            # Binarize for line detection
            _, binary = cv2.threshold(img, 0, 255, 
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Horizontal projection: sum black pixels in each row
            horizontal_sum = np.sum(binary, axis=1)
            
            # Find maximum to normalize threshold
            max_sum = np.max(horizontal_sum)
            
            if max_sum == 0:
                logging.warning("No text detected in image (empty image)")
                return []
            
            # Dynamic threshold for gap detection
            gap_threshold = self.config['gap_threshold'] * max_sum
            
            # Find all rows with text (above threshold)
            text_rows = np.where(horizontal_sum > gap_threshold)[0]
            
            if len(text_rows) == 0:
                logging.warning("No text rows detected")
                return []
            
            # Group consecutive rows into lines
            lines = []
            y_start = text_rows[0]
            y_prev = text_rows[0]
            
            for y in text_rows[1:]:
                if y != y_prev + 1:  # Gap detected
                    line_height = y_prev - y_start + 1
                    
                    if line_height >= self.config['min_line_height']:
                        lines.append((y_start, y_prev))
                    
                    y_start = y
                
                y_prev = y
            
            # Don't forget the last line
            line_height = y_prev - y_start + 1
            if line_height >= self.config['min_line_height']:
                lines.append((y_start, y_prev))
            
            logging.info(f"Detected {len(lines)} lines")
            
            return lines
        
        except Exception as e:
            logging.error(f"Error detecting lines: {e}")
            return []
    
    def deskew_line(self, line_img):
        """
        Correct text skew (rotation) in historical documents
        
        Args:
            line_img: Single line image
        
        Returns:
            Deskewed line image
        """
        
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
                
                if -10 < angle < 10:
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
        """
        Remove small noise components
        
        Args:
            line_img: Line image
        
        Returns:
            Cleaned line image
        """
        
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
        """
        Normalize line image for model input
        
        - Resize to standard height (64 px)
        - Maintain aspect ratio
        - Pad/crop to max width (1024 px)
        
        Args:
            line_img: Input line image
        
        Returns:
            Normalized line image (64 × 1024)
        """
        
        try:
            h, w = line_img.shape[:2]
            
            if h == 0 or w == 0:
                return None
            
            target_height = self.config['output_height']
            max_width = self.config['max_width']
            
            # Resize maintaining aspect ratio
            ratio = target_height / h
            new_w = int(w * ratio)
            new_w = min(max(new_w, 32), max_width)
            
            resized = cv2.resize(line_img, (new_w, target_height),
                               interpolation=cv2.INTER_CUBIC)
            
            # Pad to fixed width
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
            
            assert resized.shape == (target_height, max_width), \
                f"Unexpected shape: {resized.shape}"
            
            return resized
        
        except Exception as e:
            logging.error(f"Error normalizing line: {e}")
            return None
    
    def extract_and_save_lines(self, img, lines, page_output_dir, 
                               book_name, page_name, page_idx):
        """
        Extract individual lines and save as images in page folder
        
        Output structure:
        page_output_dir/
        ├── page_1_line_1.png
        ├── page_1_line_2.png
        ├── page_1_line_3.png
        └── ...
        
        Args:
            img: Full page image (preprocessed)
            lines: List of (y_start, y_end) tuples
            page_output_dir: Where to save this page's line images
            book_name: Book identifier
            page_name: Page identifier
            page_idx: Page number
        
        Returns:
            Tuple of (line_count, metadata_list)
        """
        
        line_count = 0
        line_metadata = []
        
        padding = self.config['padding']
        
        for line_idx, (y1, y2) in enumerate(lines, 1):  # Start from 1
            try:
                # Add padding
                y1_padded = max(0, y1 - padding)
                y2_padded = min(img.shape[0], y2 + padding)
                
                # Extract line from preprocessed image
                line_img = img[y1_padded:y2_padded, :]
                
                # Deskew if needed
                line_img = self.deskew_line(line_img)
                
                # Remove noise if needed
                line_img = self.remove_small_noise(line_img)
                
                # Normalize
                normalized = self.normalize_line(line_img)
                
                if normalized is None:
                    continue
                
                # Generate filename with page_number_line_number format
                # Examples: page_1_line_1.png, page_1_line_2.png, etc.
                line_filename = f"page_{page_idx}_line_{line_idx}.png"
                line_path = os.path.join(page_output_dir, line_filename)
                
                # Save with no compression (lossless)
                compression = self.config['save_quality']
                success = cv2.imwrite(
                    line_path,
                    normalized,
                    [cv2.IMWRITE_PNG_COMPRESSION, compression]
                )
                
                if not success:
                    logging.warning(f"Failed to save {line_filename}")
                    continue
                
                # Create metadata entry
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
                    "text": "",  # Will be filled during transcription
                    "confidence": 0.0
                }
                
                line_metadata.append(metadata_entry)
                line_count += 1
                
            except Exception as e:
                logging.error(f"Error extracting line {line_idx}: {e}")
                continue
        
        return line_count, line_metadata
    
    def process_page(self, page_path, base_output_dir, book_name, page_idx):
        """
        Process single page and extract lines into page folder
        
        Args:
            page_path: Path to preprocessed page image
            base_output_dir: Base output directory
            book_name: Book identifier
            page_idx: Page number
        
        Returns:
            Tuple of (line_count, metadata_list)
        """
        
        try:
            # Load preprocessed page
            img = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                logging.error(f"Failed to load: {page_path}")
                return 0, []
            
            # Create page-specific folder
            page_folder_name = f"page_{page_idx}"
            page_folder_path = os.path.join(base_output_dir, page_folder_name)
            
            Path(page_folder_path).mkdir(parents=True, exist_ok=True)
            
            # Detect lines
            lines = self.detect_lines(img)
            
            if not lines:
                logging.warning(f"No lines detected in {page_path}")
                return 0, []
            
            # Extract and save
            page_name = Path(page_path).stem
            line_count, metadata = self.extract_and_save_lines(
                img, lines, page_folder_path,
                book_name, page_name, page_idx
            )
            
            return line_count, metadata
        
        except Exception as e:
            logging.error(f"Error processing page {page_path}: {e}")
            return 0, []
    
    def process_all_pages(self, input_dir, output_dir, book_name=None):
        """
        Process all pages in directory
        
        Creates folder structure:
        output_dir/
        ├── page_1/
        │   ���── page_1_line_1.png
        │   ├── page_1_line_2.png
        │   └── ...
        ├── page_2/
        │   ├── page_2_line_1.png
        │   ├── page_2_line_2.png
        │   └── ...
        └── ...
        
        Args:
            input_dir: Directory with preprocessed page images
            output_dir: Where to save organized line images
            book_name: Book identifier (if None, use parent directory name)
        
        Returns:
            Total number of lines extracted
        """
        
        # Validate input
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            logging.error(f"Input directory not found: {input_dir}")
            return 0
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine book name
        if book_name is None:
            book_name = input_path.name
        
        # Display header
        print(f"\n{'='*70}")
        print(f"✂️  LINE SEGMENTATION (Organized by Page)")
        print(f"{'='*70}")
        print(f"   Input:  {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   Book:   {book_name}")
        print(f"\n   Output Structure:")
        print(f"   output_dir/")
        print(f"   ├── page_1/")
        print(f"   │   ├── page_1_line_1.png")
        print(f"   │   ├── page_1_line_2.png")
        print(f"   │   └── ...")
        print(f"   ├── page_2/")
        print(f"   │   ├── page_2_line_1.png")
        print(f"   │   └── ...")
        print(f"   └── ...\n")
        
        # Find page images
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        page_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not page_files:
            print(f"❌ No images found in {input_dir}")
            logging.warning(f"No images found in {input_dir}")
            return 0
        
        print(f"   Found {len(page_files)} pages\n")
        
        total_lines = 0
        all_metadata = []
        
        for page_idx, page_file in enumerate(page_files, 1):
            print(f"   Processing {page_idx:3d}/{len(page_files)}: {page_file.name}...", end=" ")
            
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
                
            except Exception as e:
                print(f"❌ Error: {e}")
                logging.error(f"Error processing page {page_idx}: {e}")
                continue
        
        # Save metadata
        self.metadata = all_metadata
        
        print(f"\n{'='*70}")
        print(f"✅ SEGMENTATION COMPLETE")
        print(f"{'='*70}")
        print(f"   Total lines: {total_lines}")
        print(f"   Output structure:")
        
        # Show created folders
        page_folders = sorted([f for f in output_path.iterdir() if f.is_dir()])
        for i, folder in enumerate(page_folders[:5]):
            num_files = len(list(folder.glob("*.png")))
            print(f"      {folder.name}/ → {num_files} lines")
        
        if len(page_folders) > 5:
            print(f"      ... and {len(page_folders) - 5} more folders")
        
        print(f"\n   Output: {output_dir}")
        print(f"{'='*70}\n")
        
        logging.info(f"Segmentation complete: {total_lines} lines extracted in organized folders")
        
        return total_lines
    
    def save_metadata(self, output_file):
        """
        Save segmentation metadata as JSON
        
        Args:
            output_file: Path to save metadata JSON
        """
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Metadata saved: {output_file}")
        print(f"   Total entries: {len(self.metadata)}")
        logging.info(f"Metadata saved: {output_file} with {len(self.metadata)} entries")
    
    def generate_summary_report(self, output_dir, report_file=None):
        """
        Generate summary report of segmentation
        
        Args:
            output_dir: Output directory with page folders
            report_file: Where to save report (optional)
        """
        
        output_path = Path(output_dir)
        page_folders = sorted([f for f in output_path.iterdir() if f.is_dir()])
        
        report = f"""
{'='*70}
LINE SEGMENTATION SUMMARY REPORT
{'='*70}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Output Directory: {output_dir}

Folder Structure:
"""
        
        total_lines = 0
        for folder in page_folders:
            line_files = list(folder.glob("*.png"))
            num_lines = len(line_files)
            total_lines += num_lines
            
            report += f"\n  {folder.name}/ → {num_lines} lines"
            
            # Show first few files
            for i, line_file in enumerate(sorted(line_files)[:3]):
                report += f"\n    ├─ {line_file.name}"
            
            if num_lines > 3:
                report += f"\n    └─ ... and {num_lines - 3} more"
        
        report += f"""

Statistics:
  Total page folders: {len(page_folders)}
  Total line images: {total_lines}
  Average lines per page: {total_lines / len(page_folders):.1f if page_folders else 0}

Metadata:
  File: metadata.json
  Total entries: {len(self.metadata)}

{'='*70}
"""
        
        print(report)
        
        if report_file:
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"Report saved: {report_file}")


def main():
    """Main entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Segment document images into lines organized by page folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python 03_line_segmentation.py -i "preprocessed_pages" -o "segmented_lines"
  
  # With book name
  python 03_line_segmentation.py -i "preprocessed" -o "lines" -b "book_4"
  
  # Full paths with metadata
  python 03_line_segmentation.py \\
    -i "C:\\data\\preprocessed" \\
    -o "C:\\data\\lines" \\
    -b "PORCONES" \\
    -m "C:\\data\\metadata.json"
  
  # With custom parameters
  python 03_line_segmentation.py -i "input" -o "output" \\
    --min-height 20 \\
    --gap-threshold 0.03 \\
    --no-deskew
  
  # Generate report
  python 03_line_segmentation.py -i "input" -o "output" -r "report.txt"
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory with preprocessed images'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for organized line images'
    )
    
    parser.add_argument(
        '-b', '--book-name',
        default=None,
        help='Book name for identification (optional)'
    )
    
    parser.add_argument(
        '-m', '--metadata',
        default=None,
        help='Path to save metadata JSON (optional)'
    )
    
    parser.add_argument(
        '-r', '--report',
        default=None,
        help='Path to save summary report (optional)'
    )
    
    parser.add_argument(
        '--min-height',
        type=int,
        default=10,
        help='Minimum line height in pixels'
    )
    
    parser.add_argument(
        '--gap-threshold',
        type=float,
        default=0.03,
        help='Gap threshold as fraction of max (default: 0.02)'
    )
    
    parser.add_argument(
        '--no-deskew',
        action='store_true',
        help='Disable skew correction'
    )
    
    parser.add_argument(
        '--no-denoise',
        action='store_true',
        help='Disable noise removal'
    )
    
    args = parser.parse_args()
    
    # Display header
    print("\n" + "="*70)
    print("  LINE SEGMENTATION TOOL (Organized by Page)")
    print("="*70)
    
    try:
        # Create config
        config = {
            'min_line_height': args.min_height,
            'gap_threshold': args.gap_threshold,
            'padding': 2,
            'output_height': 64,
            'max_width': 1024,
            'save_quality': 0,
            'deskew': not args.no_deskew,
            'remove_noise': not args.no_denoise,
            'noise_size': 100
        }
        
        # Initialize segmenter
        segmenter = LineSegmenter(config)
        
        # Process images
        total_lines = segmenter.process_all_pages(
            input_dir=args.input,
            output_dir=args.output,
            book_name=args.book_name
        )
        
        # Save metadata
        if args.metadata:
            segmenter.save_metadata(args.metadata)
        else:
            metadata_path = os.path.join(args.output, "metadata.json")
            segmenter.save_metadata(metadata_path)
        
        # Generate report
        if args.report:
            segmenter.generate_summary_report(args.output, args.report)
        else:
            segmenter.generate_summary_report(args.output)
        
        if total_lines > 0:
            print("✅ Line segmentation completed successfully!")
            return 0
        else:
            print("❌ No lines were extracted")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1


# ============ MAIN ============
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)