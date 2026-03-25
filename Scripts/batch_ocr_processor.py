"""
Batch OCR Processing System
Process entire folders of images with EasyOCR
Supports multiple export formats and advanced filtering

Usage:
    python batch_ocr_processor.py -i ./images -o ./results
    python batch_ocr_processor.py -i ./images -o ./results --gpu
    python batch_ocr_processor.py -i folder1 folder2 -o ./results --multiple
    python batch_ocr_processor.py -i ./images -o ./results --confidence 0.7
"""

import cv2
import easyocr
import matplotlib.pyplot as plt
import os
import json
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import warnings
import argparse
import sys
from typing import List, Dict, Tuple, Optional
import logging

warnings.filterwarnings('ignore')

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_ocr.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============ BATCH OCR PROCESSOR CLASS ============
class BatchOCRProcessor:
    """
    Production-ready batch OCR processor for folders of images
    
    Features:
    - Process entire folders
    - Progress tracking with tqdm
    - Error handling and recovery
    - Metadata tracking
    - Multiple export formats (JSON, CSV, TXT, PNG)
    - Statistics and reporting
    """
    
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """
        Initialize OCR processor
        
        Args:
            languages: List of language codes (default: ['en'])
            gpu: Whether to use GPU (default: False)
        """
        if languages is None:
            languages = ['en']
        
        logger.info(f"Initializing EasyOCR with languages: {languages}, GPU: {gpu}")
        print(f"🔄 Initializing EasyOCR reader (GPU: {gpu})...")
        
        try:
            self.reader = easyocr.Reader(languages, gpu=gpu)
            print("✓ EasyOCR reader ready!")
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
        
        self.results = []
        self.failed_images = []
        self.processing_stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'total_text_regions': 0,
            'processing_time': 0,
            'start_time': None,
            'end_time': None
        }
    
    def process_single_image(self, 
                            image_path: str, 
                            output_base_dir: Optional[str] = None, 
                            padding: int = 5, 
                            save_cropped: bool = True) -> Dict:
        """
        Process a single image file
        
        Args:
            image_path: Path to image file
            output_base_dir: Directory to save outputs
            padding: Padding around detected text
            save_cropped: Whether to save cropped text regions
        
        Returns:
            dict: Processing results
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            # Create output directory
            if output_base_dir:
                output_dir = Path(output_base_dir) / image_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = None
            
            # Convert to RGB for OCR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            results = self.reader.readtext(image_rgb)
            
            # Process results
            text_regions = []
            for idx, (bbox, text, confidence) in enumerate(results):
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                y_max = min(image.shape[0], y_max + padding)
                
                # Crop
                cropped = image[y_min:y_max, x_min:x_max]
                
                region_data = {
                    'region_id': idx,
                    'text': text.strip(),
                    'confidence': float(confidence),
                    'bbox': {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'width': x_max - x_min,
                        'height': y_max - y_min
                    }
                }
                
                # Save cropped region if requested
                if save_cropped and output_dir:
                    cropped_path = output_dir / f"region_{idx:04d}.png"
                    cv2.imwrite(str(cropped_path), cropped)
                    region_data['cropped_image_path'] = str(cropped_path)
                
                text_regions.append(region_data)
            
            # Create result object
            result = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'image_size': {
                    'width': image.shape[1],
                    'height': image.shape[0]
                },
                'status': 'success',
                'total_regions': len(text_regions),
                'text_regions': text_regions,
                'full_text': ' '.join([r['text'] for r in text_regions]),
                'average_confidence': float(np.mean([r['confidence'] for r in text_regions])) if text_regions else 0,
                'output_directory': str(output_dir) if output_dir else None,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed: {image_path.name} ({len(text_regions)} regions)")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to process {image_path}: {error_msg}")
            error_result = {
                'image_path': str(image_path),
                'status': 'failed',
                'error': error_msg,
                'processing_timestamp': datetime.now().isoformat()
            }
            return error_result
    
    def process_folder(self, 
                      folder_path: str, 
                      output_base_dir: Optional[str] = None, 
                      file_patterns: List[str] = None) -> List[Dict]:
        """
        Process all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            output_base_dir: Base directory for outputs
            file_patterns: Image file patterns to match
        
        Returns:
            list: Results for all processed images
        """
        if file_patterns is None:
            file_patterns = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.PNG', '*.JPG', '*.JPEG']
        
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            print(f"❌ Folder not found: {folder_path}")
            return []
        
        # Find all image files
        image_files = []
        for pattern in file_patterns:
            image_files.extend(folder_path.glob(f"**/{pattern}"))
        
        # Remove duplicates and sort
        image_files = list(set(image_files))
        image_files.sort()
        
        if not image_files:
            logger.warning(f"No image files found in {folder_path}")
            print(f"❌ No image files found in {folder_path}")
            return []
        
        logger.info(f"Found {len(image_files)} images to process")
        print(f"✓ Found {len(image_files)} images to process")
        
        # Create output directory
        if output_base_dir:
            output_base_dir = Path(output_base_dir)
            output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images with progress bar
        self.processing_stats['start_time'] = datetime.now()
        self.processing_stats['total_images'] = len(image_files)
        
        self.results = []
        
        for image_file in tqdm(image_files, desc="Processing images", unit="img"):
            result = self.process_single_image(image_file, output_base_dir)
            self.results.append(result)
            
            if result['status'] == 'success':
                self.processing_stats['successful'] += 1
                self.processing_stats['total_text_regions'] += result['total_regions']
            else:
                self.processing_stats['failed'] += 1
                self.failed_images.append(result)
        
        self.processing_stats['end_time'] = datetime.now()
        time_delta = self.processing_stats['end_time'] - self.processing_stats['start_time']
        self.processing_stats['processing_time'] = time_delta.total_seconds()
        
        logger.info(f"Folder processing complete: {self.processing_stats['successful']}/{len(image_files)} successful")
        return self.results
    
    def save_results_json(self, output_path: str) -> None:
        """Save results as JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        print(f"✓ Results saved to {output_path}")
    
    def save_full_transcription(self, output_path: str) -> None:
        """Save all detected text as plain text"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in self.results:
                if result['status'] == 'success':
                    f.write(f"=== {result['image_name']} ===\n")
                    f.write(result['full_text'] + "\n\n")
        
        logger.info(f"Transcription saved to {output_path}")
        print(f"✓ Transcription saved to {output_path}")
    
    def save_to_csv(self, output_path: str) -> None:
        """Save results to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Region', 'Text', 'Confidence', 'X_Min', 'Y_Min', 'Width', 'Height'])
            
            for result in self.results:
                if result['status'] == 'success':
                    for region in result['text_regions']:
                        writer.writerow([
                            result['image_name'],
                            region['region_id'],
                            region['text'],
                            f"{region['confidence']:.3f}",
                            region['bbox']['x_min'],
                            region['bbox']['y_min'],
                            region['bbox']['width'],
                            region['bbox']['height']
                        ])
        
        logger.info(f"CSV saved to {output_path}")
        print(f"✓ CSV saved to {output_path}")
    
    def save_summary_report(self, output_path: str) -> None:
        """Save processing summary report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.processing_stats
        
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║              BATCH OCR PROCESSING SUMMARY REPORT               ║
╚════════════════════════════════════════════════════════════════╝

📊 PROCESSING STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Total images:           {stats['total_images']}
  • Successfully processed: {stats['successful']}
  • Failed:                 {stats['failed']}
  • Success rate:           {(stats['successful']/stats['total_images']*100):.1f}%
  • Total text regions:     {stats['total_text_regions']}
  • Processing time:        {stats['processing_time']:.2f} seconds
  • Avg time per image:     {(stats['processing_time']/stats['total_images']):.2f} seconds

📈 CONFIDENCE STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        confidences = []
        for result in self.results:
            if result['status'] == 'success' and result['average_confidence'] > 0:
                confidences.append(result['average_confidence'])
        
        if confidences:
            report += f"""  • Average confidence: {np.mean(confidences):.2%}
  • Min confidence:     {np.min(confidences):.2%}
  • Max confidence:     {np.max(confidences):.2%}
  • Std deviation:      {np.std(confidences):.2%}
"""
        
        report += f"""
📁 DETAILED RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for result in self.results:
            if result['status'] == 'success':
                report += f"""
  ✓ {result['image_name']}
    - Regions: {result['total_regions']}
    - Avg confidence: {result['average_confidence']:.2%}
    - Text length: {len(result['full_text'])} chars
"""
            else:
                report += f"""
  ✗ {Path(result['image_path']).name}
    - Error: {result['error']}
"""
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Also print to console
        print(report)
        logger.info(f"Report saved to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return self.processing_stats
    
    def get_failed_images(self) -> List[Dict]:
        """Get list of failed images"""
        return self.failed_images


# ============ MULTI FOLDER PROCESSOR CLASS ============
class MultiFolderOCRProcessor:
    """Process multiple folders in batch"""
    
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        self.processor = BatchOCRProcessor(languages=languages, gpu=gpu)
        self.folder_results = {}
    
    def process_folders(self, folder_list: List[str], output_base_dir: str) -> Dict:
        """
        Process multiple folders
        
        Args:
            folder_list: List of folder paths
            output_base_dir: Base output directory
        
        Returns:
            dict: Results for all folders
        """
        output_base_dir = Path(output_base_dir)
        
        logger.info(f"Processing {len(folder_list)} folders")
        print(f"\n📁 Processing {len(folder_list)} folders...\n")
        
        for folder in folder_list:
            folder = Path(folder)
            folder_name = folder.name
            
            print(f"\n{'='*70}")
            print(f"Processing: {folder_name}")
            print(f"{'='*70}")
            
            # Create subfolder for results
            folder_output = output_base_dir / folder_name
            
            # Process
            results = self.processor.process_folder(folder, folder_output)
            
            # Save results
            self.processor.save_results_json(folder_output / "results.json")
            self.processor.save_full_transcription(folder_output / "transcription.txt")
            self.processor.save_to_csv(folder_output / "results.csv")
            self.processor.save_summary_report(folder_output / "report.txt")
            
            self.folder_results[folder_name] = {
                'results': results,
                'stats': self.processor.get_statistics().copy()
            }
            
            logger.info(f"Completed folder: {folder_name}")
        
        return self.folder_results


# ============ ADVANCED BATCH PROCESSOR CLASS ============
class AdvancedBatchProcessor:
    """
    Advanced batch processor with filtering and advanced features
    """
    
    def __init__(self, gpu: bool = False):
        self.processor = BatchOCRProcessor(gpu=gpu)
    
    def process_with_filter(self, 
                           folder_path: str, 
                           output_dir: str,
                           min_file_size: int = 100,
                           max_file_size: Optional[int] = None,
                           confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Process images with filtering
        
        Args:
            folder_path: Folder to process
            output_dir: Output directory
            min_file_size: Minimum file size in KB
            max_file_size: Maximum file size in KB
            confidence_threshold: Only keep results above this confidence
        
        Returns:
            list: Filtered results
        """
        folder_path = Path(folder_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing with filters - Min size: {min_file_size}KB, Confidence: {confidence_threshold}")
        
        # Find images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.PNG', '*.JPG']:
            image_files.extend(folder_path.glob(f"**/{ext}"))
        
        logger.info(f"Found {len(image_files)} images, filtering...")
        print(f"Found {len(image_files)} images, filtering...")
        
        # Filter by size
        filtered_files = []
        for img_file in image_files:
            size_kb = img_file.stat().st_size / 1024
            if min_file_size <= size_kb <= (max_file_size or float('inf')):
                filtered_files.append(img_file)
        
        logger.info(f"After filtering: {len(filtered_files)} images")
        print(f"After filtering: {len(filtered_files)} images")
        
        # Process
        results = []
        for img_file in tqdm(filtered_files, desc="Processing", unit="img"):
            result = self.processor.process_single_image(img_file, output_dir)
            
            # Filter by confidence
            if result['status'] == 'success':
                result['text_regions'] = [
                    r for r in result['text_regions'] 
                    if r['confidence'] >= confidence_threshold
                ]
            
            results.append(result)
        
        return results


# ============ CLI ARGUMENT PARSER ============
def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface argument parser"""
    parser = argparse.ArgumentParser(
        description='Batch OCR Processing for Multiple Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single folder
  python batch_ocr_processor.py -i ./images -o ./results
  
  # Process with GPU
  python batch_ocr_processor.py -i ./images -o ./results --gpu
  
  # Process multiple folders
  python batch_ocr_processor.py -i folder1 folder2 folder3 -o ./results --multiple
  
  # Process with confidence filter
  python batch_ocr_processor.py -i ./images -o ./results --confidence 0.7
  
  # Process with advanced filtering
  python batch_ocr_processor.py -i ./images -o ./results --min-size 50 --max-size 5000
  
  # Specify language
  python batch_ocr_processor.py -i ./images -o ./results --lang en es fr
        """
    )
    
    parser.add_argument('-i', '--input', 
                       type=str, 
                       nargs='+', 
                       required=True,
                       help='Input folder(s) or image file(s)')
    
    parser.add_argument('-o', '--output', 
                       type=str, 
                       default='./batch_ocr_output',
                       help='Output directory (default: ./batch_ocr_output)')
    
    parser.add_argument('-m', '--multiple', 
                       action='store_true',
                       help='Process multiple folders separately')
    
    parser.add_argument('--gpu', 
                       action='store_true',
                       help='Use GPU for processing')
    
    parser.add_argument('--confidence', 
                       type=float, 
                       default=0.0,
                       help='Minimum confidence threshold (default: 0.0)')
    
    parser.add_argument('--min-size', 
                       type=int, 
                       default=100,
                       help='Minimum file size in KB (default: 100)')
    
    parser.add_argument('--max-size', 
                       type=int, 
                       default=None,
                       help='Maximum file size in KB (default: unlimited)')
    
    parser.add_argument('--lang', 
                       type=str, 
                       nargs='+',
                       default=['en'],
                       help='Language codes (default: en)')
    
    parser.add_argument('--filter', 
                       action='store_true',
                       help='Enable advanced filtering')
    
    parser.add_argument('--export-csv', 
                       action='store_true',
                       default=True,
                       help='Export results to CSV')
    
    parser.add_argument('--no-save-cropped', 
                       action='store_true',
                       help='Do not save cropped text regions')
    
    parser.add_argument('--verbose', 
                       action='store_true',
                       help='Enable verbose logging')
    
    return parser


# ============ MAIN FUNCTION ============
def main():
    """Main entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("BATCH OCR PROCESSOR")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"  • Input: {args.input}")
    print(f"  • Output: {args.output}")
    print(f"  • Languages: {args.lang}")
    print(f"  • GPU: {args.gpu}")
    print(f"  • Confidence threshold: {args.confidence}")
    
    if args.filter:
        print(f"  • File size filter: {args.min_size}KB - {args.max_size or 'unlimited'}KB")
    
    print("\n" + "="*70 + "\n")
    
    try:
        # Initialize processor
        if args.filter:
            processor = AdvancedBatchProcessor(gpu=args.gpu)
        else:
            processor = BatchOCRProcessor(languages=args.lang, gpu=args.gpu)
        
        # Process folders
        if args.multiple and len(args.input) > 1:
            # Multiple folders
            multi_processor = MultiFolderOCRProcessor(languages=args.lang, gpu=args.gpu)
            multi_processor.process_folders(args.input, args.output)
        else:
            # Single folder
            input_folder = args.input[0]
            
            if args.filter:
                results = processor.process_with_filter(
                    input_folder,
                    args.output,
                    min_file_size=args.min_size,
                    max_file_size=args.max_size,
                    confidence_threshold=args.confidence
                )
            else:
                results = processor.processor.process_folder(input_folder, args.output)
            
            # Save results
            output_path = Path(args.output)
            processor.processor.save_results_json(output_path / "results.json")
            processor.processor.save_full_transcription(output_path / "transcription.txt")
            
            if args.export_csv:
                processor.processor.save_to_csv(output_path / "results.csv")
            
            processor.processor.save_summary_report(output_path / "report.txt")
        
        # Print completion message
        print("\n" + "="*70)
        print("✓ PROCESSING COMPLETE!")
        print("="*70)
        print(f"\n📁 Results saved to: {args.output}")
        print(f"\n📊 Files generated:")
        print(f"  • results.json (detailed results)")
        print(f"  • transcription.txt (full text)")
        
        if args.export_csv:
            print(f"  • results.csv (spreadsheet)")
        
        print(f"  • report.txt (summary report)")
        print(f"  • region_*.png (cropped text regions)")
        print(f"\n✓ See batch_ocr.log for detailed logs\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()