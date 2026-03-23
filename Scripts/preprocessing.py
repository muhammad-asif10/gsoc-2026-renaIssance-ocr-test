# Scripts/preprocessing.py

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
import yaml

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=f'logs/preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HistoricalDocumentPreprocessor:
    """
    Preprocess historical documents with multiple enhancement techniques
    Optimized for 17th-century Spanish texts
    """
    
    def __init__(self, config_file="config/preprocessing_config.yaml"):
        """Initialize preprocessor with configuration"""
        
        # Load configuration
        if not os.path.exists(config_file):
            print(f"⚠️  Config file not found: {config_file}")
            print("   Using default configuration...")
            self.config = self._get_default_config()
        else:
            try:
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
                print(f"✅ Configuration loaded from: {config_file}")
            except Exception as e:
                print(f"⚠️  Error loading config: {e}")
                self.config = self._get_default_config()
        
        self.prep_config = self.config['preprocessing']
        
        print(f"   Denoise method: {self.prep_config['denoise']['method']}")
        print(f"   Lighting correction: {self.prep_config['lighting_correction']['enabled']}")
        print(f"   Contrast method: {self.prep_config['contrast']['method']}")
    
    def _get_default_config(self):
        """Return default configuration if file not found"""
        
        return {
            'preprocessing': {
                'denoise': {
                    'method': 'bilateral',
                    'bilateral_d': 9,
                    'bilateral_sigma_color': 75,
                    'bilateral_sigma_space': 75
                },
                'lighting_correction': {
                    'enabled': True,
                    'kernel_size': 101
                },
                'contrast': {
                    'method': 'clahe',
                    'clip_limit': 3.0,
                    'tile_size': 8
                }
            },
            'output': {
                'save_quality': 0
            }
        }
    
    def denoise(self, img):
        """
        Remove noise while preserving text edges
        
        For historical documents: use bilateral filtering
        - Keeps edges sharp
        - Removes noise
        - Good for ink artifacts
        """
        
        method = self.prep_config['denoise']['method']
        
        if method == 'bilateral':
            d = self.prep_config['denoise']['bilateral_d']
            sigma_color = self.prep_config['denoise']['bilateral_sigma_color']
            sigma_space = self.prep_config['denoise']['bilateral_sigma_space']
            
            denoised = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
            
        elif method == 'median':
            denoised = cv2.medianBlur(img, 5)
            
        elif method == 'morphological':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            denoised = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        else:
            denoised = img
        
        return denoised
    
    def fix_lighting(self, img):
        """
        Correct uneven illumination in scanned documents
        
        Problem: Historical pages have shadows, fading, uneven lighting
        Solution: Estimate illumination map and correct it
        """
        
        if not self.prep_config['lighting_correction']['enabled']:
            return img
        
        try:
            # Create illumination map using morphological closing
            kernel_size = self.prep_config['lighting_correction']['kernel_size']
            
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Large morphological closing = illumination map
            illumination_map = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            
            # Divide to remove illumination
            corrected = cv2.divide(img, illumination_map, scale=255)
            
            return corrected
        
        except Exception as e:
            logging.warning(f"Error fixing lighting: {e}")
            return img
    
    def enhance_contrast(self, img):
        """
        Enhance contrast using CLAHE
        
        CLAHE = Contrast Limited Adaptive Histogram Equalization
        - Better than global histogram equalization
        - Handles local variations
        - Prevents over-enhancement
        """
        
        method = self.prep_config['contrast']['method']
        
        if method == 'clahe':
            clip_limit = self.prep_config['contrast']['clip_limit']
            tile_size = self.prep_config['contrast']['tile_size']
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                   tileGridSize=(tile_size, tile_size))
            enhanced = clahe.apply(img)
        
        else:
            enhanced = img
        
        return enhanced
    
    def process_page(self, image_path, save_intermediate=False):
        """
        Complete preprocessing pipeline for one page
        
        Steps:
        1. Denoise
        2. Fix lighting
        3. Enhance contrast
        
        Args:
            image_path: Path to input image
            save_intermediate: Save intermediate steps for debugging
        
        Returns:
            Preprocessed image or None if failed
        """
        
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                logging.error(f"Failed to load image: {image_path}")
                return None
            
            # ============ STEP 1: DENOISE ============
            img = self.denoise(img)
            
            if save_intermediate:
                self._save_intermediate(image_path, img, "01_denoised")
            
            # ============ STEP 2: FIX LIGHTING ============
            img = self.fix_lighting(img)
            
            if save_intermediate:
                self._save_intermediate(image_path, img, "02_lighting_corrected")
            
            # ============ STEP 3: ENHANCE CONTRAST ============
            img = self.enhance_contrast(img)
            
            if save_intermediate:
                self._save_intermediate(image_path, img, "03_contrast_enhanced")
            
            return img
        
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            return None
    
    def _save_intermediate(self, original_path, processed_img, stage_name):
        """Save intermediate processing steps for debugging"""
        
        try:
            filename = Path(original_path).stem
            output_path = Path("data/preprocessing_debug") / f"{stage_name}_{filename}.png"
            
            Path("data/preprocessing_debug").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), processed_img)
        
        except Exception as e:
            logging.warning(f"Error saving intermediate: {e}")
    
    def process_all_pages(self, input_dir, output_dir, book_name=None, 
                         file_pattern="*", save_intermediate=False):
        """
        Process all pages in a directory
        
        Args:
            input_dir: Directory with raw page images
            output_dir: Where to save preprocessed pages
            book_name: Name for identification (optional)
            file_pattern: File pattern to match (default: all images)
            save_intermediate: Save intermediate processing stages
        
        Returns:
            Number of successfully processed pages
        """
        
        # Validate input directory
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            logging.error(f"Input directory not found: {input_dir}")
            return 0
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Display info
        print(f"\n{'='*70}")
        print(f"🎨 PREPROCESSING")
        print(f"{'='*70}")
        print(f"   Input:  {input_dir}")
        print(f"   Output: {output_dir}")
        if book_name:
            print(f"   Book:   {book_name}")
        print(f"   Intermediate save: {save_intermediate}")
        
        # Find images
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        page_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not page_files:
            print(f"\n❌ No images found in {input_dir}")
            logging.warning(f"No images found in {input_dir}")
            return 0
        
        print(f"   Found {len(page_files)} images\n")
        
        processed_count = 0
        failed_count = 0
        
        for idx, page_file in enumerate(page_files, 1):
            print(f"   Processing {idx}/{len(page_files)}: {page_file.name}...", end=" ")
            
            try:
                # Process
                processed = self.process_page(str(page_file), 
                                             save_intermediate=save_intermediate)
                
                if processed is None:
                    print("❌")
                    failed_count += 1
                    continue
                
                # Determine output filename
                if book_name:
                    output_filename = f"{book_name}_{page_file.name}"
                else:
                    output_filename = page_file.name
                
                output_file = output_path / output_filename
                
                # Save
                compression = self.config['output']['save_quality']
                success = cv2.imwrite(
                    str(output_file), 
                    processed,
                    [cv2.IMWRITE_PNG_COMPRESSION, compression]
                )
                
                if success:
                    print("✅")
                    processed_count += 1
                else:
                    print("❌ (failed to save)")
                    failed_count += 1
                
                logging.info(f"Processed: {page_file.name}")
                
            except Exception as e:
                print(f"❌ ({e})")
                logging.error(f"Error processing {page_file}: {e}")
                failed_count += 1
        
        # Summary
        print(f"\n{'='*70}")
        print(f"✅ PREPROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"   ✅ Processed: {processed_count}/{len(page_files)}")
        if failed_count > 0:
            print(f"   ❌ Failed: {failed_count}")
        print(f"   📁 Output: {output_dir}")
        print(f"{'='*70}\n")
        
        logging.info(f"Preprocessing complete: {processed_count}/{len(page_files)} processed")
        
        return processed_count


def main():
    """Main entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Preprocess historical document images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single directory
  python 02_preprocessing.py -i "raw_pages" -o "preprocessed_pages" -b "book_1"
  
  # Process with full paths
  python 02_preprocessing.py -i "C:\\Users\\data\\images" -o "C:\\Users\\data\\output"
  
  # Use custom config
  python 02_preprocessing.py -i "input" -o "output" -c "config/custom.yaml"
  
  # Save intermediate steps for debugging
  python 02_preprocessing.py -i "input" -o "output" --save-intermediate
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing raw images'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for preprocessed images'
    )
    
    parser.add_argument(
        '-b', '--book-name',
        default=None,
        help='Book name for output filename prefix (optional)'
    )
    
    parser.add_argument(
        '-c', '--config',
        default='config/preprocessing_config.yaml',
        help='Path to configuration file (default: config/preprocessing_config.yaml)'
    )
    
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate processing steps for debugging'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose output'
    )
    
    args = parser.parse_args()
    
    # Display header
    print("\n" + "="*70)
    print("  HISTORICAL DOCUMENT PREPROCESSOR")
    print("="*70)
    
    try:
        # Initialize preprocessor
        preprocessor = HistoricalDocumentPreprocessor(args.config)
        
        # Process images
        result = preprocessor.process_all_pages(
            input_dir=args.input,
            output_dir=args.output,
            book_name=args.book_name,
            save_intermediate=args.save_intermediate
        )
        
        if result > 0:
            print("✅ Preprocessing completed successfully!")
            return 0
        else:
            print("❌ No images were processed")
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