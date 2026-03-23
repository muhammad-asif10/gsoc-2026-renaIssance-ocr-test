import cv2
import numpy as np
import os
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
from matplotlib import pyplot as plt

logging.basicConfig(filename='logs/preprocessing.log', level=logging.INFO)

class HistoricalDocumentPreprocessor:
    """
    Preprocess historical documents with multiple enhancement techniques
    Optimized for 17th-century Spanish texts
    """
    
    def __init__(self, config_file="config/preprocessing_config.yaml"):
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.prep_config = self.config['preprocessing']
        print("✅ Configuration loaded")
        print(f"   Denoise method: {self.prep_config['denoise']['method']}")
        print(f"   Contrast method: {self.prep_config['contrast']['method']}")
    
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
        
        # Create illumination map using morphological closing
        kernel_size = self.prep_config['lighting_correction']['kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Large morphological closing = illumination map
        illumination_map = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Divide to remove illumination
        # This is image division in OpenCV
        corrected = cv2.divide(img, illumination_map, scale=255)
        
        return corrected
    
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
        4. (Optional) Binarize
        """
        
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            logging.error(f"Failed to load image: {image_path}")
            return None
        
        # Store original for comparison
        original = img.copy()
        
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
        
        # Result is grayscale (good for neural networks)
        # NOT binary (models work better with grayscale)
        
        return img
    
    def _save_intermediate(self, original_path, processed_img, stage_name):
        """Save intermediate processing steps for debugging"""
        
        filename = Path(original_path).stem
        output_path = f"data/preprocessing_debug/{stage_name}_{filename}.png"
        
        Path("data/preprocessing_debug").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, processed_img)
    
    def process_all_pages(self, input_dir, output_dir, book_name):
        """
        Process all pages in a book
        
        Args:
            input_dir: Directory with raw page images
            output_dir: Where to save preprocessed pages
            book_name: Name for identification
        """
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 PREPROCESSING: {book_name}")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        
        page_files = sorted([f for f in os.listdir(input_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not page_files:
            print(f"❌ No images found in {input_dir}")
            return 0
        
        print(f"   Found {len(page_files)} pages\n")
        
        processed_count = 0
        failed_count = 0
        
        for idx, page_file in enumerate(page_files):
            page_path = os.path.join(input_dir, page_file)
            
            # Process
            processed = self.process_page(page_path, save_intermediate=False)
            
            if processed is None:
                print(f"   ❌ Page {idx + 1}: Failed to process")
                failed_count += 1
                continue
            
            # Save
            output_filename = f"{book_name}_{page_file}"
            output_path = os.path.join(output_dir, output_filename)
            
            compression = self.config['output']['save_quality']
            cv2.imwrite(output_path, processed, [cv2.IMWRITE_PNG_COMPRESSION, compression])
            
            processed_count += 1
            
            # Progress
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"   ✅ Processed {idx + 1}/{len(page_files)}")
        
        print(f"\n✅ Preprocessing complete: {processed_count} pages")
        if failed_count > 0:
            print(f"⚠️  Failed: {failed_count} pages")
        
        logging.info(f"Preprocessed {book_name}: {processed_count}/{len(page_files)}")
        
        return processed_count


# ============ MAIN ============
if __name__ == "__main__":
    
    preprocessor = HistoricalDocumentPreprocessor("config/preprocessing_config.yaml")
    
    # Process books
    books = {
        "book_4": "data\books\PORCONES_23_5_1628.pdf",
        "book_5": "data\books\PORCONES_228_38_1646.pdf",
    }
    
    for book_name, input_dir in books.items():
        if os.path.exists(input_dir):
            preprocessor.process_all_pages(
                input_dir=input_dir,
                output_dir="data/preprocessed_pages",
                book_name=book_name
            )
        else:
            print(f"⚠️  Directory not found: {input_dir}")