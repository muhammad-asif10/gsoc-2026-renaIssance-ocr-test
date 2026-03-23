# scripts/01_separate_columns.py

import cv2
import numpy as np
import os
from pathlib import Path
import json
import logging

logging.basicConfig(filename='logs/column_separation.log', level=logging.INFO)

class ColumnSeparator:
    """Detect and separate columns in historical documents"""
    
    def __init__(self, output_dir="data/separated_columns"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.metadata = []
    
    def detect_columns(self, img, min_width=50, gap_threshold=0.02):
        """
        Detect vertical columns in page
        
        Method:
        1. Binarize image
        2. Calculate vertical projection (sum pixels per column)
        3. Find gaps to separate columns
        4. Group consecutive columns
        """
        
        # Binarize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Vertical projection
        vertical_sum = np.sum(binary, axis=0)
        
        # Normalize
        max_sum = np.max(vertical_sum)
        if max_sum == 0:
            return []
        
        # Threshold for detecting text columns
        threshold = gap_threshold * max_sum
        text_cols = np.where(vertical_sum > threshold)[0]
        
        if len(text_cols) == 0:
            return []
        
        # Group consecutive columns
        columns = []
        x_start = text_cols[0]
        x_prev = text_cols[0]
        
        for x in text_cols[1:]:
            if x != x_prev + 1:  # Gap detected
                col_width = x_prev - x_start
                
                if col_width >= min_width:
                    columns.append((x_start, x_prev))
                
                x_start = x
            
            x_prev = x
        
        # Don't forget last column
        col_width = x_prev - x_start
        if col_width >= min_width:
            columns.append((x_start, x_prev))
        
        return columns
    
    def extract_columns(self, img, columns, padding=5):
        """
        Extract individual columns from page
        
        Args:
            img: Full page image
            columns: List of (x1, x2) tuples
            padding: Padding around column
        
        Returns:
            List of column images
        """
        
        column_images = []
        
        for col_idx, (x1, x2) in enumerate(columns):
            # Add padding
            x1 = max(0, x1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            
            # Extract column
            column_img = img[:, x1:x2]
            
            column_images.append({
                'image': column_img,
                'index': col_idx,
                'x_start': x1,
                'x_end': x2,
                'width': x2 - x1
            })
        
        return column_images
    
    def process_page(self, image_path, book_name, page_number):
        """
        Process single page and separate columns
        
        Args:
            image_path: Path to page image
            book_name: Book identifier
            page_number: Page number
        
        Returns:
            List of extracted columns
        """
        
        print(f"   Processing page {page_number}...", end=" ")
        
        # Load image
        img = cv2.imread(image_path)
        
        if img is None:
            print("❌ Failed to load")
            logging.error(f"Failed to load: {image_path}")
            return []
        
        # Detect columns
        columns = self.detect_columns(img)
        
        if not columns:
            print("⚠️  No columns detected")
            logging.warning(f"No columns detected in {image_path}")
            return []
        
        print(f"✅ Found {len(columns)} columns")
        
        # Extract columns
        extracted_columns = self.extract_columns(img, columns)
        
        # Save each column
        for col_data in extracted_columns:
            column_img = col_data['image']
            col_idx = col_data['index']
            
            # Filename
            col_filename = f"{book_name}_page_{page_number:04d}_col_{col_idx}.png"
            col_path = os.path.join(self.output_dir, col_filename)
            
            # Save
            cv2.imwrite(col_path, column_img)
            
            # Metadata
            self.metadata.append({
                "image": col_filename,
                "book": book_name,
                "page": page_number,
                "column": col_idx,
                "original_width": col_data['width'],
                "original_height": column_img.shape[0],
                "text": ""  # Will fill during transcription
            })
        
        logging.info(f"Extracted {len(columns)} columns from page {page_number}")
        
        return extracted_columns
    
    def process_all_pages(self, input_dir, book_name):
        """Process all pages in directory"""
        
        print(f"\n🔀 COLUMN SEPARATION: {book_name}")
        print(f"   Input: {input_dir}\n")
        
        page_files = sorted([f for f in os.listdir(input_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not page_files:
            print(f"❌ No images found")
            return 0
        
        total_columns = 0
        
        for page_num, page_file in enumerate(page_files, 1):
            page_path = os.path.join(input_dir, page_file)
            
            columns = self.process_page(page_path, book_name, page_num)
            total_columns += len(columns)
        
        print(f"\n✅ Extraction complete: {total_columns} columns")
        
        return total_columns
    
    def save_metadata(self, output_file):
        """Save column metadata"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Metadata saved: {output_file}")


# ============ MAIN ============
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Separate columns in historical documents')
    parser.add_argument('-i', '--input', required=True, help='Input directory')
    parser.add_argument('-o', '--output', default='data/separated_columns', help='Output directory')
    parser.add_argument('-b', '--book', required=True, help='Book name')
    
    args = parser.parse_args()
    
    separator = ColumnSeparator(args.output)
    total = separator.process_all_pages(args.input, args.book)
    separator.save_metadata(f"{args.output}/metadata.json")
    
    print(f"\n✅ Total columns extracted: {total}")