# scripts/01_pdf_to_images.py

import fitz  # PyMuPDF
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    filename='logs/preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PDFToImagesConverter:
    """Convert PDF pages to high-quality PNG images"""
    
    def __init__(self, output_dir="data/01_raw_pages", dpi_factor=2.0):
        self.output_dir = output_dir
        self.dpi_factor = dpi_factor
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initialized PDFToImagesConverter with dpi_factor={dpi_factor}")
    
    def convert_pdf(self, pdf_path, book_name):
        """
        Convert PDF to individual page images
        
        Args:
            pdf_path: Path to PDF file
            book_name: Name for output folder
        
        Returns:
            Number of pages converted
        """
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF not found: {pdf_path}")
            logging.error(f"PDF not found: {pdf_path}")
            return 0
        
        print(f"\n📄 Converting PDF: {pdf_path}")
        print(f"   DPI Factor: {self.dpi_factor}x")
        
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            print(f"   Total pages: {total_pages}")
            
            # Create output directory for this book
            book_output_dir = os.path.join(self.output_dir, book_name)
            Path(book_output_dir).mkdir(parents=True, exist_ok=True)
            
            # Convert each page
            for page_num in range(total_pages):
                try:
                    # Get page
                    page = pdf_document[page_num]
                    
                    # Render to pixmap (higher DPI = better quality)
                    mat = fitz.Matrix(self.dpi_factor, self.dpi_factor)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Save as PNG (lossless)
                    output_filename = f"page_{page_num + 1:04d}.png"
                    output_path = os.path.join(book_output_dir, output_filename)
                    
                    pix.save(output_path)
                    
                    # Progress
                    if (page_num + 1) % 50 == 0 or page_num == 0:
                        print(f"   ✅ Converted {page_num + 1}/{total_pages}")
                    
                    logging.info(f"Converted {book_name} page {page_num + 1}")
                    
                except Exception as e:
                    print(f"   ⚠️  Error converting page {page_num + 1}: {e}")
                    logging.warning(f"Error converting {book_name} page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            
            print(f"\n✅ Successfully converted {total_pages} pages to {book_output_dir}")
            logging.info(f"Successfully converted {total_pages} pages from {pdf_path}")
            
            return total_pages
            
        except Exception as e:
            print(f"❌ Error: {e}")
            logging.error(f"Error converting PDF {pdf_path}: {e}")
            return 0
    
    def convert_multiple_books(self, books_dict):
        """
        Convert multiple books
        
        Args:
            books_dict: Dict with {book_name: pdf_path}
        
        Returns:
            Dict with conversion results
        """
        
        results = {}
        total_pages = 0
        
        print("\n" + "="*60)
        print("🚀 STARTING PDF CONVERSION")
        print("="*60)
        
        for book_name, pdf_path in books_dict.items():
            pages = self.convert_pdf(pdf_path, book_name)
            results[book_name] = pages
            total_pages += pages
        
        print("\n" + "="*60)
        print(f"📊 CONVERSION SUMMARY")
        print("="*60)
        
        for book_name, pages in results.items():
            print(f"  {book_name}: {pages} pages")
        
        print(f"\n  Total: {total_pages} pages")
        print(f"  Output: {self.output_dir}")
        print("="*60)
        
        logging.info(f"PDF conversion complete: {total_pages} pages")
        
        return results


# ============ MAIN ============
if __name__ == "__main__":
    
    # Define your books
    books_to_convert = {
        "book_4": r"C:\Open Source\GSOC\org\HumanAI\gsoc-2026-renaissance-ocr-test\GSOC_2026_HumanAI_Test\data\books\BOOK_4.pdf",
        "book_5": r"C:\Open Source\GSOC\org\HumanAI\gsoc-2026-renaissance-ocr-test\GSOC_2026_HumanAI_Test\data\books\BOOK_5.pdf",
        # Add more books
        # "book_1": r"path/to/book_1.pdf",
        # "book_2": r"path/to/book_2.pdf",
    }
    
    # Convert
    converter = PDFToImagesConverter(dpi_factor=2.0)
    results = converter.convert_multiple_books(books_to_convert)