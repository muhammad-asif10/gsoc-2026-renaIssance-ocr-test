# scripts/cleanup_bad_images.py

import cv2
import numpy as np
import os
from pathlib import Path
import logging

logging.basicConfig(filename='logs/cleanup.log', level=logging.INFO)

class BadImageDetector:
    """Detect and remove images with no text content"""
    
    @staticmethod
    def is_bad_image(image_path, brightness_threshold=50, text_threshold=0.05):
        """
        Determine if image is bad (no text content)
        
        Args:
            image_path: Path to image
            brightness_threshold: Max average brightness (too dark = bad)
            text_threshold: Min % of non-white pixels (too blank = bad)
        
        Returns:
            True if bad, False if good
        """
        
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                return True  # Can't read = bad
            
            # Check 1: Too dark (almost all black)
            brightness = np.mean(img)
            if brightness < brightness_threshold:
                return True
            
            # Check 2: Too much white (blank page)
            white_pixels = np.sum(img > 200)
            total_pixels = img.shape[0] * img.shape[1]
            white_ratio = white_pixels / total_pixels
            
            if white_ratio > 0.95:  # More than 95% white = blank
                return True
            
            # Check 3: Almost no text (all black or all white)
            black_pixels = np.sum(img < 50)
            black_ratio = black_pixels / total_pixels
            
            if black_ratio > 0.90:  # More than 90% black = bad
                return True
            
            # Check 4: No meaningful content
            # Calculate histogram to detect uniform images
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            entropy = -np.sum((hist / hist.sum()) * np.log2(hist / hist.sum() + 1e-10))
            
            if entropy < 1.0:  # Very low entropy = uniform/blank
                return True
            
            # All checks passed = good image
            return False
        
        except Exception as e:
            logging.error(f"Error checking {image_path}: {e}")
            return True  # If error, assume bad


def cleanup_directory(image_dir, output_report="cleanup_report.txt"):
    """
    Scan directory and remove bad images
    
    Args:
        image_dir: Directory with images
        output_report: Where to save report
    """
    
    print(f"\n🔍 SCANNING FOR BAD IMAGES: {image_dir}")
    print("="*70)
    
    detector = BadImageDetector()
    
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    print(f"   Total images: {len(image_files)}")
    
    bad_images = []
    good_images = []
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_file)
        
        if detector.is_bad_image(img_path):
            bad_images.append(img_file)
            print(f"   ❌ Bad: {img_file}")
        else:
            good_images.append(img_file)
            print(f"   ✅ Good: {img_file}")
    
    print("\n" + "="*70)
    print(f"RESULTS:")
    print("="*70)
    print(f"   Good images: {len(good_images)}")
    print(f"   Bad images:  {len(bad_images)}")
    print(f"   Delete ratio: {len(bad_images)/len(image_files)*100:.1f}%")
    
    # Save report
    with open(output_report, 'w') as f:
        f.write("GOOD IMAGES:\n")
        for img in good_images:
            f.write(f"  {img}\n")
        
        f.write("\n\nBAD IMAGES (TO DELETE):\n")
        for img in bad_images:
            f.write(f"  {img}\n")
    
    print(f"\n   Report saved: {output_report}")
    
    # Ask user before deleting
    response = input("\n⚠️  Delete bad images? (yes/no): ").strip().lower()
    
    if response == 'yes':
        for bad_img in bad_images:
            bad_path = os.path.join(image_dir, bad_img)
            try:
                os.remove(bad_path)
                print(f"   🗑️  Deleted: {bad_img}")
                logging.info(f"Deleted: {bad_img}")
            except Exception as e:
                print(f"   ⚠️  Failed to delete {bad_img}: {e}")
                logging.error(f"Failed to delete {bad_img}: {e}")
        
        print(f"\n✅ Deleted {len(bad_images)} bad images")
    else:
        print(f"\n⏭️  Cleanup cancelled")
    
    return good_images, bad_images


# ============ MAIN ============
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove bad quality images')
    parser.add_argument('-i', '--input', required=True, help='Directory with images')
    parser.add_argument('--dry-run', action='store_true', help='Just report, don\'t delete')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Directory not found: {args.input}")
        exit(1)
    
    good, bad = cleanup_directory(args.input)