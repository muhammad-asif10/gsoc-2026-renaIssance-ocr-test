# scripts/detect_lines_historical.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

class HistoricalBookLineDetector:
    """
    Detect text lines in historical/old book pages
    
    Handles:
    - Degraded quality
    - Variable baseline
    - Old typography
    - Poor contrast
    """
    
    def __init__(self):
        pass
    
    def load_image(self, image_path):
        """Load image"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Cannot read: {image_path}")
            return None
        print(f"✅ Loaded: {image_path} ({img.shape})")
        return img
    
    def preprocess_historical(self, img):
        """
        Preprocess for historical documents
        
        Steps:
        1. Convert to grayscale
        2. Denoise
        3. Enhance contrast
        4. Fix uneven lighting
        """
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise using bilateral filter (keeps edges sharp)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Fix uneven lighting
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
        illumination = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        corrected = cv2.divide(enhanced, illumination, scale=255)
        
        return corrected
    
    def detect_lines_projection(self, img, gap_threshold=0.015, min_height=12):
        """
        Detect lines using horizontal projection
        
        Best method for historical documents
        """
        
        print(f"\n🔍 Detecting lines (projection method)...")
        print(f"   Gap threshold: {gap_threshold}")
        print(f"   Min height: {min_height}")
        
        # Binarize
        _, binary = cv2.threshold(img, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection
        horizontal_sum = np.sum(binary, axis=1)
        
        # Normalize
        max_sum = np.max(horizontal_sum)
        if max_sum == 0:
            print("❌ No text detected (empty image)")
            return [], horizontal_sum
        
        print(f"   Max projection: {max_sum}")
        
        # Threshold
        threshold = gap_threshold * max_sum
        print(f"   Threshold: {threshold:.1f}")
        
        text_rows = np.where(horizontal_sum > threshold)[0]
        print(f"   Text rows found: {len(text_rows)}")
        
        if len(text_rows) == 0:
            return [], horizontal_sum
        
        # Find continuous text regions (lines)
        lines = []
        y_start = text_rows[0]
        y_prev = text_rows[0]
        
        for y in text_rows[1:]:
            if y != y_prev + 1:  # Gap found
                line_height = y_prev - y_start + 1
                
                if line_height >= min_height:
                    lines.append((y_start, y_prev))
                
                y_start = y
            
            y_prev = y
        
        # Don't forget last line
        line_height = y_prev - y_start + 1
        if line_height >= min_height:
            lines.append((y_start, y_prev))
        
        return lines, horizontal_sum
    
    def detect_lines_adaptive(self, img):
        """
        Adaptive line detection for very poor quality
        
        Uses different strategies:
        1. Morphological operations
        2. Adaptive thresholding
        3. Connected components
        """
        
        print(f"\n🔍 Detecting lines (adaptive method)...")
        
        # Morphological preprocessing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.erode(img, kernel, iterations=1)
        morph = cv2.dilate(morph, kernel, iterations=1)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(morph, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Horizontal projection
        horizontal_sum = np.sum(binary, axis=1)
        
        max_sum = np.max(horizontal_sum)
        if max_sum == 0:
            return [], horizontal_sum
        
        threshold = 0.01 * max_sum
        text_rows = np.where(horizontal_sum > threshold)[0]
        
        if len(text_rows) == 0:
            return [], horizontal_sum
        
        # Find lines
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
        
        print(f"   Found {len(lines)} lines")
        
        return lines, horizontal_sum
    
    def visualize_lines_only(self, img_path, lines, horizontal_projection, title="Line Detection"):
        """
        Show ONLY detected lines and graph side by side
        NO original or grayscale image
        """
        
        img = cv2.imread(img_path)
        img_copy = img.copy()
        
        print(f"\n📊 Visualizing {len(lines)} detected lines...")
        
        # Draw rectangles
        for idx, (y1, y2) in enumerate(lines):
            cv2.rectangle(img_copy, (0, y1), (img.shape[1], y2), 
                         (0, 255, 0), 2)
            cv2.putText(img_copy, str(idx + 1), (10, (y1 + y2) // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Create figure with 2 subplots
        fig = plt.figure(figsize=(18, 8))
        
        # LEFT: Detected lines on image
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Detected Lines: {len(lines)}", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # RIGHT: Projection graph
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(horizontal_projection, range(len(horizontal_projection)), linewidth=2, color='blue')
        ax2.set_title("Horizontal Projection", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Pixel Count", fontsize=11)
        ax2.set_ylabel("Row Number", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # Draw line boundaries on projection
        for y1, y2 in lines:
            ax2.axhline(y=y1, color='g', linestyle='--', alpha=0.5, linewidth=1)
            ax2.axhline(y=y2, color='g', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.show()
        
        return img_copy
    
    def extract_line_images(self, img, lines, output_dir):
        """Extract and save individual line images"""
        
        print(f"\n💾 Saving {len(lines)} line images to {output_dir}...")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for idx, (y1, y2) in enumerate(lines, 1):
            # Extract line
            line_img = img[y1:y2, :]
            
            # Normalize height
            h, w = line_img.shape[:2]
            target_h = 64
            ratio = target_h / h
            new_w = int(w * ratio)
            line_img_resized = cv2.resize(line_img, (new_w, target_h), 
                                         interpolation=cv2.INTER_CUBIC)
            
            # Pad to standard width
            if line_img_resized.shape[1] < 1024:
                pad = 1024 - line_img_resized.shape[1]
                line_img_resized = cv2.copyMakeBorder(
                    line_img_resized, 0, 0, 0, pad,
                    cv2.BORDER_CONSTANT, value=255
                )
            
            # Save
            output_file = Path(output_dir) / f"line_{idx:03d}.png"
            cv2.imwrite(str(output_file), line_img_resized)
        
        print(f"✅ Saved {len(lines)} line images")
    
    def save_results(self, lines, output_file):
        """Save line information to file"""
        
        results = {
            "total_lines": len(lines),
            "lines": []
        }
        
        for idx, (y1, y2) in enumerate(lines, 1):
            results["lines"].append({
                "line_number": idx,
                "y_start": int(y1),
                "y_end": int(y2),
                "height": int(y2 - y1)
            })
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")
    
    def process_image(self, image_path, method='projection', gap_threshold=0.015,
                     min_height=12, visualize=True, save_lines=None, save_results=None):
        """Complete process"""
        
        print("\n" + "="*70)
        print("📖 HISTORICAL DOCUMENT LINE DETECTION")
        print("="*70)
        
        # Load
        img = self.load_image(image_path)
        if img is None:
            return None
        
        print(f"   Image shape: {img.shape}")
        
        # Preprocess
        print("\n🔧 Preprocessing for historical document...")
        preprocessed = self.preprocess_historical(img)
        print("   ✅ Preprocessing complete")
        
        # Detect lines
        if method == 'projection':
            lines, horizontal_sum = self.detect_lines_projection(
                preprocessed, gap_threshold=gap_threshold, min_height=min_height
            )
        elif method == 'adaptive':
            lines, horizontal_sum = self.detect_lines_adaptive(preprocessed)
        else:
            lines, horizontal_sum = self.detect_lines_projection(
                preprocessed, gap_threshold=gap_threshold, min_height=min_height
            )
        
        # Results
        print(f"\n✅ DETECTION COMPLETE")
        print(f"{'='*70}")
        print(f"   Total lines detected: {len(lines)}")
        
        if len(lines) > 0:
            heights = [y2-y1 for y1,y2 in lines]
            print(f"   Average line height: {np.mean(heights):.1f} px")
            print(f"   Min line height: {np.min(heights):.0f} px")
            print(f"   Max line height: {np.max(heights):.0f} px")
        
        print(f"{'='*70}\n")
        
        # Print first 10 lines
        print("   First 10 detected lines:")
        for idx, (y1, y2) in enumerate(lines[:10], 1):
            height = y2 - y1
            print(f"      Line {idx:2d}: y={y1:4d}-{y2:4d} (height={height:3d}px)")
        
        if len(lines) > 10:
            print(f"      ... and {len(lines) - 10} more lines\n")
        
        # Visualize - ONLY lines and graph
        if visualize:
            self.visualize_lines_only(image_path, lines, horizontal_sum)
        
        # Save line images
        if save_lines:
            preprocessed_for_save = self.preprocess_historical(img)
            self.extract_line_images(preprocessed_for_save, lines, save_lines)
        
        # Save results
        if save_results:
            self.save_results(lines, save_results)
        
        return lines


def main():
    parser = argparse.ArgumentParser(
        description='Detect lines in historical book pages'
    )
    
    parser.add_argument('image', help='Image file path')
    parser.add_argument('--method', choices=['projection', 'adaptive'], 
                       default='projection', help='Detection method')
    parser.add_argument('--gap-threshold', type=float, default=0.015,
                       help='Gap threshold (default: 0.015)')
    parser.add_argument('--min-height', type=int, default=12,
                       help='Minimum line height (default: 12)')
    parser.add_argument('--save-lines', help='Directory to save line images')
    parser.add_argument('--save-results', help='File to save results JSON')
    parser.add_argument('--no-visualize', action='store_true', 
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    detector = HistoricalBookLineDetector()
    lines = detector.process_image(
        args.image,
        method=args.method,
        gap_threshold=args.gap_threshold,
        min_height=args.min_height,
        visualize=not args.no_visualize,
        save_lines=args.save_lines,
        save_results=args.save_results
    )


if __name__ == "__main__":
    main()