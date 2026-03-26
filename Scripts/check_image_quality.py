import cv2
import numpy as np
from pathlib import Path

def check_image(image_path):
    """Quick check if image is good or bad"""
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Cannot read image")
        return False
    
    brightness = np.mean(img)
    white_pixels = np.sum(img > 200) / (img.shape[0] * img.shape[1])
    black_pixels = np.sum(img < 50) / (img.shape[0] * img.shape[1])
    
    print(f"\n📊 IMAGE ANALYSIS: {Path(image_path).name}")
    print(f"   Size: {img.shape}")
    print(f"   Brightness: {brightness:.1f}/255")
    print(f"   White pixels: {white_pixels*100:.1f}%")
    print(f"   Black pixels: {black_pixels*100:.1f}%")
    
    # Assessment
    is_bad = False
    reasons = []
    
    if brightness < 50:
        reasons.append("Too dark")
        is_bad = True
    
    if white_pixels > 0.95:
        reasons.append("Too blank (mostly white)")
        is_bad = True
    
    if black_pixels > 0.90:
        reasons.append("Too dark (mostly black)")
        is_bad = True
    
    if is_bad:
        print(f"\n   ❌ BAD IMAGE - Reasons:")
        for reason in reasons:
            print(f"      • {reason}")
        return False
    else:
        print(f"\n   ✅ GOOD IMAGE - Keep it")
        return True


# Test on your problematic image
if __name__ == "__main__":
    check_image("C:\Open Source\GSOC\org\HumanAI\gsoc-2026-renaissance-ocr-test\GSOC_2026_HumanAI_Test\data\Buendia_Instruccion\lines\page_023\lines\page_023_line_0043.png")