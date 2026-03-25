# save as: line_detection_diagnostics.py
"""
Line Detection Diagnostics Tool
Analyze and visualize line detection parameters for optimal results

Usage:
    python line_detection_diagnostics.py -i image.png
    python line_detection_diagnostics.py -i image.png --interactive
    python line_detection_diagnostics.py -i folder/ --batch
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from typing import List, Tuple, Optional
import logging

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ LINE DETECTION ANALYZER CLASS ============
class LineDetectionAnalyzer:
    """
    Analyze and visualize line detection with different parameters
    """
    
    def __init__(self, image_path: str):
        """
        Initialize analyzer with image
        
        Args:
            image_path: Path to image file
        """
        self.image_path = Path(image_path)
        self.img_original = None
        self.img_gray = None
        self.img_binary = None
        self.horizontal_sum = None
        
        self._load_image()
    
    def _load_image(self) -> None:
        """Load and validate image"""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        # Load as color first
        self.img_original = cv2.imread(str(self.image_path))
        if self.img_original is None:
            raise ValueError(f"Cannot read image: {self.image_path}")
        
        # Convert to grayscale
        self.img_gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        
        logger.info(f"Loaded image: {self.image_path.name} ({self.img_gray.shape})")
    
    def print_diagnostics(self) -> None:
        """Print image diagnostics"""
        if self.img_gray is None:
            return
        
        print(f"\n{'='*70}")
        print(f"📊 IMAGE DIAGNOSTICS: {self.image_path.name}")
        print(f"{'='*70}")
        print(f"  Dimensions:    {self.img_gray.shape[1]} × {self.img_gray.shape[0]} pixels")
        print(f"  Brightness:    {np.mean(self.img_gray):.1f}/255")
        print(f"  Contrast:      {np.std(self.img_gray):.1f}")
        print(f"  Min intensity: {np.min(self.img_gray)}")
        print(f"  Max intensity: {np.max(self.img_gray)}")
        
        # Calculate text coverage
        _, binary = cv2.threshold(self.img_gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_coverage = (np.sum(binary) / binary.size) * 100
        print(f"  Text coverage: {text_coverage:.1f}%")
    
    def _detect_lines(self, gap_threshold: float, min_height: int) -> List[Tuple[int, int]]:
        """
        Detect lines with given parameters
        
        Args:
            gap_threshold: Threshold for gap detection (0.0-1.0)
            min_height: Minimum line height in pixels
        
        Returns:
            List of (y_start, y_end) tuples
        """
        # Binarize
        _, binary = cv2.threshold(self.img_gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection
        horizontal_sum = np.sum(binary, axis=1)
        
        # Threshold
        max_sum = np.max(horizontal_sum)
        if max_sum == 0:
            return []
        
        threshold = gap_threshold * max_sum
        text_rows = np.where(horizontal_sum > threshold)[0]
        
        if len(text_rows) == 0:
            return []
        
        # Detect lines
        lines = []
        y_start = text_rows[0]
        y_prev = text_rows[0]
        
        for y in text_rows[1:]:
            if y != y_prev + 1:
                if y_prev - y_start >= min_height:
                    lines.append((y_start, y_prev))
                y_start = y
            y_prev = y
        
        # Add last line
        if y_prev - y_start >= min_height:
            lines.append((y_start, y_prev))
        
        return lines
    
    def visualize_parameter_grid(self, 
                                gap_values: Optional[List[float]] = None,
                                height_values: Optional[List[int]] = None,
                                output_path: Optional[str] = None) -> None:
        """
        Create grid visualization of different parameters
        
        Args:
            gap_values: List of gap thresholds to test
            height_values: List of minimum heights to test
            output_path: Path to save visualization
        """
        if gap_values is None:
            gap_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        
        if height_values is None:
            height_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        # Create figure
        fig, axes = plt.subplots(len(gap_values), len(height_values), 
                                figsize=(20, 16))
        fig.suptitle(f'Line Detection Parameter Analysis\n{self.image_path.name}',
                    fontsize=16, fontweight='bold')
        
        logger.info(f"Creating grid: {len(gap_values)} × {len(height_values)} = {len(gap_values)*len(height_values)} visualizations")
        
        # Process each combination
        for idx_gap, gap_thresh in enumerate(gap_values):
            for idx_height, min_height in enumerate(height_values):
                ax = axes[idx_gap, idx_height]
                
                try:
                    # Detect lines
                    lines = self._detect_lines(gap_thresh, min_height)
                    
                    # Visualize
                    img_display = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB).copy()
                    
                    for y1, y2 in lines:
                        cv2.rectangle(img_display, (0, y1), (img_display.shape[1], y2),
                                    (0, 255, 0), 2)
                    
                    ax.imshow(img_display)
                    ax.set_title(f"Gap={gap_thresh:.2f}, H={min_height}\nLines: {len(lines)}",
                               fontsize=8)
                    ax.axis('off')
                    
                except Exception as e:
                    logger.error(f"Error processing gap={gap_thresh}, height={min_height}: {e}")
                    ax.text(0.5, 0.5, f"Error: {str(e)[:30]}", ha='center', va='center')
                    ax.axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            logger.info(f"Parameter grid saved: {output_path}")
            print(f"✓ Parameter grid saved: {output_path}")
        
        plt.show()
    
    def visualize_projection(self, output_path: Optional[str] = None) -> None:
        """
        Visualize horizontal projection profile
        
        Args:
            output_path: Path to save visualization
        """
        # Binarize
        _, binary = cv2.threshold(self.img_gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection
        horizontal_sum = np.sum(binary, axis=1)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        axes[0].imshow(self.img_gray, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Projection graph
        axes[1].plot(horizontal_sum, range(len(horizontal_sum)), linewidth=2, color='#3498db')
        axes[1].fill_betweenx(range(len(horizontal_sum)), 0, horizontal_sum,
                            alpha=0.3, color='#3498db')
        axes[1].set_title('Horizontal Projection (Text Detection)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Pixel Count', fontsize=11)
        axes[1].set_ylabel('Row Number', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Projection graph saved: {output_path}")
            print(f"✓ Projection graph saved: {output_path}")
        
        plt.show()
    
    def visualize_single_detection(self, gap_threshold: float = 0.02, 
                                   min_height: int = 20,
                                   output_path: Optional[str] = None) -> None:
        """
        Visualize single detection with parameters
        
        Args:
            gap_threshold: Gap threshold for detection
            min_height: Minimum line height
            output_path: Path to save visualization
        """
        lines = self._detect_lines(gap_threshold, min_height)
        
        # Binarize for projection
        _, binary = cv2.threshold(self.img_gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_sum = np.sum(binary, axis=1)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Detection visualization
        img_display = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB).copy()
        for i, (y1, y2) in enumerate(lines):
            cv2.rectangle(img_display, (0, y1), (img_display.shape[1], y2),
                        (0, 255, 0), 2)
            cv2.putText(img_display, str(i+1), (10, (y1+y2)//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        axes[0].imshow(img_display)
        axes[0].set_title(f'Detected Lines: {len(lines)}\nGap={gap_threshold}, MinH={min_height}',
                         fontsize=12, fontweight='bold')
        axes[0].axis('on')
        
        # Projection
        axes[1].plot(horizontal_sum, range(len(horizontal_sum)), linewidth=2, color='#3498db')
        axes[1].fill_betweenx(range(len(horizontal_sum)), 0, horizontal_sum,
                            alpha=0.3, color='#3498db')
        
        # Mark detected lines
        for y1, y2 in lines:
            axes[1].axhline(y=y1, color='green', linestyle='--', alpha=0.5, linewidth=1)
            axes[1].axhline(y=y2, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        
        axes[1].set_title('Horizontal Projection with Line Boundaries', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Pixel Count', fontsize=11)
        axes[1].set_ylabel('Row Number', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Detection visualization saved: {output_path}")
            print(f"✓ Detection visualization saved: {output_path}")
        
        # Print statistics
        if lines:
            heights = np.array([y2 - y1 for y1, y2 in lines])
            print(f"\n📈 Line Statistics:")
            print(f"  • Total lines: {len(lines)}")
            print(f"  • Avg height: {np.mean(heights):.1f} px")
            print(f"  • Height range: {np.min(heights):.0f} - {np.max(heights):.0f} px")
        else:
            print("\n⚠️ No lines detected with these parameters!")
        
        plt.show()


# ============ CLI INTERFACE ============
def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line parser"""
    parser = argparse.ArgumentParser(
        description='Line Detection Diagnostics Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python line_detection_diagnostics.py -i image.png
  
  # Test custom parameters
  python line_detection_diagnostics.py -i image.png --gap 0.01 0.05 --height 10 20 30
  
  # Single detection visualization
  python line_detection_diagnostics.py -i image.png --single --gap 0.02 --height 20
  
  # Batch process folder
  python line_detection_diagnostics.py -i folder/ --batch
        """
    )
    
    parser.add_argument('-i', '--input', 
                       type=str, 
                       required=True,
                       help='Input image file or folder')
    
    parser.add_argument('--gap',
                       type=float,
                       nargs='+',
                       default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
                       help='Gap threshold values to test')
    
    parser.add_argument('--height',
                       type=int,
                       nargs='+',
                       default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                       help='Min height values to test')
    
    parser.add_argument('--single',
                       action='store_true',
                       help='Run single detection (requires --gap and --height)')
    
    parser.add_argument('--batch',
                       action='store_true',
                       help='Process all images in folder')
    
    parser.add_argument('--output',
                       type=str,
                       default=None,
                       help='Output directory for visualizations')
    
    parser.add_argument('--no-projection',
                       action='store_true',
                       help='Skip projection visualization')
    
    return parser


def main():
    """Main entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else Path(f"./diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    print("\n" + "="*70)
    print("LINE DETECTION DIAGNOSTICS")
    print("="*70)
    
    try:
        # Single image mode
        if input_path.is_file():
            analyzer = LineDetectionAnalyzer(str(input_path))
            analyzer.print_diagnostics()
            
            if args.single:
                # Single detection
                gap = args.gap[0] if args.gap else 0.02
                height = args.height[0] if args.height else 20
                
                print(f"\n🔍 Single Detection Visualization")
                print(f"  Gap threshold: {gap}")
                print(f"  Min height: {height}")
                
                analyzer.visualize_single_detection(
                    gap_threshold=gap,
                    min_height=height,
                    output_path=output_dir / f"detection_{input_path.stem}.png" if args.output else None
                )
            else:
                # Parameter grid
                print(f"\n📊 Creating parameter grid...")
                analyzer.visualize_parameter_grid(
                    gap_values=args.gap,
                    height_values=args.height,
                    output_path=output_dir / f"grid_{input_path.stem}.png" if args.output else None
                )
                
                # Projection
                if not args.no_projection:
                    print(f"📈 Creating projection visualization...")
                    analyzer.visualize_projection(
                        output_path=output_dir / f"projection_{input_path.stem}.png" if args.output else None
                    )
        
        # Batch mode
        elif input_path.is_dir() and args.batch:
            print(f"\n📁 Batch processing folder: {input_path}")
            
            image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
            print(f"Found {len(image_files)} images")
            
            for img_file in image_files:
                print(f"\n  Processing: {img_file.name}")
                try:
                    analyzer = LineDetectionAnalyzer(str(img_file))
                    analyzer.print_diagnostics()
                    analyzer.visualize_parameter_grid(
                        gap_values=args.gap,
                        height_values=args.height,
                        output_path=output_dir / f"grid_{img_file.stem}.png"
                    )
                except Exception as e:
                    logger.error(f"Error processing {img_file.name}: {e}")
        
        else:
            raise ValueError("Input must be an image file or folder with --batch flag")
        
        print(f"\n✓ Complete!")
        if args.output:
            print(f"  Visualizations saved to: {output_dir}")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    from datetime import datetime
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        sys.argv.append('-h')
    
    main()