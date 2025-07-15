#!/usr/bin/env python3
"""
Standalone Image Extraction Tool
Re-processes existing page images to extract embedded images using computer vision.
Can be used to re-run image extraction with different parameters.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from dataclasses import dataclass, asdict

# Import configuration system
from extractor_config import ImageDetectionConfig

@dataclass
class ExtractedImage:
    """Structure for extracted images"""
    filename: str
    format: str
    width: int
    height: int
    page_number: int = None
    caption: str = None
    metadata: Dict[str, Any] = None

class ImageExtractor:
    """Standalone image extraction from page images"""
    
    def __init__(self, output_dir: Path, config: ImageDetectionConfig = None):
        """
        Initialize the image extractor with configuration.
        
        Args:
            output_dir: Directory to save extracted images
            config: ImageDetectionConfig instance, or None for default
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Use provided config or create default
        self.config = config if config is not None else ImageDetectionConfig()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def extract_images_from_page(self, page_image_path: Path, page_num: int) -> List[ExtractedImage]:
        """
        Extract images from a single page image.
        
        Args:
            page_image_path: Path to the page image file
            page_num: Page number for metadata
            
        Returns:
            List of extracted image information
        """
        extracted_images = []
        
        try:
            # Read the page image
            page_img = cv2.imread(str(page_image_path))
            if page_img is None:
                self.logger.error(f"Could not read image: {page_image_path}")
                return extracted_images
            
            # Convert to grayscale
            gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Edge detection approach
            extracted_images.extend(self._extract_via_edge_detection(page_img, gray, page_num))
            
            # Method 2: Connected components approach
            extracted_images.extend(self._extract_via_connected_components(page_img, gray, page_num))
            
            self.logger.info(f"Extracted {len(extracted_images)} images from page {page_num}")
            
        except Exception as e:
            self.logger.error(f"Failed to extract images from page {page_num}: {e}")
        
        return extracted_images
    
    def _extract_via_edge_detection(self, page_img: np.ndarray, gray: np.ndarray, page_num: int) -> List[ExtractedImage]:
        """Extract images using edge detection and contour analysis."""
        extracted_images = []
        
        try:
            # Edge detection with configurable parameters
            edges = cv2.Canny(gray, self.config.canny_low_threshold, self.config.canny_high_threshold, 
                            apertureSize=self.config.canny_aperture_size)
            
            # Morphological operations to connect broken edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.morph_kernel_size)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            extracted_count = 0
            
            for i, contour in enumerate(contours):
                # Calculate area
                area = cv2.contourArea(contour)
                
                if area < self.config.min_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Apply size and aspect ratio filters
                aspect_ratio = w / h
                if (aspect_ratio > self.config.max_aspect_ratio or 
                    aspect_ratio < self.config.min_aspect_ratio or 
                    w < self.config.min_width or 
                    h < self.config.min_height):
                    continue
                
                # Extract the region
                roi = page_img[y:y+h, x:x+w]
                
                # Check if this looks like an image
                if self._is_likely_image_region(roi):
                    # Save the extracted image
                    extracted_filename = f"page_{page_num:03d}_edge_{extracted_count:03d}.png"
                    extracted_path = self.output_dir / extracted_filename
                    
                    cv2.imwrite(str(extracted_path), roi)
                    
                    # Create image metadata
                    image_info = ExtractedImage(
                        filename=extracted_filename,
                        format="PNG",
                        width=w,
                        height=h,
                        page_number=page_num,
                        metadata={
                            "source": "CV_extraction",
                            "extraction_method": "edge_detection",
                            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                            "contour_index": i
                        }
                    )
                    
                    extracted_images.append(image_info)
                    extracted_count += 1
                    
        except Exception as e:
            self.logger.warning(f"Edge detection extraction failed for page {page_num}: {e}")
        
        return extracted_images
    
    def _extract_via_connected_components(self, page_img: np.ndarray, gray: np.ndarray, page_num: int) -> List[ExtractedImage]:
        """Extract images using connected components analysis."""
        extracted_images = []
        
        try:
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=self.config.connectivity)
            
            extracted_count = 0
            
            for i in range(1, num_labels):  # Skip background (label 0)
                x, y, w, h, area = stats[i]
                
                # Filter by size
                if area < self.config.cc_min_area or w < self.config.min_width or h < self.config.min_height:
                    continue
                
                # Extract the region
                roi = page_img[y:y+h, x:x+w]
                
                # Check if this looks like an image
                if self._is_likely_image_region(roi):
                    # Save the extracted image
                    extracted_filename = f"page_{page_num:03d}_component_{extracted_count:03d}.png"
                    extracted_path = self.output_dir / extracted_filename
                    
                    cv2.imwrite(str(extracted_path), roi)
                    
                    # Create image metadata
                    image_info = ExtractedImage(
                        filename=extracted_filename,
                        format="PNG",
                        width=w,
                        height=h,
                        page_number=page_num,
                        metadata={
                            "source": "CV_extraction",
                            "extraction_method": "connected_components",
                            "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                            "area": area,
                            "component_label": i
                        }
                    )
                    
                    extracted_images.append(image_info)
                    extracted_count += 1
                    
        except Exception as e:
            self.logger.warning(f"Connected components extraction failed for page {page_num}: {e}")
        
        return extracted_images
    
    def _is_likely_image_region(self, roi: np.ndarray) -> bool:
        """
        Determine if a region is likely an image rather than text.
        
        Args:
            roi: Region of interest (image patch)
            
        Returns:
            True if likely an image, False if likely text
        """
        if roi.size == 0:
            return False
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
        
        # Calculate image characteristics
        height, width = gray_roi.shape
        
        # Skip very small regions (already filtered, but double-check)
        if height < self.config.min_height or width < self.config.min_width:
            return False
        
        # Calculate variance (images tend to have more variance than text)
        variance = np.var(gray_roi)
        
        # Calculate edge density
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Calculate histogram entropy (images tend to have more diverse pixel values)
        hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
        hist_norm = hist / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        # Heuristic scoring
        image_score = 0
        
        if variance > self.config.variance_threshold:
            image_score += 1
        if self.config.edge_density_min < edge_density < self.config.edge_density_max:
            image_score += 1
        if entropy > self.config.entropy_threshold:
            image_score += 1
        
        # Additional check: look for text-like patterns
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_density = np.sum(horizontal_lines > 0) / (width * height)
        
        if horizontal_density > self.config.horizontal_line_threshold:
            image_score -= 1
        
        return image_score >= 2
    
    def process_directory(self, input_dir: Path, pattern: str = "page_*.png") -> List[ExtractedImage]:
        """
        Process all page images in a directory.
        
        Args:
            input_dir: Directory containing page images
            pattern: Glob pattern for page images
            
        Returns:
            List of all extracted images
        """
        all_extracted_images = []
        
        # Find all page images
        page_images = sorted(input_dir.glob(pattern))
        
        if not page_images:
            self.logger.warning(f"No page images found in {input_dir} matching pattern {pattern}")
            return all_extracted_images
        
        self.logger.info(f"Found {len(page_images)} page images to process")
        
        for page_image_path in page_images:
            # Extract page number from filename
            try:
                page_num = int(page_image_path.stem.split('_')[1])
            except (IndexError, ValueError):
                # Fallback to sequential numbering
                page_num = len(all_extracted_images) + 1
            
            # Extract images from this page
            extracted_images = self.extract_images_from_page(page_image_path, page_num)
            all_extracted_images.extend(extracted_images)
        
        return all_extracted_images
    
    def save_manifest(self, extracted_images: List[ExtractedImage], manifest_path: Path):
        """Save extracted images manifest to JSON file."""
        manifest_data = [asdict(img) for img in extracted_images]
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved manifest with {len(extracted_images)} images to {manifest_path}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Extract images from page images using computer vision")
    parser.add_argument("input_dir", help="Directory containing page images")
    parser.add_argument("-o", "--output", help="Output directory for extracted images")
    parser.add_argument("-p", "--pattern", default="page_*.png", help="Glob pattern for page images")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to configuration JSON file (image_detection section)")
    
    # Image detection parameters (for backward compatibility)
    parser.add_argument("--min-area", type=int, help="Minimum area in pixels")
    parser.add_argument("--min-width", type=int, help="Minimum width in pixels")
    parser.add_argument("--min-height", type=int, help="Minimum height in pixels")
    parser.add_argument("--max-aspect-ratio", type=float, help="Maximum aspect ratio")
    parser.add_argument("--min-aspect-ratio", type=float, help="Minimum aspect ratio")
    parser.add_argument("--canny-low", type=int, help="Lower Canny threshold")
    parser.add_argument("--canny-high", type=int, help="Upper Canny threshold")
    parser.add_argument("--variance-threshold", type=int, help="Minimum pixel variance")
    parser.add_argument("--edge-density-min", type=float, help="Minimum edge density")
    parser.add_argument("--edge-density-max", type=float, help="Maximum edge density")
    parser.add_argument("--entropy-threshold", type=float, help="Minimum entropy")
    parser.add_argument("--horizontal-line-threshold", type=float, help="Max horizontal line density")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        try:
            from extractor_config import DocumentExtractorConfig
            full_config = DocumentExtractorConfig.from_file(Path(args.config))
            config = full_config.image_detection
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1
    else:
        config = ImageDetectionConfig()
    
    # Apply CLI overrides
    if args.min_area:
        config.min_area = args.min_area
    if args.min_width:
        config.min_width = args.min_width
    if args.min_height:
        config.min_height = args.min_height
    if args.max_aspect_ratio:
        config.max_aspect_ratio = args.max_aspect_ratio
    if args.min_aspect_ratio:
        config.min_aspect_ratio = args.min_aspect_ratio
    if args.canny_low:
        config.canny_low_threshold = args.canny_low
    if args.canny_high:
        config.canny_high_threshold = args.canny_high
    if args.variance_threshold:
        config.variance_threshold = args.variance_threshold
    if args.edge_density_min:
        config.edge_density_min = args.edge_density_min
    if args.edge_density_max:
        config.edge_density_max = args.edge_density_max
    if args.entropy_threshold:
        config.entropy_threshold = args.entropy_threshold
    if args.horizontal_line_threshold:
        config.horizontal_line_threshold = args.horizontal_line_threshold
    
    # Set up paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    output_dir = Path(args.output) if args.output else input_dir / "extracted_images"
    
    # Initialize extractor with configuration
    extractor = ImageExtractor(output_dir=output_dir, config=config)
    
    # Process all images
    extracted_images = extractor.process_directory(input_dir, args.pattern)
    
    # Save manifest
    manifest_path = output_dir / "extracted_images_manifest.json"
    extractor.save_manifest(extracted_images, manifest_path)
    
    print(f"Extracted {len(extracted_images)} images to {output_dir}")
    print(f"Manifest saved to {manifest_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())