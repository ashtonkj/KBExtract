#!/usr/bin/env python3
"""
Configuration system for Document Extractor
Centralizes all default values, thresholds, and magic numbers with clear documentation.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional


@dataclass
class OCRConfig:
    """OCR processing configuration"""
    # Basic OCR settings
    dpi: int = 400                          # OCR processing DPI (higher = better quality, slower)
    language: str = "eng+chi_sim+chi_tra"   # Tesseract language codes
    page_segmentation_mode: int = 3         # PSM: 3=automatic page segmentation
    ocr_engine_mode: int = 3                # OEM: 3=default based on what's available
    enable_preprocessing: bool = True        # Apply image preprocessing before OCR
    
    # Confidence thresholds
    low_confidence_threshold: int = 60       # Words below this confidence flagged for review
    min_avg_confidence: float = 0.3         # Minimum average confidence to accept OCR result
    
    # Text quality thresholds
    min_text_length: int = 10               # Minimum characters for valid text
    min_scanned_page_text: int = 50         # Threshold to detect scanned pages (chars)
    
    # Image preprocessing parameters
    gaussian_blur_kernel: tuple = (1, 1)    # Kernel size for noise reduction
    adaptive_threshold_block_size: int = 11  # Block size for adaptive thresholding
    adaptive_threshold_c: int = 2           # Constant subtracted from mean
    morphology_kernel_size: tuple = (1, 1)  # Morphological operation kernel size


@dataclass 
class ImageDetectionConfig:
    """Computer vision image detection configuration"""
    # Size filtering
    min_area: int = 75000                   # Minimum pixel area for detected regions
    min_width: int = 250                    # Minimum width in pixels
    min_height: int = 250                   # Minimum height in pixels
    max_aspect_ratio: float = 4.0           # Maximum width/height ratio
    min_aspect_ratio: float = 0.25          # Minimum width/height ratio
    
    # Edge detection (Canny algorithm)
    canny_low_threshold: int = 100          # Lower threshold for edge detection
    canny_high_threshold: int = 200         # Upper threshold for edge detection
    canny_aperture_size: int = 3            # Aperture size for Sobel operator
    
    # Morphological operations
    morph_kernel_size: tuple = (3, 3)       # Kernel for morphological operations
    
    # Image quality assessment
    variance_threshold: int = 1000          # Minimum pixel variance for images
    edge_density_min: float = 0.05          # Minimum edge density for images
    edge_density_max: float = 0.3           # Maximum edge density for images
    entropy_threshold: float = 6.0          # Minimum histogram entropy for images
    horizontal_line_threshold: float = 0.1   # Max horizontal line density (text indicator)
    
    # Connected components analysis
    connectivity: int = 8                   # Connectivity for connected components (4 or 8)
    cc_min_area: int = 60000               # Minimum area for connected components
    
    # Image classification thresholds
    large_image_threshold: int = 300000     # Area threshold for large images
    medium_image_threshold: int = 100000    # Area threshold for medium images
    square_aspect_ratio_min: float = 0.7   # Minimum aspect ratio for square images
    square_aspect_ratio_max: float = 1.3   # Maximum aspect ratio for square images
    relevance_score_threshold: float = 0.7  # Threshold for relevant images


@dataclass
class TextProcessingConfig:
    """Text processing and quality assessment configuration"""
    # Text quality scoring
    min_word_count: int = 10                # Minimum words for quality assessment
    min_text_length_quality: int = 20       # Minimum length for quality scoring
    optimal_text_length: int = 100          # Length that gets quality bonus
    quality_length_bonus: float = 1.1       # Bonus multiplier for good length
    
    # Word length assessment
    min_reasonable_word_length: int = 2     # Minimum reasonable average word length
    max_reasonable_word_length: int = 15    # Maximum reasonable average word length
    word_length_bonus: float = 1.05         # Bonus for reasonable word lengths
    
    # Character quality thresholds
    min_clean_char_ratio: float = 0.8       # Minimum ratio of clean characters
    
    # Text preprocessing
    max_char_repetition: int = 2            # Maximum character repetition allowed
    excessive_whitespace_threshold: int = 3  # Spaces to normalize to single space


@dataclass
class MartialArtsConfig:
    """Martial arts content detection configuration"""
    # Content relevance scoring
    technique_relevance_weight: float = 0.3  # Weight for technique mentions
    chinese_relevance_weight: float = 1.0   # Weight for Chinese character ratio
    keyword_relevance_weight: float = 0.1   # Weight for keyword density
    
    # Chinese text thresholds
    min_chinese_ratio_multilingual: float = 0.1  # Threshold for multilingual chunks
    
    # Technique detection patterns (regex patterns for technique names)
    technique_patterns: List[str] = None
    
    # Martial arts keywords for relevance scoring
    martial_arts_keywords: List[str] = None
    
    def __post_init__(self):
        if self.technique_patterns is None:
            self.technique_patterns = [
                r'[A-Z][a-z]+ [A-Z][a-z]+ (?:Fist|Palm|Kick|Strike|Stance|Step|Form|Block)',
                r'(?:Taolu|Kata|Form|Set|Routine) \d+',
                r'(?:Sifu|Shifu|Master|Sensei|Laoshi) [A-Z][a-z]+',
                r'(?:Stance|Step|Position|Guard): [A-Z][a-z]+',
                r'(?:Shaolin|Wudang|Tai Chi|Kung Fu|Gong Fu|Qi Gong|Nei Gong)',
                r'(?:Dan Tian|Meridian|Qi|Chi|Jin|Li|Jing)',
                r'(?:Eight|Five|Seven|Nine|Ten|Twelve) .{1,20}(?:Animals|Elements|Stars|Palms|Fists)',
            ]
        
        if self.martial_arts_keywords is None:
            self.martial_arts_keywords = [
                'martial', 'arts', 'kungfu', 'kung fu', 'taichi', 'tai chi', 'qigong', 'qi gong',
                'shaolin', 'wudang', 'internal', 'external', 'power', 'energy', 'meditation',
                'breathing', 'posture', 'movement', 'technique', 'application', 'training',
                'practice', 'master', 'student', 'lineage', 'tradition', 'ancient', 'classical'
            ]


@dataclass
class RAGConfig:
    """RAG system optimization configuration"""
    # Chunking parameters
    target_chunk_size: int = 512            # Target token count per chunk
    max_chunk_size: int = 600              # Maximum token count per chunk
    min_chunk_size: int = 50               # Minimum token count per chunk
    technique_chunk_threshold: int = 300    # Length threshold for technique-based chunking
    
    # Quality filtering
    min_quality_score: float = 0.6         # Minimum quality score for embedding chunks
    min_confidence_score: float = 0.7      # Minimum confidence for high-quality chunks
    max_low_confidence_regions: int = 2     # Max low-confidence words before flagging
    
    # Content type weights for quality scoring
    content_type_weights: Dict[str, float] = None
    
    # Image relevance scoring
    image_type_scores: Dict[str, float] = None
    technique_demo_bonus: float = 0.1       # Bonus per technique demonstration
    ocr_text_bonus: float = 0.3            # Bonus for OCR text content
    caption_bonus: float = 0.2             # Bonus for caption analysis
    large_image_bonus: float = 1.1         # Bonus for large images
    large_image_area_threshold: int = 200000 # Area threshold for large image bonus
    
    def __post_init__(self):
        if self.content_type_weights is None:
            self.content_type_weights = {
                'heading': 1.2,
                'paragraph': 1.0,
                'table': 1.1,
                'caption': 0.8,
                'error': 0.3
            }
        
        if self.image_type_scores is None:
            self.image_type_scores = {
                "photo": 0.8,
                "illustration": 0.9,
                "diagram": 0.7,
                "icon": 0.3
            }


@dataclass
class PDFConfig:
    """PDF-specific processing configuration"""
    # Font size thresholds for content classification
    heading_font_threshold: float = 14.0    # Minimum font size for headings
    large_heading_threshold: float = 18.0   # Font size for level 1 headings
    medium_heading_threshold: float = 16.0  # Font size for level 2 headings
    small_heading_threshold: float = 14.0   # Font size for level 3 headings
    
    # Content classification
    caption_max_length: int = 100           # Maximum length for captions
    uppercase_heading_threshold: float = 0.8 # Ratio of uppercase for heading detection
    
    # Page processing
    zoom_factor: float = 2.0                # Zoom factor for page to image conversion
    
    # Table extraction
    enable_table_extraction: bool = True     # Enable pdfplumber table extraction


@dataclass
class DocumentExtractorConfig:
    """Main configuration object containing all subsystem configs"""
    # Subsystem configurations
    ocr: OCRConfig = None
    image_detection: ImageDetectionConfig = None
    text_processing: TextProcessingConfig = None
    martial_arts: MartialArtsConfig = None
    rag: RAGConfig = None
    pdf: PDFConfig = None
    
    # General settings
    output_dir: str = "extracted_documents"
    extract_embedded_images: bool = True
    verbose_logging: bool = False
    
    def __post_init__(self):
        if self.ocr is None:
            self.ocr = OCRConfig()
        if self.image_detection is None:
            self.image_detection = ImageDetectionConfig()
        if self.text_processing is None:
            self.text_processing = TextProcessingConfig()
        if self.martial_arts is None:
            self.martial_arts = MartialArtsConfig()
        if self.rag is None:
            self.rag = RAGConfig()
        if self.pdf is None:
            self.pdf = PDFConfig()
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'DocumentExtractorConfig':
        """Load configuration from JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DocumentExtractorConfig':
        """Create configuration from dictionary"""
        # Extract subsystem configs
        ocr_config = OCRConfig(**config_dict.get('ocr', {}))
        image_config = ImageDetectionConfig(**config_dict.get('image_detection', {}))
        text_config = TextProcessingConfig(**config_dict.get('text_processing', {}))
        ma_config = MartialArtsConfig(**config_dict.get('martial_arts', {}))
        rag_config = RAGConfig(**config_dict.get('rag', {}))
        pdf_config = PDFConfig(**config_dict.get('pdf', {}))
        
        # Extract general settings
        general_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['ocr', 'image_detection', 'text_processing', 
                                     'martial_arts', 'rag', 'pdf']}
        
        return cls(
            ocr=ocr_config,
            image_detection=image_config,
            text_processing=text_config,
            martial_arts=ma_config,
            rag=rag_config,
            pdf=pdf_config,
            **general_settings
        )
    
    def to_file(self, config_path: Path):
        """Save configuration to JSON file"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'ocr': asdict(self.ocr),
            'image_detection': asdict(self.image_detection),
            'text_processing': asdict(self.text_processing),
            'martial_arts': asdict(self.martial_arts),
            'rag': asdict(self.rag),
            'pdf': asdict(self.pdf),
            'output_dir': self.output_dir,
            'extract_embedded_images': self.extract_embedded_images,
            'verbose_logging': self.verbose_logging
        }
    
    def validate(self) -> List[str]:
        """Validate configuration values and return list of issues"""
        issues = []
        
        # OCR validation
        if self.ocr.dpi < 72 or self.ocr.dpi > 1200:
            issues.append("OCR DPI should be between 72 and 1200")
        
        if self.ocr.low_confidence_threshold < 0 or self.ocr.low_confidence_threshold > 100:
            issues.append("OCR confidence threshold should be between 0 and 100")
        
        # Image detection validation
        if self.image_detection.min_area < 1000:
            issues.append("Minimum image area should be at least 1000 pixels")
        
        if self.image_detection.min_aspect_ratio >= self.image_detection.max_aspect_ratio:
            issues.append("Minimum aspect ratio should be less than maximum aspect ratio")
        
        # Text processing validation
        if self.text_processing.min_clean_char_ratio < 0 or self.text_processing.min_clean_char_ratio > 1:
            issues.append("Clean character ratio should be between 0 and 1")
        
        # RAG validation
        if self.rag.min_chunk_size >= self.rag.max_chunk_size:
            issues.append("Minimum chunk size should be less than maximum chunk size")
        
        if self.rag.min_quality_score < 0 or self.rag.min_quality_score > 1:
            issues.append("Quality score should be between 0 and 1")
        
        return issues
    
    def create_example_config(self, output_path: Path):
        """Create an example configuration file with documentation"""
        example_config = {
            "_comment": "Document Extractor Configuration",
            "_description": "Centralized configuration for all extraction parameters",
            
            "output_dir": "extracted_documents",
            "extract_embedded_images": True,
            "verbose_logging": False,
            
            "ocr": {
                "_comment": "OCR processing settings",
                "dpi": 400,
                "language": "eng+chi_sim+chi_tra",
                "page_segmentation_mode": 3,
                "ocr_engine_mode": 3,
                "enable_preprocessing": True,
                "low_confidence_threshold": 60,
                "min_avg_confidence": 0.3,
                "min_text_length": 10,
                "min_scanned_page_text": 50
            },
            
            "image_detection": {
                "_comment": "Computer vision image detection settings",
                "min_area": 75000,
                "min_width": 250,
                "min_height": 250,
                "max_aspect_ratio": 4.0,
                "min_aspect_ratio": 0.25,
                "canny_low_threshold": 100,
                "canny_high_threshold": 200,
                "variance_threshold": 1000,
                "edge_density_min": 0.05,
                "edge_density_max": 0.3,
                "entropy_threshold": 6.0
            },
            
            "text_processing": {
                "_comment": "Text quality assessment and processing",
                "min_word_count": 10,
                "min_text_length_quality": 20,
                "optimal_text_length": 100,
                "min_clean_char_ratio": 0.8
            },
            
            "martial_arts": {
                "_comment": "Martial arts content detection settings",
                "technique_relevance_weight": 0.3,
                "chinese_relevance_weight": 1.0,
                "keyword_relevance_weight": 0.1,
                "min_chinese_ratio_multilingual": 0.1
            },
            
            "rag": {
                "_comment": "RAG system optimization settings",
                "target_chunk_size": 512,
                "max_chunk_size": 600,
                "min_chunk_size": 50,
                "min_quality_score": 0.6,
                "min_confidence_score": 0.7,
                "max_low_confidence_regions": 2
            },
            
            "pdf": {
                "_comment": "PDF-specific processing settings",
                "heading_font_threshold": 14.0,
                "large_heading_threshold": 18.0,
                "medium_heading_threshold": 16.0,
                "zoom_factor": 2.0,
                "enable_table_extraction": True
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(example_config, f, indent=2, ensure_ascii=False)


# Create default configuration instance
DEFAULT_CONFIG = DocumentExtractorConfig()


def load_config(config_path: Optional[Path] = None) -> DocumentExtractorConfig:
    """
    Load configuration from file or return default configuration.
    
    Args:
        config_path: Path to configuration file, None for default config
        
    Returns:
        DocumentExtractorConfig instance
    """
    if config_path and config_path.exists():
        return DocumentExtractorConfig.from_file(config_path)
    else:
        return DocumentExtractorConfig()