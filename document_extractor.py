#!/usr/bin/env python3
"""
Document Content Extractor
Extracts text and images from various document formats while preserving structure.
Supports: PDF, EPUB, DOCX, DJVU
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
import mimetypes
import base64

# PDF processing
import PyPDF2
import fitz  # PyMuPDF for better PDF handling
import pdfplumber

# EPUB processing
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# DOCX processing
import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

# DJVU processing (requires external tools)
import subprocess
import tempfile

# OCR for images and DJVU
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Common utilities
import re
import shutil
from datetime import datetime

# Configuration system
from extractor_config import DocumentExtractorConfig, load_config

@dataclass
class ExtractedContent:
    """Structure for extracted content with RAG optimization"""
    text: str
    content_type: str  # paragraph, heading, table, caption, etc.
    level: int = 0  # for headings
    page_number: Optional[int] = None
    chapter: Optional[str] = None
    
    # RAG-specific fields
    semantic_context: Optional[str] = None  # surrounding context
    technique_mentions: List[str] = None    # detected technique names
    confidence_score: float = 1.0           # OCR confidence
    chunk_id: Optional[str] = None          # for RAG chunking
    quality_score: float = 1.0              # content quality for RAG
    raw_text: Optional[str] = None          # original OCR text before processing
    low_confidence_regions: List[str] = None # uncertain OCR regions
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.technique_mentions is None:
            self.technique_mentions = []
        if self.low_confidence_regions is None:
            self.low_confidence_regions = []

@dataclass
class ExtractedImage:
    """Structure for extracted images with RAG optimization"""
    filename: str
    format: str
    width: int
    height: int
    page_number: Optional[int] = None
    caption: Optional[str] = None
    
    # RAG-specific fields
    image_type: Optional[str] = None        # diagram, photo, illustration, etc.
    technique_demonstrations: List[str] = None # detected techniques shown
    ocr_text: Optional[str] = None          # text extracted from image
    embedding_vector: Optional[List[float]] = None # for multimodal RAG
    relevance_score: float = 1.0            # importance for martial arts content
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.technique_demonstrations is None:
            self.technique_demonstrations = []

@dataclass
class DocumentStructure:
    """Overall document structure"""
    title: str
    author: Optional[str] = None
    format: str = ""
    pages: int = 0
    chapters: List[str] = None
    content: List[ExtractedContent] = None
    images: List[ExtractedImage] = None
    metadata: Dict[str, Any] = None

class DocumentExtractor:
    """Main document extraction class"""
    
    def __init__(self, config: Optional[DocumentExtractorConfig] = None, **kwargs):
        """
        Initialize DocumentExtractor with configuration.
        
        Args:
            config: DocumentExtractorConfig instance, or None for default
            **kwargs: Override specific config values (for backward compatibility)
        """
        # Load configuration
        if config is None:
            self.config = DocumentExtractorConfig()
        else:
            self.config = config
        
        # Apply any kwargs overrides for backward compatibility
        if 'output_dir' in kwargs:
            self.config.output_dir = kwargs['output_dir']
        if 'extract_embedded_images' in kwargs:
            self.config.extract_embedded_images = kwargs['extract_embedded_images']
        if 'ocr_preprocessing' in kwargs:
            self.config.ocr.enable_preprocessing = not kwargs.get('no_ocr_preprocessing', False)
        if 'ocr_dpi' in kwargs:
            self.config.ocr.dpi = kwargs['ocr_dpi']
        if 'ocr_lang' in kwargs:
            self.config.ocr.language = kwargs['ocr_lang']
        if 'ocr_psm' in kwargs:
            self.config.ocr.page_segmentation_mode = kwargs['ocr_psm']
        if 'ocr_oem' in kwargs:
            self.config.ocr.ocr_engine_mode = kwargs['ocr_oem']
        
        # Setup paths and logging
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def extract_document(self, file_path: str) -> DocumentStructure:
        """Extract content from a document based on its format"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        extension = file_path.suffix.lower()
        
        self.logger.info(f"Processing {file_path.name} ({extension})")
        
        # Create output directory for this document
        doc_output_dir = self.output_dir / file_path.stem
        doc_output_dir.mkdir(exist_ok=True)
        
        # Route to appropriate extractor
        if extension == '.pdf':
            return self._extract_pdf(file_path, doc_output_dir)
        elif extension == '.epub':
            return self._extract_epub(file_path, doc_output_dir)
        elif extension == '.docx':
            return self._extract_docx(file_path, doc_output_dir)
        elif extension == '.djvu':
            return self._extract_djvu(file_path, doc_output_dir)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _extract_pdf(self, file_path: Path, output_dir: Path) -> DocumentStructure:
        """Extract content from PDF files"""
        self.logger.info(f"Extracting PDF: {file_path.name}")
        
        # Initialize structure
        doc_structure = DocumentStructure(
            title=file_path.stem,
            format="PDF",
            content=[],
            images=[],
            chapters=[],
            metadata={}
        )
        
        # Open PDF with PyMuPDF for comprehensive extraction
        pdf_doc = fitz.open(str(file_path))
        doc_structure.pages = len(pdf_doc)
        
        # Extract metadata
        metadata = pdf_doc.metadata
        doc_structure.metadata = metadata
        doc_structure.title = metadata.get('title', file_path.stem)
        doc_structure.author = metadata.get('author')
        
        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Extract content page by page
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            
            # Extract text with structure
            text_dict = page.get_text("dict")
            extracted_text = self._process_pdf_page_structure(text_dict, page_num + 1, doc_structure)
            
            # Check if this is a scanned page (little to no extractable text)
            is_scanned_page = len(extracted_text.strip()) < self.config.ocr.min_scanned_page_text
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                self._extract_pdf_image(pdf_doc, img, page_num + 1, img_index, images_dir, doc_structure)
            
            # If this appears to be a scanned page, convert to image and apply OCR + image extraction
            if is_scanned_page and self.config.extract_embedded_images:
                self.logger.info(f"Detected scanned PDF page {page_num + 1}, applying OCR and image extraction")
                try:
                    # Convert page to image
                    zoom = self.config.pdf.zoom_factor
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                    page_img_path = images_dir / f"page_{page_num + 1:03d}_ocr.png"
                    pix.save(str(page_img_path))
                    
                    # Process with OCR and image extraction
                    self._process_page_with_ocr(page_img_path, page_num + 1, images_dir, doc_structure)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process scanned page {page_num + 1}: {e}")
        
        pdf_doc.close()
        
        # Also try pdfplumber for better table extraction
        self._extract_pdf_tables(file_path, doc_structure)
        
        return doc_structure
    
    def _process_pdf_page_structure(self, text_dict: Dict, page_num: int, doc_structure: DocumentStructure) -> str:
        """Process PDF page structure to identify headings, paragraphs, etc.
        
        Returns:
            Combined text from all blocks on the page
        """
        blocks = text_dict.get("blocks", [])
        all_page_text = ""
        
        for block in blocks:
            if "lines" in block:
                # Text block
                block_text = ""
                font_sizes = []
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_sizes.append(span["size"])
                    block_text += line_text + "\n"
                
                block_text = block_text.strip()
                if not block_text:
                    continue
                
                all_page_text += block_text + " "
                
                # Determine content type based on font size and formatting
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                content_type = self._classify_pdf_content(block_text, avg_font_size)
                
                # Extract heading level
                level = 0
                if content_type == "heading":
                    level = self._determine_heading_level(avg_font_size)
                    if block_text not in doc_structure.chapters:
                        doc_structure.chapters.append(block_text)
                
                content = ExtractedContent(
                    text=block_text,
                    content_type=content_type,
                    level=level,
                    page_number=page_num,
                    metadata={"font_size": avg_font_size}
                )
                
                doc_structure.content.append(content)
        
        return all_page_text
    
    def _classify_pdf_content(self, text: str, font_size: float) -> str:
        """Classify content type based on text and formatting"""
        # Simple heuristics - can be enhanced
        if font_size > self.config.pdf.heading_font_threshold:
            return "heading"
        elif (len(text) < self.config.pdf.caption_max_length and 
              text.isupper() and 
              len(text.upper()) / len(text) > self.config.pdf.uppercase_heading_threshold):
            return "heading"
        elif text.startswith(("Figure", "Table", "Image")):
            return "caption"
        else:
            return "paragraph"
    
    def _determine_heading_level(self, font_size: float) -> int:
        """Determine heading level based on font size"""
        if font_size > self.config.pdf.large_heading_threshold:
            return 1
        elif font_size > self.config.pdf.medium_heading_threshold:
            return 2
        elif font_size > self.config.pdf.small_heading_threshold:
            return 3
        else:
            return 4
    
    def _extract_pdf_image(self, pdf_doc, img, page_num: int, img_index: int, images_dir: Path, doc_structure: DocumentStructure):
        """Extract image from PDF"""
        xref = img[0]
        pix = fitz.Pixmap(pdf_doc, xref)
        
        if pix.n - pix.alpha < 4:  # Skip if not RGB/RGBA
            img_filename = f"page_{page_num:03d}_img_{img_index:03d}.png"
            img_path = images_dir / img_filename
            
            if pix.alpha:
                pix.pil_save(str(img_path))
            else:
                pix.pil_save(str(img_path))
            
            # Create image metadata
            image_info = ExtractedImage(
                filename=img_filename,
                format="PNG",
                width=pix.width,
                height=pix.height,
                page_number=page_num,
                metadata={"xref": xref}
            )
            
            doc_structure.images.append(image_info)
        
        pix = None
    
    def _extract_pdf_tables(self, file_path: Path, doc_structure: DocumentStructure):
        """Extract tables from PDF using pdfplumber"""
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table_index, table in enumerate(tables):
                        if table:
                            # Convert table to structured format
                            table_text = self._format_table_as_text(table)
                            
                            content = ExtractedContent(
                                text=table_text,
                                content_type="table",
                                page_number=page_num + 1,
                                metadata={"table_index": table_index}
                            )
                            
                            doc_structure.content.append(content)
        except Exception as e:
            self.logger.warning(f"Table extraction failed for {file_path.name}: {e}")
    
    def _format_table_as_text(self, table: List[List[str]]) -> str:
        """Format table data as structured text"""
        if not table:
            return ""
        
        # Simple markdown-like table format
        formatted = []
        for row_index, row in enumerate(table):
            if row:
                formatted.append(" | ".join(str(cell) if cell else "" for cell in row))
                if row_index == 0:  # Header separator
                    formatted.append("-" * len(formatted[0]))
        
        return "\n".join(formatted)
    
    def _extract_epub(self, file_path: Path, output_dir: Path) -> DocumentStructure:
        """Extract content from EPUB files"""
        self.logger.info(f"Extracting EPUB: {file_path.name}")
        
        # Initialize structure
        doc_structure = DocumentStructure(
            title=file_path.stem,
            format="EPUB",
            content=[],
            images=[],
            chapters=[],
            metadata={}
        )
        
        # Open EPUB
        book = epub.read_epub(str(file_path))
        
        # Extract metadata
        doc_structure.title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else file_path.stem
        doc_structure.author = book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else None
        
        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Extract content from each item
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                self._process_epub_html(soup, doc_structure)
            elif item.get_type() == ebooklib.ITEM_IMAGE:
                # Image content
                self._extract_epub_image(item, images_dir, doc_structure)
        
        return doc_structure
    
    def _process_epub_html(self, soup: BeautifulSoup, doc_structure: DocumentStructure):
        """Process HTML content from EPUB"""
        # Extract chapters from navigation
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(element.name[1])
            text = element.get_text(strip=True)
            
            if text:
                if level <= 2:  # Consider h1 and h2 as chapters
                    doc_structure.chapters.append(text)
                
                content = ExtractedContent(
                    text=text,
                    content_type="heading",
                    level=level,
                    metadata={"tag": element.name}
                )
                doc_structure.content.append(content)
        
        # Extract paragraphs
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                content = ExtractedContent(
                    text=text,
                    content_type="paragraph",
                    metadata={"tag": "p"}
                )
                doc_structure.content.append(content)
        
        # Extract tables
        for table in soup.find_all('table'):
            table_text = self._format_html_table(table)
            if table_text:
                content = ExtractedContent(
                    text=table_text,
                    content_type="table",
                    metadata={"tag": "table"}
                )
                doc_structure.content.append(content)
        
        # Extract image captions
        for img in soup.find_all('img'):
            alt_text = img.get('alt', '')
            if alt_text:
                content = ExtractedContent(
                    text=alt_text,
                    content_type="caption",
                    metadata={"tag": "img", "src": img.get('src', '')}
                )
                doc_structure.content.append(content)
    
    def _format_html_table(self, table) -> str:
        """Format HTML table as structured text"""
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                cells.append(td.get_text(strip=True))
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    
    def _extract_epub_image(self, item, images_dir: Path, doc_structure: DocumentStructure):
        """Extract image from EPUB"""
        img_filename = item.get_name().split('/')[-1]
        img_path = images_dir / img_filename
        
        # Write image data
        with open(img_path, 'wb') as f:
            f.write(item.get_content())
        
        # Try to get image dimensions
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                img_format = img.format
        except:
            width = height = 0
            img_format = "unknown"
        
        image_info = ExtractedImage(
            filename=img_filename,
            format=img_format,
            width=width,
            height=height,
            metadata={"media_type": item.get_type()}
        )
        
        doc_structure.images.append(image_info)
    
    def _extract_docx(self, file_path: Path, output_dir: Path) -> DocumentStructure:
        """Extract content from DOCX files"""
        self.logger.info(f"Extracting DOCX: {file_path.name}")
        
        # Initialize structure
        doc_structure = DocumentStructure(
            title=file_path.stem,
            format="DOCX",
            content=[],
            images=[],
            chapters=[],
            metadata={}
        )
        
        # Open DOCX
        doc = docx.Document(str(file_path))
        
        # Extract metadata
        core_props = doc.core_properties
        doc_structure.title = core_props.title or file_path.stem
        doc_structure.author = core_props.author
        doc_structure.metadata = {
            "subject": core_props.subject,
            "created": core_props.created.isoformat() if core_props.created else None,
            "modified": core_props.modified.isoformat() if core_props.modified else None
        }
        
        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Extract content from document elements
        for element in doc.element.body:
            if isinstance(element, CT_P):
                # Paragraph
                paragraph = Paragraph(element, doc)
                self._process_docx_paragraph(paragraph, doc_structure)
            elif isinstance(element, CT_Tbl):
                # Table
                table = Table(element, doc)
                self._process_docx_table(table, doc_structure)
        
        # Extract images
        self._extract_docx_images(doc, images_dir, doc_structure)
        
        return doc_structure
    
    def _process_docx_paragraph(self, paragraph: Paragraph, doc_structure: DocumentStructure):
        """Process DOCX paragraph"""
        text = paragraph.text.strip()
        if not text:
            return
        
        # Determine content type based on style
        style_name = paragraph.style.name if paragraph.style else "Normal"
        
        if "Heading" in style_name:
            content_type = "heading"
            level = int(style_name.split()[-1]) if style_name.split()[-1].isdigit() else 1
            if level <= 2:
                doc_structure.chapters.append(text)
        elif "Caption" in style_name:
            content_type = "caption"
            level = 0
        else:
            content_type = "paragraph"
            level = 0
        
        content = ExtractedContent(
            text=text,
            content_type=content_type,
            level=level,
            metadata={"style": style_name}
        )
        
        doc_structure.content.append(content)
    
    def _process_docx_table(self, table: Table, doc_structure: DocumentStructure):
        """Process DOCX table"""
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                row_data.append(cell.text.strip())
            table_data.append(row_data)
        
        table_text = self._format_table_as_text(table_data)
        
        content = ExtractedContent(
            text=table_text,
            content_type="table",
            metadata={"rows": len(table_data), "columns": len(table_data[0]) if table_data else 0}
        )
        
        doc_structure.content.append(content)
    
    def _extract_docx_images(self, doc: DocxDocument, images_dir: Path, doc_structure: DocumentStructure):
        """Extract images from DOCX"""
        # This is a simplified version - full implementation would need python-docx2txt or similar
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_part = rel.target_part
                    img_filename = rel.target_ref.split('/')[-1]
                    img_path = images_dir / img_filename
                    
                    with open(img_path, 'wb') as f:
                        f.write(image_part.blob)
                    
                    # Try to get image dimensions
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            img_format = img.format
                    except:
                        width = height = 0
                        img_format = "unknown"
                    
                    image_info = ExtractedImage(
                        filename=img_filename,
                        format=img_format,
                        width=width,
                        height=height,
                        metadata={"relationship": rel.target_ref}
                    )
                    
                    doc_structure.images.append(image_info)
                except Exception as e:
                    self.logger.warning(f"Failed to extract image {rel.target_ref}: {e}")
    
    def _extract_djvu(self, file_path: Path, output_dir: Path) -> DocumentStructure:
        """Extract content from DJVU files using OCR"""
        self.logger.info(f"Extracting DJVU: {file_path.name}")
        
        # Initialize structure
        doc_structure = DocumentStructure(
            title=file_path.stem,
            format="DJVU",
            content=[],
            images=[],
            chapters=[],
            metadata={}
        )
        
        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        try:
            # Check if djvulibre tools are available
            djvused_available = shutil.which("djvused") is not None
            ddjvu_available = shutil.which("ddjvu") is not None
            
            if not (djvused_available and ddjvu_available):
                self.logger.error("DJVU tools not available. Install with: sudo apt install djvulibre-bin")
                content = ExtractedContent(
                    text="DJVU extraction requires djvulibre-bin package. Install with: sudo apt install djvulibre-bin",
                    content_type="error",
                    metadata={"missing_tools": ["djvused", "ddjvu"]}
                )
                doc_structure.content.append(content)
                return doc_structure
            
            # Convert DJVU to images using ddjvu
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Get page count
                result = subprocess.run(
                    ["djvused", str(file_path), "-e", "n"],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    page_count = int(result.stdout.strip())
                    doc_structure.pages = page_count
                    
                    # Extract each page
                    for page_num in range(1, page_count + 1):
                        page_img = temp_path / f"page_{page_num:03d}.png"
                        
                        # Convert page to image (use ppm format then convert to png)
                        page_ppm = temp_path / f"page_{page_num:03d}.ppm"
                        subprocess.run([
                            "ddjvu", "-format=ppm", "-page={}".format(page_num),
                            str(file_path), str(page_ppm)
                        ])
                        
                        # Convert ppm to png using PIL
                        if page_ppm.exists():
                            with Image.open(page_ppm) as img:
                                img.save(page_img, "PNG")
                        
                        if page_img.exists():
                            # OCR the page with enhanced configuration
                            ocr_result = self._perform_ocr_with_config(page_img)
                            
                            if ocr_result['text'].strip():
                                content = ExtractedContent(
                                    text=ocr_result['text'],
                                    content_type="paragraph",
                                    page_number=page_num,
                                    raw_text=ocr_result['raw_text'],
                                    confidence_score=ocr_result['confidence_score'],
                                    quality_score=ocr_result['quality_score'],
                                    technique_mentions=ocr_result['technique_mentions'],
                                    low_confidence_regions=ocr_result['low_confidence_regions'],
                                    metadata={
                                        "source": "Enhanced_OCR", 
                                        "language": self.ocr_lang, 
                                        "dpi": self.ocr_dpi,
                                        "chinese_ratio": ocr_result['chinese_ratio'],
                                        "has_techniques": ocr_result['has_techniques']
                                    }
                                )
                                doc_structure.content.append(content)
                            
                            # Save the page image
                            final_img_path = images_dir / f"page_{page_num:03d}.png"
                            shutil.copy2(page_img, final_img_path)
                            
                            # Get image info
                            with Image.open(final_img_path) as img:
                                width, height = img.size
                            
                            image_info = ExtractedImage(
                                filename=f"page_{page_num:03d}.png",
                                format="PNG",
                                width=width,
                                height=height,
                                page_number=page_num,
                                metadata={"source": "DJVU_conversion"}
                            )
                            
                            doc_structure.images.append(image_info)
                            
                            # NEW: Extract images from within the page using computer vision
                            if self.extract_embedded_images:
                                self.logger.info(f"Extracting embedded images from DJVU page {page_num}")
                                self._detect_and_extract_images_from_page(final_img_path, page_num, images_dir, doc_structure)
                else:
                    self.logger.error(f"Failed to get page count from DJVU: {result.stderr}")
                    content = ExtractedContent(
                        text=f"Failed to process DJVU file: {result.stderr}",
                        content_type="error",
                        metadata={"djvused_error": result.stderr}
                    )
                    doc_structure.content.append(content)
                
        except Exception as e:
            self.logger.error(f"DJVU extraction failed: {e}")
            content = ExtractedContent(
                text=f"DJVU extraction failed: {e}",
                content_type="error",
                metadata={"error": str(e)}
            )
            doc_structure.content.append(content)
        
        return doc_structure
    
    def _detect_and_extract_images_from_page(self, page_image_path: Path, page_num: int, images_dir: Path, doc_structure: DocumentStructure):
        """
        Detect and extract images from a page using computer vision.
        
        Args:
            page_image_path: Path to the page image
            page_num: Page number
            images_dir: Directory to save extracted images
            doc_structure: Document structure to add images to
        """
        try:
            # Read the page image
            page_img = cv2.imread(str(page_image_path))
            if page_img is None:
                return
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Edge detection to find rectangular regions
            edges = cv2.Canny(gray, 
                            self.config.image_detection.canny_low_threshold,
                            self.config.image_detection.canny_high_threshold, 
                            apertureSize=self.config.image_detection.canny_aperture_size)
            
            # Use morphological operations to connect broken edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.image_detection.morph_kernel_size)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            extracted_count = 0
            
            for i, contour in enumerate(contours):
                # Calculate area
                area = cv2.contourArea(contour)
                
                if area < self.config.image_detection.min_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very thin rectangles (likely text lines) and very small regions
                aspect_ratio = w / h
                if (aspect_ratio > self.config.image_detection.max_aspect_ratio or 
                    aspect_ratio < self.config.image_detection.min_aspect_ratio or 
                    w < self.config.image_detection.min_width or 
                    h < self.config.image_detection.min_height):
                    continue
                
                # Extract the region
                roi = page_img[y:y+h, x:x+w]
                
                # Check if this looks like an image (not just text)
                if self._is_likely_image_region(roi):
                    # Save the extracted image
                    extracted_filename = f"page_{page_num:03d}_extracted_{extracted_count:03d}.png"
                    extracted_path = images_dir / extracted_filename
                    
                    cv2.imwrite(str(extracted_path), roi)
                    
                    # Add to structure
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
                            "aspect_ratio": aspect_ratio
                        }
                    )
                    
                    doc_structure.images.append(image_info)
                    extracted_count += 1
            
            # Method 2: Template matching for common image patterns
            self._extract_images_by_template_matching(page_img, page_num, images_dir, doc_structure, extracted_count)
            
        except Exception as e:
            self.logger.warning(f"Image extraction failed for page {page_num}: {e}")
    
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
        
        # Skip very small regions
        if (height < self.config.image_detection.min_height or 
            width < self.config.image_detection.min_width):
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
        
        # Heuristic scoring using config values
        image_score = 0
        
        if variance > self.config.image_detection.variance_threshold:
            image_score += 1
        if (self.config.image_detection.edge_density_min < edge_density < 
            self.config.image_detection.edge_density_max):
            image_score += 1
        if entropy > self.config.image_detection.entropy_threshold:
            image_score += 1
        
        # Additional check: look for text-like patterns
        # Text regions often have horizontal structures
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_density = np.sum(horizontal_lines > 0) / (width * height)
        
        if horizontal_density > self.config.image_detection.horizontal_line_threshold:
            image_score -= 1
        
        return image_score >= 2
    
    def _extract_images_by_template_matching(self, page_img: np.ndarray, page_num: int, images_dir: Path, doc_structure: DocumentStructure, start_count: int):
        """
        Extract images using template matching for common patterns.
        This method looks for rectangular regions with image-like characteristics.
        """
        try:
            gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding to separate foreground from background
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find connected components that might be images
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                thresh, connectivity=self.config.image_detection.connectivity)
            
            extracted_count = start_count
            
            for i in range(1, num_labels):  # Skip background (label 0)
                x, y, w, h, area = stats[i]
                
                # Filter by size using config values
                if (area < self.config.image_detection.cc_min_area or 
                    w < self.config.image_detection.min_width or 
                    h < self.config.image_detection.min_height):
                    continue
                
                # Extract the region
                roi = page_img[y:y+h, x:x+w]
                
                # Check if this looks like an image
                if self._is_likely_image_region(roi):
                    # Save the extracted image
                    extracted_filename = f"page_{page_num:03d}_extracted_{extracted_count:03d}.png"
                    extracted_path = images_dir / extracted_filename
                    
                    cv2.imwrite(str(extracted_path), roi)
                    
                    # Add to structure
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
                    
                    doc_structure.images.append(image_info)
                    extracted_count += 1
                    
        except Exception as e:
            self.logger.warning(f"Template matching extraction failed for page {page_num}: {e}")
    
    def _preprocess_image_for_ocr(self, image_path: Path) -> Path:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path: Path to the original image
            
        Returns:
            Path to the preprocessed image
        """
        if not self.config.ocr.enable_preprocessing:
            return image_path
            
        try:
            # Read the original image
            img = cv2.imread(str(image_path))
            if img is None:
                return image_path
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, self.config.ocr.gaussian_blur_kernel, 0)
            
            # Apply adaptive thresholding for better text contrast
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                self.config.ocr.adaptive_threshold_block_size, 
                self.config.ocr.adaptive_threshold_c
            )
            
            # Morphological operations to clean up the image
            kernel = np.ones(self.config.ocr.morphology_kernel_size, np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Resize image for better OCR (if DPI is specified)
            if self.config.ocr.dpi != 300:  # Default is 300 DPI
                height, width = cleaned.shape
                # Scale factor based on DPI
                scale_factor = self.config.ocr.dpi / 300.0
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Save preprocessed image
            preprocessed_path = image_path.parent / f"preprocessed_{image_path.name}"
            cv2.imwrite(str(preprocessed_path), cleaned)
            
            return preprocessed_path
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed for {image_path}: {e}")
            return image_path
    
    def _perform_ocr_with_config(self, image_path: Path) -> Dict[str, Any]:
        """
        Perform OCR with enhanced configuration options and post-processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with processed text, raw text, and confidence data
        """
        try:
            # Preprocess image for better OCR
            processed_image_path = self._preprocess_image_for_ocr(image_path)
            
            # Configure Tesseract with custom settings
            custom_config = f'--oem {self.config.ocr.ocr_engine_mode} --psm {self.config.ocr.page_segmentation_mode} -l {self.config.ocr.language}'
            
            # Additional OCR configuration for better accuracy
            if self.config.ocr.dpi != 300:
                custom_config += f' --dpi {self.config.ocr.dpi}'
            
            # Perform OCR
            raw_ocr_text = pytesseract.image_to_string(
                Image.open(processed_image_path), 
                config=custom_config
            )
            
            # Get confidence data
            try:
                confidence_data = pytesseract.image_to_data(
                    Image.open(processed_image_path),
                    config=custom_config,
                    output_type=pytesseract.Output.DICT
                )
            except:
                confidence_data = None
            
            # Post-process the OCR text
            processed_result = self._post_process_ocr_text(raw_ocr_text, confidence_data)
            
            # Clean up preprocessed image if it's different from original
            if processed_image_path != image_path and processed_image_path.exists():
                processed_image_path.unlink()
                
            return processed_result
            
        except Exception as e:
            self.logger.warning(f"Enhanced OCR failed for {image_path}, falling back to basic OCR: {e}")
            # Fallback to basic OCR
            raw_text = pytesseract.image_to_string(Image.open(image_path), lang=self.config.ocr.language)
            return self._post_process_ocr_text(raw_text, None)
    
    def _post_process_ocr_text(self, raw_text: str, confidence_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Conservative post-processing for martial arts domain OCR text.
        
        Args:
            raw_text: Raw OCR output text
            confidence_data: Tesseract confidence data
            
        Returns:
            Dictionary with processed text and metadata
        """
        # Character-level OCR fixes (safe substitutions only)
        ocr_fixes = {
            r'\brn\b': 'm',              # common OCR error: rn -> m
            r'\bcl\b': 'd',              # cl misread as d
            r'\bli\b': 'h',              # li misread as h
            rf'(\w)\1{{{self.config.text_processing.max_char_repetition},}}': r'\1\1',  # reduce excessive repetition
            rf'\s{{{self.config.text_processing.excessive_whitespace_threshold},}}': ' ',  # normalize whitespace
            r'[^\w\s\u4e00-\u9fff.,;:!?()"\'-]': '',  # remove garbage characters but keep Chinese
        }
        
        processed_text = raw_text
        for pattern, replacement in ocr_fixes.items():
            processed_text = re.sub(pattern, replacement, processed_text)
        
        # Extract confidence scores and identify low-confidence regions
        confidence_scores = []
        low_confidence_regions = []
        avg_confidence = 1.0
        
        if confidence_data:
            try:
                confidences = [int(conf) for conf in confidence_data['conf'] if int(conf) > 0]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences) / 100.0  # normalize to 0-1
                    
                    # Identify low confidence words
                    words = confidence_data['text']
                    word_confidences = confidence_data['conf']
                    
                    for word, conf in zip(words, word_confidences):
                        if word.strip() and int(conf) < self.config.ocr.low_confidence_threshold:
                            low_confidence_regions.append(word.strip())
            except Exception as e:
                self.logger.debug(f"Could not process confidence data: {e}")
        
        # Detect martial arts content
        martial_arts_analysis = self._detect_martial_arts_content(processed_text)
        
        return {
            'text': processed_text.strip(),
            'raw_text': raw_text,
            'confidence_score': avg_confidence,
            'low_confidence_regions': low_confidence_regions,
            'technique_mentions': martial_arts_analysis['techniques'],
            'chinese_ratio': martial_arts_analysis['chinese_ratio'],
            'has_techniques': martial_arts_analysis['has_techniques'],
            'quality_score': self._calculate_text_quality(processed_text, avg_confidence)
        }
    
    def _detect_martial_arts_content(self, text: str) -> Dict[str, Any]:
        """
        Detect and tag martial arts-specific content.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with martial arts content analysis
        """
        # Use technique patterns from config
        technique_patterns = self.config.martial_arts.technique_patterns
        
        detected_techniques = []
        for pattern in technique_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            detected_techniques.extend(matches)
        
        # Remove duplicates while preserving order
        detected_techniques = list(dict.fromkeys(detected_techniques))
        
        # Calculate Chinese character ratio
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', ''))  # exclude spaces from count
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        
        # Use martial arts keywords from config
        martial_arts_keywords = self.config.martial_arts.martial_arts_keywords
        
        keyword_count = sum(1 for keyword in martial_arts_keywords if keyword.lower() in text.lower())
        
        return {
            'techniques': detected_techniques,
            'chinese_ratio': chinese_ratio,
            'has_techniques': len(detected_techniques) > 0,
            'keyword_density': keyword_count / len(text.split()) if text.split() else 0,
            'content_relevance': min(1.0, (
                len(detected_techniques) * self.config.martial_arts.technique_relevance_weight + 
                chinese_ratio * self.config.martial_arts.chinese_relevance_weight + 
                keyword_count * self.config.martial_arts.keyword_relevance_weight))
        }
    
    def _calculate_text_quality(self, text: str, confidence_score: float) -> float:
        """
        Calculate content quality score for RAG prioritization.
        
        Args:
            text: Text content to score
            confidence_score: OCR confidence score (0-1)
            
        Returns:
            Quality score (0-1)
        """
        if not text or len(text.strip()) < self.config.text_processing.min_text_length_quality:
            return 0.0
        
        score = confidence_score
        
        # Length scoring (prefer substantial content)
        text_length = len(text.strip())
        if text_length < self.config.text_processing.min_text_length_quality:
            score *= 0.5
        elif text_length > self.config.text_processing.optimal_text_length:
            score = min(1.0, score * self.config.text_processing.quality_length_bonus)
        
        # Character quality (penalize excessive special characters)
        clean_chars = len(re.findall(r'[\w\s\u4e00-\u9fff.,;:!?()"\'-]', text))
        char_quality = clean_chars / len(text) if text else 0
        if char_quality < self.config.text_processing.min_clean_char_ratio:
            score *= char_quality
        
        # Readability (prefer balanced whitespace)
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if (self.config.text_processing.min_reasonable_word_length <= avg_word_length <= 
                self.config.text_processing.max_reasonable_word_length):
                score = min(1.0, score * self.config.text_processing.word_length_bonus)
        
        return max(0.0, min(1.0, score))
    
    def _process_page_with_ocr(self, page_image_path: Path, page_num: int, images_dir: Path, doc_structure: DocumentStructure):
        """
        Process a page image using OCR and extract images from it.
        This method is used for full OCR workflows on scanned documents.
        
        Args:
            page_image_path: Path to the page image file
            page_num: Page number
            images_dir: Directory to save extracted images
            doc_structure: Document structure to update
        """
        try:
            # OCR the page with enhanced configuration
            ocr_result = self._perform_ocr_with_config(page_image_path)
            
            if ocr_result['text'].strip():
                content = ExtractedContent(
                    text=ocr_result['text'],
                    content_type="paragraph",
                    page_number=page_num,
                    raw_text=ocr_result['raw_text'],
                    confidence_score=ocr_result['confidence_score'],
                    quality_score=ocr_result['quality_score'],
                    technique_mentions=ocr_result['technique_mentions'],
                    low_confidence_regions=ocr_result['low_confidence_regions'],
                    metadata={
                        "source": "Enhanced_OCR", 
                        "language": self.ocr_lang, 
                        "dpi": self.ocr_dpi,
                        "chinese_ratio": ocr_result['chinese_ratio'],
                        "has_techniques": ocr_result['has_techniques']
                    }
                )
                doc_structure.content.append(content)
            
            # Extract images from within the page using computer vision
            if self.extract_embedded_images:
                self.logger.info(f"Extracting embedded images from OCR page {page_num}")
                self._detect_and_extract_images_from_page(page_image_path, page_num, images_dir, doc_structure)
            
        except Exception as e:
            self.logger.error(f"OCR processing failed for page {page_num}: {e}")
            content = ExtractedContent(
                text=f"OCR processing failed for page {page_num}: {e}",
                content_type="error",
                page_number=page_num,
                metadata={"error": str(e)}
            )
            doc_structure.content.append(content)
    
    def _create_rag_chunks(self, doc_structure: DocumentStructure) -> List[Dict]:
        """
        Create semantically meaningful chunks for RAG embeddings.
        
        Args:
            doc_structure: Document structure to chunk
            
        Returns:
            List of RAG-optimized chunks
        """
        chunks = []
        current_chunk = []
        current_context = ""
        chunk_id = 0
        
        for content in doc_structure.content:
            # Preserve technique descriptions as complete chunks
            if content.content_type == "heading":
                if current_chunk:
                    chunks.append(self._finalize_chunk(current_chunk, current_context, chunk_id))
                    chunk_id += 1
                    current_chunk = []
                current_context = content.text
            
            current_chunk.append(content)
            
            # Smart chunking based on content type and length
            chunk_text = " ".join([c.text for c in current_chunk])
            
            # Chunk at natural boundaries
            should_chunk = (
                len(chunk_text) > self.config.rag.target_chunk_size or  # size threshold
                content.content_type == "table" or  # tables are standalone
                (content.content_type == "paragraph" and len(chunk_text) > self.config.rag.technique_chunk_threshold and 
                 hasattr(content, 'technique_mentions') and content.technique_mentions)  # technique descriptions
            )
            
            if should_chunk:
                chunks.append(self._finalize_chunk(current_chunk, current_context, chunk_id))
                chunk_id += 1
                current_chunk = []
        
        # Handle remaining content
        if current_chunk:
            chunks.append(self._finalize_chunk(current_chunk, current_context, chunk_id))
        
        return chunks
    
    def _finalize_chunk(self, content_list: List[ExtractedContent], context: str, chunk_id: int) -> Dict:
        """
        Create a finalized chunk for RAG processing.
        
        Args:
            content_list: List of content items to include
            context: Current chapter/section context
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Dictionary representing the chunk
        """
        if not content_list:
            return {}
        
        combined_text = " ".join([c.text for c in content_list])
        
        # Aggregate technique mentions
        all_techniques = []
        total_confidence = 0
        total_quality = 0
        low_confidence_regions = []
        
        for content in content_list:
            if hasattr(content, 'technique_mentions') and content.technique_mentions:
                all_techniques.extend(content.technique_mentions)
            if hasattr(content, 'confidence_score'):
                total_confidence += content.confidence_score
            if hasattr(content, 'quality_score'):
                total_quality += content.quality_score
            if hasattr(content, 'low_confidence_regions') and content.low_confidence_regions:
                low_confidence_regions.extend(content.low_confidence_regions)
        
        # Remove duplicate techniques
        unique_techniques = list(dict.fromkeys(all_techniques))
        
        # Calculate average scores
        avg_confidence = total_confidence / len(content_list) if content_list else 0
        avg_quality = total_quality / len(content_list) if content_list else 0
        
        # Determine chunk type
        chunk_types = [c.content_type for c in content_list]
        primary_type = max(set(chunk_types), key=chunk_types.count)
        
        return {
            'chunk_id': f"chunk_{chunk_id:04d}",
            'text': combined_text.strip(),
            'context': context,
            'content_types': list(set(chunk_types)),
            'primary_type': primary_type,
            'page_numbers': list(set([c.page_number for c in content_list if c.page_number])),
            'technique_mentions': unique_techniques,
            'confidence_score': avg_confidence,
            'quality_score': avg_quality,
            'low_confidence_regions': list(set(low_confidence_regions)),
            'word_count': len(combined_text.split()),
            'char_count': len(combined_text),
            'has_techniques': len(unique_techniques) > 0,
            'chinese_ratio': len(re.findall(r'[\u4e00-\u9fff]', combined_text)) / len(combined_text) if combined_text else 0
        }
    
    def _extract_image_content_for_rag(self, doc_structure: DocumentStructure) -> List[Dict]:
        """
        Process images for RAG context with OCR and classification.
        
        Args:
            doc_structure: Document structure containing images
            
        Returns:
            List of processed images with RAG metadata
        """
        processed_images = []
        
        for image in doc_structure.images:
            # Determine image type based on metadata and filename
            image_type = self._classify_image_type(image)
            
            # Extract text from image if it looks like a diagram or has text
            ocr_text = ""
            if image_type in ["diagram", "illustration", "mixed"]:
                try:
                    img_path = Path(image.filename)
                    if img_path.exists():
                        ocr_result = self._perform_ocr_with_config(img_path)
                        ocr_text = ocr_result['text']
                except Exception as e:
                    self.logger.debug(f"Could not OCR image {image.filename}: {e}")
            
            # Detect martial arts techniques in image
            technique_demonstrations = []
            if image_type in ["photo", "illustration"] and (image.caption or ocr_text):
                analysis_text = f"{image.caption or ''} {ocr_text}".strip()
                if analysis_text:
                    ma_analysis = self._detect_martial_arts_content(analysis_text)
                    technique_demonstrations = ma_analysis['techniques']
            
            # Calculate relevance score
            relevance_score = self._calculate_image_relevance(image, image_type, technique_demonstrations, ocr_text)
            
            processed_image = {
                'filename': image.filename,
                'format': image.format,
                'dimensions': {'width': image.width, 'height': image.height},
                'page_number': image.page_number,
                'caption': image.caption,
                'image_type': image_type,
                'ocr_text': ocr_text,
                'technique_demonstrations': technique_demonstrations,
                'relevance_score': relevance_score,
                'area': image.width * image.height,
                'aspect_ratio': image.width / image.height if image.height > 0 else 0,
                'metadata': image.metadata or {}
            }
            
            processed_images.append(processed_image)
        
        return processed_images
    
    def _classify_image_type(self, image: ExtractedImage) -> str:
        """Classify image type for RAG processing."""
        # Use metadata hints if available
        if image.metadata and 'source' in image.metadata:
            source = image.metadata['source']
            if 'CV_extraction' in source:
                return "illustration"  # Computer vision extracted images are likely illustrations
        
        # Size-based classification
        area = image.width * image.height
        aspect_ratio = image.width / image.height if image.height > 0 else 1
        
        if area > 300000:  # Large images
            if 0.7 < aspect_ratio < 1.3:  # Square-ish
                return "photo"
            else:
                return "diagram"
        elif area > 100000:  # Medium images
            return "illustration"
        else:  # Small images
            return "icon"
    
    def _calculate_image_relevance(self, image: ExtractedImage, image_type: str, techniques: List[str], ocr_text: str) -> float:
        """Calculate image relevance score for martial arts content."""
        score = 0.5  # base score
        
        # Type-based scoring
        type_scores = {
            "photo": 0.8,
            "illustration": 0.9,
            "diagram": 0.7,
            "icon": 0.3
        }
        score *= type_scores.get(image_type, 0.5)
        
        # Technique demonstration bonus
        if techniques:
            score += len(techniques) * 0.1
        
        # OCR text content bonus
        if ocr_text and len(ocr_text.strip()) > 20:
            ma_analysis = self._detect_martial_arts_content(ocr_text)
            score += ma_analysis['content_relevance'] * 0.3
        
        # Caption analysis
        if image.caption:
            ma_analysis = self._detect_martial_arts_content(image.caption)
            score += ma_analysis['content_relevance'] * 0.2
        
        # Size consideration (larger images often more important)
        area = image.width * image.height
        if area > 200000:
            score *= 1.1
        
        return min(1.0, score)
    
    def save_structure_for_rag(self, doc_structure: DocumentStructure, output_dir: Path):
        """
        Save document structure in RAG-optimized format.
        
        Args:
            doc_structure: Document structure to save
            output_dir: Output directory
        """
        # Save original structure first
        self.save_structure(doc_structure, output_dir)
        
        # Create RAG-specific outputs
        rag_chunks = self._create_rag_chunks(doc_structure)
        processed_images = self._extract_image_content_for_rag(doc_structure)
        
        # Filter high-quality content for embeddings
        high_quality_chunks = [
            chunk for chunk in rag_chunks 
            if (chunk.get('quality_score', 0) > self.config.rag.min_quality_score and 
                chunk.get('word_count', 0) > self.config.rag.min_chunk_size)
        ]
        
        # Extract content needing review
        review_needed = [
            chunk for chunk in rag_chunks
            if (chunk.get('confidence_score', 1) < self.config.rag.min_confidence_score or 
                len(chunk.get('low_confidence_regions', [])) > self.config.rag.max_low_confidence_regions)
        ]
        
        # Build technique index
        technique_index = {}
        for chunk in rag_chunks:
            for technique in chunk.get('technique_mentions', []):
                if technique not in technique_index:
                    technique_index[technique] = []
                technique_index[technique].append({
                    'chunk_id': chunk['chunk_id'],
                    'context': chunk.get('context', ''),
                    'page_numbers': chunk.get('page_numbers', [])
                })
        
        # Create RAG output structure
        rag_output = {
            'document_metadata': {
                'title': doc_structure.title,
                'author': doc_structure.author,
                'format': doc_structure.format,
                'pages': doc_structure.pages,
                'total_chunks': len(rag_chunks),
                'high_quality_chunks': len(high_quality_chunks),
                'total_images': len(processed_images),
                'relevant_images': len([img for img in processed_images if img['relevance_score'] > 0.7])
            },
            'rag_chunks': high_quality_chunks,
            'all_chunks': rag_chunks,
            'processed_images': processed_images,
            'low_confidence_content': review_needed,
            'technique_index': technique_index,
            'extraction_stats': {
                'avg_chunk_quality': sum(c.get('quality_score', 0) for c in rag_chunks) / len(rag_chunks) if rag_chunks else 0,
                'avg_confidence': sum(c.get('confidence_score', 0) for c in rag_chunks) / len(rag_chunks) if rag_chunks else 0,
                'total_techniques': len(technique_index),
                'multilingual_chunks': len([c for c in rag_chunks if c.get('chinese_ratio', 0) > 0.1])
            }
        }
        
        # Save RAG-ready JSON
        rag_path = output_dir / "rag_ready.json"
        with open(rag_path, 'w', encoding='utf-8') as f:
            json.dump(rag_output, f, indent=2, ensure_ascii=False, default=str)
        
        # Save embeddings-ready text chunks
        embeddings_path = output_dir / "embeddings_chunks.jsonl"
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            for chunk in high_quality_chunks:
                embedding_doc = {
                    'id': chunk['chunk_id'],
                    'text': chunk['text'],
                    'metadata': {
                        'context': chunk.get('context', ''),
                        'techniques': chunk.get('technique_mentions', []),
                        'page_numbers': chunk.get('page_numbers', []),
                        'quality_score': chunk.get('quality_score', 0),
                        'has_techniques': chunk.get('has_techniques', False),
                        'chinese_ratio': chunk.get('chinese_ratio', 0)
                    }
                }
                f.write(json.dumps(embedding_doc, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved RAG-optimized structure to {output_dir}")
        self.logger.info(f"Created {len(high_quality_chunks)} high-quality chunks for embedding")
        self.logger.info(f"Identified {len(technique_index)} unique techniques")
    
    def save_structure(self, doc_structure: DocumentStructure, output_dir: Path):
        """Save extracted structure to JSON and text files"""
        # Save as JSON
        json_path = output_dir / "structure.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(doc_structure), f, indent=2, ensure_ascii=False, default=str)
        
        # Save as structured text
        text_path = output_dir / "content.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {doc_structure.title}\n")
            f.write(f"Author: {doc_structure.author or 'Unknown'}\n")
            f.write(f"Format: {doc_structure.format}\n")
            f.write(f"Pages: {doc_structure.pages}\n")
            f.write("\n" + "="*50 + "\n\n")
            
            current_chapter = None
            for content in doc_structure.content:
                if content.content_type == "heading" and content.level <= 2:
                    current_chapter = content.text
                    f.write(f"\n{'#' * content.level} {content.text}\n\n")
                elif content.content_type == "heading":
                    f.write(f"\n{'#' * content.level} {content.text}\n\n")
                elif content.content_type == "paragraph":
                    f.write(f"{content.text}\n\n")
                elif content.content_type == "table":
                    f.write(f"TABLE:\n{content.text}\n\n")
                elif content.content_type == "caption":
                    f.write(f"CAPTION: {content.text}\n\n")
        
        # Save image manifest
        if doc_structure.images:
            images_manifest_path = output_dir / "images_manifest.json"
            with open(images_manifest_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(img) for img in doc_structure.images], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved structure to {output_dir}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Extract text and images from various document formats")
    parser.add_argument("input_path", help="Path to document file or directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-embedded-images", action="store_true", help="Disable computer vision-based image extraction from pages")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--create-config", type=str, help="Create example configuration file at specified path")
    
    # RAG optimization options
    parser.add_argument("--rag-mode", action="store_true", help="Enable RAG-optimized output with chunking and post-processing")
    
    # OCR enhancement options (for backward compatibility)
    parser.add_argument("--no-ocr-preprocessing", action="store_true", help="Disable OCR image preprocessing")
    parser.add_argument("--ocr-dpi", type=int, help="OCR processing DPI")
    parser.add_argument("--ocr-lang", help="OCR language")
    parser.add_argument("--ocr-psm", type=int, help="OCR page segmentation mode")
    parser.add_argument("--ocr-oem", type=int, help="OCR engine mode")
    
    args = parser.parse_args()
    
    # Handle config creation
    if args.create_config:
        config = DocumentExtractorConfig()
        config.create_example_config(Path(args.create_config))
        print(f"Created example configuration file at: {args.create_config}")
        return
    
    # Load configuration
    if args.config:
        config = load_config(Path(args.config))
    else:
        config = DocumentExtractorConfig()
    
    # Apply CLI overrides
    if args.output:
        config.output_dir = args.output
    if args.verbose:
        config.verbose_logging = True
        logging.getLogger().setLevel(logging.DEBUG)
    if args.no_embedded_images:
        config.extract_embedded_images = False
    if args.no_ocr_preprocessing:
        config.ocr.enable_preprocessing = False
    if args.ocr_dpi:
        config.ocr.dpi = args.ocr_dpi
    if args.ocr_lang:
        config.ocr.language = args.ocr_lang
    if args.ocr_psm:
        config.ocr.page_segmentation_mode = args.ocr_psm
    if args.ocr_oem:
        config.ocr.ocr_engine_mode = args.ocr_oem
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("Configuration validation errors:")
        for issue in issues:
            print(f"  - {issue}")
        return
    
    # Initialize extractor with configuration
    extractor = DocumentExtractor(config=config)
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Single file
        try:
            doc_structure = extractor.extract_document(input_path)
            output_dir = extractor.output_dir / input_path.stem
            
            if args.rag_mode:
                extractor.save_structure_for_rag(doc_structure, output_dir)
                print(f"Extracted {input_path.name} to {output_dir} (RAG-optimized)")
            else:
                extractor.save_structure(doc_structure, output_dir)
                print(f"Extracted {input_path.name} to {output_dir}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Directory of files
        supported_extensions = {'.pdf', '.epub', '.docx', '.djvu'}
        files_processed = 0
        
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    print(f"Processing {file_path.name}...")
                    doc_structure = extractor.extract_document(file_path)
                    output_dir = extractor.output_dir / file_path.stem
                    
                    if args.rag_mode:
                        extractor.save_structure_for_rag(doc_structure, output_dir)
                        print(f"  ✓ Extracted to {output_dir} (RAG-optimized)")
                    else:
                        extractor.save_structure(doc_structure, output_dir)
                        print(f"  ✓ Extracted to {output_dir}")
                    
                    files_processed += 1
                except Exception as e:
                    print(f"  ✗ Error processing {file_path.name}: {e}")
        
        print(f"\nProcessed {files_processed} files")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()