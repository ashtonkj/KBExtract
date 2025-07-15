# Document Content Extractor

A comprehensive Python tool for extracting text and images from various document formats while preserving structure. Optimized for Retrieval Augmented Generation (RAG) systems and knowledge base construction with advanced OCR post-processing for martial arts and multilingual content.

## Features

- **Multi-format support**: PDF, EPUB, DOCX, DJVU
- **RAG-optimized output**: Intelligent chunking and quality scoring for embedding systems
- **Conservative OCR post-processing**: Domain-aware text correction without aggressive spell checking
- **Martial arts content detection**: Specialized recognition of techniques, terminology, and multilingual content
- **Structure preservation**: Maintains headings, chapters, paragraphs, tables, and captions
- **Image extraction**: Extracts all images with metadata and proper naming
- **Computer vision image detection**: Advanced CV-based extraction of embedded images from scanned pages
- **OCR capability**: Full OCR support with confidence scoring and quality metrics
- **Batch processing**: Process single files or entire directories
- **Structured output**: JSON, human-readable text, and RAG-ready formats
- **Comprehensive metadata**: Author, title, page count, and format-specific information

## Installation

### System Dependencies

#### Ubuntu/Debian:
```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# Install DJVU tools (for DJVU support)
sudo apt install djvulibre-bin

# Install Tesseract OCR (for DJVU text extraction)
sudo apt install tesseract-ocr tesseract-ocr-eng

# Optional: Additional language packs
sudo apt install tesseract-ocr-chi-sim tesseract-ocr-jpn tesseract-ocr-kor
```

#### macOS:
```bash
# Using Homebrew
brew install djvulibre tesseract

# Using MacPorts
sudo port install djvulibre tesseract
```

### Python Dependencies

Install required Python packages:

```bash
# Using pip
pip install -r requirements.txt

# Or install individually
pip install PyPDF2 PyMuPDF pdfplumber ebooklib python-docx beautifulsoup4 pytesseract Pillow lxml opencv-python numpy
```

Or use the provided installation script:

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

## Usage

### Command Line Interface

#### Extract a single document:
```bash
python3 document_extractor.py "/path/to/document.pdf"
```

#### Extract all documents from a directory:
```bash
python3 document_extractor.py "/path/to/documents/directory"
```

#### Specify output directory:
```bash
python3 document_extractor.py "/path/to/document.pdf" -o "/path/to/output"
```

#### Enable verbose logging:
```bash
python3 document_extractor.py "/path/to/document.pdf" -v
```

#### RAG-optimized extraction (recommended for knowledge bases):
```bash
python3 document_extractor.py "/path/to/document.pdf" --rag-mode
```

#### Using configuration files:
```bash
# Create an example configuration file
python3 document_extractor.py --create-config my_config.json

# Use configuration file
python3 document_extractor.py "/path/to/document.pdf" --config my_config.json
```

#### Disable computer vision image extraction:
```bash
python3 document_extractor.py "/path/to/document.pdf" --no-embedded-images
```

#### Enhanced OCR (with improved defaults):
```bash
# Default: 400 DPI, English + Chinese (Traditional & Simplified), PSM 3
python3 document_extractor.py "/path/to/document.djvu"

# English only for faster processing
python3 document_extractor.py "/path/to/document.djvu" --ocr-lang eng

# Maximum quality OCR with RAG optimization
python3 document_extractor.py "/path/to/document.djvu" --ocr-dpi 600 --rag-mode
```

#### Standalone image extraction (less sensitive defaults):
```bash
# Default: 75000 area, 250x250 min size, 4:1 max aspect ratio
python3 image_extractor.py "extracted_documents/Document_Name/images"

# More sensitive detection (smaller images)
python3 image_extractor.py "extracted_documents/Document_Name/images" \
    --min-area 50000 --min-width 200 --min-height 200 -o "reextracted_images"
```

### Python API

```python
from document_extractor import DocumentExtractor

# Initialize extractor with improved defaults
# (400 DPI, English + Chinese, less sensitive image extraction)
extractor = DocumentExtractor(output_dir="my_extracts")

# English only for faster processing
extractor = DocumentExtractor(output_dir="my_extracts", ocr_lang="eng")

# Maximum quality OCR
extractor = DocumentExtractor(
    output_dir="my_extracts",
    ocr_dpi=600,
    ocr_lang="eng+chi_sim+chi_tra"
)

# Extract a single document
doc_structure = extractor.extract_document("/path/to/document.pdf")

# Save results (standard format)
extractor.save_structure(doc_structure, Path("output/document_name"))

# Save with RAG optimization (recommended for knowledge bases)
extractor.save_structure_for_rag(doc_structure, Path("output/document_name"))
```

## Configuration System

The document extractor now uses a centralized configuration system that eliminates magic numbers and provides clear documentation for all parameters.

### Creating Configuration Files

Generate an example configuration file with all available options:

```bash
python3 document_extractor.py --create-config my_config.json
```

This creates a JSON file with all configuration options and their default values:

```json
{
  "_comment": "Document Extractor Configuration",
  "output_dir": "extracted_documents",
  "extract_embedded_images": true,
  "verbose_logging": false,
  
  "ocr": {
    "_comment": "OCR processing settings",
    "dpi": 400,
    "language": "eng+chi_sim+chi_tra",
    "page_segmentation_mode": 3,
    "low_confidence_threshold": 60
  },
  
  "image_detection": {
    "_comment": "Computer vision image detection settings",
    "min_area": 75000,
    "min_width": 250,
    "min_height": 250,
    "canny_low_threshold": 100,
    "canny_high_threshold": 200
  },
  
  "rag": {
    "_comment": "RAG system optimization settings",
    "target_chunk_size": 512,
    "min_quality_score": 0.6,
    "min_confidence_score": 0.7
  }
}
```

### Using Configuration Files

```bash
# Use configuration file
python3 document_extractor.py document.pdf --config my_config.json

# Override specific values via CLI
python3 document_extractor.py document.pdf --config my_config.json --ocr-dpi 600
```

### Configuration Categories

#### OCR Settings (`ocr`)
- **dpi**: OCR processing resolution (72-1200)
- **language**: Tesseract language codes
- **page_segmentation_mode**: Text layout analysis mode (0-13)
- **low_confidence_threshold**: Words below this flagged for review (0-100)
- **enable_preprocessing**: Apply image enhancement before OCR

#### Image Detection (`image_detection`)
- **min_area**: Minimum pixel area for detected images
- **min_width/min_height**: Minimum dimensions in pixels
- **canny_low/high_threshold**: Edge detection sensitivity
- **variance_threshold**: Minimum pixel variance for images
- **edge_density_min/max**: Edge density range for image classification

#### RAG Optimization (`rag`)
- **target_chunk_size**: Target tokens per chunk (for embeddings)
- **min_quality_score**: Minimum quality for embedding chunks (0-1)
- **min_confidence_score**: Minimum OCR confidence for high-quality chunks (0-1)
- **max_low_confidence_regions**: Max uncertain words before flagging for review

#### Martial Arts Detection (`martial_arts`)
- **technique_relevance_weight**: Weight for technique mentions in scoring
- **chinese_relevance_weight**: Weight for Chinese character ratio
- **technique_patterns**: Regex patterns for technique detection

#### Text Processing (`text_processing`)
- **min_clean_char_ratio**: Minimum ratio of valid characters (0-1)
- **optimal_text_length**: Text length that gets quality bonus
- **max_char_repetition**: Maximum allowed character repetition

### Python API with Configuration

```python
from extractor_config import DocumentExtractorConfig

# Load from file
config = DocumentExtractorConfig.from_file("my_config.json")

# Create custom configuration
config = DocumentExtractorConfig()
config.ocr.dpi = 600
config.image_detection.min_area = 50000
config.rag.target_chunk_size = 256

# Use with extractor
extractor = DocumentExtractor(config=config)
```

### Validation

The configuration system includes validation to catch common errors:

```bash
# If you set invalid values, you'll get helpful error messages
# Configuration validation errors:
#   - OCR DPI should be between 72 and 1200
#   - Minimum aspect ratio should be less than maximum aspect ratio
#   - Quality score should be between 0 and 1
```

## Output Structure

Each processed document creates a subdirectory with the following structure:

### Standard Output Structure

```
document_name/
├── structure.json          # Complete structured data
├── content.txt            # Human-readable text content
├── images/               # Extracted images directory
│   ├── page_001.png
│   ├── page_002.png
│   └── ...
└── images_manifest.json   # Image metadata
```

### RAG-Optimized Output Structure (--rag-mode)

```
document_name/
├── structure.json          # Complete structured data
├── content.txt            # Human-readable text content
├── rag_ready.json         # RAG-optimized content with chunking
├── embeddings_chunks.jsonl # High-quality chunks for embedding
├── images/               # Extracted images directory
│   ├── page_001.png
│   ├── page_002.png
│   └── ...
└── images_manifest.json   # Image metadata
```

### JSON Structure

The `structure.json` file contains:

```json
{
  "title": "Document Title",
  "author": "Author Name",
  "format": "PDF",
  "pages": 150,
  "chapters": ["Chapter 1", "Chapter 2", ...],
  "content": [
    {
      "text": "Content text",
      "content_type": "heading|paragraph|table|caption",
      "level": 1,
      "page_number": 1,
      "chapter": "Chapter Name",
      "confidence_score": 0.95,
      "quality_score": 0.87,
      "technique_mentions": ["White Crane Spreads Wings", "Single Whip"],
      "low_confidence_regions": ["unclear_word1", "unclear_word2"],
      "metadata": {...}
    }
  ],
  "images": [
    {
      "filename": "page_001.png",
      "format": "PNG",
      "width": 800,
      "height": 600,
      "page_number": 1,
      "caption": "Image caption",
      "image_type": "illustration",
      "technique_demonstrations": ["Tiger Claw stance"],
      "relevance_score": 0.85,
      "metadata": {...}
    }
  ],
  "metadata": {...}
}
```

### RAG-Ready JSON Structure

The `rag_ready.json` file contains optimized content for RAG systems:

```json
{
  "document_metadata": {
    "title": "Document Title",
    "total_chunks": 45,
    "high_quality_chunks": 38,
    "total_techniques": 127,
    "relevant_images": 23
  },
  "rag_chunks": [
    {
      "chunk_id": "chunk_0001",
      "text": "Semantic chunk text optimized for embedding...",
      "context": "Chapter 3: Advanced Techniques",
      "technique_mentions": ["Iron Palm", "Golden Bell"],
      "confidence_score": 0.89,
      "quality_score": 0.92,
      "has_techniques": true,
      "chinese_ratio": 0.15,
      "page_numbers": [12, 13]
    }
  ],
  "processed_images": [
    {
      "filename": "page_012_img_001.png",
      "image_type": "illustration",
      "ocr_text": "Figure 3.1: Iron Palm Training",
      "technique_demonstrations": ["Iron Palm"],
      "relevance_score": 0.91
    }
  ],
  "technique_index": {
    "Iron Palm": [
      {"chunk_id": "chunk_0001", "context": "Training Methods"},
      {"chunk_id": "chunk_0023", "context": "Applications"}
    ]
  },
  "low_confidence_content": [
    {
      "chunk_id": "chunk_0015",
      "confidence_score": 0.63,
      "low_confidence_regions": ["unclear_term1", "unclear_term2"]
    }
  ]
}
```

## RAG System Optimization

### Conservative OCR Post-Processing

The extractor includes domain-aware OCR post-processing designed for martial arts and multilingual content:

#### Safe Character-Level Corrections
- **Visual OCR errors**: Fixes common misrecognitions like `rn` → `m`, `cl` → `d`, `li` → `h`
- **Character repetition**: Reduces excessive character duplication from OCR artifacts
- **Whitespace normalization**: Cleans up irregular spacing without affecting content
- **Garbage character removal**: Removes OCR artifacts while preserving Chinese characters

#### Confidence-Based Quality Assessment
- **Per-word confidence scoring**: Uses Tesseract confidence data to identify uncertain regions
- **Low-confidence flagging**: Marks words with <60% confidence for manual review
- **Quality metrics**: Calculates overall text quality based on multiple factors

#### Domain-Aware Processing
- **Martial arts terminology preservation**: Avoids "correcting" valid technique names and terminology
- **Multilingual content support**: Handles mixed English/Chinese text appropriately
- **Technical vocabulary protection**: Preserves specialized terms, proper nouns, and transliterations

### Intelligent Chunking for RAG

#### Semantic Boundary Detection
- **Technique preservation**: Keeps complete technique descriptions together
- **Context-aware splitting**: Maintains semantic coherence across chunk boundaries
- **Natural breakpoints**: Splits at headings, tables, and content type changes
- **Size optimization**: Targets 512-token chunks for optimal embedding performance

#### Quality-Based Filtering
- **High-quality chunks**: Filters content with quality scores >0.6 for embedding
- **Confidence thresholds**: Excludes low-confidence OCR content from primary embeddings
- **Content type weighting**: Prioritizes headings and technique descriptions
- **Review flagging**: Identifies content needing manual verification

### Martial Arts Content Detection

#### Pattern Recognition
- **Technique names**: Detects patterns like "White Crane Spreads Wings", "Iron Palm"
- **Terminology identification**: Recognizes martial arts vocabulary and concepts
- **Multilingual support**: Handles Chinese terms, transliterations, and English descriptions
- **Context analysis**: Uses surrounding text to improve detection accuracy

#### Technique Indexing
- **Cross-referencing**: Maps techniques to multiple mentions across the document
- **Context preservation**: Associates techniques with their instructional context
- **Hierarchical organization**: Links techniques to chapters and sections

### Image Processing for RAG

#### Multimodal Content Support
- **Image classification**: Categorizes images as photos, illustrations, diagrams, or icons
- **OCR text extraction**: Extracts text from technical diagrams and illustrations
- **Technique demonstration detection**: Identifies images showing martial arts techniques
- **Relevance scoring**: Ranks images by importance for martial arts knowledge

#### RAG Integration Ready
- **Embedding preparation**: Structured data ready for multimodal embedding models
- **Caption enhancement**: Combines original captions with OCR text and technique detection
- **Cross-reference linking**: Connects images to related text chunks

### Usage for Knowledge Base Construction

#### Command Line
```bash
# Extract with full RAG optimization
python3 document_extractor.py "Kung_Fu_Manual.pdf" --rag-mode

# High-quality OCR with RAG chunking
python3 document_extractor.py "Martial_Arts_Book.djvu" --ocr-dpi 600 --rag-mode
```

#### Python API
```python
# Extract and process for knowledge base
extractor = DocumentExtractor(output_dir="knowledge_base")
doc_structure = extractor.extract_document("martial_arts_text.pdf")
extractor.save_structure_for_rag(doc_structure, output_dir)

# Access RAG-ready data
import json
with open("output_dir/rag_ready.json") as f:
    rag_data = json.load(f)

# High-quality chunks for embedding
chunks = rag_data['rag_chunks']
techniques = rag_data['technique_index']
relevant_images = [img for img in rag_data['processed_images'] 
                   if img['relevance_score'] > 0.7]
```

## Format Support

### PDF Files (.pdf)
- **Text extraction**: Using PyMuPDF and PyPDF2
- **Image extraction**: All embedded images with metadata
- **Computer vision image detection**: Automatic detection and extraction of images from scanned pages
- **Table extraction**: Using pdfplumber for enhanced table detection
- **Structure detection**: Automatic heading and paragraph classification
- **Font analysis**: Content type classification based on font size and style
- **Scanned page detection**: Automatic OCR application for image-based pages

### EPUB Files (.epub)
- **Chapter extraction**: Automatic chapter detection from navigation
- **HTML processing**: Clean text extraction from HTML content
- **Image extraction**: All embedded images (PNG, JPG, SVG)
- **Metadata extraction**: Title, author, publisher information
- **Structure preservation**: Headings, paragraphs, and content hierarchy

### DOCX Files (.docx)
- **Text extraction**: Full document text with formatting
- **Table extraction**: Complete table structure preservation
- **Image extraction**: All embedded images and media
- **Style analysis**: Heading detection based on document styles
- **Metadata extraction**: Document properties and creation info

### DJVU Files (.djvu)
- **OCR text extraction**: Full text via Tesseract OCR
- **Page conversion**: Convert all pages to PNG images
- **Computer vision image detection**: Automatic extraction of images embedded within pages
- **Batch processing**: Efficient handling of multi-page documents
- **Error handling**: Graceful handling of OCR failures

## Content Types

The extractor classifies content into the following types:

- **heading**: Document headings (levels 1-6)
- **paragraph**: Regular text paragraphs
- **table**: Structured table data
- **caption**: Image and figure captions
- **error**: Error messages and warnings

## Configuration

### Computer Vision Image Detection

The extractor now includes advanced computer vision capabilities to detect and extract images from scanned pages. This feature uses OpenCV to:

- **Detect rectangular regions**: Identify potential image boundaries using edge detection
- **Filter text regions**: Distinguish between images and text using statistical analysis
- **Extract clean images**: Save detected images as separate PNG files
- **Preserve metadata**: Include extraction method, bounding boxes, and quality metrics

#### Tuning Parameters

For overly aggressive extraction, you can:

1. **Disable the feature entirely**:
   ```bash
   python3 document_extractor.py document.djvu --no-embedded-images
   ```

2. **Adjust sensitivity in code** by modifying these parameters in `_detect_and_extract_images_from_page()`:
   - `min_area`: Minimum pixel area for detected images (default: 75000)
   - `Canny edge detection`: Edge detection thresholds (default: 100, 200)
   - `Size filters`: Minimum width/height (default: 250px)

3. **Customize detection logic** in `_is_likely_image_region()`:
   - `variance`: Pixel variance threshold for image detection
   - `edge_density`: Edge density ranges for filtering
   - `entropy`: Histogram entropy thresholds

### OCR Quality Improvements

The extractor includes several OCR enhancements for better text recognition:

#### Automatic Image Preprocessing
- **Gaussian blur**: Reduces noise in scanned images
- **Adaptive thresholding**: Improves text contrast for varying lighting
- **Morphological operations**: Cleans up text and connects broken characters
- **DPI scaling**: Optimizes image resolution for OCR processing

#### OCR Configuration Options
- **Language support**: Multi-language OCR with language packs
- **Page segmentation modes**: Different strategies for text layout analysis
- **OCR engine modes**: Choose between legacy and neural network engines
- **DPI optimization**: Configurable resolution for quality vs speed trade-offs

#### Usage Examples
```bash
# Default: High-quality OCR with 400 DPI, English + Chinese
python3 document_extractor.py document.djvu

# English only for faster processing
python3 document_extractor.py document.djvu --ocr-lang eng

# Maximum quality OCR
python3 document_extractor.py document.djvu --ocr-dpi 600

# Different page segmentation mode for complex layouts
python3 document_extractor.py document.djvu --ocr-psm 1

# Disable preprocessing if it's causing issues
python3 document_extractor.py document.djvu --no-ocr-preprocessing
```

### OCR Languages

Configure Tesseract for different languages:

```bash
# Install additional language packs
sudo apt install tesseract-ocr-fra tesseract-ocr-deu tesseract-ocr-spa
sudo apt install tesseract-ocr-chi-sim tesseract-ocr-chi-tra tesseract-ocr-jpn tesseract-ocr-kor

# Use multiple languages
python3 document_extractor.py document.djvu --ocr-lang fra+eng+deu
```

### Standalone Image Extraction

For situations where you want to re-process image extraction with different parameters:

```bash
# Re-extract images with improved defaults (less sensitive)
python3 image_extractor.py "extracted_documents/Document_Name/images"

# More sensitive detection (smaller images)
python3 image_extractor.py "extracted_documents/Document_Name/images" \
    --min-area 50000 \
    --min-width 200 \
    --min-height 200 \
    --max-aspect-ratio 5.0 \
    -o "reextracted_images"
```

The standalone tool allows you to:
- **Adjust detection sensitivity** without re-running OCR
- **Fine-tune parameters** for specific document types
- **Experiment with settings** to find optimal extraction
- **Process existing extractions** with new algorithms

### Custom Output Formats

Extend the `save_structure` method to create custom output formats:

```python
def custom_save_format(doc_structure, output_dir):
    # Custom processing logic
    pass
```

## Error Handling

The extractor includes comprehensive error handling:

- **Missing dependencies**: Clear error messages with installation instructions
- **Corrupted files**: Graceful handling of damaged documents
- **OCR failures**: Fallback to image-only extraction
- **Format detection**: Automatic format detection with validation

## Performance Considerations

- **Large documents**: Processing time scales with document size
- **DJVU files**: OCR processing is CPU-intensive
- **Batch processing**: Memory usage increases with concurrent extractions
- **Image extraction**: Disk space requirements for image-heavy documents

## Troubleshooting

### Common Issues

1. **"tesseract is not installed"**
   ```bash
   sudo apt install tesseract-ocr
   ```

2. **"djvused: command not found"**
   ```bash
   sudo apt install djvulibre-bin
   ```

3. **"Cannot import name 'CT_Table'"**
   - Update python-docx: `pip install --upgrade python-docx`

4. **Memory errors with large files**
   - Process files individually rather than in batches
   - Increase system memory or use swap space

### Debugging

Enable verbose logging for detailed processing information:

```bash
python3 document_extractor.py document.pdf -v
```

## Dependencies

### Required Python Packages

- **PyPDF2**: PDF text extraction
- **PyMuPDF (fitz)**: Advanced PDF processing
- **pdfplumber**: PDF table extraction
- **ebooklib**: EPUB processing
- **python-docx**: DOCX document handling
- **beautifulsoup4**: HTML parsing
- **pytesseract**: OCR text extraction
- **Pillow**: Image processing
- **lxml**: XML processing
- **opencv-python**: Computer vision for image detection
- **numpy**: Numerical computing for image processing

### System Requirements

- **Python 3.8+**
- **Linux/macOS/Windows**
- **2GB+ RAM** (recommended for large documents)
- **djvulibre-bin** (for DJVU support)
- **tesseract-ocr** (for OCR functionality)

## License

This project is open source. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

## Examples

### Extract a martial arts manual:
```bash
python3 document_extractor.py "Shaolin Kung Fu Manual.pdf"
```

### Process a directory of academic papers:
```bash
python3 document_extractor.py "/home/user/papers" -o "/home/user/extracted_papers"
```

### Extract with custom settings:
```python
extractor = DocumentExtractor(output_dir="custom_output")
doc = extractor.extract_document("document.epub")
print(f"Extracted {len(doc.content)} content items")
print(f"Found {len(doc.images)} images")
```

## Version History

- **v1.0.0**: Initial release with PDF, EPUB, DOCX, DJVU support
- **v1.1.0**: Added table extraction and improved structure detection
- **v1.2.0**: Enhanced OCR capabilities and error handling
- **v1.3.0**: Added computer vision-based image detection and extraction from scanned pages
- **v2.0.0**: **RAG System Optimization Release**
  - Conservative OCR post-processing for domain-specific content
  - Martial arts terminology detection and preservation
  - Intelligent chunking for embedding systems
  - Confidence scoring and quality assessment
  - RAG-ready output formats with technique indexing
  - Multimodal image processing with relevance scoring
  - Knowledge base construction optimization