# Usage Examples

## OCR Quality Improvements

### Basic Enhanced OCR (Improved Defaults)
```bash
# Uses enhanced OCR with preprocessing, 400 DPI, English + Chinese (Traditional & Simplified), PSM 3
python3 document_extractor.py document.djvu
```

### High-Quality OCR for Scanned Documents
```bash
# Higher DPI for better quality, disable preprocessing if it's causing issues
python3 document_extractor.py document.djvu --ocr-dpi 600 --ocr-lang eng
```

### Multi-language OCR
```bash
# Chinese and English OCR
python3 document_extractor.py document.djvu --ocr-lang chi_sim+eng

# Japanese and English OCR
python3 document_extractor.py document.djvu --ocr-lang jpn+eng
```

### OCR Mode Optimization
```bash
# PSM 6: Uniform block of text (default)
python3 document_extractor.py document.djvu --ocr-psm 6

# PSM 3: Fully automatic page segmentation, no OSD
python3 document_extractor.py document.djvu --ocr-psm 3

# PSM 1: Automatic page segmentation with OSD
python3 document_extractor.py document.djvu --ocr-psm 1
```

### Disable OCR Preprocessing
```bash
# If preprocessing is causing issues with your specific documents
python3 document_extractor.py document.djvu --no-ocr-preprocessing
```

## Standalone Image Extraction

### Re-extract Images with Different Parameters
```bash
# Extract images from already processed page images
python3 image_extractor.py "extracted_documents/Document_Name/images" -o "reextracted_images"
```

### Default Less Aggressive Image Extraction
```bash
# Default: 75000 area, 250x250 min size, 4:1 max aspect ratio (less sensitive)
python3 image_extractor.py "extracted_documents/Document_Name/images"

# Even less aggressive
python3 image_extractor.py "extracted_documents/Document_Name/images" \
    --min-area 100000 \
    --min-width 300 \
    --min-height 300 \
    --max-aspect-ratio 3.0
```

### More Sensitive Image Detection
```bash
# Lower thresholds for detecting smaller images
python3 image_extractor.py "extracted_documents/Document_Name/images" \
    --min-area 20000 \
    --min-width 150 \
    --min-height 150 \
    --canny-low 50 \
    --canny-high 150
```

### Fine-tune for Text vs Image Detection
```bash
# Adjust variance and entropy thresholds
python3 image_extractor.py "extracted_documents/Document_Name/images" \
    --variance-threshold 500 \
    --entropy-threshold 5.0 \
    --edge-density-min 0.03 \
    --edge-density-max 0.4
```

## Combined Workflows

### High-Quality Processing with Custom Image Extraction
```bash
# Step 1: Extract with enhanced OCR but no image extraction
python3 document_extractor.py document.djvu \
    --ocr-dpi 600 \
    --ocr-lang eng \
    --no-embedded-images \
    -o "first_pass"

# Step 2: Custom image extraction with fine-tuned parameters
python3 image_extractor.py "first_pass/Document_Name/images" \
    --min-area 75000 \
    --min-width 250 \
    --min-height 250 \
    --max-aspect-ratio 4.0 \
    -o "first_pass/Document_Name/extracted_images"
```

### Batch Processing with Different OCR Settings
```bash
# Process a directory of documents with optimized OCR settings
python3 document_extractor.py "/path/to/documents" \
    --ocr-dpi 400 \
    --ocr-lang eng \
    --ocr-psm 3 \
    --no-embedded-images \
    -o "batch_processed"
```

## OCR Quality Tips

### Language Codes
- `eng` - English
- `chi_sim` - Chinese Simplified
- `chi_tra` - Chinese Traditional
- `jpn` - Japanese
- `kor` - Korean
- `fra` - French
- `deu` - German
- `spa` - Spanish
- `rus` - Russian
- `ara` - Arabic

### Page Segmentation Modes (PSM)
- `0` - Orientation and script detection (OSD) only
- `1` - Automatic page segmentation with OSD
- `2` - Automatic page segmentation, no OSD
- `3` - Fully automatic page segmentation, no OSD (default)
- `4` - Assume a single column of text of variable sizes
- `5` - Assume a single uniform block of vertically aligned text
- `6` - Assume a single uniform block of text (default for this tool)
- `7` - Treat the image as a single text line
- `8` - Treat the image as a single word
- `9` - Treat the image as a single word in a circle
- `10` - Treat the image as a single character
- `11` - Sparse text. Find as much text as possible in no particular order
- `12` - Sparse text with OSD
- `13` - Raw line. Treat the image as a single text line, bypassing hacks

### OCR Engine Modes (OEM)
- `0` - Legacy engine only
- `1` - Neural nets LSTM engine only
- `2` - Legacy + LSTM engines
- `3` - Default, based on what is available (recommended)

### DPI Recommendations
- `150-200` - Fast processing, lower quality
- `300` - Good balance (default)
- `400-600` - High quality, slower processing
- `600+` - Maximum quality, very slow

## Image Extraction Parameters

### Size Filtering
- `--min-area` - Minimum pixel area (default: 75000)
- `--min-width` - Minimum width in pixels (default: 250)
- `--min-height` - Minimum height in pixels (default: 250)
- `--max-aspect-ratio` - Maximum width/height ratio (default: 4.0)
- `--min-aspect-ratio` - Minimum width/height ratio (default: 0.25)

### Edge Detection
- `--canny-low` - Lower threshold for edge detection (default: 100)
- `--canny-high` - Upper threshold for edge detection (default: 200)

### Image vs Text Classification
- `--variance-threshold` - Minimum pixel variance for images (default: 1000)
- `--edge-density-min` - Minimum edge density for images (default: 0.05)
- `--edge-density-max` - Maximum edge density for images (default: 0.3)
- `--entropy-threshold` - Minimum histogram entropy for images (default: 6.0)
- `--horizontal-line-threshold` - Max horizontal line density for text detection (default: 0.1)

## Troubleshooting

### OCR Quality Issues
1. **Blurry text**: Increase `--ocr-dpi` to 400-600
2. **Mixed languages**: Use `--ocr-lang lang1+lang2+lang3`
3. **Complex layouts**: Try `--ocr-psm 1` or `--ocr-psm 3`
4. **Preprocessing artifacts**: Use `--no-ocr-preprocessing`

### Image Extraction Issues
1. **Too many false positives**: Increase `--min-area`, `--min-width`, `--min-height`
2. **Missing images**: Decrease thresholds or use `--canny-low 50 --canny-high 150`
3. **Fragmented images**: Increase `--min-area` to 75000 or 100000
4. **Text being extracted as images**: Increase `--variance-threshold` and `--entropy-threshold`