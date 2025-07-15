#!/bin/bash

# Install system dependencies for document extraction
echo "Installing system dependencies..."

# Update package lists
sudo apt-get update

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Install system tools for DJVU processing
echo "Installing DJVU tools..."
sudo apt-get install -y djvulibre-bin

# Install Tesseract OCR
echo "Installing Tesseract OCR..."
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# Install additional language packs if needed
# sudo apt-get install -y tesseract-ocr-chi-sim tesseract-ocr-jpn tesseract-ocr-kor

echo "Dependencies installed successfully!"
echo "You can now run: python document_extractor.py [input_file_or_directory]"