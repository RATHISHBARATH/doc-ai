# ============================================================
# DOC AI Vision Service – OCR Engine Package
# ============================================================

"""
Optical Character Recognition (OCR) engine using Tesseract.

This module provides the TesseractOCR class, which wraps pytesseract
for extracting text from images.
"""

from .tesseract import TesseractOCR

__all__ = [
    "TesseractOCR",
]