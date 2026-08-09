# ============================================================
# DOC AI Vision Service – Tesseract OCR Engine
# ============================================================

import logging
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

from src.common.models import OCRResult

logger = logging.getLogger(__name__)


class TesseractOCR:
    """
    Optical Character Recognition (OCR) engine using Tesseract.

    This class wraps pytesseract and provides a simple `ocr` method
    that extracts text from an image and returns an `OCRResult` object.
    """

    def __init__(self, lang: str = "eng", config: Optional[str] = None):
        """
        Initialize the Tesseract OCR engine.

        Args:
            lang: Language(s) for OCR (default 'eng').
            config: Additional Tesseract configuration string (e.g., '--psm 6').
        """
        self.lang = lang
        self.config = config or "--psm 6"  # Assume a single uniform block of text
        self.logger = logging.getLogger(f"{__name__}.TesseractOCR")

    def ocr(self, image: np.ndarray) -> OCRResult:
        """
        Perform OCR on an image.

        Args:
            image: A numpy array representing the image (BGR format, OpenCV style).

        Returns:
            An OCRResult containing the extracted text, confidence, and bounding box.
            If no text is found, text will be an empty string and confidence will be None.
        """
        if image is None:
            self.logger.warning("Received None image, returning empty OCR result")
            return OCRResult(text="")

        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Perform OCR
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )

            # Extract text and confidence
            text_blocks = []
            confidences = []
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])
                if text and conf > 0:
                    text_blocks.append(text)
                    confidences.append(conf)

            extracted_text = " ".join(text_blocks)
            avg_confidence = sum(confidences) / len(confidences) if confidences else None

            # Get bounding box of the whole detected text region
            if len(data["text"]) > 0 and extracted_text:
                # Combine all non‑empty boxes into a single box
                xs = []
                ys = []
                ws = []
                hs = []
                for i in range(len(data["text"])):
                    if data["text"][i].strip():
                        xs.append(data["left"][i])
                        ys.append(data["top"][i])
                        ws.append(data["width"][i])
                        hs.append(data["height"][i])
                if xs and ys and ws and hs:
                    min_x = min(xs)
                    min_y = min(ys)
                    max_x = max(x + w for x, w in zip(xs, ws))
                    max_y = max(y + h for y, h in zip(ys, hs))
                    bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
                else:
                    bbox = None
            else:
                bbox = None

            result = OCRResult(
                text=extracted_text,
                confidence=avg_confidence,
                bounding_box=bbox,
            )

            self.logger.debug(f"OCR extracted text: '{extracted_text[:50]}...'")
            return result

        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            return OCRResult(text="")

    def set_language(self, lang: str) -> None:
        """Set the language(s) for OCR."""
        self.lang = lang

    def set_config(self, config: str) -> None:
        """Set the Tesseract configuration string."""
        self.config = config