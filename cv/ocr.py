import easyocr
import re
import numpy as np
from PIL import Image

reader = None

def get_reader():
    global reader
    if reader is None:
        print("⏳ Loading EasyOCR (first time is slow ~30s)...")
        reader = easyocr.Reader(['en'], gpu=False)
        print("✅ EasyOCR loaded")
    return reader

def extract_price(image: np.ndarray) -> dict:
    r = get_reader()
    results = r.readtext(image)
    all_text = " ".join([text for (_, text, conf) in results if conf > 0.4])

    price_patterns = [
        r'₹\s*(\d+(?:\.\d{1,2})?)',
        r'Rs\.?\s*(\d+(?:\.\d{1,2})?)',
        r'MRP\s*:?\s*(\d+(?:\.\d{1,2})?)',
        r'\b(\d{1,5}\.\d{2})\b',
        r'\b(\d{2,5})/-',
    ]

    found_prices = []
    for pattern in price_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        found_prices.extend([float(m) for m in matches])

    valid = [p for p in found_prices if 1 < p < 100000]

    return {
        "raw_text": all_text,
        "prices_found": valid,
        "best_price": min(valid) if valid else None,
        "confidence": "high" if valid else "none"
    }

if __name__ == "__main__":
    # Test with a dummy white image
    test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    result = extract_price(test_img)
    print("✅ OCR test result:", result)
    print("✅ cv/ocr.py working")