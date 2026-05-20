import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO

model = None
preprocess = None
device = "cpu"

CATEGORIES = [
    "fruits and vegetables",
    "dairy and eggs",
    "snacks and chips",
    "beverages and drinks",
    "bakery and bread",
    "grains and pulses",
    "personal care products",
    "household cleaning products",
    "baby products",
    "cooking oil and spices"
]

def load_model():
    global model, preprocess
    if model is None:
        print("⏳ Loading CLIP model...")
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✅ CLIP loaded")
    return model, preprocess

def classify_product(image_input) -> dict:
    import clip
    m, p = load_model()

    if isinstance(image_input, str) and image_input.startswith("http"):
        response = requests.get(image_input, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        image = Image.open(image_input).convert('RGB')

    image_tensor = p(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(
        [f"a photo of {cat}" for cat in CATEGORIES]
    ).to(device)

    with torch.no_grad():
        image_features = m.encode_image(image_tensor)
        text_features = m.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        scores = similarity[0].cpu().numpy()

    results = sorted(zip(CATEGORIES, scores.tolist()), key=lambda x: x[1], reverse=True)

    return {
        "top_category": results[0][0],
        "confidence": round(results[0][1], 3),
        "top_3": {cat: round(sc, 3) for cat, sc in results[:3]}
    }

if __name__ == "__main__":
    # Test with a plain white image
    test_img = Image.new('RGB', (224, 224), color=(200, 200, 200))
    result = classify_product(test_img)
    print("✅ Classifier result:", result)