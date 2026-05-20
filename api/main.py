import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from ml.drift import full_drift_report, all_categories_report
from ml.forecast import prophet_forecast, xgboost_forecast
from cv.ocr import extract_price
from cv.classifier import classify_product

app = FastAPI(title="PriceScope API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DB_PATH = "data/pricescope.db"

# ─── HEALTH CHECK ───────────────────────────────────────
@app.get("/")
def root():
    return {"status": "PriceScope API running", "version": "1.0.0"}

# ─── CATEGORIES ─────────────────────────────────────────
@app.get("/api/categories")
def get_categories():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT category, COUNT(*) as product_count
        FROM prices
        GROUP BY category
        ORDER BY product_count DESC
    """, conn)
    conn.close()
    return df.to_dict('records')

# ─── INFLATION INDEX (dashboard summary) ────────────────
@app.get("/api/inflation-index")
def get_inflation_index():
    reports = all_categories_report()
    return [
        {
            "category": r["category"],
            "inflation_pct": r["inflation_rate_pct"],
            "psi": r["psi"],
            "alert": r["alert_level"],
            "current_mean_price": r["current_mean_price"],
            "reference_mean_price": r["reference_mean_price"]
        }
        for r in reports
    ]

# ─── DRIFT REPORT FOR ONE CATEGORY ──────────────────────
@app.get("/api/drift/{category}")
def get_drift(category: str):
    return full_drift_report(category)

# ─── PRICE HISTORY FOR CHART ────────────────────────────
@app.get("/api/history/{category}")
def get_price_history(category: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT DATE(recorded_at) as date, AVG(price) as avg_price
        FROM prices
        WHERE category LIKE ?
        GROUP BY DATE(recorded_at)
        ORDER BY date
    """, conn, params=[f'%{category}%'])
    conn.close()
    return df.to_dict('records')

# ─── FORECAST ───────────────────────────────────────────
@app.get("/api/forecast/{category}")
def get_forecast(category: str):
    prophet = prophet_forecast(category)
    xgb = xgboost_forecast(category)
    return {
        "category": category,
        "prophet": prophet,
        "xgboost": xgb
    }

# ─── IMAGE ANALYSIS (CV endpoint) ───────────────────────
@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    img_array = np.array(image)

    price_result = extract_price(img_array)
    category_result = classify_product(image)

    return {
        "ocr": price_result,
        "classification": category_result,
        "summary": {
            "detected_price": price_result.get("best_price"),
            "product_category": category_result.get("top_category"),
            "confidence": category_result.get("confidence")
        }
    }

# ─── PRODUCTS IN CATEGORY ───────────────────────────────
@app.get("/api/products/{category}")
def get_products(category: str, limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT DISTINCT name, sub_category,
               AVG(price) as avg_price,
               MIN(price) as min_price,
               MAX(price) as max_price
        FROM prices
        WHERE category LIKE ?
        GROUP BY name
        ORDER BY avg_price DESC
        LIMIT ?
    """, conn, params=[f'%{category}%', limit])
    conn.close()
    return df.to_dict('records')