# PriceScope ⬡
### Retail Intelligence & Dynamic Price Forecasting Platform

```
 ____  ____  _  ___  ____  ____   __  __  ____  ____ 
(  _ \(  _ \( )/ __)(  __)/ ___) / _\(  )(  _ \(  __)
 ) __/ )   / )(( (__  ) _) \___ \/    \)(  ) __/ ) _) 
(__)  (__\_)(__)\___)(____)(____/\_/\_(__)(__)  (____)
                                              v 2.0.0
```

> **PriceScope** bridges physical retail intelligence with algorithmic market analysis. Drop a shelf image in — it reads the price. Point it at a category — it predicts where prices go next. Run it overnight — it alerts you before your competitors react.

---

## What It Actually Does

Most price monitoring tools give you a spreadsheet. PriceScope gives you a command centre.

It ingests BigBasket's 28,000-point product catalogue, simulates multi-month price history with realistic market drift, then runs three parallel intelligence layers on top: a statistical drift engine that catches price anomalies before they become problems, a hybrid ML forecaster that blends Meta's Prophet with XGBoost to predict 28-day price trajectories, and a computer vision pipeline that can extract prices directly from shelf photographs using EasyOCR and classify products with CLIP.

All of it surfaces in a dark terminal dashboard — not a generic admin panel, but something that looks like it belongs in a Bloomberg trading room.

---

## Architecture

```
pricescope/
├── data/
│   ├── raw/                    # BigBasket Products CSV — 28,000 SKUs
│   └── pricescope.db           # SQLite — 13,500+ time-series price records
│
├── pipeline/
│   ├── ingest.py               # ETL — CSV → simulated history → DB
│   └── scheduler.py            # APScheduler daemon — drift/forecast/health jobs
│
├── cv/
│   ├── ocr.py                  # EasyOCR + regex price extractor
│   └── classifier.py           # CLIP zero-shot product classifier
│
├── ml/
│   ├── drift.py                # PSI · KL Divergence · CUSUM drift analytics
│   └── forecast.py             # Prophet + XGBoost ensemble forecaster
│
├── api/
│   └── main.py                 # FastAPI — 14 endpoints, async, CORS-ready
│
├── frontend/
│   └── index.html              # Dark terminal dashboard — 5-tab SPA, zero dependencies
│
├── notebooks/
│   └── amazon-sales-dataset-eda.ipynb
│
├── .env                        # Secrets isolated from source
└── requirements.txt
```

---

## The 9 Engineering Pillars

### 1 · Automated Data Ingestion `pipeline/ingest.py`
Pulls BigBasket's catalogue, normalises columns, and simulates 90 days of realistic price history per product. Drift rates are category-aware — Fruits & Vegetables drift faster than Beverages. The result is a 13,500+ record SQLite time-series ready for immediate analysis.

### 2 · Chrono-Scheduling Engine `pipeline/scheduler.py`
APScheduler daemon with three independent cron jobs: drift scan every 6 hours, forecast refresh daily at 02:00 IST, and a DB health check every 30 minutes. Misfire grace periods prevent backlog pile-ups on restart. All events are logged with timestamps and alert levels.

### 3 · Computer Vision OCR Parser `cv/ocr.py`
Takes raw shelf imagery through five regex price patterns — ₹ prefix, Rs. prefix, MRP label, decimal detection, and slash notation. EasyOCR handles the text extraction; confidence thresholding at 0.4 filters out noise. Returns the `best_price` (lowest plausible detected price) alongside every candidate found.

### 4 · Deep Neural Product Classifier `cv/classifier.py`
CLIP ViT-B/32 running zero-shot classification across 10 grocery genres. No fine-tuning required — CLIP's visual-language alignment handles unseen products naturally. Returns ranked confidence scores for each category so the API consumer can make threshold decisions.

### 5 · MLOps Drift Analytics `ml/drift.py`
Three-layer detection stack that catches different drift patterns:

**Population Stability Index (PSI)** monitors structural population changes:

$$PSI = \sum_{i=1}^{k} \left(\text{Actual}_i\% - \text{Expected}_i\%\right) \times \ln\left(\frac{\text{Actual}_i\%}{\text{Expected}_i\%}\right)$$

- `PSI < 0.10` — Stable. No action needed.
- `PSI 0.10–0.20` — Moderate drift. Monitor closely.
- `PSI > 0.20` — Significant drift. Trigger alert.

**KL Divergence** quantifies information loss between distributions:

$$D_{KL}(P \| Q) = \sum_{x \in X} P(x) \ln\left(\frac{P(x)}{Q(x)}\right)$$

**CUSUM** catches slow, gradual drift that PSI misses — particularly useful for seasonal creep where prices shift 0.3% per week over months.

### 6 · Hybrid Forecasting Engine `ml/forecast.py`
Two models run in parallel and their outputs are served together:

**Prophet** isolates macro seasonality and weekly trend-lines. Configured with `changepoint_prior_scale=0.05` for conservative predictions that don't overfit recent noise.

**XGBoost** captures non-linear micro-variations using five lag features (1, 3, 7, 14, 21 days), 7-day rolling mean, 14-day rolling std, day-of-week, and week-of-year. The ensemble approach means the API consumer can choose the model most appropriate for their use case — Prophet for macro strategy, XGBoost for short-term tactical pricing.

### 7 · Async REST API Backend `api/main.py`
FastAPI on Uvicorn with 14 production endpoints. New in v2.0: `/api/summary` (platform KPI snapshot), `/api/movers` (highest-volatility products), `/api/subcategories/{category}` (price band breakdown), `/api/price-bands/{category}` (histogram distribution with percentile stats), `/api/search` (full-text product search), `/api/export/{category}` (JSON data export), and `/api/volatility` (rolling coefficient of variation per category).

### 8 · Terminal Intelligence Dashboard `frontend/index.html`
A zero-dependency single-file SPA that looks nothing like a generic admin panel. Five tabs: Overview (KPI cards, market heatmap, live drift alerts, market movers, category comparison bars), Intelligence (PSI rankings, CUSUM analysis, 90-day trend charts per category), Forecast (Prophet + XGBoost chart with confidence bands, model accuracy stats), CV Scanner (drag-and-drop image analysis with scan history), and Analytics (watchlist, sub-category price band visualiser, full product data table with search). Connects to the live API with graceful fallback to rich mock data when the backend isn't running.

Design language: dark terminal × financial data terminal. Scanline texture overlay, monospaced data fonts (`DM Mono`, `Space Mono`), display headings in `Syne`, 48px grid background, animated ticker tape with live category prices.

### 9 · Environment & Dependency Isolation `.env` · `requirements.txt`
`.env` keeps database paths, API keys, and cloud credentials out of source. `requirements.txt` pins specific versions — no `>=` ranges that quietly break at 3am.

---

## Quick Start

### Prerequisites
- Python 3.11 or 3.12
- Node.js not required — the frontend is a single static HTML file

### 1. Clone & install

```bash
git clone https://github.com/yourhandle/pricescope.git
cd pricescope
pip install -r requirements.txt
```

### 2. Build the database

```bash
python pipeline/ingest.py
# ✅ DB initialized
# ✅ Loaded N clean products
# ✅ Generated 13,500+ price records
# ✅ TOTAL RECORDS IN DB: 13,500
```

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
# INFO: PriceScope API v2.0 running on http://127.0.0.1:8000
```

### 4. Open the dashboard

```bash
# Option A — open directly in browser
open frontend/index.html

# Option B — serve via the API
open http://127.0.0.1:8000/dashboard
```

### 5. (Optional) Start the scheduler

```bash
python pipeline/scheduler.py
# Runs drift scans, forecast refreshes, and DB health checks on schedule
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + endpoint list |
| GET | `/api/summary` | Platform KPI snapshot |
| GET | `/api/categories` | All categories with product counts |
| GET | `/api/inflation-index` | PSI + inflation rate for every category |
| GET | `/api/drift/{category}` | Full drift report (PSI, KL, CUSUM) |
| GET | `/api/history/{category}` | Daily avg/min/max price history |
| GET | `/api/forecast/{category}` | Prophet + XGBoost 28-day forecast |
| GET | `/api/products/{category}` | Product-level price breakdown |
| GET | `/api/movers` | Highest-volatility products (30-day window) |
| GET | `/api/subcategories/{category}` | Price bands by sub-category |
| GET | `/api/price-bands/{category}` | Histogram distribution + percentile stats |
| GET | `/api/volatility` | Rolling coefficient of variation per category |
| GET | `/api/search?q={term}` | Full-text product + category search |
| GET | `/api/export/{category}` | Full category data export as JSON |
| POST | `/api/analyze-image` | CV price extraction + CLIP classification |

Query parameters where applicable: `days` (history window), `periods` (forecast horizon), `limit` (result count), `bins` (histogram resolution).

---

## Dashboard Features

| Feature | Location | Description |
|---------|----------|-------------|
| Live Ticker Tape | Top bar | Scrolling real-time category prices and % changes |
| Market Heatmap | Overview | Colour-coded inflation heatmap — green to deep red |
| Drift Alerts | Overview | Pulsing alerts ranked by PSI severity |
| Market Movers | Overview | Top-priced products with 30-day delta |
| Category Comparison | Overview | Horizontal bar chart comparing avg prices |
| PSI Rankings | Intelligence | All categories sorted by drift severity |
| 90-Day Trend | Intelligence | Per-category sparkline with drift metrics |
| CUSUM + KL Panel | Intelligence | Full statistical breakdown on click |
| Forecast Chart | Forecast | SVG line chart with confidence bands and split marker |
| Model Stats | Forecast | Accuracy, MAPE, predicted vs current |
| Drag-Drop Scanner | CV Scanner | Upload any shelf image — extracts price, classifies product |
| Scan History | CV Scanner | Last 5 scans with timestamps |
| Watchlist | Analytics | Pin categories, track with mini sparklines |
| Sub-Category Bands | Analytics | Min/avg/max price ranges visualised |
| Product Deep-Dive | Analytics | Sortable table with live search, price spread column |

---

## Key Technical Decisions

**Why SQLite instead of Postgres?**
Zero-config deployment. The database is a single file you can copy, back up, or inspect in DB Browser without a server. For production scaling, switching to Postgres requires only changing `DB_PATH` to a connection string.

**Why a single-file frontend?**
No build step. No Node.js. No `npm install`. The dashboard opens in any browser by double-clicking the file. The API connection is optional — it falls back to mock data seamlessly, so you can demo the UI without a running backend.

**Why Prophet + XGBoost together?**
They fail in different directions. Prophet struggles with abrupt structural breaks but handles seasonality beautifully. XGBoost captures sudden spikes but needs feature engineering. Serving both lets downstream consumers pick the appropriate signal for their decision horizon.

**Why CUSUM alongside PSI?**
PSI detects sudden distributional shifts. CUSUM detects gradual directional drift — a price that rises 0.3% every week for 3 months has a low PSI at any single checkpoint but a very high CUSUM score. The combination eliminates both blind spots.

---

## Configuration

```bash
# .env
DB_PATH=data/pricescope.db
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PATH=frontend/

# Scheduler intervals (cron syntax)
DRIFT_CRON=0 */6 * * *
FORECAST_CRON=0 2 * * *
HEALTH_CRON=*/30 * * * *
```

---

## Roadmap

- **Multi-source ingestion** — Amazon, Flipkart, JioMart adapters alongside BigBasket
- **Alert webhooks** — POST to Slack / email when PSI exceeds threshold
- **Competitor price comparison** — side-by-side drift across sources for same SKU
- **Anomaly isolation** — highlight specific products driving category-level drift
- **Time-travel queries** — reconstruct market state at any historical date
- **PDF export** — generate drift + forecast reports for stakeholder distribution
- **PostgreSQL adapter** — production-scale database backend
- **Docker compose** — one-command full-stack deployment

---

## License

MIT — see `LICENSE`.

---

*Built on BigBasket's open product catalogue. Forecasting models adapted from Meta's Prophet and the XGBoost library. CV pipeline uses OpenAI CLIP and JaidedAI EasyOCR.*
