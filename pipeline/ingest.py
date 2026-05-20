import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import random

DB_PATH = "data/pricescope.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT,
            name TEXT,
            category TEXT,
            sub_category TEXT,
            price REAL,
            image_url TEXT,
            source TEXT,
            recorded_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("✅ DB initialized")

def load_bigbasket():
    path = "data/raw/BigBasket Products.csv"
    df = pd.read_csv(path)
    print(f"Raw columns: {df.columns.tolist()}")

    df = df.rename(columns={
        'product': 'name',
        'sale_price': 'price',
    })

    df['image_url'] = ''
    df = df[['name', 'category', 'sub_category', 'price', 'image_url']].dropna()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df[df['price'] > 0]
    df['product_id'] = df['name'].str.lower().str.replace(' ', '_').str[:50]
    df['source'] = 'bigbasket'
    print(f"✅ Loaded {len(df)} clean products")
    return df

def simulate_price_history(df, days=90):
    print(f"⏳ Simulating {days} days of price history for {len(df)} products...")
    records = []
    base_date = datetime.now() - timedelta(days=days)

    drift_rates = {
        'Fruits & Vegetables': 0.003,
        'Bakery, Cakes & Dairy': 0.002,
        'Snacks & Branded Foods': 0.001,
        'Beverages': 0.0005,
        'Cleaning & Household': 0.0008,
    }

    for _, row in df.iterrows():
        base_price = row['price']
        daily_drift = drift_rates.get(row['category'], 0.001)

        for day in range(days):
            noise = random.gauss(0, 0.008)
            price = base_price * (1 + daily_drift * day + noise)
            price = max(1.0, round(price, 2))
            records.append({
                'product_id': row['product_id'],
                'name': row['name'],
                'category': row['category'],
                'sub_category': row.get('sub_category', ''),
                'price': price,
                'image_url': '',
                'source': 'bigbasket',
                'recorded_at': (base_date + timedelta(days=day)).isoformat()
            })

    result = pd.DataFrame(records)
    print(f"✅ Generated {len(result)} price records")
    return result

def store_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('prices', conn, if_exists='append', index=False)
    conn.close()
    print("✅ Stored to DB")

def verify_db():
    conn = sqlite3.connect(DB_PATH)
    count = pd.read_sql("SELECT COUNT(*) as total FROM prices", conn).iloc[0]['total']
    cats = pd.read_sql(
        "SELECT category, COUNT(*) as n FROM prices GROUP BY category ORDER BY n DESC LIMIT 5",
        conn
    )
    conn.close()
    print(f"\n✅ TOTAL RECORDS IN DB: {count}")
    print("\nTop categories:")
    print(cats.to_string(index=False))

if __name__ == "__main__":
    init_db()
    df = load_bigbasket()
    history_df = simulate_price_history(df.head(150))
    store_to_db(history_df)
    verify_db()
    print("\n🎉 Day 1 complete! DB is ready.")