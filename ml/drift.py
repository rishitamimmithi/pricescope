import pandas as pd
import numpy as np
from scipy import stats
import sqlite3

DB_PATH = "data/pricescope.db"

def load_category_prices(category: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT price, recorded_at, name 
        FROM prices 
        WHERE category LIKE ?
        ORDER BY recorded_at
    """
    df = pd.read_sql(query, conn, params=[f'%{category}%'])
    conn.close()
    df['recorded_at'] = pd.to_datetime(df['recorded_at'])
    return df

def calculate_psi(reference: np.ndarray, current: np.ndarray, bins=10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  = stable
    PSI 0.1-0.2 = moderate drift
    PSI > 0.2  = significant drift — ALERT
    """
    ref_counts, bin_edges = np.histogram(reference, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = (ref_counts + 1e-6) / len(reference)
    curr_pct = (curr_counts + 1e-6) / len(current)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return round(float(psi), 4)

def calculate_kl_divergence(reference: np.ndarray, current: np.ndarray) -> float:
    """KL divergence — how much current distribution diverges from reference."""
    ref_counts, bin_edges = np.histogram(reference, bins=20, density=True)
    curr_counts, _ = np.histogram(current, bins=bin_edges, density=True)

    ref_smooth = ref_counts + 1e-10
    curr_smooth = curr_counts + 1e-10

    kl = stats.entropy(curr_smooth, ref_smooth)
    return round(float(kl), 4)

def cusum_detector(prices: list, threshold: float = 5.0) -> dict:
    """
    CUSUM: catches slow gradual drift that PSI misses.
    Returns where drift started.
    """
    mean = np.mean(prices[:30])
    std = np.std(prices[:30]) + 1e-6

    cusum_pos, cusum_neg = 0, 0
    k = 0.5
    drift_start = None

    for i, price in enumerate(prices):
        z = (price - mean) / std
        cusum_pos = max(0, cusum_pos + z - k)
        cusum_neg = max(0, cusum_neg - z - k)

        if cusum_pos > threshold or cusum_neg > threshold:
            if drift_start is None:
                drift_start = i

    return {
        "drift_detected": drift_start is not None,
        "drift_start_index": drift_start,
        "final_cusum_pos": round(cusum_pos, 2),
        "final_cusum_neg": round(cusum_neg, 2)
    }

def full_drift_report(category: str) -> dict:
    df = load_category_prices(category)

    if len(df) < 60:
        return {"error": f"Not enough data for {category}"}

    # Daily average price
    daily_avg = df.groupby(df['recorded_at'].dt.date)['price'].mean().reset_index()
    daily_avg = daily_avg.sort_values('recorded_at')

    n = len(daily_avg)
    reference = daily_avg['price'].values[:n//2]
    current = daily_avg['price'].values[n//2:]

    psi = calculate_psi(reference, current)
    kl = calculate_kl_divergence(reference, current)
    cusum = cusum_detector(daily_avg['price'].tolist())

    inflation_rate = (current.mean() - reference.mean()) / reference.mean() * 100

    if psi > 0.2 or cusum['drift_detected']:
        alert = "🔴 HIGH — significant price drift"
    elif psi > 0.1:
        alert = "🟡 MEDIUM — monitor closely"
    else:
        alert = "🟢 LOW — stable"

    return {
        "category": category,
        "psi": psi,
        "kl_divergence": kl,
        "cusum": cusum,
        "inflation_rate_pct": round(inflation_rate, 2),
        "reference_mean_price": round(reference.mean(), 2),
        "current_mean_price": round(current.mean(), 2),
        "alert_level": alert,
        "data_points": n
    }

def all_categories_report() -> list:
    conn = sqlite3.connect(DB_PATH)
    categories = pd.read_sql(
        "SELECT DISTINCT category FROM prices", conn
    )['category'].tolist()
    conn.close()

    results = []
    for cat in categories:
        report = full_drift_report(cat)
        if 'error' not in report:
            results.append(report)

    results.sort(key=lambda x: abs(x['inflation_rate_pct']), reverse=True)
    return results

if __name__ == "__main__":
    print("🔍 Running drift detection on all categories...\n")
    reports = all_categories_report()

    for r in reports:
        print(f"Category: {r['category']}")
        print(f"  PSI: {r['psi']} | KL: {r['kl_divergence']}")
        print(f"  Inflation: {r['inflation_rate_pct']}%")
        print(f"  Price: ₹{r['reference_mean_price']} → ₹{r['current_mean_price']}")
        print(f"  Alert: {r['alert_level']}")
        print(f"  CUSUM drift: {r['cusum']['drift_detected']}")
        print()

    print(f"✅ Drift report complete. {len(reports)} categories analyzed.")