import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import sqlite3
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "data/pricescope.db"

def get_category_timeseries(category: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT DATE(recorded_at) as date, AVG(price) as avg_price
        FROM prices WHERE category LIKE ?
        GROUP BY DATE(recorded_at)
        ORDER BY date
    """, conn, params=[f'%{category}%'])
    conn.close()
    return df

def prophet_forecast(category: str, periods: int = 28) -> dict:
    df = get_category_timeseries(category)
    if len(df) < 30:
        return {"error": "Need at least 30 days of data"}

    prophet_df = df.rename(columns={'date': 'ds', 'avg_price': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    future_preds = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    last_actual = float(prophet_df['y'].iloc[-1])
    predicted_final = float(future_preds['yhat'].iloc[-1])
    change_pct = round((predicted_final - last_actual) / last_actual * 100, 2)

    return {
        "category": category,
        "method": "Prophet",
        "forecast_days": periods,
        "last_actual_price": round(last_actual, 2),
        "predicted_price_28d": round(predicted_final, 2),
        "predicted_change_pct": change_pct,
        "predictions": [
            {
                "date": str(row['ds'].date()),
                "predicted": round(row['yhat'], 2),
                "lower": round(row['yhat_lower'], 2),
                "upper": round(row['yhat_upper'], 2)
            }
            for _, row in future_preds.iterrows()
        ]
    }

def xgboost_forecast(category: str) -> dict:
    df = get_category_timeseries(category)
    if len(df) < 45:
        return {"error": "Need at least 45 days"}

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    for lag in [1, 3, 7, 14, 21]:
        df[f'lag_{lag}'] = df['avg_price'].shift(lag)

    df['rolling_7_mean'] = df['avg_price'].rolling(7).mean()
    df['rolling_14_std'] = df['avg_price'].rolling(14).std()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df = df.dropna()

    feature_cols = [c for c in df.columns if c not in ['date', 'avg_price']]
    X = df[feature_cols]
    y = df['avg_price']

    split = max(14, int(len(df) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])

    return {
        "category": category,
        "method": "XGBoost",
        "test_mape": round(mape, 2),
        "top_features": top_features,
        "actual_vs_predicted": [
            {"actual": round(float(a), 2), "predicted": round(float(p), 2)}
            for a, p in zip(y_test.tolist(), y_pred.tolist())
        ]
    }

def full_forecast_report(category: str) -> dict:
    print(f"  📈 Prophet forecasting {category}...")
    prophet = prophet_forecast(category)

    print(f"  🤖 XGBoost forecasting {category}...")
    xgb = xgboost_forecast(category)

    return {
        "category": category,
        "prophet": prophet,
        "xgboost": xgb
    }

if __name__ == "__main__":
    test_categories = [
        "Fruits & Vegetables",
        "Bakery, Cakes & Dairy",
        "Snacks & Branded Foods",
        "Beverages"
    ]

    print("🔮 Running price forecasting...\n")
    for cat in test_categories:
        print(f"\n{'='*50}")
        print(f"Category: {cat}")
        result = full_forecast_report(cat)

        p = result['prophet']
        x = result['xgboost']

        if 'error' not in p:
            print(f"  Prophet: ₹{p['last_actual_price']} → ₹{p['predicted_price_28d']} ({p['predicted_change_pct']}% in 28 days)")

        if 'error' not in x:
            print(f"  XGBoost MAPE: {x['test_mape']}%")
            print(f"  Top feature: {list(x['top_features'].keys())[0]}")

    print("\n✅ Forecasting complete!")