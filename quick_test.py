"""
quick_test.py
Lightweight smoke test — verifies all modules import and core logic works.
Run BEFORE installing heavy dependencies to check project structure.
  python quick_test.py
"""

import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(__file__))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results = []

def test(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, "PASS", None))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, "FAIL", str(e)))


print("\n" + "="*55)
print("  AI Demand Forecasting — Quick Test")
print("="*55 + "\n")

# ── 1. Directory structure ────────────────────────────────────
print("[1] Project Structure")
def chk_dirs():
    for d in ["data/m5","data/favorita","data/uci",
              "models/m5","models/favorita","models/uci",
              "src","api","dashboard","sota","logs","reports"]:
        assert os.path.isdir(d), f"Missing: {d}"
test("All directories present", chk_dirs)

def chk_files():
    required = [
        "requirements.txt","config.yaml","run.py","create_project.py",
        "src/preprocessing.py","src/feature_engineering.py",
        "src/evaluation.py","src/train_arima.py","src/train_lstm.py",
        "src/train_xgboost.py","src/train_tft.py","src/retrain.py",
        "api/app.py","dashboard/app_streamlit.py",
        "sota/anomaly_detection.py","sota/gnn_model.py",
        "sota/llm_integration.py","sota/multitask_tft.py",
    ]
    for f in required:
        assert os.path.isfile(f), f"Missing: {f}"
test("All source files present", chk_files)

# ── 2. Core imports ───────────────────────────────────────────
print("\n[2] Core Python Imports")
test("numpy",   lambda: __import__("numpy"))
test("pandas",  lambda: __import__("pandas"))
test("json",    lambda: __import__("json"))
test("pathlib", lambda: __import__("pathlib"))
test("logging", lambda: __import__("logging"))

# ── 3. Optional ML imports ────────────────────────────────────
print("\n[3] ML Library Imports (optional — OK if FAIL)")
ml_libs = [
    ("scikit-learn",   "sklearn"),
    ("xgboost",        "xgboost"),
    ("torch",          "torch"),
    ("statsmodels",    "statsmodels"),
    ("pmdarima",       "pmdarima"),
    ("pytorch-forecasting", "pytorch_forecasting"),
    ("shap",           "shap"),
    ("plotly",         "plotly"),
    ("streamlit",      "streamlit"),
    ("fastapi",        "fastapi"),
]
for label, mod in ml_libs:
    try:
        __import__(mod)
        print(f"  {PASS} {label}")
    except ImportError:
        print(f"  {WARN} {label} — not installed (run: pip install -r requirements.txt)")

# ── 4. Evaluation module ──────────────────────────────────────
print("\n[4] Evaluation Module")
import numpy as np

def chk_metrics():
    from src.evaluation import mae, rmse, wmape, mape, evaluate
    y  = np.array([10., 20., 30., 40., 50.])
    yp = np.array([11., 19., 31., 39., 51.])
    assert abs(mae(y, yp) - 1.0) < 0.01
    assert wmape(y, yp) < 10.0
    m = evaluate(y, yp, model_name="test")
    assert "WMAPE" in m and "MAE" in m
test("Metrics compute correctly",  chk_metrics)

def chk_baseline():
    from src.evaluation import (moving_average_forecast, seasonal_naive_forecast,
                                 naive_forecast)
    s = np.arange(1.0, 31.0)
    import pandas as pd
    series = pd.Series(s)
    f1 = naive_forecast(series, 7)
    f2 = seasonal_naive_forecast(series, 7)
    f3 = moving_average_forecast(series, 7)
    assert len(f1) == 7 and len(f2) == 7 and len(f3) == 7
test("Baseline forecasters work", chk_baseline)

# ── 5. Preprocessing ─────────────────────────────────────────
print("\n[5] Preprocessing Module")
def chk_sample_data():
    import pandas as pd
    # generate sample data inline (same logic as create_project.py)
    import random, math
    from datetime import date, timedelta
    dates = [(date(2023,1,1) + timedelta(days=i)).isoformat() for i in range(120)]
    rows  = [{"date": d, "store_id": "CA_1", "item_id": "FOODS_1",
               "sales": max(0, int(20 + math.sin(i/7)*3 + random.gauss(0,2))),
               "snap_event": 0, "sell_price": 2.0}
              for i, d in enumerate(dates)]
    df = pd.DataFrame(rows)
    assert len(df) == 120
    assert "sales" in df.columns
test("Sample data generation", chk_sample_data)

def chk_sample_file():
    import pandas as pd
    p = "data/m5/m5_sample.csv"
    if os.path.exists(p):
        df = pd.read_csv(p)
        assert len(df) > 0
    # else skip — run create_project.py first
test("Sample CSV readable (if exists)", chk_sample_file)

# ── 6. Feature engineering ────────────────────────────────────
print("\n[6] Feature Engineering")
def chk_date_feats():
    import pandas as pd
    from src.feature_engineering import add_date_features
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=30), "sales": range(30)})
    df = add_date_features(df)
    assert "day_of_week" in df.columns
    assert "sin_week"    in df.columns
test("Date features", chk_date_feats)

def chk_lag_feats():
    import pandas as pd
    from src.feature_engineering import add_lag_features
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=40),
        "sku_id": "A", "store_id": "S",
        "sales": np.arange(40.0)
    })
    df = add_lag_features(df, lags=[1, 7])
    assert "lag_1" in df.columns and "lag_7" in df.columns
test("Lag features", chk_lag_feats)

# ── 7. Anomaly Detection (no heavy deps) ──────────────────────
print("\n[7] SOTA — Anomaly Detection")
def chk_anomaly():
    import pandas as pd
    from sota.anomaly_detection import StatisticalDetector
    s = pd.Series([10,11,10,12,9,10,100,10,11,10,9,11,10])
    det  = StatisticalDetector(zscore_thresh=2.5)
    mask = det.detect(s)
    assert mask.sum() >= 1, "Expected at least 1 anomaly (spike at 100)"
test("Statistical anomaly detection", chk_anomaly)

# ── 8. Config ─────────────────────────────────────────────────
print("\n[8] Configuration")
def chk_config():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "datasets" in cfg
    assert "models"   in cfg
    assert "mlops"    in cfg
try:
    test("config.yaml loads", chk_config)
except ImportError:
    print(f"  {WARN} PyYAML not installed — skipping")

# ── Summary ───────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")

print("\n" + "="*55)
print(f"  Results: {passed}/{total} passed  |  {failed} failed")
print("="*55)

if failed == 0:
    print("\n\033[92m✓ All tests passed! Project structure is correct.\033[0m")
    print("\nNext steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. python create_project.py    (generates sample data)")
    print("  3. python run.py train --dataset M5 --model arima")
    print("  4. python run.py api           (Terminal 1)")
    print("  5. python run.py dashboard     (Terminal 2)")
else:
    print(f"\n\033[93m{failed} test(s) failed — check messages above.\033[0m")

print()
