"""
create_project.py
Auto-setup script for AI-Powered Supply Chain Demand Forecasting Project.
Run this FIRST after cloning / unzipping the project.
"""

import os
import sys
import json
import subprocess
import platform

# ── colour helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def cprint(msg, colour=GREEN): print(f"{colour}{msg}{RESET}")
def header(msg):               print(f"\n{BOLD}{CYAN}{'='*60}\n  {msg}\n{'='*60}{RESET}")
def warn(msg):                 print(f"{YELLOW}⚠  {msg}{RESET}")
def err(msg):                  print(f"{RED}✗  {msg}{RESET}")
def ok(msg):                   print(f"{GREEN}✓  {msg}{RESET}")

# ── directory tree ───────────────────────────────────────────────────────────
DIRECTORIES = [
    "data/m5", "data/favorita", "data/uci",
    "models/m5", "models/favorita", "models/uci",
    "src", "api", "dashboard", "sota", "logs", "reports",
]

# ── sample-data generators ───────────────────────────────────────────────────
import random
import math

def _date_range(start="2020-01-01", periods=365):
    from datetime import date, timedelta
    d = date.fromisoformat(start)
    return [(d + timedelta(days=i)).isoformat() for i in range(periods)]

def _write_csv(path, rows, header_row):
    with open(path, "w") as f:
        f.write(",".join(header_row) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def create_m5_sample():
    """Tiny M5-style CSV for offline testing."""
    dates  = _date_range(periods=200)
    stores = ["CA_1", "TX_1"]
    items  = ["FOODS_1_001", "FOODS_1_002", "HOBBIES_1_001"]
    rows   = []
    for store in stores:
        for item in items:
            base = random.randint(5, 30)
            for i, d in enumerate(dates):
                promo = 1 if random.random() < 0.15 else 0
                sales = max(0, int(base + math.sin(i / 7) * 3 + random.gauss(0, 2) + promo * 5))
                rows.append([d, store, item, sales, promo,
                              round(random.uniform(1.5, 4.5), 2)])
    _write_csv("data/m5/m5_sample.csv", rows,
               ["date","store_id","item_id","sales","snap_event","sell_price"])
    ok("data/m5/m5_sample.csv created")

def create_favorita_sample():
    dates  = _date_range(periods=200)
    stores = [1, 2]
    items  = [101, 102, 103]
    rows   = []
    for store in stores:
        for item in items:
            base = random.randint(10, 50)
            for i, d in enumerate(dates):
                promo = 1 if random.random() < 0.2 else 0
                sales = max(0, int(base + math.sin(i / 7) * 5 + random.gauss(0, 3) + promo * 8))
                rows.append([d, store, item, sales, promo,
                              round(random.uniform(0.5, 3.0), 2)])
    _write_csv("data/favorita/favorita_sample.csv", rows,
               ["date","store_nbr","item_nbr","unit_sales","onpromotion","transactions"])
    ok("data/favorita/favorita_sample.csv created")

def create_uci_sample():
    dates  = _date_range(periods=200)
    skus   = ["SKU_001", "SKU_002", "SKU_003"]
    rows   = []
    for sku in skus:
        base = random.randint(20, 80)
        for i, d in enumerate(dates):
            sales = max(0, int(base + math.sin(i / 14) * 10 + random.gauss(0, 5)))
            rows.append([d, sku, sales,
                          round(random.uniform(5, 25), 2),
                          random.randint(50, 300)])
    _write_csv("data/uci/uci_sample.csv", rows,
               ["date","sku_id","quantity","unit_price","stock_level"])
    ok("data/uci/uci_sample.csv created")

# ── dataset-info JSON ─────────────────────────────────────────────────────────
DATASET_INFO = {
    "M5": {
        "name": "M5 Forecasting Competition (Walmart)",
        "source": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        "sample_file": "data/m5/m5_sample.csv",
        "date_col": "date", "target_col": "sales",
        "sku_col": "item_id", "store_col": "store_id",
        "promotion_col": "snap_event", "price_col": "sell_price",
        "description": "Hierarchical sales data for 42,840 Walmart products across 10 stores.",
    },
    "Favorita": {
        "name": "Corporación Favorita Grocery Sales",
        "source": "https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting",
        "sample_file": "data/favorita/favorita_sample.csv",
        "date_col": "date", "target_col": "unit_sales",
        "sku_col": "item_nbr", "store_col": "store_nbr",
        "promotion_col": "onpromotion", "price_col": "transactions",
        "description": "Grocery sales data from Ecuadorian retailer with promotions.",
    },
    "UCI": {
        "name": "UCI Online Retail / Time Series Dataset",
        "source": "https://archive.ics.uci.edu/ml/datasets/Online+Retail",
        "sample_file": "data/uci/uci_sample.csv",
        "date_col": "date", "target_col": "quantity",
        "sku_col": "sku_id", "store_col": None,
        "promotion_col": None, "price_col": "unit_price",
        "description": "Transactional data for a UK-based online retailer.",
    },
}

# ── README builder ─────────────────────────────────────────────────────────────
def create_readme():
    content = """# 📊 AI-Powered Supply Chain Demand Forecasting

## Quick Start (Windows 10 / i5 CPU / VS Code)

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\\Scripts\\activate          # Windows
# source venv/bin/activate      # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run setup (creates dirs + sample data)
python create_project.py

# 4. Train models (start with ARIMA baseline)
python src/train_arima.py --dataset M5

# 5. Start API server
uvicorn api.app:app --reload --port 8000

# 6. Launch Dashboard (new terminal)
streamlit run dashboard/app_streamlit.py
```

## Dataset Download Instructions

### M5 Forecasting (Walmart)
1. Go to: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
2. Accept competition rules → Download `m5-forecasting-accuracy.zip`
3. Extract and place `sales_train_evaluation.csv` → `data/m5/`

### Favorita Grocery
1. Go to: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data
2. Download `train.csv.gz` → extract → place in `data/favorita/`

### UCI Online Retail
1. Go to: https://archive.ics.uci.edu/ml/datasets/Online+Retail
2. Download `Online Retail.xlsx` → save as CSV → place in `data/uci/`

> **Note:** Sample datasets are auto-generated by `create_project.py` for immediate testing without downloading.

## Project Structure
```
demand_forecasting_project/
├── data/{m5, favorita, uci}/     ← datasets
├── src/                          ← training scripts
├── models/{m5, favorita, uci}/   ← saved models
├── api/app.py                    ← FastAPI backend
├── dashboard/app_streamlit.py    ← Streamlit dashboard
├── sota/                         ← SOTA model stubs
├── logs/                         ← training & API logs
└── reports/                      ← evaluation reports
```

## API Usage
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"dataset":"M5","sku":"FOODS_1_001","horizon":14}'
```

## Model Levels
| Level | Models | Purpose |
|-------|--------|---------|
| Basic | ARIMA/SARIMA | Statistical baseline |
| Intermediate | LSTM, XGBoost | ML with exogenous vars |
| Advanced | TFT (pytorch-forecasting) | SOTA interpretable |
| SOTA | GNN, Multi-task TFT | Research-grade |
"""
    with open("README.md", "w") as f:
        f.write(content)
    ok("README.md created")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    header("AI-Powered Supply Chain Demand Forecasting — Project Setup")

    # 1. directories
    cprint("\n[1/4] Creating directory structure ...", CYAN)
    for d in DIRECTORIES:
        os.makedirs(d, exist_ok=True)
        ok(f"  {d}/")

    # 2. sample data
    cprint("\n[2/4] Generating sample datasets ...", CYAN)
    create_m5_sample()
    create_favorita_sample()
    create_uci_sample()

    # 3. dataset config
    cprint("\n[3/4] Writing dataset_config.json ...", CYAN)
    with open("data/dataset_config.json", "w") as f:
        json.dump(DATASET_INFO, f, indent=2)
    ok("data/dataset_config.json")

    # 4. README
    cprint("\n[4/4] Creating README.md ...", CYAN)
    create_readme()

    # done
    header("Setup Complete!")
    print(f"""
{BOLD}Next Steps:{RESET}
  1. {CYAN}pip install -r requirements.txt{RESET}
  2. {CYAN}python src/train_arima.py --dataset M5{RESET}
  3. {CYAN}uvicorn api.app:app --reload{RESET}  (terminal 1)
  4. {CYAN}streamlit run dashboard/app_streamlit.py{RESET}  (terminal 2)

{YELLOW}Optional:{RESET} Download full datasets from Kaggle/UCI and place in data/{{m5,favorita,uci}}/
""")

if __name__ == "__main__":
    main()
