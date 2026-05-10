"""
src/train_xgboost.py  v2.0
XGBoost tabular forecasting — full dataset, SHAP disabled for compatibility.
"""
import argparse, json, logging, os, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from src.preprocessing import preprocess
from src.feature_engineering import build_features
from src.evaluation import evaluate, save_report, print_metrics

try:
    import xgboost as xgb
    XGB = True
except ImportError:
    XGB = False
    log.warning("xgboost not installed")

FEATURE_COLS = [
    "lag_1","lag_7","lag_14","lag_28",
    "roll_mean_7","roll_mean_14","roll_mean_28",
    "roll_std_7","roll_std_28","roll_max_7","roll_max_28",
    "day_of_week","day_of_month","week_of_year","month","quarter",
    "is_weekend","is_holiday","is_month_start","is_month_end",
    "sin_week","cos_week","sin_year","cos_year",
    "promo","price","price_change_pct","price_vs_mean",
    "promo_next_7","promo_rolling",
    "web_trend","macro_cpi","competitor_price","inventory_level",
    "sku_mean","sku_std","sku_cv","sku_id_enc","store_id_enc",
]

XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, reg_alpha=0.1,
    objective="reg:squarederror", eval_metric="rmse",
    n_jobs=-1, random_state=42, early_stopping_rounds=20,
)

def get_feats(df):
    return [c for c in FEATURE_COLS if c in df.columns]

def train_xgb_direct(df, dataset, horizon=14):
    if not XGB: raise ImportError("xgboost not installed")
    model_dir = Path(f"models/{dataset.lower()}/xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = get_feats(df)
    log.info(f"Features: {feat_cols}")

    # Build multi-step dataset
    records=[]
    for sku_id, grp in df.groupby("sku_id"):
        grp=grp.sort_values("date").copy()
        for h in range(1, horizon+1):
            tmp=grp.copy()
            tmp["target"] = tmp["sales"].shift(-h)
            tmp["horizon"] = h
            records.append(tmp.dropna(subset=["target"]))

    all_df   = pd.concat(records, ignore_index=True)
    fcols    = [c for c in feat_cols+["horizon"] if c in all_df.columns]
    X        = all_df[fcols].fillna(0)
    y        = all_df["target"].values
    split    = int(0.8*len(X))
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y[:split], y[split:]

    log.info(f"Training XGBoost on {len(X_train):,} samples...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=50)

    y_pred = np.clip(model.predict(X_val), 0, None)
    m = evaluate(y_val, y_pred, y_pred*0.8, y_pred*1.2, model_name="XGBoost")
    print_metrics(m)

    # Built-in feature importance (no SHAP - version conflict avoided)
    importances = dict(sorted(
        zip(fcols, model.feature_importances_.tolist()),
        key=lambda x:-x[1])[:15])
    m["feature_importance"] = importances

    model_path = model_dir/"xgboost_model.pkl"
    with open(model_path,"wb") as f:
        pickle.dump({"model":model,"features":fcols,"importance":importances}, f)
    log.info(f"Model saved → {model_path}")
    return m

def train_dataset(dataset="M5", horizon=14, max_skus=None, category=None, store=None):
    log.info(f"=== XGBoost Training — {dataset} ===")
    df, meta = preprocess(dataset, max_skus=max_skus,
                          category_filter=category, store_filter=store)
    df = build_features(df, dataset=dataset)
    summary = train_xgb_direct(df, dataset, horizon)
    summary.update({"dataset":dataset,"model":"XGBoost"})
    save_report(summary, dataset, "xgboost")
    log.info(f"XGBoost WMAPE = {summary['WMAPE']:.2f}%")
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",  default="M5")
    p.add_argument("--horizon",  type=int, default=14)
    p.add_argument("--max_skus", type=int, default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--store",    default=None)
    a=p.parse_args()
    r=train_dataset(a.dataset, a.horizon, a.max_skus, a.category, a.store)
    print(json.dumps({k:v for k,v in r.items() if not isinstance(v,dict)}, indent=2))
