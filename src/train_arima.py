"""
src/train_arima.py  v2.0
ARIMA/SARIMA — trains per SKU, saves models, handles large datasets.
"""
import argparse, json, logging, os, pickle, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from src.preprocessing import preprocess
from src.evaluation import evaluate, save_report, print_metrics, wmape_improvement

try:
    from pmdarima import auto_arima
    PMDARIMA = True
except ImportError:
    PMDARIMA = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS = True
except ImportError:
    STATSMODELS = False

def split_series(s, n=28): return s.iloc[:-n], s.iloc[-n:]

def fit_model(train):
    if PMDARIMA:
        return auto_arima(train, seasonal=True, m=7, stepwise=True,
                          suppress_warnings=True, error_action="ignore",
                          max_p=2, max_q=2, max_P=1, max_Q=1, n_jobs=1), True
    if STATSMODELS:
        m = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,7),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return m, False
    raise ImportError("Install statsmodels or pmdarima")

def predict(model, horizon, use_pm):
    if use_pm:
        fc, ci = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
        return np.clip(fc,0,None), np.clip(ci[:,0],0,None), np.clip(ci[:,1],0,None)
    fc = model.get_forecast(steps=horizon)
    m  = fc.predicted_mean.values
    ci = fc.conf_int(alpha=0.05).values
    return np.clip(m,0,None), np.clip(ci[:,0],0,None), np.clip(ci[:,1],0,None)

def train_dataset(dataset="M5", horizon=28, max_skus=None,
                  category=None, store=None):
    model_dir = Path(f"models/{dataset.lower()}/arima")
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"=== ARIMA Training — {dataset} ===")
    df, meta = preprocess(dataset, max_skus=max_skus,
                          category_filter=category, store_filter=store)

    all_results=[]; models_meta={}
    skus = df["sku_id"].unique()
    log.info(f"Training on {len(skus)} SKUs...")

    for i, sku_id in enumerate(skus):
        log.info(f"  [{i+1}/{len(skus)}] SKU: {sku_id}")
        # Aggregate by date (sum across stores)
        series = (df[df["sku_id"]==sku_id]
                  .groupby("date")["sales"].sum()
                  .asfreq("D").fillna(0))
        if len(series) < horizon+30:
            log.warning(f"    Too short — skip")
            continue
        train, test = split_series(series, horizon)
        try:
            model, use_pm = fit_model(train)
            fc_m, fc_lo, fc_hi = predict(model, len(test), use_pm)
        except Exception as e:
            log.error(f"    Fit failed: {e}")
            from src.evaluation import seasonal_naive_forecast
            fc_m  = seasonal_naive_forecast(train, len(test))
            fc_lo = fc_m*0.8; fc_hi = fc_m*1.2
            model = None; use_pm = False

        m = evaluate(test.values, fc_m, fc_lo, fc_hi, model_name=f"ARIMA_{sku_id}")
        all_results.append(m)
        if (i+1) % 10 == 0: print_metrics(m)

        safe = str(sku_id).replace("/","_")
        mp   = model_dir/f"{safe}.pkl"
        with open(mp,"wb") as f:
            pickle.dump({"model":model,"use_pmdarima":use_pm,
                         "series_tail":train.tail(30).tolist()}, f)
        models_meta[str(sku_id)] = {"path":str(mp),"metrics":m}

    # Naive baseline
    baseline=[]
    for sku_id in df["sku_id"].unique():
        series = (df[df["sku_id"]==sku_id]
                  .groupby("date")["sales"].sum()
                  .asfreq("D").fillna(0))
        if len(series)<horizon+30: continue
        train,test = split_series(series,horizon)
        nb = np.full(len(test), train.mean())
        baseline.append(evaluate(test.values,nb,model_name="Naive"))

    avg_w = np.mean([r["WMAPE"] for r in all_results]) if all_results else 99
    avg_n = np.mean([r["WMAPE"] for r in baseline]) if baseline else 100
    imp   = wmape_improvement(avg_n, avg_w)

    log.info(f"\n{'='*50}")
    log.info(f"ARIMA avg WMAPE : {avg_w:.2f}%")
    log.info(f"Naive baseline  : {avg_n:.2f}%")
    log.info(f"Improvement     : {imp:.1f}%  (target >=15%)")
    log.info(f"{'='*50}")

    summary = {"dataset":dataset,"model":"ARIMA","n_skus":len(all_results),
               "avg_WMAPE":round(avg_w,4),"baseline_WMAPE":round(avg_n,4),
               "improvement_pct":imp,"target_met":bool(imp>=15)}
    save_report(summary, dataset, "arima")
    with open(model_dir/"models_meta.json","w") as f:
        json.dump(models_meta, f, indent=2, default=str)
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",  default="M5")
    p.add_argument("--horizon",  type=int, default=28)
    p.add_argument("--max_skus", type=int, default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--store",    default=None)
    a=p.parse_args()
    r=train_dataset(a.dataset, a.horizon, a.max_skus, a.category, a.store)
    print(json.dumps(r, indent=2))
