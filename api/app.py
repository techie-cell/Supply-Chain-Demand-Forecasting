"""
api/app.py  v2.0
FastAPI backend — supports full M5 dataset, all SKUs.
"""
import json, logging, os, pickle, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="AI Supply Chain Demand Forecasting API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    dataset:  str = "M5"
    sku:      str = "FOODS_1_001_CA_1_evaluation"
    horizon:  int = Field(14, ge=1, le=30)
    model:    str = "auto"

class RetrainRequest(BaseModel):
    dataset: str = "M5"
    models:  List[str] = ["arima","xgboost"]
    force:   bool = False

# ── cache ────────────────────────────────────────────────────────────────────
_data_cache: Dict[str, pd.DataFrame] = {}
_meta_cache: Dict[str, dict] = {}

def get_df(dataset: str) -> pd.DataFrame:
    if dataset not in _data_cache:
        from src.preprocessing import preprocess
        df, meta = preprocess(dataset)
        _data_cache[dataset] = df
        _meta_cache[dataset] = meta
    return _data_cache[dataset]

def get_meta(dataset: str) -> dict:
    if dataset not in _meta_cache:
        get_df(dataset)
    return _meta_cache.get(dataset, {})

# ── model loading ─────────────────────────────────────────────────────────────
def find_model(dataset: str, model: str = "auto"):
    base  = Path(f"models/{dataset.lower()}")
    order = ["xgboost","arima","lstm"] if model=="auto" else [model]
    for m in order:
        if m=="xgboost":
            p = base/"xgboost"/"xgboost_model.pkl"
            if p.exists():
                with open(p,"rb") as f: return pickle.load(f), "XGBoost"
        if m=="arima":
            d = base/"arima"
            if d.exists():
                pkls=[p for p in d.glob("*.pkl") if p.name!="models_meta.json"]
                if pkls:
                    with open(pkls[0],"rb") as f: return pickle.load(f), "ARIMA"
    return None, "SeasonalNaive"

# ── inference ─────────────────────────────────────────────────────────────────
def _xgb_forecast(art, df_sku, horizon):
    from src.feature_engineering import build_features
    df_fe = build_features(df_sku)
    row   = df_fe.sort_values("date").tail(1).copy()
    model = art["model"]; feats = art["features"]
    preds=[]
    for h in range(1, horizon+1):
        row["horizon"]=h
        x=row[[c for c in feats if c in row.columns]].fillna(0)
        preds.append(float(np.clip(model.predict(x)[0],0,None)))
    p=np.array(preds)
    return p, p*0.8, p*1.2

def _arima_forecast(art, horizon):
    model=art.get("model"); use_pm=art.get("use_pmdarima",True)
    if model is None: raise ValueError("No model")
    if use_pm:
        fc,ci=model.predict(n_periods=horizon,return_conf_int=True)
        return np.clip(fc,0,None),np.clip(ci[:,0],0,None),np.clip(ci[:,1],0,None)
    fc=model.get_forecast(steps=horizon)
    m=fc.predicted_mean.values; ci=fc.conf_int().values
    return np.clip(m,0,None),np.clip(ci[:,0],0,None),np.clip(ci[:,1],0,None)

def _naive_forecast(series, horizon):
    import math
    s=series.values
    w=min(28,len(s)); ma=np.mean(s[-w:])
    tr=np.polyfit(np.arange(w),s[-w:],1)[0] if w>1 else 0
    fc=np.array([max(0,ma+tr*i+math.sin(i/7)*2) for i in range(horizon)])
    return fc, fc*0.8, fc*1.2

def generate_forecast(dataset, sku, horizon, model_pref="auto"):
    t0  = time.time()
    df  = get_df(dataset)
    dsk = df[df["sku_id"]==sku].sort_values("date")
    if dsk.empty:
        avail=df["sku_id"].unique().tolist()[:5]
        raise HTTPException(404, f"SKU '{sku}' not found. Examples: {avail}")

    art, model_name = find_model(dataset, model_pref)
    series = (dsk.groupby("date")["sales"].sum()
                 .asfreq("D").fillna(0))
    try:
        if model_name=="XGBoost" and art: fc,lo,hi=_xgb_forecast(art,dsk,horizon)
        elif model_name=="ARIMA" and art: fc,lo,hi=_arima_forecast(art,horizon)
        else: raise ValueError("fallback")
    except Exception:
        fc,lo,hi=_naive_forecast(series,horizon)
        model_name="SeasonalNaive"

    last  = dsk["date"].max()
    dates = [(pd.Timestamp(last)+pd.Timedelta(days=i+1)).strftime("%Y-%m-%d")
             for i in range(horizon)]
    return {
        "forecast":            [round(float(v),2) for v in fc],
        "confidence_interval": {"lower":[round(float(v),2) for v in lo],
                                "upper":[round(float(v),2) for v in hi]},
        "model_used":   model_name,
        "dataset":      dataset,
        "sku":          sku,
        "horizon":      horizon,
        "dates":        dates,
        "inference_ms": round((time.time()-t0)*1000,1),
    }

# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health(): return {"status":"ok","version":"2.0","timestamp":datetime.now().isoformat()}

@app.get("/datasets")
def list_datasets():
    return {"datasets":["M5","Favorita","UCI"],
            "note":"M5 is the full Walmart dataset with 42,840 SKUs"}

@app.get("/meta/{dataset}")
def dataset_meta(dataset:str):
    try:
        from src.preprocessing import get_metadata
        return get_metadata(dataset.upper())
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/skus/{dataset}")
def list_skus(dataset:str, category:str=None, store:str=None, limit:int=1000):
    try:
        m = get_meta(dataset.upper())
        skus = m.get("skus",[])
        if category: skus=[s for s in skus if s.startswith(category)]
        return {"dataset":dataset,"total":len(skus),"skus":skus[:limit]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/categories/{dataset}")
def list_categories(dataset:str):
    try:
        m = get_meta(dataset.upper())
        return {"dataset":dataset,"categories":m.get("categories",[])}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict")
def predict(req: PredictRequest):
    ds = req.dataset.upper()
    log.info(f"Predict: {ds} | {req.sku} | h={req.horizon} | model={req.model}")
    return generate_forecast(ds, req.sku, req.horizon, req.model)

@app.get("/metrics/{dataset}/{model}")
def get_metrics(dataset:str, model:str):
    p=f"reports/{dataset}_{model}_eval.json"
    if not os.path.exists(p): raise HTTPException(404, f"No report: {p}")
    with open(p) as f: return json.load(f)

@app.post("/retrain")
def trigger_retrain(req:RetrainRequest, bg:BackgroundTasks):
    def _do():
        from src.retrain import retrain_models
        retrain_models(req.dataset, models=req.models, force=req.force)
    bg.add_task(_do)
    return {"status":"started","dataset":req.dataset,"models":req.models}

@app.get("/forecast/batch/{dataset}")
def batch(dataset:str, horizon:int=14, limit:int=10, category:str=None):
    ds=dataset.upper()
    m=get_meta(ds)
    skus=m.get("skus",[])
    if category: skus=[s for s in skus if s.startswith(category)]
    results={}
    for sku in skus[:limit]:
        try: r=generate_forecast(ds,sku,horizon); results[sku]={"forecast":r["forecast"],"model":r["model_used"]}
        except Exception as e: results[sku]={"error":str(e)}
    return {"dataset":ds,"horizon":horizon,"results":results}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
