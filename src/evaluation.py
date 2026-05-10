"""
src/evaluation.py  v2.0
Metrics: MAE, RMSE, MAPE, WMAPE, SMAPE, PI Coverage.
JSON-safe serialization for all numpy types.
"""
import json, os
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

def _safe(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):    return bool(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, dict):           return {k:_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)):   return [_safe(i) for i in obj]
    return obj

def mae(y,p):   return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def rmse(y,p):  return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def mape(y,p,eps=1e-8):
    y,p=np.asarray(y),np.asarray(p)
    m=np.abs(y)>eps
    return float(np.mean(np.abs((y[m]-p[m])/y[m]))*100) if m.any() else 0.0
def wmape(y,p,eps=1e-8):
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    tot=np.sum(np.abs(y))
    return float(np.sum(np.abs(y-p))/tot*100) if tot>eps else 0.0
def smape(y,p,eps=1e-8):
    y,p=np.asarray(y),np.asarray(p)
    d=(np.abs(y)+np.abs(p))/2+eps
    return float(np.mean(np.abs(y-p)/d)*100)
def pi_coverage(y,lo,hi):
    y,lo,hi=np.asarray(y),np.asarray(lo),np.asarray(hi)
    return float(np.mean((y>=lo)&(y<=hi)))

def evaluate(y_true, y_pred, lower=None, upper=None, model_name="model"):
    y,p=np.asarray(y_true,dtype=float),np.asarray(y_pred,dtype=float)
    m={"model":model_name,"MAE":round(mae(y,p),4),"RMSE":round(rmse(y,p),4),
       "MAPE":round(mape(y,p),4),"WMAPE":round(wmape(y,p),4),
       "SMAPE":round(smape(y,p),4),"n_obs":len(y)}
    if lower is not None and upper is not None:
        m["PI_Coverage_95"]=round(pi_coverage(y,lower,upper),4)
        m["PI_Width"]=round(float(np.mean(np.asarray(upper)-np.asarray(lower))),4)
    return m

def compare_models(results): return pd.DataFrame(results).sort_values("WMAPE").reset_index(drop=True)
def wmape_improvement(base, model): return round((base-model)/base*100,2) if base else 0.0

def evaluate_per_sku(df,actual="sales",pred="pred",sku="sku_id"):
    return pd.DataFrame([{**evaluate(g[actual].values,g[pred].values,model_name=str(s)),"sku_id":s}
                         for s,g in df.groupby(sku)]).sort_values("WMAPE")

def save_report(metrics, dataset, model, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    fname=f"{out_dir}/{dataset}_{model}_eval.json"
    with open(fname,"w",encoding="utf-8") as f:
        json.dump(_safe(metrics), f, indent=2)
    return fname

def print_metrics(m):
    print(f"\n{'─'*45}")
    print(f"  Model : {m.get('model','?')}")
    print(f"{'─'*45}")
    for k,v in m.items():
        if k!="model": print(f"  {k:<20} {v}")
    print(f"{'─'*45}\n")

def naive_forecast(s,h):  return np.full(h,float(s.iloc[-1]))
def seasonal_naive_forecast(s,h,season=7):
    out=[]
    for i in range(h):
        idx=len(s)-season+(i%season)
        out.append(float(s.iloc[max(0,idx)]))
    return np.array(out)
def moving_average_forecast(s,h,w=7): return np.full(h,float(s.tail(w).mean()))

if __name__=="__main__":
    y=np.abs(np.random.randn(100)*20+50)
    p=y+np.random.randn(100)*5
    print_metrics(evaluate(y,p,p-10,p+10,model_name="Test"))
