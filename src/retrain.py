"""
src/retrain.py
MLOps pipeline: data-drift detection + automated retraining triggers.
Monitors incoming data and retrains when drift or staleness detected.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/retrain.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

from src.preprocessing import preprocess


# ─────────────────────────── drift detection ────────────────────────────────

class DataDriftDetector:
    """Simple statistical drift detection using PSI and KS test."""

    def __init__(self, reference_df: pd.DataFrame,
                 target_col: str = "sales",
                 threshold_psi: float = 0.2,
                 threshold_ks: float = 0.05):
        self.reference = reference_df
        self.target    = target_col
        self.thr_psi   = threshold_psi
        self.thr_ks    = threshold_ks
        self.ref_stats = self._compute_stats(reference_df[target_col])

    def _compute_stats(self, series: pd.Series) -> Dict:
        return {
            "mean": float(series.mean()),
            "std":  float(series.std()),
            "p25":  float(series.quantile(0.25)),
            "p50":  float(series.quantile(0.50)),
            "p75":  float(series.quantile(0.75)),
            "p95":  float(series.quantile(0.95)),
        }

    def _psi(self, ref: pd.Series, curr: pd.Series, bins: int = 10) -> float:
        """Population Stability Index."""
        breakpoints = np.percentile(ref, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        ref_pct  = np.histogram(ref,  bins=breakpoints)[0] / (len(ref) + 1e-8)
        curr_pct = np.histogram(curr, bins=breakpoints)[0] / (len(curr) + 1e-8)
        ref_pct  = np.clip(ref_pct,  1e-6, None)
        curr_pct = np.clip(curr_pct, 1e-6, None)
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)

    def _ks_p_value(self, ref: pd.Series, curr: pd.Series) -> float:
        from scipy.stats import ks_2samp
        _, p = ks_2samp(ref.values, curr.values)
        return float(p)

    def detect(self, current_df: pd.DataFrame) -> Dict:
        curr_series = current_df[self.target]
        curr_stats  = self._compute_stats(curr_series)
        ref_series  = self.reference[self.target]

        psi     = self._psi(ref_series, curr_series)
        ks_p    = self._ks_p_value(ref_series, curr_series)
        mean_ch = abs(curr_stats["mean"] - self.ref_stats["mean"]) / (self.ref_stats["mean"] + 1e-8)
        std_ch  = abs(curr_stats["std"]  - self.ref_stats["std"])  / (self.ref_stats["std"] + 1e-8)

        drift_detected = (psi > self.thr_psi) or (ks_p < self.thr_ks) or (mean_ch > 0.3)

        return {
            "drift_detected":     drift_detected,
            "psi":                round(psi, 4),
            "ks_p_value":         round(ks_p, 4),
            "mean_change_pct":    round(mean_ch * 100, 2),
            "std_change_pct":     round(std_ch * 100, 2),
            "ref_mean":           round(self.ref_stats["mean"], 3),
            "curr_mean":          round(curr_stats["mean"], 3),
            "threshold_psi":      self.thr_psi,
            "threshold_ks":       self.thr_ks,
        }


# ─────────────────────────── stale-model check ──────────────────────────────

def is_model_stale(model_dir: str, dataset: str, max_age_hours: int = 24) -> bool:
    """Check if any model was trained more than max_age_hours ago."""
    pattern = f"models/{model_dir.lower()}"
    for root, _, files in os.walk(pattern):
        for f in files:
            if f.endswith((".pkl", ".pt", ".ckpt")):
                mtime = os.path.getmtime(os.path.join(root, f))
                age_h = (time.time() - mtime) / 3600
                if age_h < max_age_hours:
                    return False
    return True


# ─────────────────────────── retraining ─────────────────────────────────────

def retrain_models(dataset: str,
                   models: list = None,
                   force: bool = False) -> Dict:
    """Retrain specified (or all) models for a dataset."""
    if models is None:
        models = ["arima", "xgboost", "lstm"]

    results = {}
    log.info(f"[retrain] Starting retraining for dataset={dataset}, models={models}")

    for model in models:
        log.info(f"[retrain] Training {model.upper()} …")
        try:
            if model == "arima":
                from src.train_arima import train_dataset
                r = train_dataset(dataset, max_skus=5)
            elif model == "xgboost":
                from src.train_xgboost import train_dataset
                r = train_dataset(dataset, max_skus=10)
            elif model == "lstm":
                from src.train_lstm import train_dataset
                r = train_dataset(dataset, max_skus=3, epochs=10)
            elif model == "tft":
                from src.train_tft import train_tft
                r = train_tft(dataset, max_skus=2, max_epochs=5)
            else:
                log.warning(f"Unknown model: {model}")
                continue
            results[model] = r
            log.info(f"[retrain] {model.upper()} done – WMAPE={r.get('avg_WMAPE', r.get('WMAPE','?'))}")
        except Exception as e:
            log.error(f"[retrain] {model} failed: {e}")
            results[model] = {"error": str(e)}

    # log summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset":   dataset,
        "models":    results,
    }
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/retrain_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ─────────────────────────── main pipeline ──────────────────────────────────

def run_pipeline(dataset: str,
                 check_drift: bool = True,
                 force_retrain: bool = False,
                 models: list = None) -> Dict:

    log.info(f"=== MLOps Retraining Pipeline — {dataset} ===")
    os.makedirs("logs", exist_ok=True)

    # 1. load data
    try:
        df, meta = preprocess(dataset, max_skus=10)
    except Exception as e:
        log.error(f"Preprocessing failed: {e}")
        return {"error": str(e)}

    # 2. split into reference (first 80%) and current (last 20%)
    split_idx  = int(0.8 * len(df))
    ref_df     = df.iloc[:split_idx]
    curr_df    = df.iloc[split_idx:]

    report = {
        "dataset":   dataset,
        "timestamp": datetime.now().isoformat(),
        "n_ref":     len(ref_df),
        "n_curr":    len(curr_df),
    }

    # 3. drift detection
    drift_result = {}
    if check_drift and len(curr_df) > 0:
        log.info("Running drift detection …")
        detector    = DataDriftDetector(ref_df)
        drift_result = detector.detect(curr_df)
        report["drift"] = drift_result
        log.info(f"Drift detected: {drift_result['drift_detected']}  "
                 f"PSI={drift_result['psi']}  KS_p={drift_result['ks_p_value']}")

    # 4. decide whether to retrain
    should_retrain = (
        force_retrain
        or drift_result.get("drift_detected", False)
        or is_model_stale(dataset, dataset)
    )

    if should_retrain:
        log.info("⚡ Retraining triggered …")
        retrain_result = retrain_models(dataset, models=models, force=force_retrain)
        report["retrain"] = retrain_result
    else:
        log.info("✓ No retraining needed (models fresh, no drift)")
        report["retrain"] = {"skipped": True, "reason": "No drift, models fresh"}

    report_path = f"logs/pipeline_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Pipeline report → {report_path}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MLOps Retraining Pipeline")
    p.add_argument("--dataset",       default="M5", choices=["M5","Favorita","UCI"])
    p.add_argument("--force",         action="store_true", help="Force retrain even without drift")
    p.add_argument("--models",        nargs="+", default=["arima","xgboost"],
                   choices=["arima","xgboost","lstm","tft"])
    p.add_argument("--no-drift",      action="store_true", help="Skip drift check")
    args = p.parse_args()

    result = run_pipeline(
        dataset       = args.dataset,
        check_drift   = not args.no_drift,
        force_retrain = args.force,
        models        = args.models,
    )
    print(json.dumps({k: v for k, v in result.items()
                      if not isinstance(v, dict)}, indent=2))
