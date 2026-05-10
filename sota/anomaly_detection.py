"""
sota/anomaly_detection.py
Supply-chain anomaly detection using Isolation Forest + statistical methods.
Detects: demand spikes, supply disruptions, data-quality issues.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─────────────────────────── feature builder ─────────────────────────────────

def build_anomaly_features(df_sku: pd.DataFrame,
                            target_col: str = "sales") -> pd.DataFrame:
    s = df_sku[target_col].values.astype(float)
    n = len(s)
    feat = pd.DataFrame(index=df_sku.index)
    feat["sales"]          = s
    feat["lag1"]           = np.concatenate([[np.nan], s[:-1]])
    feat["lag7"]           = np.concatenate([np.full(7, np.nan), s[:-7]])
    feat["roll_mean_7"]    = pd.Series(s).rolling(7, min_periods=1).mean().values
    feat["roll_std_7"]     = pd.Series(s).rolling(7, min_periods=1).std().fillna(0).values
    feat["zscore"]         = (s - feat["roll_mean_7"]) / (feat["roll_std_7"] + 1e-8)
    feat["change"]         = np.concatenate([[0], np.diff(s)])
    feat["pct_change"]     = np.concatenate([[0], np.diff(s) / (np.abs(s[:-1]) + 1e-8)])
    feat["day_of_week"]    = pd.to_datetime(df_sku["date"]).dt.dayofweek.values
    feat["month"]          = pd.to_datetime(df_sku["date"]).dt.month.values
    feat = feat.fillna(0)
    return feat


# ─────────────────────────── detectors ───────────────────────────────────────

class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.model      = IsolationForest(contamination=contamination,
                                          n_estimators=n_estimators,
                                          random_state=42, n_jobs=-1)
        self.scaler     = StandardScaler()
        self.is_trained = False

    def fit(self, feat_df: pd.DataFrame) -> "IsolationForestDetector":
        X = self.scaler.fit_transform(feat_df.values)
        self.model.fit(X)
        self.is_trained = True
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Returns 1 = normal, -1 = anomaly."""
        X = self.scaler.transform(feat_df.values)
        return self.model.predict(X)

    def score(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Anomaly score (lower = more anomalous)."""
        X = self.scaler.transform(feat_df.values)
        return self.model.decision_function(X)


class StatisticalDetector:
    """Z-score + IQR based anomaly detection."""

    def __init__(self, zscore_thresh: float = 3.0, iqr_mult: float = 3.0):
        self.z_thresh = zscore_thresh
        self.iqr_mult = iqr_mult

    def detect(self, series: pd.Series) -> pd.Series:
        """Return boolean mask: True = anomaly."""
        s    = series.values
        mean = np.mean(s)
        std  = np.std(s) + 1e-8
        z    = np.abs((s - mean) / std)

        q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
        iqr    = q3 - q1
        lower  = q1 - self.iqr_mult * iqr
        upper  = q3 + self.iqr_mult * iqr

        return pd.Series((z > self.z_thresh) | (s < lower) | (s > upper),
                          index=series.index)


# ─────────────────────────── combined pipeline ───────────────────────────────

class AnomalyDetectionPipeline:
    def __init__(self, contamination: float = 0.05):
        self.if_detector   = IsolationForestDetector(contamination)
        self.stat_detector = StatisticalDetector()
        self.results       = {}

    def run(self, df: pd.DataFrame,
            target_col: str = "sales",
            sku_col: str = "sku_id") -> pd.DataFrame:
        """Run anomaly detection on all SKUs. Returns annotated DataFrame."""
        all_results = []

        for sku_id, grp in df.groupby(sku_col):
            grp = grp.sort_values("date").copy()

            # statistical
            stat_mask = self.stat_detector.detect(grp[target_col])
            grp["stat_anomaly"] = stat_mask.values

            # isolation forest
            try:
                feat = build_anomaly_features(grp, target_col)
                self.if_detector.fit(feat)
                if_labels = self.if_detector.predict(feat)
                if_scores = self.if_detector.score(feat)
                grp["if_anomaly"] = (if_labels == -1)
                grp["if_score"]   = if_scores
            except Exception as e:
                log.warning(f"IF failed for SKU {sku_id}: {e}")
                grp["if_anomaly"] = False
                grp["if_score"]   = 0.0

            grp["is_anomaly"] = grp["stat_anomaly"] | grp["if_anomaly"]
            grp["sku_id"]     = sku_id
            all_results.append(grp)

        result_df = pd.concat(all_results, ignore_index=True)

        summary = {
            "total_points":    len(result_df),
            "anomalies_found": int(result_df["is_anomaly"].sum()),
            "anomaly_rate":    round(result_df["is_anomaly"].mean() * 100, 2),
            "skus_with_anomalies": int(result_df.groupby("sku_id")["is_anomaly"].any().sum()),
        }
        self.results = summary
        log.info(f"Anomaly detection complete: {summary}")
        return result_df

    def get_anomaly_report(self, result_df: pd.DataFrame) -> Dict:
        """Summarise anomalies by SKU."""
        report = {}
        for sku_id, grp in result_df.groupby("sku_id"):
            anomalies = grp[grp["is_anomaly"]]
            if len(anomalies) > 0:
                report[str(sku_id)] = {
                    "n_anomalies":  len(anomalies),
                    "dates":        anomalies["date"].dt.strftime("%Y-%m-%d").tolist(),
                    "values":       anomalies["sales"].tolist(),
                    "types":        {
                        "statistical": int(anomalies["stat_anomaly"].sum()),
                        "isolation_forest": int(anomalies["if_anomaly"].sum()),
                    }
                }
        return report


# ─────────────────────────── supply disruption ───────────────────────────────

def detect_supply_disruptions(df: pd.DataFrame,
                               zero_run_threshold: int = 3) -> Dict:
    """Detect extended zero-sales periods (potential stockout / disruption)."""
    disruptions = {}
    for sku_id, grp in df.groupby("sku_id"):
        grp    = grp.sort_values("date")
        sales  = grp["sales"].values
        dates  = grp["date"].values
        runs   = []
        count  = 0
        start  = None
        for i, v in enumerate(sales):
            if v == 0:
                if count == 0:
                    start = dates[i]
                count += 1
            else:
                if count >= zero_run_threshold:
                    runs.append({"start": str(start), "length_days": count})
                count = 0
        if runs:
            disruptions[str(sku_id)] = runs
    return disruptions


# ─────────────────────────── main ────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, ".")
    from src.preprocessing import preprocess

    df, meta = preprocess("M5", max_skus=5)
    pipeline  = AnomalyDetectionPipeline()
    result_df = pipeline.run(df)

    print("\nAnomaly Summary:")
    print(json.dumps(pipeline.results, indent=2))

    report = pipeline.get_anomaly_report(result_df)
    print("\nPer-SKU Report:")
    print(json.dumps({k: v for k, v in list(report.items())[:2]}, indent=2))

    disruptions = detect_supply_disruptions(df)
    print(f"\nSupply Disruptions (≥3 zero-sales days): {len(disruptions)} SKUs affected")
