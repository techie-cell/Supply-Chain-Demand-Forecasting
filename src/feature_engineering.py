"""
src/feature_engineering.py
Feature engineering for demand forecasting models.
Produces lag features, rolling statistics, date features,
holiday indicators, and exogenous variables.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


# ─────────────────────────── date / calendar ────────────────────────────────

def add_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = pd.to_datetime(df[date_col])
    df["day_of_week"]    = d.dt.dayofweek          # 0=Mon
    df["day_of_month"]   = d.dt.day
    df["week_of_year"]   = d.dt.isocalendar().week.astype(int)
    df["month"]          = d.dt.month
    df["quarter"]        = d.dt.quarter
    df["year"]           = d.dt.year
    df["is_weekend"]     = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"]   = d.dt.is_month_end.astype(int)
    # Fourier terms for weekly & annual seasonality
    t = np.arange(len(df))
    df["sin_week"]  = np.sin(2 * np.pi * t / 7)
    df["cos_week"]  = np.cos(2 * np.pi * t / 7)
    df["sin_year"]  = np.sin(2 * np.pi * t / 365.25)
    df["cos_year"]  = np.cos(2 * np.pi * t / 365.25)
    return df


# ─────────────────────────── lag features ───────────────────────────────────

def add_lag_features(df: pd.DataFrame,
                     target_col: str = "sales",
                     group_cols: List[str] = ["sku_id", "store_id"],
                     lags: List[int] = [1, 7, 14, 21, 28]) -> pd.DataFrame:
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    return df


# ─────────────────────────── rolling stats ──────────────────────────────────

def add_rolling_features(df: pd.DataFrame,
                         target_col: str = "sales",
                         group_cols: List[str] = ["sku_id", "store_id"],
                         windows: List[int] = [7, 14, 28]) -> pd.DataFrame:
    for w in windows:
        shifted = df.groupby(group_cols)[target_col].shift(1)
        rolled  = shifted.groupby(df.groupby(group_cols).ngroup())
        df[f"roll_mean_{w}"]  = (df.groupby(group_cols)[target_col]
                                   .shift(1)
                                   .groupby(df[group_cols].apply(tuple, axis=1))
                                   .transform(lambda s: s.rolling(w, min_periods=1).mean()))
        df[f"roll_std_{w}"]   = (df.groupby(group_cols)[target_col]
                                   .shift(1)
                                   .groupby(df[group_cols].apply(tuple, axis=1))
                                   .transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0)))
        df[f"roll_max_{w}"]   = (df.groupby(group_cols)[target_col]
                                   .shift(1)
                                   .groupby(df[group_cols].apply(tuple, axis=1))
                                   .transform(lambda s: s.rolling(w, min_periods=1).max()))
    return df


def add_rolling_features_safe(df: pd.DataFrame,
                               target_col: str = "sales",
                               group_cols: List[str] = ["sku_id", "store_id"],
                               windows: List[int] = [7, 14, 28]) -> pd.DataFrame:
    """Memory-efficient version using apply per group."""
    results = []
    for keys, grp in df.groupby(group_cols):
        s = grp[target_col].copy()
        for w in windows:
            grp[f"roll_mean_{w}"] = s.shift(1).rolling(w, min_periods=1).mean()
            grp[f"roll_std_{w}"]  = s.shift(1).rolling(w, min_periods=1).std().fillna(0)
            grp[f"roll_max_{w}"]  = s.shift(1).rolling(w, min_periods=1).max()
        results.append(grp)
    return pd.concat(results).sort_index()


# ─────────────────────────── promo / price ──────────────────────────────────

def add_promo_features(df: pd.DataFrame,
                       promo_col: str = "promo",
                       price_col: str = "price") -> pd.DataFrame:
    if promo_col in df.columns:
        df["promo_next_7"]  = df[promo_col].shift(-7).fillna(0)
        df["promo_rolling"] = df[promo_col].rolling(7, min_periods=1).sum()
    if price_col in df.columns:
        df["price_change_pct"] = df[price_col].pct_change().fillna(0)
        df["price_vs_mean"]    = df[price_col] / df[price_col].rolling(28, min_periods=1).mean()
    return df


# ─────────────────────────── static (SKU) features ──────────────────────────

def add_static_features(df: pd.DataFrame,
                        group_cols: List[str] = ["sku_id", "store_id"]) -> pd.DataFrame:
    """Encode SKU / store IDs as integers for embedding models."""
    for col in group_cols:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].astype("category").cat.codes
    return df


# ─────────────────────────── target transforms ──────────────────────────────

def log_transform(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    df[f"{target_col}_log"] = np.log1p(df[target_col])
    return df


def add_demand_statistics(df: pd.DataFrame,
                           target_col: str = "sales",
                           group_cols: List[str] = ["sku_id"]) -> pd.DataFrame:
    """Global SKU-level statistics (static over time)."""
    stats = df.groupby(group_cols)[target_col].agg(
        sku_mean="mean", sku_std="std", sku_max="max", sku_median="median"
    ).reset_index()
    stats["sku_cv"] = stats["sku_std"] / (stats["sku_mean"] + 1e-8)  # coeff variation
    return df.merge(stats, on=group_cols, how="left")


# ─────────────────────────── full pipeline ──────────────────────────────────

def build_features(df: pd.DataFrame,
                   dataset: str = "M5",
                   lags: List[int] = [1, 7, 14, 28],
                   rolling_windows: List[int] = [7, 14, 28],
                   log_target: bool = False) -> pd.DataFrame:
    """
    Master feature engineering pipeline.
    Returns feature-enriched DataFrame; NaN rows (from lags) are dropped.
    """
    group_cols = ["sku_id", "store_id"]
    df = df.copy()

    df = add_date_features(df)
    df = add_lag_features(df, lags=lags, group_cols=group_cols)
    df = add_rolling_features_safe(df, windows=rolling_windows, group_cols=group_cols)
    df = add_promo_features(df)
    df = add_demand_statistics(df)
    df = add_static_features(df, group_cols=group_cols)
    if log_target:
        df = log_transform(df)

    # drop rows where lags are NaN
    max_lag = max(lags)
    df = df.dropna(subset=[f"lag_{max_lag}"])
    df.reset_index(drop=True, inplace=True)
    return df


def get_feature_columns(lags=[1,7,14,28], windows=[7,14,28]) -> List[str]:
    """Return the list of feature column names (for model input)."""
    cols  = ["day_of_week","day_of_month","week_of_year","month","quarter","year",
             "is_weekend","is_month_start","is_month_end",
             "sin_week","cos_week","sin_year","cos_year","is_holiday",
             "web_trend","macro_gdp_idx","macro_cpi","competitor_price","inventory_level",
             "promo","price","price_change_pct","price_vs_mean",
             "promo_next_7","promo_rolling",
             "sku_mean","sku_std","sku_max","sku_median","sku_cv",
             "sku_id_enc","store_id_enc"]
    cols += [f"lag_{l}"         for l in lags]
    cols += [f"roll_mean_{w}"   for w in windows]
    cols += [f"roll_std_{w}"    for w in windows]
    cols += [f"roll_max_{w}"    for w in windows]
    return cols


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import preprocess

    df, meta = preprocess("M5", max_skus=5)
    df_fe = build_features(df)
    print(df_fe.shape)
    print(df_fe.columns.tolist())
    print(df_fe.head(3))
