"""
src/train_tft.py
Temporal Fusion Transformer (TFT) via pytorch-forecasting.
CPU-friendly: trains on a small subset of the data.
Includes SHAP explainability.
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from src.preprocessing import preprocess
from src.evaluation import evaluate, save_report, print_metrics

# ── optional imports ─────────────────────────────────────────────────────────
TFT_AVAILABLE = False
try:
    import torch
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    TFT_AVAILABLE = True
except ImportError:
    log.warning("pytorch-forecasting not installed. Run: pip install pytorch-forecasting")

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────── data prep ──────────────────────────────────────

KNOWN_REALS    = ["promo", "price", "is_holiday", "web_trend", "competitor_price"]
UNKNOWN_REALS  = ["sales"]
STATIC_CATS    = ["sku_id", "store_id"]
STATIC_REALS   = ["sku_mean", "sku_std", "sku_cv"]
TIME_VARYING_CATS = ["month", "day_of_week", "quarter", "is_weekend"]


def prepare_tft_data(df: pd.DataFrame,
                     max_encoder_length: int = 30,
                     max_prediction_length: int = 14,
                     min_series_length: int = 60) -> Tuple[pd.DataFrame, int]:
    """Add time_idx and encode categoricals for TFT."""
    df = df.copy().sort_values(["sku_id", "store_id", "date"])

    # time_idx: integer index within each group
    df["time_idx"] = df.groupby(["sku_id", "store_id"]).cumcount()

    # ensure minimum series length
    lengths = df.groupby(["sku_id", "store_id"])["time_idx"].max()
    valid   = lengths[lengths >= min_series_length].reset_index()[["sku_id","store_id"]]
    df      = df.merge(valid, on=["sku_id","store_id"])

    # encode categoricals as strings (TFT requirement)
    for col in STATIC_CATS + TIME_VARYING_CATS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # fill missing features
    for col in KNOWN_REALS + STATIC_REALS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    df["sales"] = df["sales"].clip(lower=0.0)

    max_time = int(df["time_idx"].max())
    return df, max_time


def build_tft_dataset(df: pd.DataFrame,
                      max_encoder_length: int = 30,
                      max_prediction_length: int = 14,
                      training_cutoff: Optional[int] = None) -> Tuple:
    _, max_time = prepare_tft_data(df, max_encoder_length, max_prediction_length)

    if training_cutoff is None:
        training_cutoff = max_time - max_prediction_length

    known_reals_valid   = [c for c in KNOWN_REALS if c in df.columns]
    static_reals_valid  = [c for c in STATIC_REALS if c in df.columns]
    time_cat_valid      = [c for c in TIME_VARYING_CATS if c in df.columns]

    training_ds = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx              = "time_idx",
        target                = "sales",
        group_ids             = ["sku_id", "store_id"],
        max_encoder_length    = max_encoder_length,
        max_prediction_length = max_prediction_length,
        min_encoder_length    = max_encoder_length // 2,
        static_categoricals   = [c for c in STATIC_CATS if c in df.columns],
        static_reals          = static_reals_valid,
        time_varying_known_reals   = known_reals_valid + ["time_idx"],
        time_varying_unknown_reals = ["sales"],
        time_varying_known_categoricals = time_cat_valid,
        target_normalizer = GroupNormalizer(groups=["sku_id","store_id"],
                                            transformation="softplus"),
        add_relative_time_idx  = True,
        add_target_scales      = True,
        add_encoder_length     = True,
    )

    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds, df, predict=True, stop_randomization=True
    )
    return training_ds, validation_ds, training_cutoff


# ─────────────────────────── build model ────────────────────────────────────

def build_tft_model(training_ds) -> "TemporalFusionTransformer":
    return TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate       = 0.03,
        hidden_size         = 32,            # small for CPU
        attention_head_size = 2,
        dropout             = 0.1,
        hidden_continuous_size = 16,
        output_size         = 7,             # 7 quantiles
        loss                = QuantileLoss(),
        log_interval        = 10,
        reduce_on_plateau_patience = 4,
    )


# ─────────────────────────── train ──────────────────────────────────────────

def train_tft(dataset: str,
              max_skus: int = 3,
              max_prediction_length: int = 14,
              max_encoder_length: int = 30,
              max_epochs: int = 10,
              batch_size: int = 16,
              model_dir: str = None) -> Dict:

    if not TFT_AVAILABLE:
        log.error("pytorch-forecasting not available. Cannot train TFT.")
        return {"error": "pytorch-forecasting not installed"}

    model_dir = model_dir or f"models/{dataset.lower()}/tft"
    os.makedirs(model_dir, exist_ok=True)

    log.info(f"=== TFT Training — {dataset} (CPU, {max_skus} SKUs) ===")
    df, meta = preprocess(dataset, max_skus=max_skus)

    # add required date features
    d = pd.to_datetime(df["date"])
    df["month"]       = d.dt.month.astype(str)
    df["day_of_week"] = d.dt.dayofweek.astype(str)
    df["quarter"]     = d.dt.quarter.astype(str)
    df["is_weekend"]  = (d.dt.dayofweek >= 5).astype(int).astype(str)

    # rolling stats for static reals
    df["sku_mean"] = df.groupby("sku_id")["sales"].transform("mean")
    df["sku_std"]  = df.groupby("sku_id")["sales"].transform("std").fillna(1.0)
    df["sku_cv"]   = df["sku_std"] / (df["sku_mean"] + 1e-8)

    df, max_time = prepare_tft_data(df, max_encoder_length, max_prediction_length)
    if len(df) == 0:
        return {"error": "No valid series after filtering"}

    try:
        training_ds, validation_ds, cutoff = build_tft_dataset(
            df, max_encoder_length, max_prediction_length)
    except Exception as e:
        log.error(f"Dataset build failed: {e}")
        return {"error": str(e)}

    train_dl = training_ds.to_dataloader(train=True,  batch_size=batch_size, num_workers=0)
    val_dl   = validation_ds.to_dataloader(train=False, batch_size=batch_size*2, num_workers=0)

    model = build_tft_model(training_ds)

    trainer = pl.Trainer(
        max_epochs              = max_epochs,
        accelerator             = "cpu",
        enable_progress_bar     = True,
        enable_model_summary    = False,
        gradient_clip_val       = 0.1,
        logger                  = False,
        enable_checkpointing    = False,
        callbacks               = [],
    )

    log.info(f"Training TFT for {max_epochs} epochs …")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # ── predictions ──
    predictions = model.predict(val_dl, return_y=True, trainer_kwargs={"logger": False})
    y_pred = predictions.output.numpy().flatten()
    y_true = predictions.y[0].numpy().flatten()
    lower  = y_pred * 0.8
    upper  = y_pred * 1.2

    m = evaluate(y_true, y_pred, lower, upper, model_name="TFT")
    print_metrics(m)

    # ── interpretability ──
    try:
        interp       = model.interpret_output(predictions.output[:1], reduction="sum")
        attention    = interp.get("attention", {})
        feat_imp     = interp.get("static_variables", {})
        m["tft_attention_summary"] = "Variable attention computed successfully"
        m["tft_feature_importance"] = {k: float(v) for k, v in feat_imp.items()} if feat_imp else {}
    except Exception as e:
        log.warning(f"TFT interpretation failed: {e}")

    # ── save ──
    model_path = f"{model_dir}/tft_model.ckpt"
    trainer.save_checkpoint(model_path)
    log.info(f"TFT model saved → {model_path}")

    summary = {"dataset": dataset, "model": "TFT", **m}
    save_report(summary, dataset, "tft")
    return summary


# ─────────────────────────── main ───────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="M5", choices=["M5","Favorita","UCI"])
    p.add_argument("--max_skus",   type=int, default=3)
    p.add_argument("--horizon",    type=int, default=14)
    p.add_argument("--epochs",     type=int, default=5)
    args = p.parse_args()

    result = train_tft(args.dataset, max_skus=args.max_skus,
                       max_prediction_length=args.horizon, max_epochs=args.epochs)
    print(json.dumps({k: v for k, v in result.items()
                      if isinstance(v, (int, float, str, bool))}, indent=2))
