"""
src/train_lstm.py
LSTM (PyTorch, CPU-friendly) for multi-horizon demand forecasting.
Includes exogenous variables: promotions, holidays, external features.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from src.preprocessing import preprocess
from src.feature_engineering import build_features, get_feature_columns
from src.evaluation import evaluate, save_report, print_metrics, wmape_improvement


# ─────────────────────────── dataset ────────────────────────────────────────

EXOG_COLS = ["promo", "is_holiday", "day_of_week", "month", "is_weekend",
             "web_trend", "macro_cpi", "price_change_pct", "sin_week", "cos_week"]


class DemandDataset(Dataset):
    def __init__(self, sequences: List[Tuple[np.ndarray, float]]):
        self.data = sequences

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor([y])


def make_sequences(df_sku: pd.DataFrame,
                   seq_len: int = 30,
                   horizon: int = 7,
                   feature_cols: List[str] = None) -> List:
    """Create (input_window, future_target) pairs for one SKU."""
    if feature_cols is None:
        feature_cols = EXOG_COLS
    valid_cols = [c for c in feature_cols if c in df_sku.columns]

    sales  = df_sku["sales"].values.astype(np.float32)
    # normalise sales per SKU (min-max)
    s_min, s_max = sales.min(), sales.max()
    sales_n = (sales - s_min) / (s_max - s_min + 1e-8)

    feats = df_sku[valid_cols].values.astype(np.float32)
    # simple z-score normalise features
    f_mean, f_std = feats.mean(0), feats.std(0) + 1e-8
    feats = (feats - f_mean) / f_std

    sequences = []
    for i in range(len(sales_n) - seq_len - horizon + 1):
        x_sales = sales_n[i : i + seq_len].reshape(-1, 1)
        x_feats = feats[i : i + seq_len]
        x = np.concatenate([x_sales, x_feats], axis=1)
        y = sales_n[i + seq_len + horizon - 1]           # predict horizon steps ahead
        sequences.append((x, y))
    return sequences, s_min, s_max


# ─────────────────────────── model ──────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])          # last time step


# ─────────────────────────── training loop ──────────────────────────────────

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 20,
                lr: float = 1e-3) -> List[float]:
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    crit      = nn.MSELoss()
    history   = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(crit(model(xb), yb).item())

        avg_t, avg_v = np.mean(losses), np.mean(val_losses)
        scheduler.step(avg_v)
        history.append(avg_v)
        if epoch % 5 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:3d}/{epochs}  train={avg_t:.4f}  val={avg_v:.4f}")

    return history


# ─────────────────────────── predict + confidence ───────────────────────────

def mc_dropout_predict(model: nn.Module,
                       x: torch.Tensor,
                       n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo dropout for uncertainty estimation."""
    model.train()   # keep dropout on
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x.unsqueeze(0)).item())
    preds = np.array(preds)
    return preds.mean(), np.percentile(preds, 2.5), np.percentile(preds, 97.5)


# ─────────────────────────── main pipeline ──────────────────────────────────

def train_dataset(dataset: str,
                  horizon: int = 14,
                  seq_len: int = 30,
                  max_skus: int = 5,
                  epochs: int = 15,
                  batch_size: int = 32,
                  model_dir: str = None) -> Dict:

    model_dir = model_dir or f"models/{dataset.lower()}/lstm"
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cpu")

    log.info(f"=== LSTM Training — {dataset} (CPU) ===")
    df, meta = preprocess(dataset, max_skus=max_skus)
    df = build_features(df, dataset=dataset)

    all_metrics = []

    for sku_id in df["sku_id"].unique():
        log.info(f"  SKU: {sku_id}")
        df_sku = df[df["sku_id"] == sku_id].copy().sort_values("date").reset_index(drop=True)

        if len(df_sku) < seq_len + horizon + 20:
            log.warning("    Too short – skipping")
            continue

        seqs, s_min, s_max = make_sequences(df_sku, seq_len, horizon)
        if len(seqs) < 10:
            continue

        split = int(0.8 * len(seqs))
        train_ds = DemandDataset(seqs[:split])
        val_ds   = DemandDataset(seqs[split:])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size)

        input_size = 1 + len([c for c in EXOG_COLS if c in df_sku.columns])
        model = LSTMForecaster(input_size=input_size).to(device)
        train_model(model, train_dl, val_dl, epochs=epochs)

        # ── evaluate on val set ──
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb).numpy().flatten()
                y_pred.extend(preds)
                y_true.extend(yb.numpy().flatten())

        # denormalise
        y_true = np.array(y_true) * (s_max - s_min) + s_min
        y_pred = np.clip(np.array(y_pred) * (s_max - s_min) + s_min, 0, None)
        lower  = y_pred * 0.8
        upper  = y_pred * 1.2

        m = evaluate(y_true, y_pred, lower, upper, model_name=f"LSTM_{sku_id}")
        all_metrics.append(m)
        print_metrics(m)

        # ── save ──
        safe_sku = str(sku_id).replace("/", "_")
        torch.save({
            "state_dict": model.state_dict(),
            "input_size": input_size,
            "s_min": s_min, "s_max": s_max,
            "seq_len": seq_len, "horizon": horizon,
        }, f"{model_dir}/{safe_sku}.pt")

    avg_wmape = np.mean([m["WMAPE"] for m in all_metrics]) if all_metrics else 99.0
    summary = {
        "dataset": dataset, "model": "LSTM",
        "n_skus": len(all_metrics),
        "avg_WMAPE": round(avg_wmape, 4),
    }
    save_report(summary, dataset, "lstm")
    log.info(f"LSTM avg WMAPE = {avg_wmape:.2f}%")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="M5", choices=["M5","Favorita","UCI"])
    p.add_argument("--horizon",   type=int, default=14)
    p.add_argument("--max_skus",  type=int, default=3)
    p.add_argument("--epochs",    type=int, default=10)
    args = p.parse_args()
    result = train_dataset(args.dataset, args.horizon, max_skus=args.max_skus, epochs=args.epochs)
    print(json.dumps(result, indent=2))
