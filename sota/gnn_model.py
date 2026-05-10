"""
sota/gnn_model.py
Graph Neural Network (GNN) for inter-product demand relationships.
Captures substitutes and complements via learned graph edges.

SOTA STRUCTURE — requires: torch-geometric (pip install torch-geometric)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
log = logging.getLogger(__name__)

# Optional torch-geometric import
try:
    from torch_geometric.data import Data, DataLoader as GeoDataLoader
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False
    log.warning("torch-geometric not installed. Install: pip install torch-geometric")


# ─────────────────────────── graph construction ───────────────────────────────

def build_correlation_graph(df: pd.DataFrame,
                             sku_col: str = "sku_id",
                             target_col: str = "sales",
                             corr_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build adjacency from Pearson correlation between SKU sales time series.
    Returns (edge_index [2, E], edge_weight [E]).
    """
    pivot  = df.pivot_table(index="date", columns=sku_col, values=target_col).fillna(0)
    corr   = pivot.corr().values
    skus   = pivot.columns.tolist()
    n      = len(skus)

    src, dst, weights = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr[i, j]) >= corr_threshold:
                src.append(i)
                dst.append(j)
                weights.append(corr[i, j])

    if not src:
        # fully connected fallback
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i); dst.append(j); weights.append(1.0)

    edge_index  = np.array([src, dst], dtype=np.int64)
    edge_weight = np.array(weights,    dtype=np.float32)
    return edge_index, edge_weight, skus


def build_sku_node_features(df: pd.DataFrame,
                              sku_col: str = "sku_id",
                              target_col: str = "sales") -> np.ndarray:
    """Build node feature matrix [n_skus, n_features]."""
    stats = df.groupby(sku_col)[target_col].agg(
        ["mean","std","min","max",
         lambda s: s.quantile(0.25),
         lambda s: s.quantile(0.75),
         lambda s: s.autocorr(1) if len(s) > 1 else 0]
    ).fillna(0)
    return stats.values.astype(np.float32)


# ─────────────────────────── GNN models ──────────────────────────────────────

class GATDemandForecaster(nn.Module):
    """Graph Attention Network for demand forecasting."""

    def __init__(self, in_channels: int, hidden: int = 64,
                 heads: int = 4, out_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        if not GEO_AVAILABLE:
            raise ImportError("torch-geometric required")
        self.gat1   = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.gat2   = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.lstm   = nn.LSTM(hidden, hidden // 2, batch_first=True)
        self.fc     = nn.Linear(hidden // 2, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # graph convolution
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        # temporal (LSTM over node feature as sequence proxy)
        x_seq, _ = self.lstm(x.unsqueeze(1))
        out = self.fc(x_seq.squeeze(1))
        return out


class GCNDemandForecaster(nn.Module):
    """Simpler GCN alternative for CPU-constrained environments."""

    def __init__(self, in_channels: int, hidden: int = 32, out_channels: int = 1):
        super().__init__()
        if not GEO_AVAILABLE:
            raise ImportError("torch-geometric required")
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc    = nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x)


# ─────────────────────────── training stub ───────────────────────────────────

class GNNDemandPipeline:
    """
    Pipeline wrapper: graph construction → training → inference.
    Training is a stub for structural completeness (requires torch-geometric).
    """

    def __init__(self, model_type: str = "gcn"):
        self.model_type  = model_type
        self.model       = None
        self.skus        = []
        self.edge_index  = None

    def build_graph(self, df: pd.DataFrame) -> "GNNDemandPipeline":
        self.edge_index, self.edge_weight, self.skus = build_correlation_graph(df)
        self.node_features = build_sku_node_features(df)
        log.info(f"Graph built: {len(self.skus)} nodes, {self.edge_index.shape[1]} edges")
        return self

    def build_model(self) -> "GNNDemandPipeline":
        if not GEO_AVAILABLE:
            log.error("Install torch-geometric to use GNN models")
            return self
        in_ch = self.node_features.shape[1]
        if self.model_type == "gat":
            self.model = GATDemandForecaster(in_ch)
        else:
            self.model = GCNDemandForecaster(in_ch)
        log.info(f"GNN model ({self.model_type}) built: {sum(p.numel() for p in self.model.parameters()):,} params")
        return self

    def train(self, df: pd.DataFrame, epochs: int = 20, lr: float = 1e-3):
        """Training stub — implement full temporal GNN training here."""
        if not GEO_AVAILABLE or self.model is None:
            log.warning("GNN training skipped (torch-geometric not available)")
            return {"status": "skipped", "reason": "torch-geometric not installed"}

        x    = torch.FloatTensor(self.node_features)
        edge = torch.LongTensor(self.edge_index)
        opt  = torch.optim.Adam(self.model.parameters(), lr=lr)
        crit = nn.MSELoss()

        # dummy target: SKU mean sales (replace with actual horizon targets)
        y = torch.FloatTensor(self.node_features[:, 0:1])

        self.model.train()
        losses = []
        for ep in range(1, epochs + 1):
            opt.zero_grad()
            pred = self.model(x, edge)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if ep % 5 == 0:
                log.info(f"  GNN epoch {ep}/{epochs}  loss={loss.item():.4f}")

        return {"epochs": epochs, "final_loss": losses[-1]}

    def predict(self, sku_id: str) -> Dict:
        """Placeholder inference."""
        if not GEO_AVAILABLE or self.model is None:
            return {"forecast": list(np.random.uniform(10, 100, 14).round(1)),
                    "note": "mock – torch-geometric not available"}
        self.model.eval()
        with torch.no_grad():
            x    = torch.FloatTensor(self.node_features)
            edge = torch.LongTensor(self.edge_index)
            pred = self.model(x, edge)
        if sku_id in self.skus:
            idx = self.skus.index(sku_id)
            return {"sku": sku_id, "forecast_score": float(pred[idx, 0])}
        return {"error": "SKU not in graph"}


# ─────────────────────────── main ────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, ".")
    from src.preprocessing import preprocess

    print("=" * 55)
    print("  GNN Demand Forecasting — SOTA Module")
    print("=" * 55)

    df, _ = preprocess("M5", max_skus=5)

    pipeline = GNNDemandPipeline(model_type="gcn")
    pipeline.build_graph(df)
    pipeline.build_model()

    result = pipeline.train(df, epochs=10)
    print(f"\nTraining result: {result}")

    pred = pipeline.predict(df["sku_id"].iloc[0])
    print(f"Sample prediction: {pred}")
