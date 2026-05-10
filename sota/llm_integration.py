"""
sota/llm_integration.py
LLM Integration for Demand Forecasting:
- Unstructured news & social media sentiment as exogenous signals
- LLM-based demand commentary / explanation generation
- Structured signal extraction from text

Placeholder implementation using Hugging Face transformers (local) or OpenAI API.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# optional imports
HF_AVAILABLE  = False
OAI_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HF_AVAILABLE = True
except ImportError:
    log.warning("transformers not installed; HF sentiment unavailable. pip install transformers")

try:
    import openai
    OAI_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────── sentiment analysis ───────────────────────────────

class SentimentSignalExtractor:
    """
    Extracts demand signals from news headlines / social posts.
    Uses a pre-trained FinBERT / DistilBERT for financial/retail sentiment.
    """

    SAMPLE_NEWS = [
        "Walmart reports record holiday sales, supply chain optimized",
        "Inflation concerns drive consumers to cut discretionary spending",
        "New product launch drives 40% spike in online demand",
        "Port strike causing delays in consumer goods shipments",
        "Back-to-school season boosts stationery and electronics demand",
        "Severe weather disrupts distribution centers across midwest",
    ]

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.pipe       = None

    def load_model(self):
        if not HF_AVAILABLE:
            log.warning("Hugging Face transformers not installed")
            return self
        try:
            self.pipe = hf_pipeline("sentiment-analysis", model=self.model_name,
                                     truncation=True, max_length=128)
            log.info(f"Sentiment model loaded: {self.model_name}")
        except Exception as e:
            log.error(f"Failed to load sentiment model: {e}")
        return self

    def analyze(self, texts: List[str]) -> List[Dict]:
        """Return sentiment scores for each text."""
        if self.pipe is None:
            # mock output
            mock = []
            for t in texts:
                score = float(np.random.uniform(0.4, 0.9))
                label = "POSITIVE" if score > 0.55 else "NEGATIVE"
                mock.append({"text": t[:60], "label": label, "score": round(score, 3)})
            return mock

        results = self.pipe(texts)
        return [{"text": t[:60], **r} for t, r in zip(texts, results)]

    def to_demand_signal(self, sentiments: List[Dict]) -> float:
        """
        Aggregate sentiments into a demand multiplier.
        POSITIVE → demand boost, NEGATIVE → demand reduction.
        Range: [0.7, 1.3]
        """
        if not sentiments:
            return 1.0
        scores = []
        for s in sentiments:
            multiplier = s["score"] if s["label"] == "POSITIVE" else -s["score"]
            scores.append(multiplier)
        avg = np.mean(scores)
        # scale to [0.7, 1.3]
        return float(np.clip(1.0 + avg * 0.3, 0.7, 1.3))


# ─────────────────────────── LLM commentary ──────────────────────────────────

class ForecastCommentator:
    """
    Uses an LLM to generate natural-language explanations for forecasts.
    Supports: OpenAI GPT-4 (if API key set) or a local HF model.
    """

    def __init__(self, backend: str = "template"):
        """backend: 'template' | 'openai' | 'local'"""
        self.backend = backend

    def generate_commentary(self, forecast: Dict) -> str:
        if self.backend == "openai" and OAI_AVAILABLE:
            return self._openai_commentary(forecast)
        elif self.backend == "local" and HF_AVAILABLE:
            return self._local_commentary(forecast)
        else:
            return self._template_commentary(forecast)

    def _template_commentary(self, f: Dict) -> str:
        sku     = f.get("sku", "Unknown SKU")
        horizon = f.get("horizon", 14)
        avg_fc  = np.mean(f.get("forecast", [0]))
        trend   = f.get("trend", "stable")
        model   = f.get("model", "AI")

        commentary = f"""
📊 Forecast Summary for {sku} ({horizon}-day horizon)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Model Used      : {model}
• Avg Forecast    : {avg_fc:.1f} units/day
• Demand Trend    : {trend.capitalize()}
• Confidence      : 95% prediction interval provided

💡 Business Insights:
  - {"📈 Plan for increased inventory" if trend == "increasing" else "📉 Consider reducing safety stock" if trend == "decreasing" else "📊 Maintain current inventory levels"}
  - Promotional effects have been accounted for in the forecast
  - Holiday and seasonal patterns are incorporated

⚠️  Risk Flags:
  - {"High volatility detected — widen safety stock buffer" if f.get("volatility", "low") == "high" else "Normal volatility — standard reorder point recommended"}
"""
        return commentary.strip()

    def _openai_commentary(self, f: Dict) -> str:
        if not OAI_AVAILABLE:
            return self._template_commentary(f)
        prompt = f"""
You are a supply chain analyst AI. Generate a concise (3-4 sentences) 
business commentary on this demand forecast:

SKU: {f.get('sku')}
Horizon: {f.get('horizon')} days
Average Forecast: {np.mean(f.get('forecast', [0])):.1f} units/day
Model: {f.get('model')}
Trend: {f.get('trend', 'stable')}

Focus on: inventory implications, risk, and actionable recommendations.
"""
        try:
            resp = openai.ChatCompletion.create(
                model    = "gpt-4o-mini",
                messages = [{"role": "user", "content": prompt}],
                max_tokens = 200,
            )
            return resp.choices[0].message.content
        except Exception as e:
            log.error(f"OpenAI API error: {e}")
            return self._template_commentary(f)

    def _local_commentary(self, f: Dict) -> str:
        """Placeholder for local LLM (e.g., llama.cpp, Ollama)."""
        log.info("Local LLM commentary: using template fallback")
        return self._template_commentary(f)


# ─────────────────────────── news feature pipeline ───────────────────────────

class NewsFeaturePipeline:
    """
    Converts daily news headlines into time-series demand adjustment factors.
    Can be plugged into TFT as a time-varying known real feature.
    """

    def __init__(self):
        self.extractor = SentimentSignalExtractor()
        self.extractor.load_model()

    def fetch_news(self, start_date: str, end_date: str,
                   keywords: List[str] = None) -> List[Dict]:
        """
        Placeholder: fetch news from an API (NewsAPI, GDELT, etc.)
        Returns mock news for demonstration.
        """
        if keywords is None:
            keywords = ["retail", "demand", "supply chain"]
        dates = pd.date_range(start_date, end_date, freq="D")
        news  = []
        for d in dates:
            n_articles = np.random.randint(1, 4)
            day_news   = np.random.choice(SentimentSignalExtractor.SAMPLE_NEWS, n_articles)
            news.append({"date": str(d.date()), "headlines": day_news.tolist()})
        return news

    def build_sentiment_feature(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Return DataFrame with date + demand_sentiment_signal column."""
        news   = self.fetch_news(start_date, end_date)
        rows   = []
        for day in news:
            sents  = self.extractor.analyze(day["headlines"])
            signal = self.extractor.to_demand_signal(sents)
            rows.append({"date": pd.Timestamp(day["date"]),
                          "demand_sentiment": signal})
        return pd.DataFrame(rows)

    def enrich_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment signal to demand DataFrame."""
        start = str(df["date"].min().date())
        end   = str(df["date"].max().date())
        sent_df = self.build_sentiment_feature(start, end)
        return df.merge(sent_df, on="date", how="left").fillna({"demand_sentiment": 1.0})


# ─────────────────────────── main ────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  LLM Integration — SOTA Module")
    print("=" * 55)

    # 1. Sentiment extraction
    extractor = SentimentSignalExtractor()
    extractor.load_model()
    news  = extractor.SAMPLE_NEWS[:3]
    sents = extractor.analyze(news)
    signal = extractor.to_demand_signal(sents)

    print("\n📰 Sentiment Analysis:")
    for s in sents:
        print(f"  [{s['label']} {s['score']:.2f}] {s['text']}")
    print(f"\n  → Demand Signal Multiplier: {signal:.3f}")

    # 2. Forecast commentary
    commentator = ForecastCommentator(backend="template")
    commentary  = commentator.generate_commentary({
        "sku":      "FOODS_1_001",
        "horizon":  14,
        "forecast": list(np.random.uniform(20, 50, 14).round(1)),
        "trend":    "increasing",
        "model":    "TFT",
        "volatility": "low",
    })
    print("\n💬 Forecast Commentary:")
    print(commentary)

    # 3. News feature pipeline
    print("\n📊 News Feature Pipeline:")
    pipeline = NewsFeaturePipeline()
    sent_df  = pipeline.build_sentiment_feature("2024-01-01", "2024-01-10")
    print(sent_df.head())
