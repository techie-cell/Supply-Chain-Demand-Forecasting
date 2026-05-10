# AI Supply Chain Demand Forecasting

## Overview

This project is an enterprise-grade AI-powered demand forecasting system developed for FMCG and retail supply chain optimization.

The system uses Temporal Fusion Transformer (TFT), Deep Learning, Explainable AI, and MLOps concepts to generate accurate multi-horizon forecasts across multiple datasets.

---

# Business Problem

Large retailers experience:
- Stockouts on high-demand products
- Excess inventory on slow-moving items
- Poor forecast accuracy using traditional methods

This system solves these challenges using advanced AI forecasting techniques.

---

# Features

- Multi-horizon forecasting
- Temporal Fusion Transformer (TFT)
- LSTM forecasting
- XGBoost forecasting
- ARIMA/SARIMA baseline models
- Multi-dataset support
- Explainable AI with SHAP
- Forecast visualization dashboard
- FastAPI deployment
- Automated retraining pipeline
- Drift detection
- Anomaly detection
- Multi-task forecasting support

---

# Datasets Used

## 1. M5 Forecasting Dataset
- Walmart sales forecasting
- 42,000+ SKU demand series

## 2. Favorita Grocery Sales Dataset
- Ecuador retail sales dataset
- Promotion and holiday analysis

## 3. UCI Time Series Datasets
- Benchmark forecasting datasets

---

# Tech Stack

- Python
- PyTorch
- PyTorch Forecasting
- Temporal Fusion Transformer
- SHAP
- FastAPI
- Streamlit
- XGBoost
- Docker

---

# Project Structure

```bash
demand_forecasting_project/
│
├── api/
├── dashboard/
├── data/
├── logs/
├── models/
├── reports/
├── sota/
├── src/
├── requirements.txt
├── TRAIN_MODELS.bat
├── START_API.bat
├── START_DASHBOARD.bat
└── README.md

## Forecasting Workflow

## Data Collection
- Sales history
- Promotions
- Holidays
- Inventory data
- External indicators

## Data Preprocessing
- Missing value handling
- Outlier detection
- Time indexing

## Feature Engineering
- Lag features
- Rolling averages
- Moving statistics
- Holiday indicators
- Promotional features
- Seasonality encoding
- Trend features

## Baseline Models
- ARIMA
- SARIMA

## Intermediate Models
- LSTM
- XGBoost

## Advanced Models
- Temporal Fusion Transformer (TFT)

## Explainability
- SHAP analysis
- Attention visualization

## Deployment
- FastAPI backend
- Streamlit dashboard

---

# Model Performance

Target achieved:
- 15–20% WMAPE reduction compared to baseline forecasting

---

# Dashboard Features

- Dataset selection
- Forecast visualization
- SKU-level analysis
- Sales trends
- Prediction confidence intervals
- Model comparison charts
- Forecast accuracy metrics

---

# API Endpoints

## Generate Forecast

```http
POST /forecast
```

## Health Check

```http
GET /health
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/techie-cell/Supply-Chain-Demand-Forecasting.git
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment (Windows)

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train Models

```bash
python src/train_tft.py
```

---

# Run API

```bash
uvicorn api.app:app --reload
```

---

# Run Dashboard

```bash
streamlit run dashboard/app_streamlit.py
```

---

# Future Enhancements

- Multi-task TFT
- Graph Neural Networks
- LLM integration
- Social media demand signals
- Reinforcement Learning optimization
- AWS deployment
- Real-time retraining

---

# Industry Applications

- Retail & FMCG
- Manufacturing
- Logistics
- E-commerce
- Pharmaceuticals
├── START_DASHBOARD.bat
└── README.md
