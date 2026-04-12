# USD/INR Close Price Prediction — Quant Strategy

A machine learning pipeline for predicting next-day (T+1) USD/INR closing prices and generating long/short trading signals. Uses daily OHLC data from 2003–2023 with a walk-forward validation framework to simulate live trading conditions.

---

## Project Overview

The project compares four models for directional prediction and price forecasting:

| Model | Type | Directional Accuracy (2021–2023 avg) |
|---|---|---|
| Statistical (GARCH + EMA threshold) | Rule-based | Baseline |
| **Decision Tree ★** | ML Classifier | **74.68%** — best overall |
| KNN | Instance-based | 50.19% |
| LDA + Ridge (Two-Stage) | Linear ML | 51.65% |

All models share a common GARCH(1,1) volatility + rolling std regime classification backbone. The direction model is swapped per experiment while the magnitude prediction formula stays the same.

---

## Features Used (10 total)

| Feature | Description |
|---|---|
| `feat` | EMA₅ − EMA₂₀ (trend spread) |
| `feat_change` | feat_t − feat_{t−1} (trend acceleration) |
| `ret_1d` | 1-day return |
| `ret_3d` | 3-day return |
| `price_vs_ema20` | (Close − EMA₂₀) / EMA₂₀ |
| `vol_ratio` | std₅ / std₂₅ |
| `garch_vol` | GARCH(1,1) conditional volatility σ_t |
| `intraday_body` | (Close − Open) / Open |
| `wick_upper` | (High − Close) / (High − Low + ε) |
| `close_in_range` | (Close − Low) / (High − Low + ε) |

---

## Requirements

```
numpy
pandas
plotly
scikit-learn
scipy
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

Place the raw data file in the project root (already included):

```
USD_INR_Exchange.csv
```

Expected columns: `Date, Open, High, Low, Close`
Period covered: **2003 – 2023** (daily OHLC)
Train split: 2003–2020 | Test split: 2021–2023

---

## Project Structure

```
Final_code_ppt/
│
├── USD_INR_Exchange.csv      # Raw OHLC data
├── README.txt
│
├── stats_model1.py           # Statistical model — Part 1 (GARCH + EMA threshold direction)
├── stats_model1.txt          # Notes / output for stats model 1
├── stats_model2.py           # Statistical model — Part 2 (regime-based magnitude synthesis)
├── stats_model2.txt          # Notes / output for stats model 2
│
├── dec_tree.py               # Decision Tree direction + GARCH magnitude (best model)
├── dec_tree.txt              # Notes / output for decision tree
│
├── knn.py                    # KNN direction + GARCH magnitude
├── knn.txt                   # Notes / output for KNN
│
├── polyregression.py         # LDA + Polynomial Ridge two-stage model
├── poly_reg.txt              # Notes / output for poly regression
│
├── plots.py                  # Shared plotting utilities
│
├── ppt.html                  # Full presentation (all 14 slides)
├── features.html             # Feature overview interactive plot
│
├── features.png              # Feature engineering visual
├── decision_tree.png         # Decision Tree predictions plot (2021)
├── knn.png                   # KNN predictions plot (2021)
├── polynomial_reg.png        # LDA + Ridge predictions plot (2021)
├── Stats_model.png           # Statistical model predictions plot (2021)
└── usd_inr.png               # USD/INR full time-series chart
```

---

## Running the Code

### 1. Statistical Model

```bash
python stats_model1.py
python stats_model2.py
```

Runs GARCH(1,1) volatility estimation + EMA threshold direction model with walk-forward validation. Outputs per-year error metrics and actual vs predicted plots.

### 2. Decision Tree Model (Best)

```bash
python dec_tree.py
```

Trains a depth-tuned Decision Tree classifier on all 10 features. Direction output is combined with GARCH magnitude for final price prediction. Achieves ~74.68% directional accuracy on the 2021–2023 test set.

### 3. KNN Model

```bash
python knn.py
```

Standardises features and searches k ∈ {3, 5, 7, 9, 11, 15, 21, 31} via 5-fold CV per walk-forward window. Direction output combined with GARCH magnitude.

### 4. LDA + Polynomial Ridge Model

```bash
python polyregression.py
```

Two-stage model: LDA for direction (momentum/trend features) + RidgeCV for magnitude (volatility/candle features). Walk-forward refit every 7 days.

### 5. Generate Plots

```bash
python plots.py
```

Generates all visualisation outputs used in the presentation.

---

## Walk-Forward Validation

All models use the same framework:

- **Initial train window:** 2003–2020
- **Test period:** 2021–2023 (695 trading days)
- **Refit frequency:** Every 7 days
- **Rolling window:** Last 1000 trading days per refit
- **Zero lookahead bias** enforced throughout

---

## Key Results (Test Set: 2021–2023)

| Model | Avg Abs Err (%) | Max Err (%) | Dir Acc (avg) |
|---|---|---|---|
| Statistical | 0.2362 | 1.3941 | — |
| **Decision Tree** | **0.2082** | **1.1016** | **74.68%** |
| KNN | 0.2357 | 1.5129 | 50.19% |
| LDA + Ridge | 0.3456 | 1.5622 | 51.65% |

> A directional accuracy above ~51–55% is sufficient to generate positive-expectancy signals after transaction costs in a liquid FX pair like USD/INR.

---

## Team

| Name | Roll No. | Contribution |
|---|---|---|
| Kriti Agarwal | B24EE1037 | Feature engineering, GARCH modeling, statistical model, EDA, walk-forward framework |
| Vanshika Mehta | B24EE1084 | Feature engineering, GARCH modeling, statistical model, EDA, walk-forward framework |
| Tanishka Trivedi | B24EE1080 | Decision Tree, KNN, LDA+Ridge models, evaluation pipeline, model comparison |
| Vempati Nityan | B24EE1087 | Decision Tree, KNN, LDA+Ridge models, evaluation pipeline, model comparison |
