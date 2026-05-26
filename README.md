# Stock Market Anomaly Detection

Detects unusual market days and stock days using only daily price and volume data. No labels, no news, no complex models — just rolling statistics and unsupervised clustering.

> Educational analytics only — not investment advice.

---

## What counts as unusual?

A day is unusual when today's behaviour is extreme relative to its recent history, measured using rolling windows (past data only, no look-ahead).

**Stock-level triggers — any one fires:**
- `|ret_z| > 2.5` — large return relative to recent history
- `vol_z > 2.5` — unusually high volume
- `range_pct > 95` — unusually wide intraday range

**Market-level triggers — either fires:**
- `|market_ret|` above its own rolling 95th percentile
- `breadth < 0.3` — fewer than 30% of tickers positive

---

## Features

| Feature | Formula | Window |
|---------|---------|--------|
| Return z-score | `(ret - rolling_mean) / rolling_std` | 63 days |
| Volume z-score | `(log(V) - rolling_mean) / rolling_std` | 21 days |
| Intraday range percentile | percentile rank of `(High-Low)/Close` vs past W days | 63 days |

All features use past data only — no leakage.

---

## Anomaly Types

| Type | Condition |
|------|-----------|
| `crash` | `ret < 0` and `\|ret_z\| > 2.5` |
| `spike` | `ret > 0` and `\|ret_z\| > 2.5` |
| `volume_shock` | `vol_z > 2.5` (can co-exist with crash/spike) |
| `cluster_only` | flagged by KMeans or DBSCAN but not rule-based |

---

## Detectors

### Rule-based
Simple threshold triggers on `ret_z`, `vol_z`, `range_pct`. Interpretable and fast.

### K-Means
- Fit once on Train (2018), centroids frozen
- Anomaly = distance to nearest centroid > cluster's q-th percentile threshold
- Tuned on Val (2019) to hit 2–8% flag rate

### DBSCAN
- Walk-forward: refit monthly on expanding window
- Anomaly = points labeled -1 (noise)
- Scaler fitted on Train only, never refit

### Consensus
- **Union** — flag if KMeans OR DBSCAN flags (higher recall)
- **Intersection** — flag if both flag (higher precision)

---

## Dataset

Kaggle NASDAQ daily OHLCV CSVs — one per ticker.  
Link: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset  
Columns: `Date, Open, High, Low, Close, Adj Close, Volume`  
Universe: `QQQ, AAPL, MSFT, AMZN, NVDA, TSLA`  
Always use **Adj Close** for returns — accounts for splits and dividends.

---

## Train / Val / Test Split

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2018 | Fit models, scaler |
| Validation | 2019 | Tune thresholds, target 2–8% flag rate |
| Test | 2020 Q1 (Jan–Mar) | Locked thresholds, final evaluation |

---

## Project Structure

```
stock-anomaly/
├── data/
│   ├── stocks/               # Kaggle CSVs (one per ticker)
│   └── processed.csv        # Cleaned merged CSV
├── src/
│   ├── data_loader.py
│   ├── feature.py         # ret_z, vol_z, range_pct
│   ├── rule_based.py      # anomaly_flag, type, why
│   ├── market.py          # market_ret, breadth, market_anomaly_flag
│   ├── kmeans.py          # KMeansDetector
│   ├── dbscan.py          # DBSCANDetector
│   ├── walkforward.py     # full pipeline, writes CSVs
│   ├── query.py           # CLI date query
│   └── monthly.py         # CLI monthly report
├── outputs/
│   ├── anomaly_cards/     # anomaly_card.csv
│   └── market_days/       # market_days.csv
├── README.md
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run full pipeline

```bash
python -m src.walkforward --universe QQQ,AAPL,MSFT,NVDA,AMZN,TSLA
```

Writes two CSVs to `outputs/`:
- `anomaly_cards/anomaly_card.csv` — per ticker per day
- `market_days/market_days.csv` — per date

### 3. Query a specific date

```bash
python -m src.query --date 2020-02-27
```

Output:
```
=== Market Status: 2020-02-27 ===
Market Return : -0.070
Breadth       : 0.0
Market Flag   : ANOMALOUS

=== Anomalous Tickers ===
ticker     ret_z    vol_z  range_pct               type                                  why
  AAPL -4.128862 2.681428  96.825397 crash+volume_shock |ret_z|>2.5; vol_z>2.5; range_pct>95
  MSFT -5.316050 2.865146  98.412698 crash+volume_shock |ret_z|>2.5; vol_z>2.5; range_pct>95
   QQQ -4.981904 2.581589 100.000000 crash+volume_shock |ret_z|>2.5; vol_z>2.5; range_pct>95
```

### 4. Monthly mini-report

```bash
python -m src.monthly --month 2020-02
```

---

## Output Files

### `anomaly_card.csv`

One row per ticker per day. Captures what happened to each stock.

```
date, ticker, anomaly_flag, type, ret, ret_z, vol_z, range_pct, why
2020-02-27, AAPL, 1, crash+volume_shock, -0.065, -4.13, 2.68, 96.8, |ret_z|>2.5; vol_z>2.5; range_pct>95
```

### `market_days.csv`

One row per date. Captures overall market behaviour.

```
date, market_ret, breadth, market_anomaly_flag
2020-02-27, -0.069, 0.0, 1
```

---

## Sanity Check — Feb 27 2020

The worst single day of the COVID crash:

| Ticker | ret | ret_z | vol_z | range_pct |
|--------|-----|-------|-------|-----------|
| MSFT | -0.070 | -5.32 | 2.87 | 98.4 |
| QQQ | -0.050 | -4.98 | 2.58 | 100.0 |
| AAPL | -0.065 | -4.13 | 2.68 | 96.8 |

Market: `market_ret=-0.069`, `breadth=0.0`, `market_anomaly_flag=1`

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
```