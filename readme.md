# Unemployment Rate Forecasting — Republic of Kazakhstan

## Problem Description

Unemployment is one of the most important macroeconomic indicators.  When
unemployment rises, households lose income, consumption declines, and the
government must increase social spending.  Accurate forecasts help policymakers
plan budgets, design labour-market programmes, and anticipate economic downturns.

This project forecasts the **monthly unemployment rate** $y_t$ for Kazakhstan
over a 24-month horizon (January 2024 – December 2025) using structural
macroeconomic indicators and both classical econometric and deep-learning methods.

> **Key question:** Given exchange rates, oil prices, interest rates, GDP growth,
> and the RUB/KZT cross-rate, can we predict how unemployment will change?

**Note:** CPI columns are excluded — CPI forecasting is handled in a separate module.

---

## Data Sources

| Indicator | Frequency | Source | Period |
|-----------|-----------|--------|--------|
| Unemployment rate (%) | Monthly | stat.gov.kz | 2010–2025 |
| USD/KZT exchange rate | Daily $\to$ monthly | National Bank of Kazakhstan | 2010–2026 |
| Brent crude oil (USD/barrel) | Monthly | Compiled dataset | 2010–2025 |
| GDP growth (%) | Monthly | stat.gov.kz / World Bank | 2010–2025 |
| Gold price (USD/oz) | Monthly | Compiled dataset | 2010–2025 |
| Interest rate (%) | Monthly | National Bank of Kazakhstan | 2010–2025 |
| National Bank base rate (%) | Irregular $\to$ monthly | nationalbank.kz | 2015–2026 |
| RUB/KZT exchange rate | Monthly | Derived via yfinance (USD/KZT $\div$ USD/RUB) | 2010–2025 |
| USD/KZT volatility | Derived | Monthly std of daily rates | 2010–2026 |

### Data Quality

- Primary dataset: **192 rows, zero missing values**
- RUB/KZT: The local file (`RUB_KZT_mon.csv`) has only 24 months (Apr 2024–Mar 2026).
  Historical data (2010–2024) is **derived** from yfinance by computing
  $\text{RUB/KZT} = \text{USD/KZT} \times \text{RUB/USD}$.

---

## Methodology

### Feature Engineering — Stationarity via Differencing

Non-stationary regressors are transformed to first-differences:

$$\Delta x_t = x_t - x_{t-1}$$

This removes unit-root trends.  The final feature matrix:

| Feature | Transformation |
|---------|---------------|
| $y_{t-1}$, $y_{t-12}$ | Autoregressive lags of the target |
| `usd_kzt_diff` | $\Delta\text{USD/KZT}_t$ |
| `Oil_crude_brent_diff` | $\Delta\text{Brent}_t$ |
| `Base_Rate_diff` | $\Delta\text{Base Rate}_t$ |
| `rub_kzt_diff` | $\Delta\text{RUB/KZT}_t$ |
| `GDP_growth` | Already a rate (YoY %), kept as-is |
| `interest_rate` | Level — captures policy stance |
| `gold_price_usd_avg` | Level — safe-haven proxy |
| `usd_kzt_volatility` | Monthly std of daily exchange rates |

**Not yet available:** `STEI`, `Retail_Trade_diff`, `Industrial_Production_diff` —
upload from [stat.gov.kz](https://stat.gov.kz) to complete the full specification.

### Train / Test Split

$$\text{Train:} \quad t \in [2015\text{-}01,\; 2023\text{-}12] \quad (n = 108)$$

$$\text{Test:} \quad t \in [2024\text{-}01,\; 2025\text{-}12] \quad (n = 24)$$

### Models

#### 1. SARIMA $(2,1,1)(1,1,1)_{12}$

Univariate seasonal ARIMA — **baseline**.

$$\Phi(B^{12})\,\phi(B)\,(1-B)(1-B^{12})\,y_t = c + \Theta(B^{12})\,\theta(B)\,\varepsilon_t$$

#### 2. SARIMAX + Exogenous

Same SARIMA structure augmented with differenced regressors.

#### 3. ARDL (Autoregressive Distributed Lag)

$$y_t = \alpha + \beta_1 y_{t-1} + \beta_{12} y_{t-12} + \gamma'\Delta X_t + \varepsilon_t$$

HC1 robust standard errors.

#### 4. Elastic Net

$$\min_w \frac{1}{2n}\|y - Xw\|_2^2 + \alpha\left[\lambda\|w\|_1 + \frac{1-\lambda}{2}\|w\|_2^2\right]$$

#### 5. Ridge Regression

$$\min_w \frac{1}{2n}\|y - Xw\|_2^2 + \alpha\|w\|_2^2$$

#### 6. Random Forest

An ensemble of $B$ regression trees:

$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

Feature importance measured by mean decrease in impurity (MDI).

#### 7. ANN — Multi-Layer Perceptron

$$\text{Input}(p) \to 128 \to 64 \to 32 \to 1$$

ReLU activation, Adam optimiser, early stopping.

#### 8. RNN — LSTM

$$h_t = \text{LSTM}(x_t,\; h_{t-1})$$

LSTM(64) $\to$ Dropout(0.2) $\to$ LSTM(32) $\to$ Dropout(0.2) $\to$ Dense(1).  Lookback $L = 12$.

### Evaluation Metrics

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

$$\text{MASE} = \frac{\text{MAE}}{\frac{1}{n-1}\sum_{i=2}^{n}|y_i - y_{i-1}|}$$

---

## Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.13 | Programming language |
| pandas | 2.x | Data manipulation |
| NumPy | 2.x | Numerical computation |
| statsmodels | 0.14+ | SARIMA, SARIMAX, OLS, ADF tests |
| scikit-learn | 1.x | ElasticNet, Ridge, Random Forest, MLP, scaling |
| TensorFlow / Keras | 2.x | LSTM recurrent neural network |
| matplotlib | 3.x | LaTeX-style visualisations |
| seaborn | 0.13+ | Heatmaps |
| yfinance | 0.2+ | Historical RUB/USD and Brent crude |

---

## Project Structure

```
PROJECT_DIPLOMA/
├── data/
│   ├── cpi_data/
│   │   └── diploma_dataset.xlsx        # Primary dataset (192 months)
│   ├── indicators/
│   │   └── energymetrics_Kazakhstan_1990-2026.xlsx
│   └── unemp/
│       ├── unemployment_rate.xlsx
│       ├── USD_TENGE.xlsx               # Daily USD/KZT
│       ├── National Bank Base Rate.xlsx  # Central bank repo rate
│       ├── RUB_KZT_mon.csv             # Monthly RUB/KZT (24 obs)
│       └── POILBREUSDM.xlsx
├── unemployment.ipynb                   # Main forecasting notebook
├── code/parts/                          # Development notebooks
├── plots/                               # Saved figures
└── README.md                            # This file
```

---

## References

1. Bureau of National Statistics of Kazakhstan — [stat.gov.kz](https://stat.gov.kz)
2. National Bank of Kazakhstan — [nationalbank.kz](https://nationalbank.kz)
3. Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed.
4. Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press.
5. Pesaran, M.H. & Shin, Y. (1998). An autoregressive distributed-lag modelling approach. Cambridge University Press.
6. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8).
7. Zou, H. & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2).
8. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
