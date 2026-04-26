#!/usr/bin/env python3
"""
Unemployment Rate Forecasting Pipeline -- Republic of Kazakhstan
================================================================
Diploma Project: Monthly unemployment rate forecasting (24-month horizon)
using macroeconomic indicators and multiple model families.

Steps:
  1. Data Validation
  2. Exploratory Data Analysis (EDA)
  3. Preprocessing
  4. Model Building (8 models)
  5. Training & Evaluation (TimeSeriesSplit, hyperparameter tuning)

Outputs:
  - figures/           -- all EDA and forecast plots (PNG + PGF for LaTeX)
  - results/           -- CSV tables for the LaTeX report
  - models/            -- serialised best models

Usage:
  python pipeline.py
"""

import os
import sys
import json
import warnings
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATA        = ROOT / "data"
FIG_DIR     = ROOT / "figures"
RES_DIR     = ROOT / "results"
MODEL_DIR   = ROOT / "models"

TRAIN_END   = "2023-12"
TEST_START  = "2024-01"
N_SPLITS    = 5          # TimeSeriesSplit folds
RANDOM_STATE = 42
LOOKBACK    = 12          # LSTM lookback window
FORECAST_H  = 24         # months

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

for d in (FIG_DIR, RES_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1, denom)
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100


def mase(y_true, y_pred, y_train, m=12):
    """Mean Absolute Scaled Error (relative to seasonal naïve)."""
    n = len(y_train)
    d = np.mean(np.abs(np.diff(y_train, n=m)))
    if d == 0:
        d = 1e-8
    return np.mean(np.abs(y_true - y_pred)) / d


def all_metrics(y_true, y_pred, y_train):
    return {
        "MAE":   mean_absolute_error(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2":    r2_score(y_true, y_pred),
        "MASE":  mase(y_true, y_pred, y_train.values),
    }


def save_fig(name, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()


def print_header(title):
    sep = "=" * 70
    print(f"\n{sep}\n  {title}\n{sep}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1 -- DATA VALIDATION
# ══════════════════════════════════════════════════════════════════════
print_header("STEP 1 - DATA VALIDATION")

raw = pd.read_excel(DATA / "cpi_data" / "diploma_dataset.xlsx")
print(f"Raw shape: {raw.shape}")
print(f"Columns : {list(raw.columns)}")

# Check for target variable
target_candidates = [c for c in raw.columns
                     if "unemploy" in c.lower() or "unemp" in c.lower()]
if not target_candidates:
    sys.exit("ERROR: No unemployment target variable found in dataset. Aborting.")

target_raw = target_candidates[0]
print(f"\nTarget variable found: '{target_raw}'")

# Rename for consistency
raw = raw.rename(columns={target_raw: "unemployed_rate"})

# Drop CPI sub-component columns (handled separately)
cpi_cols = [c for c in raw.columns if c not in [
    "Date", "unemployed_rate", "total_cpi_yoy", "gold_price_usd_avg",
    "usd_kzt", "interest_rate", "Oil_crude_brent", "GDP_growth"
]]
df = raw.drop(columns=cpi_cols).copy()
print(f"Dropped {len(cpi_cols)} CPI sub-component columns.")

# Parse dates
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df.set_index("Date", inplace=True)
df.index.freq = "MS"   # monthly start

# ---------- Merge additional macro series ----------

# National Bank base rate
nbk = pd.read_excel(DATA / "unemp" / "National Bank Base Rate.xlsx")
nbk.columns = ["Date_str", "Base_Rate", "Corridor"]
nbk["Date_str"] = nbk["Date_str"].astype(str).str.replace(r"\*", "", regex=True).str.strip()
nbk["Date"] = pd.to_datetime(nbk["Date_str"], dayfirst=True, format="mixed")
nbk = nbk[["Date", "Base_Rate"]].set_index("Date").resample("MS").last().ffill()
df = df.join(nbk, how="left")
df["Base_Rate"] = df["Base_Rate"].ffill().bfill()

# USD/KZT daily -> monthly volatility
usd_raw = pd.read_excel(DATA / "unemp" / "USD_TENGE.xlsx")
usd_raw.columns = usd_raw.columns[:3].tolist() if len(usd_raw.columns) >= 3 else usd_raw.columns.tolist()
date_col = usd_raw.columns[0]
rate_col = usd_raw.columns[1]
usd_raw[date_col] = pd.to_datetime(usd_raw[date_col], dayfirst=True, format="mixed")
# Ensure rate column is numeric
if usd_raw[rate_col].dtype == object:
    usd_raw[rate_col] = usd_raw[rate_col].astype(str).str.replace(",", ".").str.strip()
    usd_raw[rate_col] = pd.to_numeric(usd_raw[rate_col], errors="coerce")
usd_monthly_vol = (usd_raw.set_index(date_col)[rate_col]
                   .resample("MS").std()
                   .rename("usd_kzt_volatility"))
df = df.join(usd_monthly_vol, how="left")
df["usd_kzt_volatility"] = df["usd_kzt_volatility"].ffill().bfill()

# Brent oil (cross-check / fill)
brent = pd.read_excel(DATA / "unemp" / "POILBREUSDM.xlsx")
brent.columns = ["Date", "Brent"]
brent["Date"] = pd.to_datetime(brent["Date"])
brent = brent.set_index("Date").resample("MS").last()
# already have Oil_crude_brent -- use brent to fill any NaNs
if "Oil_crude_brent" in df.columns:
    df["Oil_crude_brent"] = df["Oil_crude_brent"].fillna(brent["Brent"])

# RUB/KZT -- attempt local CSV first; derive from yfinance as fallback
rub_path = DATA / "unemp" / "RUB_KZT_mon.csv"
if rub_path.exists():
    rub_local = pd.read_csv(rub_path)
    rub_local.columns = rub_local.columns[:7].tolist()
    date_col_rub = rub_local.columns[0]
    close_col_rub = [c for c in rub_local.columns if "close" in c.lower() or "price" in c.lower()]
    if close_col_rub:
        rub_local["Date"] = pd.to_datetime(rub_local[date_col_rub])
        rub_series = rub_local.set_index("Date")[close_col_rub[0]].resample("MS").last()
    else:
        rub_local["Date"] = pd.to_datetime(rub_local[date_col_rub])
        rub_series = rub_local.set_index("Date").iloc[:, 0].resample("MS").last()
    rub_series.name = "rub_kzt"
    df = df.join(rub_series, how="left")

# Fill rub_kzt with yfinance if many NaNs
if "rub_kzt" in df.columns and df["rub_kzt"].isna().sum() > len(df) * 0.5:
    try:
        import yfinance as yf
        usdrub = yf.download("RUBUSD=X", start="2010-01-01",
                             end=df.index[-1].strftime("%Y-%m-%d"),
                             progress=False)["Close"]
        usdrub_monthly = usdrub.resample("MS").mean()
        usdrub_monthly.name = "USD_RUB"
        derived_rub_kzt = df["usd_kzt"] / usdrub_monthly
        derived_rub_kzt.name = "rub_kzt"
        df["rub_kzt"] = df["rub_kzt"].fillna(derived_rub_kzt)
    except Exception as e:
        print(f"  [WARN] yfinance fallback failed: {e}")

# Convert string columns with comma decimals to numeric
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Final fill
df = df.ffill().bfill()

print(f"\nFinal dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
print(f"Missing values:\n{df.isna().sum()}")

# Correlation screening
corr_with_target = df.corrwith(df["unemployed_rate"]).drop("unemployed_rate").sort_values(key=abs, ascending=False)
print("\nCorrelation with unemployment:")
print(corr_with_target.to_string())

# Save validation summary
val_summary = pd.DataFrame({
    "Feature": df.columns,
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "min": df.min(),
    "max": df.max(),
    "mean": df.mean(),
    "std": df.std(),
}).reset_index(drop=True)
val_summary.to_csv(RES_DIR / "01_data_validation.csv", index=False)


# ══════════════════════════════════════════════════════════════════════
# STEP 2 -- EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print_header("STEP 2 - EXPLORATORY DATA ANALYSIS")

# 2a. Summary statistics
desc = df.describe().T
desc.to_csv(RES_DIR / "02_summary_statistics.csv")
print(desc.to_string())

# 2b. Time series plot of target
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df["unemployed_rate"], linewidth=1.5, color="steelblue")
ax.axvline(pd.Timestamp(TEST_START), color="red", linestyle="--", label="Train/Test split")
ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_title("Monthly Unemployment Rate -- Kazakhstan (2010-2025)")
ax.legend()
save_fig("02a_unemployment_timeseries")

# 2c. Seasonal decomposition
try:
    decomp = seasonal_decompose(df["unemployed_rate"], model="additive", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], title="Observed")
    decomp.trend.plot(ax=axes[1], title="Trend")
    decomp.seasonal.plot(ax=axes[2], title="Seasonal")
    decomp.resid.plot(ax=axes[3], title="Residual")
    for a in axes:
        a.set_xlabel("")
    save_fig("02c_seasonal_decomposition")
    print("Seasonal decomposition: OK")
except Exception as e:
    print(f"Seasonal decomposition warning: {e}")

# 2d. ADF test for stationarity
print("\nAugmented Dickey-Fuller Tests:")
adf_results = []
for col in df.columns:
    series = df[col].dropna()
    if len(series) < 20 or series.std() == 0:
        continue
    result = adfuller(series, autolag="AIC")
    stat, pval = result[0], result[1]
    stationary = "Yes" if pval < 0.05 else "No"
    adf_results.append({
        "Variable": col,
        "ADF Statistic": round(stat, 4),
        "p-value": round(pval, 4),
        "Stationary (5%)": stationary
    })
    print(f"  {col:30s}  ADF={stat:8.4f}  p={pval:.4f}  => {stationary}")

adf_df = pd.DataFrame(adf_results)
adf_df.to_csv(RES_DIR / "02d_adf_tests.csv", index=False)

# 2e. Correlation matrix
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix")
save_fig("02e_correlation_matrix")
corr_matrix.to_csv(RES_DIR / "02e_correlation_matrix.csv")
print("Correlation matrix saved.")

# 2f. ACF/PACF of target
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df["unemployed_rate"].dropna(), lags=36, ax=axes[0], title="ACF -- Unemployment Rate")
plot_pacf(df["unemployed_rate"].dropna(), lags=36, ax=axes[1], title="PACF -- Unemployment Rate")
save_fig("02f_acf_pacf")

# 2g. Distribution of target
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["unemployed_rate"].hist(bins=30, ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Histogram -- Unemployment Rate")
axes[0].set_xlabel("Unemployment Rate (%)")
stats.probplot(df["unemployed_rate"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot -- Unemployment Rate")
save_fig("02g_distribution")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 -- PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
print_header("STEP 3 - PREPROCESSING")

df_model = df.copy()

# 3a. Interpolation for missing values (time series continuity)
n_missing_before = df_model.isna().sum().sum()
df_model = df_model.interpolate(method="time")
df_model = df_model.ffill().bfill()
n_missing_after = df_model.isna().sum().sum()
print(f"Missing values: {n_missing_before} -> {n_missing_after} (interpolation)")
print("Justification: Time-based interpolation preserves temporal continuity,")
print("  which is critical for time series models (ARIMA, LSTM).")
print("  K-NN imputation would ignore the temporal ordering of observations.")

# 3b. Feature engineering -- lagged target & differenced regressors
df_model["y_lag1"]  = df_model["unemployed_rate"].shift(1)
df_model["y_lag12"] = df_model["unemployed_rate"].shift(12)

diff_cols = ["usd_kzt", "Oil_crude_brent", "Base_Rate", "gold_price_usd_avg"]
if "rub_kzt" in df_model.columns:
    diff_cols.append("rub_kzt")

for col in diff_cols:
    if col in df_model.columns:
        df_model[f"{col}_diff"] = df_model[col].diff()

# Drop raw level columns that were differenced (keep diffs)
df_model.drop(columns=[c for c in diff_cols if c in df_model.columns],
              inplace=True, errors="ignore")

# Drop initial rows with NaNs from lagging/differencing
df_model.dropna(inplace=True)

print(f"\nFinal model dataset: {df_model.shape}")
feature_cols = [c for c in df_model.columns if c != "unemployed_rate"]
print(f"Features ({len(feature_cols)}): {feature_cols}")

# 3c. Train/test split
train = df_model.loc[:TRAIN_END]
test  = df_model.loc[TEST_START:]

X_train = train.drop(columns=["unemployed_rate"])
y_train = train["unemployed_rate"]
X_test  = test.drop(columns=["unemployed_rate"])
y_test  = test["unemployed_rate"]

print(f"\nTrain: {X_train.shape[0]} samples  ({train.index.min().date()} -> {train.index.max().date()})")
print(f"Test:  {X_test.shape[0]} samples  ({test.index.min().date()} -> {test.index.max().date()})")

# 3d. Scaling for neural networks
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = pd.DataFrame(scaler_X.fit_transform(X_train),
                           columns=X_train.columns, index=X_train.index)
X_test_sc  = pd.DataFrame(scaler_X.transform(X_test),
                           columns=X_test.columns, index=X_test.index)
y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_sc  = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

print("StandardScaler fitted on training data, applied to test data.")

# Save preprocessing info
preproc_info = {
    "interpolation_method": "time-based",
    "missing_before": int(n_missing_before),
    "missing_after": int(n_missing_after),
    "lag_features": ["y_lag1", "y_lag12"],
    "differenced_features": [f"{c}_diff" for c in diff_cols],
    "scaling": "StandardScaler",
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "n_features": int(X_train.shape[1]),
}
with open(RES_DIR / "03_preprocessing.json", "w") as f:
    json.dump(preproc_info, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
# STEP 4 & 5 -- MODEL BUILDING, TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════
print_header("STEP 4-5 - MODEL BUILDING & EVALUATION")

results = {}
predictions = {}
tscv = TimeSeriesSplit(n_splits=N_SPLITS)


# ── Model 1: Naive Seasonal Forecast (baseline) ──────────────────────
print("\n[1/8] Naive Seasonal Forecast")
naive_pred = []
for i, idx in enumerate(y_test.index):
    # Use value from 12 months before
    target_date = idx - pd.DateOffset(months=12)
    if target_date in df["unemployed_rate"].index:
        naive_pred.append(df["unemployed_rate"].loc[target_date])
    else:
        # Fallback: last known value
        naive_pred.append(y_train.iloc[-1])

naive_pred = np.array(naive_pred)
results["Naive Seasonal"] = all_metrics(y_test, naive_pred, y_train)
predictions["Naive Seasonal"] = naive_pred
print(f"  MAE={results['Naive Seasonal']['MAE']:.4f}  RMSE={results['Naive Seasonal']['RMSE']:.4f}")


# ── Model 2: Linear Regression (OLS with robust SE) ──────────────────
print("\n[2/8] Linear Regression (OLS)")
X_train_const = sm.add_constant(X_train)
X_test_const  = sm.add_constant(X_test)
ols_model = sm.OLS(y_train, X_train_const).fit(cov_type="HC1")
ols_pred = ols_model.predict(X_test_const).values

results["Linear Regression"] = all_metrics(y_test, ols_pred, y_train)
predictions["Linear Regression"] = ols_pred
print(f"  MAE={results['Linear Regression']['MAE']:.4f}  RMSE={results['Linear Regression']['RMSE']:.4f}")
print(f"  Adj. R2={ols_model.rsquared_adj:.4f}")


# ── Model 3: SARIMA(2,1,1)(1,1,1)₁₂ ─────────────────────────────────
print("\n[3/8] SARIMA(2,1,1)(1,1,1)_12")
try:
    sarima_model = SARIMAX(
        y_train,
        order=(2, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=500)

    sarima_pred = sarima_model.get_forecast(steps=len(y_test)).predicted_mean.values
    results["SARIMA"] = all_metrics(y_test, sarima_pred, y_train)
    predictions["SARIMA"] = sarima_pred
    print(f"  AIC={sarima_model.aic:.1f}  BIC={sarima_model.bic:.1f}")
    print(f"  MAE={results['SARIMA']['MAE']:.4f}  RMSE={results['SARIMA']['RMSE']:.4f}")
except Exception as e:
    print(f"  SARIMA failed: {e}")
    results["SARIMA"] = {k: np.nan for k in ["MAE","RMSE","MAPE","sMAPE","R2","MASE"]}


# ── Model 4: SARIMAX with exogenous variables ────────────────────────
print("\n[4/8] SARIMAX + Exogenous")
exog_cols = [c for c in X_train.columns if c.endswith("_diff") or c in ["GDP_growth", "interest_rate"]]
if len(exog_cols) >= 2:
    try:
        sarimax_model = SARIMAX(
            y_train,
            exog=X_train[exog_cols],
            order=(2, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=500)

        sarimax_pred = sarimax_model.get_forecast(
            steps=len(y_test), exog=X_test[exog_cols]
        ).predicted_mean.values
        results["SARIMAX"] = all_metrics(y_test, sarimax_pred, y_train)
        predictions["SARIMAX"] = sarimax_pred
        print(f"  Exog: {exog_cols}")
        print(f"  AIC={sarimax_model.aic:.1f}  BIC={sarimax_model.bic:.1f}")
        print(f"  MAE={results['SARIMAX']['MAE']:.4f}  RMSE={results['SARIMAX']['RMSE']:.4f}")
    except Exception as e:
        print(f"  SARIMAX failed: {e}")
        results["SARIMAX"] = {k: np.nan for k in ["MAE","RMSE","MAPE","sMAPE","R2","MASE"]}
else:
    print("  Skipping: not enough exogenous columns")
    results["SARIMAX"] = {k: np.nan for k in ["MAE","RMSE","MAPE","sMAPE","R2","MASE"]}


# ── Model 5: Elastic Net (L1+L2 regularised) ─────────────────────────
print("\n[5/8] Elastic Net (CV)")
en = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=np.logspace(-4, 1, 50),
    cv=tscv,
    max_iter=10000,
    random_state=RANDOM_STATE,
).fit(X_train_sc, y_train)

en_pred = en.predict(X_test_sc)
results["Elastic Net"] = all_metrics(y_test, en_pred, y_train)
predictions["Elastic Net"] = en_pred
print(f"  alpha={en.alpha_:.6f}  l1_ratio={en.l1_ratio_:.2f}")
print(f"  MAE={results['Elastic Net']['MAE']:.4f}  RMSE={results['Elastic Net']['RMSE']:.4f}")

# Feature importance for Elastic Net
en_coef = pd.DataFrame({"Feature": X_train.columns, "Coef": en.coef_})
en_coef = en_coef.reindex(en_coef["Coef"].abs().sort_values(ascending=False).index)
en_coef.to_csv(RES_DIR / "05_elasticnet_coefs.csv", index=False)


# ── Model 6: Random Forest Regressor ─────────────────────────────────
print("\n[6/8] Random Forest Regressor (RandomizedSearchCV)")
rf_params = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, 0.8],
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE),
    param_distributions=rf_params,
    n_iter=40,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
).fit(X_train, y_train)

rf = rf_search.best_estimator_
rf_pred = rf.predict(X_test)
results["Random Forest"] = all_metrics(y_test, rf_pred, y_train)
predictions["Random Forest"] = rf_pred
print(f"  Best params: {rf_search.best_params_}")
print(f"  MAE={results['Random Forest']['MAE']:.4f}  RMSE={results['Random Forest']['RMSE']:.4f}")

# Feature importance
fi = pd.DataFrame({"Feature": X_train.columns, "Importance": rf.feature_importances_})
fi = fi.sort_values("Importance", ascending=False)
fi.to_csv(RES_DIR / "06_rf_feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(fi["Feature"], fi["Importance"], color="steelblue")
ax.set_xlabel("Mean Decrease in Impurity")
ax.set_title("Random Forest -- Feature Importance")
ax.invert_yaxis()
save_fig("06_rf_feature_importance")


# ── Model 7: XGBoost / Gradient Boosting ─────────────────────────────
print("\n[7/8] Gradient Boosting Regressor")
gb_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "min_samples_leaf": [2, 5, 10],
}

gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    param_distributions=gb_params,
    n_iter=40,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
).fit(X_train, y_train)

gb = gb_search.best_estimator_
gb_pred = gb.predict(X_test)
results["Gradient Boosting"] = all_metrics(y_test, gb_pred, y_train)
predictions["Gradient Boosting"] = gb_pred
print(f"  Best params: {gb_search.best_params_}")
print(f"  MAE={results['Gradient Boosting']['MAE']:.4f}  RMSE={results['Gradient Boosting']['RMSE']:.4f}")


# ── Model 8: LSTM Recurrent Neural Network ───────────────────────────
print("\n[8/8] LSTM-RNN")

def create_lstm_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# Prepare sequences from scaled data
all_X_sc = pd.DataFrame(
    scaler_X.transform(df_model.drop(columns=["unemployed_rate"])),
    columns=feature_cols,
    index=df_model.index
)
all_y_sc = scaler_y.transform(df_model["unemployed_rate"].values.reshape(-1, 1)).ravel()

X_seq, y_seq = create_lstm_sequences(all_X_sc.values, all_y_sc, LOOKBACK)

# Split sequences into train/test
train_size = len(y_train) - LOOKBACK
test_size  = len(y_test)
X_lstm_train = X_seq[:train_size]
y_lstm_train = y_seq[:train_size]
X_lstm_test  = X_seq[train_size:train_size + test_size]
y_lstm_test  = y_seq[train_size:train_size + test_size]

print(f"  LSTM train sequences: {X_lstm_train.shape}")
print(f"  LSTM test  sequences: {X_lstm_test.shape}")

# Build LSTM (PyTorch)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden1=64, hidden2=32):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc1   = nn.Linear(hidden2, 16)
        self.fc2   = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])   # last timestep
        out = torch.relu(self.fc1(out))
        return self.fc2(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTMModel(X_lstm_train.shape[2]).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Prepare DataLoader
train_ds = TensorDataset(
    torch.tensor(X_lstm_train, dtype=torch.float32),
    torch.tensor(y_lstm_train, dtype=torch.float32),
)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)

# Validation split (last 15%)
val_size = max(1, int(len(X_lstm_train) * 0.15))
X_val_t = torch.tensor(X_lstm_train[-val_size:], dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_lstm_train[-val_size:], dtype=torch.float32).to(device)

train_losses, val_losses = [], []
best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(200):
    lstm_model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = lstm_model(xb).squeeze()
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    train_losses.append(epoch_loss / len(train_ds))

    lstm_model.eval()
    with torch.no_grad():
        val_pred = lstm_model(X_val_t).squeeze()
        val_loss = criterion(val_pred, y_val_t).item()
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in lstm_model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 20:
            break

if best_state:
    lstm_model.load_state_dict(best_state)

# Predict
if X_lstm_test.shape[0] > 0:
    lstm_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_lstm_test, dtype=torch.float32).to(device)
        lstm_pred_sc = lstm_model(X_test_t).squeeze().cpu().numpy()
    lstm_pred = scaler_y.inverse_transform(lstm_pred_sc.reshape(-1, 1)).ravel()

    n_pred = min(len(lstm_pred), len(y_test))
    lstm_y_true = y_test.iloc[:n_pred]
    lstm_pred = lstm_pred[:n_pred]

    results["LSTM"] = all_metrics(lstm_y_true, lstm_pred, y_train)
    predictions["LSTM"] = lstm_pred
    print(f"  Epochs trained: {len(train_losses)}")
    print(f"  MAE={results['LSTM']['MAE']:.4f}  RMSE={results['LSTM']['RMSE']:.4f}")
else:
    print("  LSTM: Not enough test sequences")
    results["LSTM"] = {k: np.nan for k in ["MAE","RMSE","MAPE","sMAPE","R2","MASE"]}

# LSTM training curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label="Train Loss")
ax.plot(val_losses, label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("LSTM Training History")
ax.legend()
save_fig("08_lstm_training")


# ══════════════════════════════════════════════════════════════════════
# RESULTS COMPILATION
# ══════════════════════════════════════════════════════════════════════
print_header("RESULTS SUMMARY")

results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
results_df = results_df.sort_values("MAE")
print(results_df.round(4).to_string())
results_df.round(4).to_csv(RES_DIR / "05_model_comparison.csv")

# Best model
best = results_df["MAE"].idxmin()
print(f"\n* Best model by MAE: {best}")
print(f"  MAE  = {results_df.loc[best, 'MAE']:.4f}")
print(f"  RMSE = {results_df.loc[best, 'RMSE']:.4f}")
print(f"  R2   = {results_df.loc[best, 'R2']:.4f}")


# ── Forecast vs Actual plot ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(y_test.index, y_test.values, "k-", linewidth=2, label="Actual", zorder=10)

colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
for (name, pred), color in zip(predictions.items(), colors):
    n = min(len(pred), len(y_test))
    ax.plot(y_test.index[:n], pred[:n], "--", color=color, alpha=0.8, label=name)

ax.set_xlabel("Date")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_title("Forecast vs Actual -- All Models (Test Period)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
save_fig("09_forecast_vs_actual")


# ── Metrics bar chart ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, metric in zip(axes.ravel(), ["MAE", "RMSE", "MAPE", "sMAPE"]):
    vals = results_df[metric].sort_values()
    bars = ax.barh(vals.index, vals.values, color="steelblue")
    ax.set_xlabel(metric)
    ax.set_title(metric)
    for bar, val in zip(bars, vals.values):
        ax.text(val, bar.get_y() + bar.get_height()/2, f" {val:.4f}",
                va="center", fontsize=9)
save_fig("10_metrics_comparison")


# ── Residual analysis for best model ─────────────────────────────────
if best in predictions:
    best_pred = predictions[best]
    n = min(len(best_pred), len(y_test))
    residuals = y_test.values[:n] - best_pred[:n]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(y_test.index[:n], residuals, "o-", markersize=3)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"Residuals -- {best}")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=15, edgecolor="white", color="steelblue")
    axes[1].set_title("Residual Distribution")

    stats.probplot(residuals, plot=axes[2])
    axes[2].set_title("Residual Q-Q Plot")
    save_fig("11_residual_analysis")


# ══════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION RESULTS (5-fold TimeSeriesSplit)
# ══════════════════════════════════════════════════════════════════════
print_header("CROSS-VALIDATION (TimeSeriesSplit, k=5)")

cv_results = {}
cv_models = {
    "Elastic Net": ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=tscv,
                                max_iter=10000, random_state=RANDOM_STATE),
    "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 50), cv=N_SPLITS),
    "Random Forest": rf_search.best_estimator_,
    "Gradient Boosting": gb_search.best_estimator_,
}

for name, model in cv_models.items():
    fold_scores = []
    for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        if name in ["Elastic Net", "Ridge"]:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_va_s = sc.transform(X_va)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_va_s)
        else:
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)

        fold_scores.append(mean_absolute_error(y_va, preds))

    cv_results[name] = {
        "CV_MAE_mean": np.mean(fold_scores),
        "CV_MAE_std": np.std(fold_scores),
    }
    print(f"  {name:25s}  MAE = {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

cv_df = pd.DataFrame(cv_results).T
cv_df.to_csv(RES_DIR / "05_cv_results.csv")


# ══════════════════════════════════════════════════════════════════════
# SAVE PREDICTION DATA FOR LATEX PLOTS
# ══════════════════════════════════════════════════════════════════════
pred_df = pd.DataFrame({"Date": y_test.index, "Actual": y_test.values})
for name, pred in predictions.items():
    n = min(len(pred), len(y_test))
    col_name = name.replace(" ", "_")
    pred_df[col_name] = np.nan
    pred_df.loc[:n-1, col_name] = pred[:n]

pred_df.to_csv(RES_DIR / "predictions.csv", index=False)


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print_header("PIPELINE COMPLETE")
print(f"Figures saved to: {FIG_DIR}")
print(f"Results saved to: {RES_DIR}")
print(f"Best model: {best} (MAE = {results_df.loc[best, 'MAE']:.4f})")
print(f"\nAll output files:")
for d in (FIG_DIR, RES_DIR):
    for f in sorted(d.iterdir()):
        print(f"  {f.relative_to(ROOT)}")
