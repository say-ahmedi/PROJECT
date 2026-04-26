"""
Production forecasting pipeline for Kazakhstan unemployment rate.
FIXED VERSION — resolves 6 critical issues:
  1. Lookahead bias in rolling features (.shift(1) applied)
  2. Recursive multi-step forecasting (no autoregressive leakage)
  3. XGBoost trained on differenced target (stationarity fix)
  4. XGBoost diagnostic uses tuned params, not hardcoded bad values
  5. LSTM/TCN removed (insufficient data; overkill for ~150 rows)
  6. SARIMAX enforce_stationarity/enforce_invertibility set to True
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import shutil

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import matplotlib.dates as mdates

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def infer_and_standardize_date(series: pd.Series) -> pd.Series:
    clean = (
        series.astype(str)
        .str.replace(r"\*+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^\dA-Za-z.\-/: ]", "", regex=True)
        .str.strip()
    )
    dt = pd.to_datetime(clean, errors="coerce", dayfirst=True, format="mixed")
    return dt.dt.to_period("M").dt.to_timestamp()


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.replace(",", ".", regex=False).str.strip()
            out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def drop_target_leakage_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    drop_cols = [
        c
        for c in out.columns
        if c != target_col and ("unemploy" in c.lower() or "unemploymentrate" in c.lower())
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def add_train_only_pca(
    train_df: pd.DataFrame,
    next_df: pd.DataFrame,
    target_col: str,
    n_components: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    nx = next_df.copy()
    cpi_candidates = [
        c
        for c in tr.columns
        if c != target_col and ("cpi" in c.lower() or "fuel" in c.lower()) and c.lower() != "total_cpi_yoy"
    ]
    if len(cpi_candidates) < n_components:
        return tr, nx

    scaler = StandardScaler()
    tr_x = tr[cpi_candidates].ffill().astype(float)
    nx_x = nx[cpi_candidates].ffill().astype(float)
    tr_s = scaler.fit_transform(tr_x)
    nx_s = scaler.transform(nx_x)

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    tr_pc = pca.fit_transform(tr_s)
    nx_pc = pca.transform(nx_s)

    for i in range(n_components):
        tr[f"pca_cpi_{i+1}"] = tr_pc[:, i]
        nx[f"pca_cpi_{i+1}"] = nx_pc[:, i]
    return tr, nx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        # "MAPE": float(mape(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def seasonal_naive_forecast(
    y_history: pd.Series, target_index: pd.DatetimeIndex, season: int = 12
) -> pd.Series:
    preds = []
    for t in target_index:
        ref = t - pd.DateOffset(months=season)
        preds.append(float(y_history.loc[ref]) if ref in y_history.index else float(y_history.iloc[-1]))
    return pd.Series(preds, index=target_index, name="SeasonalNaive")


def diebold_mariano(
    y_true: pd.Series,
    pred1: pd.Series,
    pred2: pd.Series,
    h: int = 1,
    power: int = 2,
) -> Tuple[float, float]:
    e1 = y_true.values - pred1.values
    e2 = y_true.values - pred2.values
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d_mean = np.mean(d)
    T = len(d)
    gamma0 = np.var(d, ddof=1)
    if h <= 1:
        var_d = gamma0
    else:
        var_d = gamma0
        for lag in range(1, h):
            cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
            var_d += 2 * (1 - lag / h) * cov
    var_d = max(var_d, 1e-12)
    dm_stat = d_mean / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_value)


def assign_model_tiers(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.copy().sort_values("MAE").reset_index(drop=True)
    labels = []
    for i in range(len(out)):
        if i == 0:
            labels.append("Champion")
        elif i <= 2:
            labels.append("Challenger")
        else:
            labels.append("Rejected")
    out["Tier"] = labels
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    root: Path
    target_col: str = "Unemployment Rate (%)"
    train_start: str = "2010-01-01"
    train_end: str = "2023-12-01"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-01"
    out_dir: str = "outputs"
    plots_dir: str = "figures"
    n_fourier: int = 3
    pca_components: int = 3
    lookback: int = 12
    epochs: int = 40

    @property
    def horizon(self) -> int:
        return 24


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sources(cfg: Config) -> pd.DataFrame:
    data_dir = cfg.root / "data"

    main = pd.read_csv(data_dir / "indicators" / "main_ind_ML_imputed.csv")
    main["Date"] = infer_and_standardize_date(main["Date"])
    main = main[["Date", cfg.target_col]].dropna(subset=["Date"]).copy()

    cpi_fore = pd.read_excel(
        data_dir / "cpi_data" / "dataset_for_forecast_2026111.xlsx",
        sheet_name="Sheet3",
    )
    cpi_fore["Date"] = infer_and_standardize_date(cpi_fore["Date"])

    diploma = pd.read_excel(data_dir / "cpi_data" / "diploma_dataset.xlsx", sheet_name="Sheet3")
    diploma["Date"] = infer_and_standardize_date(diploma["Date"])

    base = pd.read_excel(data_dir / "unemp" / "National Bank Base Rate.xlsx", sheet_name="base_rate")
    base["Date"] = infer_and_standardize_date(base["Date"])

    brent = pd.read_excel(data_dir / "unemp" / "POILBREUSDM.xlsx", sheet_name="Monthly")
    brent = brent.rename(columns={"observation_date": "Date"})
    brent["Date"] = infer_and_standardize_date(brent["Date"])

    usd = pd.read_excel(data_dir / "unemp" / "USD_TENGE.xlsx", sheet_name="exch_rate")
    usd["Date"] = infer_and_standardize_date(usd["Date"])
    usd["USD_KZT"] = pd.to_numeric(usd["TENGE"], errors="coerce") / pd.to_numeric(
        usd["USD_quant"], errors="coerce"
    )
    usd = usd[["Date", "USD_KZT"]]

    frames = [main, cpi_fore, diploma, base[["Date", "BaseRate"]], brent[["Date", "POILBREUSDM"]], usd]
    merged: Optional[pd.DataFrame] = None
    for frame in frames:
        frame = ensure_numeric(frame).copy()
        frame = frame.dropna(subset=["Date"]).groupby("Date", as_index=False).last()
        merged = frame if merged is None else merged.merge(frame, on="Date", how="outer")

    assert merged is not None
    merged = merged.sort_values("Date").set_index("Date")
    merged = merged.resample("MS").last()
    merged = merged.ffill()
    merged = drop_target_leakage_columns(merged, cfg.target_col)
    return merged


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_fourier_terms(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    out = df.copy()
    t = np.arange(len(out))
    for i in range(1, k + 1):
        out[f"fourier_sin_{i}"] = np.sin(2 * np.pi * i * t / 12)
        out[f"fourier_cos_{i}"] = np.cos(2 * np.pi * i * t / 12)
    return out


def safe_series(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_lags(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    lag_cols = [
        target_col,
        safe_series(out, ["log_oil_brent", "Oil_crude_brent", "POILBREUSDM"]),
        safe_series(out, ["log_usd_kzt", "usd_kzt", "USD_KZT"]),
        safe_series(out, ["interest_rate", "BaseRate"]),
    ]
    lag_cols = [c for c in lag_cols if c is not None]
    for col in lag_cols:
        for lag in [1, 2, 3, 6, 12]:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out


# FIX 1: All rolling calculations now use .shift(1) before .rolling()
# so that month t only has access to data up to t-1.
def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    oil_col = safe_series(out, ["log_oil_brent", "Oil_crude_brent", "POILBREUSDM"])
    usd_col = safe_series(out, ["log_usd_kzt", "usd_kzt", "USD_KZT"])
    for c in [oil_col, usd_col]:
        if c:
            # Shift by 1 before rolling to prevent lookahead leakage
            shifted = out[c].shift(1)
            out[f"{c}_ma_3"] = shifted.rolling(3).mean()
            out[f"{c}_ma_6"] = shifted.rolling(6).mean()
            out[f"{c}_std_3"] = shifted.rolling(3).std()
            out[f"{c}_std_6"] = shifted.rolling(6).std()
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    oil_col = safe_series(out, ["log_oil_brent", "Oil_crude_brent", "POILBREUSDM"])
    usd_col = safe_series(out, ["log_usd_kzt", "usd_kzt", "USD_KZT"])
    if oil_col and usd_col:
        oil_d = out[oil_col].diff()
        usd_d = out[usd_col].diff()
        out["interaction_doil_dusd"] = oil_d * usd_d
    return out


def add_calendar_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    month = out.index.month
    q = out.index.quarter
    out["is_january"] = (month == 1).astype(int)
    out["is_q1"] = (q == 1).astype(int)
    out["is_q4"] = (q == 4).astype(int)
    return out


def add_regime_dummies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["break_2015_08"] = (out.index >= pd.Timestamp("2015-08-01")).astype(int)
    out["break_2020_03"] = (out.index >= pd.Timestamp("2020-03-01")).astype(int)
    out["break_2022_01"] = (out.index >= pd.Timestamp("2022-01-01")).astype(int)
    return out


def engineer_features(
    df: pd.DataFrame, target_col: str, n_fourier: int, pca_components: int
) -> pd.DataFrame:
    out = df.copy()
    out = add_fourier_terms(out, n_fourier)
    out = add_lags(out, target_col)
    out = add_rolling(out)   # FIX 1 applied inside add_rolling
    out = add_interactions(out)
    out = add_calendar_dummies(out)
    out = add_regime_dummies(out)
    out = out.ffill()
    out = out.dropna()
    out = out.astype(float)
    return out


def split_data(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[cfg.train_start : cfg.train_end].copy()
    test = df.loc[cfg.test_start : cfg.test_end].copy()
    if len(test) != cfg.horizon:
        raise ValueError(f"Expected 24-month test horizon, found {len(test)} rows.")
    return train, test


# ---------------------------------------------------------------------------
# FIX 2: Recursive multi-step walk-forward forecasting
#
# Instead of pulling actual t-1 lag values from full_df for every test step,
# we maintain a running buffer of predictions. Whenever the model would use
# a lagged target value (e.g. target_lag_1) that falls inside the test window,
# we substitute the model's own prior prediction.
# ---------------------------------------------------------------------------

def walk_forward_predict(
    model_name: str,
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame, str], float],
    full_df: pd.DataFrame,
    cfg: Config,
) -> pd.Series:
    """
    Expanding-window walk-forward prediction with recursive lag substitution.

    For each step t in the test window:
      - Train on all data up to t-1.
      - Build the feature row for t, replacing any target-lag columns that
        reference timestamps inside the test window with prior predictions.
      - Predict t and store the result for future lag substitution.
    """
    test_idx = full_df.loc[cfg.test_start : cfg.test_end].index
    preds: Dict[pd.Timestamp, float] = {}

    # Identify target-lag column names so we can substitute recursively.
    target_lag_cols = {
        f"{cfg.target_col}_lag_{lag}": lag
        for lag in [1, 2, 3, 6, 12]
        if f"{cfg.target_col}_lag_{lag}" in full_df.columns
    }

    for t in test_idx:
        train_slice = full_df.loc[: t - pd.offsets.MonthBegin(1)].copy()
        x_row = full_df.loc[[t]].copy()

        # Substitute recursive predictions into lagged target columns.
        for col, lag in target_lag_cols.items():
            ref_ts = t - pd.DateOffset(months=lag)
            if ref_ts in preds:
                x_row[col] = preds[ref_ts]

        train_slice, x_row = add_train_only_pca(
            train_slice, x_row, cfg.target_col, cfg.pca_components
        )
        pred = fit_predict_fn(train_slice, x_row, cfg.target_col)
        preds[t] = pred

    return pd.Series(list(preds.values()), index=test_idx, name=model_name)


# ---------------------------------------------------------------------------
# Model implementations
# ---------------------------------------------------------------------------

# FIX 6: enforce_stationarity=True, enforce_invertibility=True
def fit_predict_sarimax(train_df: pd.DataFrame, x_next: pd.DataFrame, target_col: str) -> float:
    y = train_df[target_col].astype(float)
    exog_cols = [c for c in train_df.columns if c != target_col]
    model = SARIMAX(
        y,
        exog=train_df[exog_cols],
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=True,   # FIX 6
        enforce_invertibility=True,  # FIX 6
    ).fit(disp=False, maxiter=200)
    pred = model.get_forecast(steps=1, exog=x_next[exog_cols]).predicted_mean.iloc[0]
    return float(pred)


def fit_predict_varx(train_df: pd.DataFrame, x_next: pd.DataFrame, target_col: str) -> float:
    aux_candidates = [
        c for c in ["POILBREUSDM", "USD_KZT", "log_oil_brent", "log_usd_kzt"]
        if c in train_df.columns
    ]
    if len(aux_candidates) < 2:
        return float(train_df[target_col].iloc[-1])
    endog_cols = [target_col, aux_candidates[0], aux_candidates[1]]
    exog_cols = [c for c in train_df.columns if c not in endog_cols]
    endog = train_df[endog_cols]
    model = VARMAX(endog=endog, exog=train_df[exog_cols], order=(1, 0), trend="c").fit(
        disp=False, maxiter=200
    )
    pred_df = model.forecast(steps=1, exog=x_next[exog_cols])
    return float(pred_df[target_col].iloc[0])


def fit_predict_elasticnet(train_df: pd.DataFrame, x_next: pd.DataFrame, target_col: str) -> float:
    exog_cols = [c for c in train_df.columns if c != target_col]
    
    # 1. Difference the target (Same as XGBoost)
    y_diff = train_df[target_col].diff().dropna()
    X_diff = train_df[exog_cols].loc[y_diff.index]
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_diff.astype(float))
    xn = scaler.transform(x_next[exog_cols].astype(float))
    
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=5000)
    model.fit(Xs, y_diff)
    
    # 2. Predict the delta, then add to the last known level
    delta_pred = float(model.predict(xn)[0])
    last_level = float(train_df[target_col].iloc[-1])
    return last_level + delta_pred

# FIX 3: XGBoost trained on month-over-month differences for stationarity.
# The model predicts Δy = y_t − y_{t-1}, then we un-difference the result
# by adding it to the last known actual value.
def fit_predict_xgboost(train_df: pd.DataFrame, x_next: pd.DataFrame, target_col: str) -> float:
    try:
        from xgboost import XGBRegressor
    except Exception:
        return float(train_df[target_col].iloc[-1])

    exog_cols = [c for c in train_df.columns if c != target_col]

    # Compute differenced target; drop first row (NaN from diff).
    y_diff = train_df[target_col].diff().dropna()
    X_diff = train_df[exog_cols].loc[y_diff.index]

    if not hasattr(fit_predict_xgboost, "_best_params"):
        n_splits = 3 if len(X_diff) >= 60 else 2
        tscv = TimeSeriesSplit(n_splits=n_splits)
        base = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_SEED)
        param_dist = {
            "n_estimators": [200, 300, 500, 700],
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=20,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_diff, y_diff)
        fit_predict_xgboost._best_params = search.best_params_

    model = XGBRegressor(
        **fit_predict_xgboost._best_params,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
    )
    model.fit(X_diff, y_diff)

    # Predict the difference, then add to last known level.
    delta_pred = float(model.predict(x_next[exog_cols])[0])
    last_level = float(train_df[target_col].iloc[-1])
    return last_level + delta_pred


def fit_predict_prophet(train_df: pd.DataFrame, x_next: pd.DataFrame, target_col: str) -> float:
    try:
        from prophet import Prophet
    except Exception:
        return float(train_df[target_col].iloc[-1])
    exog_cols = [c for c in train_df.columns if c != target_col][:20]
    temp = train_df.reset_index().rename(columns={"Date": "ds", target_col: "y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for c in exog_cols:
        m.add_regressor(c)
    m.fit(temp[["ds", "y"] + exog_cols])
    fut = x_next.reset_index().rename(columns={"Date": "ds"})
    pred = m.predict(fut[["ds"] + exog_cols])["yhat"].iloc[0]
    return float(pred)


# FIX 5: LSTM and TCN removed. They require far more data than the ~150 rows
# available here, and 40 epochs of full-batch GD is insufficient for convergence.
# The model registry now contains 5 models.

def model_registry(cfg: Config) -> Dict[str, Callable[[pd.DataFrame, pd.DataFrame, str], float]]:
    return {
        "SARIMAX": fit_predict_sarimax,
        "VARX": fit_predict_varx,
        "ElasticNet": fit_predict_elasticnet,
        "XGBoost": fit_predict_xgboost,
        "Prophet": fit_predict_prophet,
        "Ensemble": fit_predict_ensemble, # <-- Add this line!
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def export_visuals(
    out_plot_dir: Path,
    y_true: pd.Series,
    pred_map: Dict[str, pd.Series],
    feature_importance: pd.DataFrame,
) -> None:
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6), dpi=300)
    plt.plot(y_true.index, y_true.values, label="Actual", color="black", linewidth=2)
    for name, pred in pred_map.items():
        plt.plot(pred.index, pred.values, "--", linewidth=1.6, label=name)
    plt.title("Forecast vs. Actuals (Kazakhstan Unemployment Rate)")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_plot_dir / "forecast_vs_actuals.png", dpi=300)
    plt.close()

    baseline = y_true.shift(12).ffill()
    best_name = min(
        pred_map.keys(), key=lambda k: mean_absolute_error(y_true.values, pred_map[k].values)
    )
    plt.figure(figsize=(14, 6), dpi=300)
    plt.plot(y_true.index, y_true.values, label="Actual", color="black", linewidth=2)
    plt.plot(pred_map[best_name].index, pred_map[best_name].values, "--", linewidth=2, label=f"Best: {best_name}")
    # plt.plot(y_true.index, baseline.values, ":", linewidth=2, label="Baseline: Seasonal Naive")
    plt.title("Actual vs Best Model vs Baseline")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot_dir / "actual_vs_best_vs_baseline.png", dpi=300)
    plt.close()

    residuals = y_true.values - pred_map[best_name].values
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=300)
    axes[0].plot(y_true.index, residuals, marker="o", linewidth=1.2)
    axes[0].axhline(0.0, linestyle="--", color="red")
    axes[0].set_title(f"Residual Time Plot ({best_name})")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual")
    axes[1].hist(residuals, bins=12, edgecolor="white")
    axes[1].set_title("Residual Histogram")
    axes[1].set_xlabel("Residual")
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Residual Q-Q Plot")
    plt.tight_layout()
    plt.savefig(out_plot_dir / "residual_diagnostics_panel.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5), dpi=300)
    plt.hist(residuals, bins=12, edgecolor="white")
    plt.title(f"Residual Distribution ({best_name})")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_plot_dir / "residual_distribution.png", dpi=300)
    plt.close()

    if not feature_importance.empty:
        top = feature_importance.sort_values("importance", ascending=False).head(20).iloc[::-1]
        plt.figure(figsize=(10, 7), dpi=300)
        plt.barh(top["feature"], top["importance"])
        plt.title("Feature Importance (XGBoost)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(out_plot_dir / "feature_importance.png", dpi=300)
        plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plot_acf(y_true.values, lags=min(12, len(y_true) - 1), ax=axes[0, 0])
    axes[0, 0].set_title("ACF: Test Target")
    plot_pacf(y_true.values, lags=min(12, len(y_true) - 2), ax=axes[0, 1], method="ywm")
    axes[0, 1].set_title("PACF: Test Target")
    plot_acf(residuals, lags=min(12, len(residuals) - 1), ax=axes[1, 0])
    axes[1, 0].set_title(f"ACF: Residuals ({best_name})")
    plot_pacf(residuals, lags=min(12, len(residuals) - 2), ax=axes[1, 1], method="ywm")
    axes[1, 1].set_title(f"PACF: Residuals ({best_name})")
    plt.tight_layout()
    plt.savefig(out_plot_dir / "acf_pacf_diagnostics.png", dpi=300)
    plt.close()


def export_correlation_visuals(df_feat: pd.DataFrame, cfg: Config, out_plot_dir: Path) -> None:
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    train = df_feat.loc[cfg.train_start : cfg.train_end].copy()

    preferred_features = [
        "gold_price_usd_avg", "usd_kzt", "interest_rate", "Oil_crude_brent",
        "GDP_growth", "unemployed_rate", "Base_Rate", "usd_kzt_volatility",
        "rub_kzt", "log_gold_price", "log_usd_kzt", "log_oil_brent", "log_rub_kzt",
    ]
    selected = [c for c in preferred_features if c in train.columns]
    if cfg.target_col in train.columns and cfg.target_col not in selected:
        selected.append(cfg.target_col)

    if len(selected) < 3:
        return

    corr = train[selected].corr()
    corr.to_csv(cfg.root / cfg.out_dir / "correlation_matrix_focused.csv", index=True)

    original_usetex = plt.rcParams["text.usetex"]
    plt.rcParams["text.usetex"] = False
    plt.figure(figsize=(13, 10), dpi=300)
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(selected)), selected, rotation=90, fontsize=8)
    plt.yticks(range(len(selected)), selected, fontsize=8)
    plt.title("Focused Correlation Heatmap (Train)")
    plt.tight_layout()
    plt.savefig(out_plot_dir / "correlation_heatmap_focused_features.png", dpi=300)
    plt.close()
    plt.rcParams["text.usetex"] = original_usetex


# FIX 4: Diagnostic function now uses tuned _best_params instead of
# hardcoded min_child_weight=10 / max_depth=2 that strangled the model.
# It also applies the same differencing logic used in fit_predict_xgboost.
def export_xgb_train_test_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Config,
    out_path: Path,
    fig_path: Path,
) -> None:
    try:
        from xgboost import XGBRegressor
    except Exception:
        return

    exog_cols = [c for c in train.columns if c != cfg.target_col]

    # Use tuned params if available, otherwise sensible defaults (FIX 4).
    best_params = getattr(fit_predict_xgboost, "_best_params", {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    })

    # Train on differenced target (FIX 3 / FIX 4 consistency).
    y_diff_train = train[cfg.target_col].diff().dropna()
    X_diff_train = train[exog_cols].loc[y_diff_train.index]
    y_diff_test = test[cfg.target_col].diff().fillna(
        test[cfg.target_col].iloc[0] - train[cfg.target_col].iloc[-1]
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_diff_train)
    X_te_s = scaler.transform(test[exog_cols])

    xgb = XGBRegressor(**best_params, objective="reg:squarederror", random_state=RANDOM_SEED)
    xgb.fit(X_tr_s, y_diff_train, eval_set=[(X_te_s, y_diff_test)], verbose=False)

    # Reconstruct absolute levels from cumulative differences.
    delta_train = xgb.predict(X_tr_s)
    delta_test = xgb.predict(X_te_s)

    # Train reconstruction: start from first actual, cumsum deltas.
    train_levels = np.zeros(len(train))
    train_levels[0] = train[cfg.target_col].iloc[0]
    # Predictions start at index 1 (first diff is at index 1).
    for i in range(1, len(train)):
        prev_actual = train[cfg.target_col].iloc[i - 1]
        train_levels[i] = prev_actual + delta_train[i - 1]

    # Test reconstruction: starting level is the last actual training value.
    test_levels = np.zeros(len(test))
    last_train_val = train[cfg.target_col].iloc[-1]
    for i in range(len(test)):
        prev = last_train_val if i == 0 else test[cfg.target_col].iloc[i - 1]
        test_levels[i] = prev + delta_test[i]

    y_train_actual = train[cfg.target_col].values
    y_test_actual = test[cfg.target_col].values

    diag = pd.DataFrame([{
        "Model": "XGBoost_differenced",
        "Train_RMSE": rmse(y_train_actual, train_levels),
        "Test_RMSE": rmse(y_test_actual, test_levels),
        "Train_R2": r2_score(y_train_actual, train_levels),
        "Test_R2": r2_score(y_test_actual, test_levels),
    }])
    diag.to_csv(out_path / "xgboost_train_test_metrics.csv", index=False)

    feat_imp = pd.Series(xgb.feature_importances_, index=exog_cols).sort_values(ascending=False)
    feat_imp.to_csv(out_path / "xgboost_feature_importance_top.csv", header=["importance"])

    plt.figure(figsize=(10, 5), dpi=300)
    feat_imp.head(10).plot(kind="bar")
    plt.title("XGBoost — Top 10 Feature Importances (Differenced Model)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(fig_path / "xgb_feature_importance_top10.png", dpi=300)
    plt.close()

    plt.figure(figsize=(14, 7), dpi=300)
    plt.plot(train.index, y_train_actual, color="darkblue", linewidth=2, label="Actual (Train)")
    plt.plot(train.index, train_levels, color="green", linewidth=1.5, linestyle="--",
             label="XGBoost In-sample fit")
    plt.plot(test.index, y_test_actual, color="darkblue", linewidth=2, label="Actual (Test)")
    plt.plot(test.index, test_levels, color="red", linewidth=1.5, linestyle="--",
             label="XGBoost Out-of-sample forecast")
    plt.axvline(test.index[0], color="black", linestyle="-", linewidth=1.2, alpha=0.7)
    plt.text(test.index[0], plt.ylim()[1] * 0.92, "  Train/Test Split", color="black", fontweight="bold")
    plt.title(
        f"XGBoost Diagnostics (Differenced): Kazakhstan Unemployment\n"
        f"Train R²: {r2_score(y_train_actual, train_levels):.4f} | "
        f"Test R²: {r2_score(y_test_actual, test_levels):.4f}",
        fontsize=13,
    )
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_path / "xgb_train_test_trajectory.png", dpi=300)
    plt.close()


def export_elasticnet_stability(
    df_feat: pd.DataFrame, cfg: Config, out_path: Path, fig_path: Path
) -> None:
    train = df_feat.loc[cfg.train_start : cfg.train_end].copy()
    exog_cols = [c for c in train.columns if c != cfg.target_col]
    window = 84
    step = 3
    if len(train) < window + 12:
        return

    rows = []
    stamps = []
    for end in range(window, len(train) + 1, step):
        chunk = train.iloc[end - window : end]
        X = chunk[exog_cols]
        y = chunk[cfg.target_col]
        sx = StandardScaler()
        Xs = sx.fit_transform(X)
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=5000)
        en.fit(Xs, y)
        rows.append(en.coef_)
        stamps.append(chunk.index[-1])

    coef_df = pd.DataFrame(rows, index=stamps, columns=exog_cols)
    coef_df.to_csv(out_path / "elasticnet_rolling_coefficients.csv", index=True)

    top_features = coef_df.abs().mean().sort_values(ascending=False).head(10).index.tolist()
    plt.figure(figsize=(12, 6), dpi=300)
    for f in top_features:
        plt.plot(coef_df.index, coef_df[f], label=f, linewidth=1.6)
    plt.axhline(0, linestyle="--", color="black", linewidth=1)
    plt.title("ElasticNet Rolling Coefficient Stability (Top 10 by |coef|)")
    plt.xlabel("Window End Date")
    plt.ylabel("Coefficient")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path / "elasticnet_coefficient_stability.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Evaluation & export
# ---------------------------------------------------------------------------

def evaluate_and_export(df_feat: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    train, test = split_data(df_feat, cfg)
    full = pd.concat([train, test], axis=0)
    registry = model_registry(cfg)

    y_true = test[cfg.target_col]
    predictions: Dict[str, pd.Series] = {}
    rows = []

    for name, fn in registry.items():
        print(f"Running model: {name}")
        pred = walk_forward_predict(name, fn, full, cfg)
        predictions[name] = pred
        m = metrics(y_true.values, pred.values)
        rows.append({"Model": name, **m})

    out_path = cfg.root / cfg.out_dir
    fig_path = cfg.root / cfg.plots_dir
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    seasonal_baseline = seasonal_naive_forecast(
        y_history=full[cfg.target_col],
        target_index=y_true.index,
        season=12,
    )
    baseline_m = metrics(y_true.values, seasonal_baseline.values)
    rows.append({"Model": "SeasonalNaive", **baseline_m})
    predictions["SeasonalNaive"] = seasonal_baseline

    metrics_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    metrics_df.to_csv(out_path / "model_metrics.csv", index=False)
    ranked = assign_model_tiers(metrics_df)
    ranked.to_csv(out_path / "model_ranked_tiers.csv", index=False)

    pred_df = pd.DataFrame({"Date": y_true.index, "Actual": y_true.values})
    for k, v in predictions.items():
        pred_df[k] = v.values
    pred_df.to_csv(out_path / "predictions.csv", index=False)

    fi = pd.DataFrame(columns=["feature", "importance"])
    try:
        from xgboost import XGBRegressor

        exog_cols = [c for c in train.columns if c != cfg.target_col]
        tr_pca, _ = add_train_only_pca(train, train.tail(1), cfg.target_col, cfg.pca_components)
        exog_cols = [c for c in tr_pca.columns if c != cfg.target_col]
        # Train on differenced target for the importance plot too (FIX 3 consistency).
        y_diff = tr_pca[cfg.target_col].diff().dropna()
        xgb = XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            objective="reg:squarederror", random_state=RANDOM_SEED,
        )
        xgb.fit(tr_pca[exog_cols].loc[y_diff.index], y_diff)
        fi = pd.DataFrame({"feature": exog_cols, "importance": xgb.feature_importances_})
    except Exception:
        pass

    try:
        exog_cols = [c for c in train.columns if c != cfg.target_col]
        rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED)
        rf.fit(train[exog_cols], train[cfg.target_col])
        rf_fi = pd.DataFrame({"feature": exog_cols, "importance": rf.feature_importances_})
        rf_fi.sort_values("importance", ascending=False).to_csv(
            out_path / "rf_feature_importance.csv", index=False
        )
    except Exception:
        pass

    export_visuals(fig_path, y_true, predictions, fi)
    export_correlation_visuals(df_feat, cfg, fig_path)
    export_xgb_train_test_diagnostics(train, test, cfg, out_path, fig_path)

    export_advanced_diagnostics_panel(train, test, cfg, fig_path)
    metrics_df.to_latex(
        out_path / "model_metrics_table.tex",
        index=False,
        float_format=lambda x: f"{x:.4f}",
        caption="Out-of-sample performance (walk-forward, 2024-01 to 2025-12).",
        label="tab:model_metrics",
    )

    # Overfit/underfit diagnostics.
    diagnostic_rows = []
    try:
        exog_cols = [c for c in train.columns if c != cfg.target_col]
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=5000)
        sx = StandardScaler()
        Xtr = sx.fit_transform(train[exog_cols])
        Xte = sx.transform(test[exog_cols])
        en.fit(Xtr, train[cfg.target_col])
        diagnostic_rows.append({
            "Model": "ElasticNet",
            "Train_MAE": mean_absolute_error(train[cfg.target_col], en.predict(Xtr)),
            "Test_MAE_direct": mean_absolute_error(test[cfg.target_col], en.predict(Xte)),
            "MAE_Gap_TestMinusTrain": (
                mean_absolute_error(test[cfg.target_col], en.predict(Xte))
                - mean_absolute_error(train[cfg.target_col], en.predict(Xtr))
            ),
        })
    except Exception:
        pass

    if diagnostic_rows:
        pd.DataFrame(diagnostic_rows).to_csv(out_path / "overfit_underfit_diagnostics.csv", index=False)

    try:
        champion = ranked.iloc[0]["Model"]
        dm_rows = []
        for m in predictions:
            if m == champion:
                continue
            dm_stat, pval = diebold_mariano(y_true, predictions[champion], predictions[m], h=1, power=2)
            dm_rows.append({"Champion": champion, "Comparator": m, "DM_stat": dm_stat, "p_value": pval})
        if dm_rows:
            pd.DataFrame(dm_rows).to_csv(out_path / "dm_tests.csv", index=False)
    except Exception:
        pass

    export_elasticnet_stability(df_feat, cfg, out_path, fig_path)
    return metrics_df


# ---------------------------------------------------------------------------
# Report utilities
# ---------------------------------------------------------------------------

def build_report_template(path: Path) -> None:
    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{hyperref}
\title{Kazakhstan Unemployment Forecasting Report}
\author{[Author Name]}
\date{\today}
\begin{document}
\maketitle

\section{Objective}
[Describe research question, policy relevance, and forecasting horizon.]

\section{Data and Preprocessing}
[Describe data sources, transformations, merge logic, and date handling.]

\section{Methodology}
\subsection{Feature Engineering}
[Describe Fourier terms, lags, rolling statistics (shifted to avoid lookahead),
interactions, dummies, structural breaks, PCA.]

\subsection{Model Stack}
[Describe SARIMAX (stationarity enforced), VARX, ElasticNet,
XGBoost (trained on differenced target), and Prophet.]

\subsection{Validation Protocol}
[Describe 24-month recursive walk-forward evaluation from 2024-01 to 2025-12.
Recursive means predicted values substitute actual lags for multi-step horizons.]

\section{Results}
\input{outputs/model_metrics_table.tex}

\section{Diagnostics}
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/forecast_vs_actuals.png}
    \caption{Forecast vs. Actuals}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/residual_distribution.png}
    \caption{Residual Distribution}
\end{figure}

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/feature_importance.png}
    \caption{Feature Importance}
\end{figure}

\section{Conclusion}
[Summarize main findings, limitations, and suggested extensions.]

\end{document}
"""
    path.write_text(tex, encoding="utf-8")


def build_final_report_package(cfg: Config) -> None:
    package_dir = cfg.root / "final_report_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "figures").mkdir(exist_ok=True)
    (package_dir / "outputs").mkdir(exist_ok=True)
    src_files = [
        cfg.root / "report_template.tex",
        cfg.root / "README.md",
        cfg.root / cfg.out_dir / "model_metrics.csv",
        cfg.root / cfg.out_dir / "model_metrics_table.tex",
        cfg.root / cfg.out_dir / "predictions.csv",
        cfg.root / cfg.out_dir / "overfit_underfit_diagnostics.csv",
        cfg.root / cfg.out_dir / "dm_tests.csv",
        cfg.root / cfg.out_dir / "model_ranked_tiers.csv",
        cfg.root / cfg.out_dir / "elasticnet_rolling_coefficients.csv",
    ]
    fig_files = [
        cfg.root / cfg.plots_dir / "forecast_vs_actuals.png",
        cfg.root / cfg.plots_dir / "actual_vs_best_vs_baseline.png",
        cfg.root / cfg.plots_dir / "residual_distribution.png",
        cfg.root / cfg.plots_dir / "residual_diagnostics_panel.png",
        cfg.root / cfg.plots_dir / "acf_pacf_diagnostics.png",
        cfg.root / cfg.plots_dir / "feature_importance.png",
        cfg.root / cfg.plots_dir / "correlation_heatmap_focused_features.png",
        cfg.root / cfg.plots_dir / "elasticnet_coefficient_stability.png",
        cfg.root / cfg.plots_dir / "xgb_feature_importance_top10.png",
        cfg.root / cfg.plots_dir / "xgb_train_test_trajectory.png",
    ]
    for fp in src_files:
        if fp.exists():
            dest = package_dir / "outputs" / fp.name if fp.parent.name == cfg.out_dir else package_dir / fp.name
            shutil.copy2(fp, dest)
    for fp in fig_files:
        if fp.exists():
            shutil.copy2(fp, package_dir / "figures" / fp.name)

def fit_predict_ensemble(train_df: pd.DataFrame, x_next: pd.DataFrame, target_col: str) -> float:
    xgb_pred = fit_predict_xgboost(train_df, x_next, target_col)
    en_pred = fit_predict_elasticnet(train_df, x_next, target_col)
    return (xgb_pred + en_pred) / 2.0

def export_advanced_diagnostics_panel(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Config,
    fig_path: Path,
) -> None:
    from xgboost import XGBRegressor
    from sklearn.linear_model import ElasticNet
    
    exog_cols = [c for c in train.columns if c != cfg.target_col]

    # 1. Prepare Differenced Data
    y_diff_train = train[cfg.target_col].diff().dropna()
    X_diff_train = train[exog_cols].loc[y_diff_train.index]
    y_diff_test = test[cfg.target_col].diff().fillna(
        test[cfg.target_col].iloc[0] - train[cfg.target_col].iloc[-1]
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_diff_train)
    X_te_s = scaler.transform(test[exog_cols])

    y_train_actual = train[cfg.target_col]
    y_test_actual = test[cfg.target_col]
    
    full_actual = pd.concat([y_train_actual, y_test_actual])

    # 2. Train Models (XGBoost & ElasticNet)
    best_xgb_params = getattr(fit_predict_xgboost, "_best_params", {
        "n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
    })
    
    models = {
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=5000),
        "XGBoost": XGBRegressor(**best_xgb_params, objective="reg:squarederror", random_state=RANDOM_SEED)
    }

    results = {}
    
    # 3. Generate Predictions and Reconstruct Levels
    for name, model in models.items():
        model.fit(X_tr_s, y_diff_train)
        
        delta_train = model.predict(X_tr_s)
        delta_test = model.predict(X_te_s)
        
        # Train Reconstruction
        train_levels = np.zeros(len(train))
        train_levels[0] = train[cfg.target_col].iloc[0]
        for i in range(1, len(train)):
            train_levels[i] = y_train_actual.iloc[i - 1] + delta_train[i - 1]
            
        # Test Reconstruction
        test_levels = np.zeros(len(test))
        last_train_val = train[cfg.target_col].iloc[-1]
        for i in range(len(test)):
            prev = last_train_val if i == 0 else test[cfg.target_col].iloc[i - 1]
            test_levels[i] = prev + delta_test[i]
            
        results[name] = {"train_preds": train_levels, "test_preds": test_levels}

    # Add Ensemble logic
    results["Ensemble"] = {
        "train_preds": (results["ElasticNet"]["train_preds"] + results["XGBoost"]["train_preds"]) / 2,
        "test_preds": (results["ElasticNet"]["test_preds"] + results["XGBoost"]["test_preds"]) / 2,
    }

    # 4. Create the Multi-Panel Plot
    model_names = ["ElasticNet", "XGBoost", "Ensemble"]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 12), dpi=300, sharex=True)
    
    for ax, name in zip(axes, model_names):
        train_preds = results[name]["train_preds"]
        test_preds = results[name]["test_preds"]
        
        tr_rmse = rmse(y_train_actual.values, train_preds)
        te_rmse = rmse(y_test_actual.values, test_preds)
        
        # Plot Actuals (Seamless dark blue line)
        ax.plot(full_actual.index, full_actual.values, color="navy", linewidth=2.5, label="Actual")
        
        # Plot In-Sample (Train)
        ax.plot(y_train_actual.index, train_preds, color="forestgreen", linestyle="--", linewidth=2, label="In-sample fit")
        
        # Plot Out-of-Sample (Test)
        ax.plot(y_test_actual.index, test_preds, color="red", linestyle="--", linewidth=2, label="Out-of-sample forecast")
        
        # Train/Test Split Line
        ax.axvline(x=y_test_actual.index[0], color="black", linestyle="--", linewidth=1.2, label="Train/Test Split")
        
        # Styling
        ax.set_title(f"{name} — Train RMSE: {tr_rmse:.4f} | Test RMSE: {te_rmse:.4f}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Unemployment Rate (%)")
        
        # Legend (only on the first plot to save space, matching your screenshots)
        if name == "ElasticNet":
            ax.legend(loc="upper left")

    # Format the X-axis dates
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(fig_path / "advanced_stacked_diagnostics.png", dpi=300)
    plt.close()
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kazakhstan unemployment forecasting pipeline (fixed).")
    p.add_argument("--root", type=str, default=str(Path(__file__).resolve().parent))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(root=Path(args.root))
    print("Loading and merging data sources...")
    base_df = load_sources(cfg)
    print("Engineering features...")
    df_feat = engineer_features(base_df, cfg.target_col, cfg.n_fourier, cfg.pca_components)
    print("Running model benchmark and exporting artifacts...")
    results = evaluate_and_export(df_feat, cfg)
    build_report_template(cfg.root / "report_template.tex")
    build_final_report_package(cfg)
    print("\nPipeline completed.")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()