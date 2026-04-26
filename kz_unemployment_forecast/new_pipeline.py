# unemployment_pipeline_single_file.py
# -*- coding: utf-8 -*-

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Stats / econometrics
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL

# ML
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional ML/DL libs
XGBOOST_AVAILABLE = True
PROPHET_AVAILABLE = True
TORCH_AVAILABLE = True

try:
    from xgboost import XGBRegressor
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from prophet import Prophet
except Exception:
    PROPHET_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    TORCH_AVAILABLE = False

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# CONFIG
# =========================
@dataclass
class Config:
    # Input files from your message
    main_ind_csv: str = "data/indicators/main_ind_ML_imputed.csv"
    cpi_xlsx: str = "data/cpi_data/dataset_for_forecast_2026111.xlsx"
    cpi_sheet: str = "Sheet3"
    diploma_xlsx: str = "data/cpi_data/diploma_dataset.xlsx"
    diploma_sheet: str = "Sheet3"
    base_rate_xlsx: str = "data/unemp/National Bank Base Rate.xlsx"
    base_rate_sheet: str = "base_rate"
    oil_xlsx: str = "data/unemp/POILBREUSDM.xlsx"
    oil_sheet: str = "Monthly"
    unemp_xlsx: str = "data/unemp/unemployment_rate.xlsx"
    unemp_sheet: str = "rate"
    fx_xlsx: str = "data/unemp/USD_TENGE.xlsx"
    fx_sheet: str = "exch_rate"

    # Pipeline options
    date_col: str = "Date"
    target_col: str = "UnemploymentRate"
    freq: str = "MS"  # monthly start frequency
    test_horizon: int = 24  # last 24 months as walk-forward test window
    min_train_size: int = 96  # first fold train length
    forecast_h: int = 1       # 1-step ahead walk-forward

    # Output
    out_dir: str = "reports"
    fig_dir: str = "reports/figures"


# =========================
# UTILITIES
# =========================
def ensure_dirs(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)


def monthly_from_daily_fx(df_fx: pd.DataFrame) -> pd.DataFrame:
    # expected columns: Date, USD_quant, TENGE
    d = df_fx.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["usd_kzt"] = d["TENGE"] / d["USD_quant"]
    d = d.set_index("Date").sort_index()
    # monthly average exchange rate
    d_m = d["usd_kzt"].resample("MS").mean().to_frame()
    return d_m.reset_index()


def infer_and_standardize_date(df: pd.DataFrame, date_candidates=("Date", "observation_date")) -> pd.DataFrame:
    d = df.copy()
    date_col = None
    for c in date_candidates:
        if c in d.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("No date column found.")
        
    # FIX: Clean the data first. 
    # Convert to string, remove asterisks, and strip any accidental whitespace.
    cleaned_dates = d[date_col].astype(str).str.replace('*', '', regex=False).str.strip()
    
    # Now safely parse the cleaned string
    d["Date"] = pd.to_datetime(cleaned_dates, dayfirst=True, format='mixed')
    
    d = d.drop(columns=[c for c in [date_col] if c != "Date"])
    return d
def safe_log(x):
    return np.log(np.where(x <= 0, np.nan, x))


def add_fourier_terms(df: pd.DataFrame, period: int = 12, K: int = 3) -> pd.DataFrame:
    d = df.copy()
    t = np.arange(len(d))
    for k in range(1, K + 1):
        d[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        d[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return d


def add_calendar_dummies(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = d["Date"].dt.month
    d["quarter"] = d["Date"].dt.quarter
    d["is_january"] = (d["month"] == 1).astype(int)
    d["is_q1"] = (d["quarter"] == 1).astype(int)
    d["is_q4"] = (d["quarter"] == 4).astype(int)
    return d


def add_regime_dummies(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["regime_2015_08"] = (d["Date"] >= pd.Timestamp("2015-08-01")).astype(int)
    d["regime_2020_03"] = (d["Date"] >= pd.Timestamp("2020-03-01")).astype(int)
    d["regime_2022_01"] = (d["Date"] >= pd.Timestamp("2022-01-01")).astype(int)
    return d


def add_lag_features(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    d = df.copy()
    for l in lags:
        d[f"{col}_lag{l}"] = d[col].shift(l)
    return d


def add_rolling_features(df: pd.DataFrame, col: str, windows=(3, 6)) -> pd.DataFrame:
    d = df.copy()
    for w in windows:
        d[f"{col}_ma_{w}"] = d[col].rolling(w).mean()
        d[f"{col}_std_{w}"] = d[col].rolling(w).std()
    return d


def metric_frame(y_true, y_pred) -> Dict[str, float]:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def print_stationarity_report(series: pd.Series, name: str):
    s = series.dropna()
    if len(s) < 24:
        print(f"[WARN] {name}: too short for stationarity tests.")
        return
    try:
        adf_p = adfuller(s, autolag="AIC")[1]
    except Exception:
        adf_p = np.nan
    try:
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan
    print(f"{name:30s} | ADF p={adf_p:.4f} | KPSS p={kpss_p:.4f}")


# =========================
# DATA INGESTION + MERGE
# =========================
def load_and_merge(cfg: Config) -> pd.DataFrame:
    # 1) Core unemployment
    unemp = pd.read_excel(cfg.unemp_xlsx, sheet_name=cfg.unemp_sheet)
    unemp = infer_and_standardize_date(unemp)
    # expected column UnemploymentRate
    if "UnemploymentRate" not in unemp.columns:
        # fallback search
        cand = [c for c in unemp.columns if "unemp" in c.lower()]
        if not cand:
            raise ValueError("Could not find unemployment column in unemployment_rate.xlsx")
        unemp = unemp.rename(columns={cand[0]: "UnemploymentRate"})
    unemp = unemp[["Date", "UnemploymentRate"]]

    # 2) main indicators csv
    main_ind = pd.read_csv(cfg.main_ind_csv)
    main_ind = infer_and_standardize_date(main_ind)
    # rename to safer snake_case
    ren = {}
    for c in main_ind.columns:
        if c == "Date":
            continue
        ren[c] = (
            c.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
            .replace("/", "_")
            .replace("-", "_")
        )
    main_ind = main_ind.rename(columns=ren)

    # Ensure unemployment column there if needed
    if "unemployment_rate_pct" in main_ind.columns:
        main_ind = main_ind.rename(columns={"unemployment_rate_pct": "unemp_rate_main_ind"})

    # 3) cpi/main macro
    cpi = pd.read_excel(cfg.cpi_xlsx, sheet_name=cfg.cpi_sheet)
    cpi = infer_and_standardize_date(cpi)

    # 4) diploma dataset
    diploma = pd.read_excel(cfg.diploma_xlsx, sheet_name=cfg.diploma_sheet)
    diploma = infer_and_standardize_date(diploma)
    # normalize typo
    if "Unemployement" in diploma.columns:
        diploma = diploma.rename(columns={"Unemployement": "Unemployment_diploma"})

    # 5) base rate
    br = pd.read_excel(cfg.base_rate_xlsx, sheet_name=cfg.base_rate_sheet)
    br = infer_and_standardize_date(br)

    # 6) oil
    oil = pd.read_excel(cfg.oil_xlsx, sheet_name=cfg.oil_sheet)
    oil = infer_and_standardize_date(oil)
    if "POILBREUSDM" in oil.columns:
        oil = oil.rename(columns={"POILBREUSDM": "oil_brent_usd"})

    # 7) daily fx -> monthly
    fx = pd.read_excel(cfg.fx_xlsx, sheet_name=cfg.fx_sheet)
    fx = infer_and_standardize_date(fx)
    fx_m = monthly_from_daily_fx(fx)

    # merge all
    dfs = [unemp, main_ind, cpi, diploma, br, oil, fx_m]
    out = dfs[0]
    for i, d in enumerate(dfs[1:], start=1):
        out = out.merge(d, on="Date", how="outer", suffixes=("", f"_dup{i}"))

    out = out.sort_values("Date").drop_duplicates(subset=["Date"])
    out = out.set_index("Date").asfreq(cfg.freq).reset_index()

    # basic imputation
    out = out.ffill().bfill()

    return out


# =========================
# FEATURE ENGINEERING
# =========================
def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()

    # choose canonical unemployment target
    if cfg.target_col not in d.columns:
        if "Unemployment" in d.columns:
            d[cfg.target_col] = d["Unemployment"]
        elif "Unemployment Rate (%)" in d.columns:
            d[cfg.target_col] = d["Unemployment Rate (%)"]
        else:
            raise ValueError(f"Target column `{cfg.target_col}` not found and no fallback found.")

    # common transforms if present
    if "usd_kzt" in d.columns:
        d["log_usd_kzt_auto"] = safe_log(d["usd_kzt"])
        d["d_usd_kzt"] = d["usd_kzt"].diff()
    elif "log_usd_kzt" in d.columns:
        d["d_log_usd_kzt"] = d["log_usd_kzt"].diff()

    if "oil_brent_usd" in d.columns:
        d["log_oil_brent_auto"] = safe_log(d["oil_brent_usd"])
        d["d_oil_brent"] = d["oil_brent_usd"].diff()
    elif "log_oil_brent" in d.columns:
        d["d_log_oil_brent"] = d["log_oil_brent"].diff()

    # Fourier / calendar / regime
    d = add_fourier_terms(d, period=12, K=3)
    d = add_calendar_dummies(d)
    d = add_regime_dummies(d)

    # target lags
    d = add_lag_features(d, cfg.target_col, [1, 2, 3, 6, 12])

    # exog lags if exist
    candidate_lag_cols = [c for c in ["usd_kzt", "oil_brent_usd", "interest_rate", "total_cpi_yoy", "real_wage_yoy_pct"] if c in d.columns]
    for c in candidate_lag_cols:
        d = add_lag_features(d, c, [1, 2, 3, 6])

    # rolling stats for oil/fx if exist
    if "oil_brent_usd" in d.columns:
        d = add_rolling_features(d, "oil_brent_usd", windows=(3, 6))
    if "usd_kzt" in d.columns:
        d = add_rolling_features(d, "usd_kzt", windows=(3, 6))

    # interaction term example
    if "d_oil_brent" in d.columns and "d_usd_kzt" in d.columns:
        d["interaction_dOil_dUSD"] = d["d_oil_brent"] * d["d_usd_kzt"]
    elif "d_log_oil_brent" in d.columns and "d_log_usd_kzt" in d.columns:
        d["interaction_dOil_dUSD"] = d["d_log_oil_brent"] * d["d_log_usd_kzt"]

    # regime interactions
    if "d_usd_kzt" in d.columns:
        d["regime2015_x_dusd"] = d["regime_2015_08"] * d["d_usd_kzt"]

    # Optional PCA from CPI-like columns
    cpi_like = [c for c in d.columns if "cpi" in c.lower() and c != cfg.target_col.lower()]
    # remove target-ish names
    cpi_like = [c for c in cpi_like if "unemploy" not in c.lower()]
    # use at most 13-ish columns but safe
    cpi_like = [c for c in cpi_like if pd.api.types.is_numeric_dtype(d[c])]
    if len(cpi_like) >= 3:
        scaler = StandardScaler()
        Xc = scaler.fit_transform(d[cpi_like].fillna(method="ffill").fillna(method="bfill"))
        pca = PCA(n_components=min(3, len(cpi_like)))
        pcs = pca.fit_transform(Xc)
        for i in range(pcs.shape[1]):
            d[f"cpi_pc{i+1}"] = pcs[:, i]
        print(f"[INFO] PCA added from {len(cpi_like)} CPI-like cols. Explained variance: {pca.explained_variance_ratio_}")

    # Drop rows with NA from lag/rolling construction
    d = d.dropna().reset_index(drop=True)
    return d


# =========================
# MODEL HELPERS
# =========================
def get_feature_cols(df: pd.DataFrame, cfg: Config) -> List[str]:
    exclude = {"Date", cfg.target_col, "month", "quarter"}
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feats


def walk_forward_splits(n: int, min_train: int, horizon: int, test_horizon: int) -> List[Tuple[int, int, int]]:
    # returns (train_end, test_start, test_end)
    # train: [0:train_end), test: [test_start:test_end)
    splits = []
    start_test = n - test_horizon
    for t in range(max(min_train, start_test), n - horizon + 1):
        splits.append((t, t, t + horizon))
    return splits


# =========================
# MODELS
# =========================
def model_seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, cfg: Config):
    # monthly seasonality = 12
    if len(train) >= 12:
        pred = np.repeat(train[cfg.target_col].iloc[-12], len(test))
    else:
        pred = np.repeat(train[cfg.target_col].iloc[-1], len(test))
    return pred


def model_sarimax(train: pd.DataFrame, test: pd.DataFrame, cfg: Config, feature_cols: List[str]):
    y_tr = train[cfg.target_col]
    X_tr = train[feature_cols]
    X_te = test[feature_cols]
    # You can tune (p,d,q)x(P,D,Q,12)
    model = SARIMAX(
        y_tr, exog=X_tr,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = model.fit(disp=False)
    pred = fit.predict(start=len(y_tr), end=len(y_tr) + len(test) - 1, exog=X_te)
    return np.array(pred)


def model_varx(train: pd.DataFrame, test: pd.DataFrame, cfg: Config, feature_cols: List[str]):
    # VAR uses only endogenous matrix; include target + selected exogenous-like variables as joint system
    joint_cols = [cfg.target_col] + feature_cols[: min(6, len(feature_cols))]
    tr = train[joint_cols].copy()
    te = test[joint_cols].copy()

    # keep numerics and finite
    tr = tr.replace([np.inf, -np.inf], np.nan).dropna()
    te = te.replace([np.inf, -np.inf], np.nan).dropna()

    if len(tr) < 24 or len(te) == 0:
        return model_seasonal_naive(train, test, cfg)

    model = VAR(tr)
    fit = model.fit(maxlags=2, ic="aic")
    lag_order = fit.k_ar
    forecast_input = tr.values[-lag_order:]
    fc = fit.forecast(y=forecast_input, steps=len(test))
    pred = fc[:, 0]  # first col = target
    if len(pred) < len(test):
        pred = np.pad(pred, (0, len(test)-len(pred)), mode='edge')
    return pred[:len(test)]


def model_prophet(train: pd.DataFrame, test: pd.DataFrame, cfg: Config, feature_cols: List[str]):
    if not PROPHET_AVAILABLE:
        return model_seasonal_naive(train, test, cfg)

    tr = train.copy()
    te = test.copy()
    trp = tr[["Date", cfg.target_col] + feature_cols].rename(columns={"Date": "ds", cfg.target_col: "y"})
    tep = te[["Date"] + feature_cols].rename(columns={"Date": "ds"})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    for c in feature_cols:
        m.add_regressor(c)
    m.fit(trp)

    fc = m.predict(tep)
    return fc["yhat"].values


def model_xgboost(train: pd.DataFrame, test: pd.DataFrame, cfg: Config, feature_cols: List[str]):
    if not XGBOOST_AVAILABLE:
        # fallback to LASSO
        xtr = train[feature_cols]
        ytr = train[cfg.target_col]
        xte = test[feature_cols]
        mdl = LassoCV(cv=3, random_state=42).fit(xtr, ytr)
        return mdl.predict(xte)

    xtr = train[feature_cols]
    ytr = train[cfg.target_col]
    xte = test[feature_cols]

    mdl = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    mdl.fit(xtr, ytr)
    return mdl.predict(xte)


# ---------- DL ----------
class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len=12):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_t = self.y[idx + self.seq_len]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)


class LSTMReg(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        out = o[:, -1, :]
        out = self.drop(out)
        out = self.fc(out).squeeze(-1)
        return out


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, padding=pad)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, dilation=dilation, padding=pad)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.shape[-1]]  # causal trim
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = out[:, :, :x.shape[-1]]
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCNReg(nn.Module):
    def __init__(self, n_features, channels=32):
        super().__init__()
        self.tcn1 = TCNBlock(n_features, channels, dilation=1)
        self.tcn2 = TCNBlock(channels, channels, dilation=2)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        o = self.tcn1(x)
        o = self.tcn2(o)
        out = o[:, :, -1]
        out = self.fc(out).squeeze(-1)
        return out


def train_dl_model(model, train_loader, epochs=40, lr=1e-3):
    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model


def predict_dl_next(model, X_all_scaled, seq_len):
    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_all_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
        yhat = model(x).cpu().numpy().ravel()[0]
    return yhat


def model_lstm(train: pd.DataFrame, test: pd.DataFrame, cfg: Config, feature_cols: List[str]):
    if not TORCH_AVAILABLE or len(train) < 60:
        return model_seasonal_naive(train, test, cfg)

    seq_len = 12
    preds = []

    for i in range(len(test)):
        hist = pd.concat([train, test.iloc[:i]], axis=0).reset_index(drop=True)
        x = hist[feature_cols].values
        y = hist[cfg.target_col].values

        scaler = StandardScaler()
        x_s = scaler.fit_transform(x)

        if len(hist) <= seq_len + 10:
            preds.append(hist[cfg.target_col].iloc[-1])
            continue

        ds = SeqDataset(x_s, y, seq_len=seq_len)
        dl = DataLoader(ds, batch_size=16, shuffle=True)

        mdl = LSTMReg(n_features=x_s.shape[1], hidden=32, dropout=0.2)
        mdl = train_dl_model(mdl, dl, epochs=30, lr=1e-3)

        # next-step prediction using latest sequence
        yhat = predict_dl_next(mdl, x_s, seq_len=seq_len)
        preds.append(yhat)

    return np.array(preds)


def model_tcn(train: pd.DataFrame, test: pd.DataFrame, cfg: Config, feature_cols: List[str]):
    if not TORCH_AVAILABLE or len(train) < 60:
        return model_seasonal_naive(train, test, cfg)

    seq_len = 12
    preds = []

    for i in range(len(test)):
        hist = pd.concat([train, test.iloc[:i]], axis=0).reset_index(drop=True)
        x = hist[feature_cols].values
        y = hist[cfg.target_col].values

        scaler = StandardScaler()
        x_s = scaler.fit_transform(x)

        if len(hist) <= seq_len + 10:
            preds.append(hist[cfg.target_col].iloc[-1])
            continue

        ds = SeqDataset(x_s, y, seq_len=seq_len)
        dl = DataLoader(ds, batch_size=16, shuffle=True)

        mdl = TCNReg(n_features=x_s.shape[1], channels=32)
        mdl = train_dl_model(mdl, dl, epochs=30, lr=1e-3)

        yhat = predict_dl_next(mdl, x_s, seq_len=seq_len)
        preds.append(yhat)

    return np.array(preds)


# =========================
# EVALUATION + PLOTS
# =========================
def plot_forecast(df_pred: pd.DataFrame, cfg: Config, model_name: str):
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(13, 6), dpi=180)
    plt.plot(df_pred["Date"], df_pred["actual"], label="Actual", linewidth=2.5)
    plt.plot(df_pred["Date"], df_pred["pred"], label=f"{model_name} Pred", linewidth=2.5)
    plt.title(f"{model_name}: Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = os.path.join(cfg.fig_dir, f"{model_name.lower().replace(' ', '_')}_forecast_vs_actual.png")
    plt.savefig(out)
    plt.close()


def plot_residuals(df_pred: pd.DataFrame, cfg: Config, model_name: str):
    d = df_pred.copy()
    d["resid"] = d["actual"] - d["pred"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=180)
    axes[0].plot(d["Date"], d["resid"])
    axes[0].axhline(0, ls="--", c="black")
    axes[0].set_title("Residuals Over Time")
    axes[0].tick_params(axis='x', rotation=45)

    sns.histplot(d["resid"], kde=True, ax=axes[1])
    axes[1].set_title("Residual Distribution")

    sns.boxplot(x=d["resid"], ax=axes[2])
    axes[2].set_title("Residual Boxplot")
    plt.tight_layout()
    out = os.path.join(cfg.fig_dir, f"{model_name.lower().replace(' ', '_')}_residuals.png")
    plt.savefig(out)
    plt.close()


# =========================
# MAIN BENCHMARK
# =========================
def run_pipeline(cfg: Config):
    ensure_dirs(cfg)

    # 1) load
    raw = load_and_merge(cfg)
    print(f"[INFO] Raw merged shape: {raw.shape}")

    # 2) quick stationarity checks
    print("\n=== Stationarity (ADF/KPSS) ===")
    for col in [cfg.target_col, "usd_kzt", "oil_brent_usd", "interest_rate", "total_cpi_yoy"]:
        if col in raw.columns:
            print_stationarity_report(raw[col], col)

    # 3) STL on target
    if cfg.target_col in raw.columns and raw[cfg.target_col].notna().sum() > 36:
        stl_series = raw.set_index("Date")[cfg.target_col].dropna()
        stl = STL(stl_series, period=12, robust=True).fit()
        fig = stl.plot()
        fig.set_size_inches(12, 8)
        fig.savefig(os.path.join(cfg.fig_dir, "stl_target_decomposition.png"), dpi=180)
        plt.close(fig)

    # 4) features
    df = build_features(raw, cfg)
    print(f"[INFO] Feature dataset shape: {df.shape}")

    feature_cols = get_feature_cols(df, cfg)
    print(f"[INFO] Number of feature columns: {len(feature_cols)}")

    # 5) walk-forward setup
    n = len(df)
    splits = walk_forward_splits(
        n=n,
        min_train=cfg.min_train_size,
        horizon=cfg.forecast_h,
        test_horizon=cfg.test_horizon
    )
    print(f"[INFO] Walk-forward folds: {len(splits)}")

    # 6) models (exact 7)
    model_funcs = {
        "SeasonalNaive": lambda tr, te: model_seasonal_naive(tr, te, cfg),
        "SARIMAX": lambda tr, te: model_sarimax(tr, te, cfg, feature_cols),
        "VARX": lambda tr, te: model_varx(tr, te, cfg, feature_cols),
        "Prophet": lambda tr, te: model_prophet(tr, te, cfg, feature_cols),
        "XGBoost": lambda tr, te: model_xgboost(tr, te, cfg, feature_cols),
        "LSTM": lambda tr, te: model_lstm(tr, te, cfg, feature_cols),
        "TCN": lambda tr, te: model_tcn(tr, te, cfg, feature_cols),
    }

    all_metrics = []
    all_preds = {}

    for model_name, fn in model_funcs.items():
        y_true_all, y_pred_all, dates_all = [], [], []
        print(f"\n[RUN] {model_name}")

        for (train_end, test_start, test_end) in splits:
            train = df.iloc[:train_end].copy()
            test = df.iloc[test_start:test_end].copy()
            if len(test) == 0:
                continue

            try:
                pred = fn(train, test)
            except Exception as e:
                print(f"[WARN] {model_name} failed on fold; fallback SeasonalNaive. Error={str(e)[:120]}")
                pred = model_seasonal_naive(train, test, cfg)

            # safety
            pred = np.array(pred).ravel()
            if len(pred) != len(test):
                if len(pred) == 0:
                    pred = np.repeat(train[cfg.target_col].iloc[-1], len(test))
                elif len(pred) < len(test):
                    pred = np.pad(pred, (0, len(test)-len(pred)), mode='edge')
                else:
                    pred = pred[:len(test)]

            y_true_all.extend(test[cfg.target_col].values.tolist())
            y_pred_all.extend(pred.tolist())
            dates_all.extend(test["Date"].tolist())

        met = metric_frame(y_true_all, y_pred_all)
        met["model"] = model_name
        all_metrics.append(met)

        pred_df = pd.DataFrame({"Date": dates_all, "actual": y_true_all, "pred": y_pred_all})
        pred_df = pred_df.drop_duplicates(subset=["Date"]).sort_values("Date")
        all_preds[model_name] = pred_df

        # save model prediction file + plots
        pred_df.to_csv(os.path.join(cfg.out_dir, f"pred_{model_name}.csv"), index=False)
        plot_forecast(pred_df, cfg, model_name)
        plot_residuals(pred_df, cfg, model_name)

    # 7) Save metrics
    metrics_df = pd.DataFrame(all_metrics)[["model", "MAE", "RMSE", "MAPE", "R2"]].sort_values("RMSE")
    metrics_path = os.path.join(cfg.out_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # metrics bar chart
    plt.figure(figsize=(12, 6), dpi=180)
    melt = metrics_df.melt(id_vars="model", var_name="metric", value_name="value")
    sns.barplot(data=melt, x="model", y="value", hue="metric")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.fig_dir, "model_comparison_metrics.png"))
    plt.close()

    print("\n=== Final Metrics ===")
    print(metrics_df.to_string(index=False))
    print(f"\n[INFO] Saved metrics: {metrics_path}")
    print(f"[INFO] Figures folder: {cfg.fig_dir}")


if __name__ == "__main__":
    cfg = Config()
    run_pipeline(cfg)