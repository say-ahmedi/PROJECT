import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing


# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR.parent / "data" / "unemployment_rate.xlsx"

SHEET_NAME = "unemployed_rate"
DATE_COL = "Date"
Y_COL = "UnemploymentRate"

TRAIN_START = "2015-01-01"
TRAIN_END   = "2023-12-01"
TEST_START  = "2024-01-01"
TEST_END    = "2025-12-01"

FREQ = "MS"
MAX_LAG = 12
ARIMA_ORDER = (1, 1, 1)
H_FUTURE = 3


# ---------- METRICS ----------
def mase(y_true, y_pred, y_train, m=1):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale

def eval_metrics(y_true, y_pred, y_train):
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float).reindex(y_true.index)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2, mase(y_true, y_pred, y_train, m=1)

def make_supervised(df, y_col, max_lag=12):
    out = df.copy()
    for lag in range(1, max_lag + 1):
        out[f"lag_{lag}"] = out[y_col].shift(lag)

    out["month"] = out.index.month
    out["month_sin"] = np.sin(2*np.pi*out["month"]/12.0)
    out["month_cos"] = np.cos(2*np.pi*out["month"]/12.0)

    out = out.dropna()
    feat_cols = [f"lag_{lag}" for lag in range(1, max_lag + 1)] + ["month_sin", "month_cos"]
    return out, feat_cols


# ---------- RECURSIVE FORECAST FOR ML ----------
def recursive_forecast_ml(model, last_row, steps=3):
    lags = [float(last_row[f"lag_{i}"]) for i in range(1, 13)]  # lag_1..lag_12
    start_date = last_row.name + pd.offsets.MonthBegin(1)
    idx = pd.date_range(start=start_date, periods=steps, freq="MS")

    preds = []
    for d in idx:
        month = d.month
        x = np.array(lags + [np.sin(2*np.pi*month/12.0), np.cos(2*np.pi*month/12.0)], dtype=float).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        lags = [yhat] + lags[:11]

    return pd.Series(preds, index=idx)


# ---------- 1) LOAD ----------
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%m/%d/%y", errors="coerce")
df = df.dropna(subset=[DATE_COL, Y_COL]).sort_values(DATE_COL).set_index(DATE_COL)
df = df.asfreq(FREQ)
df[Y_COL] = df[Y_COL].astype(float).ffill()
df = df.dropna(subset=[Y_COL])

y = df[Y_COL]
y_train_ts = y.loc[TRAIN_START:TRAIN_END]
y_test_ts  = y.loc[TEST_START:TEST_END]

print("Train:", y_train_ts.index.min().date(), "->", y_train_ts.index.max().date(), "n=", len(y_train_ts))
print("Test :", y_test_ts.index.min().date(),  "->", y_test_ts.index.max().date(),  "n=", len(y_test_ts))


# ---------- 2) BASELINE ----------
pred_naive = y.shift(1).loc[y_test_ts.index]
naive_mae, naive_rmse, naive_r2, naive_mase = eval_metrics(y_test_ts, pred_naive, y_train_ts)


# ---------- 3) SUPERVISED (lags + season features) ----------
df_sup, feat_cols = make_supervised(df[[Y_COL]], Y_COL, max_lag=MAX_LAG)

train_df = df_sup.loc[TRAIN_START:TRAIN_END]
test_df  = df_sup.loc[TEST_START:TEST_END]

X_train = train_df[feat_cols]
y_train = train_df[Y_COL]
X_test  = test_df[feat_cols]
y_test  = test_df[Y_COL]


# ---------- 4) LINEAR REGRESSION ----------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = pd.Series(lr.predict(X_test), index=y_test.index)
lr_mae, lr_rmse, lr_r2, lr_mase = eval_metrics(y_test, pred_lr, y_train)


# ---------- 5) ELASTIC NET (CV) ----------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(max_iter=300000, random_state=42))
])

param_grid = {
    "model__alpha": np.logspace(-10, 0, 40),
    "model__l1_ratio": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
}

tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(pipe, param_grid, scoring="neg_root_mean_squared_error", cv=tscv, n_jobs=-1)
grid.fit(X_train, y_train)

best_enet = grid.best_estimator_
print("\nBest ENet params:", grid.best_params_)

pred_enet = pd.Series(best_enet.predict(X_test), index=y_test.index)
enet_mae, enet_rmse, enet_r2, enet_mase = eval_metrics(y_test, pred_enet, y_train)


# ---------- 6) ARIMA ----------
arima_fit = ARIMA(y_train_ts, order=ARIMA_ORDER).fit()
pred_arima = arima_fit.forecast(steps=len(y_test_ts))
pred_arima.index = y_test_ts.index
a_mae, a_rmse, a_r2, a_mase = eval_metrics(y_test_ts, pred_arima, y_train_ts)


# ---------- 7) ETS (optional benchmark) ----------
# keep only one ETS to stay simple (Seasonal 12 if possible, else SES)
try:
    ets_fit = ExponentialSmoothing(
        y_train_ts, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=12,
        initialization_method="estimated"
    ).fit()
    pred_ets = ets_fit.forecast(len(y_test_ts))
    pred_ets.index = y_test_ts.index
    ets_name = "ETS-Seasonal12"
except:
    ets_fit = SimpleExpSmoothing(y_train_ts, initialization_method="estimated").fit()
    pred_ets = ets_fit.forecast(len(y_test_ts))
    pred_ets.index = y_test_ts.index
    ets_name = "ETS-SES"

ets_mae, ets_rmse, ets_r2, ets_mase = eval_metrics(y_test_ts, pred_ets, y_train_ts)


# ---------- 8) RESULTS TABLE ----------
results = pd.DataFrame([
    ["naive",        naive_mae, naive_rmse, naive_r2, naive_mase],
    ["linreg",       lr_mae,    lr_rmse,    lr_r2,    lr_mase],
    ["elastic_net",  enet_mae,  enet_rmse,  enet_r2,  enet_mase],
    [f"arima{ARIMA_ORDER}", a_mae, a_rmse, a_r2, a_mase],
    [ets_name,       ets_mae,   ets_rmse,   ets_r2,   ets_mase],
], columns=["model", "MAE", "RMSE", "R2", "MASE(m=1)"]).sort_values("RMSE")

print("\n=== Model comparison (sorted by RMSE) ===")
print(results.to_string(index=False))


plt.plot(df['UnemploymentRate'], color='blue', label='Unemployment rate')
plt.title("Unemployment Rate in Kazakhstan through years.")
plt.xlabel("Date")
plt.ylabel("Percent (%)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("unemployment_rate_kz.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ---------- 7.5) PLOT: TEST FORECASTS vs ACTUAL (best lines only) ----------
# Recommended: plot Actual + Naive + LinReg (2nd best) to keep it readable

plt.figure(figsize=(12, 6))
plt.plot(y_test_ts, label="Actual (test)")
plt.plot(pred_naive, label="Naive (t-1)")
plt.plot(pred_lr.reindex(y_test_ts.index), label="LinReg (lags+season)")
plt.title("Unemployment rate: Forecast vs Actual (Test period 2024–2025)")
plt.xlabel("Date")
plt.ylabel("Percent (%)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig("forecast_test_2024_2025.pdf", format="pdf", bbox_inches="tight")
plt.show()