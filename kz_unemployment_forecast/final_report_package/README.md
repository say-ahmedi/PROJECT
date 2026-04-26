# Kazakhstan Unemployment Forecasting Pipeline

Production-ready, end-to-end forecasting pipeline for monthly unemployment rate in Kazakhstan using econometric, machine learning, and deep learning models.

## Scope

- **Target**: `Unemployment Rate (%)`
- **Frequency**: monthly (`MS`)
- **Training period**: `2010-01` to `2023-12`
- **Test horizon**: `2024-01` to `2025-12` (24 months)
- **Validation**: expanding-window walk-forward (one-step ahead across 24 months)

## Data Inputs

The pipeline ingests and merges the following sources from `data/`:

- `indicators/main_ind_ML_imputed.csv` (target)
- `cpi_data/dataset_for_forecast_2026111.xlsx` (`Sheet3`)
- `cpi_data/diploma_dataset.xlsx` (`Sheet3`)
- `unemp/National Bank Base Rate.xlsx` (`base_rate`)
- `unemp/POILBREUSDM.xlsx` (`Monthly`)
- `unemp/USD_TENGE.xlsx` (`exch_rate`)

## Core Data Engineering

1. **Robust date parsing** via `infer_and_standardize_date()`:
   - strips footnotes and markers (e.g., `24.04.2026*`)
   - parses mixed string formats with `dayfirst=True`, `format="mixed"`
   - maps all timestamps to monthly-start index (`MS`)
2. **Merge strategy**: outer joins on `Date`, then monthly resampling (`MS`).
3. **Missing data treatment**: forward-fill followed by backward-fill.

## Feature Engineering (7 required blocks)

1. Fourier annual seasonality terms (`k=1,2,3`)
2. Lags (`t-1, t-2, t-3, t-6, t-12`) for target and key exogenous variables
3. Rolling means/std (3 and 6 months) for oil and USD/KZT
4. Interaction term: `ΔOil × ΔUSD/KZT`
5. Calendar dummies: January, Q1, Q4 indicators
6. Structural break regime dummies: `2015-08`, `2020-03`, `2022-01`
7. PCA factors (3 components) from CPI-related sub-indices

## Model Benchmark Suite (exactly 7 models)

1. SARIMAX (with exogenous regressors)
2. VARX
3. ElasticNet
4. XGBoost
5. Prophet
6. LSTM (PyTorch)
7. TCN (PyTorch)

## Artifacts Produced

After execution:

- `outputs/model_metrics.csv` (MAE, RMSE, MAPE, R² comparison)
- `outputs/predictions.csv` (actuals and per-model forecasts)
- `outputs/model_metrics_table.tex` (LaTeX table)
- `figures/forecast_vs_actuals.png`
- `figures/residual_distribution.png`
- `figures/feature_importance.png`
- `report_template.tex`

## Reproducibility

### 1) Environment

Recommended Python: `3.10+`

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels openpyxl xgboost prophet torch
```

### 2) Run

From project root:

```bash
python unemployment_pipeline.py
```

Optional:

```bash
python unemployment_pipeline.py --epochs 20
```

### 3) Verify outputs

Confirm files exist in:

- `outputs/`
- `figures/`
- `report_template.tex`

## Pipeline Architecture

- `load_sources()` -> ingestion, schema harmonization, date standardization, monthly merge
- `engineer_features()` -> applies all seven feature-engineering strategies
- `walk_forward_predict()` -> expanding-window one-step forecasting over 24 months
- model-specific fit/predict functions -> unified registry interface
- `evaluate_and_export()` -> metrics, predictions, plots, LaTeX table
- `main()` -> orchestration entry point

## Notes for Production Usage

- The pipeline includes graceful fallback for optional libraries (e.g., `Prophet`, `XGBoost`, `torch`) by reverting to last-observation prediction when unavailable.
- For strict production deployment, pin package versions and run under a locked environment (e.g., `requirements.txt` + CI).
