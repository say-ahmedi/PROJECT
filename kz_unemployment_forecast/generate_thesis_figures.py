"""
Generate all figures and tables needed for the diploma thesis LaTeX document.
This script:
  1. Runs the unemployment pipeline (reuses existing outputs if available)
  2. Performs XGBoost hyperparameter tuning with GridSearchCV and exports results
  3. Generates publication-quality figures for the thesis
"""

from __future__ import annotations
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "predictions.csv", parse_dates=["Date"], index_col="Date")
    return df


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "model_metrics.csv")


def load_rf_importance() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "rf_feature_importance.csv")


# ---------------------------------------------------------------
# Figure 1: Walk-forward forecast vs actuals (all models)
# ---------------------------------------------------------------
def fig_forecast_vs_actuals(preds: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(preds.index, preds["Actual"], color="black", linewidth=2.5, label="Actual", zorder=5)

    model_styles = {
        "ElasticNet": {"color": "#2196F3", "ls": "-", "lw": 1.8},
        "XGBoost": {"color": "#FF5722", "ls": "--", "lw": 1.8},
        "Ensemble": {"color": "#4CAF50", "ls": "-.", "lw": 1.8},
        "SARIMAX": {"color": "#9C27B0", "ls": ":", "lw": 1.5},
        "VARX": {"color": "#FF9800", "ls": ":", "lw": 1.5},
        "Prophet": {"color": "#795548", "ls": ":", "lw": 1.5},
    }

    for model, style in model_styles.items():
        if model in preds.columns:
            ax.plot(preds.index, preds[model], label=model, **style)

    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title("Walk-Forward Out-of-Sample Forecasts vs. Actual Unemployment Rate (2024--2025)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_forecast_vs_actuals.png")
    plt.close(fig)
    print("[OK] thesis_forecast_vs_actuals.png")


# ---------------------------------------------------------------
# Figure 2: Model comparison bar chart (RMSE, MAE, R²)
# ---------------------------------------------------------------
def fig_model_comparison_bars(metrics: pd.DataFrame):
    models_order = ["ElasticNet", "XGBoost", "Ensemble", "SARIMAX", "VARX", "Prophet", "SeasonalNaive"]
    df = metrics.set_index("Model").loc[[m for m in models_order if m in metrics["Model"].values]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#795548", "#607D8B"]

    # MAE
    axes[0].barh(df.index[::-1], df["MAE"][::-1], color=colors[:len(df)][::-1])
    axes[0].set_xlabel("MAE")
    axes[0].set_title("Mean Absolute Error")
    for i, v in enumerate(df["MAE"][::-1]):
        axes[0].text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=9)

    # RMSE
    axes[1].barh(df.index[::-1], df["RMSE"][::-1], color=colors[:len(df)][::-1])
    axes[1].set_xlabel("RMSE")
    axes[1].set_title("Root Mean Squared Error")
    for i, v in enumerate(df["RMSE"][::-1]):
        axes[1].text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=9)

    # R²
    axes[2].barh(df.index[::-1], df["R2"][::-1], color=colors[:len(df)][::-1])
    axes[2].set_xlabel("$R^2$")
    axes[2].set_title("Coefficient of Determination ($R^2$)")
    for i, v in enumerate(df["R2"][::-1]):
        axes[2].text(max(v + 0.01, 0.01), i, f"{v:.4f}", va="center", fontsize=9)

    fig.suptitle("Comparative Performance Metrics Across All Models", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_model_comparison_bars.png")
    plt.close(fig)
    print("[OK] thesis_model_comparison_bars.png")


# ---------------------------------------------------------------
# Figure 3: Residual analysis for top 3 models
# ---------------------------------------------------------------
def fig_residual_analysis(preds: pd.DataFrame):
    top_models = ["ElasticNet", "XGBoost", "Ensemble"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for i, model in enumerate(top_models):
        residuals = preds["Actual"] - preds[model]

        # Time-series residual plot
        axes[0, i].plot(preds.index, residuals, color=colors[i], linewidth=1.2)
        axes[0, i].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0, i].set_title(f"{model} Residuals")
        axes[0, i].set_ylabel("Residual (pp)")
        axes[0, i].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        axes[0, i].tick_params(axis="x", rotation=45)
        axes[0, i].grid(True, alpha=0.3)

        # Residual histogram
        axes[1, i].hist(residuals, bins=12, color=colors[i], alpha=0.7, edgecolor="black")
        axes[1, i].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1, i].set_xlabel("Residual (pp)")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].set_title(f"{model} Distribution")

        mu, sigma = residuals.mean(), residuals.std()
        axes[1, i].text(0.05, 0.95, f"$\\mu$={mu:.4f}\n$\\sigma$={sigma:.4f}",
                        transform=axes[1, i].transAxes, fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Residual Diagnostics for Top-Performing Models", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_residual_analysis.png")
    plt.close(fig)
    print("[OK] thesis_residual_analysis.png")


# ---------------------------------------------------------------
# Figure 4: Feature importance comparison (RF)
# ---------------------------------------------------------------
def fig_feature_importance(rf_imp: pd.DataFrame):
    top_n = 15
    rf_top = rf_imp.nlargest(top_n, "importance")

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = range(len(rf_top))
    bars = ax.barh(y_pos, rf_top["importance"].values[::-1], color="#2196F3", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    labels = [f.replace("Unemployment Rate (%)", "Unemp Rate").replace("_", " ") for f in rf_top["feature"].values[::-1]]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Feature Importance (Gini Impurity Reduction)")
    ax.set_title("Random Forest: Top 15 Feature Importances")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, rf_top["importance"].values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_feature_importance_rf.png")
    plt.close(fig)
    print("[OK] thesis_feature_importance_rf.png")


# ---------------------------------------------------------------
# Figure 5: XGBoost hyperparameter tuning (GridSearchCV heatmap)
# ---------------------------------------------------------------
def fig_xgboost_tuning():
    from unemployment_pipeline import load_sources, engineer_features, Config, add_train_only_pca

    cfg = Config(root=ROOT)
    raw = load_sources(cfg)
    df = engineer_features(raw, cfg.target_col, cfg.n_fourier, cfg.pca_components)

    train = df.loc[cfg.train_start:cfg.train_end].copy()
    test = df.loc[cfg.test_start:cfg.test_end].copy()

    exog_cols = [c for c in train.columns if c != cfg.target_col]

    y_diff = train[cfg.target_col].diff().dropna()
    X_diff = train[exog_cols].loc[y_diff.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_diff)

    # Grid search over key hyperparameters
    param_grid = {
        "n_estimators": [100, 200, 300, 500, 700],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    n_splits = 3 if len(X_diff) >= 60 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)

    base = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_grid,
        n_iter=80,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    search.fit(X_scaled, y_diff)

    best_params = search.best_params_
    print(f"[XGBoost] Best params: {best_params}")
    print(f"[XGBoost] Best CV MAE: {-search.best_score_:.6f}")

    # Save best params
    pd.DataFrame([best_params]).to_csv(OUT_DIR / "xgboost_best_params.csv", index=False)

    # Save CV results
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(OUT_DIR / "xgboost_cv_results.csv", index=False)

    # --- Figure 5a: Learning rate vs n_estimators heatmap ---
    # Extract unique lr and n_est values from the search
    results_df = cv_results[["param_learning_rate", "param_n_estimators", "mean_test_score"]].copy()
    results_df.columns = ["lr", "n_est", "score"]
    results_df["score"] = -results_df["score"]  # flip sign back to MAE

    pivot = results_df.groupby(["lr", "n_est"])["score"].mean().reset_index()
    pivot_table = pivot.pivot(index="lr", columns="n_est", values="score")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Heatmap
    if pivot_table.shape[0] > 1 and pivot_table.shape[1] > 1:
        valid_pivot = pivot_table.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if valid_pivot.shape[0] > 1 and valid_pivot.shape[1] > 1:
            im = axes[0].imshow(valid_pivot.values, cmap="YlOrRd_r", aspect="auto")
            axes[0].set_xticks(range(len(valid_pivot.columns)))
            axes[0].set_xticklabels(valid_pivot.columns, fontsize=9)
            axes[0].set_yticks(range(len(valid_pivot.index)))
            axes[0].set_yticklabels([f"{v:.3f}" for v in valid_pivot.index], fontsize=9)
            axes[0].set_xlabel("n_estimators")
            axes[0].set_ylabel("learning_rate")
            axes[0].set_title("Mean CV MAE: Learning Rate vs. n\\_estimators")
            plt.colorbar(im, ax=axes[0], label="MAE")
        else:
            axes[0].text(0.5, 0.5, "Insufficient grid coverage", ha="center", va="center", transform=axes[0].transAxes)
            axes[0].set_title("Mean CV MAE: Learning Rate vs. n\\_estimators")
    else:
        axes[0].text(0.5, 0.5, "Insufficient grid coverage", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("Mean CV MAE: Learning Rate vs. n\\_estimators")

    # --- Figure 5b: CV score distribution across iterations ---
    sorted_scores = np.sort(-cv_results["mean_test_score"].values)
    axes[1].plot(range(1, len(sorted_scores)+1), sorted_scores, "o-", color="#FF5722", markersize=4, linewidth=1)
    axes[1].axhline(-search.best_score_, color="green", linestyle="--", linewidth=1.5, label=f"Best MAE = {-search.best_score_:.6f}")
    axes[1].set_xlabel("Configuration Rank")
    axes[1].set_ylabel("Mean CV MAE")
    axes[1].set_title("XGBoost Hyperparameter Search: Ranked CV Scores")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("XGBoost Hyperparameter Tuning Results", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_xgboost_tuning.png")
    plt.close(fig)
    print("[OK] thesis_xgboost_tuning.png")

    # --- Figure: XGBoost train vs test trajectory ---
    best_model = XGBRegressor(**best_params, objective="reg:squarederror", random_state=RANDOM_SEED)

    y_diff_test = test[cfg.target_col].diff().dropna()
    X_test = test[exog_cols]
    X_test_scaled = scaler.transform(X_test)

    best_model.fit(X_scaled, y_diff)

    delta_train = best_model.predict(X_scaled)
    delta_test = best_model.predict(X_test_scaled)

    train_levels = np.zeros(len(train))
    train_levels[0] = train[cfg.target_col].iloc[0]
    for i in range(1, len(train)):
        idx = i - 1
        if idx < len(delta_train):
            train_levels[i] = train[cfg.target_col].iloc[i-1] + delta_train[idx]
        else:
            train_levels[i] = train[cfg.target_col].iloc[i]

    test_levels = np.zeros(len(test))
    test_levels[0] = train[cfg.target_col].iloc[-1] + delta_test[0]
    for i in range(1, len(test)):
        test_levels[i] = test[cfg.target_col].iloc[i-1] + delta_test[i]

    train_r2 = r2_score(train[cfg.target_col].values, train_levels)
    test_r2 = r2_score(test[cfg.target_col].values, test_levels)
    train_rmse = math.sqrt(mean_squared_error(train[cfg.target_col].values, train_levels))
    test_rmse = math.sqrt(mean_squared_error(test[cfg.target_col].values, test_levels))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train.index, train[cfg.target_col], color="darkblue", linewidth=2, label="Actual (Train)")
    ax.plot(train.index, train_levels, color="green", linewidth=1.5, linestyle="--", label="XGBoost Fit (Train)")
    ax.plot(test.index, test[cfg.target_col], color="darkblue", linewidth=2, label="Actual (Test)")
    ax.plot(test.index, test_levels, color="red", linewidth=1.5, linestyle="--", label="XGBoost Forecast (Test)")
    ax.axvline(test.index[0], color="black", linestyle="-", linewidth=1.2, alpha=0.7)
    ax.text(test.index[0], ax.get_ylim()[1] * 0.98, "  Train/Test Split", color="black", fontweight="bold", fontsize=10)
    ax.set_title(f"XGBoost (Tuned): Train R²={train_r2:.4f}, Test R²={test_r2:.4f} | Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_xgb_train_test.png")
    plt.close(fig)
    print("[OK] thesis_xgb_train_test.png")

    # XGBoost feature importance from tuned model
    feat_imp = pd.Series(best_model.feature_importances_, index=exog_cols).sort_values(ascending=False)
    feat_imp_df = feat_imp.reset_index()
    feat_imp_df.columns = ["feature", "importance"]
    feat_imp_df.to_csv(OUT_DIR / "xgboost_tuned_feature_importance.csv", index=False)

    top_feats = feat_imp_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = [f.replace("Unemployment Rate (%)", "Unemp Rate").replace("_", " ") for f in top_feats["feature"].values[::-1]]
    ax.barh(range(len(top_feats)), top_feats["importance"].values[::-1], color="#FF5722", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost (Tuned): Top 15 Feature Importances")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_feature_importance_xgb.png")
    plt.close(fig)
    print("[OK] thesis_feature_importance_xgb.png")

    # Save tuning diagnostics summary
    tuning_summary = {
        "Best_MAE_CV": -search.best_score_,
        "Train_RMSE": train_rmse,
        "Test_RMSE": test_rmse,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        **best_params,
    }
    pd.DataFrame([tuning_summary]).to_csv(OUT_DIR / "xgboost_tuning_summary.csv", index=False)

    return best_params


# ---------------------------------------------------------------
# Figure 6: Correlation heatmap of key features
# ---------------------------------------------------------------
def fig_correlation_heatmap():
    corr = pd.read_csv(OUT_DIR / "correlation_matrix_focused.csv", index_col=0)

    labels_map = {
        "gold_price_usd_avg": "Gold Price",
        "usd_kzt": "USD/KZT",
        "Oil_crude_brent": "Brent Oil",
        "GDP_growth": "GDP Growth",
        "log_gold_price": "log(Gold)",
        "log_usd_kzt": "log(USD/KZT)",
        "log_oil_brent": "log(Brent)",
        "log_rub_kzt": "log(RUB/KZT)",
        "Unemployment Rate (%)": "Unemp. Rate",
    }

    display_labels = [labels_map.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(display_labels)))
    ax.set_yticklabels(display_labels, fontsize=9)

    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_title("Correlation Matrix of Key Macroeconomic Features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_correlation_heatmap.png")
    plt.close(fig)
    print("[OK] thesis_correlation_heatmap.png")


# ---------------------------------------------------------------
# Figure 7: Multi-horizon forecast degradation
# ---------------------------------------------------------------
def fig_horizon_degradation():
    horizon = pd.read_csv(OUT_DIR / "horizon_model_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = horizon["Model"].unique()
    colors_map = {
        "ElasticNet": "#2196F3",
        "SARIMAX": "#9C27B0",
        "VARX": "#FF9800",
        "GRU": "#E91E63",
        "RandomForest": "#4CAF50",
    }

    for model in models:
        mdf = horizon[horizon["Model"] == model]
        color = colors_map.get(model, "#607D8B")
        axes[0].plot(mdf["Horizon"], mdf["RMSE"], "o-", color=color, label=model, linewidth=2, markersize=8)
        axes[1].plot(mdf["Horizon"], mdf["R2"], "o-", color=color, label=model, linewidth=2, markersize=8)

    axes[0].set_xlabel("Forecast Horizon (months)")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE vs. Forecast Horizon")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([3, 6, 12])

    axes[1].set_xlabel("Forecast Horizon (months)")
    axes[1].set_ylabel("$R^2$")
    axes[1].set_title("$R^2$ vs. Forecast Horizon")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks([3, 6, 12])

    fig.suptitle("Multi-Horizon Forecast Performance Degradation", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_horizon_degradation.png")
    plt.close(fig)
    print("[OK] thesis_horizon_degradation.png")


# ---------------------------------------------------------------
# Figure 8: Diebold-Mariano test visualization
# ---------------------------------------------------------------
def fig_dm_tests():
    dm = pd.read_csv(OUT_DIR / "dm_tests.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = []
    for _, row in dm.iterrows():
        if row["p_value"] < 0.01:
            colors.append("#4CAF50")  # strong significance
        elif row["p_value"] < 0.05:
            colors.append("#8BC34A")  # significant
        elif row["p_value"] < 0.10:
            colors.append("#FFC107")  # marginal
        else:
            colors.append("#F44336")  # not significant

    bars = ax.barh(dm["Comparator"], -dm["DM_stat"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Diebold--Mariano Statistic (magnitude)")
    ax.set_title("Statistical Significance: ElasticNet vs. Comparators")

    for bar, (_, row) in zip(bars, dm.iterrows()):
        label = f"p={row['p_value']:.4f}" if row['p_value'] >= 0.001 else f"p<0.001"
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, label,
                va="center", fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="p < 0.01"),
        Patch(facecolor="#8BC34A", label="p < 0.05"),
        Patch(facecolor="#FFC107", label="p < 0.10"),
        Patch(facecolor="#F44336", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "thesis_dm_tests.png")
    plt.close(fig)
    print("[OK] thesis_dm_tests.png")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Generating thesis figures...")
    print("=" * 60)

    preds = load_predictions()
    metrics = load_metrics()
    rf_imp = load_rf_importance()

    fig_forecast_vs_actuals(preds)
    fig_model_comparison_bars(metrics)
    fig_residual_analysis(preds)
    fig_feature_importance(rf_imp)
    fig_correlation_heatmap()
    fig_horizon_degradation()
    fig_dm_tests()

    # XGBoost tuning (this also generates its own figures)
    best_params = fig_xgboost_tuning()

    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)
