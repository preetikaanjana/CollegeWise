"""
Error Analysis Module for CollegeWise Placement Predictor.

Generates:
1. Actual vs. Predicted scatter plot
2. Residuals vs. Predicted plot
3. Prediction error distribution histogram
4. Top absolute prediction error diagnosis table with institutional domain reasoning.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from src.models.train import prepare_training_data, FEATURE_COLS, TARGET_COL, MODELS_DIR, REPORTS_DIR


def run_error_analysis(df_path: str = r"d:\projects\college\data\processed\colleges_features.parquet"):
    """Perform rigorous residual analysis and outlier diagnosis."""
    print("Running comprehensive ML error analysis...")
    df = pd.read_parquet(df_path)
    X, y, valid_df = prepare_training_data(df)

    # Recreate identical holdout test split
    X_train_val, X_test, y_train_val, y_test, df_train_val, df_test = train_test_split(
        X, y, valid_df, test_size=0.15, random_state=42
    )

    # Load production model
    model_path = os.path.join(MODELS_DIR, "placement_predictor.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model = joblib.load(model_path)
    test_preds = model.predict(X_test)
    residuals = y_test - test_preds
    abs_errors = np.abs(residuals)

    df_test_eval = df_test.copy()
    df_test_eval["actual_median_lpa"] = y_test
    df_test_eval["predicted_median_lpa"] = np.round(test_preds, 2)
    df_test_eval["residual"] = np.round(residuals, 2)
    df_test_eval["absolute_error"] = np.round(abs_errors, 2)
    df_test_eval["pct_error"] = np.round((abs_errors / y_test) * 100, 1)

    # 1. Actual vs. Predicted Plot
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=test_preds, color="#4F46E5", s=90, alpha=0.85, edgecolor="black", linewidth=0.8)
    min_val = min(y_test.min(), test_preds.min()) - 1
    max_val = max(y_test.max(), test_preds.max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], color="#EF4444", linestyle="--", linewidth=1.5, label="Perfect Fit (y = x)")
    plt.title("Actual vs. Predicted Median Package (Holdout Test Set)", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Verified Median Package (₹ LPA)", fontsize=10)
    plt.ylabel("Predicted Median Package (₹ LPA)", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(REPORTS_DIR, "ml_actual_vs_predicted.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"Saved: {p1}")

    # 2. Residuals vs. Predicted Plot (Homoscedasticity Check)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=test_preds, y=residuals, color="#065F46", s=90, alpha=0.85, edgecolor="black", linewidth=0.8)
    plt.axhline(0, color="#EF4444", linestyle="--", linewidth=1.5)
    plt.title("Residuals vs. Predicted Package", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Median Package (₹ LPA)", fontsize=10)
    plt.ylabel("Residual (Actual - Predicted) [₹ LPA]", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p2 = os.path.join(REPORTS_DIR, "ml_residuals.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Saved: {p2}")

    # 3. Residual Distribution Histogram
    plt.figure(figsize=(7, 4.5))
    sns.histplot(residuals, kde=True, color="#7C3AED", bins=10)
    plt.axvline(0, color="#EF4444", linestyle="--", linewidth=1.5)
    plt.title("Distribution of Prediction Errors (Residuals)", fontsize=12, fontweight="bold")
    plt.xlabel("Prediction Error (Actual - Predicted) [₹ LPA]", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p3 = os.path.join(REPORTS_DIR, "ml_residual_distribution.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"Saved: {p3}")

    # 4. Outlier & Error Diagnosis Table
    top_errors = df_test_eval.sort_values(by="absolute_error", ascending=False).head(5)
    print("\n--- Top Prediction Errors on Holdout Test Set ---")
    disp_cols = ["college_name", "state", "ownership", "actual_median_lpa", "predicted_median_lpa", "residual", "absolute_error"]
    print(top_errors[disp_cols].to_string())

    # Save error report to JSON
    err_report_path = os.path.join(REPORTS_DIR, "error_analysis_report.json")
    top_errors[disp_cols].to_json(err_report_path, orient="records", indent=2)
    print(f"Saved error diagnosis data to: {err_report_path}")

    return df_test_eval


if __name__ == "__main__":
    run_error_analysis()
