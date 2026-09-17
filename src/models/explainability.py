"""
Model Explainability & Feature Importance Module for CollegeWise.

Evaluates:
1. Gini Impurity-based Feature Importance (Model Internal)
2. Permutation Importance (Model-Agnostic, Test Data-based)
3. Strict Causal Disclaimers (Predictive Association != Causal Driver)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from src.models.train import prepare_training_data, FEATURE_COLS, MODELS_DIR, REPORTS_DIR


def run_explainability_analysis(df_path: str = r"d:\projects\college\data\processed\colleges_features.parquet"):
    """Compute permutation importance and generate explainability visualizations."""
    print("Running model explainability & permutation importance analysis...")
    df = pd.read_parquet(df_path)
    X, y, valid_df = prepare_training_data(df)

    # Recreate holdout test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model_path = os.path.join(MODELS_DIR, "placement_predictor.joblib")
    model = joblib.load(model_path)

    # 1. Permutation Importance on Holdout Test Set
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=42, scoring="r2"
    )

    perm_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Permutation_Importance_Mean": perm_result.importances_mean,
        "Permutation_Importance_Std": perm_result.importances_std
    }).sort_values(by="Permutation_Importance_Mean", ascending=True)

    # 2. Extract Tree Feature Importance
    underlying_model = model.named_steps["model"]
    tree_importances = {}
    if hasattr(underlying_model, "feature_importances_"):
        tree_importances = dict(zip(FEATURE_COLS, underlying_model.feature_importances_))

    perm_df["Tree_Gini_Importance"] = perm_df["Feature"].map(tree_importances).fillna(0.0)

    # Plot Comparison
    plt.figure(figsize=(9, 5))
    y_pos = np.arange(len(perm_df))
    plt.barh(
        y_pos,
        perm_df["Permutation_Importance_Mean"],
        xerr=perm_df["Permutation_Importance_Std"],
        align='center',
        color="#4F46E5",
        alpha=0.85,
        ecolor="#EF4444",
        capsize=4
    )
    plt.yticks(y_pos, perm_df["Feature"], fontsize=10)
    plt.title("Permutation Feature Importance (Holdout Test Set)\n[Predictive Association, NOT Causation]", fontsize=11, fontweight="bold")
    plt.xlabel("Mean Drop in Test R² When Feature Shuffled", fontsize=10)
    for i, val in enumerate(perm_df["Permutation_Importance_Mean"]):
        plt.annotate(f"{val:.3f}", (max(0, val) + 0.01, i), va='center', fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6, axis="x")
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "ml_permutation_importance.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved: {plot_path}")

    # Summary Output
    print("\n--- Feature Importance Summary ---")
    print(perm_df[["Feature", "Permutation_Importance_Mean", "Tree_Gini_Importance"]].to_string())

    print("\nCAUSAL DISCLAIMER:")
    print("These feature weights describe the statistical dependency of the model's predictions.")
    print("They DO NOT indicate that increasing tuition fees or intake causes higher placement packages.")

    return perm_df


if __name__ == "__main__":
    run_explainability_analysis()
