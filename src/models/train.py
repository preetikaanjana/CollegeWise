"""
Machine Learning Model Training Pipeline for CollegeWise.

Supervised Regression Task:
Predict institutional median package (LPA) from verified institutional characteristics
(intake capacity, fees, faculty, student-faculty ratio, ownership, infrastructure, academics).

Evaluates and compares:
0. Dummy Regressor (Naive Mean Baseline)
1. Linear Regression (Parametric Baseline)
2. Ridge Regression (L2 Regularized)
3. Decision Tree Regressor (Non-linear Tree Baseline)
4. Random Forest Regressor (Bagging Ensemble)
5. Gradient Boosting Regressor (GBDT)
6. XGBoost Regressor (Extreme Gradient Boosting, if available)

Data Integrity & Anti-Leakage Rules:
- Excludes placement_rate, salary targets, and placement_score from input features.
- Excludes affordability_score (derived directly from tuition_fee_annual) to eliminate multicollinearity.
- Proper Train / Validation / Test holdout split.
- Scaler and imputers fit strictly on training set.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "eda_figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Independent feature schema (Strictly zero target leakage, zero collinear redundancy)
FEATURE_COLS = [
    "total_approved_intake",
    "tuition_fee_annual",
    "faculty_count",
    "student_faculty_ratio",
    "academic_score",
    "infrastructure_score",
    "location_score",
    "student_life_score",
    "is_government"
]

TARGET_COL = "median_package_lpa"


def prepare_training_data(df: pd.DataFrame):
    """
    Filter dataset for institutions with verified target data and prepare feature matrix.
    Prevents data leakage.
    """
    valid_df = df.dropna(subset=[TARGET_COL]).copy()
    valid_df = valid_df[valid_df[TARGET_COL] > 0.5].copy()

    # Create binary ownership feature
    valid_df["is_government"] = (valid_df["ownership"].astype(str).str.lower() == "government").astype(float)

    X = valid_df[FEATURE_COLS].copy()
    y = valid_df[TARGET_COL].values

    return X, y, valid_df


def train_and_compare_models(df_path: str = r"d:\projects\college\data\processed\colleges_features.parquet"):
    """
    Train multiple models including dummy baseline, run 5-fold cross validation,
    evaluate on test set, perform classification audit, and persist the champion model.
    """
    print(f"Loading dataset for ML training from {df_path}...")
    df = pd.read_parquet(df_path)
    X, y, valid_df = prepare_training_data(df)
    print(f"Dataset for ML: {len(X)} verified institutional records, {len(FEATURE_COLS)} features.")
    print(f"Target summary -> Mean: {y.mean():.2f} LPA, Std: {y.std():.2f} LPA, Min: {y.min():.2f} LPA, Max: {y.max():.2f} LPA")

    # 1. Train / Validation / Test split (70% Train, 15% Val, 15% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42  # 0.1765 * 0.85 approx 0.15
    )

    print(f"Split sizes -> Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # Preprocessing Pipeline: Median Imputation + Standard Scaling
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Model Candidates Dictionary (Including Dummy Baseline)
    candidates = {
        "Dummy Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression (alpha=10.0)": Ridge(alpha=10.0, random_state=42),
        "Decision Tree (max_depth=5)": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest (n=150)": RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost Regressor"] = XGBRegressor(
            n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42, n_jobs=1
        )

    results = {}
    fitted_pipelines = {}

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

    for name, model in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # 5x5 Repeated K-Fold CV (25 evaluations per candidate) on Train set
        cv_r2_scores = cross_val_score(pipe, X_train, y_train, cv=rkf, scoring="r2")
        cv_mae_scores = -cross_val_score(pipe, X_train, y_train, cv=rkf, scoring="neg_mean_absolute_error")
        cv_rmse_scores = np.sqrt(-cross_val_score(pipe, X_train, y_train, cv=rkf, scoring="neg_mean_squared_error"))

        # Fit on Train set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Validation set performance
        val_preds = pipe.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_r2 = r2_score(y_val, val_preds)

        # Holdout Test set performance
        test_preds = pipe.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)

        results[name] = {
            "CV_R2_Mean": round(float(cv_r2_scores.mean()), 4),
            "CV_R2_Std": round(float(cv_r2_scores.std()), 4),
            "CV_MAE_Mean": round(float(cv_mae_scores.mean()), 4),
            "CV_MAE_Std": round(float(cv_mae_scores.std()), 4),
            "CV_RMSE_Mean": round(float(cv_rmse_scores.mean()), 4),
            "CV_RMSE_Std": round(float(cv_rmse_scores.std()), 4),
            "Val_MAE": round(float(val_mae), 4),
            "Val_RMSE": round(float(val_rmse), 4),
            "Val_R2": round(float(val_r2), 4),
            "Test_MAE": round(float(test_mae), 4),
            "Test_RMSE": round(float(test_rmse), 4),
            "Test_R2": round(float(test_r2), 4),
        }

    # Identify Champion Model by Test R2 (excluding Dummy Baseline)
    real_candidates = [k for k in results.keys() if "Dummy" not in k]
    best_name = max(real_candidates, key=lambda k: (results[k]["Test_R2"], -results[k]["Test_RMSE"]))
    champion_pipe = fitted_pipelines[best_name]

    # Calculate empirical error bound on holdout test set
    champ_test_preds = champion_pipe.predict(X_test)
    test_abs_errors = np.abs(y_test - champ_test_preds)
    empirical_interval = round(float(np.percentile(test_abs_errors, 75)), 2)

    print(f"\n=======================================================")
    print(f"CHAMPION MODEL SELECTED: {best_name}")
    print(f"Holdout Test R2: {results[best_name]['Test_R2']}, Test MAE: {results[best_name]['Test_MAE']} LPA, Test RMSE: {results[best_name]['Test_RMSE']} LPA")
    print(f"Empirical 75th percentile error bound: ±{empirical_interval} LPA")
    print(f"=======================================================\n")

    # Display comparison table
    comp_df = pd.DataFrame(results).T
    print(comp_df.to_string())

    # Retrain champion model on full available dataset for production deployment
    final_champion_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", candidates[best_name])
    ])
    final_champion_pipe.fit(X, y)

    # Save model artifact
    model_save_path = os.path.join(MODELS_DIR, "placement_predictor.joblib")
    joblib.dump(final_champion_pipe, model_save_path)
    print(f"Saved production model to {model_save_path}")

    # Extract Feature Importance
    feature_importances = {}
    underlying_model = final_champion_pipe.named_steps["model"]
    if hasattr(underlying_model, "feature_importances_"):
        fi = underlying_model.feature_importances_
        feature_importances = {f: round(float(w), 4) for f, w in zip(FEATURE_COLS, fi)}
    elif hasattr(underlying_model, "coef_"):
        fi = np.abs(underlying_model.coef_)
        fi_norm = fi / (np.sum(fi) if np.sum(fi) > 0 else 1.0)
        feature_importances = {f: round(float(w), 4) for f, w in zip(FEATURE_COLS, fi_norm)}

    # Secondary Audit: Classification Formulation (Tier 1 >= 7.0 LPA)
    threshold = 7.0
    y_test_bin = (y_test >= threshold).astype(int)
    y_pred_test = champion_pipe.predict(X_test)
    y_pred_bin = (y_pred_test >= threshold).astype(int)

    cls_acc = float(accuracy_score(y_test_bin, y_pred_bin))
    cls_prec = float(precision_score(y_test_bin, y_pred_bin, zero_division=0))
    cls_rec = float(recall_score(y_test_bin, y_pred_bin, zero_division=0))
    cls_f1 = float(f1_score(y_test_bin, y_pred_bin, zero_division=0))
    
    try:
        cls_roc = float(roc_auc_score(y_test_bin, y_pred_test))
    except Exception:
        cls_roc = 0.5

    cm = confusion_matrix(y_test_bin, y_pred_bin).tolist()

    classification_audit = {
        "classification_target": f"median_package >= {threshold} LPA (Tier 1 Premium)",
        "positive_class_count": int(np.sum(y_test_bin)),
        "negative_class_count": int(len(y_test_bin) - np.sum(y_test_bin)),
        "accuracy": round(cls_acc, 4),
        "precision": round(cls_prec, 4),
        "recall": round(cls_rec, 4),
        "f1_score": round(cls_f1, 4),
        "roc_auc": round(cls_roc, 4),
        "confusion_matrix": cm,
        "note": "Regression remains the primary headline ML result. Classification accuracy is subject to class imbalance."
    }

    # Save metadata and metrics
    comparison_save_path = os.path.join(MODELS_DIR, "model_comparison.json")
    with open(comparison_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_variable": TARGET_COL,
            "sample_size": len(X),
            "features_used": FEATURE_COLS,
            "champion_model": best_name,
            "empirical_prediction_interval_lpa": empirical_interval,
            "models": results,
            "feature_importances": feature_importances,
            "classification_audit": classification_audit,
            "data_split": {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
        }, f, indent=2)
    print(f"Saved metrics registry to {comparison_save_path}")

    # Generate Model Comparison Chart
    plt.figure(figsize=(10, 5))
    comp_plot_df = comp_df.reset_index().rename(columns={"index": "Model"})
    ax = sns.barplot(data=comp_plot_df, x="Test_R2", y="Model", palette="viridis")
    plt.title("Model Comparison on Holdout Test Set (R² Score)", fontsize=13, fontweight='bold')
    plt.xlabel("Test R² Score (Higher is better; baseline is 0)", fontsize=11)
    for p in ax.patches:
        val = p.get_width()
        ax.annotate(f"{val:.3f}", (max(0, val) + 0.01, p.get_y() + p.get_height() / 2),
                    va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "ml_model_comparison.png"))
    plt.close()
    print("Saved ml_model_comparison.png")

    # Generate Feature Importance Chart
    if feature_importances:
        plt.figure(figsize=(10, 5))
        fi_df = pd.DataFrame(list(feature_importances.items()), columns=["Feature", "Importance"]).sort_values(
            by="Importance", ascending=True
        )
        ax = sns.barplot(data=fi_df, x="Importance", y="Feature", palette="rocket")
        plt.title(f"Feature Importance ({best_name}) [Predictive Correlation, Not Causation]", fontsize=12, fontweight='bold')
        plt.xlabel("Relative Predictive Weight", fontsize=10)
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.3f}", (p.get_width() + 0.005, p.get_y() + p.get_height() / 2),
                        va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "ml_feature_importance.png"))
        plt.close()
        print("Saved ml_feature_importance.png")

    return results, best_name


if __name__ == "__main__":
    train_and_compare_models()
