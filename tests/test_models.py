"""
Unit tests for Machine Learning model prediction service.
"""

import os
import pandas as pd
import pytest
from src.models.predict import predict_median_package, load_model


def test_model_loading():
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict")


def test_predict_verified_college():
    row = pd.Series({
        "college_name": "Test IIT",
        "median_package_lpa": 14.5,
        "data_source": "NIRF MHRD Disclosures",
        "data_year": 2021
    })
    res = predict_median_package(row)
    assert res["predicted"] is False
    assert res["median_package_lpa"] == 14.5
    assert "NIRF" in res["source"]


def test_predict_unverified_college():
    row = pd.Series({
        "college_name": "Test Private Engineering College",
        "median_package_lpa": None,
        "total_approved_intake": 480,
        "tuition_fee_annual": 120000,
        "faculty_count": 60,
        "student_faculty_ratio": 16.0,
        "academic_score": 70.0,
        "infrastructure_score": 72.0,
        "location_score": 80.0,
        "student_life_score": 70.0,
        "ownership": "Private"
    })
    res = predict_median_package(row)
    assert res["predicted"] is True
    assert res["derived"] is True
    assert 1.5 <= res["median_package_lpa"] <= 25.0
    assert "confidence_lower_lpa" in res
    assert "confidence_upper_lpa" in res
    assert res["confidence_lower_lpa"] <= res["median_package_lpa"] <= res["confidence_upper_lpa"]


def test_model_comparison_artifact():
    import json
    json_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_comparison.json")
    assert os.path.exists(json_path), "model_comparison.json must exist"
    
    with open(json_path, "r") as f:
        metrics = json.load(f)
    
    models = metrics.get("models", {})
    # Verify dummy baseline is included
    assert "Dummy Baseline (Mean)" in models
    dummy_mae = models["Dummy Baseline (Mean)"]["Test_MAE"]
    dummy_r2 = models["Dummy Baseline (Mean)"]["Test_R2"]
    
    # Champion Gradient Boosting should decisively outperform dummy baseline
    assert "Gradient Boosting" in models
    gb_mae = models["Gradient Boosting"]["Test_MAE"]
    gb_r2 = models["Gradient Boosting"]["Test_R2"]
    
    assert gb_mae < dummy_mae, f"Champion MAE ({gb_mae}) must beat Dummy MAE ({dummy_mae})"
    assert gb_r2 > 0.55, f"Champion R2 ({gb_r2}) should exceed 0.55 (got {gb_r2})"
    assert dummy_r2 <= 0.0, f"Dummy baseline R2 must be <= 0 (got {dummy_r2})"
    
    # Verify cross-validation stability metrics exist
    gb_entry = models["Gradient Boosting"]
    assert "CV_MAE_Mean" in gb_entry and "CV_MAE_Std" in gb_entry
    assert "CV_R2_Mean" in gb_entry and "CV_R2_Std" in gb_entry
    assert gb_entry["CV_MAE_Std"] < 1.5, "CV MAE standard deviation should be stable (< 1.5 LPA)"


def test_no_target_leakage():
    """
    Asserts zero target leakage:
    1. Feature schema strictly excludes placement outcomes.
    2. No feature has Pearson correlation > 0.90 with target variable.
    """
    from src.models.train import FEATURE_COLS, TARGET_COL, prepare_training_data
    import pandas as pd
    import numpy as np

    forbidden_targets = ["median_package_lpa", "placement_rate", "placed_students_count", "placement_score"]
    for col in forbidden_targets:
        assert col not in FEATURE_COLS, f"Target-derived column {col} found in FEATURE_COLS (target leakage)!"

    # Check correlation with target in processed features
    features_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "colleges_features.parquet")
    df = pd.read_parquet(features_path)
    X, y, valid_df = prepare_training_data(df)
    
    for feat in FEATURE_COLS:
        col_vals = pd.to_numeric(X[feat], errors="coerce").fillna(X[feat].median())
        corr = np.abs(np.corrcoef(col_vals, y)[0, 1])
        assert corr < 0.90, f"Feature {feat} has suspect high correlation ({corr:.3f}) with target (possible leakage)!"


def test_predict_uncertainty_and_data_sufficiency():
    """Verify uncertainty bounds and data sufficiency warning levels."""
    # 1. High coverage / Verified
    high_row = pd.Series({
        "college_name": "Verified IIT",
        "median_package_lpa": 16.0,
        "data_source": "NIRF Mandatory Disclosure",
        "data_year": 2021
    })
    res_high = predict_median_package(high_row)
    assert res_high["data_sufficiency"] == "🟢 High Data Coverage"
    assert res_high["uncertainty_lpa"] == 0.0
    assert res_high["status"] == "Verified Disclosure"

    # 2. Moderate coverage / Statutory AICTE
    mod_row = pd.Series({
        "college_name": "State Engineering College",
        "median_package_lpa": None,
        "total_approved_intake": 600,
        "tuition_fee_annual": 85000,
        "faculty_count": 45,
        "student_faculty_ratio": 15.0,
        "academic_score": 68.0,
        "infrastructure_score": 70.0,
        "location_score": 60.0,
        "student_life_score": 60.0,
        "ownership": "Government"
    })
    res_mod = predict_median_package(mod_row)
    assert res_mod["data_sufficiency"] == "🟡 Moderate Data Coverage"
    assert res_mod["uncertainty_lpa"] > 0
    assert res_mod["confidence_lower_lpa"] <= res_mod["median_package_lpa"] <= res_mod["confidence_upper_lpa"]

    # 3. Limited coverage / Baseline
    lim_row = pd.Series({
        "college_name": "Unknown Baseline College",
        "median_package_lpa": None,
        "total_approved_intake": None,
        "tuition_fee_annual": None,
        "faculty_count": None,
        "student_faculty_ratio": None,
        "ownership": "Private"
    })
    res_lim = predict_median_package(lim_row)
    assert res_lim["data_sufficiency"] == "⚪ Limited Data Coverage"
    assert res_lim["uncertainty_lpa"] >= res_mod["uncertainty_lpa"]

