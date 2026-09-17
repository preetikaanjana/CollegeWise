"""
Unit tests for data preprocessing and cleaner module.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing.cleaner import (
    standardize_state,
    standardize_ownership,
    clean_college_name,
    detect_and_remove_duplicates,
    sanitize_numeric_boundaries,
    analyze_missing_values
)


def test_standardize_state():
    assert standardize_state("nct of delhi") == "Delhi"
    assert standardize_state("tamilnadu") == "Tamil Nadu"
    assert standardize_state("telengana") == "Telangana"
    assert standardize_state("karnataka") == "Karnataka"
    assert standardize_state(None) == "Unknown"


def test_standardize_ownership():
    assert standardize_ownership("Government") == "Government"
    assert standardize_ownership("Central University") == "Government"
    assert standardize_ownership("State Government University") == "Government"
    assert standardize_ownership("Private-Self Financing") == "Private"
    assert standardize_ownership("Deemed to be University(Pvt)") == "Private"
    assert standardize_ownership(None) == "Private"


def test_clean_college_name():
    assert clean_college_name("  \"IIT DELHI\"  ") == "IIT DELHI"
    assert clean_college_name("Anna   University  ") == "Anna University"
    assert clean_college_name(None) == "Unknown Institution"


def test_detect_and_remove_duplicates():
    df = pd.DataFrame([
        {"college_id": "C1", "college_name": "College A", "state": "Delhi"},
        {"college_id": "C1", "college_name": "College A", "state": "Delhi"},  # Duplicate ID
        {"college_id": "C2", "college_name": "College A", "state": "Delhi"},  # Duplicate Name+State
        {"college_id": "C3", "college_name": "College B", "state": "Karnataka"}
    ])
    cleaned, removed = detect_and_remove_duplicates(df)
    assert removed == 2
    assert len(cleaned) == 2


def test_sanitize_numeric_boundaries():
    df = pd.DataFrame([
        {"placement_rate": 105.0, "median_package_lpa": 120.0, "tuition_fee_annual": 50.0},
        {"placement_rate": 85.0, "median_package_lpa": 8.5, "tuition_fee_annual": 120000.0}
    ])
    sanitized = sanitize_numeric_boundaries(df)
    # Out of bounds should become NaN
    assert pd.isna(sanitized.loc[0, "placement_rate"])
    assert pd.isna(sanitized.loc[0, "median_package_lpa"])
    assert pd.isna(sanitized.loc[0, "tuition_fee_annual"])
    # In bounds should remain intact
    assert sanitized.loc[1, "placement_rate"] == 85.0
    assert sanitized.loc[1, "median_package_lpa"] == 8.5
    assert sanitized.loc[1, "tuition_fee_annual"] == 120000.0
