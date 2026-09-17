"""
Unit tests for feature engineering and dimension scoring module.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.engineering import (
    compute_placement_score,
    compute_academic_score,
    compute_affordability_score,
    compute_infrastructure_score,
    compute_location_score,
    compute_student_life_score,
    compute_confidence_score,
    engineer_features
)


def test_compute_placement_score():
    # Premier institute placement
    score_high = compute_placement_score(placement_rate=95.0, median_package_lpa=16.0, higher_studies_rate=12.0)
    assert score_high is not None
    assert 75.0 <= score_high <= 100.0

    # Moderate placement
    score_mod = compute_placement_score(placement_rate=60.0, median_package_lpa=4.0, higher_studies_rate=5.0)
    assert score_mod is not None
    assert 30.0 <= score_mod <= 70.0

    # Missing placement data should return None (not 0!)
    score_missing = compute_placement_score(placement_rate=None, median_package_lpa=None, higher_studies_rate=None)
    assert score_missing is None


def test_compute_academic_score():
    # Institute of National Importance
    score_ini = compute_academic_score("Institute of National Importance (INI)", "Central University", sfr=12.0, research_lakhs=150.0)
    assert score_ini >= 90.0

    # AICTE approved standard college
    score_std = compute_academic_score("AICTE Approved", "Private-Self Financing", sfr=None, research_lakhs=None)
    assert 55.0 <= score_std <= 75.0


def test_compute_affordability_score():
    # Low cost government college (very affordable)
    score_govt = compute_affordability_score(estimated_total_cost=55000.0, ownership="Government")
    assert score_govt >= 85.0

    # High fee private college
    score_pvt = compute_affordability_score(estimated_total_cost=300000.0, ownership="Private")
    assert score_pvt < 60.0


def test_compute_location_score():
    # Close to metro (< 25km)
    score_close = compute_location_score(dist_metro_km=15.0, state="Delhi")
    assert score_close >= 85.0

    # Remote location
    score_far = compute_location_score(dist_metro_km=250.0, state="Odisha")
    assert score_far <= 65.0

    # State match bonus
    score_matched = compute_location_score(dist_metro_km=15.0, state="Karnataka", preferred_state="Karnataka")
    assert score_matched >= score_close


def test_compute_confidence_score():
    full_row = pd.Series({
        "college_name": "Test College",
        "state": "Delhi",
        "branches": "CSE",
        "tuition_fee_annual": 100000.0,
        "placement_rate": 80.0,
        "median_package_lpa": 6.5,
        "faculty_count": 120,
        "latitude": 28.6
    })
    assert compute_confidence_score(full_row) == 100.0

    empty_row = pd.Series({})
    assert compute_confidence_score(empty_row) == 0.0
