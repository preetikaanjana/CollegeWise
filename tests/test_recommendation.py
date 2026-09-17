"""
Unit tests for Recommendation Engine and Explainability Module.
"""

import pandas as pd
import pytest
from src.recommendation.engine import RecommendationEngine, normalize_weights, DIMENSIONS
from src.recommendation.explainer import generate_recommendation_explanation


def test_normalize_weights():
    # Normal weights summing to 100
    weights_100 = {"placement": 35, "campus": 25, "academics": 15, "affordability": 10, "location": 10, "student_life": 5}
    norm_w = normalize_weights(weights_100)
    assert sum(norm_w.values()) == pytest.approx(1.0, rel=1e-3)
    assert norm_w["placement"] == 0.35

    # Unnormalized weights (e.g. sum to 50)
    weights_50 = {"placement": 20, "campus": 10, "academics": 10, "affordability": 5, "location": 5, "student_life": 0}
    norm_50 = normalize_weights(weights_50)
    assert sum(norm_50.values()) == pytest.approx(1.0, rel=1e-3)
    assert norm_50["placement"] == 0.40  # 20 / 50 = 0.40

    # All zeros should fallback to equal weighting
    zeros = {d: 0 for d in DIMENSIONS}
    norm_zeros = normalize_weights(zeros)
    assert sum(norm_zeros.values()) == pytest.approx(1.0, rel=1e-3)


def test_recommendation_engine_filtering():
    engine = RecommendationEngine()
    weights = {"placement": 30, "campus": 20, "academics": 20, "affordability": 10, "location": 10, "student_life": 10}

    # Filter by state
    recs_state = engine.recommend(user_weights=weights, preferred_state="Karnataka", top_k=5)
    assert len(recs_state) > 0
    assert (recs_state["state"] == "Karnataka").all()

    # Filter by ownership
    recs_govt = engine.recommend(user_weights=weights, ownership="Government", top_k=5)
    assert len(recs_govt) > 0
    assert (recs_govt["ownership"] == "Government").all()


def test_recommendation_explainer():
    weights = {"placement": 0.4, "academics": 0.2, "campus": 0.15, "affordability": 0.15, "location": 0.05, "student_life": 0.05}
    col_row = pd.Series({
        "college_name": "Test Engineering Institute",
        "state": "Maharashtra",
        "placement_rate": 88.0,
        "median_package_lpa": 8.0,
        "academic_score": 85.0,
        "affordability_score": 70.0,
        "infrastructure_score": 75.0,
        "tuition_fee_annual": 85000.0,
        "estimated_total_cost_annual": 110000.0,
        "has_placement_data": True
    })

    exp = generate_recommendation_explanation(col_row, weights, {"preferred_state": "Maharashtra", "max_budget": 150000})
    assert len(exp["why_this_college"]) > 0
    # Should mention placement because placement weight is 40% and PR is 88%
    assert any("placement" in r.lower() for r in exp["why_this_college"])
    # Should mention state match
    assert any("maharashtra" in r.lower() for r in exp["why_this_college"])


def test_recommendation_sensitivity_analysis():
    engine = RecommendationEngine()
    
    weights_placement = {"placement": 70, "campus": 10, "academics": 10, "affordability": 5, "location": 3, "student_life": 2}
    weights_affordability = {"placement": 5, "campus": 5, "academics": 10, "affordability": 70, "location": 5, "student_life": 5}
    
    sens = engine.run_sensitivity_analysis(weights_placement, weights_affordability, top_k=10)
    
    assert "recs_a" in sens
    assert "recs_b" in sens
    assert "comparison" in sens
    assert "factor_deltas" in sens
    
    assert len(sens["recs_a"]) > 0
    assert len(sens["recs_b"]) > 0
    
    # Check factor deltas
    assert sens["factor_deltas"]["placement"] < 0  # Decreased in B
    assert sens["factor_deltas"]["affordability"] > 0  # Increased in B
    
    # Check comparison dataframe has delta metrics
    comp = sens["comparison"]
    assert "score_delta" in comp.columns
    assert "rank_shift" in comp.columns
    assert "rank_scenario_a" in comp.columns
    assert "rank_scenario_b" in comp.columns
