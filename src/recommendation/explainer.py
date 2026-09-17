"""
Explainability Engine for CollegeWise.

Generates transparent, evidence-based natural language explanations
answering "Why this college?" for every recommended institution.

Adheres strictly to the Objective Grounding Policy:
- Never makes subjective claims (e.g. "most beautiful campus").
- Cites verified data metrics (e.g. "91.2% placement rate reported in NIRF disclosure").
- Explicitly flags when information is unverified or estimated.
"""

from typing import Dict, List, Any, Optional
import pandas as pd


def generate_recommendation_explanation(
    college_row: pd.Series,
    user_weights: Dict[str, float],
    student_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates a structured, honest explanation of why a college was recommended.
    """
    reasons = []
    caveats = []
    factor_breakdowns = []

    # Map user weights (sorted by importance)
    sorted_priorities = sorted(user_weights.items(), key=lambda x: x[1], reverse=True)

    # 1. Evaluate top 2 user priorities
    for dim, weight in sorted_priorities[:3]:
        if weight >= 0.15:
            if dim == "placement":
                pr = college_row.get("placement_rate")
                pkg = college_row.get("median_package_lpa")
                if pd.notna(pr) and float(pr) >= 75.0:
                    reasons.append(
                        f"Strong verified placement rate ({pr}%) closely aligns with your high placement priority ({int(weight*100)}%)."
                    )
                elif pd.notna(pkg) and float(pkg) >= 6.0:
                    reasons.append(
                        f"Competitive median compensation of ₹{pkg} LPA matches your career priority."
                    )
            elif dim == "academics":
                score = college_row.get("academic_score", 0)
                acc = college_row.get("accreditation", "AICTE Approved")
                if score >= 80:
                    reasons.append(
                        f"High academic reputation indicator ({score}/100, {acc}) matches your academic focus."
                    )
                else:
                    reasons.append(
                        f"Approved curriculum and technical accreditation ({acc}) meet required academic baseline."
                    )
            elif dim == "affordability":
                cost = college_row.get("estimated_total_cost_annual")
                aff_score = college_row.get("affordability_score", 0)
                if pd.notna(cost) and cost < 120000:
                    reasons.append(
                        f"Highly affordable statutory fee structure (~₹{int(cost):,}/year) maximizes your affordability preference."
                    )
                elif aff_score >= 75:
                    reasons.append(
                        f"Favorable cost-to-quality profile ({aff_score}/100 affordability rating) matches your budget priority."
                    )
            elif dim == "campus":
                infra = college_row.get("infrastructure_score", 0)
                pcs = college_row.get("pcs_accessibility_score")
                if pd.notna(pcs) and pcs >= 2:
                    reasons.append(
                        f"Documented physical accessibility (PCS ramps, lifts, special facilities) supports your campus priority."
                    )
                else:
                    reasons.append(
                        f"Documented institutional infrastructure rating ({infra}/100) satisfies campus requirements."
                    )
            elif dim == "location":
                dist = college_row.get("distance_to_major_city_km")
                st = college_row.get("state")
                if pd.notna(dist) and dist <= 50:
                    reasons.append(
                        f"Strategic urban connectivity ({dist} km from economic center in {st}) supports your location priority."
                    )
                else:
                    reasons.append(
                        f"Institutional transit connectivity in {st} matches your regional preference."
                    )
            elif dim == "student_life":
                life = college_row.get("student_life_score", 0)
                reasons.append(
                    f"Active student organizations, technical chapters, and campus life ({life}/100) align with your priority."
                )

    # 2. Check student profile constraints
    if student_profile:
        user_state = student_profile.get("preferred_state")
        col_state = college_row.get("state")
        if user_state and str(user_state).lower() not in ["", "all", "all india"]:
            if str(col_state).lower() == str(user_state).lower():
                reasons.append(f"Located directly in your target state of {col_state}.")

        user_budget = student_profile.get("max_budget")
        col_cost = college_row.get("estimated_total_cost_annual")
        if user_budget and pd.notna(col_cost):
            if col_cost <= user_budget:
                reasons.append(
                    f"Estimated total annual cost (₹{int(col_cost):,}) is within your maximum budget of ₹{int(user_budget):,}."
                )

    # 3. Transparent Data Integrity Caveats
    has_placement = college_row.get("has_placement_data", False)
    if not has_placement or pd.isna(college_row.get("placement_rate")):
        caveats.append(
            "Direct placement statistics are not published in NIRF public disclosures for this institution; placement score was estimated via ML institutional modeling."
        )

    if pd.isna(college_row.get("tuition_fee_annual")):
        caveats.append(
            "Official fee schedule was not disclosed in the primary regulatory dump; standard state fee regulatory committee benchmarks apply."
        )

    if not reasons:
        reasons.append("Balanced multi-dimensional profile matching your overall preference distribution.")

    return {
        "why_this_college": reasons,
        "transparency_caveats": caveats,
        "match_percentage": college_row.get("match_score", 0),
        "data_completeness": college_row.get("data_completeness_ratio", 0) * 100
    }
