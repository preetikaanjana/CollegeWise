"""
Preprocessing Pipeline for CollegeWise.

Orchestrates cleaning, standardization, deduplication, schema validation,
and saves the cleaned production dataset.
"""

import os
import pandas as pd
from src.preprocessing.cleaner import (
    standardize_state,
    standardize_ownership,
    clean_college_name,
    detect_and_remove_duplicates,
    sanitize_numeric_boundaries,
    analyze_missing_values
)


def run_preprocessing_pipeline(
    raw_csv_path: str = r"d:\projects\college\data\raw\colleges_raw.csv",
    output_parquet_path: str = r"d:\projects\college\data\processed\colleges_clean.parquet",
    output_csv_path: str = r"d:\projects\college\data\processed\colleges_clean.csv"
) -> pd.DataFrame:
    """Execute end-to-end preprocessing pipeline."""
    print(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    initial_rows = len(df)
    print(f"Initial raw rows: {initial_rows}")

    # 1. Clean Names & Text
    df["college_name"] = df["college_name"].apply(clean_college_name)
    df["state"] = df["state"].apply(standardize_state)
    df["ownership"] = df["institution_type"].apply(standardize_ownership)

    # 2. Deduplication
    df_dedup, removed = detect_and_remove_duplicates(df)
    print(f"Removed {removed} duplicates. Remaining: {len(df_dedup)}")

    # 3. Numeric Sanitation & Domain Boundary Checking
    df_clean = sanitize_numeric_boundaries(df_dedup)

    # 4. Analyze Missing Values
    missing_report = analyze_missing_values(df_clean)
    print("\n--- Top Missing Value Columns ---")
    print(missing_report.head(10)[["column", "missing_percentage", "available_count"]].to_string())

    # Automated Data Integrity & Boundary Validation Checks
    valid_pkgs = df_clean["median_package_lpa"].dropna()
    assert len(valid_pkgs) >= 20, f"Expected verified packages, got {len(valid_pkgs)}"
    assert valid_pkgs.nunique() >= 15, "Target must have variance; cannot be a static constant"
    assert (valid_pkgs > 0).all(), "Packages must be strictly positive"
    if "tuition_fee_annual" in df_clean.columns:
        valid_fees = df_clean["tuition_fee_annual"].dropna()
        assert (valid_fees >= 0).all(), "Fees cannot be negative"
    print("\nAutomated Data Quality Validation: PASSED (Verified positive packages, realistic variance, no negative fees).")

    # 5. Save Outputs
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df_clean.to_parquet(output_parquet_path, index=False)
    df_clean.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"\nCleaned dataset saved successfully to:\n- {output_parquet_path}\n- {output_csv_path}")

    return df_clean


if __name__ == "__main__":
    run_preprocessing_pipeline()
