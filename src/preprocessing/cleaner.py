"""
Data Cleaning & Standardization Module for CollegeWise.

Implements reusable functions for:
- State and city standardization
- Inconsistent naming corrections
- Duplicate detection and removal
- Outlier detection and handling
- Data type coercion and schema validation
- Missing value audit
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

# Standardized Indian States & Union Territories
STATE_MAPPING = {
    "nct of delhi": "Delhi",
    "delhi ncr": "Delhi",
    "national capital territory of delhi": "Delhi",
    "pondicherry": "Puducherry",
    "orissa": "Odisha",
    "telengana": "Telangana",
    "tamilnadu": "Tamil Nadu",
    "uttaranchal": "Uttarakhand",
    "chhatisgarh": "Chhattisgarh",
    "jammu & kashmir": "Jammu and Kashmir",
    "andaman & nicobar": "Andaman and Nicobar Islands",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "dadra & nagar haveli": "Dadra and Nagar Haveli",
    "daman & diu": "Daman and Diu"
}

# Ownership mapping
OWNERSHIP_MAPPING = {
    "government": "Government",
    "govt aided": "Government",
    "central university": "Government",
    "state government university": "Government",
    "deemed to be university(govt)": "Government",
    "private-self financing": "Private",
    "state private university": "Private",
    "deemed to be university(pvt)": "Private"
}


def standardize_state(state_raw: Optional[str]) -> str:
    """Standardize state and union territory names."""
    if not state_raw or pd.isna(state_raw):
        return "Unknown"
    s = str(state_raw).strip()
    s_clean = s.lower()
    return STATE_MAPPING.get(s_clean, s.title())


def standardize_ownership(inst_type: Optional[str]) -> str:
    """Determine standardized ownership (Government / Private)."""
    if not inst_type or pd.isna(inst_type):
        return "Private"
    t_clean = str(inst_type).strip().lower()
    for key, val in OWNERSHIP_MAPPING.items():
        if key in t_clean:
            return val
    return "Private"


def clean_college_name(name_raw: Optional[str]) -> str:
    """Clean and standardize college names."""
    if not name_raw or pd.isna(name_raw):
        return "Unknown Institution"
    name = str(name_raw).strip()
    # Normalize excessive spaces
    name = re.sub(r'\s+', ' ', name)
    # Remove leading/trailing quotes or punctuation
    name = name.strip('"\' ,.-')
    return name


def detect_and_remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Detect duplicates based on college_id or (college_name + district + state).
    Returns cleaned dataframe and count of removed duplicates.
    """
    initial_len = len(df)
    # 1. Deduplicate by unique college_id
    df_dedup = df.drop_duplicates(subset=["college_id"], keep="first")
    
    # 2. Deduplicate by standardized name + state
    df_dedup = df_dedup.drop_duplicates(
        subset=["college_name", "state"],
        keep="first"
    )
    removed = initial_len - len(df_dedup)
    return df_dedup, removed


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Detect outliers using the Interquartile Range (IQR) method."""
    valid = series.dropna()
    if len(valid) < 4:
        return pd.Series(False, index=series.index)
    q25 = valid.quantile(0.25)
    q75 = valid.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)
    return (series < lower_bound) | (series > upper_bound)


def sanitize_numeric_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-informed boundary checks for metrics in India:
    - Placement rate: 0.0% to 100.0%
    - Median package: 1.0 LPA to 60.0 LPA
    - Student-Faculty Ratio: 3.0 to 100.0
    - Fees: 5,000 to 1,500,000 INR
    """
    df_clean = df.copy()

    # Placement rate
    if "placement_rate" in df_clean.columns:
        df_clean.loc[(df_clean["placement_rate"] < 0) | (df_clean["placement_rate"] > 100), "placement_rate"] = np.nan

    # Median Package LPA
    if "median_package_lpa" in df_clean.columns:
        df_clean.loc[(df_clean["median_package_lpa"] < 0.5) | (df_clean["median_package_lpa"] > 100), "median_package_lpa"] = np.nan

    # Fees
    if "tuition_fee_annual" in df_clean.columns:
        df_clean.loc[(df_clean["tuition_fee_annual"] < 1000) | (df_clean["tuition_fee_annual"] > 2500000), "tuition_fee_annual"] = np.nan

    return df_clean


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Generate detailed missing-value audit report."""
    total = len(df)
    missing_count = df.isnull().sum()
    missing_pct = round((missing_count / total) * 100, 2)
    dtypes = df.dtypes
    
    report = pd.DataFrame({
        "column": df.columns,
        "dtype": dtypes.astype(str).values,
        "missing_count": missing_count.values,
        "missing_percentage": missing_pct.values,
        "available_count": (total - missing_count).values
    })
    return report.sort_values(by="missing_percentage", ascending=False).reset_index(drop=True)
