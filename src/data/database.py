"""
Database Management Module for CollegeWise.

Manages SQLite storage of college master data, placements, academics,
infrastructure, and provenance data.
"""

import os
import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "collegewise.db")


def init_database(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialize database tables and indexes."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Master Colleges Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS colleges (
        college_id TEXT PRIMARY KEY,
        college_name TEXT NOT NULL,
        university_name TEXT,
        institution_type TEXT,
        ownership TEXT,
        state TEXT,
        city TEXT,
        district TEXT,
        address TEXT,
        latitude REAL,
        longitude REAL,
        established_year INTEGER,
        accreditation TEXT,
        affiliated_university TEXT,
        degrees TEXT,
        branches TEXT,
        total_approved_intake INTEGER,
        faculty_count INTEGER,
        student_count INTEGER,
        student_faculty_ratio REAL,
        research_consultancy_lakhs REAL,
        academic_indicators TEXT,
        placement_rate REAL,
        median_package_lpa REAL,
        average_package_lpa REAL,
        highest_package_lpa REAL,
        higher_studies_rate REAL,
        recruiting_companies TEXT,
        internship_information TEXT,
        tuition_fee_annual REAL,
        hostel_fee_annual REAL,
        estimated_total_cost_annual REAL,
        scholarship_information TEXT,
        campus_area_acres REAL,
        hostel_available TEXT,
        library_facility TEXT,
        laboratories_facility TEXT,
        sports_facilities TEXT,
        medical_facilities TEXT,
        internet_connectivity TEXT,
        pcs_accessibility_score INTEGER,
        other_facilities TEXT,
        distance_to_major_city_km REAL,
        railway_connectivity TEXT,
        airport_connectivity TEXT,
        public_transport_information TEXT,
        clubs_and_societies TEXT,
        cultural_events TEXT,
        extracurricular_activities TEXT,
        data_source TEXT,
        data_source_url TEXT,
        data_year INTEGER,
        has_placement_data INTEGER,
        data_completeness_ratio REAL,
        missing_data_percentage REAL
    );
    """)

    # Create Indexes for fast filtering
    cur.execute("CREATE INDEX IF NOT EXISTS idx_colleges_state ON colleges(state);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_colleges_ownership ON colleges(ownership);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_colleges_inst_type ON colleges(institution_type);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_colleges_placement ON colleges(placement_rate);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_colleges_fee ON colleges(estimated_total_cost_annual);")

    conn.commit()
    return conn


def load_csv_to_sqlite(csv_path: str, db_path: str = DEFAULT_DB_PATH) -> int:
    """Load curated raw or processed CSV into the SQLite database."""
    conn = init_database(db_path)
    df = pd.read_csv(csv_path)

    # Clean boolean
    if "has_placement_data" in df.columns:
        df["has_placement_data"] = df["has_placement_data"].astype(int)

    # Insert into database
    df.to_sql("colleges", conn, if_exists="replace", index=False)
    conn.commit()
    count = len(df)
    conn.close()
    print(f"Successfully loaded {count} colleges into SQLite database at {db_path}")
    return count


def query_colleges(
    state: Optional[str] = None,
    ownership: Optional[str] = None,
    max_budget: Optional[float] = None,
    min_placement_rate: Optional[float] = None,
    branch_keyword: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Query colleges with optional filter criteria."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM colleges WHERE 1=1"
    params: List[Any] = []

    if state:
        query += " AND state = ?"
        params.append(state)
    if ownership:
        query += " AND ownership = ?"
        params.append(ownership)
    if max_budget is not None:
        query += " AND (estimated_total_cost_annual <= ? OR estimated_total_cost_annual IS NULL)"
        params.append(max_budget)
    if min_placement_rate is not None:
        query += " AND (placement_rate >= ? OR placement_rate IS NULL)"
        params.append(min_placement_rate)
    if branch_keyword:
        query += " AND branches LIKE ?"
        params.append(f"%{branch_keyword}%")

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


if __name__ == "__main__":
    raw_csv = r"d:\projects\college\data\raw\colleges_raw.csv"
    load_csv_to_sqlite(raw_csv)
