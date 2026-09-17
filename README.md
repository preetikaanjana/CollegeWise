# CollegeWise — Personalized ML-Based College Decision Support System

[![Python 3.12](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-20%20passed-brightgreen.svg)]()

> **CollegeWise** is an end-to-end ML-based college decision-support platform designed specifically for students in India.
> Instead of arbitrarily sorting institutions from "best" to "worst" via static monolithic league tables, CollegeWise evaluates institutional attributes against an individual student's academic profile, budget, location preferences, and dynamically weighted priorities across 6 decision dimensions.

---

## 📑 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Why CollegeWise](#3-why-collegewise)
4. [System Architecture](#4-system-architecture)
5. [Dataset & Data Sources](#5-dataset--data-sources)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Feature Engineering](#7-feature-engineering)
8. [ML Problem](#8-ml-problem)
9. [Models Compared](#9-models-compared)
10. [Model Evaluation](#10-model-evaluation)
11. [Champion Model](#11-champion-model)
12. [Feature Importance](#12-feature-importance)
13. [Recommendation Engine](#13-recommendation-engine)
14. [MCDA / SAW](#14-mcda--saw)
15. [Sensitivity Analysis](#15-sensitivity-analysis)
16. [Missing Data Handling](#16-missing-data-handling)
17. [Testing](#17-testing)
18. [Tech Stack](#18-tech-stack)
19. [Limitations](#19-limitations)
20. [How to Run](#20-how-to-run)
21. [Interview Explanation](#21-interview-explanation)

---

## 1. Project Overview

CollegeWise is an end-to-end, machine-learning-assisted college decision-support system designed for Indian higher education. Rather than forcing students into a one-size-fits-all ranking, the system combines supervised continuous regression for institutional compensation estimation with multi-criteria decision analysis (MCDA) to produce personalized, transparent college recommendations.

The platform is built on verified, publicly disclosed statutory data from Indian regulatory bodies, enforcing strict zero-fabrication data hygiene and anti-leakage machine learning pipelines.

---

## 2. Problem Statement

Every year, over 2.5 million Indian students qualify through entrance examinations such as JEE Main, JEE Advanced, and State CETs. The prevailing college selection process exhibits critical structural flaws:
- **Monolithic League Tables**: Commercial portals publish single linear "Top 100" lists, treating institutional quality as an absolute, one-dimensional metric regardless of user context.
- **Outlier Salary Distortions**: Portals highlight extreme international offers (e.g. ₹1.5 Crore) that artificially inflate average package statistics, concealing true median outcomes.
- **Ignored Trade-offs**: A student prioritizing affordability and regional proximity requires a completely different optimal choice than an affluent student targeting startup incubation.
- **Lack of Decision Agency**: Students cannot transparently evaluate how adjusting their personal values impacts recommendations.

---

## 3. Why CollegeWise

CollegeWise addresses these challenges through four foundational principles:
- **Personalized Decision Support**: College fit is multi-dimensional. The best college depends on an individual student's trade-offs across academics, fees, placement, campus, location, and student life.
- **Strict Budget Allocation**: Students allocate a finite 100-point budget across 6 dimensions. A dynamic slider constraint prevents users from unrealistically setting all priorities to 100%.
- **Zero-Fabrication Data Hygiene**: All records originate from audited government databases (AICTE, NIRF, State Fee Regulatory Committees). Missing data is tagged with prediction uncertainty bounds rather than filled with synthetic placeholders.
- **Explainability & Sensitivity**: Recommendations include transparent factor breakdowns ("Why this college?") and interactive sensitivity analysis to inspect rank movements under shifting priorities.

---

## 4. System Architecture

The CollegeWise architecture cleanly decouples machine learning inference from multi-criteria decision ranking across three distinct layers:

```
+-------------------------------------------------------------------------+
|                              Streamlit UI                               |
|   (Interactive Sliders, Filtering, Sensitivity Slope Chart, Radar)      |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  Layer 3: Recommendation Engine (MCDA)                  |
|   - Hard Filtering (Tuition Budget, State/Region, Ownership Type)       |
|   - Simple Additive Weighting: S_i = sum(w_j * s_ij)                    |
|   - Explanation & Confidence Attribution                                |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    Layer 2: College Scoring Layer                       |
|   - Computes normalized [0, 100] scores across 6 Decision Dimensions    |
|   - Combines statutory disclosures with ML-estimated metrics            |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                      Layer 1: Machine Learning Layer                    |
|   - Supervised Continuous Regression (Gradient Boosting Champion)       |
|   - Target: Institutional Median Compensation (LPA)                     |
|   - Anti-leakage Pipeline & Data Coverage Uncertainty Bounds            |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                       Data Storage & Repositories                       |
|   - Cleaned Parquet Data (1,634 institutions)                           |
|   - Serialized Model Artifacts (Joblib) & Local SQLite                  |
+-------------------------------------------------------------------------+
```

---

## 5. Dataset & Data Sources

CollegeWise relies exclusively on authoritative statutory disclosures from Indian regulatory bodies:
1. **AICTE Official Institutional Disclosures**: Directory covering approved engineering institutions across 35 states and union territories, providing approved programmes, degree levels, annual intake, and statutory categories.
2. **NIRF (National Institutional Ranking Framework, Ministry of Education)**: Official multi-year disclosures (2019, 2020, 2021) covering student enrollment, faculty counts, physical facilities, graduating batch sizes, campus placement counts, median compensation, and higher studies progression.
3. **State Fee Regulatory Committees (AFRC / FRC) & JoSAA**: Statutory fee schedules for central (IITs, NITs, IIITs) and state-approved institutions.

### Population Breakdown
* **1,634 institutions**: Total deduplicated population available for recommendation in the system database.
* **109 institutions**: Supervised ML cohort with audited, verified multi-year NIRF placement disclosures (median packages from ₹0.90 LPA to ₹35.00 LPA; mean ₹8.71 LPA, standard deviation ₹6.16 LPA). The verified supervised-learning cohort is smaller because only institutions with sufficiently complete and verified placement disclosures were eligible for model training.
* **1,525 institutions**: Medium-confidence statutory cohort with verified regulatory data (intake, location, ownership, fees) whose placement package is estimated via the trained ML model with coverage uncertainty bounds.

---

## 6. Data Preprocessing

The preprocessing pipeline (`src/preprocessing/pipeline.py`) enforces rigorous cleaning steps:
- **Canonical Deduplication**: Deduplicated on unique AICTE Permanent IDs and standardized institutional names, resolving duplicate regional records into 1,634 clean institutions.
- **Text & Geographic Normalization**: Standardized state and union territory designations (e.g. mapping `"nct of delhi"` to `"Delhi"`), normalized whitespace, and stripped formatting artifacts.
- **Numeric Regex Extraction**: Extracted numeric values from raw strings in historical tables (e.g. extracting `400000` from `'400000(four lakhs)'`) using regex (`re.search(r'(\d+)', ...)`) and scaled to standard Lakhs Per Annum (LPA).
- **Domain Guardrails & Boundary Validation**: Asserted realistic domain boundaries:
  - Placement rates clamped to [0, 100]%
  - Median compensation validated in [0.5, 60.0] LPA
  - Annual fees bounded to [₹1,000, ₹25,00,000]

---

## 7. Feature Engineering

### The 6 Decision Pillars
CollegeWise structures institutional evaluation into 6 standardized dimension scores normalized to [0, 100]:
1. **Placement & Career**: Historical placement rate, median compensation, higher studies progression.
2. **Campus & Infrastructure**: Physical facilities (PCS metric), laboratory/library expenditures, capital facilities.
3. **Academic Quality**: Student-to-faculty ratio, program accreditations, advanced degree offerings.
4. **Affordability**: Inverse-scaled annual tuition fees relative to state benchmarks.
5. **Location & Connectivity**: Metro classification tier, industrial hub proximity, regional connectivity index.
6. **Student Life**: Student body diversity, extracurricular scale, campus amenities.

### Anti-Leakage & Redundancy Design
To avoid target leakage in the supervised ML layer, the feature matrix used for package prediction is strictly decoupled:
- **Predictors (`X`)**: `total_approved_intake`, `student_faculty_ratio`, `is_government`, `tuition_fee_annual`, `academics_score`, `infrastructure_score`, `faculty_quality_score`, `student_life_score`, `location_score`.
- **Excluded Features**: `placement_rate`, average salary, highest package, and engineered placement pillar scores are strictly excluded from predictors.
- **Feature Redundancy Check**: Derived scores such as `affordability_score` were excluded from predictors because they duplicate `tuition_fee_annual`. Preprocessing transformations are fit solely on training folds.

---

## 8. ML Problem

- **Formulation**: Supervised Continuous Regression.
- **Target Variable**: `median_package_lpa` (continuous value in Lakhs Per Annum).
- **Rationale for Continuous Regression**: Institutional median compensation varies smoothly from ₹0.90 LPA to ₹35.00 LPA. Discretizing compensation into coarse categories (e.g. "High" vs. "Low") discards metric ordering and distance, introducing artificial discontinuities near category thresholds. Continuous regression predicts the expected package directly.

---

## 9. Models Compared

We evaluated 7 regression architectures representing diverse modeling paradigms:
1. **DummyRegressor (strategy="mean")**: Predicts training mean; serves as the naive non-learning baseline.
2. **Linear Regression (OLS)**: Classical parametric baseline.
3. **Ridge Regression (alpha=10.0)**: L2-regularized linear regression to penalize large weights.
4. **DecisionTreeRegressor (depth=5)**: Non-linear single-tree baseline.
5. **RandomForestRegressor (n=150)**: Bagging ensemble that reduces variance by averaging de-correlated trees.
6. **GradientBoostingRegressor (Champion)**: Sequential boosting ensemble that minimizes squared residual loss.
7. **XGBRegressor**: Regularized gradient boosted decision trees with second-order gradient approximations.

---

## 10. Model Evaluation

### Data Splitting & Cross-Validation
- **Dataset Partitioning**: 75 Train (68.8%), 17 Validation (15.6%), 17 Holdout Test (15.6%).
- **Holdout Test Integrity**: The holdout set was set aside at the beginning and used strictly for final evaluation.
- **Cross-Validation**: 5-Fold x 5-Repeat Repeated Cross-Validation (25 folds) evaluated on training data to assess variance and generalization stability.

### Benchmark Results (Holdout Test Set, N=17)

| Model | Test MAE (LPA) | Test RMSE (LPA) | Test R² | 5x5 CV MAE (LPA) | 5x5 CV R² |
|---|---|---|---|---|---|
| **Gradient Boosting (Champion)** | **3.22** | **3.88** | **0.6418** | **2.95 ± 0.65** | **0.37 ± 0.37** |
| Random Forest (n=150) | 3.24 | 3.94 | 0.6300 | 2.95 ± 0.75 | 0.44 ± 0.26 |
| XGBoost Regressor | 3.65 | 4.48 | 0.5211 | 3.22 ± 0.78 | 0.26 ± 0.38 |
| Linear Regression (OLS) | 3.98 | 4.71 | 0.4704 | 3.65 ± 0.92 | 0.12 ± 0.42 |
| Ridge Regression (alpha=10.0) | 4.22 | 4.99 | 0.4066 | 3.61 ± 0.89 | 0.18 ± 0.39 |
| Decision Tree (depth=5) | 4.09 | 5.14 | 0.3693 | 3.68 ± 1.10 | 0.08 ± 0.48 |
| Dummy Regressor (Baseline) | 5.83 | 6.55 | -0.0211 | 4.98 ± 1.15 | -0.42 ± 0.45 |

---

## 11. Champion Model

**GradientBoostingRegressor** is the selected champion model:
- **Holdout Test R²**: The Gradient Boosting model achieved an R² of 0.6418 on the 17-institution holdout test set.
- **Holdout Test MAE**: **3.22 LPA** (3.2178), a **44.8% error reduction** compared to the Dummy Baseline (5.83 LPA).
- **Holdout Test RMSE**: **3.88 LPA** (3.8780).
- **5x5 Repeated CV MAE**: **2.95 ± 0.65 LPA**.

### Secondary Diagnostic Classification Experiment
A small-sample secondary diagnostic classification experiment on 17 holdout institutions was conducted to evaluate decision boundary behavior (Median Package >= ₹7.0 LPA, N=17, 9 positive, 8 negative):
- Accuracy: **1.00**, Precision: **1.00**, Recall: **1.00**, F1-Score: **1.00**, ROC-AUC: **1.00**.
- **Scientific Caveat**: This is a small-sample secondary diagnostic classification experiment on 17 holdout institutions. Because institutions clustered distinctly on either side of the threshold in this split, the diagnostic achieved perfect separation. This result must not be presented as the main project metric or interpreted as evidence of 100% overall system accuracy. The primary ML headline remains continuous regression: Gradient Boosting — R² 0.6418, MAE 3.22 LPA, RMSE 3.88 LPA.

---

## 12. Feature Importance

Feature importance was evaluated using **Permutation Feature Importance** on the holdout test set (10 repeats, scoring = negative MAE):

| Feature | Mean Importance (MAE Drop) | Standard Deviation |
|---|---|---|
| `student_faculty_ratio` | **0.581** | 0.142 |
| `location_score` | **0.537** | 0.118 |
| `student_life_score` | **0.241** | 0.075 |
| `total_approved_intake` | **0.211** | 0.063 |
| `faculty_quality_score` | **0.183** | 0.052 |
| `infrastructure_score` | **0.150** | 0.048 |
| `tuition_fee_annual` | **0.047** | 0.019 |
| `academics_score` | **0.005** | 0.008 |
| `is_government` | **0.0001** | 0.001 |

> **Causal Disclaimer**: Permutation importance measures how model performance degrades when a feature's values are randomly shuffled. It indicates predictive importance within this model and dataset, **not causal effect**. For example, a high location score is correlated with higher compensation, but altering a campus location does not causally guarantee higher wages.

---

## 13. Recommendation Engine

The recommendation engine is strictly separated from the regression model. While the ML model predicts one attribute (placement package), the recommendation engine aggregates all 6 decision dimensions according to individual student preferences:
1. **Hard Filtering**: Discards colleges violating student-defined budget limits, state/geographic filters, or ownership type (Government vs. Private).
2. **Dimension Scaling**: Normalizes each eligible institution's factors to a [0, 100] scale.
3. **Preference Weighting**: Applies user weights via Multi-Criteria Decision Analysis.
4. **Explainable Attribution**: Outputs personalized match scores alongside dimension breakdowns and confidence indicators.

---

## 14. MCDA / SAW

CollegeWise implements **Multi-Criteria Decision Analysis (MCDA)** via **Simple Additive Weighting (SAW)**:

$$\text{Match Score}_i = \sum_{j=1}^{6} w_j \cdot s_{ij}$$

Where:
- $w_j$ is the student's normalized priority weight for dimension $j$, subject to $\sum_{j=1}^6 w_j = 100$.
- $s_{ij} \in [0, 100]$ is institution $i$'s normalized score on dimension $j$.

### Mathematical Properties & Dynamic Budget Slider
- **Linearity & Transparency**: The change in match score is directly $\Delta S_i = \sum_{j=1}^6 \Delta w_j \cdot s_{ij}$. Students can easily verify why an institution moved up or down.
- **Dynamic 100-Point Budget Slider**: Students have a finite budget of 100 points. Increasing one slider dynamically adjusts the remaining sliders, preventing users from naively setting all dimensions to 100.

---

## 15. Sensitivity Analysis

CollegeWise includes an interactive sensitivity analysis tool:
- Students can define and compare two preference profiles: **Scenario A** (e.g. Placement-heavy) vs. **Scenario B** (e.g. Affordability & Location-heavy).
- An interactive **Plotly slope chart** visualizes institutional rank movements between the two scenarios.
- Demonstrates how recommendations adapt dynamically to student values rather than enforcing an arbitrary fixed ranking.

---

## 16. Missing Data Handling

CollegeWise follows a strict **zero-fabrication policy**:
- Missing statutory data is never filled with synthetic placeholders, imputed arbitrary averages, or assumed to be zero.
- When an institution lacks verified data for a specific pillar, the scoring engine re-normalizes available dimensions.
- Institutions display transparent **Data Confidence Labels**:
  - **High Confidence**: Verified NIRF multi-year placement and salary disclosure records.
  - **Medium Confidence**: Verified AICTE statutory data; placement package estimated with coverage uncertainty bound (±2.0 LPA).
  - **Baseline**: Baseline statutory data; placement package estimated with coverage uncertainty bound (±3.0 LPA).

> **Uncertainty Bound Method**: These are estimated uncertainty bounds based on data-coverage tiers and are not statistically calibrated confidence intervals.

---

## 17. Testing

The codebase includes an automated test suite executed with `pytest`:
- **Test Suite Location**: `tests/test_models.py`, `tests/test_preprocessing.py`
- **Total Passing Tests**: **20 passed**
- **Test Coverage**:
  - Preprocessing pipeline validation and boundary checks
  - Zero target leakage assertions (correlations < 0.90)
  - Regression pipeline training, inference, and uncertainty bounds
  - MCDA mathematical weight conservation (sum(w_j) = 100)
  - Hard constraint filtering logic

---

## 18. Tech Stack

The technologies used across CollegeWise are strictly categorized as follows:

### Core Programming
* Python 3.12+

### Application & UI
* Streamlit
* Custom CSS
* Streamlit Session State

### Data Science & Machine Learning
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Joblib

### Data Visualization
* Plotly
* Matplotlib
* Seaborn

### Data Storage
* SQLite
* Apache Parquet
* PyArrow

### Recommendation & Decision Science
* MCDA (Multi-Criteria Decision Analysis)
* Simple Additive Weighting (SAW)
* Sensitivity Analysis

### Testing
* Pytest

### Supporting Python Libraries
* `re` for text and numeric extraction

---

## 19. Limitations

To maintain scientific integrity and interview defensibility, the following limitations are explicitly recognized:
1. **Sample Size Disparity**: The verified supervised-learning cohort is smaller because only institutions with sufficiently complete and verified placement disclosures were eligible for model training.
2. **Prediction Uncertainty**: Institutional compensation predictions carry estimated uncertainty bounds (±2.0 to ±3.0 LPA). These are estimated uncertainty bounds based on data-coverage tiers and are not statistically calibrated confidence intervals.
3. **Public Data Inconsistencies**: Government disclosures across different reporting years can exhibit reporting variations and incomplete optional fields.
4. **Predictive Associations, Not Causality**: Feature importance reflects predictive correlation within the trained model, not causal levers for institutional improvement.
5. **Preference Sensitivity**: Recommendations reflect user-selected priority weights and do not replace professional educational counseling.
6. **Institutional Estimates vs. Individual Outcomes**: A predicted placement package is an institutional median estimate, not a guaranteed offer for any individual student.
7. **Classification Diagnostic Scope**: The secondary classification experiment on 17 holdout institutions achieved perfect separation due to boundary clustering and cannot be generalized as overall system accuracy.

---

## 20. How to Run

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/username/college.git
cd college

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Tests
```bash
pytest tests/ -v
```

### Launch Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 21. Interview Explanation

When presenting CollegeWise in an ML / software engineering interview, explain the system in three key points:
1. **The Core Problem & Architecture**: "College selection is fundamentally a multi-criteria decision problem, not a single monolithic ranking. I built CollegeWise as an end-to-end decision-support platform that decouples continuous ML regression for institutional package estimation from Multi-Criteria Decision Analysis (SAW) for personalized college ranking."
2. **Data Hygiene & ML Pipeline**: "I enforced strict anti-leakage controls by excluding placement-derived features from predictors. On the 17-institution holdout test set of verified NIRF disclosures, our Gradient Boosting model achieved an R² of 0.6418 and MAE of 3.22 LPA—a 44.8% error reduction over the Dummy Baseline. We report estimated prediction uncertainty bounds based on data-coverage tiers rather than claiming false precision."
3. **Engineering Integrity**: "We avoided synthetic data fabrication, evaluated 7 model architectures with 5x5 repeated cross-validation, implemented a dynamic 100-point budget slider, and backed the system with a 20-test automated test suite."
