# System Methodology & Engineering Architecture — CollegeWise

CollegeWise rejects the traditional, naive paradigm of ranking colleges from "best" to "worst" via monolithic league tables. Institutional optimality is inherently multidimensional: a first-generation student with a strict budget seeking high return-on-investment has completely different optimal college choices than an affluent student prioritizing advanced research laboratories or vibrant campus life.

This document details the complete mathematical formulation, data science engineering, and decision-theoretic foundations of CollegeWise.

---

## 1. System Architecture & End-to-End Pipeline

```
                                  DATA INGESTION
                     ┌──────────────────────────────────────┐
                     │ AICTE Statutory Directory (13k+ rec) │
                     │ NIRF Disclosures (2019, 2020, 2021)  │
                     │ State Fee Regulatory Gazettes        │
                     └──────────────────┬───────────────────┘
                                        │
                                        ▼
                               DATA PREPROCESSING
                     ┌──────────────────────────────────────┐
                     │ • Regex Extraction & Normalization   │
                     │ • Deduplication (AICTE Permanent ID) │
                     │ • State & Ownership Standardization  │
                     │ • Boundary Sanity Assertions         │
                     └──────────────────┬───────────────────┘
                                        │
                                        ▼
                               FEATURE ENGINEERING
                     ┌──────────────────────────────────────┐
                     │ • 6 Decision Pillars (0 to 100)      │
                     │ • Anti-Leakage Feature Formulation   │
                     │ • Data Confidence Tier Assignment    │
                     └─────────┬──────────────────┬─────────┘
                               │                  │
                               ▼                  ▼
                    SUPERVISED ML MODEL        MCDA ENGINE
                 ┌──────────────────────┐  ┌──────────────────────┐
                 │ 9 Non-Leaking Feats  │  │ Hard Filter Matrix   │
                 │ 5-Fold CV + Test Set │  │ Simple Additive Wgt  │
                 │ Champion: GBDT Regr  │  │ Dynamic 100pt Budget │
                 │ R²=0.64, MAE=3.22LPA │  │ Sensitivity Analysis │
                 └──────────────────────┘  └──────────────────────┘
                               │                  │
                               └─────────┬────────┘
                                         ▼
                             STREAMLIT PRESENTATION
                 ┌──────────────────────────────────────────────┐
                 │ 🎯 Discover & Recommend Feed (100pt Budget)  │
                 │ 🔬 Interactive Sensitivity Analysis          │
                 │ 📊 Side-by-Side Multi-College Radar Chart    │
                 │ 🔍 Deep Dive Institutional Audit Provenance  │
                 └──────────────────────────────────────────────┘
```

---

## 2. Feature Engineering: The 6 Decision Pillars

All 6 pillars are normalized onto an intuitive, continuous scale of **0.0 to 100.0**.

### Pillar A: Placement & Career Score ($S_{\text{placement}}$)
Evaluates institutional employment outcomes using verified NIRF mandatory filings:
$$S_{\text{placement}} = 0.40 \cdot \text{PR}_{\text{norm}} + 0.50 \cdot \text{Sal}_{\text{norm}} + 0.10 \cdot \text{HS}_{\text{norm}}$$

- **Placement Rate Component ($\text{PR}_{\text{norm}}$)**:
  $$\text{PR}_{\text{norm}} = \text{clip}\left(\frac{\text{Students Placed}}{\text{Graduating Students}} \times 100, 0, 100\right)$$
- **Logarithmic Median Salary Utility ($\text{Sal}_{\text{norm}}$)**: Scaled via logarithmic returns to reflect diminishing marginal utility above ₹20 LPA:
  $$\text{Sal}_{\text{norm}} = \text{clip}\left(\frac{\ln(1 + \text{Median LPA}) - \ln(1 + 2.0)}{\ln(1 + 25.0) - \ln(1 + 2.0)} \times 100, 10, 100\right)$$
  *(Economic Rationale: Moving from ₹3.5 LPA to ₹7.0 LPA fundamentally alters a graduate's standard of living; moving from ₹22 LPA to ₹25 LPA provides marginal utility).*
- **Progression to Higher Studies ($\text{HS}_{\text{norm}}$)**: Progression to IISc, IITs, or top global universities:
  $$\text{HS}_{\text{norm}} = \text{clip}\left(\frac{\text{Higher Studies Rate}}{25.0} \times 100, 0, 100\right)$$
- **Missing Data Handling**: If an institution has not filed NIRF graduation outcomes, $S_{\text{placement}}$ is stored as `NULL`. It is never assumed to be zero.

---

### Pillar B: Academic Quality Score ($S_{\text{academics}}$)
Reflects statutory tier recognition, faculty density, and sponsored research:
$$S_{\text{academics}} = \text{clip}\left(B_{\text{tier}} + \Delta_{\text{SFR}} + \Delta_{\text{research}}, 40.0, 99.0\right)$$

- **Statutory Tier Baseline ($B_{\text{tier}}$)**:
  - Institute of National Importance (IIT / NIT / IIIT / IISc): $94.0$
  - Central University / NIRF Ranked (Top Tier): $88.0$
  - State Government University / Deemed (Govt): $80.0$
  - State Govt Engineering College / NBA Accredited: $76.0$
  - Standard AICTE Approved Autonomous / Private: $62.0$
- **Student-to-Faculty Ratio Adjustment ($\Delta_{\text{SFR}}$)**:
  - $\text{SFR} \le 15.0$ (Ideal AICTE norm): $+4.0$
  - $15.0 < \text{SFR} \le 20.0$: $+2.0$
  - $\text{SFR} > 35.0$: $-4.0$
- **Sponsored Research Grants ($\Delta_{\text{research}}$)**:
  - Sponsored research & consultancy $> ₹100\text{ Lakhs}$: $+4.0$
  - Sponsored research $> ₹20\text{ Lakhs}$: $+2.0$

---

### Pillar C: Affordability Score ($S_{\text{affordability}}$)
Models financial accessibility. Lower total annual expenditure yields higher utility:
$$S_{\text{affordability}} = 96.0 - \left(\frac{\text{Total Cost} - 25,000}{325,000} \times 66.0\right) + \Delta_{\text{govt\_aid}}$$

- Baseline statutory cost ranges from ₹25,000/year (State Govt College) to ₹3,50,000+/year (Private University).
- Government colleges receive a statutory $+4.0$ bonus reflecting fee waivers (Central Sector Scheme, Post-Matric, PMSSS).
- If fees are undisclosed, a sector-normative baseline is applied with an explicit transparency disclosure.

---

### Pillar D: Campus & Infrastructure Score ($S_{\text{campus}}$)
Based on verified Physical Facilities for Physically Challenged Students (PCS) and technical infrastructure:
$$S_{\text{campus}} = \text{clip}\left(B_{\text{infra}} + \Delta_{\text{PCS}} + \Delta_{\text{hostel}}, 50.0, 98.0\right)$$
- **PCS Accessibility Index ($\Delta_{\text{PCS}}$)**: Verified lifts, ramps, and accessible toilets (+4 points per facility, up to +12).
- **Residential Hostel Housing ($\Delta_{\text{hostel}}$)**: $+4.0$ for documented on-campus student housing.

---

### Pillar E: Location & Connectivity Score ($S_{\text{location}}$)
Evaluates urban accessibility and regional industry connectivity:
- Distance to major tier-1 economic hub ($\le 25\text{ km}$: $92$, $25-60\text{ km}$: $84$, $60-120\text{ km}$: $74$, $> 200\text{ km}$: $55$).
- Dynamic user state preference: If the institution matches the student's selected target state, a $+8.0$ alignment bonus is applied.

---

### Pillar F: Student Life & Extracurriculars Score ($S_{\text{student\_life}}$)
Measures documented student organizations, technical chapters (IEEE, ACM, SAE, CSI), cultural festivals, and sports complexes ($50.0$ to $95.0$).

---

## 3. Personalized Recommendation Engine (MCDA)

The recommendation engine executes in two discrete stages:

### Stage 1: Hard Constraint Filtering
Eliminates non-viable institutions based on absolute student criteria:
1. **Branch Availability**: Institution must offer the student's selected branch (e.g., Computer Science, Mechanical).
2. **Budget Ceiling**: $\text{Estimated Annual Cost} \le \text{Max Budget}$ (undisclosed colleges are preserved but flagged).
3. **Geographic Preference**: Filter by state if specified.
4. **Ownership Model**: Filter by Government or Private if specified.

### Stage 2: Soft Multi-Attribute Utility Ranking (Simple Additive Weighting)
User preference weights $w_j \in [0, 100]$ are normalized to sum to $1.0$:
$$\tilde{w}_j = \frac{w_j}{\sum_{k=1}^{6} w_k}$$

For each candidate college $i$, let $\mathcal{A}_i \subseteq \{1, \dots, 6\}$ denote the subset of dimensions for which verified data is available.

To prevent penalizing unranked colleges as "zero quality", weights are re-normalized over the active set $\mathcal{A}_i$:
$$\hat{w}_{j, i} = \frac{\tilde{w}_j}{\sum_{m \in \mathcal{A}_i} \tilde{w}_m}$$

The final personalized match score is:
$$\text{Match Score}_i = \sum_{j \in \mathcal{A}_i} \hat{w}_{j, i} \cdot s_{ij}$$

The transparency completeness index is simultaneously recorded:
$$\text{Completeness}_i = \frac{|\mathcal{A}_i|}{6} \times 100\%$$

---

## 4. Recommendation Sensitivity & Rank Perturbation Analysis

To prove that recommendations dynamically and reliably shift when a student changes their preferences, CollegeWise includes an **interactive sensitivity analysis engine** (`run_sensitivity_analysis`).

### Mathematical Formulation of Rank Shift
When a student perturbs their preference vector from $\mathbf{w}_A$ to $\mathbf{w}_B$ ($\Delta w_j = w_{B, j} - w_{A, j}$), the score delta for college $i$ is:

$$\Delta S_i = \sum_{j=1}^{6} \Delta w_j \cdot s_{ij}$$

- If college $i$ has high placement performance ($s_{i, \text{placement}} = 95$) but high tuition ($s_{i, \text{affordability}} = 30$), shifting weight from Placement to Affordability produces a strong negative $\Delta S_i$.
- Conversely, affordable government engineering colleges ($s_{i, \text{affordability}} = 92$) experience positive $\Delta S_i$ and vault upward in rank.

---

## 5. Supervised Machine Learning Pipeline

### Objective & Target Selection
- **Target Variable**: Institutional Median Package ($\text{LPA}$) from verified NIRF disclosures ($N = 109$).
- **Justification**: Median package represents the authentic 50th percentile graduate compensation, eliminating skew from rare international outlier packages.
- **Why Regression over Classification**: Continuous compensation varies smoothly from ₹0.90 LPA to ₹35.00 LPA (mean ₹8.71 LPA, std ₹6.16 LPA). Discretizing this into arbitrary classes loses distance information and misrepresents borderline colleges. Regression is maintained as the primary headline ML task.

### Anti-Data Leakage Protocol
To strictly avoid target leakage, the following variables are **strictly excluded** from input features:
- `placement_rate`
- `average_package_lpa`
- `highest_package_lpa`
- `placement_score`
- `affordability_score` (excluded to eliminate collinearity with `tuition_fee_annual`)

### Input Feature Matrix ($X \in \mathbb{R}^9$)
1. `total_approved_intake` (Institutional scale)
2. `tuition_fee_annual` (Financial resource indicator)
3. `faculty_count` (Sanctioned instructional staff)
4. `student_faculty_ratio` (Instructional density)
5. `academic_score` (Statutory tier & accreditation)
6. `infrastructure_score` (Campus facilities)
7. `location_score` (Connectivity)
8. `student_life_score` (Campus environment)
9. `is_government` (Ownership binary indicator)

### Benchmark Results (Holdout Test Set)

| Model Architecture | 5x5 Repeated CV MAE | 5x5 Repeated CV $R^2$ | Val MAE (LPA) | Val RMSE (LPA) | Val $R^2$ | Test MAE (LPA) | Test RMSE (LPA) | Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Mean)** | 4.43 $\pm$ 0.68 | -0.12 $\pm$ 0.23 | 5.89 | 7.28 | -0.0120 | 5.83 | 6.55 | -0.0211 | Naive Baseline |
| **Linear Regression** | 3.60 $\pm$ 0.80 | 0.13 $\pm$ 0.36 | 4.37 | 5.57 | 0.4075 | 3.98 | 4.72 | 0.4704 | Linear Baseline |
| **Ridge Regression ($\alpha=10$)** | 3.40 $\pm$ 0.68 | 0.25 $\pm$ 0.26 | 4.42 | 5.61 | 0.3993 | 4.22 | 4.99 | 0.4066 | Regularized Linear |
| **Decision Tree ($d=5$)** | 3.90 $\pm$ 0.79 | -0.31 $\pm$ 1.07 | 4.09 | 5.96 | 0.3229 | 2.92 | 4.04 | 0.6122 | Non-Linear Baseline |
| **Random Forest ($n=150$)** | 2.95 $\pm$ 0.75 | 0.44 $\pm$ 0.26 | 4.00 | 5.86 | 0.3446 | 3.24 | 3.94 | 0.6300 | Ensemble Bagging |
| **Gradient Boosting** | **2.95 $\pm$ 0.65** | **0.37 $\pm$ 0.37** | **3.83** | **5.39** | **0.4456** | **3.22** | **3.88** | **0.6418** | **Champion Model** |
| **XGBoost Regressor** | 2.94 $\pm$ 0.60 | 0.39 $\pm$ 0.33 | 4.48 | 5.84 | 0.3498 | 3.10 | 3.95 | 0.6293 | Regularized Boosting |

---

## 6. Explainable AI (XAI) Framework

For every recommended college, CollegeWise generates an evidence-backed rationale answering **"Why this college?"**:
1. **Priority Alignment**: Connects the student's highest weighted sliders ($\ge 15\%$) to documented institutional strengths.
2. **Budget Verification**: Validates whether the statutory tuition fee conforms to the user's budget ceiling.
3. **Location Confirmation**: Confirms geographic proximity to the student's target state/city.
4. **Data Caveats**: Explicitly notes if the institution does not participate in NIRF disclosures and highlights that the score relies on verified regulatory indicators.
