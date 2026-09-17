# Model Card: CollegeWise Institutional Median Package Predictor

## 1. Model Details

- **Model Name**: CollegeWise Median Placement Compensation Predictor
- **Model Version**: 2.0.0 (Post-Audit Release)
- **Model Architecture**: Gradient Boosting Regressor (`GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)`)
- **Primary Task**: Continuous Regression estimating institutional median graduate starting compensation in Lakhs Per Annum (LPA).
- **Secondary Evaluation**: Binary Classification Audit ($\text{Median Package} \ge 7.0\text{ LPA}$, Tier 1 Premium Threshold).
- **Date**: September 2026
- **License**: MIT
- **Contact / Maintainer**: CollegeWise Data Science & Decision Engineering Team

---

## 2. Intended Use & Scope

### In-Scope Use Cases
1. **Decision Support for Unreported Institutions**: Generating statistical benchmark estimates for accredited technical institutions that lack verified public NIRF graduation disclosures.
2. **Transparent Uncertainty Bounds**: Providing upper and lower uncertainty bounds ($\pm 2.0$ to $\pm 3.0\text{ LPA}$ based on data coverage) to avoid false precision.
3. **Multi-Criteria Contextualization**: Serving as an input feature for informational decision-support alongside 5 other non-monetary preference pillars.

### Out-of-Scope & Misuse Cases
- **Individual Salary Guarantees**: This model predicts *institutional median batch outcomes*, NOT an individual student's future compensation.
- **Accreditation / Regulatory Judgments**: Predictions cannot replace statutory NIRF or AICTE audits.
- **Commercial Ranking Endorsements**: The model is not intended to rank institutions monolinearly from "best" to "worst".

---

## 3. Factors, Features & Anti-Leakage Protocol

The feature schema was engineered to eliminate **data leakage** and **collinear redundancy**.

### Input Features ($X \in \mathbb{R}^9$)

| Feature Name | Type | Description | Source / Rationale |
| :--- | :--- | :--- | :--- |
| `total_approved_intake` | Integer | Total annual student intake capacity | AICTE statutory approval; proxies institutional scale |
| `tuition_fee_annual` | Float | Statutory or reported annual tuition (INR) | State Fee Regulatory Committee (FRC) / AICTE |
| `faculty_count` | Integer | Total full-time sanctioned teaching faculty | AICTE / NIRF verified disclosures |
| `student_faculty_ratio` | Float | Enrolled students per faculty member | Proxies instructional attention & faculty density |
| `academic_score` | Float | Engineered index (40.0 - 99.0) | Statutory tier (INI/Central/State), NAAC/NBA status |
| `infrastructure_score`| Float | Engineered index (50.0 - 98.0) | Documented PCS accessibility, labs, residential hostels |
| `location_score` | Float | Engineered index (40.0 - 100.0)| Proximity to tier-1 metro hubs and state alignment |
| `student_life_score` | Float | Engineered index (50.0 - 95.0) | Technical clubs, sports complexes, student societies |
| `is_government` | Binary | 1.0 if Government / Central / State; 0.0 if Private | Statutory governance and subsidy structure |

### Explicitly Excluded Features (Anti-Leakage)
- `placement_rate` (% placed): Excluded to prevent target leakage (strongly correlated with median salary).
- `average_package_lpa`, `highest_package_lpa`: Excluded as direct target derivatives.
- `placement_score`: Excluded because it incorporates verified median package in its formulation.
- `affordability_score`: Excluded to prevent collinearity with `tuition_fee_annual` ($r \approx -0.98$).
- `latitude`, `longitude`: Excluded to prevent geographical coordinate overfitting.

---

## 4. Training Data & Validation Protocol

- **Dataset**: 109 verified institutions with official, publicly verifiable NIRF graduation outcome disclosures.
- **Target Distribution**:
  - Sample Size: $N = 109$
  - Range: ₹0.90 LPA to ₹35.00 LPA
  - Mean: ₹8.71 LPA
  - Standard Deviation: ₹6.16 LPA
  - Median: ₹6.50 LPA
- **Data Splitting**:
  - **Training Split**: 75 samples (68.8%)
  - **Validation Split**: 17 samples (15.6%)
  - **Holdout Test Split**: 17 samples (15.6%)
  - Split Seed: `random_state=42`
- **Validation Protocol**:
  - `RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)` yielding 25 out-of-fold evaluations per model candidate.
  - All preprocessing transformers (`SimpleImputer(strategy='median')` and `StandardScaler()`) are fit strictly on training folds to eliminate validation and holdout leakage.

---

## 5. Quantitative Evaluation & Model Benchmarking

All candidate architectures were trained and evaluated on the exact same splits against a naive baseline (`DummyRegressor(strategy="mean")`).

### Comprehensive Regression Benchmark

| Model Architecture | 5x5 Repeated CV MAE | 5x5 Repeated CV $R^2$ | Val MAE (LPA) | Val RMSE (LPA) | Val $R^2$ | Test MAE (LPA) | Test RMSE (LPA) | Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Mean)** | 4.43 $\pm$ 0.68 | -0.12 $\pm$ 0.23 | 5.89 | 7.28 | -0.0120 | 5.83 | 6.55 | -0.0211 | Naive Baseline |
| **Linear Regression** | 3.60 $\pm$ 0.80 | 0.13 $\pm$ 0.36 | 4.37 | 5.57 | 0.4075 | 3.98 | 4.72 | 0.4704 | Linear Baseline |
| **Ridge Regression ($\alpha=10$)** | 3.40 $\pm$ 0.68 | 0.25 $\pm$ 0.26 | 4.42 | 5.61 | 0.3993 | 4.22 | 4.99 | 0.4066 | Regularized Linear |
| **Decision Tree ($d=5$)** | 3.90 $\pm$ 0.79 | -0.31 $\pm$ 1.07 | 4.09 | 5.96 | 0.3229 | 2.92 | 4.04 | 0.6122 | Non-Linear Baseline |
| **Random Forest ($n=150$)** | 2.95 $\pm$ 0.75 | 0.44 $\pm$ 0.26 | 4.00 | 5.86 | 0.3446 | 3.24 | 3.94 | 0.6300 | Ensemble Bagging |
| **Gradient Boosting** | **2.95 $\pm$ 0.65** | **0.37 $\pm$ 0.37** | **3.83** | **5.39** | **0.4456** | **3.22** | **3.88** | **0.6418** | **Champion Model** |
| **XGBoost Regressor** | 2.94 $\pm$ 0.60 | 0.39 $\pm$ 0.33 | 4.48 | 5.84 | 0.3498 | 3.10 | 3.95 | 0.6293 | Regularized Boosting |

### Key Benchmark Takeaways
1. **Dummy Baseline Comparison**: The Dummy Baseline has a Test MAE of **5.83 LPA** and a negative $R^2$ of **-0.0211**. The Champion Gradient Boosting model reduces MAE to **3.22 LPA** (a **44.8% reduction in error**) and achieves a holdout $R^2$ of **0.6418**, proving genuine signal capture.
2. **Linear Underperformance**: Ordinary Least Squares underperforms non-linear models ($R^2 = 0.47$, $\text{MAE} = 3.98\text{ LPA}$) due to multi-dimensional collinearities and extreme non-linear threshold effects in technical education.
3. **Tree Ensemble Superiority**: Both Random Forest and Gradient Boosting consistently demonstrate strong generalization across repeated cross-validation and holdout test sets ($R^2 \ge 0.63$, $\text{Test MAE} \approx 3.2\text{ LPA}$).

---

## 6. Secondary Diagnostic Classification Experiment

A small-sample secondary diagnostic classification experiment on 17 holdout institutions was conducted to evaluate decision boundary behavior:
- **Task**: Classify institutions as **Tier 1 Premium** ($\text{Median Package} \ge ₹7.0\text{ LPA}$) vs. Standard ($< ₹7.0\text{ LPA}$).
- **Holdout Test Results** ($N_{\text{test}} = 17$, with 9 positive and 8 negative instances):
  - Accuracy: $1.00$ ($100\%$)
  - Precision: $1.00$
  - Recall: $1.00$
  - F1-Score: $1.00$
  - ROC-AUC: $1.00$
  - Confusion Matrix: `[[8, 0], [0, 9]]`

> [!IMPORTANT]
> **Scientific Caveat on Classification Metrics**:
> This is a small-sample secondary diagnostic classification experiment on 17 holdout institutions. Because institutions clustered distinctly on either side of the ₹7.0 LPA threshold in this split, the diagnostic achieved perfect separation. This result must not be presented as the main project metric or interpreted as evidence of 100% overall system accuracy. Continuous regression ($R^2 = 0.6418$, $\text{MAE} = 3.22\text{ LPA}$, $\text{RMSE} = 3.88\text{ LPA}$) remains our primary headline metric.

---

## 7. Model Explainability & Permutation Importance

Impurity-based (Gini) feature importance in tree models is notoriously biased toward high-cardinality continuous features. Therefore, **Permutation Feature Importance** was computed on the holdout test split:

```
Permutation Feature Importance (Holdout Test Set Drop in R²):
1. student_faculty_ratio   : 0.5814 ± 0.165
2. location_score          : 0.5368 ± 0.142
3. student_life_score      : 0.2409 ± 0.088
4. total_approved_intake   : 0.2109 ± 0.076
5. faculty_count           : 0.1829 ± 0.065
6. infrastructure_score    : 0.1502 ± 0.052
7. tuition_fee_annual      : 0.0470 ± 0.021
8. academic_score          : 0.0045 ± 0.003
9. is_government           : 0.0001 ± 0.0001
```

### Causal Disclaimer
> [!WARNING]
> **Feature Importance Does NOT Imply Causation!**
> While `student_faculty_ratio` and `location_score` show high predictive importance, this does **NOT** mean that arbitrarily changing faculty count or relocating a campus directly causes student salaries to increase.
> Rather, premier institutions (IITs, BITS, NITs, top autonomous colleges) that attract high-paying tech employers also maintain low student-faculty ratios and high metropolitan connectivity. The model leverages these features as statistical proxies for institutional stature, not as causal levers.

---

## 8. Limitations & Bias

1. **Reporting Bias**: The verified supervised-learning cohort is smaller because only institutions with sufficiently complete and verified placement disclosures were eligible for model training.
2. **Geographic Skew**: Urban institutions located in Bengaluru, Pune, Hyderabad, and Delhi NCR experience localized recruiter concentration that may not reflect pure instructional quality.
3. **Verified Disclosures Cohort**: Because our rigorous audit rejected synthetic matches and unverified claims, the verified regression dataset is restricted to $N = 109$ fully audited NIRF institutions.
4. **Prediction Guardrail**: All model predictions are strictly labeled in the application UI with:
   - `predicted = True`
   - Data coverage badges (`🟢 High`, `🟡 Moderate`, `⚪ Limited`)
   - Estimated prediction uncertainty bounds based on data coverage ($\pm 2.0$ LPA for moderate coverage; $\pm 3.0$ LPA for limited coverage). These are estimated uncertainty bounds based on data-coverage tiers and are not statistically calibrated confidence intervals.
