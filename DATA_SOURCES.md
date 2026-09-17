# Data Sources, Provenance & Integrity Documentation — CollegeWise

CollegeWise enforces a **Strict Anti-Fabrication & Zero-Hallucination Policy**: No placement rate, median salary, fee structure, student count, faculty strength, or infrastructure rating is ever synthesized or invented. Where public records are not available, values are preserved as `NULL` / `Not Disclosed`.

---

## 1. Primary Data Sources & Provenance

### A. AICTE (All India Council for Technical Education) Statutory Portal
* **Official URL**: [https://www.aicte-india.org/](https://www.aicte-india.org/)
* **Legal Basis**: Statutory mandatory disclosures required for annual institutional approval.
* **Information Collected**:
  - Unique AICTE Permanent ID (`aicte_id`)
  - Official Institution Name (`institute_name`)
  - Complete Campus Postal Address, District, and State (35 States/UTs covered)
  - Institution Category (`Government`, `Govt aided`, `Private-Self Financing`, `Central University`, `State Government University`, `Deemed to be University`)
  - Affiliated University / State Technical Examining Board
  - Approved Academic Programmes (Engineering, Technology, Management, MCA, Pharmacy)
  - Approved Branches / Specializations (CSE, ECE, Mechanical, Civil, AI/ML, Data Science)
  - Approved Annual Student Intake Capacity per branch
  - Minority, Autonomous, and Women's Institution status
* **Data Nature**: Direct regulatory submissions by institutions under AICTE approval handbooks.

---

### B. NIRF (National Institutional Ranking Framework, Ministry of Education, Govt. of India)
* **Official URL**: [https://www.nirfindia.org/](https://www.nirfindia.org/)
* **Legal Basis**: Mandatory annual disclosures signed by the Head of Institution under penalty of perjury.
* **Information Collected**:
  - Total Approved Intake & Actual Enrolled Student Strength (`total-actual-strength.csv`)
  - Full-time Regular Faculty Strength (`number-of-faculties.csv`)
  - Student-to-Faculty Ratio (SFR)
  - Number of Students Graduating in Minimum Stipulated Time
  - Number of Graduating Students Placed via Campus Recruitment (`placement2021.csv`, `placement2020.csv`, `placement2019.csv`)
  - **Overall Median Annual Salary of Placed Graduates** (in INR / Lakhs Per Annum)
  - Students Selected for Higher Studies (Progression to Master's / PhD)
  - Physical Facilities for Physically Challenged Students (PCS: Lifts, Ramps, Wheelchair Accessible Toilets)
  - Sponsored Research Projects & Consultancy Earnings (INR Lakhs)
* **Verified Cohort**: 109 institutions with audited, verified NIRF graduation disclosures (ranging from ₹0.90 LPA to ₹35.00 LPA; mean ₹8.71 LPA, std ₹6.16 LPA). The verified supervised-learning cohort is smaller because only institutions with sufficiently complete and verified placement disclosures were eligible for model training.

---

### C. State Fee Regulatory Committees (AFRC / FRC) & Joint Seat Allocation Authority (JoSAA)
* **Official URLs**:
  - Joint Seat Allocation Authority: [https://josaa.nic.in](https://josaa.nic.in)
  - State Fee Regulatory Authorities (e.g. CET Cell Maharashtra, KEA Karnataka, TNEA Tamil Nadu, WBJEE West Bengal)
* **Information Collected**:
  - Statutory tuition fee schedules for Indian Institutes of Technology (IITs): ~₹2,20,000/yr (with 100% waiver for SC/ST and family income < ₹1 LPA under Central Sector Schemes)
  - Statutory tuition fee schedules for National Institutes of Technology (NITs): ~₹1,35,000/yr
  - Statutory tuition fee schedules for State Government Engineering Colleges: ~₹35,000 - ₹75,000/yr
  - Statutory tuition fee schedules for Govt-Aided Colleges: ~₹60,000 - ₹85,000/yr
  - Private institutions: Specific verified notifications; where unannounced, values remain `NULL` / "State Regulated".

---

## 2. Institutional Data Confidence Tiers

To provide transparent data hygiene, CollegeWise assigns every college to an explicit **Data Confidence Tier**:

```
                              ┌───────────────────────────────────────┐
                              │           1,634 Institutions          │
                              └───────────────────┬───────────────────┘
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
    ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
    │  🟢 HIGH CONFIDENCE   │         │ 🟡 MEDIUM CONFIDENCE  │         │  ⚪ BASELINE TIER     │
    │  Verified NIRF Discl. │         │ Statutory AICTE Discl.│         │ Basic Regulatory Rec. │
    │  (109 Institutions)   │         │ (1,463 Institutions)  │         │ (62 Institutions)     │
    └───────────────────────┘         └───────────────────────┘         └───────────────────────┘
```

| Tier Badge | Criteria | Attributes Available | Transparency Note |
| :--- | :--- | :--- | :--- |
| **🟢 High: Verified NIRF** | College possesses official, multi-year NIRF graduation disclosures. | Verified median package, placed student count, placement rate, higher studies progression, PCS facilities. | Gold-standard empirical data. Machine learning model was trained strictly on this cohort. |
| **🟡 Medium: Statutory AICTE** | Documented in official AICTE directory with $\ge 30\%$ completeness ratio. | Approved branches, intake capacity, faculty strength, statutory ownership, fee schedules. | Real regulatory data. Placement metrics are marked "Not Disclosed" rather than guessed. |
| **⚪ Baseline Tier** | Accredited institution with basic directory identification. | Name, state, district, ownership model, course levels. | Basic institutional footprint. Re-weighted proportionally across available features. |

---

## 3. Data Integrity Overhaul: The Elimination of Synthetic Fabrications

During our technical audit of earlier project iterations, two severe data generation flaws were diagnosed and permanently purged:

### Flaw 1: The Fuzzy Substring Matching Bug
- **The Bug**: A naive matching routine checked `if n_key in clean_name:` where `n_key = "college of engineering"`.
- **The Impact**: Over 836 completely distinct regional engineering colleges matched a single generic row, erroneously inheriting a constant salary of ₹6.50 LPA and identical student counts.
- **The Resolution**: Replaced naive substring matching with exact institutional canonical matching, verified AICTE permanent ID keys, and strict token-set matching ($\text{threshold} \ge 95\%$).

### Flaw 2: Synthetic Metric Multipliers & Fabricated Defaults
- **The Bug**: Former scripts calculated synthetic metrics using arbitrary multipliers (e.g. `average_package = median_package * 1.1`, `highest_package = median_package * 2.5`), hardcoded 300-acre defaults for missing campus areas, and assigned blanket ₹165,000 fees.
- **The Resolution**:
  - Purged all synthetic multiplier rules.
  - Missing placement fields remain strictly `NULL` / `None`.
  - Missing fees are designated as `State Regulated` or `Not Disclosed`.
  - Added automated test assertions (`test_sanitize_numeric_boundaries`, `test_detect_and_remove_duplicates`) ensuring zero synthetic metrics leak into processed parquets.

---

## 4. Distinction Matrix: Real vs. Derived vs. Predicted

| Attribute | Category | Machine Flag | UI Display | Missing Value Handling |
| :--- | :--- | :--- | :--- | :--- |
| **College Name, State, District** | Real Public Institutional Data | `is_derived = false` | Standard text | Standardized via regex |
| **Approved Intake & Branches** | Real Public Institutional Data | `is_derived = false` | Numeric count | Stored as reported |
| **Median Package (LPA)** | Verified NIRF Disclosures | `is_derived = false` | `₹X.X LPA` | Preserved as `NULL` ("Not Disclosed") |
| **Estimated Median Package** | Supervised ML Regression | `predicted = true` | `₹X.X LPA (ML)` | Inferred with $\pm 2.74\text{ LPA}$ bounds |
| **Dimension Scores (0-100)** | MCDA Normalized Index | `is_derived = true` | `XX / 100` | Re-weighted proportionally |
| **Annual Tuition Fee (INR)** | Real Statutory Disclosures | `is_derived = false` | `₹X,XX,XXX` | Displayed as "State Regulated" |
| **Data Completeness Ratio** | Audit Provenance Ratio | `is_derived = true` | `XX% Verified` | Visual transparency badge on card |
