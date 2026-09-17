# -*- coding: utf-8 -*-
"""
CollegeWise: Comprehensive Technical Architecture, Implementation & Interview Defense Handbook.
Generates an interview-grade, professional PDF document using ReportLab.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = r"d:\projects\college\CollegeWise_Comprehensive_System_Architecture_and_Interview_Defense.pdf"

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page count and headers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Cover page decoration
        if self._pageNumber == 1:
            self.setFillColor(colors.HexColor("#0F2042"))
            self.rect(0, 11 * 72 - 18, 8.5 * 72, 18, fill=1, stroke=0)
            self.setFillColor(colors.HexColor("#1E40AF"))
            self.rect(0, 0, 8.5 * 72, 16, fill=1, stroke=0)
            self.restoreState()
            return

        # Running Header
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#1E40AF"))
        self.drawString(45, 11 * 72 - 30, "COLLEGEWISE")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(110, 11 * 72 - 30, "|  Technical Architecture & System Defense Handbook")
        
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(45, 11 * 72 - 35, 8.5 * 72 - 45, 11 * 72 - 35)

        # Running Footer
        self.line(45, 38, 8.5 * 72 - 45, 38)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(45, 26, "Preetika Anjana — ML / Software Engineering Interview Handbook (CollegeWise)")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 45, 26, page_text)
        
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0F2042")      # Deep Navy
    SECONDARY = colors.HexColor("#1E40AF")    # Royal Blue
    ACCENT = colors.HexColor("#0D9488")       # Teal
    TEXT_DARK = colors.HexColor("#1E293B")    # Slate Dark
    TEXT_MUTED = colors.HexColor("#64748B")   # Slate Muted
    BG_LIGHT = colors.HexColor("#F8FAFC")     # Card Light Gray
    BORDER_LIGHT = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=TEXT_DARK,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=TEXT_DARK,
        spaceAfter=5
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=TEXT_DARK
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1E293B")
    )

    callout_bold = ParagraphStyle(
        'CalloutBold',
        parent=callout_style,
        fontName='Helvetica-Bold'
    )

    def make_callout(title, text, bg_hex="#F0FDF4", border_hex="#10B981"):
        content = [
            Paragraph(f"<b>{title}</b>", callout_bold),
            Spacer(1, 3),
            Paragraph(text, callout_style)
        ]
        t = Table([[content]], colWidths=[522])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_hex)),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor(border_hex)),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        return t

    story = []

    # =========================================================================
    # SECTION 1: HEADER & EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("CollegeWise", title_style))
    story.append(Paragraph("Personalized ML-Based College Decision Support System", subtitle_style))
    
    meta_text = (
        "<b>Author:</b> Preetika Anjana &nbsp;|&nbsp; "
        "<b>Target Role:</b> Infosys Specialist Programmer L3 / ML Software Engineer &nbsp;|&nbsp; "
        "<b>Repository:</b> github.com/preetikaanjana/CollegeWise"
    )
    story.append(Paragraph(meta_text, body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceBefore=4, spaceAfter=8))

    story.append(Paragraph("1. Executive Summary & Core Motivation", h1_style))
    p1 = (
        "<b>The Problem in Indian Higher Education:</b> Annually, over 2.5 million students clear engineering "
        "examinations (JEE Main, Advanced, State CETs). The existing college selection ecosystem relies on "
        "monolithic 'Top 100' league tables that enforce an arbitrary, one-size-fits-all ranking. These tables "
        "ignore individual family budget constraints, regional connectivity, and personal career goals. Furthermore, "
        "commercial portals routinely report inflated average compensation driven by extreme international outliers "
        "(e.g., ₹1.5 Crore), masking the true 50th-percentile outcomes of graduating cohorts."
    )
    story.append(Paragraph(p1, body_style))

    p2 = (
        "<b>The CollegeWise Solution:</b> CollegeWise is an end-to-end decision-support platform designed to transform "
        "college discovery from a static ranking list into an individualized, multi-criteria decision process. It combines "
        "supervised continuous machine learning for institutional compensation estimation with Multi-Criteria Decision "
        "Analysis (MCDA) via Simple Additive Weighting (SAW). The platform guarantees strict data hygiene (zero synthetic "
        "data fabrication), anti-leakage ML design, explainable recommendations ('Why this college?'), and interactive "
        "sensitivity analysis."
    )
    story.append(Paragraph(p2, body_style))

    # Core Stats Table
    stats_data = [
        [
            Paragraph("<b>Total Recommendation Pool</b>", table_header),
            Paragraph("<b>Verified Supervised ML Cohort</b>", table_header),
            Paragraph("<b>Supervised ML Splits</b>", table_header),
            Paragraph("<b>Champion Model</b>", table_header),
            Paragraph("<b>Holdout Test R² / MAE</b>", table_header)
        ],
        [
            Paragraph("<b>1,634</b> Institutions<br/>(AICTE Disclosures)", table_cell),
            Paragraph("<b>109</b> Verified NIRF<br/>(Full Placement Discl.)", table_cell),
            Paragraph("<b>75</b> Train (68.8%)<br/><b>17</b> Val / <b>17</b> Test", table_cell),
            Paragraph("<b>Gradient Boosting</b><br/>(n=100, lr=0.08, d=3)", table_cell),
            Paragraph("<b>R² = 0.6418</b><br/><b>MAE = 3.22 LPA</b>", table_cell)
        ]
    ]
    t_stats = Table(stats_data, colWidths=[105, 105, 100, 110, 102])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('BACKGROUND', (0,1), (-1,1), BG_LIGHT),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 2: END-TO-END SYSTEM ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("2. System Architecture & Clean Layer Separation", h1_style))
    arch_desc = (
        "A critical engineering strength of CollegeWise is the <b>strict decoupling of the Machine Learning "
        "layer from the Recommendation layer</b>. Machine learning models should never directly 'rank' colleges; "
        "rather, ML predicts an objective, unobserved attribute (expected institutional median package), while the "
        "MCDA engine evaluates multi-dimensional institutional fit based on the human user's personal priority weights."
    )
    story.append(Paragraph(arch_desc, body_style))

    arch_layers = [
        [
            Paragraph("<b>Layer</b>", table_header),
            Paragraph("<b>Component & Technology</b>", table_header),
            Paragraph("<b>Responsibilities & Engineering Guarantees</b>", table_header)
        ],
        [
            Paragraph("<b>Presentation & UI Layer</b>", table_cell),
            Paragraph("Streamlit, Custom CSS, Session State, Plotly", table_cell),
            Paragraph("Interactive 100-pt budget sliders with mutual constraints; multi-college radar charts; sensitivity slope chart; responsive card grid; session state caching.", table_cell)
        ],
        [
            Paragraph("<b>Decision & Recommendation Layer</b>", table_cell),
            Paragraph("MCDA Engine, Simple Additive Weighting (SAW)", table_cell),
            Paragraph("Hard filtering (budget ceilings, state, ownership); linear weighted aggregation; dynamic missing-feature re-normalization; transparent 'Why this college?' attribution.", table_cell)
        ],
        [
            Paragraph("<b>Scoring & Normalization Layer</b>", table_cell),
            Paragraph("Feature Engine, Min-Max & Inverted Scalers", table_cell),
            Paragraph("Maps diverse institutional metrics (intake, fees, infrastructure) into 6 standardized continuous scores in [0, 100]. Incorporates verified or ML-estimated placement figures.", table_cell)
        ],
        [
            Paragraph("<b>Supervised ML Layer</b>", table_cell),
            Paragraph("Gradient Boosting Champion, Scikit-Learn, Joblib", table_cell),
            Paragraph("Anti-leakage feature matrix ($X \\in \\mathbb{R}^9$); continuous median package forecasting; empirical uncertainty bounds (±2.0 / ±3.0 LPA); out-of-fold cross-validation.", table_cell)
        ],
        [
            Paragraph("<b>Data Storage & Provenance Layer</b>", table_cell),
            Paragraph("Apache Parquet, PyArrow, SQLite", table_cell),
            Paragraph("High-performance columnar storage for fast vector filtering; zero-fabrication data schema; deduplicated canonical records with persistent AICTE IDs.", table_cell)
        ]
    ]
    t_arch = Table(arch_layers, colWidths=[110, 130, 282])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT])
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 3: FRONTEND ARCHITECTURE & UX INNOVATIONS
    # =========================================================================
    story.append(Paragraph("3. Frontend Architecture & Interactive Features", h1_style))
    story.append(Paragraph(
        "The frontend is implemented in <b>Streamlit</b> with custom CSS styling and state-machine synchronization. "
        "It provides four distinct decision-support modules:",
        body_style
    ))

    features_list = [
        "<b>Dynamic 100-Point Budget Slider Mechanism:</b> Solves the common user failure where individuals set every "
        "priority to 100%. The system enforces a strict mathematical constraint: $\\sum_{j=1}^6 w_j = 100$. When a "
        "student increases one priority, the remaining sliders dynamically adjust proportionally, compelling realistic "
        "trade-off evaluation.",
        "<b>Interactive Sensitivity Analysis (Scenario A vs. Scenario B):</b> Allows students to define two separate "
        "priority profiles and observe real-time institutional rank movements on a Plotly slope chart. Displacements are "
        "framed neutrally: rank shifts illustrate alignment with human priorities, not absolute changes in institutional quality.",
        "<b>Multi-College Comparison Radar Charts:</b> Side-by-side multi-axial visualization across all 6 decision pillars "
        "for 2 to 4 shortlisted institutions, enabling instant visual diagnosis of trade-offs.",
        "<b>College Deep Dive & Provenance Badges:</b> Transparent inspection of underlying data with Data Confidence badges: "
        "🟢 High Coverage (audited NIRF disclosure), 🟡 Moderate Coverage (statutory AICTE data, ML package ±2.0 LPA bound), "
        "and ⚪ Baseline (statutory identification, ML package ±3.0 LPA bound)."
    ]
    for item in features_list:
        story.append(Paragraph(f"• {item}", bullet_style))

    story.append(Spacer(1, 6))

    # Callout: Budget Slider
    slider_callout = (
        "<b>Interview Note on the Dynamic Slider:</b> In Streamlit, each slider is bound to <code>st.session_state</code>. "
        "When an <code>on_change</code> callback fires for slider $k$, the difference $\\Delta = w_k^{new} - w_k^{old}$ is "
        "subtracted proportionally from the remaining 5 sliders, followed by clamping to $[0, 100]$ and rounding. This "
        "guarantees that the student's priority budget always sums exactly to 100 points without jarring page reloads."
    )
    story.append(make_callout("UX Engineering: 100-Point Budget Invariant", slider_callout, "#EFF6FF", "#3B82F6"))
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 4: DATASET PROVENANCE & ANTI-FABRICATION HYGIENE
    # =========================================================================
    story.append(Paragraph("4. Data Provenance & Anti-Fabrication Data Hygiene", h1_style))
    story.append(Paragraph(
        "Commercial college counseling platforms frequently fabricate data through synthetic multipliers or naive defaults. "
        "CollegeWise enforces a rigorous <b>Zero-Fabrication Policy</b> backed exclusively by official government sources:",
        body_style
    ))

    data_sources = [
        "<b>AICTE Official Disclosures:</b> Directory of approved institutions across 35 states/UTs, providing approved "
        "programmes, intake capacity, statutory ownership categories, and AICTE Permanent IDs.",
        "<b>NIRF (Ministry of Education, Govt. of India):</b> Official audited disclosures (2019-2021) covering enrolled strength, "
        "faculty counts, graduating batch size, campus recruitment placements, and median annual compensation (LPA).",
        "<b>State Fee Regulatory Committees (AFRC/FRC) & JoSAA:</b> Statutory tuition fee schedules for central (IITs/NITs) "
        "and state-approved institutions."
    ]
    for ds in data_sources:
        story.append(Paragraph(f"• {ds}", bullet_style))

    story.append(Paragraph(
        "<b>Why the ML Dataset ($N=109$) is Smaller than the Recommendation Pool ($N=1,634$):</b><br/>"
        "The verified supervised-learning cohort is smaller because only institutions with sufficiently complete and "
        "verified placement disclosures were eligible for model training. Rather than fabricating fake placement figures "
        "for the remaining 1,525 institutions, we trained supervised models strictly on the 109 verified records and used "
        "the model to infer packages with explicit, coverage-based prediction uncertainty bounds.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 5: FEATURE ENGINEERING & ANTI-LEAKAGE PROTOCOL
    # =========================================================================
    story.append(Paragraph("5. Feature Engineering & Anti-Leakage Protocol", h1_style))
    story.append(Paragraph(
        "Institutional capability is structured into <b>6 standardized decision pillars</b> normalized to $[0, 100]$: "
        "<b>Placement & Career</b>, <b>Campus & Infrastructure</b>, <b>Academic Quality</b>, <b>Affordability</b>, "
        "<b>Location & Connectivity</b>, and <b>Student Life</b>.",
        body_style
    ))

    leakage_callout = (
        "<b>Strict Anti-Leakage Feature Matrix ($X \\in \\mathbb{R}^9$):</b><br/>"
        "To prevent target leakage when predicting median package, the supervised ML feature matrix strictly excludes: "
        "<code>placement_rate</code>, <code>placed_students_count</code>, average salary, highest salary, and the engineered "
        "<code>placement_score</code>. Furthermore, <code>affordability_score</code> was excluded because it duplicates "
        "<code>tuition_fee_annual</code> ($r \\approx -0.98$). Automated unit tests assert that no input feature has a "
        "correlation $> 0.90$ with the target variable."
    )
    story.append(make_callout("ML Engineering: Zero Target Leakage Guarantee", leakage_callout, "#FFFBEB", "#F59E0B"))
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 6: SUPERVISED ML CONTINUOUS REGRESSION & BENCHMARKS
    # =========================================================================
    story.append(Paragraph("6. Supervised ML: Continuous Regression Benchmark", h1_style))
    story.append(Paragraph(
        "<b>Formulation:</b> Continuous regression predicting <code>median_package_lpa</code> (mean ₹8.71 LPA, std ₹6.16 LPA, "
        "min ₹0.90 LPA, max ₹35.00 LPA). Continuous regression was chosen over classification because compensation varies "
        "smoothly; discretizing into arbitrary buckets ('High' vs 'Low') discards metric distance and creates artificial threshold errors.",
        body_style
    ))

    # Model Comparison Table
    model_table_data = [
        [
            Paragraph("<b>Model Architecture</b>", table_header),
            Paragraph("<b>5x5 Repeated CV MAE</b>", table_header),
            Paragraph("<b>5x5 Repeated CV R²</b>", table_header),
            Paragraph("<b>Test MAE (LPA)</b>", table_header),
            Paragraph("<b>Test RMSE (LPA)</b>", table_header),
            Paragraph("<b>Test R²</b>", table_header),
            Paragraph("<b>Architecture Status</b>", table_header)
        ],
        [
            Paragraph("<b>Gradient Boosting</b>", table_cell),
            Paragraph("<b>2.95 ± 0.65</b>", table_cell),
            Paragraph("<b>0.37 ± 0.37</b>", table_cell),
            Paragraph("<b>3.22</b>", table_cell),
            Paragraph("<b>3.88</b>", table_cell),
            Paragraph("<b>0.6418</b>", table_cell),
            Paragraph("<b>Champion Model</b>", table_cell)
        ],
        [
            Paragraph("Random Forest (n=150)", table_cell),
            Paragraph("2.95 ± 0.75", table_cell),
            Paragraph("0.44 ± 0.26", table_cell),
            Paragraph("3.24", table_cell),
            Paragraph("3.94", table_cell),
            Paragraph("0.6300", table_cell),
            Paragraph("Ensemble Bagging", table_cell)
        ],
        [
            Paragraph("XGBoost Regressor", table_cell),
            Paragraph("3.22 ± 0.78", table_cell),
            Paragraph("0.26 ± 0.38", table_cell),
            Paragraph("3.65", table_cell),
            Paragraph("4.48", table_cell),
            Paragraph("0.5211", table_cell),
            Paragraph("Regularized Boosting", table_cell)
        ],
        [
            Paragraph("Linear Regression (OLS)", table_cell),
            Paragraph("3.65 ± 0.92", table_cell),
            Paragraph("0.12 ± 0.42", table_cell),
            Paragraph("3.98", table_cell),
            Paragraph("4.71", table_cell),
            Paragraph("0.4704", table_cell),
            Paragraph("Linear Baseline", table_cell)
        ],
        [
            Paragraph("Ridge (α=10.0)", table_cell),
            Paragraph("3.61 ± 0.89", table_cell),
            Paragraph("0.18 ± 0.39", table_cell),
            Paragraph("4.22", table_cell),
            Paragraph("4.99", table_cell),
            Paragraph("0.4066", table_cell),
            Paragraph("L2 Regularized", table_cell)
        ],
        [
            Paragraph("Decision Tree (depth=5)", table_cell),
            Paragraph("3.68 ± 1.10", table_cell),
            Paragraph("0.08 ± 0.48", table_cell),
            Paragraph("4.09", table_cell),
            Paragraph("5.14", table_cell),
            Paragraph("0.3693", table_cell),
            Paragraph("Single Tree Baseline", table_cell)
        ],
        [
            Paragraph("Dummy Regressor (Mean)", table_cell),
            Paragraph("4.98 ± 1.15", table_cell),
            Paragraph("-0.42 ± 0.45", table_cell),
            Paragraph("5.83", table_cell),
            Paragraph("6.55", table_cell),
            Paragraph("-0.0211", table_cell),
            Paragraph("Naive Central Tendency", table_cell)
        ]
    ]
    t_models = Table(model_table_data, colWidths=[110, 80, 80, 65, 65, 55, 67])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('ALIGN', (1,0), (-2,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#EFF6FF"), colors.white, BG_LIGHT, colors.white, BG_LIGHT, colors.white, BG_LIGHT])
    ]))
    story.append(t_models)
    story.append(Spacer(1, 6))

    # Champion Performance Bullet Points
    story.append(Paragraph(
        "<b>Key Takeaways from the Empirical Benchmark:</b><br/>"
        "• <b>Dummy Baseline Defeat:</b> The naive Dummy baseline achieved a negative Test $R^2$ (-0.0211) and Test MAE of 5.83 LPA. "
        "The champion Gradient Boosting model reduced MAE to 3.22 LPA—a <b>44.8% error reduction</b>, verifying genuine signal capture.<br/>"
        "• <b>Non-Linear Superiority:</b> Linear regression underperformed ($R^2=0.4704$, MAE=3.98 LPA) because technical education "
        "compensation exhibits threshold effects (e.g., small elite IITs command high packages despite smaller intake, whereas massive "
        "regional colleges report lower compensation). Tree ensembles split feature space into rectangular decision regions that model these interactions natively.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Permutation Feature Importance Table
    pfi_data = [
        [
            Paragraph("<b>Rank</b>", table_header),
            Paragraph("<b>Feature Name</b>", table_header),
            Paragraph("<b>Mean Importance (MAE Drop)</b>", table_header),
            Paragraph("<b>Std Dev</b>", table_header),
            Paragraph("<b>Domain Meaning & Statistical Rationale</b>", table_header)
        ],
        [
            Paragraph("1", table_cell),
            Paragraph("<code>student_faculty_ratio</code>", table_cell),
            Paragraph("<b>0.581</b>", table_cell),
            Paragraph("± 0.142", table_cell),
            Paragraph("Proxies instructional attention and elite institutional resource density.", table_cell)
        ],
        [
            Paragraph("2", table_cell),
            Paragraph("<code>location_score</code>", table_cell),
            Paragraph("<b>0.537</b>", table_cell),
            Paragraph("± 0.118", table_cell),
            Paragraph("Proximity to major metropolitan IT/industrial hubs (Bengaluru, Hyderabad, Pune).", table_cell)
        ],
        [
            Paragraph("3", table_cell),
            Paragraph("<code>student_life_score</code>", table_cell),
            Paragraph("<b>0.241</b>", table_cell),
            Paragraph("± 0.075", table_cell),
            Paragraph("Extracurricular scale, technical societies, and peer network vibrancy.", table_cell)
        ],
        [
            Paragraph("4", table_cell),
            Paragraph("<code>total_approved_intake</code>", table_cell),
            Paragraph("<b>0.211</b>", table_cell),
            Paragraph("± 0.063", table_cell),
            Paragraph("Institutional batch scale and recruiter talent pool breadth.", table_cell)
        ],
        [
            Paragraph("5", table_cell),
            Paragraph("<code>faculty_quality_score</code>", table_cell),
            Paragraph("<b>0.183</b>", table_cell),
            Paragraph("± 0.052", table_cell),
            Paragraph("Ph.D. faculty ratio and sponsored research track record.", table_cell)
        ],
        [
            Paragraph("6", table_cell),
            Paragraph("<code>infrastructure_score</code>", table_cell),
            Paragraph("<b>0.150</b>", table_cell),
            Paragraph("± 0.048", table_cell),
            Paragraph("Physical campus facilities, laboratories, and computing assets.", table_cell)
        ],
        [
            Paragraph("7", table_cell),
            Paragraph("<code>tuition_fee_annual</code>", table_cell),
            Paragraph("<b>0.047</b>", table_cell),
            Paragraph("± 0.019", table_cell),
            Paragraph("Capital expenditure capability vs. state regulatory caps.", table_cell)
        ]
    ]
    t_pfi = Table(pfi_data, colWidths=[30, 120, 95, 60, 217])
    t_pfi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('ALIGN', (2,0), (3,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT])
    ]))
    story.append(t_pfi)
    story.append(Spacer(1, 6))

    # Causal & Classification Disclaimer
    causal_callout = (
        "<b>Important Scientific Disclaimers:</b><br/>"
        "• <b>Correlation vs. Causation:</b> Permutation importance indicates predictive importance within the model, "
        "<b>not real-world causation</b>. Relocating a campus does not causally increase student salaries; location proxies for "
        "recruiter concentration and ecosystem maturity.<br/>"
        "• <b>Secondary Diagnostic Classification Experiment ($N=17$):</b> When audited on a binary threshold (Median Package $\\ge$ ₹7.0 LPA), "
        "the model achieved 100% Accuracy, Precision, Recall, and F1. This was a small-sample secondary diagnostic experiment on 17 holdout institutions "
        "where institutions naturally clustered across the boundary. It is <b>not</b> the headline ML metric; continuous regression remains our primary metric.<br/>"
        "• <b>Prediction Uncertainty Bounds:</b> Estimated bounds (±2.0 to ±3.0 LPA) are based on data-coverage tiers and are "
        "<b>not statistically calibrated confidence intervals</b>."
    )
    story.append(make_callout("Scientific Integrity: Causal & Metric Boundaries", causal_callout, "#FFFBEB", "#F59E0B"))
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 7: MULTI-CRITERIA DECISION ANALYSIS (MCDA / SAW)
    # =========================================================================
    story.append(Paragraph("7. Recommendation Science: Multi-Criteria Decision Analysis", h1_style))
    story.append(Paragraph(
        "CollegeWise implements <b>Simple Additive Weighting (SAW)</b>, the foundational technique of Multi-Criteria "
        "Decision Analysis (MCDA). The personalized match score for college $i$ is formulated as:",
        body_style
    ))

    formula_text = (
        "<font size='9' color='#0F2042'><b>Match Score<sub>i</sub> = "
        "&sum;<sub>j=1..6</sub> ( w<sub>j</sub> &times; s<sub>ij</sub> )</b></font>"
        "&nbsp;&nbsp;&nbsp;&nbsp;subject to&nbsp;&nbsp;&nbsp;&nbsp;"
        "<font size='9' color='#0F2042'><b>&sum;<sub>j=1..6</sub> w<sub>j</sub> = 100</b></font>"
        "&nbsp;&nbsp;and&nbsp;&nbsp;"
        "<font size='9' color='#0F2042'><b>s<sub>ij</sub> &isin; [0, 100]</b></font>"
    )
    story.append(Paragraph(formula_text, body_style))
    story.append(Spacer(1, 4))

    mcda_points = [
        "<b>Linearity & Explainability:</b> The change in an institution's score under priority adjustments is strictly "
        "$\\Delta S_i = \\sum \\Delta w_j \\cdot s_{ij}$. Unlike black-box collaborative filtering or non-linear TOPSIS "
        "(which introduces rank-reversal paradoxes when candidates are added/removed), SAW provides full mathematical transparency.",
        "<b>Hard Constraint Filtering:</b> Prior to scoring, candidates are filtered through strict constraints: annual tuition "
        "budget ceilings, state/regional location filters, and ownership type (Government vs. Private).",
        "<b>Zero-Fabrication Re-normalization:</b> If an institution lacks verified data for a non-essential dimension, the scoring "
        "engine dynamically re-normalizes the match score across available verified pillars rather than penalizing the college with a zero."
    ]
    for pt in mcda_points:
        story.append(Paragraph(f"• {pt}", bullet_style))

    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 8: TECH STACK IN-DEPTH ANALYSIS
    # =========================================================================
    story.append(Paragraph("8. Technology Stack: Working Principles & Justifications", h1_style))
    
    tech_data = [
        [
            Paragraph("<b>Category</b>", table_header),
            Paragraph("<b>Technology</b>", table_header),
            Paragraph("<b>Role in CollegeWise</b>", table_header),
            Paragraph("<b>How it Works Under the Hood</b>", table_header)
        ],
        [
            Paragraph("<b>Core Language</b>", table_cell),
            Paragraph("<b>Python 3.12+</b>", table_cell),
            Paragraph("Runtime environment for entire pipeline", table_cell),
            Paragraph("High-performance bytecode compilation, enhanced error tracebacks, standard library support.", table_cell)
        ],
        [
            Paragraph("<b>Application & UI</b>", table_cell),
            Paragraph("<b>Streamlit</b>", table_cell),
            Paragraph("Reactive web interface & dashboard", table_cell),
            Paragraph("Runs Python scripts top-to-bottom on interaction; uses session state to preserve widget state across reruns.", table_cell)
        ],
        [
            Paragraph("<b>Data Science</b>", table_cell),
            Paragraph("<b>Pandas & NumPy</b>", table_cell),
            Paragraph("Data manipulation & vectorized math", table_cell),
            Paragraph("C-optimized 2D dataframes and contiguous memory ndarrays executing SIMD-accelerated linear algebra.", table_cell)
        ],
        [
            Paragraph("<b>Machine Learning</b>", table_cell),
            Paragraph("<b>Scikit-Learn</b>", table_cell),
            Paragraph("Pipelines, regression, cross-validation", table_cell),
            Paragraph("Standardized fit/transform interfaces; ColumnTransformers preventing data leakage across folds.", table_cell)
        ],
        [
            Paragraph("<b>Gradient Boosting</b>", table_cell),
            Paragraph("<b>GBDT & XGBoost</b>", table_cell),
            Paragraph("Champion compensation predictors", table_cell),
            Paragraph("Sequentially fits decision trees to pseudo-residuals of loss function using shrinkage learning rate.", table_cell)
        ],
        [
            Paragraph("<b>Visualization</b>", table_cell),
            Paragraph("<b>Plotly</b>", table_cell),
            Paragraph("Interactive slope & radar charts", table_cell),
            Paragraph("Generates declarative JSON specifications rendered via browser-side Plotly.js (WebGL/SVG).", table_cell)
        ],
        [
            Paragraph("<b>Data Storage</b>", table_cell),
            Paragraph("<b>Parquet & PyArrow</b>", table_cell),
            Paragraph("High-speed columnar storage", table_cell),
            Paragraph("Dictionary encoding, Snappy compression, column-chunk skipping without loading entire tables into RAM.", table_cell)
        ],
        [
            Paragraph("<b>Database</b>", table_cell),
            Paragraph("<b>SQLite</b>", table_cell),
            Paragraph("Local relational metadata storage", table_cell),
            Paragraph("Serverless, zero-configuration ACID-compliant single-file database for structured audit queries.", table_cell)
        ],
        [
            Paragraph("<b>Testing</b>", table_cell),
            Paragraph("<b>Pytest</b>", table_cell),
            Paragraph("Automated test suite (20 tests)", table_cell),
            Paragraph("Test discovery, parametrized fixtures, assertion rewriting for exact verification of boundaries and leakage.", table_cell)
        ],
        [
            Paragraph("<b>Text Extraction</b>", table_cell),
            Paragraph("<b>Python <code>re</code></b>", table_cell),
            Paragraph("Regex parsing of raw salary strings", table_cell),
            Paragraph("Deterministic finite automaton (DFA) regex engine parsing numeric amounts from government PDF tables.", table_cell)
        ]
    ]
    t_tech = Table(tech_data, colWidths=[80, 85, 135, 222])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT])
    ]))
    story.append(t_tech)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 9: 27 COMPREHENSIVE INTERVIEW DEFENSE Q&AS
    # =========================================================================
    story.append(Paragraph("9. Comprehensive Technical Interview Defense (27 Core Q&As)", h1_style))
    story.append(Paragraph(
        "Tailored for the <b>Infosys Specialist Programmer L3 / ML Software Engineer</b> technical round:",
        body_style
    ))

    qa_list = [
        (
            "1. What is CollegeWise?",
            "CollegeWise is an end-to-end, machine-learning-assisted college decision-support platform designed for Indian engineering aspirants. It replaces static league tables with personalized multi-criteria recommendations and continuous salary forecasting. The platform integrates supervised regression with Multi-Criteria Decision Analysis (MCDA) across 1,634 verified institutions."
        ),
        (
            "2. What problem does it solve?",
            "It solves the mismatch between monolithic college rankings and individual student needs. Commercial portals typically rank institutions on a single linear scale and highlight extreme outlier packages, ignoring family budget, location, and academic trade-offs. CollegeWise empowers students to express personal priorities and receive transparent, explainable recommendations."
        ),
        (
            "3. Why did you build it?",
            "Over 2.5 million students clear engineering entrance examinations annually in India, yet available counseling tools rely on commercial incentives or opaque formulas. I built CollegeWise to democratize data-driven, authentic educational decision-making using official statutory disclosures. The platform guarantees zero-fabrication data hygiene and complete explainability for every recommendation."
        ),
        (
            "4. What is the ML problem?",
            "The machine learning problem is formulated as supervised continuous regression to predict institutional median graduate starting compensation (in Lakhs Per Annum). The model learns non-linear relationships between an institution's scale, infrastructure, student-faculty ratio, location, and fees to forecast expected placement compensation."
        ),
        (
            "5. Why regression?",
            "Institutional median compensation varies continuously across a wide range (₹0.90 LPA to ₹35.00 LPA). Discretizing compensation into arbitrary classes (such as 'High', 'Medium', and 'Low') discards critical metric ordering, distance information, and creates artificial boundary thresholds. Continuous regression directly estimates expected compensation alongside data coverage uncertainty bounds."
        ),
        (
            "6. What is the target variable?",
            "The target variable is median_package_lpa, representing the audited median annual starting salary of graduating students in Lakhs Per Annum. We intentionally selected the median rather than the average because campus placement distributions are heavily skewed by extreme international outliers. Median salary reflects the authentic 50th percentile outcome for a graduating student."
        ),
        (
            "7. What features did you use?",
            "We used 9 predictors capturing institutional fundamentals: total_approved_intake, student_faculty_ratio, is_government, tuition_fee_annual, academics_score, infrastructure_score, faculty_quality_score, student_life_score, and location_score. All features reflect statutory disclosures and pre-college institutional capabilities. Target-derived variables like placement rates and average packages were strictly excluded to eliminate leakage."
        ),
        (
            "8. Why did you choose these models?",
            "We selected 7 regression architectures to span the full spectrum from naive baseline to regularized linear models and non-linear ensembles: DummyRegressor, Linear Regression (OLS), Ridge Regression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, and XGBRegressor. Comparing simple and complex models ensured that performance gains were empirically justified over baseline central tendency."
        ),
        (
            "9. Why Gradient Boosting?",
            "The Gradient Boosting model achieved an R² of 0.6418 on the 17-institution holdout test set, with a Test MAE of 3.22 LPA and 5x5 Repeated CV MAE of 2.95 LPA. Its sequential residual-fitting mechanism effectively captures complex non-linear thresholds and multi-attribute interactions without requiring artificial polynomial feature transformations. It reduced Dummy Baseline error by 44.8%."
        ),
        (
            "10. What is R²?",
            "R² (Coefficient of Determination) is a standard regression evaluation metric that measures the proportion of target variance explained by model predictions relative to a naive mean baseline (R² = 1 - [sum(y - y_hat)² / sum(y - y_bar)²]). An R² of 1.0 represents perfect prediction, while 0.0 equals predicting the mean. The Gradient Boosting model achieved an R² of 0.6418 on the 17-institution holdout test set; however, because the holdout sample is small, we evaluate it alongside MAE (3.22 LPA) and RMSE (3.88 LPA) rather than treating R² as an over-generalized claim."
        ),
        (
            "11. What is MAE?",
            "Mean Absolute Error (MAE) computes the average absolute magnitude of prediction errors: MAE = (1/N) * sum(|y_i - y_hat_i|). Unlike squared error metrics, MAE treats all errors linearly and is directly interpretable in the original units of the target variable. Our champion model achieved a holdout MAE of 3.22 LPA (~₹3.22 Lakhs)."
        ),
        (
            "12. What is RMSE?",
            "Root Mean Squared Error (RMSE) is the square root of the average squared differences between predicted and actual values: RMSE = sqrt((1/N) * sum((y_i - y_hat_i)²)). Because errors are squared before averaging, RMSE penalizes large forecasting mistakes more severely than MAE. Our champion model achieved a holdout RMSE of 3.88 LPA."
        ),
        (
            "13. Why use multiple metrics?",
            "Using R², MAE, and RMSE together provides a comprehensive, balanced evaluation of model quality. R² evaluates overall variance explanation relative to baseline, MAE provides an intuitive, robust measure of typical error in LPA, and RMSE highlights the presence of large outlier prediction errors. Relying on a single metric can obscure localized performance failures."
        ),
        (
            "14. What is cross-validation?",
            "Cross-validation is a resampling technique that partitions data into complementary subsets to train on one partition and test on the other across multiple rounds. It evaluates how reliably a predictive model generalizes to an independent dataset rather than overfitting to a single arbitrary train/test split. It is especially critical for small-to-medium datasets where evaluation variance can be high."
        ),
        (
            "15. Why repeated K-fold?",
            "Standard K-fold cross-validation depends on a single random partitioning of data, which can produce noisy performance estimates on smaller datasets (N=109). Repeated K-fold (5 folds repeated 5 times, yielding 25 total folds) averages evaluation across multiple distinct partitionings to produce stable mean metrics and empirical confidence spreads (MAE = 2.95 ± 0.65 LPA). This guards against reporting split-dependent lucky results."
        ),
        (
            "16. What is data leakage?",
            "Data leakage occurs when information from outside the training dataset—particularly from the target variable or the holdout test set—is inadvertently introduced into the model training pipeline. Leakage creates artificially inflated training and validation metrics that collapse when the model is deployed on truly novel, real-world data."
        ),
        (
            "17. How did you prevent leakage?",
            "We enforced three structural controls: First, target-derived variables (placement rates, average packages, and engineered placement scores) were strictly excluded from predictor matrices. Second, all feature transformers (imputers, scalers) were fitted strictly on training folds within cross-validation and pipeline routines, never globally on the entire dataset. Third, an untouched holdout test set (N=17) was isolated prior to model selection and evaluated only once."
        ),
        (
            "18. What is permutation importance?",
            "Permutation feature importance measures the increase in model prediction error after randomly shuffling the values of a specific feature while keeping all other features unchanged. A substantial increase in error indicates that the model relies heavily on that feature for its predictions. Shuffling breaks the relationship between the feature and the target in a model-agnostic manner."
        ),
        (
            "19. Does feature importance imply causation?",
            "No, feature importance strictly reflects predictive association within the trained model and dataset, not real-world causation. For instance, while location_score exhibits high permutation importance, relocating a college campus would not causally guarantee higher graduate salaries. Confounding variables such as historical institutional funding, industry ecosystem maturity, and student selectivity drive the observed correlations."
        ),
        (
            "20. Why is the ML dataset smaller than the recommendation database?",
            "The verified supervised-learning cohort is smaller because only institutions with sufficiently complete and verified placement disclosures were eligible for model training. The recommendation database contains 1,634 institutions with verified statutory AICTE details (intake, programs, location, fees), while 109 institutions had verified multi-year NIRF placement records. Rather than fabricating fake placement figures, we trained supervised models strictly on verified records and used the model to estimate packages with clear data-coverage-based uncertainty bounds."
        ),
        (
            "21. How does MCDA work?",
            "Multi-Criteria Decision Analysis (MCDA) is a decision science framework for evaluating and ranking alternative choices across conflicting qualitative and quantitative criteria. It normalizes distinct dimension metrics onto a common scale and aggregates them based on user-defined priority weights. This allows students to transparently evaluate trade-offs rather than relying on arbitrary external rankings."
        ),
        (
            "22. Why SAW?",
            "We chose Simple Additive Weighting (SAW) because it is linear, mathematically transparent, and easily explainable to students: Match Score_i = sum(w_j * s_ij). Students can immediately understand how adjustments to their priority weights affect the score (Delta S_i = sum(Delta w_j * s_ij)). More complex non-linear methods (such as TOPSIS or AHP) introduce rank-reversal paradoxes and distance metrics that obscure intuitive user feedback."
        ),
        (
            "23. How do the six weights work?",
            "The six weights correspond to Placement, Campus, Academics, Affordability, Location, and Student Life, constrained to sum strictly to 100 points (sum(w_j) = 100). Each weight represents the percentage importance a student assigns to that institutional pillar. The recommendation engine multiplies each normalized pillar score (s_ij in [0, 100]) by the student's corresponding weight to compute a composite match score between 0 and 100."
        ),
        (
            "24. How do you prevent all sliders from becoming 100?",
            "We implemented a dynamic budget slider mechanism in the application session state that enforces a strict 100-point total sum. When a student increases one slider, the remaining unallocated points are proportionally decremented across the other sliders, or capped so the sum cannot exceed 100. This forces realistic decision-making by requiring students to make explicit trade-offs."
        ),
        (
            "25. How does sensitivity analysis work?",
            "Sensitivity analysis allows students to define two different priority profiles—Scenario A and Scenario B—and directly observe institutional rank movements on an interactive Plotly slope chart. By keeping institutional attributes constant and varying only user preference weights, students can see which colleges remain robust across profiles versus which ones are highly sensitive to priority changes."
        ),
        (
            "26. How do you handle missing data?",
            "We enforce a strict zero-fabrication policy where missing values are never replaced with invented data or treated as zero. When statutory data is missing for a non-essential dimension, the scoring engine dynamically re-normalizes the match score across available verified pillars. Furthermore, each college is assigned an explicit Data Confidence badge (High, Medium, Baseline) with coverage uncertainty bounds (±2.0 or ±3.0 LPA). These are estimated uncertainty bounds based on data-coverage tiers and are not statistically calibrated confidence intervals."
        ),
        (
            "27. What are the limitations?",
            "The verified supervised-learning cohort is smaller because only institutions with sufficiently complete and verified placement disclosures were eligible for model training. Institutional salary predictions carry estimated uncertainty bounds (±2.0 to ±3.0 LPA) based on data-coverage tiers, not statistically calibrated confidence intervals. Feature importance indicates predictive correlation rather than causation. Recommendations reflect user-selected priority weights under MCDA. Finally, the secondary classification experiment on 17 holdout institutions achieved perfect separation due to boundary clustering and cannot be generalized as overall system accuracy."
        )
    ]

    for q, a in qa_list:
        qa_flowable = [
            Paragraph(f"<b>{q}</b>", h3_style),
            Paragraph(a, body_style),
            Spacer(1, 4)
        ]
        story.append(KeepTogether(qa_flowable))

    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 10: QUALITY ASSURANCE & TESTING SUITE
    # =========================================================================
    story.append(Paragraph("10. Automated Testing & Verification Suite", h1_style))
    story.append(Paragraph(
        "CollegeWise includes an end-to-end automated test suite executed via <b>pytest</b> (20/20 passing tests). "
        "The suite rigorously asserts mathematical and architectural invariants:",
        body_style
    ))

    tests_summary = [
        "<b>Data Boundary Sanitization (<code>test_sanitize_numeric_boundaries</code>):</b> Asserts placement rates "
        "$\\in [0, 100]\\%$, fees $\\in [₹1,000, ₹25,00,000]$, and median packages $\\in [0.5, 60.0]$ LPA.",
        "<b>Zero Target Leakage (<code>test_no_target_leakage</code>):</b> Asserts that target columns are strictly absent "
        "from predictor feature vectors and verifies that no input feature has a Pearson correlation $> 0.90$ with the target.",
        "<b>Inference Uncertainty Bounds (<code>test_predict_uncertainty_and_data_sufficiency</code>):</b> Asserts that verified "
        "colleges report 0.0 LPA uncertainty and that unverified colleges receive data-coverage uncertainty bounds (±2.0 or ±3.0 LPA).",
        "<b>MCDA Weight Conservation (<code>test_normalize_weights</code>):</b> Asserts that normalized weights sum strictly to 100.0 "
        "regardless of input perturbations.",
        "<b>Sensitivity Slope Stability (<code>test_recommendation_sensitivity_analysis</code>):</b> Verifies that rank displacement "
        "vectors reflect only preference shifts and preserve institutional factor invariance."
    ]
    for ts in tests_summary:
        story.append(Paragraph(f"• {ts}", bullet_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=SECONDARY, spaceBefore=4, spaceAfter=8))
    story.append(Paragraph(
        "<b>Conclusion:</b> CollegeWise delivers an interview-defensible, scientifically grounded decision-support system. "
        "By grounding recommendations in authentic statutory disclosures, decoupling continuous ML package forecasting from Multi-Criteria "
        "Decision Analysis, enforcing a dynamic 100-point budget slider, and backing all operations with an automated test suite, "
        "the project embodies production-grade software and ML engineering excellence.",
        body_style
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built at: {PDF_OUTPUT_PATH}")

if __name__ == "__main__":
    build_pdf()
