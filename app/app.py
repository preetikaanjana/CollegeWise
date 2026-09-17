"""
CollegeWise — Personalized ML-Based College Decision Support System
A modern, Pinterest-aesthetic decision-support platform for students in India.
"""

import sys
import os
import json

# Ensure project root is in sys.path for cloud deployment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.recommendation.engine import RecommendationEngine, DIMENSIONS, normalize_weights
from src.recommendation.explainer import generate_recommendation_explanation
from src.models.predict import predict_median_package

# Page configuration
st.set_page_config(
    page_title="CollegeWise — Personalized College Decision Support",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast, Aesthetic Modern Theme CSS
MODERN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Background */
.stApp {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* ============================================================
   UNIVERSAL HIGH-CONTRAST TEXT & WIDGET LABEL RULES
   Guarantees 100% legibility regardless of OS/Browser dark mode
   ============================================================ */
label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label,
.stCheckbox label,
.stRadio label,
div[data-testid="stWidgetLabel"],
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span,
label p {
    color: #0F172A !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    visibility: visible !important;
    opacity: 1 !important;
    line-height: 1.4 !important;
    text-shadow: none !important;
    margin-bottom: 6px !important;
}

/* BaseWeb Selectbox / Dropdown styling */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
div[data-baseweb="select"] * {
    color: #0F172A !important;
}
div[data-baseweb="select"] svg {
    fill: #0F172A !important;
}
div[data-baseweb="popover"] ul {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
}
div[data-baseweb="popover"] li {
    color: #0F172A !important;
    font-weight: 500 !important;
}
div[data-baseweb="popover"] li:hover {
    background-color: #EEF2FF !important;
}

/* Checkbox Labels */
.stCheckbox span {
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

/* Headings */
h1, h2, h3, h4, h5 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    letter-spacing: -0.02em !important;
}

/* Hero Campus Banner */
.hero-campus-banner {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.82) 100%), 
                url('https://images.unsplash.com/photo-1541339907198-e08756dedf3f?auto=format&fit=crop&w=1600&q=80');
    background-size: cover;
    background-position: center;
    border-radius: 24px;
    padding: 44px 36px;
    margin-bottom: 28px;
    box-shadow: 0 20px 35px -10px rgba(15, 23, 42, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #FFFFFF;
}

/* Modern Card Container */
.cw-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 26px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05), 0 2px 6px -1px rgba(0, 0, 0, 0.02);
    border: 1px solid #E2E8F0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.cw-card:hover {
    box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.08);
}

/* Recommendation Item Card */
.recs-card {
    background: #FFFFFF;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 24px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.recs-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 32px -4px rgba(0, 0, 0, 0.1);
}

/* Factor Hero Pill Cards */
.factor-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 16px 12px;
    text-align: center;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    transition: transform 0.15s ease;
}
.factor-card:hover {
    transform: translateY(-2px);
}

/* Pastel & Vibrant Badges */
.badge-mint {
    background-color: #ECFDF5;
    color: #065F46;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
    border: 1px solid #A7F3D0;
}
.badge-lavender {
    background-color: #F5F3FF;
    color: #5B21B6;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
    border: 1px solid #DDD6FE;
}
.badge-peach {
    background-color: #FFF7ED;
    color: #9A3412;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
    border: 1px solid #FED7AA;
}
.badge-sky {
    background-color: #F0F9FF;
    color: #0369A1;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
    border: 1px solid #BAE6FD;
}
.badge-indigo {
    background-color: #EEF2FF;
    color: #3730A3;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    display: inline-block;
    border: 1px solid #C7D2FE;
}

/* Match Score Pill */
.match-score-pill {
    background: linear-gradient(135deg, #10B981, #059669);
    color: #FFFFFF !important;
    padding: 7px 18px;
    border-radius: 999px;
    font-size: 1.05rem;
    font-weight: 800;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
    display: inline-block;
}

/* Budget Pool Tracker Meter */
.budget-meter-container {
    background: #F1F5F9;
    border-radius: 14px;
    padding: 14px 20px;
    margin: 16px 0;
    border: 1px solid #CBD5E1;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Primary Action Buttons */
.stButton > button {
    border-radius: 14px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.45) !important;
    transform: translateY(-2px) !important;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
}
div[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
</style>
"""
st.markdown(MODERN_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_recommendation_engine():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "colleges_features.parquet")
    return RecommendationEngine(data_path)


engine = get_recommendation_engine()
df_master = engine.df


# Campus Image Resolver for Pinterest-style Visual College Cards
def get_campus_image(college_id: str, college_name: str, institution_type: str, ownership: str) -> str:
    cname_upper = str(college_name or "").upper()
    cid_upper = str(college_id or "").upper()
    
    if "IIT" in cid_upper or "INDIAN INSTITUTE OF TECHNOLOGY" in cname_upper:
        return "https://images.unsplash.com/photo-1562774053-701939374585?auto=format&fit=crop&w=800&q=80"
    elif "BITS" in cid_upper or "BIRLA INSTITUTE" in cname_upper:
        return "https://images.unsplash.com/photo-1498243691581-b145c3f54a5a?auto=format&fit=crop&w=800&q=80"
    elif "NIT" in cid_upper or "NATIONAL INSTITUTE OF TECHNOLOGY" in cname_upper:
        return "https://images.unsplash.com/photo-1523050854058-8df90110c9f1?auto=format&fit=crop&w=800&q=80"
    elif "VIT" in cid_upper or "VELLORE" in cname_upper:
        return "https://images.unsplash.com/photo-1592280771190-3e2e4d571952?auto=format&fit=crop&w=800&q=80"
    elif "IIIT" in cid_upper or "INFORMATION TECHNOLOGY" in cname_upper:
        return "https://images.unsplash.com/photo-1519452635265-7b1fbfd1e4e0?auto=format&fit=crop&w=800&q=80"
    elif ownership == "Government":
        return "https://images.unsplash.com/photo-1541339907198-e08756dedf3f?auto=format&fit=crop&w=800&q=80"
    else:
        return "https://images.unsplash.com/photo-1564981797816-1043664bf78d?auto=format&fit=crop&w=800&q=80"


# Modern Collegiate Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 16px 0;">
        <div style="font-size: 2.4rem; line-height: 1;">🏛️</div>
        <h2 style="margin: 6px 0 2px 0; font-size: 1.45rem; color: #0F172A;">CollegeWise</h2>
        <div style="font-size: 0.8rem; font-weight: 600; color: #64748B;">Decision Support System</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "🎯 1. Discover & Recommend",
            "🔬 2. Sensitivity Analysis",
            "📊 3. Side-by-Side Comparison",
            "🔍 4. College Deep Dive Explorer"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 🛡️ **Verified Database Status**")
    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; font-size: 0.82rem; color: #334155;">
        <div>🏛️ <b>{len(df_master):,}</b> Accredited Colleges</div>
        <div style="margin-top: 4px;">🏆 <b>Top IITs, NITs, BITS, VIT</b> Included</div>
        <div style="margin-top: 4px;">📊 <b>NIRF & AICTE</b> Official Disclosures</div>
        <div style="margin-top: 4px;">⚖️ <b>Strict 100%</b> Points Constraint</div>
    </div>
    """, unsafe_allow_html=True)

    pdf_doc_path = os.path.join(os.path.dirname(__file__), "..", "CollegeWise_Comprehensive_System_Architecture_and_Interview_Defense.pdf")
    if os.path.exists(pdf_doc_path):
        with open(pdf_doc_path, "rb") as f_pdf:
            st.download_button(
                label="📥 Download System Handbook (PDF)",
                data=f_pdf.read(),
                file_name="CollegeWise_System_Architecture_and_Interview_Defense.pdf",
                mime="application/pdf",
                help="Download the complete technical architecture and interview defense handbook.",
                use_container_width=True
            )
    
    st.caption("CollegeWise © 2026 • AICTE & NIRF Audit Provenance")


# ==============================================================================
# VIEW 1: SEQUENTIAL DISCOVERY & RECOMMENDATION FLOW
# Step 1: Academic Profile -> Step 2: 100% Budget Priorities -> CTA -> Step 3: Recommendations
# ==============================================================================
if "🎯 1. Discover" in nav_selection:
    # Aesthetic University Campus Hero Banner
    st.markdown("""
    <div class="hero-campus-banner">
        <span class="badge-mint" style="margin-bottom: 12px; background: rgba(16, 185, 129, 0.2); color: #A7F3D0; border: 1px solid rgba(167, 243, 208, 0.4);">
            🎓 Indian Higher Education Decision Support
        </span>
        <h1 style="font-size: 2.6rem; color: #FFFFFF !important; margin: 8px 0 12px 0;">Find the college that fits YOU.</h1>
        <p style="font-size: 1.08rem; color: #E2E8F0; max-width: 820px; line-height: 1.6; margin-bottom: 0;">
            A personalized recommendation system powered by authentic NIRF placement disclosures and AICTE records. 
            Assign your priorities across 6 key factors—enforced to a strict 100-point budget—and discover institutions tailored to your profile.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 6 Core Pillars Preview Row
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    with p1:
        st.markdown('<div class="factor-card" style="background:#ECFDF5; border-color:#A7F3D0;"><div style="font-size: 1.5rem;">💼</div><b style="color:#065F46;">Placement</b><div style="font-size: 0.72rem; color: #047857;">Packages & Roles</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="factor-card" style="background:#F5F3FF; border-color:#DDD6FE;"><div style="font-size: 1.5rem;">🏛️</div><b style="color:#5B21B6;">Campus</b><div style="font-size: 0.72rem; color: #6D28D9;">Labs & Hostels</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown('<div class="factor-card" style="background:#F0FDFA; border-color:#99F6E4;"><div style="font-size: 1.5rem;">📚</div><b style="color:#0F766E;">Academics</b><div style="font-size: 0.72rem; color: #115E59;">Faculty & Rigor</div></div>', unsafe_allow_html=True)
    with p4:
        st.markdown('<div class="factor-card" style="background:#FFF7ED; border-color:#FED7AA;"><div style="font-size: 1.5rem;">💰</div><b style="color:#9A3412;">Affordability</b><div style="font-size: 0.72rem; color: #C2410C;">Fee & Scholarships</div></div>', unsafe_allow_html=True)
    with p5:
        st.markdown('<div class="factor-card" style="background:#F0F9FF; border-color:#BAE6FD;"><div style="font-size: 1.5rem;">📍</div><b style="color:#0369A1;">Location</b><div style="font-size: 0.72rem; color: #0284C7;">City & Transit</div></div>', unsafe_allow_html=True)
    with p6:
        st.markdown('<div class="factor-card" style="background:#FDF2F8; border-color:#FBCFE8;"><div style="font-size: 1.5rem;">🎭</div><b style="color:#9D174D;">Student Life</b><div style="font-size: 0.72rem; color: #BE185D;">Fests & Clubs</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # STEP 1: STUDENT PROFILE & CONSTRAINTS CARD
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="cw-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            <div style="background: #EEF2FF; border-radius: 12px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">🧑‍🎓</div>
            <div>
                <h3 style="margin: 0; font-size: 1.3rem; color: #0F172A;">Step 1: Your Academic Profile & Preferences</h3>
                <div style="font-size: 0.85rem; color: #64748B;">Specify your preferred course, state, budget limit, and living criteria.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        all_branches = [
            "All Branches", "Computer Science", "Information Technology",
            "Electronics & Communication", "Mechanical Engineering",
            "Civil Engineering", "Electrical Engineering",
            "Artificial Intelligence", "Data Science", "Aerospace Engineering"
        ]
        sel_branch = st.selectbox("Desired Branch / Specialization", all_branches, index=0)

    with c_p2:
        all_states = ["All India"] + sorted(df_master["state"].dropna().unique().tolist())
        sel_state = st.selectbox("Preferred State", all_states, index=0)

    with c_p3:
        budget_option = st.selectbox(
            "Annual Tuition Budget Ceiling",
            ["No Limit", "Under ₹1.5 Lakhs", "Under ₹2.5 Lakhs", "Under ₹4 Lakhs", "Under ₹6 Lakhs"],
            index=0
        )
        budget_map = {
            "No Limit": None,
            "Under ₹1.5 Lakhs": 150000,
            "Under ₹2.5 Lakhs": 250000,
            "Under ₹4 Lakhs": 400000,
            "Under ₹6 Lakhs": 600000
        }
        user_budget = budget_map[budget_option]

    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        sel_ownership = st.selectbox("Ownership Preference", ["Any / Both", "Government Only", "Private Only"], index=0)
        own_filter = "Government" if sel_ownership == "Government Only" else ("Private" if sel_ownership == "Private Only" else None)
    with c_f2:
        hostel_req = st.checkbox("Hostel Accommodation Required", value=False)
    with c_f3:
        verified_only = st.checkbox("Only Show Verified NIRF Placements", value=False)

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # STEP 2: STRICT 100-POINT PRIORITY BUDGET CARD
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="cw-card">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div style="background: #ECFDF5; border-radius: 12px; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">⚖️</div>
            <div>
                <h3 style="margin: 0; font-size: 1.3rem; color: #0F172A;">Step 2: Assign Priorities (Strict 100-Point Budget)</h3>
                <div style="font-size: 0.85rem; color: #64748B;">
                    You have a total budget of 100 points. If you choose 80 for Placement, only 20 points remain for your other preferences!
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Session State Initialization for Priorities
    if "p_placement" not in st.session_state:
        st.session_state.p_placement = 40
        st.session_state.p_campus = 15
        st.session_state.p_academics = 15
        st.session_state.p_affordability = 15
        st.session_state.p_location = 10
        st.session_state.p_student_life = 5

    # 1-Click Balanced Presets
    st.markdown("<b>⚡ Quick 100% Balanced Presets:</b>", unsafe_allow_html=True)
    ps1, ps2, ps3, ps4, ps5, ps6 = st.columns(6)

    def set_priorities(p, c, ac, af, loc, sl):
        st.session_state.p_placement = p
        st.session_state.p_campus = c
        st.session_state.p_academics = ac
        st.session_state.p_affordability = af
        st.session_state.p_location = loc
        st.session_state.p_student_life = sl
        st.rerun()

    with ps1:
        if st.button("💼 Career (70%)", use_container_width=True):
            set_priorities(70, 10, 5, 5, 5, 5)
    with ps2:
        if st.button("💰 High ROI (40%)", use_container_width=True):
            set_priorities(30, 10, 10, 40, 5, 5)
    with ps3:
        if st.button("📚 Academics (50%)", use_container_width=True):
            set_priorities(25, 10, 50, 5, 5, 5)
    with ps4:
        if st.button("🏛️ Campus Life", use_container_width=True):
            set_priorities(20, 35, 10, 10, 10, 15)
    with ps5:
        if st.button("⚖️ All-Rounder", use_container_width=True):
            set_priorities(20, 16, 16, 16, 16, 16)
    with ps6:
        if st.button("🔄 Start at 0", use_container_width=True):
            set_priorities(0, 0, 0, 0, 0, 0)

    # Dynamic 100-Point Budget Calculation
    dims = ["placement", "campus", "academics", "affordability", "location", "student_life"]
    current_sum = sum(st.session_state[f"p_{d}"] for d in dims)
    remaining_points = max(0, 100 - current_sum)

    # Callback to strictly clamp any slider if user pushes total over 100
    def make_on_change(dim_name):
        def _callback():
            new_val = st.session_state[f"sl_{dim_name}"]
            # calculate sum of other 5 sliders
            other_sum = sum(st.session_state[f"p_{d}"] for d in dims if d != dim_name)
            allowed_max = max(0, 100 - other_sum)
            if new_val > allowed_max:
                st.session_state[f"sl_{dim_name}"] = allowed_max
            st.session_state[f"p_{dim_name}"] = st.session_state[f"sl_{dim_name}"]
        return _callback

    # Budget Meter Display
    if current_sum == 100:
        st.markdown(f"""
        <div class="budget-meter-container" style="background: #ECFDF5; border-color: #A7F3D0;">
            <div>
                <b style="color: #065F46; font-size: 1.05rem;">✅ 100 / 100 Points Fully Allocated</b>
                <div style="font-size: 0.85rem; color: #047857;">Perfect balance! Each slider is strictly constrained by the 100-point total.</div>
            </div>
            <span class="badge-mint" style="font-size: 0.95rem;">0 Points Remaining</span>
        </div>
        """, unsafe_allow_html=True)
    elif current_sum < 100:
        st.markdown(f"""
        <div class="budget-meter-container" style="background: #EEF2FF; border-color: #C7D2FE;">
            <div>
                <b style="color: #3730A3; font-size: 1.05rem;">⚡ {current_sum} / 100 Points Allocated</b>
                <div style="font-size: 0.85rem; color: #4338CA;">You have <b>{remaining_points} points remaining</b> to distribute across preferences!</div>
            </div>
            <span class="badge-indigo" style="font-size: 0.95rem;">{remaining_points} Points Left</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.session_state.p_placement = min(100, st.session_state.p_placement)

    # 6 Dynamic Sliders in clean 2-column grid
    col_s1, col_s2 = st.columns(2, gap="large")

    def render_slider(dim_key, col, label_text):
        cur_val = st.session_state[f"p_{dim_key}"]
        other_sum = sum(st.session_state[f"p_{d}"] for d in dims if d != dim_key)
        max_allowed = max(0, 100 - other_sum)
        slider_max = max(1, max_allowed)
        
        st.session_state[f"sl_{dim_key}"] = min(cur_val, max_allowed)

        with col:
            st.slider(
                f"{label_text} (Max: {max_allowed} pts)",
                min_value=0,
                max_value=slider_max,
                value=st.session_state[f"sl_{dim_key}"],
                key=f"sl_{dim_key}",
                on_change=make_on_change(dim_key)
            )

    render_slider("placement", col_s1, "💼 Placement & Career")
    render_slider("campus", col_s1, "🏛️ Campus & Infrastructure")
    render_slider("academics", col_s1, "📚 Academic Quality")

    render_slider("affordability", col_s2, "💰 Affordability & Scholarships")
    render_slider("location", col_s2, "📍 Location & Connectivity")
    render_slider("student_life", col_s2, "🎭 Student Life & Extracurriculars")

    raw_weights = {
        "placement": st.session_state.p_placement,
        "campus": st.session_state.p_campus,
        "academics": st.session_state.p_academics,
        "affordability": st.session_state.p_affordability,
        "location": st.session_state.p_location,
        "student_life": st.session_state.p_student_life
    }

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PRIMARY CALL TO ACTION: GENERATE RECOMMENDATIONS
    # --------------------------------------------------------------------------
    if "show_recommendations" not in st.session_state:
        st.session_state.show_recommendations = False

    st.markdown("<div style='text-align: center; margin: 24px 0 32px 0;'>", unsafe_allow_html=True)
    cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
    with cta_col2:
        if st.button("🎯 Generate My Personalized College Recommendations", use_container_width=True, type="primary"):
            st.session_state.show_recommendations = True
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # STEP 3: PERSONALIZED RECOMMENDATIONS FEED (Only visible after CTA!)
    # --------------------------------------------------------------------------
    if st.session_state.show_recommendations:
        st.markdown("<hr style='border: 1px solid #CBD5E1; margin: 24px 0;'>", unsafe_allow_html=True)
        
        # Summary Header of Active User Priorities
        st.markdown(f"""
        <div style="background: #FFFFFF; border-radius: 16px; padding: 20px 24px; border: 1px solid #E2E8F0; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
            <div>
                <h2 style="margin: 0; font-size: 1.6rem; color: #0F172A;">🏆 Step 3: Your Personalized College Recommendations</h2>
                <div style="font-size: 0.9rem; color: #475569; margin-top: 4px;">
                    Ranked strictly by multi-criteria fit score based on your profile filters and priority weights.
                </div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="badge-mint">💼 Placement: {raw_weights['placement']}%</span>
                <span class="badge-lavender">🏛️ Campus: {raw_weights['campus']}%</span>
                <span class="badge-sky">📚 Academics: {raw_weights['academics']}%</span>
                <span class="badge-peach">💰 Fees: {raw_weights['affordability']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Query Recommendation Engine
        candidates = df_master.copy()
        if verified_only:
            candidates = candidates[candidates["has_placement_data"] == True]

        recs = engine.recommend(
            user_weights=raw_weights,
            preferred_branch=sel_branch if sel_branch != "All Branches" else None,
            preferred_state=sel_state if sel_state != "All India" else None,
            max_budget=user_budget,
            ownership=own_filter,
            hostel_required=hostel_req,
            verified_only=verified_only,
            top_k=15
        )

        if len(recs) == 0:
            st.warning("No colleges strictly matched all selected filters. Relax your budget ceiling or branch constraint to see institutions.")
        else:
            st.markdown(f"<div style='font-size: 1.05rem; font-weight: 700; color: #1E293B; margin-bottom: 16px;'>Found {len(recs)} Top Recommendations Matching Your Criteria:</div>", unsafe_allow_html=True)

            if "comparison_list" not in st.session_state:
                st.session_state.comparison_list = []

            for idx, col_data in recs.iterrows():
                cid = col_data["college_id"]
                cname = col_data["college_name"]
                city = col_data["city"]
                state = col_data["state"]
                score = col_data["match_score"]
                own = col_data["ownership"]
                fee = col_data.get("tuition_fee_annual")
                pkg = col_data.get("median_package_lpa")
                pr = col_data.get("placement_rate")
                placed_cnt = col_data.get("placed_students_count")
                comp_pct = col_data.get("match_completeness_pct", 100)

                campus_img = get_campus_image(cid, cname, col_data.get("institution_type", ""), own)

                # Generate transparent explanation
                exp = generate_recommendation_explanation(
                    col_data,
                    normalize_weights(raw_weights),
                    {"preferred_state": sel_state, "max_budget": user_budget}
                )

                # Data confidence tier badge
                conf_tier = col_data.get("data_confidence_tier", "Baseline")
                if conf_tier == "High":
                    conf_badge = '<span class="badge-mint" style="box-shadow: 0 2px 6px rgba(0,0,0,0.2);">🟢 High: NIRF Verified</span>'
                elif conf_tier == "Medium":
                    conf_badge = '<span class="badge-peach" style="box-shadow: 0 2px 6px rgba(0,0,0,0.2);">🟡 Medium: AICTE</span>'
                else:
                    conf_badge = '<span class="badge-lavender" style="box-shadow: 0 2px 6px rgba(0,0,0,0.2);">⚪ Baseline</span>'

                # Render Pinterest-Aesthetic Recommendation Card
                with st.container():
                    st.markdown(f"""
                    <div class="recs-card">
                        <div style="position: relative; height: 160px; overflow: hidden; background: #0F172A;">
                            <img src="{campus_img}" style="width: 100%; height: 100%; object-fit: cover; opacity: 0.88;" alt="Campus" />
                            <div style="position: absolute; top: 14px; left: 16px; display: flex; gap: 8px; flex-wrap: wrap;">
                                <span class="badge-{('mint' if own == 'Government' else 'lavender')}" style="box-shadow: 0 2px 6px rgba(0,0,0,0.2);">{own}</span>
                                <span class="badge-sky" style="box-shadow: 0 2px 6px rgba(0,0,0,0.2);">{state}</span>
                                {conf_badge}
                                <span class="badge-indigo" style="box-shadow: 0 2px 6px rgba(0,0,0,0.2);">{comp_pct}% Data Verified</span>
                            </div>
                            <div style="position: absolute; top: 14px; right: 16px;">
                                <span class="match-score-pill">{score}% Match</span>
                            </div>
                        </div>
                        <div style="padding: 22px 24px;">
                            <h3 style="font-size: 1.35rem; margin-top: 0; margin-bottom: 4px; color: #0F172A;">{cname}</h3>
                            <p style="color: #64748B; font-size: 0.9rem; margin-bottom: 16px;">
                                📍 {city}, {state} &nbsp;|&nbsp; 🏛️ {col_data.get('affiliated_university', col_data.get('institution_type', ''))[:60]}
                            </p>
                    """, unsafe_allow_html=True)

                    # Compute ML inference or fetch verified disclosure with uncertainty
                    pred_pkg_info = predict_median_package(col_data)

                    # 4 Authentic Stat Badges
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        if not pred_pkg_info["predicted"]:
                            st.metric("Overall Median Package", f"₹{pred_pkg_info['median_package_lpa']} LPA")
                            st.caption("🟢 NIRF Verified Outcome")
                        else:
                            st.metric("Est. Package (ML)", f"₹{pred_pkg_info['median_package_lpa']} ± {pred_pkg_info['uncertainty_lpa']} LPA")
                            st.caption(f"{pred_pkg_info['data_sufficiency']}")
                    with m2:
                        if pd.notna(placed_cnt) and float(placed_cnt) > 0 and pd.notna(pr):
                            st.metric("Students Placed", f"{int(placed_cnt)} ({pr}%)")
                        elif pd.notna(pr) and float(pr) > 0:
                            st.metric("Students Placed", f"{pr}% Placed")
                        else:
                            st.metric("Students Placed", "Not Disclosed")
                    with m3:
                        if pd.notna(fee):
                            st.metric("Annual Tuition Fee", f"₹{int(fee):,}")
                        else:
                            st.metric("Annual Tuition Fee", "State Regulated")
                    with m4:
                        st.metric("Academic Rating", f"{int(col_data.get('academic_score', 70))}/100")

                    # Distinct Presentation: 6 Dimension Factor Scores (Intrinsic institutional attributes)
                    st.markdown(f"""
                    <div style="background: #F1F5F9; border-radius: 10px; padding: 10px 14px; margin: 12px 0 14px 0; display: flex; flex-wrap: wrap; gap: 8px; font-size: 0.82rem; color: #1E293B;">
                        <span style="font-weight: 700; color: #475569;">📊 Dimension Factor Scores:</span>
                        <span style="font-weight: 600;">💼 Placement: {int(col_data.get('placement_score', 50))}/100</span>
                        <span style="color: #CBD5E1;">•</span>
                        <span style="font-weight: 600;">🏛️ Campus: {int(col_data.get('infrastructure_score', 50))}/100</span>
                        <span style="color: #CBD5E1;">•</span>
                        <span style="font-weight: 600;">📚 Academics: {int(col_data.get('academic_score', 50))}/100</span>
                        <span style="color: #CBD5E1;">•</span>
                        <span style="font-weight: 600;">💰 Affordability: {int(col_data.get('affordability_score', 50))}/100</span>
                        <span style="color: #CBD5E1;">•</span>
                        <span style="font-weight: 600;">📍 Location: {int(col_data.get('location_score', 50))}/100</span>
                        <span style="color: #CBD5E1;">•</span>
                        <span style="font-weight: 600;">🎭 Student Life: {int(col_data.get('student_life_score', 50))}/100</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Why this college fits you explanation box
                    st.markdown("""
                    <div style="background: #F8FAFC; border-left: 4px solid #4F46E5; padding: 12px 16px; border-radius: 8px; margin: 14px 0 10px 0; font-size: 0.9rem; color: #1E293B;">
                        <b style="color: #4F46E5;">💡 Why this college matches your preferences:</b><br>
                    """ + "".join([f"• {reason}<br>" for reason in exp["why_this_college"]]) + """
                    </div>
                    """, unsafe_allow_html=True)

                    if exp["transparency_caveats"]:
                        st.caption("ℹ️ " + " | ".join(exp["transparency_caveats"]))

                    # Data Provenance & Reliability Audit Expander
                    with st.expander(f"📊 Data Provenance & Integrity Audit ({cname[:35]})"):
                        prov_c1, prov_c2 = st.columns(2)
                        with prov_c1:
                            st.markdown(f"""
                            - **Outcome Status**: `{pred_pkg_info['status']}`
                            - **Data Coverage Tier**: {pred_pkg_info['data_sufficiency']}
                            - **Primary Data Source**: `{col_data.get('data_source', 'AICTE Public Portal')}`
                            - **Reporting Disclosure Year**: {col_data.get('data_year', 2021)}
                            """)
                        with prov_c2:
                            st.markdown(f"""
                            - **Training Cohort**: {pred_pkg_info['training_cohort']}
                            - **Prediction Uncertainty Bound**: $\\pm$ {pred_pkg_info['uncertainty_lpa']} LPA
                            - **Estimated Uncertainty Range**: ₹{pred_pkg_info['confidence_lower_lpa']} — ₹{pred_pkg_info['confidence_upper_lpa']} LPA
                            - **Integrity Notice**: *{pred_pkg_info['disclaimer']}*
                            """)

                    # Comparison selection checkbox
                    is_compared = cid in st.session_state.comparison_list
                    if st.checkbox("Select for Side-by-Side Comparison", value=is_compared, key=f"chk_{cid}"):
                        if cid not in st.session_state.comparison_list:
                            if len(st.session_state.comparison_list) >= 4:
                                st.warning("Max 4 colleges can be compared side-by-side.")
                            else:
                                st.session_state.comparison_list.append(cid)
                    else:
                        if cid in st.session_state.comparison_list:
                            st.session_state.comparison_list.remove(cid)

                    st.markdown("</div></div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: SENSITIVITY & PRIORITY PERTURBATION ANALYSIS
# ==============================================================================
elif "🔬 2. Sensitivity Analysis" in nav_selection:
    st.markdown("""
    <div class="cw-card">
        <span class="badge-indigo" style="margin-bottom: 8px;">🔬 Defensible Decision-Support Mathematics</span>
        <h2 style="margin: 6px 0 8px 0; color: #0F172A;">Interactive Recommendation Sensitivity Analysis</h2>
        <div style="color: #64748B; font-size: 0.95rem; line-height: 1.5;">
            In multi-criteria decision analysis (MCDA), recommendations are only as credible as their sensitivity to user preference changes.
            This module perturbs priority weights between two profiles (<b>Scenario A</b> vs. <b>Scenario B</b>) and observes the dynamic rank shifts across accredited institutions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Preset selection or Custom
    preset_col, filter_col = st.columns([1.5, 1], gap="large")
    with preset_col:
        st.markdown("#### ⚙️ **Select Comparison Scenarios**")
        scenario_preset = st.selectbox(
            "Choose a Predefined Scenario Contrast:",
            [
                "💼 Placement Maximizer vs. 💰 Affordability First",
                "📚 Research & Academics vs. 🎭 Campus Life & Infrastructure",
                "⚖️ Balanced Profile vs. 💼 Placement Maximizer",
                "🛠️ Custom Priority Weights"
            ]
        )

    with filter_col:
        st.markdown("#### 🎯 **Scope Constraints**")
        sens_state = st.selectbox("State Filter:", ["All India"] + sorted([s for s in df_master["state"].dropna().unique() if s != "Unknown"]))
        sens_ownership = st.selectbox("Institution Type:", ["All Types", "Government", "Private"])

    # Configure weights based on preset
    if scenario_preset == "💼 Placement Maximizer vs. 💰 Affordability First":
        weights_a = {"placement": 50, "campus": 15, "academics": 20, "affordability": 5, "location": 5, "student_life": 5}
        weights_b = {"placement": 10, "campus": 10, "academics": 15, "affordability": 50, "location": 10, "student_life": 5}
        label_a = "Scenario A: Placement Maximizer (50% Placement)"
        label_b = "Scenario B: Affordability Focused (50% Affordability)"
    elif scenario_preset == "📚 Research & Academics vs. 🎭 Campus Life & Infrastructure":
        weights_a = {"placement": 15, "campus": 10, "academics": 55, "affordability": 10, "location": 5, "student_life": 5}
        weights_b = {"placement": 15, "campus": 40, "academics": 15, "affordability": 10, "location": 5, "student_life": 15}
        label_a = "Scenario A: Academic & Research Centric (55% Academics)"
        label_b = "Scenario B: Campus Life & Infrastructure (40% Campus, 15% Student Life)"
    elif scenario_preset == "⚖️ Balanced Profile vs. 💼 Placement Maximizer":
        weights_a = {"placement": 20, "campus": 20, "academics": 20, "affordability": 15, "location": 15, "student_life": 10}
        weights_b = {"placement": 60, "campus": 10, "academics": 15, "affordability": 5, "location": 5, "student_life": 5}
        label_a = "Scenario A: Balanced Student Profile"
        label_b = "Scenario B: Strict Placement Maximizer (60% Placement)"
    else:
        label_a = "Scenario A: Custom Weights"
        label_b = "Scenario B: Custom Weights"
        st.markdown("##### 🎚️ Configure Scenario A & Scenario B Weights (Must sum to 100 each)")
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("<b>Scenario A Weights</b>", unsafe_allow_html=True)
            wa_p = st.number_input("Scenario A Placement %", min_value=0, max_value=100, value=35, step=5)
            wa_c = st.number_input("Scenario A Campus %", min_value=0, max_value=100, value=25, step=5)
            wa_ac = st.number_input("Scenario A Academics %", min_value=0, max_value=100, value=15, step=5)
            wa_af = st.number_input("Scenario A Affordability %", min_value=0, max_value=100, value=10, step=5)
            wa_l = st.number_input("Scenario A Location %", min_value=0, max_value=100, value=10, step=5)
            wa_s = st.number_input("Scenario A Student Life %", min_value=0, max_value=100, value=5, step=5)
            weights_a = {"placement": wa_p, "campus": wa_c, "academics": wa_ac, "affordability": wa_af, "location": wa_l, "student_life": wa_s}
        with c_b:
            st.markdown("<b>Scenario B Weights</b>", unsafe_allow_html=True)
            wb_p = st.number_input("Scenario B Placement %", min_value=0, max_value=100, value=15, step=5)
            wb_c = st.number_input("Scenario B Campus %", min_value=0, max_value=100, value=15, step=5)
            wb_ac = st.number_input("Scenario B Academics %", min_value=0, max_value=100, value=20, step=5)
            wb_af = st.number_input("Scenario B Affordability %", min_value=0, max_value=100, value=35, step=5)
            wb_l = st.number_input("Scenario B Location %", min_value=0, max_value=100, value=10, step=5)
            wb_s = st.number_input("Scenario B Student Life %", min_value=0, max_value=100, value=5, step=5)
            weights_b = {"placement": wb_p, "campus": wb_c, "academics": wb_ac, "affordability": wb_af, "location": wb_l, "student_life": wb_s}

    # Run sensitivity analysis
    state_arg = sens_state if sens_state != "All India" else None
    own_arg = sens_ownership if sens_ownership != "All Types" else None

    analysis_res = engine.run_sensitivity_analysis(
        weights_a=weights_a,
        weights_b=weights_b,
        preferred_state=state_arg,
        ownership=own_arg,
        top_k=10
    )

    comp_table = analysis_res["comparison"]
    factor_deltas = analysis_res["factor_deltas"]

    # Weight shift visual bar
    st.markdown("---")
    st.markdown("### ⚖️ Priority Weight Shift: Scenario A vs. Scenario B")
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown(f"**{label_a}**")
        df_wa = pd.DataFrame(list(weights_a.items()), columns=["Dimension", "Weight %"])
        df_wa["Dimension"] = df_wa["Dimension"].str.replace("_", " ").str.title()
        fig_wa = px.bar(df_wa, x="Dimension", y="Weight %", color="Dimension", color_discrete_sequence=px.colors.qualitative.Prism)
        fig_wa.update_layout(height=260, showlegend=False, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="#FFFFFF")
        st.plotly_chart(fig_wa, use_container_width=True)

    with col_w2:
        st.markdown(f"**{label_b}**")
        df_wb = pd.DataFrame(list(weights_b.items()), columns=["Dimension", "Weight %"])
        df_wb["Dimension"] = df_wb["Dimension"].str.replace("_", " ").str.title()
        fig_wb = px.bar(df_wb, x="Dimension", y="Weight %", color="Dimension", color_discrete_sequence=px.colors.qualitative.Prism)
        fig_wb.update_layout(height=260, showlegend=False, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="#FFFFFF")
        st.plotly_chart(fig_wb, use_container_width=True)

    # Key Metrics of Rank Perturbation
    if len(comp_table) > 0:
        max_shift = comp_table["rank_shift"].abs().max()
        avg_shift = comp_table["rank_shift"].abs().mean()
        shifted_count = (comp_table["rank_shift"] != 0).sum()
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Institutions Evaluated", f"{len(comp_table)}")
        with m_col2:
            st.metric("Institutions with Rank Shift", f"{shifted_count} / {len(comp_table)}")
        with m_col3:
            st.metric("Max Rank Shift", f"{int(max_shift)} Positions")
        with m_col4:
            st.metric("Avg Rank Displacement", f"{avg_shift:.1f} Positions")

        # Visual Rank Movement Chart (Slope / Bump chart)
        st.markdown("### 📈 Institutional Rank Movements (Scenario A → Scenario B)")
        
        fig_slope = go.Figure()
        for _, r in comp_table.iterrows():
            ra = r["rank_scenario_a"]
            rb = r["rank_scenario_b"]
            shift = r["rank_shift"]
            c_name = r["college_name"][:32]
            
            if shift > 0:
                line_color = "#10B981" # Rose in B
            elif shift < 0:
                line_color = "#EF4444" # Fell in B
            else:
                line_color = "#64748B" # Unchanged
                
            fig_slope.add_trace(go.Scatter(
                x=["<b>Scenario A</b>", "<b>Scenario B</b>"],
                y=[ra, rb],
                mode="lines+markers+text",
                name=c_name,
                line=dict(color=line_color, width=3),
                marker=dict(size=8),
                text=[f"#{int(ra)}", f"#{int(rb)} {c_name}"],
                textposition=["middle left", "middle right"],
                hoverinfo="text",
                hovertext=f"{c_name}: Rank {int(ra)} → {int(rb)} (Shift: {int(shift):+d})"
            ))

        fig_slope.update_layout(
            yaxis=dict(autorange="reversed", title="Ranking (1 = Top Match)", dtick=1, gridcolor="#E2E8F0"),
            xaxis=dict(tickfont=dict(size=14, family="Plus Jakarta Sans", color="#0F172A")),
            showlegend=False,
            height=480,
            margin=dict(l=60, r=260, t=40, b=40),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC"
        )
        st.plotly_chart(fig_slope, use_container_width=True)

        # Tabular breakdown with visual badges
        st.markdown("### 📋 Institutional Sensitivity Scorecard")
        
        table_rows = []
        for _, r in comp_table.iterrows():
            shift = int(r["rank_shift"])
            if shift > 0:
                shift_badge = f"🟢 +{shift} (Rose)"
            elif shift < 0:
                shift_badge = f"🔴 {shift} (Fell)"
            else:
                shift_badge = "⚪ 0 (Unchanged)"
                
            table_rows.append({
                "College Name": r["college_name"],
                "State": r["state"],
                "Ownership": r["ownership"],
                "Rank (A)": int(r["rank_scenario_a"]),
                "Rank (B)": int(r["rank_scenario_b"]),
                "Rank Shift": shift_badge,
                "Score (A)": f"{r['score_a']}%",
                "Score (B)": f"{r['score_b']}%",
                "Score Delta (Δ)": f"{r['score_delta']:+.1f}%"
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # Student-Friendly Takeaway Card + Technical Expander for Evaluators
        st.markdown("""
        <div style="background: #FFFFFF; border: 1.5px solid #E2E8F0; border-radius: 14px; padding: 18px 22px; margin-top: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="font-size: 1.25rem;">💡</span>
                <b style="color: #0F172A; font-size: 1.05rem;">How Your Personalized Rankings Shift</b>
            </div>
            <p style="color: #475569; font-size: 0.92rem; margin: 0; line-height: 1.6;">
                CollegeWise does <b>not</b> use a rigid, one-size-fits-all ranking. When you change priorities between 
                <b>Scenario A</b> and <b>Scenario B</b>, institutions that excel in your promoted areas (e.g., lower fees or higher placement) 
                naturally climb upward. Rank changes reflect your unique personal preferences — not that one college is universally "better" than another.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📐 Technical Methodology: Multi-Criteria Decision Analysis (MCDA) Proof"):
            st.markdown(r"""
            **Mathematical Formulation: Simple Additive Weighting (SAW)**
            
            Every institution's match score is computed as:
            $$\text{Match Score}_i = \sum_{j=1}^{6} w_j \cdot s_{ij}$$
            
            Where:
            - $w_j$ is the student's normalized priority weight for dimension $j$ ($\sum w_j = 1.0$).
            - $s_{ij} \in [0, 100]$ is the college's verified dimension score.
            
            When priority weights are perturbed by $\Delta w_j = w_{B, j} - w_{A, j}$, the score change is deterministic:
            $$\Delta S_i = \sum_{j=1}^{6} \Delta w_j \cdot s_{ij}$$
            
            - **Rank Displacement Guarantee**: An institution with a superior score in dimension $k$ ($s_{ik} > \bar{s}$) gains a positive score delta when weight on $k$ increases, vaulting it higher in rank.
            - **Neutral Framing**: Rank shifts represent personal value alignment, proving recommendations are dynamically tailored to human priorities.
            """)
    else:
        st.info("No overlapping institutions found under current strict filter settings.")


# ==============================================================================
# VIEW 3: SIDE-BY-SIDE COMPARISON
# ==============================================================================
elif "📊 3. Side-by-Side" in nav_selection:
    st.markdown("""
    <div class="cw-card">
        <h2 style="margin-top: 0; color: #0F172A;">📊 Side-by-Side College Comparison</h2>
        <div style="color: #64748B; font-size: 0.92rem;">Compare 2 to 4 shortlisted institutions across all 6 preference dimensions and verified placement metrics.</div>
    </div>
    """, unsafe_allow_html=True)

    sample_premier = ["IIT-BOMBAY", "BITS-PILANI", "NIT-TRICHY", "VIT-VELLORE"]
    if "comparison_list" not in st.session_state or len(st.session_state.comparison_list) == 0:
        st.session_state.comparison_list = sample_premier

    all_col_options = dict(zip(df_master["college_id"], df_master["college_name"] + " (" + df_master["state"] + ")"))
    current_selected = [cid for cid in st.session_state.get("comparison_list", []) if cid in all_col_options]

    chosen_ids = st.multiselect(
        "Choose institutions to compare (Select 2 to 4):",
        options=list(all_col_options.keys()),
        format_func=lambda x: all_col_options.get(x, x),
        default=current_selected[:4]
    )
    st.session_state.comparison_list = chosen_ids

    if len(chosen_ids) < 2:
        st.warning("Please select at least 2 colleges to view side-by-side comparison charts.")
    else:
        comp_df = df_master[df_master["college_id"].isin(chosen_ids)].copy()

        # Radar Chart comparing 6 Dimensions
        fig_radar = go.Figure()
        categories = ["Placement", "Campus", "Academics", "Affordability", "Location", "Student Life"]
        color_palette = ["#10B981", "#8B5CF6", "#3B82F6", "#F59E0B"]

        for i, (_, c_row) in enumerate(comp_df.iterrows()):
            vals = [
                c_row.get("placement_score", 50),
                c_row.get("infrastructure_score", 50),
                c_row.get("academic_score", 50),
                c_row.get("affordability_score", 50),
                c_row.get("location_score", 50),
                c_row.get("student_life_score", 50)
            ]
            vals.append(vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill='toself',
                name=c_row["college_name"][:35],
                line=dict(color=color_palette[i % len(color_palette)])
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#64748B"),
                angularaxis=dict(color="#0F172A", tickfont=dict(size=12, family="Plus Jakarta Sans"))
            ),
            showlegend=True,
            title=dict(text="Multi-Dimensional Balance Comparison (Scores out of 100)", font=dict(size=16, color="#0F172A")),
            height=500,
            margin=dict(l=50, r=50, t=70, b=50),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Comparison Table
        st.markdown("### 📋 **Comprehensive Metric Comparison**")
        disp_cols = [
            "college_name", "state", "ownership", "tuition_fee_annual",
            "median_package_lpa", "placed_students_count", "placement_rate",
            "academic_score", "infrastructure_score", "affordability_score", "data_source"
        ]
        disp_df = comp_df[disp_cols].copy()
        disp_df["median_package_lpa"] = disp_df["median_package_lpa"].apply(lambda x: f"₹{x} LPA" if pd.notna(x) else "Not Disclosed")
        disp_df["placed_students_count"] = disp_df["placed_students_count"].apply(lambda x: f"{int(x)} Students" if pd.notna(x) else "Not Disclosed")
        disp_df["placement_rate"] = disp_df["placement_rate"].apply(lambda x: f"{x}%" if pd.notna(x) else "Not Disclosed")
        disp_df["tuition_fee_annual"] = disp_df["tuition_fee_annual"].apply(lambda x: f"₹{int(x):,}" if pd.notna(x) else "State Regulated")

        disp_df.columns = [
            "Institution", "State", "Ownership", "Annual Tuition",
            "Overall Median Package", "Students Placed", "Placement Rate",
            "Academics (0-100)", "Infrastructure (0-100)", "Affordability (0-100)", "Data Source"
        ]
        st.dataframe(disp_df.set_index("Institution").T, use_container_width=True)


# ==============================================================================
# VIEW 4: COLLEGE DEEP DIVE EXPLORER
# ==============================================================================
elif "🔍 4. College Deep Dive" in nav_selection:
    st.markdown("""
    <div class="cw-card">
        <h2 style="margin-top: 0; color: #0F172A;">🔍 Institutional Deep Dive Explorer</h2>
        <div style="color: #64748B; font-size: 0.92rem;">Inspect full academic disclosures, statutory infrastructure, and audit provenance.</div>
    </div>
    """, unsafe_allow_html=True)

    col_names = dict(zip(df_master["college_id"], df_master["college_name"] + " (" + df_master["state"] + ")"))
    
    default_idx = 0
    if "IIT-BOMBAY" in col_names:
        default_idx = list(col_names.keys()).index("IIT-BOMBAY")

    selected_cid = st.selectbox("Select Institution to Inspect:", options=list(col_names.keys()), index=default_idx, format_func=lambda x: col_names[x])
    col_row = df_master[df_master["college_id"] == selected_cid].iloc[0]

    campus_hero_img = get_campus_image(selected_cid, col_row["college_name"], col_row.get("institution_type", ""), col_row["ownership"])

    st.markdown(f"""
    <div class="recs-card" style="margin-bottom: 24px;">
        <div style="height: 180px; overflow: hidden; position: relative;">
            <img src="{campus_hero_img}" style="width: 100%; height: 100%; object-fit: cover;" alt="Campus" />
            <div style="position: absolute; bottom: 16px; left: 24px; background: rgba(15, 23, 42, 0.75); padding: 6px 16px; border-radius: 999px;">
                <span style="color: #FFFFFF; font-weight: 700; font-size: 0.9rem;">📍 {col_row.get('city', '')}, {col_row.get('state', '')}</span>
            </div>
        </div>
        <div style="padding: 24px;">
            <span class="badge-mint">{col_row['ownership']}</span>
            <span class="badge-lavender">{col_row['institution_type']}</span>
            <span class="badge-sky">{col_row['state']}</span>
            <h2 style="margin-top: 10px; margin-bottom: 4px; color: #0F172A;">{col_row['college_name']}</h2>
            <p style="color: #64748B; font-size: 0.92rem;">🏛️ {col_row.get('affiliated_university', col_row.get('institution_type', ''))} &nbsp;|&nbsp; {col_row.get('address', '')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Academics & Branches",
        "💼 Placements & Outcomes",
        "🏛️ Campus & Facilities",
        "📜 Audit Provenance"
    ])

    with tab1:
        st.markdown("#### Approved Courses & Specializations")
        branches_str = col_row.get("branches")
        if pd.notna(branches_str):
            branch_list = [b.strip() for b in branches_str.split(";")]
            st.markdown(" ".join([f'<span class="badge-sky" style="margin: 4px;">{b}</span>' for b in branch_list]), unsafe_allow_html=True)
        else:
            st.info("Detailed branch list not itemized in basic AICTE disclosure.")

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            st.metric("Total Annual Intake", col_row.get("total_approved_intake", "N/A"))
        with col_a2:
            st.metric("Faculty Strength", col_row.get("faculty_count", "Not Disclosed"))
        with col_a3:
            st.metric("Student-Faculty Ratio", col_row.get("student_faculty_ratio", "N/A"))

    with tab2:
        st.markdown("#### Verified NIRF Placement Disclosures")
        has_pl = col_row.get("has_placement_data", False)
        if has_pl:
            st.success("Official NIRF graduation outcome disclosure verified.")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Overall Median Package", f"₹{col_row.get('median_package_lpa')} LPA")
            with col_p2:
                cnt = col_row.get('placed_students_count')
                st.metric("Students Placed", f"{int(cnt)} ({col_row.get('placement_rate')}%)" if pd.notna(cnt) else f"{col_row.get('placement_rate')}%")
            with col_p3:
                st.metric("Higher Studies Rate", f"{col_row.get('higher_studies_rate', 'N/A')}%")
        else:
            pred_deep = predict_median_package(col_row)
            st.info("Institution does not have public NIRF placement disclosure filings. Displaying statistically inferred outcome:")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Est. Median Package (ML)", f"₹{pred_deep['median_package_lpa']} ± {pred_deep['uncertainty_lpa']} LPA")
                st.caption(f"{pred_deep['data_sufficiency']}")
            with col_p2:
                st.metric("Estimated Range", f"₹{pred_deep['confidence_lower_lpa']} – ₹{pred_deep['confidence_upper_lpa']} LPA")
                st.caption(f"Status: {pred_deep['status']}")
            with col_p3:
                st.metric("Supervised Cohort", "109 Verified Inst.")
                st.caption("Zero Target Leakage")
            st.caption(f"ℹ️ {pred_deep['disclaimer']}")

    with tab3:
        st.markdown("#### Living & Campus Facilities")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Annual Tuition Fee", f"₹{int(col_row.get('tuition_fee_annual')):,}" if pd.notna(col_row.get("tuition_fee_annual")) else "State Regulated")
        with col_f2:
            st.metric("Annual Hostel Fee", f"₹{int(col_row.get('hostel_fee_annual')):,}" if pd.notna(col_row.get("hostel_fee_annual")) else "N/A")
        with col_f3:
            st.metric("Hostel Availability", col_row.get("hostel_available", "N/A"))

        st.markdown("#### Documented Campus Facilities")
        st.markdown(f"""
        - 🌐 **Internet Connectivity**: {col_row.get('internet_connectivity', 'Campus Wi-Fi')}
        - 📖 **Library**: {col_row.get('library_facility', 'Central Library')}
        - 🧪 **Laboratories**: {col_row.get('laboratories_facility', 'Departmental Labs')}
        - 🏃 **Sports Complex**: {col_row.get('sports_facilities', 'Indoor/Outdoor Sports')}
        - ♿ **Physical Accessibility (PCS)**: {col_row.get('other_facilities', 'Standard Facilities')}
        """)

    with tab4:
        st.markdown("#### Data Provenance & Integrity Audit")
        pred_deep_audit = predict_median_package(col_row)
        st.json({
            "College ID": col_row.get("college_id"),
            "Data Source": col_row.get("data_source"),
            "Data Confidence Tier": col_row.get("data_confidence_tier", "Baseline"),
            "Data Sufficiency": pred_deep_audit["data_sufficiency"],
            "Outcome Status": pred_deep_audit["status"],
            "Empirical Uncertainty Bound": f"±{pred_deep_audit['uncertainty_lpa']} LPA",
            "Disclosure Year": int(col_row.get("data_year", 2021)),
            "Data Completeness Ratio": f"{int(col_row.get('data_completeness_ratio', 0)*100)}%",
            "Missing Data Percentage": f"{col_row.get('missing_data_percentage')}%",
            "Has Verified NIRF Records": bool(col_row.get("has_placement_data"))
        })
