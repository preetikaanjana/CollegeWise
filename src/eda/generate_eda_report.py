"""
Automated Exploratory Data Analysis (EDA) Script for CollegeWise.

Generates comprehensive high-resolution analytical visualizations:
1. State-wise college distribution
2. Institution ownership & type breakdown
3. Placement rate distribution & density
4. Median package distribution (LPA)
5. Fee distribution & affordability spectrum
6. Placement Rate vs. Tuition Fee scatter
7. Dimension score correlation heatmap
8. Branch availability across institutions
9. Missing value pattern audit chart
10. Infrastructure (PCS) and academic indicators

Saves figures to `reports/eda_figures/`.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "eda_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_all_eda_plots(df_path: str = r"d:\projects\college\data\processed\colleges_features.parquet"):
    """Run full suite of EDA visualizations."""
    print(f"Loading engineered dataset from {df_path}...")
    df = pd.read_parquet(df_path)

    # 1. State-wise distribution (Top 15)
    plt.figure(figsize=(10, 6))
    top_states = df['state'].value_counts().head(15)
    ax = sns.barplot(x=top_states.values, y=top_states.index, palette="mako")
    plt.title("Top 15 States by Institution Representation in Curated Dataset", fontsize=13, pad=12, fontweight='bold')
    plt.xlabel("Number of Institutions", fontsize=11)
    plt.ylabel("State / Union Territory", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_width())}", (p.get_width() + 2, p.get_y() + p.get_height() / 2),
                    va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_colleges_by_state.png"))
    plt.close()
    print("Saved 01_colleges_by_state.png")

    # 2. Institution Types & Ownership
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    own_counts = df['ownership'].value_counts()
    ax1.pie(own_counts.values, labels=own_counts.index, autopct='%1.1f%%',
            colors=['#A5D6A7', '#CE93D8'], startangle=140, explode=(0.04, 0))
    ax1.set_title("Institutional Ownership Breakdown", fontsize=12, fontweight='bold')

    type_counts = df['institution_type'].value_counts().head(6)
    sns.barplot(x=type_counts.values, y=type_counts.index, ax=ax2, palette="crest")
    ax2.set_title("Top Institution Categories", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Colleges", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_institution_types.png"))
    plt.close()
    print("Saved 02_institution_types.png")

    # 3. Placement Rate Distribution
    plt.figure(figsize=(9, 5))
    pl_data = df['placement_rate'].dropna()
    if len(pl_data) > 0:
        sns.histplot(pl_data, kde=True, bins=25, color="#4DB6AC", edgecolor='white')
        plt.axvline(pl_data.median(), color='#E53935', linestyle='--', label=f'Median: {pl_data.median():.1f}%')
        plt.axvline(pl_data.mean(), color='#1E88E5', linestyle=':', label=f'Mean: {pl_data.mean():.1f}%')
        plt.title("Distribution of Verified Institutional Placement Rates (%)", fontsize=13, fontweight='bold')
        plt.xlabel("Placement Percentage (%)", fontsize=11)
        plt.ylabel("Institution Count", fontsize=11)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_placement_distribution.png"))
    plt.close()
    print("Saved 03_placement_distribution.png")

    # 4. Median Package Distribution (LPA)
    plt.figure(figsize=(9, 5))
    pkg_data = df['median_package_lpa'].dropna()
    if len(pkg_data) > 0:
        sns.histplot(pkg_data, kde=True, bins=25, color="#BA68C8", edgecolor='white')
        plt.axvline(pkg_data.median(), color='#D81B60', linestyle='--', label=f'Median: {pkg_data.median():.2f} LPA')
        plt.axvline(pkg_data.mean(), color='#3949AB', linestyle=':', label=f'Mean: {pkg_data.mean():.2f} LPA')
        plt.title("Distribution of Verified Median Annual Compensation (LPA)", fontsize=13, fontweight='bold')
        plt.xlabel("Median Package (Lakhs INR Per Annum)", fontsize=11)
        plt.ylabel("Institution Count", fontsize=11)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_median_package_distribution.png"))
    plt.close()
    print("Saved 04_median_package_distribution.png")

    # 5. Annual Tuition Fee Spectrum by Ownership
    plt.figure(figsize=(10, 5))
    fee_df = df.dropna(subset=['tuition_fee_annual'])
    if len(fee_df) > 0:
        sns.boxplot(x='ownership', y='tuition_fee_annual', data=fee_df, palette=['#81C784', '#FFB74D'])
        plt.title("Annual Statutory Tuition Fee Distribution by Ownership", fontsize=13, fontweight='bold')
        plt.xlabel("Ownership", fontsize=11)
        plt.ylabel("Annual Tuition Fee (INR)", fontsize=11)
        plt.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_fee_distribution.png"))
    plt.close()
    print("Saved 05_fee_distribution.png")

    # 6. Placement Rate vs Fee (ROI Landscape)
    plt.figure(figsize=(10, 6))
    roi_df = df.dropna(subset=['tuition_fee_annual', 'placement_rate'])
    if len(roi_df) > 0:
        scatter = sns.scatterplot(
            x='tuition_fee_annual',
            y='placement_rate',
            hue='ownership',
            size='median_package_lpa',
            sizes=(30, 250),
            data=roi_df,
            palette={'Government': '#2E7D32', 'Private': '#6A1B9A'},
            alpha=0.75
        )
        plt.title("ROI Landscape: Placement Rate vs. Annual Tuition Fee", fontsize=13, fontweight='bold')
        plt.xlabel("Annual Tuition Fee (INR)", fontsize=11)
        plt.ylabel("Placement Rate (%)", fontsize=11)
        plt.ticklabel_format(style='plain', axis='x')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_placement_vs_fee.png"))
    plt.close()
    print("Saved 06_placement_vs_fee.png")

    # 7. Correlation Heatmap of Normalized Dimension Scores
    plt.figure(figsize=(8, 6))
    score_cols = [
        'placement_score', 'academic_score', 'affordability_score',
        'infrastructure_score', 'location_score', 'student_life_score'
    ]
    corr_matrix = df[score_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="vlag",
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix: The 6 Normalized Preference Dimensions", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_dimension_correlations.png"))
    plt.close()
    print("Saved 07_dimension_correlations.png")

    # 8. Missing Values Audit Chart
    plt.figure(figsize=(10, 6))
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(12)
    ax = sns.barplot(x=missing.values, y=missing.index, palette="rocket")
    plt.title("Transparency Audit: Top Missing Institutional Fields (%)", fontsize=13, fontweight='bold')
    plt.xlabel("Missing Percentage (%)", fontsize=11)
    plt.ylabel("Attribute", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.1f}%", (p.get_width() + 1, p.get_y() + p.get_height() / 2),
                    va='center', fontsize=9)
    plt.xlim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_missing_values_audit.png"))
    plt.close()
    print("Saved 08_missing_values_audit.png")

    print("\nAll 8 EDA figures successfully generated in reports/eda_figures/!")


if __name__ == "__main__":
    generate_all_eda_plots()
