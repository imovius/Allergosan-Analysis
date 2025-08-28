# Main orchestration script - ties together all the analysis components
import os
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from .data_cleaning import get_modeling_dataframe
    from .clustering import cluster_and_profile_enhanced, create_value_comparison
    from .config import *
except ImportError:
    from data_cleaning import get_modeling_dataframe
    from clustering import cluster_and_profile_enhanced, create_value_comparison
    from config import *

def run_comprehensive_analysis():
    """
    Run the complete customer segmentation analysis with both approaches:
    - Pass A: With LTV included in FAMD
    - Pass B: Behavioral only (LTV as overlay)
    """
    
    print("="*60)
    print("COMPREHENSIVE CUSTOMER SEGMENTATION ANALYSIS")
    print("="*60)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {SEED}")
    print()
    
    # Load and prepare data
    print("Step 1: Loading and cleaning data...")
    df_model_base = get_modeling_dataframe()
    print(f"Final dataset: {df_model_base.shape[0]:,} customers, {df_model_base.shape[1]} features")
    print()
    
    # Pass A: WITH LTV in FAMD
    print("Step 2: Pass A - Including LTV in factor analysis")
    print("-" * 50)
    
    passA_cat = BASE_CAT[:]  # same categoricals
    passA_num = BASE_NUM[:] + [c for c in VALUE_COLS if c in df_model_base.columns]  # add value numerics
    
    resA = cluster_and_profile_enhanced(
        tag="A_withLTV", 
        df_base=df_model_base, 
        famd_vars_cat=passA_cat, 
        famd_vars_num=passA_num, 
        include_value_in_famd=True, 
        force_k=FORCE_K_A
    )
    
    # Pass B: WITHOUT LTV in FAMD
    print("\nStep 3: Pass B - Behavioral features only (LTV as overlay)")
    print("-" * 50)
    
    passB_cat = BASE_CAT[:]
    passB_num = BASE_NUM[:]  # exclude value numerics here
    
    resB = cluster_and_profile_enhanced(
        tag="B_noLTV", 
        df_base=df_model_base, 
        famd_vars_cat=passB_cat, 
        famd_vars_num=passB_num, 
        include_value_in_famd=False, 
        force_k=FORCE_K_B
    )
    
    # Step 4: Create cross-approach comparisons
    print("\nStep 4: Creating cross-approach comparisons")
    print("-" * 50)
    
    # Value comparison
    results_dict = {
        "A_withLTV": resA,
        "B_noLTV": resB
    }
    
    create_value_comparison(results_dict)
    
    # Individual value summaries
    def create_value_summary(df_profiles, tag):
        keep = ["cluster", "n_customers", "net_ltv_mean", "net_ltv_median", 
                "avg_order_value_mean", "avg_order_value_median"]
        return df_profiles[[c for c in keep if c in df_profiles.columns]].copy().assign(tag=tag)
    
    valA = create_value_summary(resA["profiles"], "A_withLTV")
    valB = create_value_summary(resB["profiles"], "B_noLTV")
    
    valA.to_csv(os.path.join(CLUSTER_DIR, "A_value_by_cluster.csv"), index=False)
    valB.to_csv(os.path.join(CLUSTER_DIR, "B_value_by_cluster.csv"), index=False)
    
    # Step 5: Generate comprehensive summary report
    print("\nStep 5: Generating comprehensive summary report")
    print("-" * 50)
    
    create_summary_report(resA, resB, df_model_base)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Key Output Directories:")
    print(f"  - FAMD interpretation files: {FAMD_DIR}/")
    print(f"  - Clustering & profiling files: {CLUSTER_DIR}/")
    print()
    print("Key Files for Review:")
    print("  FAMD Analysis:")
    print("    - *_factor_interpretation.csv")
    print("    - *_relationship_matrix.csv")
    print("    - *_loading_matrix.csv")
    print("    - *_correlation_circle.png")
    print()
    print("  Clustering Results:")
    print("    - *_cluster_profiles.csv")
    print("    - *_personas.txt")
    print("    - clustering_value_comparison.csv")
    print("    - COMPREHENSIVE_ANALYSIS_SUMMARY.txt")
    
    return resA, resB

def create_summary_report(resA: dict, resB: dict, df_model: pd.DataFrame):
    """Create comprehensive summary report."""
    
    summary_report = []
    summary_report.append("# Enhanced FAMD Customer Segmentation Analysis Summary")
    summary_report.append("=" * 70)
    summary_report.append("")
    summary_report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    summary_report.append(f"Total Quiz Takers Analyzed: {len(df_model):,}")
    summary_report.append(f"Random Seed: {SEED}")
    summary_report.append(f"Features Used: {len([c for c in df_model.columns if c != 'customer_id'])}")
    summary_report.append("")
    
    summary_report.append("## APPROACH COMPARISON")
    summary_report.append("")
    summary_report.append("### PASS A (With LTV in FAMD)")
    summary_report.append(f"- Optimal Clusters: {resA['chosen_k']}")
    summary_report.append(f"- Silhouette Score: {resA['best_s']:.4f}")
    summary_report.append(f"- FAMD Components Retained: {resA['n_keep']} ({resA['cumvar']*100:.1f}% variance)")
    summary_report.append("- Strategy: Include customer value metrics directly in factor analysis")
    summary_report.append("- Use Case: Value-driven segmentation for premium targeting")
    summary_report.append("")
    
    summary_report.append("### PASS B (Behavioral Only)")
    summary_report.append(f"- Optimal Clusters: {resB['chosen_k']}")
    summary_report.append(f"- Silhouette Score: {resB['best_s']:.4f}")
    summary_report.append(f"- FAMD Components Retained: {resB['n_keep']} ({resB['cumvar']*100:.1f}% variance)")
    summary_report.append("- Strategy: Pure behavioral segmentation with value overlay")
    summary_report.append("- Use Case: Behavioral targeting independent of current value")
    summary_report.append("")
    
    summary_report.append("## KEY FILES GENERATED")
    summary_report.append("")
    summary_report.append("### Factor Analysis Interpretation:")
    summary_report.append("- *_explained_variance.csv - Component eigenvalues and variance explained")
    summary_report.append("- *_variable_contributions.csv - Variable importance in factors")
    summary_report.append("- *_factor_interpretation.csv - Business interpretation of factors")
    summary_report.append("- *_relationship_matrix.csv - Variable relationships (R², φ², η²)")
    summary_report.append("- *_loading_matrix.csv - True factor loadings with signs")
    summary_report.append("- *_correlation_circle.png - Visual factor interpretation")
    summary_report.append("")
    
    summary_report.append("### Clustering Results:")
    summary_report.append("- *_cluster_profiles.csv - Detailed cluster characteristics")
    summary_report.append("- *_personas.txt - Business-friendly cluster descriptions")
    summary_report.append("- *_silhouette.png/.csv - Cluster quality analysis")
    summary_report.append("- *_famd_scatter.png - Clusters in factor space")
    summary_report.append("- clustering_value_comparison.csv - Cross-approach comparison")
    summary_report.append("")
    
    summary_report.append("### Factor Scores:")
    summary_report.append("- *_factor_scores.csv - Customer positions in factor space")
    summary_report.append("")
    
    summary_report.append("## ANALYSIS INSIGHTS")
    summary_report.append("")
    
    # Add cluster size comparison
    summary_report.append("### Cluster Size Distribution:")
    summary_report.append("Pass A (With LTV):")
    for _, row in resA["profiles"].iterrows():
        summary_report.append(f"  Cluster {int(row['cluster'])}: {int(row['n_customers']):,} customers")
    
    summary_report.append("")
    summary_report.append("Pass B (Behavioral Only):")
    for _, row in resB["profiles"].iterrows():
        summary_report.append(f"  Cluster {int(row['cluster'])}: {int(row['n_customers']):,} customers")
    
    summary_report.append("")
    summary_report.append("## RECOMMENDATIONS FOR WORKING SESSION")
    summary_report.append("")
    summary_report.append("1. **Factor Interpretation Review**")
    summary_report.append("   - Compare factor loadings between approaches")
    summary_report.append("   - Identify key behavioral drivers vs. value drivers")
    summary_report.append("   - Review relationship matrices for variable dependencies")
    summary_report.append("")
    
    summary_report.append("2. **Cluster Business Validation**")
    summary_report.append("   - Review persona narratives for business sense")
    summary_report.append("   - Compare cluster value distributions")
    summary_report.append("   - Assess cluster sizes for practical targeting")
    summary_report.append("")
    
    summary_report.append("3. **Strategic Decision**")
    summary_report.append("   - Choose between value-driven vs. behavioral segmentation")
    summary_report.append("   - Consider hybrid approaches based on insights")
    summary_report.append("   - Plan implementation and targeting strategies")
    summary_report.append("")
    
    summary_report.append("## TECHNICAL NOTES")
    summary_report.append(f"- Coverage threshold: {COVERAGE*100:.0f}% (columns with sufficient data)")
    summary_report.append(f"- Dominance threshold: {DOMINANCE_THRESH*100:.1f}% (max category dominance)")
    summary_report.append(f"- Variance target: {EXPLAINED_TARGET*100:.0f}% (cumulative explained variance)")
    summary_report.append(f"- Feature pruning: {PRUNE_FRACTION*100:.0f}% lowest contributors removed")
    summary_report.append("")
    
    # Save report
    with open(os.path.join(CLUSTER_DIR, "COMPREHENSIVE_ANALYSIS_SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_report))
    
    print("Comprehensive summary report created")

if __name__ == "__main__":
    # Set numpy random seed for reproducibility
    np.random.seed(SEED)
    
    # Run the complete analysis
    results_A, results_B = run_comprehensive_analysis()
    
    print(f"\nFinal Results Summary:")
    print(f"Pass A: {results_A['chosen_k']} clusters, silhouette={results_A['best_s']:.3f}")
    print(f"Pass B: {results_B['chosen_k']} clusters, silhouette={results_B['best_s']:.3f}")
