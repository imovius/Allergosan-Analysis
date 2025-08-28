#!/usr/bin/env python3
"""
Generate factor-to-factor correlation matrix for Hannah and Klaus.

This script reads the factor scores from our FAMD analysis and calculates
correlations between the factors themselves.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

def generate_factor_correlation_matrix(factor_scores_path, output_dir, tag):
    """Generate and save factor-to-factor correlation matrix."""
    
    # Read factor scores
    if not os.path.exists(factor_scores_path):
        print(f"Factor scores file not found: {factor_scores_path}")
        return None
    
    factor_scores = pd.read_csv(factor_scores_path)
    
    # Get factor columns (exclude customer_index if present)
    factor_cols = [col for col in factor_scores.columns if col.startswith('Factor_')]
    
    if len(factor_cols) == 0:
        print(f"No factor columns found in {factor_scores_path}")
        return None
    
    print(f"Found {len(factor_cols)} factors: {factor_cols}")
    
    # Calculate correlation matrix between factors
    factor_data = factor_scores[factor_cols]
    correlation_matrix = factor_data.corr()
    
    # Save correlation matrix as CSV
    csv_path = os.path.join(output_dir, f"{tag}_factor_correlation_matrix.csv")
    correlation_matrix.to_csv(csv_path)
    print(f"Saved factor correlation matrix: {csv_path}")
    
    # Create heatmap visualization
    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(correlation_matrix, dtype=bool)
    np.fill_diagonal(mask, True)  # Mask diagonal
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', mask=mask,
                cmap='RdBu_r', center=0, square=True,
                cbar_kws={'label': 'Factor-to-Factor Correlation'})
    
    # Add diagonal values as text (should be 1.0)
    for i in range(len(correlation_matrix)):
        plt.text(i + 0.5, i + 0.5, '1.000',
                ha='center', va='center', color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black'))
    
    plt.title(f'Factor-to-Factor Correlation Matrix\n{tag.replace("_", " ").title()}')
    plt.xlabel('Factors')
    plt.ylabel('Factors')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{tag}_factor_correlation_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved factor correlation heatmap: {plot_path}")
    
    return correlation_matrix

def main():
    """Generate factor correlation matrices for both approaches."""
    
    # Output directory
    output_dir = "FAMD_output"
    cluster_dir = "quiz_only/outputs"
    
    # Check if directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for both approaches
    approaches = [
        ("A_withLTV_final", "Approach A (With LTV)"),
        ("B_noLTV_final", "Approach B (Without LTV)")
    ]
    
    results = {}
    
    for tag, description in approaches:
        print(f"\n=== {description} ===")
        
        # Path to factor scores file
        factor_scores_path = os.path.join(cluster_dir, f"{tag}_factor_scores.csv")
        
        # Generate correlation matrix
        corr_matrix = generate_factor_correlation_matrix(factor_scores_path, output_dir, tag)
        
        if corr_matrix is not None:
            results[tag] = corr_matrix
            
            # Print summary statistics
            print(f"\nFactor Correlation Summary for {description}:")
            print(f"  - Number of factors: {len(corr_matrix)}")
            
            # Get off-diagonal correlations
            off_diag = corr_matrix.values[~np.eye(len(corr_matrix), dtype=bool)]
            print(f"  - Off-diagonal correlation range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")
            print(f"  - Mean absolute off-diagonal correlation: {np.abs(off_diag).mean():.3f}")
            
            # Identify highest correlations
            corr_copy = corr_matrix.copy()
            np.fill_diagonal(corr_copy.values, np.nan)  # Remove diagonal
            
            # Find max absolute correlation
            max_corr_idx = np.unravel_index(np.nanargmax(np.abs(corr_copy.values)), corr_copy.shape)
            max_corr_val = corr_copy.iloc[max_corr_idx]
            factor1 = corr_copy.index[max_corr_idx[0]]
            factor2 = corr_copy.columns[max_corr_idx[1]]
            
            print(f"  - Highest correlation: {factor1} <-> {factor2} = {max_corr_val:.3f}")
    
    # Create summary report
    summary_path = os.path.join(output_dir, "Factor_Correlation_Analysis_Summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Factor-to-Factor Correlation Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis shows how the factors derived from FAMD correlate with each other. ")
        f.write("In ideal factor analysis, factors should be largely uncorrelated (orthogonal), ")
        f.write("indicating they capture independent dimensions of customer behavior.\n\n")
        
        f.write("## Results\n\n")
        for tag, description in approaches:
            if tag in results:
                corr_matrix = results[tag]
                f.write(f"### {description}\n\n")
                f.write(f"- **Factors analyzed**: {len(corr_matrix)}\n")
                
                off_diag = corr_matrix.values[~np.eye(len(corr_matrix), dtype=bool)]
                f.write(f"- **Correlation range**: [{off_diag.min():.3f}, {off_diag.max():.3f}]\n")
                f.write(f"- **Mean absolute correlation**: {np.abs(off_diag).mean():.3f}\n\n")
                
                f.write(f"**Files generated**:\n")
                f.write(f"- `{tag}_factor_correlation_matrix.csv`\n")
                f.write(f"- `{tag}_factor_correlation_heatmap.png`\n\n")
        
        f.write("## Interpretation\n\n")
        f.write("- **Low correlations** (|r| < 0.3): Factors capture independent customer dimensions\n")
        f.write("- **Moderate correlations** (0.3 ≤ |r| < 0.7): Some overlap between factor meanings\n")
        f.write("- **High correlations** (|r| ≥ 0.7): Factors may be measuring similar constructs\n\n")
        f.write("## Business Implications\n\n")
        f.write("Independent factors suggest that customer behavior is genuinely multidimensional ")
        f.write("and cannot be reduced to simpler patterns. This supports personalization strategies ")
        f.write("that account for multiple, independent behavioral dimensions.\n")
    
    print(f"\nSummary report saved: {summary_path}")
    print("\n=== Analysis Complete ===")
    print(f"Files generated in {output_dir}:")
    for tag, _ in approaches:
        print(f"  - {tag}_factor_correlation_matrix.csv")
        print(f"  - {tag}_factor_correlation_heatmap.png")
    print(f"  - Factor_Correlation_Analysis_Summary.md")

if __name__ == "__main__":
    main()
