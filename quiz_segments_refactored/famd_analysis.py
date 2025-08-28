# Factor Analysis of Mixed Data (FAMD) implementation and interpretation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
from prince import FAMD
from scipy.stats import pearsonr, chi2_contingency

try:
    from .config import *
except ImportError:
    from config import *

def generate_relationship_matrix(df_input: pd.DataFrame, variable_names: List[str], tag: str):
    """Generate FAMD-style relationship matrix (like Wikipedia Table 2)."""
    
    # Identify variable types
    quant_vars = []
    qual_vars = []
    
    for var in variable_names:
        if var in df_input.columns:
            if df_input[var].dtype == 'category' or df_input[var].dtype == 'object':
                qual_vars.append(var)
            else:
                quant_vars.append(var)
    
    print(f"  - Quantitative variables: {quant_vars}")
    print(f"  - Qualitative variables: {qual_vars}")
    
    # Create relationship matrix
    all_vars = quant_vars + qual_vars
    relationship_matrix = pd.DataFrame(index=all_vars, columns=all_vars, dtype=float)
    
    # Fill diagonal with appropriate values
    for var in all_vars:
        if var in quant_vars:
            relationship_matrix.loc[var, var] = 1.0  # R² = 1 for quantitative
        else:
            # φ² diagonal = number of categories - 1 (degrees of freedom)
            n_categories = len(df_input[var].cat.categories) if hasattr(df_input[var], 'cat') else df_input[var].nunique()
            phi_squared_max = n_categories - 1
            relationship_matrix.loc[var, var] = phi_squared_max

    # Calculate R² between quantitative variables
    for i, var1 in enumerate(quant_vars):
        for j, var2 in enumerate(quant_vars):
            if i != j:
                try:
                    r, _ = pearsonr(df_input[var1].fillna(0), df_input[var2].fillna(0))
                    relationship_matrix.loc[var1, var2] = r**2
                except Exception:
                    relationship_matrix.loc[var1, var2] = 0.0

    # Calculate φ² between qualitative variables
    for i, var1 in enumerate(qual_vars):
        for j, var2 in enumerate(qual_vars):
            if i != j:
                try:
                    # Create contingency table
                    ct = pd.crosstab(df_input[var1].fillna('missing'), df_input[var2].fillna('missing'))
                    chi2, _, _, _ = chi2_contingency(ct)
                    n = len(df_input)
                    phi_squared = chi2 / n if n > 0 else 0
                    relationship_matrix.loc[var1, var2] = phi_squared
                except Exception:
                    relationship_matrix.loc[var1, var2] = 0.0

    # Calculate η² between quantitative and qualitative variables
    def eta_squared(categorical, continuous):
        """Calculate eta-squared (correlation ratio squared)."""
        try:
            # Remove NaN values
            valid_mask = pd.notna(categorical) & pd.notna(continuous)
            if valid_mask.sum() == 0:
                return 0
            
            cat_clean = categorical[valid_mask]
            cont_clean = continuous[valid_mask]
            
            # Convert categorical to ensure proper grouping
            if hasattr(cat_clean, 'cat'):
                groups = [cont_clean[cat_clean == cat] for cat in cat_clean.cat.categories 
                         if len(cont_clean[cat_clean == cat]) > 0]
            else:
                unique_cats = cat_clean.unique()
                groups = [cont_clean[cat_clean == cat] for cat in unique_cats 
                         if len(cont_clean[cat_clean == cat]) > 0]
            
            if len(groups) < 2:
                return 0
            
            # Overall mean
            overall_mean = cont_clean.mean()
            
            # Between-group sum of squares
            ss_between = sum(len(group) * (group.mean() - overall_mean)**2 for group in groups)
            
            # Total sum of squares  
            ss_total = sum((cont_clean - overall_mean)**2)
            
            if ss_total == 0:
                return 0
            
            return ss_between / ss_total
        except Exception:
            return 0

    # Fill η² values
    for quant_var in quant_vars:
        for qual_var in qual_vars:
            eta2 = eta_squared(df_input[qual_var], df_input[quant_var])
            relationship_matrix.loc[quant_var, qual_var] = eta2
            relationship_matrix.loc[qual_var, quant_var] = eta2

    # Save relationship matrix
    relationship_matrix.round(3).to_csv(os.path.join(FAMD_DIR, f"{tag}_relationship_matrix.csv"))
    
    # Create relationship matrix visualization (diagonal white, off-diagonal colored)
    plt.figure(figsize=(10, 8))
    try:
        # Create a masked version for heatmap coloring
        plot_matrix = relationship_matrix.astype(float).copy()
        
        # Create a mask for diagonal elements
        mask = np.zeros_like(plot_matrix.values, dtype=bool)
        np.fill_diagonal(mask, True)  # Mask diagonal for coloring
        
        # Create heatmap with diagonal masked (white) but still annotated
        sns.heatmap(plot_matrix, annot=True, fmt='.2f', mask=mask,
                   cmap='YlOrRd', cbar_kws={'label': 'Off-Diagonal Relationship Strength'})
        
        # Add diagonal values as text annotations (white background)
        for i in range(len(plot_matrix)):
            plt.text(i + 0.5, i + 0.5, f'{plot_matrix.iloc[i, i]:.1f}', 
                    ha='center', va='center', color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black'))
        
        plt.title(f'{tag}: Variable Relationship Matrix\n(R² for quantitative, φ² for qualitative, η² for mixed)\nDiagonal values (white) show variable self-relationships')
        plt.tight_layout()
        plt.savefig(os.path.join(FAMD_DIR, f"{tag}_relationship_matrix_heatmap.png"), dpi=DPI)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create relationship matrix heatmap for {tag}: {e}")
        plt.close()
    
    return relationship_matrix

def extract_famd_interpretations(famd_model: FAMD, variable_names: List[str], n_components: int, tag: str, df_input: pd.DataFrame, manual_explained_variance=None):
    """Extract comprehensive FAMD interpretation outputs including relationship matrix."""
    
    # 1. Basic variance info
    eigs = np.asarray(famd_model.eigenvalues_, dtype=float)
    if manual_explained_variance is not None:
        explained_var = manual_explained_variance
    else:
        explained_var = eigs / eigs.sum()
    cumulative_var = np.cumsum(explained_var)
    
    # Create explained variance dataframe
    var_df = pd.DataFrame({
        'component': range(1, len(explained_var) + 1),
        'eigenvalue': eigs,
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'retained': [i <= n_components for i in range(1, len(explained_var) + 1)]
    })
    var_df.to_csv(os.path.join(FAMD_DIR, f"{tag}_explained_variance.csv"), index=False)
    
    # 2. Variable contributions (corrected calculation)
    contrib_raw = famd_model.column_contributions_.iloc[:, :n_components].copy()
    contrib_totals = contrib_raw.sum(axis=1)
    contrib_totals_pct = contrib_totals / contrib_totals.sum() * 100.0  # Convert to percentages
    
    # Create properly formatted dataframe
    contrib_df = contrib_raw.copy()
    contrib_df['variable'] = variable_names
    contrib_df['total_contribution_pct'] = contrib_totals_pct
    contrib_df = contrib_df.sort_values('total_contribution_pct', ascending=False)
    contrib_df.to_csv(os.path.join(FAMD_DIR, f"{tag}_variable_contributions.csv"), index=False)
    
    # 3. Factor interpretation summary
    interpretation = []
    for i in range(n_components):
        # Get component column name (0-based indexing for iloc)
        component_col = contrib_df.columns[i] if i < len(contrib_df.columns) - 2 else f'component_{i}'
        
        # Sort by this component's contribution
        if component_col in contrib_df.columns:
            comp_contrib = contrib_df.sort_values(component_col, ascending=False).head(5)
            top_vars = comp_contrib['variable'].tolist()
            top_contribs = comp_contrib[component_col].tolist()
        else:
            # Fallback to total contribution ranking
            comp_contrib = contrib_df.sort_values('total_contribution_pct', ascending=False).head(5)
            top_vars = comp_contrib['variable'].tolist()
            top_contribs = comp_contrib['total_contribution_pct'].tolist()
        
        interp = {
            'component': f'Factor_{i+1}',
            'explained_variance_pct': round(explained_var[i] * 100, 2),
            'cumulative_variance_pct': round(cumulative_var[i] * 100, 2),
            'top_variable_1': top_vars[0] if len(top_vars) > 0 else '',
            'top_contrib_1': round(top_contribs[0], 2) if len(top_contribs) > 0 else 0,
            'top_variable_2': top_vars[1] if len(top_vars) > 1 else '',
            'top_contrib_2': round(top_contribs[1], 2) if len(top_contribs) > 1 else 0,
            'top_variable_3': top_vars[2] if len(top_vars) > 2 else '',
            'top_contrib_3': round(top_contribs[2], 2) if len(top_contribs) > 2 else 0,
        }
        interpretation.append(interp)
    
    interp_df = pd.DataFrame(interpretation)
    interp_df.to_csv(os.path.join(FAMD_DIR, f"{tag}_factor_interpretation.csv"), index=False)
    
    # 4. Generate Relationship Matrix (like Wikipedia Table 2)
    print(f"[{tag}] Generating relationship matrix...")
    relationship_matrix = generate_relationship_matrix(df_input, variable_names, tag)
    
    return var_df, contrib_df, interp_df, relationship_matrix

def generate_famd_interpretation_plots(famd_model: FAMD, variable_names: List[str], df_input: pd.DataFrame, n_components: int, tag: str, manual_explained_variance=None):
    """Generate standard FAMD interpretation plots: correlation circle, variable space, category representation."""

    try:
        # 1. Correlation Circle (Figure 3 equivalent) - Variable loadings
        if hasattr(famd_model, 'column_coordinates_'):
            plt.figure(figsize=(8, 8))
            var_coords = famd_model.column_coordinates_.iloc[:, :min(2, n_components)]
            
            # Plot variables as arrows from origin
            for i, var in enumerate(variable_names):
                if i < len(var_coords):
                    x, y = var_coords.iloc[i, 0], var_coords.iloc[i, 1] if var_coords.shape[1] > 1 else 0
                    plt.arrow(0, 0, x, y, head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.7)
                    plt.text(x*1.1, y*1.1, var, fontsize=9, ha='center', va='center')
            
            # Add unit circle for reference
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--', alpha=0.5)
            plt.gca().add_patch(circle)
            
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            if manual_explained_variance is not None:
                plt.xlabel(f'Dim 1 ({manual_explained_variance[0]*100:.1f}%)')
                plt.ylabel(f'Dim 2 ({manual_explained_variance[1]*100:.1f}%)' if n_components > 1 else 'Dim 2')
            else:
                plt.xlabel(f'Dim 1 ({famd_model.percentage_of_variance_[0]:.1f}%)')
                plt.ylabel(f'Dim 2 ({famd_model.percentage_of_variance_[1]:.1f}%)' if n_components > 1 else 'Dim 2')
            plt.title(f'{tag}: FAMD Correlation Circle\n(Variable loadings on factors)')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(FAMD_DIR, f"{tag}_correlation_circle.png"), dpi=DPI)
            plt.close()
        
        # 2. Variable Quality/Contribution Plot
        if hasattr(famd_model, 'column_coordinates_') and hasattr(famd_model, 'column_contributions_'):
            plt.figure(figsize=(8, 8))
            var_coords = famd_model.column_coordinates_.iloc[:, :min(2, n_components)]
            var_contribs = famd_model.column_contributions_.iloc[:, :min(2, n_components)]
            
            # Plot with contribution-based colors
            for i, var in enumerate(variable_names):
                if i < len(var_coords):
                    x = var_coords.iloc[i, 0]
                    y = var_coords.iloc[i, 1] if var_coords.shape[1] > 1 else 0
                    contrib_total = var_contribs.iloc[i, :].sum()
                    
                    plt.scatter(x, y, s=100, c=contrib_total, cmap='YlOrRd', alpha=0.7, edgecolors='black')
                    plt.text(x*1.05, y*1.05, var, fontsize=9, ha='center', va='center')
            
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            if manual_explained_variance is not None:
                plt.xlabel(f'Dim 1 ({manual_explained_variance[0]*100:.1f}%)')
                plt.ylabel(f'Dim 2 ({manual_explained_variance[1]*100:.1f}%)' if n_components > 1 else 'Dim 2')
            else:
                plt.xlabel(f'Dim 1 ({famd_model.percentage_of_variance_[0]:.1f}%)')
                plt.ylabel(f'Dim 2 ({famd_model.percentage_of_variance_[1]:.1f}%)' if n_components > 1 else 'Dim 2')
            plt.title(f'{tag}: Variable Representation Quality\n(Color = contribution to factors)')
            plt.colorbar(label='Total Contribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FAMD_DIR, f"{tag}_variable_quality.png"), dpi=DPI)
            plt.close()
        
    except Exception as e:
        print(f"Warning: Could not create FAMD interpretation plots for {tag}: {e}")

def fit_famd_model(df_catnum: pd.DataFrame, explained_target: float, n_max: int, seed: int, tag: str) -> Tuple[FAMD, np.ndarray, int, np.ndarray]:
    """
    Enhanced FAMD fit with comprehensive outputs.
    
    Args:
        df_catnum: Dataframe with categorical and numeric variables
        explained_target: Target cumulative variance explained
        n_max: Maximum number of components
        seed: Random seed
        tag: Tag for output files
        
    Returns:
        Tuple of (fitted_model, factor_scores, n_components_kept, explained_variance)
    """
    print(f"[{tag}] Fitting FAMD model...")
    
    famd_full = FAMD(n_components=min(n_max, max(2, df_catnum.shape[1]-1)), random_state=seed)
    famd_full.fit(df_catnum)
    
    eigs = np.asarray(famd_full.eigenvalues_, dtype=float)
    expl = eigs / eigs.sum()
    cum = np.cumsum(expl)
    
    n_keep = int(np.searchsorted(cum, explained_target) + 1)
    n_keep = max(2, min(n_keep, len(expl)))
    
    print(f"[{tag}] Keeping {n_keep} components ({cum[n_keep-1]*100:.1f}% variance)")
    
    # Extract comprehensive interpretations
    var_names = df_catnum.columns.tolist()
    extract_famd_interpretations(famd_full, var_names, n_keep, tag, df_catnum, manual_explained_variance=expl)
    
    # Generate standard FAMD interpretation plots
    generate_famd_interpretation_plots(famd_full, var_names, df_catnum, n_keep, tag, manual_explained_variance=expl)
    
    # Get factor scores
    scores = famd_full.row_coordinates(df_catnum)
    X_all = scores.values if hasattr(scores, "values") else np.asarray(scores)
    X = X_all[:, :n_keep]
    
    # Save factor scores  
    scores_df = pd.DataFrame(X, columns=[f'Factor_{i+1}' for i in range(n_keep)])
    scores_df['customer_index'] = range(len(scores_df))
    scores_df.to_csv(os.path.join(CLUSTER_DIR, f"{tag}_factor_scores.csv"), index=False)
    
    # Save variable coordinates if available
    try:
        if hasattr(famd_full, 'column_coordinates_'):
            var_coords = famd_full.column_coordinates_.iloc[:, :n_keep]
            var_coords.to_csv(os.path.join(FAMD_DIR, f"{tag}_variable_coordinates.csv"))
        else:
            print(f"[{tag}] Variable coordinates not available in this FAMD implementation")
    except Exception as e:
        print(f"[{tag}] Could not save variable coordinates: {e}")
    
    # Generate true loading matrix (signed correlations)
    try:
        print(f"[{tag}] Generating true loading matrix (signed correlations)...")
        
        # Get row coordinates (factor scores) from FAMD
        row_coords = famd_full.row_coordinates(df_catnum)
        
        # Convert categorical variables to numeric for correlation calculation
        df_numeric = df_catnum.copy()
        categorical_cols = df_catnum.select_dtypes(include=['category', 'object']).columns.tolist()
        
        # Convert categorical to numeric codes for correlation calculation
        for col in categorical_cols:
            if col in df_numeric.columns:
                df_numeric[col] = pd.Categorical(df_numeric[col]).codes
        
        # Ensure all columns are numeric
        df_numeric = df_numeric.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix between original variables and factor scores
        combined_data = np.column_stack([df_numeric.values, row_coords.values[:, :n_keep]])
        corr_matrix = np.corrcoef(combined_data.T)
        
        # Extract the cross-correlations (loadings)
        n_vars = df_numeric.shape[1]
        loadings_matrix = corr_matrix[:n_vars, n_vars:]
        
        # Create loading matrix DataFrame
        loading_df = pd.DataFrame(
            loadings_matrix,
            index=df_numeric.columns,
            columns=[f'Factor_{i+1}' for i in range(n_keep)]
        )
        
        # Save true loading matrix
        loading_df.to_csv(os.path.join(FAMD_DIR, f"{tag}_loading_matrix.csv"))
        print(f"[{tag}] Saved true loading matrix with {len(loading_df)} variables")
        
        # Display summary of positive vs negative loadings
        loadings_values = loading_df.values.flatten()
        loadings_values = loadings_values[~np.isnan(loadings_values)]
        n_positive = np.sum(loadings_values > 0)
        n_negative = np.sum(loadings_values < 0)
        print(f"[{tag}] Loading summary: {n_positive} positive, {n_negative} negative correlations")
        print(f"[{tag}] Loading range: [{loadings_values.min():.3f}, {loadings_values.max():.3f}]")
        
    except Exception as e:
        print(f"[{tag}] Could not generate true loading matrix: {e}")
        import traceback
        traceback.print_exc()
    
    return famd_full, X, n_keep, expl

def prune_features_by_contribution(famd_model: FAMD, n_keep: int, famd_vars: List[str], 
                                 frac: float, min_pct: float, tag: str) -> List[str]:
    """
    Prune features based on their contribution to the first n_keep factors.
    
    Args:
        famd_model: Fitted FAMD model
        n_keep: Number of factors to consider
        famd_vars: List of variable names
        frac: Fraction of variables to prune (lowest contributing)
        min_pct: Minimum contribution percentage threshold
        tag: Tag for output files
        
    Returns:
        List of variables to keep after pruning
    """
    print(f"[{tag}] Pruning features by contribution...")
    
    contrib = famd_model.column_contributions_.copy()
    contrib_first = contrib.iloc[:, :n_keep]
    totals = (contrib_first.sum(axis=1))
    totals_pct = totals / totals.sum() * 100.0
    
    rank = pd.DataFrame({
        "variable": totals_pct.index, 
        "total_contrib_pct": totals_pct.values
    }).sort_values("total_contrib_pct", ascending=False)
    
    rank.to_csv(os.path.join(FAMD_DIR, f"{tag}_feature_contributions_ranked.csv"), index=False)
    
    n_prune = int(np.floor(frac * len(rank)))
    to_drop = set(rank.tail(n_prune)["variable"].tolist()) | set(rank[rank["total_contrib_pct"] < min_pct]["variable"].tolist())
    keep_vars = [v for v in famd_vars if v not in to_drop]
    
    pd.DataFrame({"dropped_feature": sorted(list(to_drop))}).to_csv(os.path.join(FAMD_DIR, f"{tag}_pruned_features.csv"), index=False)
    
    print(f"[{tag}] Dropped {len(to_drop)} features, keeping {len(keep_vars)}")
    
    return keep_vars

def apply_dominance_filter(df_input: pd.DataFrame, dominance_thresh: float = DOMINANCE_THRESH) -> pd.DataFrame:
    """
    Filter out variables that are too dominated by a single value.
    
    Args:
        df_input: Input dataframe
        dominance_thresh: Maximum fraction for dominant category
        
    Returns:
        Filtered dataframe
    """
    famd_cat = [c for c in df_input.columns if df_input[c].dtype == object]
    famd_num = [c for c in df_input.columns if c not in famd_cat]
    
    # Keep numeric columns with sufficient variance
    keep_num = [c for c in famd_num if pd.to_numeric(df_input[c], errors="coerce").fillna(0).std(ddof=0) > 1e-8]
    
    # Keep categorical columns without excessive dominance
    keep_cat = []
    for c in famd_cat:
        v = df_input[c].astype(str).fillna("missing")
        vc = v.value_counts(normalize=True, dropna=False)
        if (vc.size >= 2) and (vc.iloc[0] <= dominance_thresh):
            keep_cat.append(c)
    
    famd_keep = keep_cat + keep_num
    df_filtered = df_input[famd_keep].copy()
    
    # Convert categorical columns to proper dtype
    for c in keep_cat:
        df_filtered[c] = df_filtered[c].astype("category")
    
    print(f"Dominance filter: kept {len(keep_cat)} categorical + {len(keep_num)} numeric = {len(famd_keep)} total")
    
    return df_filtered

if __name__ == "__main__":
    # Test the FAMD analysis functions
    try:
        from .data_cleaning import get_modeling_dataframe
    except ImportError:
        from data_cleaning import get_modeling_dataframe
    
    df = get_modeling_dataframe()
    
    # Create a test subset for FAMD
    famd_vars = [c for c in df.columns if c != 'customer_id']
    df_famd = df[famd_vars].copy()
    df_famd = apply_dominance_filter(df_famd)
    
    print(f"\nTest FAMD with {df_famd.shape[0]} rows and {df_famd.shape[1]} features")
    
    # Fit FAMD model
    famd_model, factor_scores, n_components, explained_var = fit_famd_model(
        df_famd, EXPLAINED_TARGET, N_MAX_COMPONENTS, SEED, "test"
    )
    
    print(f"FAMD analysis completed! Kept {n_components} components explaining {np.sum(explained_var[:n_components])*100:.1f}% variance")
