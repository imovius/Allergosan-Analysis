# Enhanced FAMD Analysis with Comprehensive Factor Interpretation
# Original: two_pass_personas_value.py
# Enhanced: Comprehensive FAMD outputs for factor analysis interpretation
# Python 3.9+

import os, re, sys
import numpy as np
import pandas as pd
from typing import Optional, Set, Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from prince import FAMD  # assumes your env's FAMD with eigenvalues_ & column_contributions_

# ======================= CONFIG =======================
CSV_PATH = "raw_data_v3.csv"
FAMD_DIR = "FAMD_output"      # Core FAMD interpretation files
CLUSTER_DIR = "quiz_only/outputs"  # Clustering/profiling files
SEED     = 42

# Ensure output directories exist
os.makedirs(FAMD_DIR, exist_ok=True)
os.makedirs(CLUSTER_DIR, exist_ok=True)

# Data filters
INFLUENCER_ONLY   = False   # True: promo purchasers only; False: all quiz takers
COVERAGE          = 0.60    # keep columns with >=60% non-null (looser to bring back features)
DOMINANCE_THRESH  = 0.995   # allow dominated categoricals to pass (e.g., first_sku_bucket)

# FAMD (both passes)
EXPLAINED_TARGET  = 0.85    # keep ~85% cumulative variance
N_MAX_COMPONENTS  = 20      # upper bound for components to learn
PRUNE_FRACTION    = 0.15    # drop bottom 15% by total contribution (first n_keep comps)
MIN_CONTRIB_PCT   = 0.40    # also drop any <0.4% contribution

# Clustering
K_RANGE           = range(2, 11)  # candidate k values
FORCE_K_A         = None          # force k for Pass A (with LTV), or None
FORCE_K_B         = None          # force k for Pass B (behavior only), or None

DPI = 150
# ======================================================
# Directories already created above
np.random.seed(SEED)

# -------------------------- Helpers --------------------------
def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

SKU_TO_BUCKET: Dict[str, str] = {
    "OBSR":"stress","S-OBSR":"stress",
    "OBAB":"gut_restoration","3MGHR":"gut_restoration",
    "OBHX":"detox","S-OBHX":"detox",
    "OBBA":"immune","OLIM":"immune",
    "OBPO":"metabolism","OLPL":"metabolism","S-OBP":"metabolism",
    "OBPA":"prenatal","S-OBPA":"prenatal",
    "OBCG":"other_ob","OBCA":"other_ob","OBCD":"other_ob",
    "OBSC":"accessory",
    "LBUN":"bundle","IMBUN":"bundle","DYNDUO":"bundle",
}
QUIZ_NORMALIZE: Dict[str, str] = {
    "OMNi-BiOTiC BALANCE":"Balance",
    "OMNi-BiOTiCA® BALANCE":"Balance",
    "OMNi-BiOTiC Stress Release":"Stress Release",
    "OMNi-BiOTiCA® Stress Release":"Stress Release",
    "OMNi-BiOTiC HETOX":"Hetox",
    "OMNi-BiOTiCA® HETOX":"Hetox",
    "Omni-Biotic Power":"Power",
    "OMNi-BiOTiC Panda":"Panda",
    "OMNi-BiOTiC AB 10":"AB 10",
    "OMNi-BiOTiCA® AB 10":"AB 10",
    "Gut Health Reset Program":"Gut Health Reset",
}
QUIZ_LINE_TO_BUCKET: Dict[str, str] = {
    "Stress Release":"stress","Balance":"immune","Hetox":"detox",
    "AB 10":"gut_restoration","Gut Health Reset":"gut_restoration",
    "Panda":"prenatal","Power":"metabolism",
}
INFLUENCER_CODES = {c.lower() for c in {
    "dave20","dave","jessica15","drwillcole","dr.cain15","valeria20",
    "skinny","blonde","blonde20","carly15","tammy15","sweats15"
}}

def normalize_quiz_result(q: Optional[str]) -> Optional[str]:
    if not isinstance(q, str): return None
    qn = q.strip()
    if not qn: return None
    return QUIZ_NORMALIZE.get(qn, qn)

def quiz_to_bucket(quiz_result: Optional[str]) -> Optional[str]:
    if not isinstance(quiz_result, str): return None
    line = normalize_quiz_result(quiz_result)
    return QUIZ_LINE_TO_BUCKET.get(line) if line else None

def parse_first_sku_buckets(first_sku: Optional[str]) -> Set[str]:
    buckets: Set[str] = set()
    if not isinstance(first_sku, str) or not first_sku.strip(): return buckets
    for token in [t.strip() for t in first_sku.split(",")]:
        b = SKU_TO_BUCKET.get(token)
        if b: buckets.add(b)
    return buckets

def collapse_sku_buckets(s: Set[str]) -> str:
    if not s: return "none"
    return sorted(list(s))[0] if len(s) == 1 else "multi"

def loose_match(q_bucket: Optional[str], sku_buckets: Set[str]) -> int:
    return int(q_bucket in sku_buckets) if q_bucket and sku_buckets else 0

def map_code_group(x: Optional[str]) -> str:
    if isinstance(x, str) and x.strip():
        v = x.strip().lower()
        return v if v in INFLUENCER_CODES else "other"
    return "none"

def derive_gender(row: pd.Series) -> str:
    g = row.get("gender", None)
    if isinstance(g, str) and g.strip(): return g.strip().lower()
    im = row.get("is_male", None)
    if pd.notna(im):
        try: return "male" if int(im) == 1 else "female"
        except Exception: pass
    return "missing"

# -------------------------- Load & Prepare --------------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
if "customer_id" not in df.columns:
    raise ValueError("customer_id column is required.")

# Normalize / engineer
df["primary_goal"] = df.get("primary_goal", pd.Series(dtype=object)).replace("-", "missing")
df["bm_pattern"] = df.get("bm_pattern", pd.Series(dtype=object)).replace("unspecified", "missing")
df["quiz_result"] = df.get("quiz_result", pd.Series(dtype=object)).apply(normalize_quiz_result)
df["quiz_bucket"] = df["quiz_result"].apply(quiz_to_bucket)
df["first_sku_buckets"] = df.get("first_sku", pd.Series(dtype=object)).apply(parse_first_sku_buckets)
df["first_sku_bucket"] = df["first_sku_buckets"].apply(collapse_sku_buckets)
df["quiz_reco_match"] = df.apply(lambda r: loose_match(r.get("quiz_bucket"), r.get("first_sku_buckets")), axis=1)
df["acquisition_code_group"] = df.get("ancestor_discount_code", np.nan).apply(map_code_group)
df["gender_cat"] = df.apply(derive_gender, axis=1)

# Filter to quiz takers (+ influencers if desired)
if "quiz_taker" not in df.columns:
    raise ValueError("quiz_taker column is required.")
df_quiz = df[df["quiz_taker"].astype(str).str.lower() == "yes"].copy()
if INFLUENCER_ONLY:
    df_quiz = df_quiz[df_quiz["acquisition_code_group"].ne("none")].copy()

# Drop only hard leakage/IDs; keep value fields for later use (profiling, and Pass A features)
drop_now = [
    "email_key","quiz_date","first_order_date","last_order_date",
    "ancestor_discount_code","quiz_taker","first_sku","first_sku_buckets",
    "total_cogs","shipping_collected","shippingspend","refund_amt","refund_ratio",
    "recent_abx_flag","affiliate_segment",
]
df_quiz.drop(columns=[c for c in drop_now if c in df_quiz.columns], inplace=True, errors="ignore")

# Coverage filter (looser)
non_null_threshold = COVERAGE * len(df_quiz)
df_quiz = df_quiz.loc[:, df_quiz.notnull().sum() >= non_null_threshold].copy()
print(f"[COVERAGE] rows={len(df_quiz)}, kept_cols={len(df_quiz.columns)}")

# -------------------------- Common feature definitions --------------------------
# CATEGORICALS for personas (shared by both passes)
BASE_CAT = [
    "quiz_result","acquisition_code_group","bm_pattern","gi_symptom_cat",
    "primary_goal","gender_cat","first_sku_bucket"
]
# NUMERICS for personas (shared by both passes)
BASE_NUM = [
    "order_count","days_since_last_order","symptom_count","gut_issue_score",
    "high_stress","refund_count","quiz_reco_match"
]
# VALUE FIELDS
VALUE_COLS = [c for c in ["net_ltv","avg_order_value","gross_ltv"] if c in df_quiz.columns]

for c in BASE_CAT:
    if c in df_quiz.columns:
        df_quiz[c] = df_quiz[c].fillna("missing").astype(str)
for c in BASE_NUM:
    if c in df_quiz.columns:
        df_quiz[c] = pd.to_numeric(df_quiz[c], errors="coerce").fillna(0.0)

# -------------------------- Enhanced FAMD Utilities --------------------------
def generate_relationship_matrix(df_input: pd.DataFrame, variable_names: List[str], tag: str):
    """Generate FAMD-style relationship matrix (like Wikipedia Table 2)."""
    from scipy.stats import pearsonr, chi2_contingency
    
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
    
    # 3. Variable coordinates (loadings-like interpretation)
    # Skip variable correlations extraction as it's not reliably available in prince library
    # The relationship matrix and variable contributions provide the needed factor interpretation
    print(f"[{tag}] Variable correlations skipped - using relationship matrix and contributions instead")
    
    # 4. Factor interpretation summary
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
    
    # 5. Create factor contributions visualization
    print(f"[{tag}] Factor contributions heatmap skipped - data better viewed in CSV format")
    # Skip heatmap creation for factor contributions as values are on different scales
    # and are better interpreted from the CSV files directly
    
    # 6. Generate Relationship Matrix (like Wikipedia Table 2)
    print(f"[{tag}] Generating relationship matrix...")
    relationship_matrix = generate_relationship_matrix(df_input, variable_names, tag)
    
    return var_df, contrib_df, interp_df, relationship_matrix

def famd_fit_scores_enhanced(df_catnum: pd.DataFrame, explained_target: float, n_max: int, seed: int, tag: str):
    """Enhanced FAMD fit with comprehensive outputs."""
    famd_full = FAMD(n_components=min(n_max, max(2, df_catnum.shape[1]-1)), random_state=seed)
    famd_full.fit(df_catnum)
    # Removed debug exit
    
    eigs = np.asarray(famd_full.eigenvalues_, dtype=float)
    expl = eigs / eigs.sum()
    cum  = np.cumsum(expl)
    
    # Manual variance calculation used consistently
    
    n_keep = int(np.searchsorted(cum, explained_target) + 1)
    n_keep = max(2, min(n_keep, len(expl)))
    
    # Extract comprehensive interpretations using manual variance calculation
    var_names = df_catnum.columns.tolist()
    extract_famd_interpretations(famd_full, var_names, n_keep, tag, df_catnum, manual_explained_variance=expl)
    
    # Generate standard FAMD interpretation plots with manual variance
    generate_famd_interpretation_plots(famd_full, var_names, df_catnum, n_keep, tag, manual_explained_variance=expl)
    
    scores = famd_full.row_coordinates(df_catnum)
    X_all = scores.values if hasattr(scores, "values") else np.asarray(scores)
    X = X_all[:, :n_keep]
    
    # Save factor scores  
    scores_df = pd.DataFrame(X, columns=[f'Factor_{i+1}' for i in range(n_keep)])
    scores_df['customer_index'] = range(len(scores_df))
    scores_df.to_csv(os.path.join(CLUSTER_DIR, f"{tag}_factor_scores.csv"), index=False)
    
    # Save variable coordinates in factor space (if available)
    try:
        if hasattr(famd_full, 'column_coordinates_'):
            var_coords = famd_full.column_coordinates_.iloc[:, :n_keep]
            var_coords.to_csv(os.path.join(FAMD_DIR, f"{tag}_variable_coordinates.csv"))
        else:
            print(f"[{tag}] Variable coordinates not available in this FAMD implementation")
    except Exception as e:
        print(f"[{tag}] Could not save variable coordinates: {e}")
    
    # Generate true loading matrix (signed correlations) - ChatGPT method
    try:
        print(f"[{tag}] Generating true loading matrix (signed correlations)...")
        
        # Get row coordinates (factor scores) from FAMD
        row_coords = famd_full.row_coordinates(df_catnum)
        
        # For mixed data in FAMD, we need to handle categorical variables properly
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
        # This gives us the true loading matrix with signs
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

def famd_contrib_prune_enhanced(famd_model: FAMD, n_keep: int, famd_vars: List[str], frac: float, min_pct: float, tag: str) -> List[str]:
    contrib = famd_model.column_contributions_.copy()
    contrib_first = contrib.iloc[:, :n_keep]
    totals = (contrib_first.sum(axis=1))
    totals_pct = totals / totals.sum() * 100.0
    rank = pd.DataFrame({"variable": totals_pct.index, "total_contrib_pct": totals_pct.values}).sort_values("total_contrib_pct", ascending=False)
    rank.to_csv(os.path.join(FAMD_DIR, f"{tag}_feature_contributions_ranked.csv"), index=False)
    n_prune = int(np.floor(frac * len(rank)))
    to_drop = set(rank.tail(n_prune)["variable"].tolist()) | set(rank[rank["total_contrib_pct"] < min_pct]["variable"].tolist())
    keep_vars = [v for v in famd_vars if v not in to_drop]
    pd.DataFrame({"dropped_feature": sorted(list(to_drop))}).to_csv(os.path.join(FAMD_DIR, f"{tag}_pruned_features.csv"), index=False)
    return keep_vars

def choose_k_by_silhouette(X: np.ndarray, k_range=range(2,11), seed=42, label=""):
    sil_curve = []
    best_k, best_s = None, -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=30)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        sil_curve.append((k, s))
        if s > best_s:
            best_k, best_s = k, s
        print(f"[{label}] k={k} silhouette={s:.4f}")
    return best_k, best_s, sil_curve

def plot_sil_curve(sil_curve, path, title):
    ks = [k for k,_ in sil_curve]; ss = [s for _,s in sil_curve]
    pd.DataFrame(sil_curve, columns=["k","silhouette"]).to_csv(path.replace(".png",".csv"), index=False)
    plt.figure(figsize=(7,4)); plt.plot(ks, ss, marker="o")
    plt.xlabel("k"); plt.ylabel("Silhouette"); plt.title(title)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=DPI); plt.close()

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
        
        # 2. Variable Quality/Contribution Plot (Figure 2 equivalent)
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
        
        # 3. Category Representation (Figure 4 equivalent) - for qualitative variables
        try:
            categorical_vars = df_input.select_dtypes(include=['category', 'object']).columns.tolist()
            if categorical_vars and hasattr(famd_model, 'column_coordinates_'):
                plt.figure(figsize=(10, 8))
                
                # Get category coordinates (this varies by implementation)
                var_coords = famd_model.column_coordinates_.iloc[:, :min(2, n_components)]
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(categorical_vars)))
                
                for var_idx, var in enumerate(categorical_vars):
                    if var in variable_names:
                        var_pos = variable_names.index(var)
                        if var_pos < len(var_coords):
                            x = var_coords.iloc[var_pos, 0]
                            y = var_coords.iloc[var_pos, 1] if var_coords.shape[1] > 1 else 0
                            
                            # Plot variable position
                            plt.scatter(x, y, s=150, c=[colors[var_idx]], marker='s', 
                                      label=f'{var}', alpha=0.8, edgecolors='black')
                            
                            # Add category labels around the variable
                            categories = df_input[var].unique()
                            for cat_idx, category in enumerate(categories):
                                if pd.notna(category):
                                    # Offset categories around variable position
                                    angle = 2 * np.pi * cat_idx / len(categories)
                                    offset_x = 0.1 * np.cos(angle)
                                    offset_y = 0.1 * np.sin(angle)
                                    plt.text(x + offset_x, y + offset_y, str(category)[:10], 
                                           fontsize=8, ha='center', va='center',
                                           bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[var_idx], alpha=0.3))
                
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                plt.xlabel(f'Dim 1 ({famd_model.percentage_of_variance_[0]:.1f}%)')
                plt.ylabel(f'Dim 2 ({famd_model.percentage_of_variance_[1]:.1f}%)' if n_components > 1 else 'Dim 2')
                plt.title(f'{tag}: Categorical Variable Representation\n(Categories positioned around variable centers)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(FAMD_DIR, f"{tag}_category_representation.png"), dpi=DPI, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Warning: Could not create category representation plot for {tag}: {e}")
    
    except Exception as e:
        print(f"Warning: Could not create FAMD interpretation plots for {tag}: {e}")

def cluster_and_profile_enhanced(tag: str, df_base: pd.DataFrame, famd_vars_cat: List[str], famd_vars_num: List[str],
                        include_value_in_famd: bool, force_k: Optional[int]):
    # Build famd input
    cat_cols = [c for c in famd_vars_cat if c in df_base.columns]
    num_cols = [c for c in famd_vars_num if c in df_base.columns]
    famd_vars = cat_cols + num_cols
    famd_input = df_base[famd_vars].copy()
 
    # dominance filter
    famd_cat = [c for c in famd_input.columns if famd_input[c].dtype == object]
    famd_num = [c for c in famd_input.columns if c not in famd_cat]
    keep_num = [c for c in famd_num if pd.to_numeric(famd_input[c], errors="coerce").fillna(0).std(ddof=0) > 1e-8]
    keep_cat = []
    for c in famd_cat:
        v = famd_input[c].astype(str).fillna("missing")
        vc = v.value_counts(normalize=True, dropna=False)
        if (vc.size >= 2) and (vc.iloc[0] <= DOMINANCE_THRESH):
            keep_cat.append(c)
    famd_keep = keep_cat + keep_num
    famd_input_filt = famd_input[famd_keep].copy()
    for c in keep_cat:
        famd_input_filt[c] = famd_input_filt[c].astype("category")
    print(famd_input_filt.columns)
    
    # FAMD fit #1 (enhanced)
    famd1, X1, n_keep1, expl1 = famd_fit_scores_enhanced(famd_input_filt, EXPLAINED_TARGET, N_MAX_COMPONENTS, SEED, f"{tag}_initial")
    cumvar1 = float(np.cumsum(expl1)[n_keep1-1]); print(f"[{tag}] FAMD#1 kept {n_keep1} comps ({cumvar1*100:.1f}%)")
    
    # prune
    keep_vars = famd_contrib_prune_enhanced(famd1, n_keep1, famd_keep, PRUNE_FRACTION, MIN_CONTRIB_PCT, f"{tag}_pruning")
    if len(keep_vars) < 2: keep_vars = famd_keep

    # FAMD fit #2 (pruned, enhanced)
    famd_input_pruned = famd_input[keep_vars].copy()
    for c in keep_cat:
        if c in famd_input_pruned.columns:
            famd_input_pruned[c] = famd_input_pruned[c].astype("category")

    famd2, X2, n_keep2, expl2 = famd_fit_scores_enhanced(famd_input_pruned, EXPLAINED_TARGET, N_MAX_COMPONENTS, SEED, f"{tag}_final")
    cumvar2 = float(np.cumsum(expl2)[n_keep2-1]); print(f"[{tag}] FAMD#2 kept {n_keep2} comps ({cumvar2*100:.1f}%)")

    # choose k
    forced = force_k if isinstance(force_k, int) else None
    if forced is None:
        best_k, best_s, sil_curve = choose_k_by_silhouette(X2, K_RANGE, SEED, label=tag)
        chosen_k = best_k
    else:
        # evaluate forced k once for reporting
        km = KMeans(n_clusters=forced, random_state=SEED, n_init=30)
        labs = km.fit_predict(X2)
        s = silhouette_score(X2, labs)
        sil_curve = [(k, np.nan) for k in K_RANGE]
        chosen_k, best_s = forced, s
        print(f"[{tag}] FORCED k={chosen_k} silhouette={best_s:.4f}")

    plot_sil_curve(sil_curve, os.path.join(CLUSTER_DIR, f"{tag}_silhouette.png"), f"{tag}: Silhouette by k (FAMD pruned)")

    # final cluster labels
    kmeans = KMeans(n_clusters=chosen_k, random_state=SEED, n_init=50)
    clusters = kmeans.fit_predict(X2)

    # scatter of first two components
    plt.figure(figsize=(10,6))
    plt.scatter(X2[:,0], X2[:,1], c=clusters, s=10, cmap="tab10")
    plt.title(f"{tag}: FAMD (pruned) — KMeans k={chosen_k}")
    plt.xlabel("FAMD 1"); plt.ylabel("FAMD 2")
    plt.tight_layout(); plt.savefig(os.path.join(CLUSTER_DIR, f"{tag}_famd_scatter.png"), dpi=DPI); plt.close()

    # build profiles (include value overlays regardless of whether LTV was used)
    df_labels = df_base[["customer_id"]].copy()
    df_labels[f"cluster_{tag}"] = clusters.astype(int)
    df_join = df_labels.merge(df_quiz, on="customer_id", how="left")

    cat_profile_cols = [
        "quiz_result","bm_pattern","gi_symptom_cat",
        "acquisition_code_group","gender_cat","first_sku_bucket"
    ]
    num_profile_cols = [
        "order_count","days_since_last_order","symptom_count","gut_issue_score",
        "high_stress","refund_count","quiz_reco_match",
        # value overlays
        "net_ltv","avg_order_value","gross_ltv"
    ]

    profiles = []
    for cid, g in df_join.groupby(f"cluster_{tag}"):
        rec = {"cluster": int(cid), "n_customers": int(len(g))}
        # top categories
        for col in cat_profile_cols:
            if col in g.columns:
                vc = g[col].value_counts(normalize=True)
                if len(vc):
                    rec[f"{col}_top"] = vc.index[0]
                    rec[f"{col}_pct"] = round(vc.iloc[0]*100, 1)
        # numeric/values
        for col in num_profile_cols:
            if col in g.columns:
                series = pd.to_numeric(g[col], errors="coerce")
                rec[f"{col}_mean"] = round(series.fillna(0).mean(), 2)
                rec[f"{col}_median"] = round(series.median(), 2)
        # quiz-match %
        if "quiz_reco_match" in g.columns:
            rec["quiz_reco_match_pct"] = round(pd.to_numeric(g["quiz_reco_match"], errors="coerce").fillna(0).mean()*100, 1)
        # promo mix
        if "acquisition_code_group" in g.columns:
            promo = g["acquisition_code_group"].value_counts(normalize=True).head(3)
            for i, (code, share) in enumerate(promo.items(), start=1):
                rec[f"promo_{i}"] = f"{code}:{round(share*100,1)}%"
        profiles.append(rec)

    df_profiles = pd.DataFrame(profiles).sort_values("cluster")
    df_profiles.to_csv(os.path.join(CLUSTER_DIR, f"{tag}_cluster_profiles.csv"), index=False)

    # personas file
    def _safe(v): return v if isinstance(v, str) else "—"
    persona_lines = []
    for _, r in df_profiles.iterrows():
        blurb = (
            f"{tag} — Cluster {int(r['cluster'])} — {int(r['n_customers'])} customers\n"
            f"- Quiz result: {_safe(r.get('quiz_result_top'))}; Gender: {_safe(r.get('gender_cat_top'))}; First-SKU: {_safe(r.get('first_sku_bucket_top'))}\n"
            f"- GI/BM: {_safe(r.get('gi_symptom_cat_top'))}, {_safe(r.get('bm_pattern_top'))}; Avg symptoms: {r.get('symptom_count_mean',0)}\n"
            f"- Behavior: avg orders {r.get('order_count_mean',0)}, days since last order {r.get('days_since_last_order_mean',0)}; Quiz-match: {r.get('quiz_reco_match_pct',0)}%\n"
            f"- VALUE: mean LTV ${r.get('net_ltv_mean','—')}, mean AOV ${r.get('avg_order_value_mean','—')}\n"
            f"- Top promos: {_safe(r.get('promo_1'))} {_safe(r.get('promo_2'))}\n"
        )
        persona_lines.append(blurb)
    with open(os.path.join(CLUSTER_DIR, f"{tag}_personas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(persona_lines))

    return {
        "tag": tag,
        "chosen_k": chosen_k,
        "best_s": best_s,
        "n_keep": n_keep2,
        "cumvar": float(np.cumsum(expl2)[n_keep2-1]),
        "labels_frame": df_labels,          # columns: customer_id, cluster_tag
        "profiles": df_profiles,            # per-cluster stats incl. value overlays
    }

# -------------------------- Build base modeling frame --------------------------
# Keep personas columns + value columns on the side
keep_cols = ["customer_id"] + BASE_CAT + BASE_NUM + [c for c in VALUE_COLS if c in df_quiz.columns]
df_model_base = df_quiz[keep_cols].copy()


# -------------------------- PASS A: WITH LTV in FAMD --------------------------
print("\n=== PASS A: WITH LTV in clustering ===")
# For Pass A, include net_ltv & avg_order_value & gross_ltv **numerics** as features too
passA_cat = BASE_CAT[:]               # same categoricals
passA_num = BASE_NUM[:] + [c for c in VALUE_COLS if c in df_model_base.columns]  # add value numerics

resA = cluster_and_profile_enhanced("A_withLTV", df_model_base, passA_cat, passA_num, True, FORCE_K_A)

# -------------------------- PASS B: WITHOUT LTV in FAMD -----------------------
print("\n=== PASS B: WITHOUT LTV in clustering (value overlay only) ===")
passB_cat = BASE_CAT[:]
passB_num = BASE_NUM[:]               # exclude value numerics here
resB = cluster_and_profile_enhanced("B_noLTV", df_model_base, passB_cat, passB_num, False, FORCE_K_B)

# -------------------------- Comparison: value by cluster across passes --------
# Join LTV back in (already part of profiles). Create compact comparison tables.
def value_summary(df_profiles, tag):
    keep = ["cluster","n_customers","net_ltv_mean","net_ltv_median","avg_order_value_mean","avg_order_value_median"]
    return df_profiles[keep].copy().assign(tag=tag)

valA = value_summary(resA["profiles"], "A_withLTV")
valB = value_summary(resB["profiles"], "B_noLTV")
valA.to_csv(os.path.join(CLUSTER_DIR, "A_value_by_cluster.csv"), index=False)
valB.to_csv(os.path.join(CLUSTER_DIR, "B_value_by_cluster.csv"), index=False)

# Quick bar plots: mean LTV by cluster for each pass
def bar_value_plot(dfv, title, path):
    plt.figure(figsize=(8,4))
    plt.bar(dfv["cluster"].astype(str), pd.to_numeric(dfv["net_ltv_mean"], errors="coerce").fillna(0))
    plt.xlabel("Cluster"); plt.ylabel("Mean net LTV"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=DPI); plt.close()

bar_value_plot(valA.sort_values("cluster"), "Pass A (with LTV): mean net LTV by cluster", os.path.join(CLUSTER_DIR, "A_mean_ltv_by_cluster.png"))
bar_value_plot(valB.sort_values("cluster"), "Pass B (no LTV): mean net LTV by cluster", os.path.join(CLUSTER_DIR, "B_mean_ltv_by_cluster.png"))

# -------------------------- Enhanced Summary Report ---------------------------
# Create comprehensive summary with all factor analysis insights
summary_report = []
summary_report.append("# Enhanced FAMD Analysis Summary")
summary_report.append("=" * 50)
summary_report.append("")
summary_report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
summary_report.append(f"Total Quiz Takers Analyzed: {len(df_quiz):,}")
summary_report.append(f"Random Seed: {SEED}")
summary_report.append("")
summary_report.append("## PASS A (With LTV)")
summary_report.append(f"- Optimal Clusters: {resA['chosen_k']}")
summary_report.append(f"- Silhouette Score: {resA['best_s']:.4f}")
summary_report.append(f"- FAMD Components Retained: {resA['n_keep']} ({resA['cumvar']*100:.1f}% variance)")
summary_report.append("")
summary_report.append("## PASS B (Behavioral Only)")
summary_report.append(f"- Optimal Clusters: {resB['chosen_k']}")
summary_report.append(f"- Silhouette Score: {resB['best_s']:.4f}")
summary_report.append(f"- FAMD Components Retained: {resB['n_keep']} ({resB['cumvar']*100:.1f}% variance)")
summary_report.append("")
summary_report.append("## Key Files Generated")
summary_report.append("### Factor Analysis Interpretation:")
summary_report.append("- *_explained_variance.csv - Component eigenvalues and variance explained")
summary_report.append("- *_variable_contributions.csv - How much each variable contributes to factors")
summary_report.append("- *_factor_interpretation.csv - Business interpretation of each factor")
summary_report.append("- *_relationship_matrix.csv - Variable-to-variable relationships (R², φ², η²)")
summary_report.append("- *_relationship_matrix_heatmap.png - Visual relationship matrix (off-diagonal only)")
summary_report.append("- *_variable_coordinates.csv - Variable coordinates in factor space (if available)")
summary_report.append("")
summary_report.append("### Clustering Results:")
summary_report.append("- *_cluster_profiles.csv - Detailed cluster characteristics")
summary_report.append("- *_personas.txt - Business-friendly cluster descriptions")
summary_report.append("- *_silhouette.png/.csv - Cluster quality analysis")
summary_report.append("- *_famd_scatter.png - Factor space visualization")
summary_report.append("")
summary_report.append("### Factor Scores:")
summary_report.append("- *_factor_scores.csv - Customer positions in factor space")
summary_report.append("")
summary_report.append("## Recommendations for Working Session")
summary_report.append("1. Review factor interpretations to understand what drives customer behavior")
summary_report.append("2. Compare variable-factor correlations between Pass A and Pass B")
summary_report.append("3. Decide on optimal approach based on business objectives")
summary_report.append("4. Use factor loadings to inform marketing strategy development")

with open(os.path.join(CLUSTER_DIR, "COMPREHENSIVE_ANALYSIS_SUMMARY.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_report))

print("\n" + "="*60)
print("ENHANCED FAMD ANALYSIS COMPLETE")
print("="*60)
print(f"FAMD interpretation files saved to: {FAMD_DIR}/")
print(f"Clustering/profiling files saved to: {CLUSTER_DIR}/")
print(f"\nKey files for Hannah & Klaus:")
print(f" - {os.path.join(FAMD_DIR, '*_factor_interpretation.csv')}")
print(f" - {os.path.join(FAMD_DIR, '*_relationship_matrix.csv')} (NEW!)")
print(f" - {os.path.join(FAMD_DIR, '*_relationship_matrix_heatmap.png')} (NEW!)")
print(f" - {os.path.join(FAMD_DIR, '*_correlation_circle.png')} (NEW!)")
print(f" - {os.path.join(FAMD_DIR, '*_variable_quality.png')} (NEW!)")
print(f" - {os.path.join(CLUSTER_DIR, 'COMPREHENSIVE_ANALYSIS_SUMMARY.txt')}")
print(f"\nFAMD_output folder now contains ONLY core FAMD interpretation files!")
print("\nNEW: Relationship matrices show variable-to-variable relationships (R², φ², η²)")
print("This matches the Wikipedia FAMD Table 2 format that Hannah & Klaus expect!")
print("\nReady for working session with Hannah & Klaus!")
