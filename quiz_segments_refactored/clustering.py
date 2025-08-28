# Customer clustering and profiling functionality
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from .config import *
    from .famd_analysis import fit_famd_model, prune_features_by_contribution, apply_dominance_filter
except ImportError:
    from config import *
    from famd_analysis import fit_famd_model, prune_features_by_contribution, apply_dominance_filter

def choose_k_by_silhouette(X: np.ndarray, k_range=K_RANGE, seed=SEED, label="") -> Tuple[int, float, List[Tuple[int, float]]]:
    """
    Find optimal number of clusters using silhouette analysis.
    
    Args:
        X: Factor scores array
        k_range: Range of k values to test
        seed: Random seed
        label: Label for logging
        
    Returns:
        Tuple of (best_k, best_silhouette, silhouette_curve)
    """
    print(f"[{label}] Finding optimal k using silhouette analysis...")
    
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
    
    print(f"[{label}] Best k={best_k} with silhouette={best_s:.4f}")
    return best_k, best_s, sil_curve

def plot_silhouette_curve(sil_curve: List[Tuple[int, float]], path: str, title: str):
    """Plot silhouette scores by k value."""
    ks = [k for k, _ in sil_curve]
    ss = [s for _, s in sil_curve]
    
    # Save data
    pd.DataFrame(sil_curve, columns=["k", "silhouette"]).to_csv(path.replace(".png", ".csv"), index=False)
    
    # Create plot
    plt.figure(figsize=(7, 4))
    plt.plot(ks, ss, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()

def create_cluster_profiles(df_join: pd.DataFrame, cluster_col: str, tag: str) -> pd.DataFrame:
    """
    Create detailed cluster profiles with categorical and numeric summaries.
    
    Args:
        df_join: Dataframe with cluster assignments and features
        cluster_col: Name of cluster column
        tag: Tag for file naming
        
    Returns:
        DataFrame with cluster profiles
    """
    print(f"[{tag}] Creating cluster profiles...")
    
    cat_profile_cols = [
        "quiz_result", "bm_pattern", "gi_symptom_cat",
        "acquisition_code_group", "gender_cat", "first_sku_bucket"
    ]
    
    num_profile_cols = [
        "order_count", "days_since_last_order", "symptom_count", "gut_issue_score",
        "high_stress", "refund_count", "quiz_reco_match",
        # value overlays
        "net_ltv", "avg_order_value", "gross_ltv"
    ]
    
    profiles = []
    
    for cid, g in df_join.groupby(cluster_col):
        rec = {"cluster": int(cid), "n_customers": int(len(g))}
        
        # Top categories for categorical variables
        for col in cat_profile_cols:
            if col in g.columns:
                vc = g[col].value_counts(normalize=True)
                if len(vc):
                    rec[f"{col}_top"] = vc.index[0]
                    rec[f"{col}_pct"] = round(vc.iloc[0] * 100, 1)
        
        # Numeric summaries
        for col in num_profile_cols:
            if col in g.columns:
                series = pd.to_numeric(g[col], errors="coerce")
                rec[f"{col}_mean"] = round(series.fillna(0).mean(), 2)
                rec[f"{col}_median"] = round(series.median(), 2)
        
        # Quiz-match percentage
        if "quiz_reco_match" in g.columns:
            rec["quiz_reco_match_pct"] = round(
                pd.to_numeric(g["quiz_reco_match"], errors="coerce").fillna(0).mean() * 100, 1
            )
        
        # Promo mix (top 3 acquisition codes)
        if "acquisition_code_group" in g.columns:
            promo = g["acquisition_code_group"].value_counts(normalize=True).head(3)
            for i, (code, share) in enumerate(promo.items(), start=1):
                rec[f"promo_{i}"] = f"{code}:{round(share*100, 1)}%"
        
        profiles.append(rec)
    
    df_profiles = pd.DataFrame(profiles).sort_values("cluster")
    df_profiles.to_csv(os.path.join(CLUSTER_DIR, f"{tag}_cluster_profiles.csv"), index=False)
    
    print(f"[{tag}] Created profiles for {len(profiles)} clusters")
    return df_profiles

def create_personas_narrative(df_profiles: pd.DataFrame, tag: str):
    """Create business-friendly persona descriptions."""
    
    def _safe(v): 
        return v if isinstance(v, str) else "—"
    
    persona_lines = []
    
    for _, r in df_profiles.iterrows():
        blurb = (
            f"{tag} — Cluster {int(r['cluster'])} — {int(r['n_customers'])} customers\n"
            f"- Quiz result: {_safe(r.get('quiz_result_top'))}; Gender: {_safe(r.get('gender_cat_top'))}; First-SKU: {_safe(r.get('first_sku_bucket_top'))}\n"
            f"- GI/BM: {_safe(r.get('gi_symptom_cat_top'))}, {_safe(r.get('bm_pattern_top'))}; Avg symptoms: {r.get('symptom_count_mean', 0)}\n"
            f"- Behavior: avg orders {r.get('order_count_mean', 0)}, days since last order {r.get('days_since_last_order_mean', 0)}; Quiz-match: {r.get('quiz_reco_match_pct', 0)}%\n"
            f"- VALUE: mean LTV ${r.get('net_ltv_mean', '—')}, mean AOV ${r.get('avg_order_value_mean', '—')}\n"
            f"- Top promos: {_safe(r.get('promo_1'))} {_safe(r.get('promo_2'))}\n"
        )
        persona_lines.append(blurb)
    
    with open(os.path.join(CLUSTER_DIR, f"{tag}_personas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(persona_lines))
    
    print(f"[{tag}] Created persona narratives")

def plot_famd_clusters(X: np.ndarray, clusters: np.ndarray, chosen_k: int, tag: str):
    """Create scatter plot of clusters in FAMD space."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, s=10, cmap="tab10")
    plt.title(f"{tag}: FAMD (pruned) — KMeans k={chosen_k}")
    plt.xlabel("FAMD Component 1")
    plt.ylabel("FAMD Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join(CLUSTER_DIR, f"{tag}_famd_scatter.png"), dpi=DPI)
    plt.close()

def cluster_and_profile_enhanced(tag: str, df_base: pd.DataFrame, famd_vars_cat: List[str], 
                               famd_vars_num: List[str], include_value_in_famd: bool, 
                               force_k: Optional[int]) -> Dict:
    """
    Complete clustering pipeline: FAMD -> pruning -> final FAMD -> clustering -> profiling.
    
    Args:
        tag: Tag for output files
        df_base: Base dataframe with all features
        famd_vars_cat: Categorical variables for FAMD
        famd_vars_num: Numeric variables for FAMD  
        include_value_in_famd: Whether to include value metrics in FAMD
        force_k: Force specific number of clusters (optional)
        
    Returns:
        Dictionary with clustering results and metadata
    """
    print(f"\n=== {tag}: Enhanced Clustering Pipeline ===")
    
    # Build FAMD input
    cat_cols = [c for c in famd_vars_cat if c in df_base.columns]
    num_cols = [c for c in famd_vars_num if c in df_base.columns]
    famd_vars = cat_cols + num_cols
    famd_input = df_base[famd_vars].copy()
    
    print(f"[{tag}] Starting with {len(famd_vars)} variables: {len(cat_cols)} categorical + {len(num_cols)} numeric")
    
    # Apply dominance filter
    famd_input_filt = apply_dominance_filter(famd_input)
    print(f"[{tag}] After dominance filter: {famd_input_filt.shape}")
    
    # FAMD fit #1 (initial)
    famd1, X1, n_keep1, expl1 = fit_famd_model(
        famd_input_filt, EXPLAINED_TARGET, N_MAX_COMPONENTS, SEED, f"{tag}_initial"
    )
    cumvar1 = float(np.cumsum(expl1)[n_keep1-1])
    print(f"[{tag}] FAMD#1 kept {n_keep1} components ({cumvar1*100:.1f}% variance)")
    
    # Prune features based on contribution
    keep_vars = prune_features_by_contribution(
        famd1, n_keep1, famd_input_filt.columns.tolist(), PRUNE_FRACTION, MIN_CONTRIB_PCT, f"{tag}_pruning"
    )
    if len(keep_vars) < 2: 
        keep_vars = famd_input_filt.columns.tolist()
    
    # FAMD fit #2 (pruned)
    famd_input_pruned = famd_input[keep_vars].copy()
    categorical_cols = famd_input_pruned.select_dtypes(include=['object']).columns
    for c in categorical_cols:
        famd_input_pruned[c] = famd_input_pruned[c].astype("category")
    
    famd2, X2, n_keep2, expl2 = fit_famd_model(
        famd_input_pruned, EXPLAINED_TARGET, N_MAX_COMPONENTS, SEED, f"{tag}_final"
    )
    cumvar2 = float(np.cumsum(expl2)[n_keep2-1])
    print(f"[{tag}] FAMD#2 kept {n_keep2} components ({cumvar2*100:.1f}% variance)")
    
    # Choose optimal k
    if force_k is not None:
        # Use forced k
        km = KMeans(n_clusters=force_k, random_state=SEED, n_init=30)
        clusters = km.fit_predict(X2)
        best_s = silhouette_score(X2, clusters)
        sil_curve = [(k, np.nan) for k in K_RANGE]
        chosen_k = force_k
        print(f"[{tag}] FORCED k={chosen_k} silhouette={best_s:.4f}")
    else:
        # Find optimal k
        best_k, best_s, sil_curve = choose_k_by_silhouette(X2, K_RANGE, SEED, label=tag)
        chosen_k = best_k
    
    # Plot silhouette curve
    plot_silhouette_curve(sil_curve, os.path.join(CLUSTER_DIR, f"{tag}_silhouette.png"), 
                         f"{tag}: Silhouette by k (FAMD pruned)")
    
    # Final clustering
    kmeans = KMeans(n_clusters=chosen_k, random_state=SEED, n_init=50)
    clusters = kmeans.fit_predict(X2)
    
    # Plot clusters in FAMD space
    plot_famd_clusters(X2, clusters, chosen_k, tag)
    
    # Build comprehensive profiles
    df_labels = df_base[["customer_id"]].copy()
    df_labels[f"cluster_{tag}"] = clusters.astype(int)
    
    # Join with original data for profiling (need the full quiz dataset)
    try:
        from .data_cleaning import get_modeling_dataframe
    except ImportError:
        from data_cleaning import get_modeling_dataframe
    df_quiz = get_modeling_dataframe()
    df_join = df_labels.merge(df_quiz, on="customer_id", how="left")
    
    # Create detailed profiles
    df_profiles = create_cluster_profiles(df_join, f"cluster_{tag}", tag)
    
    # Create persona narratives
    create_personas_narrative(df_profiles, tag)
    
    print(f"[{tag}] Clustering completed: {chosen_k} clusters, silhouette={best_s:.4f}")
    
    return {
        "tag": tag,
        "chosen_k": chosen_k,
        "best_s": best_s,
        "n_keep": n_keep2,
        "cumvar": cumvar2,
        "labels_frame": df_labels,
        "profiles": df_profiles,
        "factor_scores": X2,
        "clusters": clusters
    }

def create_value_comparison(results_dict: Dict[str, Dict], output_dir: str = CLUSTER_DIR):
    """Create comparison of value metrics across different clustering approaches."""
    
    print("Creating value comparison across clustering approaches...")
    
    comparison_data = []
    
    for approach_name, results in results_dict.items():
        profiles = results["profiles"]
        
        for _, row in profiles.iterrows():
            comparison_data.append({
                "approach": approach_name,
                "cluster": row["cluster"],
                "n_customers": row["n_customers"],
                "net_ltv_mean": row.get("net_ltv_mean", 0),
                "net_ltv_median": row.get("net_ltv_median", 0),
                "avg_order_value_mean": row.get("avg_order_value_mean", 0),
                "avg_order_value_median": row.get("avg_order_value_median", 0),
                "order_count_mean": row.get("order_count_mean", 0)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, "clustering_value_comparison.csv"), index=False)
    
    # Create comparison plots
    approaches = comparison_df["approach"].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Net LTV by cluster
    ax = axes[0, 0]
    for approach in approaches:
        data = comparison_df[comparison_df["approach"] == approach]
        ax.bar([f"{approach}\nC{int(c)}" for c in data["cluster"]], 
               data["net_ltv_mean"], alpha=0.7, label=approach)
    ax.set_title("Mean Net LTV by Cluster")
    ax.set_ylabel("Net LTV ($)")
    ax.tick_params(axis='x', rotation=45)
    
    # AOV by cluster
    ax = axes[0, 1]
    for approach in approaches:
        data = comparison_df[comparison_df["approach"] == approach]
        ax.bar([f"{approach}\nC{int(c)}" for c in data["cluster"]], 
               data["avg_order_value_mean"], alpha=0.7, label=approach)
    ax.set_title("Mean AOV by Cluster")
    ax.set_ylabel("AOV ($)")
    ax.tick_params(axis='x', rotation=45)
    
    # Order count by cluster
    ax = axes[1, 0]
    for approach in approaches:
        data = comparison_df[comparison_df["approach"] == approach]
        ax.bar([f"{approach}\nC{int(c)}" for c in data["cluster"]], 
               data["order_count_mean"], alpha=0.7, label=approach)
    ax.set_title("Mean Order Count by Cluster")
    ax.set_ylabel("Order Count")
    ax.tick_params(axis='x', rotation=45)
    
    # Customer count by cluster
    ax = axes[1, 1]
    for approach in approaches:
        data = comparison_df[comparison_df["approach"] == approach]
        ax.bar([f"{approach}\nC{int(c)}" for c in data["cluster"]], 
               data["n_customers"], alpha=0.7, label=approach)
    ax.set_title("Customer Count by Cluster")
    ax.set_ylabel("Number of Customers")
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clustering_value_comparison.png"), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("Value comparison completed")

if __name__ == "__main__":
    # Test the clustering pipeline
    try:
        from .data_cleaning import get_modeling_dataframe
    except ImportError:
        from data_cleaning import get_modeling_dataframe
    
    df = get_modeling_dataframe()
    
    print("\nTesting clustering pipeline...")
    
    # Test behavioral clustering (no LTV)
    results = cluster_and_profile_enhanced(
        tag="test_behavioral",
        df_base=df,
        famd_vars_cat=BASE_CAT,
        famd_vars_num=BASE_NUM,
        include_value_in_famd=False,
        force_k=None
    )
    
    print(f"\nClustering test completed!")
    print(f"Found {results['chosen_k']} clusters with silhouette score {results['best_s']:.3f}")
    print(f"Explained {results['cumvar']*100:.1f}% variance with {results['n_keep']} factors")
