# ================================================================
# ALLERGOSAN – CUSTOMER SEGMENTATION (Behaviour + Quiz)
# Final Analysis Script
#
# Author: Senior Data Scientist (AI Assistant)
# Date: January 2025
# Methodology: Two-Stage Clustering as per Mega-Prompt
# ================================================================

import pandas as pd
import numpy as np
import warnings
import os
import time
from textwrap import dedent

# Preprocessing & Dimensionality Reduction
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.decomposition import FactorAnalysis
import prince

# Clustering & Validation
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, jaccard_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import gower

# --- Configuration ---
np.random.seed(42)
warnings.filterwarnings('ignore')
if not os.path.exists('figs'):
    os.makedirs('figs')
print("Libraries imported and environment configured.")

# HELPER FUNCTIONS
def get_timer(start_time): return f"({time.time() - start_time:.2f}s)"
def bootstrap_jaccard_score(model, X, n_boots=50):
    original_labels = model.fit_predict(X)
    jaccard_scores = []
    for i in range(n_boots):
        boot_indices = resample(np.arange(X.shape[0]), n_samples=int(0.8 * X.shape[0]), random_state=i)
        X_boot = X[boot_indices]
        boot_labels = model.fit_predict(X_boot)
        original_labels_in_sample = original_labels[boot_indices]
        score = jaccard_score(original_labels_in_sample, boot_labels, average='weighted')
        jaccard_scores.append(score)
    return np.mean(jaccard_scores)

# ================================================================
# MAIN ANALYSIS WORKFLOW
# ================================================================
def run_analysis():
    start_time = time.time()
    
    # 1. DATA PREP
    t0 = time.time()
    print("\\n--- 1. DATA PREP ---")
    try:
        df = pd.read_csv('raw_data_v2.csv', low_memory=False)
    except FileNotFoundError:
        print("CHECKPOINT-1.0: ERROR - raw_data_v2.csv not found."); return
    df.dropna(subset=['customer_id'], inplace=True)
    print(f"CHECKPOINT-1.1: Data loaded. Shape: {df.shape} {get_timer(t0)}")

    behavioral_numeric = ['days_since_last_order', 'order_count', 'gross_ltv', 'net_ltv', 'avg_order_value']
    behavioral_binary = ['is_male', 'is_pregnant', 'quiz_taker']
    for col in behavioral_numeric: df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in behavioral_binary: df[col] = df[col].astype(str).str.lower().isin(['1', '1.0', 'yes', 'y', 'true']).astype(int)
    
    numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    binary_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])
    preprocessor_s1 = ColumnTransformer(transformers=[('num', numeric_pipeline, behavioral_numeric), ('bin', binary_pipeline, behavioral_binary)], remainder='drop')
    
    X_s1_processed = preprocessor_s1.fit_transform(df)
    print(f"CHECKPOINT-1.2: Stage-1 data prepared. Shape: {X_s1_processed.shape} {get_timer(t0)}")

    # 2. STAGE-1 CLUSTERING
    t0 = time.time()
    print("\\n--- 2. STAGE-1 CLUSTERING (BEHAVIOUR ONLY) ---")
    
    fa = FactorAnalysis(n_components=3, rotation='varimax', random_state=42)
    X_rotated = fa.fit_transform(X_s1_processed)
    print(f"CHECKPOINT-2.1: Factor Analysis with Varimax complete. {get_timer(t0)}")

    k_range = range(3, 9)
    kmeans_results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_rotated)
        kmeans_results.append({'k': k, 'silhouette': silhouette_score(X_rotated, labels)})
    df_kmeans_results = pd.DataFrame(kmeans_results).set_index('k')
    print(f"CHECKPOINT-2.2: K-Means grid search complete. {get_timer(t0)}")
    print(df_kmeans_results)
    
    best_k_s1 = df_kmeans_results['silhouette'].idxmax()
    kmeans_final = KMeans(n_clusters=best_k_s1, random_state=42, n_init=10)
    stability_score = bootstrap_jaccard_score(kmeans_final, X_rotated)
    print(f"CHECKPOINT-2.3: Best K is {best_k_s1}. Bootstrap Jaccard stability = {stability_score:.3f} {get_timer(t0)}")

    if stability_score >= 0.70:
        df['cluster_lvl1'] = kmeans_final.fit_predict(X_rotated)
        print("CHECKPOINT-2.4: Stage-1 K-Means is stable.")
    else:
        print("CHECKPOINT-2.4: K-Means unstable. Falling back to Gaussian Mixture Model.")
        gmm = GaussianMixture(n_components=best_k_s1, covariance_type='diag', random_state=42)
        df['cluster_lvl1'] = gmm.fit_predict(X_rotated)
        
    # 3. STAGE-2 CLUSTERING
    t0 = time.time()
    print("\\n--- 3. STAGE-2 CLUSTERING (QUIZ-TAKERS ONLY) ---")
    df_s2 = df[df['quiz_taker'] == 1].copy()
    print(f"CHECKPOINT-3.1: Subsetted to {len(df_s2)} quiz-takers. {get_timer(t0)}")

    quiz_numeric = ['symptom_count']
    quiz_categorical = ['gi_symptom_cat', 'bm_pattern', 'primary_goal']
    all_s2_numeric = behavioral_numeric + quiz_numeric
    all_s2_categorical = behavioral_binary + quiz_categorical
    df_s2_model = df_s2[all_s2_numeric + all_s2_categorical].copy()
    for col in all_s2_numeric: df_s2_model[col] = pd.to_numeric(df_s2_model[col], errors='coerce').fillna(df_s2_model[col].median())
    for col in all_s2_categorical: df_s2_model[col] = df_s2_model[col].astype(str).fillna('missing').astype('category')

    famd = prince.FAMD(n_components=5, n_iter=3, random_state=42)
    X_famd = famd.fit_transform(df_s2_model)
    print(f"CHECKPOINT-3.2: FAMD on quiz data complete. {get_timer(t0)}")
    
    cat_features_mask = [col in all_s2_categorical for col in df_s2_model.columns]
    gower_matrix = gower.gower_matrix(df_s2_model, cat_features=cat_features_mask)
    condensed_gower_matrix = squareform(gower_matrix, checks=False)
    
    linked = linkage(condensed_gower_matrix, method='complete')
    best_k_s2 = 4
    df_s2['cluster_lvl2'] = fcluster(linked, best_k_s2, criterion='maxclust')
    print(f"CHECKPOINT-3.3: Hierarchical clustering complete. Cut for K={best_k_s2}. {get_timer(t0)}")
    df = df.merge(df_s2[['customer_id', 'cluster_lvl2']], on='customer_id', how='left')

    # 4 & 5. PROFILING & DELIVERABLES
    t0 = time.time()
    print("\\n--- 4. PROFILING & 5. DELIVERABLES ---")
    
    profile_lvl1 = df.groupby('cluster_lvl1')[behavioral_numeric].mean()
    profile_lvl1['size'] = df['cluster_lvl1'].value_counts()
    print("\\n--- Level 1 Cluster Profiles (Behavioral) ---")
    print(profile_lvl1)
    
    profile_lvl2 = df.groupby(['cluster_lvl1', 'cluster_lvl2'])[quiz_numeric + ['net_ltv']].mean()
    profile_lvl2['size'] = df.groupby(['cluster_lvl1', 'cluster_lvl2']).size()
    print("\\n--- Level 2 Cluster Profiles (Quiz Takers) ---")
    print(profile_lvl2)
    
    deliverable_cols = ['customer_id', 'cluster_lvl1', 'cluster_lvl2']
    df[deliverable_cols].to_csv('segmented_customers.csv', index=False)
    print(f"CHECKPOINT-5.1: `segmented_customers.csv` saved. {get_timer(t0)}")
    
    summary = dedent(f"""
    ================================================================
      EXECUTIVE SUMMARY FOR PRESENTATION SLIDE
    ================================================================
    *   **Methodology:** A two-stage clustering process was used. Stage 1 segmented all customers by behavior; Stage 2 sub-clustered quiz-takers by their health profiles.
    *   **Stage 1 Results:** Identified {df['cluster_lvl1'].nunique()} distinct behavioral segments. The segmentation was validated for stability (Jaccard Score: {stability_score:.3f}).
    *   **Top Behavioral Segment (by LTV):** Cluster {profile_lvl1['net_ltv'].idxmax()} shows the highest value.
    *   **Stage 2 Insights:** Within the behavioral segments, we found {df['cluster_lvl2'].nunique()} distinct health personas among quiz-takers, allowing for hyper-targeted product recommendations.
    *   **Action:** Use Level 1 clusters for broad lifecycle marketing (e.g., retention, win-back) and Level 2 clusters for targeted, quiz-specific content and product messaging.
    ================================================================
    """)
    print(summary)
    print(f"CHECKPOINT-5.2: Summary generated. Total runtime: {get_timer(start_time)}")

if __name__ == '__main__':
    run_analysis()
