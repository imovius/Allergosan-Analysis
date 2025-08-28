#!/usr/bin/env python3
"""
Customer Segmentation Analysis Script - BEHAVIORAL FOCUS
Removes obvious revenue drivers to surface interesting health & behavior patterns

This version excludes LTV/revenue from clustering to find segments based on:
- Health patterns (quiz features)
- Purchase behavior (frequency, timing, returns)
- Marketing attribution (channels, discounts)
- Product preferences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin
import prince
import lightgbm as lgb

print("=== ALLERGOSAN BEHAVIORAL SEGMENTATION ANALYSIS ===")
print("Focus: Health patterns, behavior, NOT obvious revenue drivers")
print("Starting analysis...")

# --- 1. Data Loading and Schema Audit ---
print("\n1. Loading data and performing schema audit...")

try:
    df = pd.read_csv('raw_data_v2.csv')
    print(f"✓ Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'raw_data_v2.csv' not found.")
    exit()

# Schema catalogue - BEHAVIORAL FOCUS
date_cols = ['created_at', 'quiz_date']

# BEHAVIORAL numerics (NO obvious revenue drivers)
behavioral_numeric_cols = [
    'days_since_last_order',    # Recency
    'order_count',              # Frequency (keep this as it's behavior, not value)
    'refund_count',             # Return behavior
    'avg_days_between_orders',  # Purchase rhythm
    'refund_ratio',             # Return propensity 
    'shipping_spend',           # Shipping preference
    # EXCLUDED: gross_ltv, net_ltv, avg_order_value, total_cogs (too obvious)
]

# Health & quiz features (the interesting stuff!)
health_binary_cols = [
    'stress_mental_flag','stress_physical_flag','stress_digestion_flag','high_stress',
    'took_antibiotics_recently_flag','stomach_flu_flag','digestive_meds_flag',
    'sx_bloating','sx_reflux','sx_constipation','sx_diarrhea',
    'sx_anxiety','sx_brain_fog','sx_uti','sx_acne',
    'inflammatory_condition','food_intolerance',
    'is_male','is_pregnant','probiotic_for_child_flag'
]

# Marketing & preference features  
marketing_categorical_cols = [
    'affiliate_segment', 'ancestor_discount_code', 'first_sku', 
    'bm_pattern', 'gi_symptom_cat', 'primary_goal', 'quiz_result'
]

# Create outputs directory
import os
os.makedirs('outputs', exist_ok=True)

# Schema audit
schema_audit = pd.DataFrame({
    'dtype': df.dtypes,
    'cardinality': df.nunique(),
    'missing_values': df.isnull().sum(),
    'missing_percentage': (df.isnull().sum() / len(df)) * 100
})
schema_audit.to_csv('outputs/schema_audit_behavioral.csv')
print("✓ Schema audit saved")

# --- 2. Preprocessing Pipeline ---
print("\n2. Setting up behavioral-focused preprocessing...")

# Convert date columns
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Target encoding for high-cardinality categoricals
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self.global_mean = None
        self.category_means = {}
    
    def fit(self, X, y):
        self.global_mean = y.mean()
        for col in X.columns:
            category_stats = X[[col]].join(pd.Series(y, name='target')).groupby(col)['target'].agg(['mean', 'count'])
            smoothed_means = (category_stats['mean'] * category_stats['count'] + self.global_mean * self.smoothing) / (category_stats['count'] + self.smoothing)
            self.category_means[col] = smoothed_means.to_dict()
        return self
    
    def transform(self, X):
        result = X.copy()
        for col in X.columns:
            result[col] = X[col].map(self.category_means[col]).fillna(self.global_mean)
        return result

print("✓ Preprocessing pipeline ready")

# --- 3. Quiz-subset Analysis ---
print("\n3. Analyzing quiz-taking subset...")

quiz_subset_df = df.dropna(subset=['quiz_date']).copy()
print(f"Quiz subset shape: {quiz_subset_df.shape}")
print(f"Quiz participation rate: {len(quiz_subset_df)/len(df)*100:.1f}%")

# Get available features for quiz analysis
available_behavioral_nums = [col for col in behavioral_numeric_cols if col in quiz_subset_df.columns]
available_health_flags = [col for col in health_binary_cols if col in quiz_subset_df.columns]
available_marketing_cats = [col for col in marketing_categorical_cols if col in quiz_subset_df.columns]

all_quiz_features = available_behavioral_nums + available_health_flags + available_marketing_cats
quiz_features_df = quiz_subset_df[all_quiz_features].copy()

print(f"Quiz analysis features: {len(available_behavioral_nums)} behavioral + {len(available_health_flags)} health + {len(available_marketing_cats)} marketing")

# Remove duplicates and impute
quiz_features_df = quiz_features_df.loc[:, ~quiz_features_df.columns.duplicated()]

# Simple imputation
for col in available_behavioral_nums:
    if col in quiz_features_df.columns:
        quiz_features_df[col] = quiz_features_df[col].fillna(quiz_features_df[col].median())

for col in available_health_flags:
    if col in quiz_features_df.columns:
        quiz_features_df[col] = quiz_features_df[col].fillna(0)

for col in available_marketing_cats:
    if col in quiz_features_df.columns:
        quiz_features_df[col] = quiz_features_df[col].fillna('missing')

# Run FAMD on quiz subset
print("Running FAMD on behavioral + health features...")
try:
    max_components = min(15, len(quiz_features_df.columns))
    famd = prince.FAMD(n_components=max_components, n_iter=3, random_state=42)
    famd = famd.fit(quiz_features_df)
    
    cumulative_variance = famd.cumulative_percentage_of_variance_
    n_components_to_retain = np.argmax(np.array(cumulative_variance) >= 70.0) + 1
    n_components_to_retain = min(n_components_to_retain, 8)
    
    quiz_famd_components = famd.transform(quiz_features_df)
    retained_quiz_components = quiz_famd_components.iloc[:, :n_components_to_retain]
    
    print(f"✓ FAMD complete. Retained {n_components_to_retain} components")
    
except Exception as e:
    print(f"✗ FAMD failed: {e}")
    print("Falling back to PCA on numeric features only...")
    
    numeric_only_df = quiz_features_df[available_behavioral_nums].copy()
    scaler_quiz = StandardScaler()
    quiz_scaled = scaler_quiz.fit_transform(numeric_only_df)
    
    max_pca_components = min(6, len(available_behavioral_nums))
    pca_quiz = PCA(n_components=max_pca_components, random_state=42)
    retained_quiz_components = pd.DataFrame(
        pca_quiz.fit_transform(quiz_scaled),
        index=quiz_features_df.index
    )
    n_components_to_retain = retained_quiz_components.shape[1]
    print(f"✓ PCA fallback complete. Using {n_components_to_retain} components")

# --- 4. Quiz-subset Clustering ---
print("\n4. Clustering quiz subset (behavioral + health patterns)...")

def simple_stability_score(data, clustering_algorithm, n_resamples=5):
    scores = []
    for i in range(n_resamples):
        sample_size = int(0.7 * len(data))
        sample_idx = np.random.choice(len(data), sample_size, replace=False)
        
        sample1_idx = np.random.choice(sample_idx, int(0.8 * len(sample_idx)), replace=False)
        sample2_idx = np.random.choice(sample_idx, int(0.8 * len(sample_idx)), replace=False)
        
        if len(sample1_idx) > 10 and len(sample2_idx) > 10:
            labels1 = clustering_algorithm.fit_predict(data.iloc[sample1_idx])
            clustering_algorithm2 = type(clustering_algorithm)(**clustering_algorithm.get_params())
            labels2 = clustering_algorithm2.fit_predict(data.iloc[sample2_idx])
            
            common_idx = np.intersect1d(sample1_idx, sample2_idx)
            if len(common_idx) > 5:
                overlap1, overlap2 = [], []
                for idx in common_idx:
                    pos1 = np.where(sample1_idx == idx)[0]
                    pos2 = np.where(sample2_idx == idx)[0]
                    if len(pos1) > 0 and len(pos2) > 0:
                        overlap1.append(labels1[pos1[0]])
                        overlap2.append(labels2[pos2[0]])
                
                if len(overlap1) > 3:
                    score = adjusted_rand_score(overlap1, overlap2)
                    scores.append(score)
    
    return np.mean(scores) if scores else 0

# Clustering grid for quiz subset
k_range = range(3, 7)
results = []

for k in k_range:
    print(f"  Testing K={k}...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = kmeans.fit_predict(retained_quiz_components)
    silhouette = silhouette_score(retained_quiz_components, labels)
    db_score = davies_bouldin_score(retained_quiz_components, labels)
    stability = simple_stability_score(retained_quiz_components, kmeans, n_resamples=3)
    
    results.append({
        'K': k,
        'algorithm': 'K-Means (FAMD)',
        'silhouette': silhouette,
        'davies_bouldin': db_score,
        'stability': stability
    })

metrics_df = pd.DataFrame(results)
metrics_df.to_csv('outputs/metrics_validation_behavioral.csv', index=False)

# Select optimal K
best_idx = metrics_df.loc[metrics_df['silhouette'].idxmax()]
optimal_k = int(best_idx['K'])
print(f"✓ Optimal K for quiz subset: {optimal_k}")

# Final clustering
final_kmeans_quiz = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
quiz_subset_df['quiz_cluster_id'] = final_kmeans_quiz.fit_predict(retained_quiz_components)

# --- 5. Full-base Analysis (BEHAVIORAL FOCUS) ---
print("\n5. Analyzing full customer base (behavioral patterns only)...")

# Get available features (NO REVENUE METRICS!)
available_behavioral_nums = [col for col in behavioral_numeric_cols if col in df.columns]
available_health_flags = [col for col in health_binary_cols if col in df.columns]
available_marketing_cats = [col for col in marketing_categorical_cols if col in df.columns]

print(f"Full-base features: {len(available_behavioral_nums)} behavioral + {len(available_health_flags)} health + {len(available_marketing_cats)} marketing")
print("🚫 EXCLUDED: gross_ltv, net_ltv, avg_order_value (too obvious)")

# Prepare features
full_base_features = df[available_behavioral_nums + available_health_flags + available_marketing_cats].copy()

# Imputation
for col in available_behavioral_nums:
    full_base_features[col] = full_base_features[col].fillna(full_base_features[col].median())

for col in available_health_flags:
    full_base_features[col] = full_base_features[col].fillna(0)

for col in available_marketing_cats:
    full_base_features[col] = full_base_features[col].fillna('missing').astype(str)

# Target encoding for categoricals
if available_marketing_cats:
    target_encoder = TargetEncoder(smoothing=10.0)
    cat_features = full_base_features[available_marketing_cats]
    target_values = df['order_count']  # Use order_count as proxy (behavior, not value)
    target_encoder.fit(cat_features, target_values)
    cat_encoded = target_encoder.transform(cat_features)
    
    all_features = pd.concat([
        full_base_features[available_behavioral_nums + available_health_flags],
        cat_encoded
    ], axis=1)
else:
    all_features = full_base_features[available_behavioral_nums + available_health_flags]

# Standardization
scaler = StandardScaler()
full_base_scaled = scaler.fit_transform(all_features)

print(f"Full base features shape: {full_base_scaled.shape}")

# PCA with Varimax rotation
pca = PCA(random_state=42)
pca.fit(full_base_scaled)

try:
    from factor_analyzer.rotator import Rotator
    rot = Rotator(method='varimax')
    rotated_components = rot.fit_transform(pca.components_.T)
    print("✓ Varimax rotation applied")
except ImportError:
    print("⚠ factor_analyzer not available, using unrotated components")
    rotated_components = pca.components_.T

# Retain components for 70% variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_comps_pca = np.argmax(cumvar >= 0.70) + 1
n_comps_pca = min(n_comps_pca, 8)  # Cap for memory

full_base_components = pca.transform(full_base_scaled)[:, :n_comps_pca]
full_base_components_df = pd.DataFrame(full_base_components, index=df.index)

print(f"✓ PCA complete. Using {n_comps_pca} components explaining {cumvar[n_comps_pca-1]*100:.1f}% variance")

# Full-base clustering
print("Clustering full customer base...")
full_results = []

for k in k_range:
    print(f"  Testing K={k}...")
    
    kmeans_full = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels_full = kmeans_full.fit_predict(full_base_components_df)
    
    silhouette_full = silhouette_score(full_base_components_df, labels_full)
    db_full = davies_bouldin_score(full_base_components_df, labels_full)
    stability_full = simple_stability_score(full_base_components_df, kmeans_full, n_resamples=3)
    
    full_results.append({
        'K': k,
        'algorithm': 'K-Means (PCA)',
        'silhouette': silhouette_full,
        'davies_bouldin': db_full,
        'stability': stability_full
    })

# Select optimal K for full base
full_metrics_df = pd.DataFrame(full_results)
best_full_idx = full_metrics_df.loc[full_metrics_df['silhouette'].idxmax()]
optimal_full_k = int(best_full_idx['K'])

print(f"✓ Optimal K for full base: {optimal_full_k}")

# Final full-base clustering
final_full_kmeans = KMeans(n_clusters=optimal_full_k, random_state=42, n_init=10)
df['behavioral_cluster_id'] = final_full_kmeans.fit_predict(full_base_components_df)

# --- 6. Bridge Modeling ---
print("\n6. Building bridge model...")

quiz_customers_with_clusters = df[df.index.isin(quiz_subset_df.index)].copy()

if len(quiz_customers_with_clusters) > 100:
    try:
        # Use behavioral features (not revenue) to predict clusters
        bridge_features = quiz_customers_with_clusters[available_behavioral_nums[:6]]  # Top 6 behavioral
        bridge_target = quiz_customers_with_clusters['behavioral_cluster_id']
        
        # Imputation
        for col in bridge_features.columns:
            bridge_features[col] = bridge_features[col].fillna(bridge_features[col].median())
        
        X_train, X_test, y_train, y_test = train_test_split(
            bridge_features, bridge_target, test_size=0.3, random_state=42
        )
        
        class_counts = bridge_target.value_counts()
        n_classes = len(class_counts)
        print(f"Bridge data: {len(bridge_features)} samples, {n_classes} classes")
        print("Class distribution:", class_counts.to_dict())
        
        if n_classes > 2:
            lgb_clf = lgb.LGBMClassifier(
                objective='multiclass', num_class=n_classes, random_state=42, verbose=-1
            )
        else:
            lgb_clf = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1)
        
        lgb_clf.fit(X_train, y_train)
        y_pred = lgb_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Bridge model accuracy: {accuracy:.2%}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': lgb_clf.booster_.feature_name(),
            'importance': lgb_clf.booster_.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False)
        
        print("Top behavioral predictors:")
        print(feature_importance.head().to_string(index=False))
        
    except Exception as e:
        print(f"Bridge modeling failed: {e}")
else:
    print("Insufficient data for bridge modeling")

# --- 7. POST-HOC PROFILING (Now we add back the revenue for insights!) ---
print("\n7. Profiling behavioral clusters...")

# Comprehensive profiling including revenue metrics we excluded from clustering
profile_metrics = {
    # Behavioral (what drove the clustering)
    'customers': ('email_key', 'count'),
    'avg_orders': ('order_count', 'mean'),
    'avg_recency_days': ('days_since_last_order', 'mean'),
    'refund_rate': ('refund_ratio', 'mean'),
    
    # Revenue (POST-HOC insights)
    'median_ltv': ('net_ltv', 'median'),
    'mean_ltv': ('net_ltv', 'mean'),
    'median_aov': ('avg_order_value', 'median'),
    
    # Health patterns (the interesting stuff!)
    'pct_quiz_takers': ('quiz_taker', lambda x: x.map({'No': 0, 'Yes': 1}).mean() if x.dtype == 'object' else x.mean()),
    'pct_stress_mental': ('stress_mental_flag', 'mean'),
    'pct_stress_digestive': ('stress_digestion_flag', 'mean'),
    'pct_bloating': ('sx_bloating', 'mean'),
    'pct_anxiety': ('sx_anxiety', 'mean'),
    'pct_inflammatory': ('inflammatory_condition', 'mean'),
    
    # Marketing patterns
    'pct_discount_users': ('ancestor_discount_code', lambda x: (x != 'missing').mean()),
}

# Calculate profiles
cluster_profiles = []
for cluster_id in sorted(df['behavioral_cluster_id'].unique()):
    cluster_data = df[df['behavioral_cluster_id'] == cluster_id]
    
    profile = {'cluster_id': cluster_id}
    for metric_name, (col, func) in profile_metrics.items():
        if col in df.columns:
            if callable(func):
                profile[metric_name] = func(cluster_data[col])
            else:
                profile[metric_name] = cluster_data[col].agg(func)
        else:
            profile[metric_name] = np.nan
    
    cluster_profiles.append(profile)

profiles_df = pd.DataFrame(cluster_profiles)

# Calculate LTV lifts
global_mean_ltv = df['net_ltv'].mean()
profiles_df['ltv_lift_pct'] = ((profiles_df['mean_ltv'] / global_mean_ltv) - 1) * 100

print("\n=== BEHAVIORAL CLUSTER PROFILES ===")
print(profiles_df.round(2).to_string(index=False))

# --- 8. Export Results ---
print("\n8. Exporting results...")

# Export segmented customers
final_output = df[['email_key', 'behavioral_cluster_id']].copy()
quiz_clusters = quiz_subset_df[['quiz_cluster_id']].copy()
quiz_clusters.columns = ['Q_subcluster']
final_output = final_output.join(quiz_clusters, how='left')

final_output.to_csv('outputs/segmented_customers_behavioral.csv', index=False)
print("✓ Behavioral segments saved")

# Export cluster profiles
profiles_df.to_csv('outputs/cluster_profiles_behavioral.csv', index=False)
print("✓ Cluster profiles saved")

# Export PCA loadings for interpretability
feature_names = list(all_features.columns)
pca_loadings = pd.DataFrame(
    pca.components_[:n_comps_pca].T,
    columns=[f'PC{i+1}' for i in range(n_comps_pca)],
    index=feature_names
)
pca_loadings.to_csv('outputs/pca_loadings_behavioral.csv')
print("✓ PCA loadings saved")

print("\n=== BEHAVIORAL SEGMENTATION COMPLETE ===")
print("🎯 Key insights:")
print("- Segments are driven by BEHAVIOR & HEALTH, not just spending")
print("- Revenue metrics used only for post-hoc profiling")
print("- Check cluster_profiles_behavioral.csv for actionable insights!")
print("- PCA loadings show which behaviors drive each component")

