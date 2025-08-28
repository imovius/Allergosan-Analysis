#!/usr/bin/env python3
"""
Customer Segmentation Analysis - PRE-STRATIFIED BY VALUE
Approach: Split into value bands FIRST, then cluster on stakeholder priorities within each band

Focus Features:
1. Affiliate segments
2. Promo codes  
3. Quiz answers/results
4. SKUs purchased
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin
import prince
import lightgbm as lgb

print("=== ALLERGOSAN PRE-STRATIFIED SEGMENTATION ===")
print("Strategy: Value bands FIRST → Behavioral clustering within each band")
print("Focus: Affiliate segments, promo codes, quiz patterns, SKU preferences")

# --- 1. Data Loading ---
print("\n1. Loading data...")

try:
    df = pd.read_csv('raw_data_v2.csv')
    print(f"✓ Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'raw_data_v2.csv' not found.")
    exit()

import os
os.makedirs('outputs', exist_ok=True)

# --- 2. Define Value Bands ---
print("\n2. Creating value band stratification...")

# Define LTV bands
def assign_value_band(ltv):
    if ltv < 200:
        return 'Low'
    elif ltv < 800:
        return 'Core'  
    else:
        return 'High'

df['value_band'] = df['net_ltv'].apply(assign_value_band)

# Show distribution
band_summary = df.groupby('value_band').agg({
    'email_key': 'count',
    'net_ltv': ['mean', 'median'],
    'order_count': 'mean'
}).round(2)

print("Value band distribution:")
print(band_summary)

# --- 3. Define Stakeholder Priority Features ---
print("\n3. Defining stakeholder priority features...")

# Core behavioral features (light touch - just timing)
timing_features = [
    'days_since_last_order',
    'avg_days_between_orders',
    'order_count'  # Keep this as it's behavioral frequency, not value
]

# STAKEHOLDER PRIORITIES
affiliate_features = ['affiliate_segment']
promo_features = ['ancestor_discount_code']
quiz_features = [
    'quiz_result', 'primary_goal', 'bm_pattern', 'gi_symptom_cat',
    'stress_mental_flag', 'stress_physical_flag', 'stress_digestion_flag',
    'sx_bloating', 'sx_reflux', 'sx_constipation', 'sx_diarrhea',
    'sx_anxiety', 'sx_brain_fog', 'inflammatory_condition', 'food_intolerance'
]
sku_features = ['first_sku']

# Combine all priority features
priority_numeric = [f for f in timing_features if f in df.columns]
priority_categorical = [f for f in (affiliate_features + promo_features + quiz_features + sku_features) if f in df.columns]

print(f"Priority features: {len(priority_numeric)} numeric + {len(priority_categorical)} categorical")
print(f"Categorical features: {priority_categorical}")

# --- 4. Target Encoding for High-Cardinality Categoricals ---
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

# --- 5. Clustering Function ---
def cluster_value_band(band_data, band_name, target_k_range=range(2, 6)):
    """
    Cluster customers within a specific value band using stakeholder priority features
    """
    print(f"\n--- Clustering {band_name} Value Band ({len(band_data)} customers) ---")
    
    # Prepare features
    available_numeric = [f for f in priority_numeric if f in band_data.columns]
    available_categorical = [f for f in priority_categorical if f in band_data.columns]
    
    if len(band_data) < 10:
        print(f"⚠ Only {len(band_data)} customers in {band_name} band - assigning all to single cluster")
        band_data_small = band_data.copy()
        band_data_small['cluster_id'] = f"{band_name}-0"  # Single cluster with band prefix
        return band_data_small, None
    
    # Feature preparation
    features_df = band_data[available_numeric + available_categorical].copy()
    
    # Imputation
    for col in available_numeric:
        features_df[col] = features_df[col].fillna(features_df[col].median())
    
    # Store original categorical data for profiling BEFORE encoding
    original_categoricals = {}
    for col in available_categorical:
        original_categoricals[col] = band_data[col].copy()
        features_df[col] = features_df[col].fillna('missing').astype(str)
    
    # Use Label Encoding instead of Target Encoding to preserve diversity
    if available_categorical:
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        cat_encoded_df = pd.DataFrame(index=features_df.index)
        
        for col in available_categorical:
            le = LabelEncoder()
            cat_encoded_df[col] = le.fit_transform(features_df[col])
            label_encoders[col] = le
        
        # Combine features
        all_features = pd.concat([
            features_df[available_numeric],
            cat_encoded_df
        ], axis=1)
    else:
        all_features = features_df[available_numeric]
        original_categoricals = {}
    
    # Standardization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    
    # PCA for dimensionality reduction
    n_components = min(5, scaled_features.shape[1], len(band_data)//10)  # Conservative
    if n_components < 2:
        n_components = min(2, scaled_features.shape[1])
    
    pca = PCA(n_components=n_components, random_state=42)
    pca_components = pca.fit_transform(scaled_features)
    
    print(f"PCA: {n_components} components explaining {pca.explained_variance_ratio_.sum()*100:.1f}% variance")
    
    # Clustering grid search
    results = []
    for k in target_k_range:
        if k >= len(band_data):  # Can't have more clusters than data points
            continue
            
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pca_components)
        
        # Allow smaller clusters - just need >1% of band size
        cluster_sizes = pd.Series(labels).value_counts()
        min_size_threshold = max(1, len(band_data) // 100)  # At least 1% of band or 1 customer
        if cluster_sizes.min() < min_size_threshold:
            continue
            
        silhouette = silhouette_score(pca_components, labels)
        db_score = davies_bouldin_score(pca_components, labels)
        
        results.append({
            'K': k,
            'silhouette': silhouette,
            'davies_bouldin': db_score,
            'min_cluster_size': cluster_sizes.min()
        })
        
        print(f"  K={k}: Silhouette={silhouette:.3f}, DB={db_score:.3f}, Min size={cluster_sizes.min()}")
    
    if not results:
        print(f"⚠ No valid clustering found for {band_name} band - assigning all to single cluster")
        band_data_single = band_data.copy()
        band_data_single['cluster_id'] = f"{band_name}-0"  # Single cluster with band prefix
        return band_data_single, None
    
    # Select best K based on silhouette score
    best_result = max(results, key=lambda x: x['silhouette'])
    optimal_k = best_result['K']
    
    print(f"✓ Optimal K={optimal_k} for {band_name} band")
    
    # Final clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(pca_components)
    
    # Add cluster labels with band prefix
    band_data_clustered = band_data.copy()
    band_data_clustered['cluster_id'] = [f"{band_name}-{label}" for label in final_labels]
    
    # Create feature importance info
    feature_importance = pd.DataFrame({
        'feature': all_features.columns,
        'pc1_loading': abs(pca.components_[0]),
        'pc2_loading': abs(pca.components_[1]) if n_components > 1 else 0
    }).sort_values('pc1_loading', ascending=False)
    
    # Store original categoricals for profiling
    band_data_clustered._original_categoricals = original_categoricals
    return band_data_clustered, feature_importance

# --- 6. Run Clustering for Each Value Band ---
print("\n6. Running stratified clustering...")

all_clustered_data = []
all_feature_importance = {}

for band in ['Low', 'Core', 'High']:
    band_data = df[df['value_band'] == band].copy()
    clustered_data, feature_importance = cluster_value_band(band_data, band)
    
    all_clustered_data.append(clustered_data)
    if feature_importance is not None:
        all_feature_importance[band] = feature_importance

# Combine all bands
final_df = pd.concat(all_clustered_data, ignore_index=True)

# --- 7. Comprehensive Profiling ---
print("\n7. Creating comprehensive cluster profiles...")

profile_metrics = {
    # Basic info
    'customers': ('email_key', 'count'),
    'value_band': ('value_band', lambda x: x.iloc[0]),  # Should be constant within cluster
    
    # Behavioral patterns  
    'avg_orders': ('order_count', 'mean'),
    'median_ltv': ('net_ltv', 'median'),
    'avg_recency_days': ('days_since_last_order', 'mean'),
    
    # STAKEHOLDER PRIORITIES
    
    # 1. Affiliate segments  
    'top_affiliate_segment': ('affiliate_segment', lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
    'pct_affiliate_defined': ('affiliate_segment', lambda x: (~pd.isna(x) & (x != '') & (x != 'missing')).mean()),
    
    # 2. Promo codes
    'top_promo_code': ('ancestor_discount_code', lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
    'pct_promo_users': ('ancestor_discount_code', lambda x: (~pd.isna(x) & (x != '') & (x != 'missing')).mean()),
    
    # 3. Quiz patterns
    'pct_quiz_takers': ('quiz_result', lambda x: (~pd.isna(x) & (x != '') & (x != 'missing')).mean()),
    'top_primary_goal': ('primary_goal', lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
    'pct_stress_mental': ('stress_mental_flag', 'mean'),
    'pct_stress_digestive': ('stress_digestion_flag', 'mean'),
    'pct_bloating': ('sx_bloating', 'mean'),
    'pct_anxiety': ('sx_anxiety', 'mean'),
    
    # 4. SKU patterns
    'top_first_sku': ('first_sku', lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
}

# Calculate profiles using original categorical data from each band
cluster_profiles = []

# Build profiles per band to access original categorical data
for band_data in all_clustered_data:
    if hasattr(band_data, '_original_categoricals'):
        original_cats = band_data._original_categoricals
    else:
        original_cats = {}
    
    for cluster_id in sorted(band_data['cluster_id'].unique()):
        cluster_data = band_data[band_data['cluster_id'] == cluster_id]
        
        profile = {'cluster_id': cluster_id}
        for metric_name, (col, func) in profile_metrics.items():
            try:
                # Use original categorical data if available
                if col in original_cats:
                    data_to_use = original_cats[col].loc[cluster_data.index]
                else:
                    data_to_use = cluster_data[col] if col in cluster_data.columns else None
                
                if data_to_use is not None:
                    if callable(func):
                        profile[metric_name] = func(data_to_use)
                    else:
                        profile[metric_name] = data_to_use.agg(func)
                else:
                    profile[metric_name] = 'missing_column'
            except Exception as e:
                profile[metric_name] = f'error_{str(e)[:20]}'
        
        cluster_profiles.append(profile)

profiles_df = pd.DataFrame(cluster_profiles)

# --- 8. Export Results ---
print("\n8. Exporting results...")

# Export segmented customers
final_output = final_df[['email_key', 'cluster_id', 'value_band', 'net_ltv']].copy()
final_output.to_csv('outputs/segmented_customers_prestratified.csv', index=False)
print("✓ Segmented customers saved")

# Export cluster profiles
profiles_df.to_csv('outputs/cluster_profiles_prestratified.csv', index=False)
print("✓ Cluster profiles saved")

# Export feature importance for each band
for band, importance_df in all_feature_importance.items():
    importance_df.to_csv(f'outputs/feature_importance_{band}_band.csv', index=False)

print("\n=== PRE-STRATIFIED SEGMENTATION COMPLETE ===")
print("\n📊 CLUSTER SUMMARY:")
print(profiles_df[['cluster_id', 'value_band', 'customers', 'median_ltv', 'top_affiliate_segment', 'top_promo_code', 'pct_quiz_takers']].to_string(index=False))

print("\n🎯 KEY INSIGHTS:")
print("- Each cluster is within a defined value band")
print("- Behavioral differences focus on stakeholder priorities:")
print("  • Affiliate segments")  
print("  • Promo code usage")
print("  • Quiz engagement patterns")
print("  • First SKU preferences")
print("- Check cluster_profiles_prestratified.csv for full details!")
