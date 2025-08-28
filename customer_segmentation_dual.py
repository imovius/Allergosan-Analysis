#!/usr/bin/env python3
"""
ALLERGOSAN DUAL SEGMENTATION STRATEGY
=====================================
Strategy: Separate clustering for quiz takers vs non-quiz takers
- Quiz takers: Health-based clustering (symptoms, goals, health patterns)
- Non-quiz takers: Behavioral clustering (purchase patterns, affiliate, promo, SKUs)
- Both: RFM pre-stratification within each group
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.base import BaseEstimator, TransformerMixin

print("=== ALLERGOSAN DUAL SEGMENTATION STRATEGY ===")
print("Quiz takers: Health-based clustering")
print("Non-quiz takers: Behavioral clustering")
print("Both: RFM pre-stratification")

# --- 1. Data Loading ---
print("\n1. Loading and splitting data...")

try:
    df = pd.read_csv('raw_data_v2.csv')
    print(f"✓ Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'raw_data_v2.csv' not found.")
    exit()

import os
os.makedirs('outputs', exist_ok=True)

# Split into quiz takers vs non-quiz takers
quiz_takers = df[df['quiz_taker'] == 'yes'].copy()
non_quiz_takers = df[df['quiz_taker'] != 'yes'].copy()

print(f"Quiz takers: {len(quiz_takers)} customers ({len(quiz_takers)/len(df)*100:.1f}%)")
print(f"Non-quiz takers: {len(non_quiz_takers)} customers ({len(non_quiz_takers)/len(df)*100:.1f}%)")

# --- 2. Define Value Bands ---
print("\n2. Creating value band stratification...")

def assign_value_band(ltv):
    if ltv < 200:
        return 'Low'
    elif ltv < 800:
        return 'Core'  
    else:
        return 'High'

# Apply to both groups
quiz_takers['value_band'] = quiz_takers['net_ltv'].apply(assign_value_band)
non_quiz_takers['value_band'] = non_quiz_takers['net_ltv'].apply(assign_value_band)

print("\nQuiz takers by value band:")
print(quiz_takers['value_band'].value_counts().sort_index())

print("\nNon-quiz takers by value band:")
print(non_quiz_takers['value_band'].value_counts().sort_index())

# --- 3. Define Feature Sets ---
print("\n3. Defining feature sets...")

# QUIZ TAKERS: Health-focused features
quiz_features = {
    'timing': ['days_since_last_order', 'avg_days_between_orders', 'order_count'],
    'health_goals': ['primary_goal', 'bm_pattern', 'gi_symptom_cat'],
    'symptoms': ['sx_bloating', 'sx_reflux', 'sx_constipation', 'sx_diarrhea', 'sx_anxiety', 'sx_brain_fog'],
    'stress': ['stress_mental_flag', 'stress_physical_flag', 'stress_digestion_flag'],
    'conditions': ['inflammatory_condition', 'food_intolerance'],
    'affiliate': ['affiliate_segment', 'ancestor_discount_code'],
    'products': ['first_sku']
}

# NON-QUIZ TAKERS: Behavioral-focused features
behavioral_features = {
    'timing': ['days_since_last_order', 'avg_days_between_orders', 'order_count'],
    'engagement': ['refund_count', 'refund_ratio', 'avg_order_value'],
    'marketing': ['affiliate_segment', 'ancestor_discount_code'],
    'products': ['first_sku'],
    'logistics': ['shipping_spend']
}

# --- 4. Simple Label Encoder ---
class SimpleTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=5.0):
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
def cluster_customer_group(data, group_name, feature_config, target_k_range=range(2, 6)):
    """
    Cluster a customer group (quiz/non-quiz) within value bands
    """
    print(f"\n--- Clustering {group_name} ---")
    
    all_clustered_data = []
    all_profiles = []
    
    for band in ['Low', 'Core', 'High']:
        band_data = data[data['value_band'] == band].copy()
        
        if len(band_data) < 10:
            print(f"⚠ {group_name} {band} band: Only {len(band_data)} customers - assigning single cluster")
            band_data['cluster_id'] = f"{group_name}-{band}-0"
            all_clustered_data.append(band_data)
            continue
            
        print(f"\n{group_name} {band} band: {len(band_data)} customers")
        
        # Prepare features
        all_feature_cols = []
        for category, cols in feature_config.items():
            available_cols = [c for c in cols if c in band_data.columns]
            all_feature_cols.extend(available_cols)
        
        # Split numeric and categorical
        numeric_cols = []
        categorical_cols = []
        
        for col in all_feature_cols:
            if band_data[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        if len(numeric_cols) + len(categorical_cols) < 2:
            print(f"⚠ {group_name} {band} band: Insufficient features - single cluster")
            band_data['cluster_id'] = f"{group_name}-{band}-0"
            all_clustered_data.append(band_data)
            continue
        
        # Feature preprocessing
        features_df = band_data[numeric_cols + categorical_cols].copy()
        
        # Handle missing values
        for col in numeric_cols:
            features_df[col] = features_df[col].fillna(features_df[col].median())
        
        for col in categorical_cols:
            features_df[col] = features_df[col].fillna('missing').astype(str)
        
        # Encode categorical features
        if categorical_cols:
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                features_df[col] = le.fit_transform(features_df[col])
                label_encoders[col] = le
        
        # Standardization
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        
        # PCA
        n_components = min(5, scaled_features.shape[1], len(band_data)//10)
        if n_components < 2:
            n_components = min(2, scaled_features.shape[1])
            
        pca = PCA(n_components=n_components, random_state=42)
        pca_features = pca.fit_transform(scaled_features)
        
        print(f"  PCA: {n_components} components, {pca.explained_variance_ratio_.sum()*100:.1f}% variance")
        
        # Clustering grid search
        best_silhouette = -1
        best_k = 2
        best_labels = None
        
        for k in target_k_range:
            if k >= len(band_data):
                continue
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pca_features)
            
            cluster_sizes = pd.Series(labels).value_counts()
            min_size_threshold = max(1, len(band_data) // 50)
            
            if cluster_sizes.min() < min_size_threshold:
                continue
                
            silhouette = silhouette_score(pca_features, labels)
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
                best_labels = labels
            
            print(f"    K={k}: Silhouette={silhouette:.3f}, Min size={cluster_sizes.min()}")
        
        if best_labels is None:
            print(f"  No valid clustering - single cluster")
            band_data['cluster_id'] = f"{group_name}-{band}-0"
        else:
            print(f"  ✓ Selected K={best_k}, Silhouette={best_silhouette:.3f}")
            cluster_labels = [f"{group_name}-{band}-{label}" for label in best_labels]
            band_data['cluster_id'] = cluster_labels
        
        all_clustered_data.append(band_data)
    
    return all_clustered_data

# --- 6. Run Dual Clustering ---
print("\n6. Running dual clustering strategy...")

# Cluster quiz takers (health-focused)
quiz_clustered = cluster_customer_group(quiz_takers, "Quiz", quiz_features)

# Cluster non-quiz takers (behavioral-focused) 
behavioral_clustered = cluster_customer_group(non_quiz_takers, "Behavioral", behavioral_features)

# --- 7. Combine Results ---
print("\n7. Combining results...")

all_segments = []
for segments in quiz_clustered + behavioral_clustered:
    all_segments.append(segments)

final_df = pd.concat(all_segments, ignore_index=True)

print(f"✓ Total customers segmented: {len(final_df)}")
print(f"✓ Unique clusters: {final_df['cluster_id'].nunique()}")

# --- 8. Create Profiles ---
print("\n8. Creating cluster profiles...")

# Define profiling metrics
profile_metrics = {
    'customers': ('email_key', 'count'),
    'avg_orders': ('order_count', 'mean'),
    'median_ltv': ('net_ltv', 'median'),
    'avg_recency_days': ('days_since_last_order', 'mean'),
    'top_affiliate': ('affiliate_segment', lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
    'top_promo': ('ancestor_discount_code', lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
    'top_sku': ('first_sku', lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
    'pct_quiz_takers': ('quiz_taker', lambda x: (x == 'yes').mean()),
}

# Add health metrics for quiz clusters
quiz_health_metrics = {
    'top_primary_goal': ('primary_goal', lambda x: x.mode().iloc[0] if not x.mode().empty else 'missing'),
    'pct_bloating': ('sx_bloating', 'mean'),
    'pct_anxiety': ('sx_anxiety', 'mean'),
    'pct_stress_mental': ('stress_mental_flag', 'mean'),
}

cluster_profiles = []

for cluster_id in sorted(final_df['cluster_id'].unique()):
    cluster_data = final_df[final_df['cluster_id'] == cluster_id]
    
    profile = {'cluster_id': cluster_id}
    
    # Add group type
    if cluster_id.startswith('Quiz'):
        profile['customer_type'] = 'Quiz Taker'
        metrics_to_use = {**profile_metrics, **quiz_health_metrics}
    else:
        profile['customer_type'] = 'Non-Quiz Taker' 
        metrics_to_use = profile_metrics
    
    # Calculate metrics
    for metric_name, (col, func) in metrics_to_use.items():
        if col in cluster_data.columns:
            try:
                if callable(func):
                    profile[metric_name] = func(cluster_data[col])
                else:
                    profile[metric_name] = cluster_data[col].agg(func)
            except:
                profile[metric_name] = 'error'
        else:
            profile[metric_name] = 'missing'
    
    cluster_profiles.append(profile)

profiles_df = pd.DataFrame(cluster_profiles)

# --- 9. Export Results ---
print("\n9. Exporting results...")

# Export segmented customers
final_df[['email_key', 'cluster_id', 'value_band', 'net_ltv', 'quiz_taker']].to_csv(
    'outputs/segmented_customers_dual.csv', index=False
)
print("✓ Segmented customers saved")

# Export cluster profiles
profiles_df.to_csv('outputs/cluster_profiles_dual.csv', index=False)
print("✓ Cluster profiles saved")

# --- 10. Summary ---
print("\n=== DUAL SEGMENTATION COMPLETE ===")
print(f"\n📊 SUMMARY:")
print(f"Total customers: {len(final_df):,}")
print(f"Quiz takers: {len(final_df[final_df['quiz_taker'] == 'yes']):,}")
print(f"Non-quiz takers: {len(final_df[final_df['quiz_taker'] != 'yes']):,}")
print(f"Total clusters: {final_df['cluster_id'].nunique()}")

print(f"\n🎯 CLUSTER BREAKDOWN:")
cluster_summary = profiles_df[['cluster_id', 'customer_type', 'customers', 'median_ltv']].sort_values('customers', ascending=False)
print(cluster_summary.to_string(index=False))

print(f"\n✅ STRATEGY BENEFITS:")
print("- 100% customer coverage")
print("- Health insights for quiz takers")  
print("- Behavioral insights for non-quiz takers")
print("- Value band stratification for both groups")

