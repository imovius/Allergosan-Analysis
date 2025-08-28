#!/usr/bin/env python3
"""
Customer Segmentation Analysis Script
Based on the MEGA PROMPT requirements for Allergosan project

This script implements a two-stage clustering approach:
1. Quiz-subset clustering using FAMD
2. Full-base clustering using PCA
3. Bridge modeling to connect the two approaches
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
import prince
import lightgbm as lgb
from factor_analyzer.rotator import Rotator


print("=== ALLERGOSAN CUSTOMER SEGMENTATION ANALYSIS ===")
print("Starting analysis...")

# --- 1. Data Loading and Schema Audit ---
print("\n1. Loading data and performing schema audit...")

try:
    df = pd.read_csv('raw_data_v2.csv')
    print(f" Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'raw_data_v2.csv' not found.")
    exit()

# Schema catalogue
date_cols = ['first_order_date','first_order_date', 'quiz_date']
num_cols = [
    'days_since_last_order', 'order_count', 'refund_count','avg_days_between_orders',
    'gross_ltv','refund_amt', 'net_ltv', 'avg_order_value',
    'symptom_count','total_cogs','shipping_collected',
    'shipping_spend','avg_order_value','refund_ratio',
]
binary_cols = [
    'is_male','is_pregnant','in_third_trimester_flag','probiotic_for_child_flag',
    'stress_mental_flag','stress_physical_flag','stress_digestion_flag','high_stress',
    'recent_abx_flag','took_antibiotics_recently_flag','stomach_flu_flag',
    'digestive_meds_flag','inflammatory_condition','food_intolerance',
    'sx_bloating','sx_reflux','sx_constipation','sx_diarrhea',
    'sx_anxiety','sx_brain_fog','sx_uti','sx_acne','quiz_taker'
]
cat_low_cols = ['first_sku','affiliate_segment',
'bm_pattern','gi_symptom_cat','primary_goal','quiz_result']
text_ignore = ['ancestor_discount_code']

# Schema audit
schema_audit = pd.DataFrame({
    'dtype': df.dtypes,
    'cardinality': df.nunique(),
    'missing_values': df.isnull().sum(),
    'missing_percentage': (df.isnull().sum() / len(df)) * 100
})

# Create outputs directory
import os
os.makedirs('outputs', exist_ok=True)
schema_audit.to_csv('outputs/schema_audit.csv')
print(" Schema audit saved to outputs/schema_audit.csv")

# --- 2. Preprocessing Pipeline ---
print("\n2. Setting up preprocessing pipeline...")

# Convert date columns
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Define preprocessors
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_low_cols)
    ],
    remainder='passthrough'
)

# Save preprocessing pipeline
joblib.dump(preprocessor, 'outputs/preprocessing_pipeline.joblib')
print(" Preprocessing pipeline saved")

# --- 3. Quiz-subset Analysis ---
print("\n3. Analyzing quiz-taking subset...")

# Create quiz subset
quiz_subset_df = df.dropna(subset=['quiz_date']).copy()
print(f"Quiz subset shape: {quiz_subset_df.shape}")
print(f"Quiz participation rate: {len(quiz_subset_df)/len(df)*100:.1f}%")

# Prepare quiz features - only use columns that actually exist
available_num_cols = [col for col in num_cols if col in quiz_subset_df.columns]
available_binary_cols = [col for col in binary_cols if col in quiz_subset_df.columns]
available_cat_cols = [col for col in cat_low_cols if col in quiz_subset_df.columns]

all_available_cols = available_num_cols + available_binary_cols + available_cat_cols
quiz_features_df = quiz_subset_df[all_available_cols].copy()

print(f"Available features for FAMD: {len(all_available_cols)} out of {len(num_cols + binary_cols + cat_low_cols)} defined")

# Simple imputation for FAMD
for col in available_num_cols:
    quiz_features_df[col] = quiz_features_df[col].fillna(quiz_features_df[col].median())

for col in available_binary_cols:
    quiz_features_df[col] = quiz_features_df[col].fillna(0)

for col in available_cat_cols:
    quiz_features_df[col] = quiz_features_df[col].fillna('missing')

# Check for duplicate columns and remove them
quiz_features_df = quiz_features_df.loc[:, ~quiz_features_df.columns.duplicated()]
print(f"Quiz features after removing duplicates: {quiz_features_df.shape[1]} columns")

# Run FAMD (memory-efficient version)
print("Running FAMD on quiz subset...")
try:
    # Limit components to prevent memory issues
    max_components = min(15, len(quiz_features_df.columns))
    famd = prince.FAMD(n_components=max_components, n_iter=3, random_state=42)
    famd = famd.fit(quiz_features_df)
    
    # Determine components to retain
    # The prince library has cumulative_percentage_of_variance_ attribute
    cumulative_variance = famd.cumulative_percentage_of_variance_
    
    # Find where cumulative variance reaches 70%
    n_components_to_retain = np.argmax(np.array(cumulative_variance) >= 70.0) + 1
    n_components_to_retain = min(n_components_to_retain, 8)  # Cap at 8 for memory
    
    quiz_famd_components = famd.transform(quiz_features_df)
    retained_quiz_components = quiz_famd_components.iloc[:, :n_components_to_retain]
    
    print(f" FAMD complete. Retained {n_components_to_retain} components")
    
except Exception as e:
    print(f"✗ FAMD failed: {e}")
    print("Falling back to simple PCA on numeric features only...")
    
    # Fallback to PCA on numeric features only
    numeric_only_df = quiz_features_df[available_num_cols].copy()
    
    # Standardize numeric features
    scaler_quiz = StandardScaler()
    quiz_scaled = scaler_quiz.fit_transform(numeric_only_df)
    
    # Apply PCA
    max_pca_components = min(8, len(available_num_cols))
    pca_quiz = PCA(n_components=max_pca_components, random_state=42)
    retained_quiz_components = pd.DataFrame(
        pca_quiz.fit_transform(quiz_scaled),
        index=quiz_features_df.index
    )
    n_components_to_retain = retained_quiz_components.shape[1]
    print(f" PCA fallback complete. Using {n_components_to_retain} components")

# --- 4. Quiz-subset Clustering (Memory Optimized) ---
print("\n4. Clustering quiz subset...")

def simple_stability_score(data, clustering_algorithm, n_resamples=10):
    """Simplified stability score to prevent memory issues"""
    scores = []
    
    for i in range(n_resamples):
        # Sample 70% of data
        sample_size = int(0.7 * len(data))
        sample_idx = np.random.choice(len(data), sample_size, replace=False)
        
        sample1_idx = np.random.choice(sample_idx, int(0.8 * len(sample_idx)), replace=False)
        sample2_idx = np.random.choice(sample_idx, int(0.8 * len(sample_idx)), replace=False)
        
        if len(sample1_idx) > 10 and len(sample2_idx) > 10:
            labels1 = clustering_algorithm.fit_predict(data.iloc[sample1_idx])
            
            # Create new instance for second fit
            clustering_algorithm2 = type(clustering_algorithm)(**clustering_algorithm.get_params())
            labels2 = clustering_algorithm2.fit_predict(data.iloc[sample2_idx])
            
            # Find overlap
            common_idx = np.intersect1d(sample1_idx, sample2_idx)
            if len(common_idx) > 5:
                overlap1 = []
                overlap2 = []
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
k_range = range(3, 7)  # Reduced range for memory
results = []

data_for_clustering = retained_quiz_components

for k in k_range:
    print(f"  Testing K={k}...")
    
    # K-Means only (skip Agglomerative to save memory)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)  # Reduced n_init
    
    # Calculate metrics
    labels = kmeans.fit_predict(data_for_clustering)
    silhouette = silhouette_score(data_for_clustering, labels)
    db_score = davies_bouldin_score(data_for_clustering, labels)
    
    # Simplified stability (reduced resamples)
    stability = simple_stability_score(data_for_clustering, kmeans, n_resamples=5)
    
    results.append({
        'K': k,
        'algorithm': 'K-Means (FAMD)',
        'silhouette': silhouette,
        'davies_bouldin': db_score,
        'stability': stability
    })

# Save metrics
metrics_df = pd.DataFrame(results)
metrics_df.to_csv('outputs/metrics_validation.csv', index=False)
print(" Quiz subset clustering metrics saved")

# Select optimal K
best_idx = metrics_df.loc[metrics_df['silhouette'].idxmax()]
optimal_k = int(best_idx['K'])
print(f" Optimal K for quiz subset: {optimal_k}")

# Final clustering
final_kmeans_quiz = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
quiz_subset_df['quiz_cluster_id'] = final_kmeans_quiz.fit_predict(data_for_clustering)

# --- 5. Full-base Analysis ---
print("\n5. Analyzing full customer base...")

# Use comprehensive behavioral features (not just 5 basic ones!)
behavioral_numeric_cols = [
    'days_since_last_order', 'order_count', 'refund_count', 'avg_days_between_orders',
    'gross_ltv', 'refund_amt', 'net_ltv', 'total_cogs', 'shipping_collected', 
    'shipping_spend', 'avg_order_value', 'refund_ratio'
]

behavioral_categorical_cols = [
    'affiliate_segment', 'ancestor_discount_code', 'first_sku'
]

# Get available columns
available_numeric_cols = [col for col in behavioral_numeric_cols if col in df.columns]
available_categorical_cols = [col for col in behavioral_categorical_cols if col in df.columns]
available_behavioral_cols = available_numeric_cols + available_categorical_cols

print(f"Using {len(available_behavioral_cols)} behavioral features: {len(available_numeric_cols)} numeric + {len(available_categorical_cols)} categorical")

if not available_behavioral_cols:
    print("✗ No behavioral columns found")
    exit()

full_base_features = df[available_behavioral_cols].copy()

# Proper preprocessing for mixed data types
# Handle numeric features
for col in available_numeric_cols:
    full_base_features[col] = full_base_features[col].fillna(full_base_features[col].median())

# Handle categorical features  
for col in available_categorical_cols:
    full_base_features[col] = full_base_features[col].fillna('missing').astype(str)

# Target encoding for high-cardinality categoricals to prevent memory explosion
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.global_mean = None
        self.category_means = {}
    
    def fit(self, X, y):
        self.global_mean = y.mean()
        for col in X.columns:
            category_stats = X[[col]].join(pd.Series(y, name='target')).groupby(col)['target'].agg(['mean', 'count'])
            # Smoothed target encoding
            smoothed_means = (category_stats['mean'] * category_stats['count'] + self.global_mean * self.smoothing) / (category_stats['count'] + self.smoothing)
            self.category_means[col] = smoothed_means.to_dict()
        return self
    
    def transform(self, X):
        result = X.copy()
        for col in X.columns:
            result[col] = X[col].map(self.category_means[col]).fillna(self.global_mean)
        return result

# Create preprocessing pipeline with target encoding for categoricals
if available_categorical_cols:
    # Use target encoding for high-cardinality categoricals
    target_encoder = TargetEncoder(smoothing=10.0)
    # Fit target encoder using net_ltv as target
    cat_features = full_base_features[available_categorical_cols]
    target_values = df.loc[full_base_features.index, 'net_ltv']
    target_encoder.fit(cat_features, target_values)
    cat_encoded = target_encoder.transform(cat_features)
    
    # Combine numeric and encoded categorical features
    all_features = pd.concat([
        full_base_features[available_numeric_cols],
        cat_encoded
    ], axis=1)
else:
    all_features = full_base_features[available_numeric_cols]

# Simple standardization
scaler = StandardScaler()
full_base_scaled = scaler.fit_transform(all_features)

print(f"Full base features shape after target encoding: {full_base_scaled.shape}")



# PCA with Varimax rotation for interpretability
pca = PCA(random_state=42)
pca.fit(full_base_scaled)

# Apply Varimax rotation for better interpretability
try:
    from factor_analyzer.rotator import Rotator
    rot = Rotator(method='varimax')
    rotated_components = rot.fit_transform(pca.components_.T)
    print("✓ Varimax rotation applied for interpretability")
except ImportError:
    print("⚠ factor_analyzer not available, using unrotated components")
    rotated_components = pca.components_.T

# Retain components for 70% variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_comps_pca = np.argmax(cumvar >= 0.70) + 1
n_comps_pca = min(n_comps_pca, 5)  # Cap for memory

full_base_components = pca.transform(full_base_scaled)[:, :n_comps_pca]
full_base_components_df = pd.DataFrame(full_base_components, index=df.index)

print(f" PCA complete. Using {n_comps_pca} components")

# Full-base clustering (K-Means only)
print("Clustering full customer base...")
full_results = []

for k in k_range:
    print(f"  Testing K={k}...")
    
    kmeans_full = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels_full = kmeans_full.fit_predict(full_base_components_df)
    
    silhouette_full = silhouette_score(full_base_components_df, labels_full)
    db_full = davies_bouldin_score(full_base_components_df, labels_full)
    
    # Add stability check for full base (reduced resamples)
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

print(f" Optimal K for full base: {optimal_full_k}")

# Final full-base clustering
final_kmeans_full = KMeans(n_clusters=optimal_full_k, random_state=42, n_init=10)
df['final_cluster_id'] = final_kmeans_full.fit_predict(full_base_components_df)

# --- 6. Bridge Modeling ---
print("\n6. Building bridge model...")

# Prepare bridge data
quiz_customers_with_full_clusters = df[df.index.isin(quiz_subset_df.index)].copy()

if len(quiz_customers_with_full_clusters) > 100:  # Only if we have enough data
    try:
        # Features: only use numeric features for bridge model to avoid LightGBM categorical issues
        bridge_features = quiz_customers_with_full_clusters[available_numeric_cols[:8]]  # Use first 8 numeric features
        bridge_target = quiz_customers_with_full_clusters['final_cluster_id']
        
        # Simple imputation
        for col in bridge_features.columns:
            bridge_features[col] = bridge_features[col].fillna(bridge_features[col].median() if bridge_features[col].dtype in ['int64', 'float64'] else 0)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            bridge_features, bridge_target, test_size=0.3, random_state=42
        )
        
        # Check class distribution
        class_counts = bridge_target.value_counts()
        n_classes = len(class_counts)
        print(f"Bridge data: {len(bridge_features)} samples, {n_classes} classes")
        print("Class distribution:", class_counts.to_dict())
        
        # Train LightGBM with proper multiclass configuration
        if n_classes > 2:
            lgb_clf = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=n_classes,
                random_state=42,
                verbose=-1,
                n_estimators=100,
                learning_rate=0.1
            )
        else:
            lgb_clf = lgb.LGBMClassifier(
                objective='binary',
                random_state=42,
                verbose=-1,
                n_estimators=100,
                learning_rate=0.1
            )
        
        lgb_clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = lgb_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f" Bridge model accuracy: {accuracy:.2%}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': lgb_clf.booster_.feature_name(),
            'importance': lgb_clf.booster_.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False)
        
        print("Top 5 most important features:")
        print(feature_importance.head().to_string(index=False))
        
    except Exception as e:
        print(f"Bridge modeling failed: {e}")
else:
    print("Insufficient data for bridge modeling")

# --- 7. Export Results ---
print("\n7. Exporting results...")

# Create final output
final_output = df[['email_key', 'final_cluster_id']].copy()

# Add quiz cluster where available
quiz_clusters = quiz_subset_df[['quiz_cluster_id']].copy()
quiz_clusters.columns = ['Q_subcluster']
final_output = final_output.join(quiz_clusters, how='left')

# Save segmented customers
final_output.to_csv('outputs/segmented_customers.csv', index=False)
print(" Segmented customers saved to outputs/segmented_customers.csv")

# Save PCA loadings for interpretability
feature_names = list(all_features.columns) if 'all_features' in locals() else available_numeric_cols
pca_loadings = pd.DataFrame(
    pca.components_[:n_comps_pca].T,
    columns=[f'PC{i+1}' for i in range(n_comps_pca)],
    index=feature_names
)
pca_loadings.to_csv('outputs/pca_loadings.csv')
print(" PCA loadings saved to outputs/pca_loadings.csv")

# Summary statistics
print("\n=== ANALYSIS SUMMARY ===")
print(f"Total customers: {len(df):,}")
print(f"Quiz participants: {len(quiz_subset_df):,} ({len(quiz_subset_df)/len(df)*100:.1f}%)")
print(f"Behavioral segments (full base): {optimal_full_k}")
print(f"Quiz segments: {optimal_k}")

print("\nFinal cluster distribution:")
cluster_summary = df.groupby('final_cluster_id').agg({
    'net_ltv': ['count', 'mean', 'median'],
    'order_count': 'mean'
}).round(2)
print(cluster_summary)

# Calculate cluster value lifts
global_mean_ltv = df['net_ltv'].mean()
for cluster_id in sorted(df['final_cluster_id'].unique()):
    cluster_ltv = df[df['final_cluster_id'] == cluster_id]['net_ltv'].mean()
    lift = (cluster_ltv / global_mean_ltv - 1) * 100
    print(f"Cluster {cluster_id} LTV lift: {lift:+.1f}%")

print("\n Analysis complete! Check the 'outputs/' directory for all results.")
