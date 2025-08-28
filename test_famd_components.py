"""
Test FAMD with different n_components settings to understand variance calculation
"""

import pandas as pd
import numpy as np
from prince import FAMD

# Load your data
df = pd.read_csv("raw_data_v3.csv")
print(f"Original data shape: {df.shape}")

# Quick preprocessing (simplified)
df_test = df.dropna().head(100)  # Small sample for testing
print(f"Test data shape: {df_test.shape}")

# Select mixed variables (like in your analysis) - MUST include numerical vars
test_vars = ['order_count', 'days_since_last_order', 'symptom_count', 'quiz_result', 'acquisition_code_group', 
             'bm_pattern', 'gi_symptom_cat', 'primary_goal']

df_test_sub = df_test[test_vars].copy()
print(f"Selected variables shape: {df_test_sub.shape}")

# Convert categorical columns
cat_cols = ['quiz_result', 'acquisition_code_group', 'bm_pattern', 'gi_symptom_cat', 'primary_goal']
for col in cat_cols:
    if col in df_test_sub.columns:
        df_test_sub[col] = df_test_sub[col].astype('category')

# Keep numerical columns as float
num_cols = ['order_count', 'days_since_last_order', 'symptom_count']
for col in num_cols:
    if col in df_test_sub.columns:
        df_test_sub[col] = df_test_sub[col].astype('float64')

print(f"After categorical conversion:")
print(df_test_sub.dtypes)

# Test different n_components settings
test_configs = [
    {"name": "Limited (input vars)", "n_components": len(test_vars)},
    {"name": "Double", "n_components": len(test_vars) * 2}, 
    {"name": "Large", "n_components": 50},
    {"name": "Max possible", "n_components": min(df_test_sub.shape[0]-1, 100)}
]

for config in test_configs:
    print(f"\n{'='*50}")
    print(f"Testing: {config['name']} (n_components={config['n_components']})")
    print(f"{'='*50}")
    
    try:
        famd = FAMD(n_components=config['n_components'], random_state=42)
        famd.fit(df_test_sub)
        
        # Get results
        eigenvalues = famd.eigenvalues_
        prince_variance = famd.percentage_of_variance_
        manual_variance = eigenvalues / eigenvalues.sum() * 100
        
        print(f"First 5 eigenvalues: {eigenvalues[:5].round(3)}")
        print(f"Sum of eigenvalues: {eigenvalues.sum():.3f}")
        print(f"Prince variance (first 5): {prince_variance[:5].round(3)}")
        print(f"Manual variance (first 5): {manual_variance[:5].round(3)}")
        print(f"Prince total variance (first 10): {prince_variance[:10].sum():.1f}%")
        print(f"Manual total variance (first 10): {manual_variance[:10].sum():.1f}%")
        
        # Check if they match at full components
        if config['n_components'] >= eigenvalues.shape[0]:
            print(f"Full variance - Prince: {prince_variance.sum():.1f}%")
            print(f"Full variance - Manual: {manual_variance.sum():.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
