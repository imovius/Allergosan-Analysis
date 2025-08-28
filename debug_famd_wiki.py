"""
Debug Wikipedia FAMD Example - Exact Matching
============================================
Testing different FAMD parameters and implementations to match Wikipedia exactly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prince import FAMD
import os

# ===================== EXACT WIKIPEDIA DATASET =====================
wiki_data = pd.DataFrame({
    'k1': [2, 5, 3, 4, 1, 6],           
    'k2': [4.5, 4.5, 1, 1, 1, 1],      
    'k3': [4, 4, 2, 2, 1, 2],          
    'q1': ['A', 'C', 'B', 'B', 'A', 'C'],  
    'q2': ['B', 'B', 'B', 'B', 'A', 'A'],  
    'q3': ['C', 'C', 'B', 'B', 'A', 'A']   
})

# Convert qualitative to categorical
for col in ['q1', 'q2', 'q3']:
    wiki_data[col] = wiki_data[col].astype('category')

print("EXACT Wikipedia Dataset:")
print(wiki_data)
print()

# ===================== TEST DIFFERENT FAMD CONFIGURATIONS =====================

configs_to_test = [
    {"n_components": 2, "copy": True, "check_input": True, "engine": "auto", "random_state": None},
    {"n_components": 2, "copy": True, "check_input": True, "engine": "fbpca", "random_state": 42},
    {"n_components": 2, "copy": True, "check_input": True, "engine": "sklearn", "random_state": 42},
    {"n_components": 6, "copy": True, "check_input": True, "engine": "auto", "random_state": None},  # Full components
]

for i, config in enumerate(configs_to_test):
    print(f"\n{'='*60}")
    print(f"TEST {i+1}: FAMD Configuration")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    try:
        # Try this configuration
        famd = FAMD(**config)
        famd.fit(wiki_data)
        
        # Get results
        eigenvalues = famd.eigenvalues_
        explained_variance = famd.percentage_of_variance_
        individual_coords = famd.row_coordinates(wiki_data)
        
        print(f"\nResults:")
        print(f"Eigenvalues: {eigenvalues[:2]}")
        print(f"Explained variance: {explained_variance[0]:.1f}%, {explained_variance[1]:.1f}%")
        
        print(f"\nIndividual coordinates (first 2 dims):")
        coords_df = pd.DataFrame(individual_coords.iloc[:, :2])
        coords_df.index = [f'i{i+1}' for i in range(6)]
        coords_df.columns = ['Dim1', 'Dim2']
        print(coords_df.round(3))
        
        # Check if this matches Wikipedia
        # Wikipedia Figure 1 shows:
        # i1: (~2, ~0.5), i2: (~2, ~4.5), i3: (~-3, ~-2), i4: (~-1, ~-2), i5: (~4, ~-2), i6: (~2, ~-2)
        print(f"\nWikipedia target positions:")
        print("i1: (~2, ~0.5), i2: (~2, ~4.5), i3: (~-3, ~-2)")
        print("i4: (~-1, ~-2), i5: (~4, ~-2), i6: (~2, ~-2)")
        
        # Plot this configuration
        plt.figure(figsize=(8, 6))
        plt.scatter(individual_coords.iloc[:, 0], individual_coords.iloc[:, 1], 
                   s=100, alpha=0.7, edgecolors='black')
        
        for j in range(len(individual_coords)):
            plt.annotate(f'i{j+1}', 
                        (individual_coords.iloc[j, 0], individual_coords.iloc[j, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=12)
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel(f'Dim 1 ({explained_variance[0]:.1f}%)')
        plt.ylabel(f'Dim 2 ({explained_variance[1]:.1f}%)')
        plt.title(f'Test {i+1}: FAMD Configuration\n{str(config)[:50]}...')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"debug_test_{i+1}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved as: debug_test_{i+1}.png")
        
    except Exception as e:
        print(f"ERROR with config {i+1}: {e}")

# ===================== MANUAL SCALING TEST =====================
print(f"\n{'='*60}")
print("MANUAL SCALING TEST")
print(f"{'='*60}")

# Try manual standardization of quantitative variables
wiki_data_scaled = wiki_data.copy()

# Standardize quantitative variables (mean=0, std=1)
quant_vars = ['k1', 'k2', 'k3']
for var in quant_vars:
    mean_val = wiki_data_scaled[var].mean()
    std_val = wiki_data_scaled[var].std()
    wiki_data_scaled[var] = (wiki_data_scaled[var] - mean_val) / std_val
    print(f"{var}: mean={mean_val:.2f}, std={std_val:.2f}")

print(f"\nScaled quantitative data:")
print(wiki_data_scaled[quant_vars].round(3))

# Run FAMD on scaled data
famd_scaled = FAMD(n_components=2)
famd_scaled.fit(wiki_data_scaled)

individual_coords_scaled = famd_scaled.row_coordinates(wiki_data_scaled)
explained_var_scaled = famd_scaled.percentage_of_variance_

print(f"\nScaled FAMD Results:")
print(f"Explained variance: {explained_var_scaled[0]:.1f}%, {explained_var_scaled[1]:.1f}%")
print(f"Individual coordinates:")
coords_scaled_df = pd.DataFrame(individual_coords_scaled.iloc[:, :2])
coords_scaled_df.index = [f'i{i+1}' for i in range(6)]
coords_scaled_df.columns = ['Dim1', 'Dim2']
print(coords_scaled_df.round(3))

# ===================== CHECK PRINCE LIBRARY VERSION =====================
print(f"\n{'='*60}")
print("LIBRARY INFORMATION")
print(f"{'='*60}")

try:
    import prince
    print(f"Prince version: {prince.__version__ if hasattr(prince, '__version__') else 'Unknown'}")
except:
    print("Could not get prince version")

print("\nNOTE: The Wikipedia example likely uses R's FactoMineR package,")
print("which may have different default behaviors than Python's prince library.")
print("Common differences:")
print("- Different scaling/normalization methods")
print("- Different SVD algorithms") 
print("- Different handling of categorical variables")
print("- Different coordinate sign conventions")
