#!/usr/bin/env python3
"""
Test the signed loadings generation from existing files
"""

import pandas as pd
import numpy as np
import os

# Load existing factor scores
factor_scores = pd.read_csv("quiz_only/outputs/A_withLTV_final_factor_scores.csv")
print(f"Factor scores shape: {factor_scores.shape}")
print(f"Factor columns: {[col for col in factor_scores.columns if col.startswith('Factor_')]}")

# Load existing R² coordinates  
var_coords = pd.read_csv("FAMD_output/A_withLTV_final_variable_coordinates.csv", index_col=0)
print(f"\nVariable coordinates shape: {var_coords.shape}")
print(f"Variables: {list(var_coords.index)}")

# We need the original processed data to calculate correlations
# For now, let's just show what the signed loadings would look like
print(f"\nR² coordinates (always positive):")
print(var_coords.iloc[:5, :3])

print(f"\nSigned loadings would show both positive and negative correlations")
print("Need original variable data to calculate true correlations...")
