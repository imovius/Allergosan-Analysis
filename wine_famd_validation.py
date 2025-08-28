"""
Wine FAMD Validation - Python vs R FactoMineR
==============================================
Replicating the STHDA wine example to validate prince vs FactoMineR implementation
Source: https://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/115-famd-factor-analysis-of-mixed-data-in-r-essentials/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince import FAMD
import os

# ===================== SETUP =====================
OUTPUT_DIR = "wine_validation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== LOAD WINE DATASET =====================
print("Loading wine dataset...")
try:
    # Load the wine dataset from CSV
    wine_full = pd.read_csv("wikipedia_validation/wine.csv")
    print(f"✓ Wine dataset loaded: {wine_full.shape}")
    
    # Show first few rows
    print("\nFirst 5 rows of wine dataset:")
    print(wine_full.head())
    
    # Show column names and types
    print(f"\nColumn names ({len(wine_full.columns)}):")
    for i, col in enumerate(wine_full.columns):
        print(f"{i+1:2d}. {col:<20} ({wine_full[col].dtype})")
    
except FileNotFoundError:
    print("ERROR: wine.csv not found in wikipedia_validation/")
    print("Please make sure the wine dataset is available.")
    exit(1)

# ===================== EXTRACT FAMD SUBSET =====================
# Following STHDA example: columns 1, 2, 16, 22, 29, 28, 30, 31 (1-indexed)
# Convert to 0-indexed: 0, 1, 15, 21, 28, 27, 29, 30

selected_indices = [0, 1, 15, 21, 28, 27, 29, 30]  # 0-indexed
try:
    wine_subset = wine_full.iloc[:, selected_indices]
    print(f"\n✓ FAMD subset extracted: {wine_subset.shape}")
    
    # Expected columns from STHDA:
    # Label, Soil, Plante, Acidity, Harmony, Intensity, Overall.quality, Typical
    print("\nFAMD subset columns:")
    for i, col in enumerate(wine_subset.columns):
        print(f"{i+1}. {col:<20} ({wine_subset[col].dtype})")
    
    print("\nFirst 4 rows of FAMD subset:")
    print(wine_subset.head(4))
    
except IndexError as e:
    print(f"ERROR: Cannot extract subset - {e}")
    print(f"Dataset has {wine_full.shape[1]} columns, but trying to access indices: {selected_indices}")
    exit(1)

# ===================== DATA PREPROCESSING =====================
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Convert categorical variables to category type
# First two columns should be categorical (Label, Soil)
categorical_cols = wine_subset.columns[:2].tolist()
numerical_cols = wine_subset.columns[2:].tolist()

print(f"Categorical variables: {categorical_cols}")
print(f"Numerical variables: {numerical_cols}")

# Convert to proper types
wine_processed = wine_subset.copy()
for col in categorical_cols:
    wine_processed[col] = wine_processed[col].astype('category')
    print(f"✓ {col}: {wine_processed[col].nunique()} categories - {wine_processed[col].cat.categories.tolist()}")

print(f"\nProcessed dataset structure:")
print(wine_processed.dtypes)

# Save processed dataset
wine_processed.to_csv(os.path.join(OUTPUT_DIR, "wine_famd_subset_processed.csv"), index=False)

# ===================== PERFORM FAMD WITH PRINCE =====================
print("\n" + "="*60)
print("FAMD ANALYSIS WITH PYTHON PRINCE")
print("="*60)

# Run FAMD with different configurations to match R
configs = [
    {"n_components": 5, "engine": "sklearn", "random_state": None},
    {"n_components": 8, "engine": "sklearn", "random_state": None},  # All components
    {"n_components": 5, "engine": "sklearn", "random_state": 42},
]

best_config = None
best_results = None

for i, config in enumerate(configs):
    print(f"\n--- Configuration {i+1}: {config} ---")
    
    try:
        famd = FAMD(**config)
        famd.fit(wine_processed)

        
        # Extract results
        eigenvalues = famd.eigenvalues_
        explained_variance = famd.percentage_of_variance_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Manual calculation like in Allergosan script
        eigs = np.asarray(eigenvalues, dtype=float)
        manual_expl = eigs / eigs.sum()
        manual_cum = np.cumsum(manual_expl)
        
        print(f"✓ FAMD completed successfully")
        print(f"Eigenvalues (first 5): {eigenvalues[:5].round(3)}")
        print(f"Sum of eigenvalues: {eigs.sum():.3f}")
        print(f"Manual explained variance (first 5): {manual_expl[:5]*100}")
        print(f"Prince explained variance (first 5): {explained_variance[:5]}")
        print(f"Manual cumulative variance (first 5): {manual_cum[:5]*100}")
        print(f"Prince cumulative variance (first 5): {cumulative_variance[:5]}")
        
        # Get coordinates
        individual_coords = famd.row_coordinates(wine_processed)
        variable_coords = famd.column_coordinates_ if hasattr(famd, 'column_coordinates_') else None
        
        print(f"Individual coordinates shape: {individual_coords.shape}")
        if variable_coords is not None:
            print(f"Variable coordinates shape: {variable_coords.shape}")
        
        # Save this configuration as best
        if best_config is None:
            best_config = config.copy()
            best_results = {
                'famd': famd,
                'eigenvalues': eigenvalues,
                'explained_variance': explained_variance,
                'cumulative_variance': cumulative_variance,
                'individual_coords': individual_coords,
                'variable_coords': variable_coords
            }
        
    except Exception as e:
        print(f"✗ Configuration {i+1} failed: {e}")

if best_results is None:
    print("ERROR: No FAMD configuration worked!")
    exit(1)

print(f"\n✓ Using best configuration: {best_config}")

# ===================== SAVE DETAILED RESULTS =====================
print("\n" + "="*60)
print("SAVING PYTHON RESULTS")
print("="*60)

# Save eigenvalues and variance explained
variance_df = pd.DataFrame({
    'component': range(1, len(best_results['explained_variance']) + 1),
    'eigenvalue': best_results['eigenvalues'],
    'explained_variance_pct': best_results['explained_variance'],
    'cumulative_variance_pct': best_results['cumulative_variance']
})
variance_df.to_csv(os.path.join(OUTPUT_DIR, "wine_famd_variance_python.csv"), index=False)

# Save individual coordinates
individual_coords_df = pd.DataFrame(
    best_results['individual_coords'].values[:, :5],  # First 5 dimensions
    columns=[f'Dim{i+1}' for i in range(5)],
    index=wine_processed.index
)
individual_coords_df.to_csv(os.path.join(OUTPUT_DIR, "wine_individual_coords_python.csv"))

# Save variable coordinates if available
if best_results['variable_coords'] is not None:
    variable_coords_df = pd.DataFrame(
        best_results['variable_coords'].iloc[:, :5],  # First 5 dimensions
        columns=[f'Dim{i+1}' for i in range(5)],
        index=wine_processed.columns
    )
    variable_coords_df.to_csv(os.path.join(OUTPUT_DIR, "wine_variable_coords_python.csv"))

print(f"✓ Saved variance results: wine_famd_variance_python.csv")
print(f"✓ Saved individual coordinates: wine_individual_coords_python.csv")
if best_results['variable_coords'] is not None:
    print(f"✓ Saved variable coordinates: wine_variable_coords_python.csv")

# ===================== CREATE VISUALIZATIONS =====================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# 1. Scree plot (eigenvalues)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(best_results['eigenvalues'][:8]) + 1), 
         best_results['explained_variance'][:8], 
         'bo-', linewidth=2, markersize=8)
plt.xlabel('Principal Components')
plt.ylabel('Percentage of Variance Explained (%)')
plt.title('FAMD Scree Plot - Wine Dataset (Python Prince)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 9))
for i, val in enumerate(best_results['explained_variance'][:8]):
    plt.annotate(f'{val:.1f}%', (i+1, val), textcoords="offset points", xytext=(0,10), ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "wine_scree_plot_python.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Individual plot (first 2 dimensions)
plt.figure(figsize=(10, 8))
coords = best_results['individual_coords'].iloc[:, :2]
plt.scatter(coords.iloc[:, 0], coords.iloc[:, 1], s=100, alpha=0.7, edgecolors='black')

# Label some points
for i in range(min(10, len(coords))):  # Label first 10 points to avoid clutter
    plt.annotate(f'i{i+1}', (coords.iloc[i, 0], coords.iloc[i, 1]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel(f'Dim 1 ({best_results["explained_variance"][0]:.1f}%)')
plt.ylabel(f'Dim 2 ({best_results["explained_variance"][1]:.1f}%)')
plt.title('FAMD Individual Plot - Wine Dataset (Python Prince)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "wine_individuals_plot_python.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. Variable plot (if available)
if best_results['variable_coords'] is not None:
    plt.figure(figsize=(10, 8))
    var_coords = best_results['variable_coords'].iloc[:, :2]
    
    # Plot variables
    for i, var in enumerate(wine_processed.columns):
        x, y = var_coords.iloc[i, 0], var_coords.iloc[i, 1]
        
        # Different colors for categorical vs numerical
        color = 'red' if var in categorical_cols else 'blue'
        marker = 's' if var in categorical_cols else 'o'
        
        plt.scatter(x, y, c=color, marker=marker, s=100, alpha=0.7, edgecolors='black')
        plt.annotate(var, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel(f'Dim 1 ({best_results["explained_variance"][0]:.1f}%)')
    plt.ylabel(f'Dim 2 ({best_results["explained_variance"][1]:.1f}%)')
    plt.title('FAMD Variable Plot - Wine Dataset (Python Prince)\nRed=Categorical, Blue=Numerical')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "wine_variables_plot_python.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ===================== SUMMARY REPORT =====================
summary_report = f"""
Wine FAMD Validation Results - Python Prince Implementation
===========================================================

Dataset Information:
- Total observations: {wine_processed.shape[0]}
- Total variables: {wine_processed.shape[1]}
- Categorical variables: {len(categorical_cols)} ({', '.join(categorical_cols)})
- Numerical variables: {len(numerical_cols)} ({', '.join(numerical_cols)})

FAMD Configuration Used:
{best_config}

Results (First 5 Components):
- Eigenvalues: {best_results['eigenvalues'][:5].round(3)}
- Explained Variance (%): {best_results['explained_variance'][:5].round(1)}
- Cumulative Variance (%): {best_results['cumulative_variance'][:5].round(1)}

Files Generated:
✓ wine_famd_subset_processed.csv (preprocessed dataset)
✓ wine_famd_variance_python.csv (eigenvalues and variance)
✓ wine_individual_coords_python.csv (individual coordinates)
✓ wine_variable_coords_python.csv (variable coordinates)
✓ wine_scree_plot_python.png (variance plot)
✓ wine_individuals_plot_python.png (individual factor map)
✓ wine_variables_plot_python.png (variable factor map)

Next Steps:
1. Compare these results with R FactoMineR output
2. Validate if Python prince matches R implementation
3. Use findings to validate Allergosan FAMD analysis

Reference: {best_config}
Source: https://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/115-famd-factor-analysis-of-mixed-data-in-r-essentials/
"""

with open(os.path.join(OUTPUT_DIR, "wine_validation_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_report)

print(summary_report)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("\n" + "="*60)
print("WINE FAMD VALIDATION COMPLETE")
print("="*60)
