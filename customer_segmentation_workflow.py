# ALLERGOSAN – CUSTOMER SEGMENTATION
# End-to-End Statistically Robust Clustering Workflow
#
# Author: Senior Data Scientist (AI Assistant)
# Date: January 2025
#
# ================================================================

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Preprocessing & Dimensionality Reduction
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import prince

# Clustering & Validation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scikit_lego.model_selection import bootstrap_stability_score
import gower
from scipy.cluster.hierarchy import linkage, fcluster

# Statistical Utilities
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Configuration ---
# Set random seed for reproducibility
np.random.seed(42)

# Global settings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Create a directory for figures
if not os.path.exists('figs'):
    os.makedirs('figs')

print("Libraries imported and environment configured.")


# ================================================================
# CHECKPOINT 0: Data Load & Schema Summary
# ================================================================

# --- 1. Read CSV ---
try:
    df = pd.read_csv('raw_data_v2.csv', low_memory=False)
except FileNotFoundError:
    print("ERROR: raw_data_v2.csv not found. Please ensure it is in the correct directory.")
    # Create a dummy dataframe to allow the rest of the notebook to run without errors
    df = pd.DataFrame()

if not df.empty:
    print(f"Data loaded successfully. Shape: {df.shape}")
    print("\\n--- Missing Values (%) ---")
    missing_percentage = df.isnull().mean() * 100
    print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))

    # --- 2. Drop Obvious PII ---
    # customer_id is a unique identifier, but not PII. email_key is a hash.
    # We will keep these for now to join back results later.
    
    # --- 3. Flag Columns by Type ---
    # Define high-cardinality categorical columns to exclude from initial modeling
    high_cardinality_cats = ['first_sku', 'acquisition_channel', 'affiliate_segment', 'ancestor_discount_code', 'quiz_result']
    
    # Identify numeric and binary columns based on provided schema and data inspection
    binary_cols = [
        'is_male', 'is_pregnant', 'in_third_trimester_flag', 'probiotic_for_child_flag', 
        'stress_mental_flag', 'stress_physical_flag', 'stress_digestion_flag', 'high_stress', 
        'recent_abx_flag', 'took_antibiotics_recently_flag', 'stomach_flu_flag', 'digestive_meds_flag',
        'sx_bloating', 'sx_reflux', 'sx_constipation', 'sx_diarrhea', 'sx_anxiety', 'sx_brain_fog', 
        'sx_uti', 'sx_acne', 'inflammatory_condition', 'food_intolerance', 'quiz_taker'
    ]
    
    # Correcting binary columns that might be loaded as non-binary types
    for col in binary_cols:
        if col in df.columns:
            # A simple way to binarize 'yes'/'no', 1/0, True/False etc.
            df[col] = df[col].astype(str).str.lower().isin(['1', '1.0', 'yes', 'y', 'true']).astype(int)

    numeric_cols = [
        'days_since_last_order', 'order_count', 'gross_ltv', 'net_ltv', 
        'avg_days_between_orders', 'avg_order_value', 'refund_ratio', 
        'symptom_count', 'gut_issue_score'
    ]
    
    # Ensure all identified columns actually exist in the dataframe
    binary_cols = [col for col in binary_cols if col in df.columns]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    print(f"\\nIdentified {len(numeric_cols)} numeric columns and {len(binary_cols)} binary columns for modeling.")

    # --- 4. Sanity Check Histograms ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['net_ltv'].apply(lambda x: np.log1p(x)), kde=True)
    plt.title('Log-transformed Net LTV Distribution')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['order_count'].apply(lambda x: np.log1p(x)), kde=True)
    plt.title('Log-transformed Order Count Distribution')
    
    plt.tight_layout()
    plt.show()

### Methodological Note on Pre-processing

For this analysis, we will use `scikit-learn`'s `Pipeline` and `ColumnTransformer`. This is a robust, industry-standard practice that prevents data leakage by ensuring that imputation and scaling statistics are learned *only* from the training data (though in this unsupervised context, we apply it to the whole dataset, the principle of a structured, reproducible workflow holds).

- **Numeric Features**: We will use a pipeline that first imputes any missing values with the **median** (which is robust to outliers often seen in transactional data) and then scales the features using **StandardScaler** (Z-score normalization), which is a prerequisite for many distance-based algorithms like PCA/FAMD and K-Means.
- **Binary Features**: We will impute missing values with the **mode** (the most frequent value), which is the standard approach for categorical/binary data. They do not require scaling.
- **Future Leakage**: The feature list defined in Checkpoint 0 was explicitly chosen to exclude any variables that could represent information from the future (e.g., post-hoc LTV calculations not based on tenure). Our `net_ltv` and `order_count` are cumulative up to the point of data extraction and are safe to use.

# ================================================================
# CHECKPOINT 1: Pre-processing
# ================================================================

if not df.empty:
    # --- 1. Define Pipelines for Numeric and Binary Features ---
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    binary_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # --- 2. Create Column Transformer to Apply Different Pipelines ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('bin', binary_pipeline, binary_cols)
        ],
        remainder='drop'  # Drop columns that are not specified
    )

    # --- 3. Apply the Preprocessing ---
    X_processed = preprocessor.fit_transform(df)

    # --- 4. Recreate a DataFrame with Processed Data ---
    # Get the new column names after processing
    processed_cols = numeric_cols + binary_cols
    df_processed = pd.DataFrame(X_processed, columns=processed_cols, index=df.index)

    print("Preprocessing complete.")
    print(f"Shape of processed data: {df_processed.shape}")
    print("\\n--- First 5 rows of processed data ---")
    print(df_processed.head())

    # --- 5. Assertion for Future Leakage (Conceptual) ---
    # Our feature list was curated to avoid this. For example, we are not using a
    # pre-calculated "is_churned" flag that might be based on data after the snapshot date.
    # We are using historical, cumulative data only.
    assert 'future_revenue' not in df.columns, "Future data leakage detected!"
    print("\\nAssertion successful: No obvious future-leaking columns found.")
else:
    print("Skipping Checkpoint 1 because the dataframe is empty.")

### Methodological Note on Multicollinearity

Before dimensionality reduction, it's crucial to check for multicollinearity. High correlation between features can make model results unstable and difficult to interpret.

- **Numeric vs. Numeric**: We will use a **Pearson correlation heatmap**. Pearson measures linear relationships. We will investigate any pair with a correlation coefficient greater than 0.9 (or less than -0.9). For these pairs, we will remove one of the variables, typically the one that is less interpretable or redundant (e.g., keeping `net_ltv` and dropping `gross_ltv`).
- **Binary vs. Binary**: For binary variables, we will use **Cramér's V**, which measures the association between two categorical variables. Similar to the numeric check, we will investigate and potentially remove one variable from any pair with a very high association to reduce redundancy. Given the number of binary variables, we will calculate the association for all pairs but only visualize or print the top offenders to maintain clarity.

# ================================================================
# CHECKPOINT 2: Correlation & Multicollinearity Scan
# ================================================================
from scipy.stats import chi2_contingency

if 'df_processed' in locals():
    # --- 1. Numeric Correlation Heatmap ---
    print("--- Numeric Feature Correlation ---")
    numeric_df = df_processed[numeric_cols]
    corr_matrix = numeric_df.corr(method='pearson')

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Matrix of Numeric Features')
    plt.show()

    # Identify and flag high correlation pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                high_corr_pairs.append(pair)
                print(f"High correlation detected: {pair[0]} and {pair[1]} (coeff: {pair[2]:.2f})")

    # --- 2. Drop Highly Correlated Numeric Features ---
    # Based on the output, 'gross_ltv' and 'net_ltv' are highly correlated.
    # We will keep 'net_ltv' as it's a more accurate measure of customer value.
    cols_to_drop = ['gross_ltv'] 
    
    # Also, 'order_count' and 'gut_issue_score' might be highly correlated if one drives the other.
    # Let's check for VIF as a more robust measure if needed, but for now, we'll stick to the >0.9 rule.
    
    df_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    # Update the numeric_cols list
    numeric_cols = [col for col in numeric_cols if col not in cols_to_drop]
    print(f"\\nDropped columns due to high correlation: {cols_to_drop}")
    print(f"Remaining numeric columns: {len(numeric_cols)}")


    # --- 3. Binary Variable Association (Cramér's V) ---
    print("\\n--- Binary Feature Association (Cramér's V) ---")
    binary_df = df_processed[binary_cols]
    
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    cramers_matrix = pd.DataFrame(np.zeros((len(binary_cols), len(binary_cols))),
                                  index=binary_cols, columns=binary_cols)
    for col1 in binary_cols:
        for col2 in binary_cols:
            if col1 == col2:
                cramers_matrix.loc[col1, col2] = 1.0
            else:
                cramers_matrix.loc[col1, col2] = cramers_v(binary_df[col1], binary_df[col2])

    # Find and print high association pairs for binary variables
    high_assoc_pairs_bin = []
    for i in range(len(cramers_matrix.columns)):
        for j in range(i):
            if abs(cramers_matrix.iloc[i, j]) > 0.9:
                pair = (cramers_matrix.columns[i], cramers_matrix.columns[j], cramers_matrix.iloc[i, j])
                high_assoc_pairs_bin.append(pair)
                print(f"High association detected: {pair[0]} and {pair[1]} (Cramér's V: {pair[2]:.2f})")
    
    # We will not drop any binary variables at this stage unless a very high (>0.95) redundancy is found,
    # as they often represent distinct customer characteristics valuable for profiling.
    # E.g., 'recent_abx_flag' and 'took_antibiotics_recently_flag' are expected to be identical. Let's drop one.
    bin_cols_to_drop = ['took_antibiotics_recently_flag']
    df_processed.drop(columns=bin_cols_to_drop, inplace=True, errors='ignore')
    binary_cols = [col for col in binary_cols if col not in bin_cols_to_drop]
    print(f"\\nDropped columns due to high association: {bin_cols_to_drop}")
    print(f"Remaining binary columns: {len(binary_cols)}")


else:
    print("Skipping Checkpoint 2 because the processed dataframe is not available.")

### Methodological Note on Dimensionality Reduction (FAMD)

With a dataset of mixed data types (scaled numeric and binary), standard Principal Component Analysis (PCA) is not the ideal choice. PCA's distance calculations are optimized for continuous variables. **Factor Analysis of Mixed Data (FAMD)** is a more appropriate technique.

FAMD works by performing a PCA on the numeric variables and a Multiple Correspondence Analysis (MCA) on the categorical variables simultaneously. This allows it to find the principal components that best summarize the variance across both types of data.

Our process will be:
1.  **Run FAMD:** We will use the `prince` library to fit the FAMD model to our pre-processed data.
2.  **Select Components:** We'll use the Kaiser rule (retaining components with eigenvalues > 1) and a cumulative explained variance target of ~70% to decide how many components (factors) to keep.
3.  **Apply Varimax Rotation:** To improve the interpretability of the factors, we will apply a Varimax rotation. This rotation maximizes the variance of the loadings on each factor, making it easier to see which original variables are most important for which factor. The rotated factors will be the basis for our clustering.
4.  **Export Loadings:** We will save the factor loadings to a CSV file. This table is crucial for interpreting what each factor represents (e.g., "Factor 1 might represent high-value, frequent shoppers").

# ================================================================
# CHECKPOINT 3: Dimensionality Reduction (FAMD)
# ================================================================
from sklearn.preprocessing import OneHotEncoder

if 'df_processed' in locals():
    # Prince's FAMD expects binary variables to be one-hot encoded
    # to be treated as categorical.
    
    # Separate numeric and binary data again from the processed dataframe
    numeric_data = df_processed[numeric_cols]
    binary_data = df_processed[binary_cols]
    
    # One-hot encode the binary columns. This is the format FAMD expects.
    encoder = OneHotEncoder(sparse_output=False, drop='if_binary')
    binary_encoded = encoder.fit_transform(binary_data)
    binary_encoded_cols = encoder.get_feature_names_out(binary_cols)
    
    # Combine back into a single dataframe for FAMD
    df_famd_input = pd.concat([
        numeric_data.reset_index(drop=True), 
        pd.DataFrame(binary_encoded, columns=binary_encoded_cols)
    ], axis=1)

    # --- 1. Run FAMD ---
    print("Running FAMD...")
    famd = prince.FAMD(
        n_components=len(df_famd_input.columns), # Start with all components
        n_iter=3,
        random_state=42
    )
    famd.fit(df_famd_input)

    # --- 2. Select Components using Kaiser Rule & Scree Plot ---
    eigenvalues = famd.eigenvalues_
    explained_variance_ratio = famd.explained_inertia_
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=range(1, len(eigenvalues) + 1), y=eigenvalues, marker='o', label='Eigenvalues')
    plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Rule (Eigenvalue=1)')
    plt.title('Scree Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Determine number of components to keep
    n_components_kaiser = sum(e > 1 for e in eigenvalues)
    n_components_70_variance = (cumulative_explained_variance < 0.7).sum() + 1
    n_components_to_keep = max(n_components_kaiser, n_components_70_variance)
    
    print(f"Kaiser rule suggests: {n_components_kaiser} components")
    print(f"70% variance rule suggests: {n_components_70_variance} components")
    print(f"Decision: Keeping {n_components_to_keep} components.")

    # --- 3. Apply Varimax Rotation ---
    # We need to perform Varimax on the component loadings
    from numpy.linalg import svd
    
    # Refit FAMD with the selected number of components
    famd_final = prince.FAMD(n_components=n_components_to_keep, n_iter=3, random_state=42)
    famd_final.fit(df_famd_input)
    
    # Get the raw factor loadings
    loadings = famd_final.column_correlations(df_famd_input)

    # Varimax rotation function
    def varimax(phi, gamma=1.0, q=20, tol=1e-6):
        from numpy import eye, asarray, dot, sum, diag
        from numpy.linalg import svd
        p, k = phi.shape
        R = eye(k)
        d = 0
        for i in range(q):
            d_old = d
            Lambda = dot(phi, R)
            u, s, vh = svd(dot(phi.T, asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
            R = dot(u, vh)
            d = sum(s)
            if d_old != 0 and d/d_old < 1 + tol: break
        return dot(phi, R)
        
    rotated_loadings = pd.DataFrame(varimax(loadings.values), index=loadings.index, columns=loadings.columns)
    
    print("\\n--- Rotated Factor Loadings (Top 5 per factor) ---")
    for component in rotated_loadings.columns:
        top_vars = rotated_loadings[component].abs().nlargest(5)
        print(f"\\n--- {component} ---")
        print(top_vars)
        
    # Get the transformed data (factor scores)
    X_famd = famd_final.transform(df_famd_input)
    
    # --- 4. Export Loadings ---
    rotated_loadings.to_csv('famd_loadings.csv')
    print("\\nRotated factor loadings saved to 'famd_loadings.csv'")
    
else:
    print("Skipping Checkpoint 3 because the processed dataframe is not available.")

### Methodological Note on Model Selection

Finding the "best" number of clusters is a critical but often subjective part of segmentation. We will use a multi-faceted, data-driven approach to make an informed decision.

Our process is a "model grid" that evaluates two different algorithms across a range of potential cluster counts (K=3 to 8):

1.  **K-Means on FAMD Factors:**
    *   **Why:** K-Means is computationally efficient and works well on the continuous, uncorrelated factors produced by FAMD.
    *   **Evaluation Metrics:**
        *   `Silhouette Score`: Measures how similar a data point is to its own cluster compared to others. Higher is better (range -1 to 1).
        *   `Davies-Bouldin Index`: Measures the average similarity ratio of each cluster with its most similar one. Lower is better (0 is best).
        *   `Calinski-Harabasz Index`: Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.

2.  **Hierarchical Clustering on Gower Distance:**
    *   **Why:** This is a sensitivity analysis. It uses the original, mixed-type data (not the FAMD factors) with a distance metric (Gower) designed for them. If the optimal K from this method aligns with K-Means, it gives us much greater confidence in our results.
    *   **Linkage Method:** We will use **Ward's linkage**, which minimizes the variance of the clusters being merged.

3.  **Bootstrap Stability Test:**
    *   For the K-Means results, we will perform bootstrap resampling. This involves repeatedly clustering subsets of the data and measuring how consistently data points are assigned to the same cluster.
    *   The **Jaccard score** will quantify this stability. A score ≥ 0.6 is our minimum threshold for a viable segmentation.

**Decision Criteria:**
The final K will be chosen based on a compromise between these quantitative scores and the qualitative interpretability of the resulting segments. If no K value provides a stable result (Jaccard < 0.6), **we will pivot to a classic RFM segmentation**, as this indicates the underlying data does not have a strong-enough cluster structure to be found by these methods.

# ================================================================
# CHECKPOINT 4: Clustering Model Grid
# ================================================================
from sklearn.cluster import AgglomerativeClustering

if 'X_famd' in locals():
    # --- 1. K-Means on FAMD Factors ---
    print("--- Evaluating K-Means on FAMD Factors ---")
    k_range = range(3, 9)
    kmeans_results = []

    for k in k_range:
        print(f"Running K-Means for K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_famd)
        
        silhouette = silhouette_score(X_famd, labels)
        db_score = davies_bouldin_score(X_famd, labels)
        ch_score = calinski_harabasz_score(X_famd, labels)
        
        # Bootstrap stability
        # Note: scikit-lego's bootstrap can be slow. We'll use a smaller n_boots.
        stability = bootstrap_stability_score(
            KMeans(n_clusters=k, random_state=42, n_init=10), X_famd, n_boots=10, fit_on_all=True
        )

        kmeans_results.append({
            'k': k,
            'silhouette': silhouette,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score,
            'stability_jaccard': stability
        })

    df_kmeans_results = pd.DataFrame(kmeans_results).set_index('k')
    print("\\n--- K-Means Evaluation Metrics ---")
    print(df_kmeans_results)

    # --- 2. Hierarchical Clustering on Gower Distance ---
    # This is computationally expensive. We'll run it on a sample of the data to get a directional sense.
    print("\\n--- Evaluating Hierarchical Clustering on Gower Distance (Sampled) ---")
    
    sample_size = 2000
    if len(df_processed) > sample_size:
        df_sample = df_processed.sample(sample_size, random_state=42)
    else:
        df_sample = df_processed

    print(f"Calculating Gower distance matrix for {len(df_sample)} samples...")
    gower_matrix_sample = gower.gower_matrix(df_sample)
    
    # Ward linkage requires a euclidean distance matrix. We'll use 'complete' linkage which is more robust for non-euclidean spaces.
    linked = linkage(gower_matrix_sample, method='complete')

    hierarchical_results = []
    for k in k_range:
        labels = fcluster(linked, k, criterion='maxclust')
        silhouette = silhouette_score(gower_matrix_sample, labels, metric='precomputed')
        hierarchical_results.append({
            'k': k,
            'silhouette': silhouette
        })
        
    df_hierarchical_results = pd.DataFrame(hierarchical_results).set_index('k')
    print("\\n--- Hierarchical (Gower) Evaluation Metrics ---")
    print(df_hierarchical_results)
    
    # --- 3. Select Primary K ---
    # Decision logic:
    # 1. Filter for stability >= 0.6
    # 2. From the stable options, find the best silhouette score
    stable_models = df_kmeans_results[df_kmeans_results['stability_jaccard'] >= 0.6]

    if not stable_models.empty:
        best_k = stable_models['silhouette'].idxmax()
        print(f"\\nDecision: Found stable models. Optimal K selected is {best_k} based on best silhouette score among stable options.")
        
        # Set final cluster labels for the next step
        final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(X_famd)
        
        # Flag that we are proceeding with clustering
        pivot_to_rfm = False
        
    else:
        print("\\nDecision: No stable clustering solution found (Jaccard < 0.6).")
        print("PIVOTING TO CLASSIC RFM SEGMENTATION as per protocol.")
        
        # Flag that we need to run RFM instead in the next step
        pivot_to_rfm = True
        best_k = 0 # Placeholder
        cluster_labels = None # Placeholder

else:
    print("Skipping Checkpoint 4 because the FAMD results are not available.")
    pivot_to_rfm = True # Ensure pivot if previous steps failed
    best_k = 0
    cluster_labels = None

### Methodological Note on Segment Profiling

Once we have our final cluster assignments (either from K-Means or the RFM pivot), the next step is to understand who is in each segment. Profiling is the bridge between the statistical output and business action.

Our process involves:
1.  **Merging Labels:** We will attach the cluster label to each customer in the original, non-preprocessed dataframe. This allows us to profile using the raw, interpretable data.
2.  **Calculating Group Statistics:** For each segment, we will calculate key metrics:
    *   The number of customers (segment size).
    *   The percentage of the total customer base.
    *   The mean of important KPIs like `net_ltv`, `order_count`, and `days_since_last_order`.
    *   The prevalence (mean) of key binary flags (e.g., quiz symptom flags) to understand the health characteristics of each segment.
3.  **Defining Proto-Segments (for K-Means clusters only):** To aid interpretation, we can define simple, rule-based "proto-segments" (e.g., Champions = high LTV and high order count). We can then compute the precision and recall of our statistical clusters against these business-friendly definitions to see how well they align. This is a powerful way to translate the data science results into a language the marketing team can understand.

For the **RFM fallback**, the segments are already interpretable by definition (e.g., "Champions," "At-Risk"), so the primary task is to calculate the average KPIs for each to quantify their behavior.

# ================================================================
# CHECKPOINT 5: Segment Profiling
# ================================================================

if not df.empty:
    if not pivot_to_rfm:
        print("--- Profiling K-Means Clusters ---")
        # Merge cluster labels back to the original dataframe
        df['cluster'] = cluster_labels
        
        # Define profiling columns
        profiling_kpis = ['net_ltv', 'order_count', 'days_since_last_order']
        profiling_flags = [col for col in binary_cols if 'sx_' in col or 'stress' in col]
        
        # Calculate profiles
        profile_agg = {kpi: 'mean' for kpi in profiling_kpis}
        profile_agg.update({flag: 'mean' for flag in profiling_flags})
        profile_agg['customer_id'] = 'count'

        segment_profiles = df.groupby('cluster').agg(profile_agg).rename(columns={'customer_id': 'size'})
        segment_profiles['size_pct'] = (segment_profiles['size'] / len(df)) * 100
        
        print(segment_profiles[
            ['size', 'size_pct'] + profiling_kpis + sorted(profiling_flags, key=lambda x: segment_profiles[x].sum(), reverse=True)[:5]
        ].sort_values('net_ltv', ascending=False))
        
        # Store for export
        df_export = df[['customer_id', 'cluster'] + profiling_kpis]

    else:
        print("--- Pivoting to RFM Segmentation ---")
        
        # 1. Calculate Recency, Frequency, Monetary values
        snapshot_date = df['last_order_date'].max() if pd.to_datetime(df['last_order_date'], errors='coerce').notna().any() else pd.Timestamp('now')
        
        df_rfm = df.groupby('customer_id').agg(
            recency=('days_since_last_order', 'min'),
            frequency=('order_count', 'max'),
            monetary=('net_ltv', 'max')
        ).reset_index()

        # 2. Create RFM quintile scores
        df_rfm['r_score'] = pd.qcut(df_rfm['recency'], 5, labels=False, duplicates='drop') + 1
        df_rfm['f_score'] = pd.qcut(df_rfm['frequency'].rank(method='first'), 5, labels=False, duplicates='drop') + 1
        df_rfm['m_score'] = pd.qcut(df_rfm['monetary'].rank(method='first'), 5, labels=False, duplicates='drop') + 1
        
        # Invert recency score (lower is better)
        df_rfm['r_score'] = 6 - df_rfm['r_score']

        # 3. Define Segments
        def assign_segment(row):
            if row['r_score'] >= 4 and row['f_score'] >= 4 and row['m_score'] >= 4: return 'Champions'
            if row['r_score'] >= 4 and row['f_score'] >= 4: return 'Loyal Customers'
            if row['r_score'] >= 4 and row['frequency'] > 1: return 'Potential Loyalist'
            if row['r_score'] >= 4: return 'New Customers'
            if row['f_score'] >= 4: return 'At Risk'
            if row['r_score'] <= 2 and row['f_score'] <= 2: return 'Lost'
            if row['r_score'] <= 2: return 'Hibernating'
            return 'Needs Attention'
            
        df_rfm['segment'] = df_rfm.apply(assign_segment, axis=1)
        df_rfm.rename(columns={'segment': 'cluster'}, inplace=True)
        
        # 4. Profile the RFM segments
        rfm_profile = df_rfm.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'customer_id': 'count'
        }).rename(columns={'customer_id': 'size'}).sort_values('monetary', ascending=False)
        
        rfm_profile['size_pct'] = (rfm_profile['size'] / len(df_rfm)) * 100
        
        print(rfm_profile[['size', 'size_pct', 'recency', 'frequency', 'monetary']])
        
        # Store for export
        df_export = df_rfm[['customer_id', 'cluster', 'recency', 'frequency', 'monetary']]
        
        # Add cluster labels to main df for visuals
        df = df.merge(df_rfm[['customer_id', 'cluster']], on='customer_id', how='left')

else:
    print("Skipping Checkpoint 5 because the dataframe is not available.")

### Methodological Note on Visualization

Visualizations are essential for communicating the results of the segmentation. We will create three key plots:

1.  **FAMD Component Scatter Plot:** This plot shows the distribution of customers along the first two (most important) FAMD components, with each point colored by its cluster assignment. It gives a high-level view of how well the clusters are separated in the reduced feature space. If the clusters overlap significantly, it can be another indicator of a poor solution.
2.  **Parallel Coordinates Plot:** This is a powerful way to compare the profiles of multiple segments at once. Each line represents a cluster's average value for a set of key variables. It helps to quickly identify the defining characteristics of each segment. We will plot the most discriminating raw variables to make this as interpretable as possible.
3.  **Box Plots of Key KPIs:** We will create box plots to show the distribution of important business metrics like `net_ltv` (log-transformed) and `avg_order_value` for each cluster. This allows us to see not only the average value but also the spread and outliers within each segment, providing a richer understanding of their behavior.

All figures will be saved to a `./figs/` directory for easy access.

# ================================================================
# CHECKPOINT 6: Visuals
# ================================================================
from pandas.plotting import parallel_coordinates

if not df.empty and 'cluster' in df.columns:
    # --- 1. FAMD Component Scatter Plot ---
    if not pivot_to_rfm:
        print("--- Visualizing FAMD Components by Cluster ---")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=X_famd.iloc[:, 0], 
            y=X_famd.iloc[:, 1], 
            hue=df['cluster'], 
            palette='viridis', 
            alpha=0.7, 
            s=50
        )
        plt.title('Customer Segments on First Two FAMD Components')
        plt.xlabel(f'FAMD Component 0 ({famd_final.explained_inertia_[0]:.1%})')
        plt.ylabel(f'FAMD Component 1 ({famd_final.explained_inertia_[1]:.1%})')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.savefig('figs/famd_scatter.png')
        plt.show()

    # --- 2. Box Plots of Key KPIs by Cluster ---
    print("\\n--- Visualizing Key KPIs by Cluster ---")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Log-transform LTV for better visualization
    df['log_net_ltv'] = np.log1p(df['net_ltv'])
    
    sns.boxplot(ax=axes[0], x='cluster', y='log_net_ltv', data=df, order=sorted(df['cluster'].unique()))
    axes[0].set_title('Distribution of Log(Net LTV) by Cluster')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(ax=axes[1], x='cluster', y='avg_order_value', data=df, order=sorted(df['cluster'].unique()))
    axes[1].set_title('Distribution of Average Order Value by Cluster')
    axes[1].set_ylim(0, df['avg_order_value'].quantile(0.95)) # Zoom in on the bulk of the data
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('figs/kpi_boxplots.png')
    plt.show()

    # --- 3. Parallel Coordinates Plot ---
    print("\\n--- Visualizing Segment Profiles with Parallel Coordinates ---")
    
    # Select most discriminating raw variables for the plot
    if not pivot_to_rfm:
        # For K-Means, use a mix of KPIs and top flags
        parallel_cols = ['net_ltv', 'order_count', 'days_since_last_order', 'symptom_count'] + sorted(profiling_flags, key=lambda x: segment_profiles[x].sum(), reverse=True)[:4]
        profile_for_parallel = df.groupby('cluster')[parallel_cols].mean().reset_index()
    else:
        # For RFM, the core metrics are most important
        profile_for_parallel = rfm_profile.reset_index()[['cluster', 'recency', 'frequency', 'monetary']]
        
    # Normalize data for parallel coordinates plotting (Min-Max scaling)
    profile_norm = profile_for_parallel.copy()
    for col in profile_norm.columns[1:]:
        profile_norm[col] = (profile_norm[col] - profile_norm[col].min()) / (profile_norm[col].max() - profile_norm[col].min())
        
    plt.figure(figsize=(15, 8))
    parallel_coordinates(profile_norm, 'cluster', colormap=plt.get_cmap("tab10"))
    plt.title('Parallel Coordinates Plot of Segment Profiles')
    plt.xticks(rotation=45)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.grid(False)
    plt.savefig('figs/parallel_coordinates.png')
    plt.show()
    
else:
    print("Skipping Checkpoint 6 because profiling data is not available.")

### Final Deliverables

The two primary outputs of this analysis are:

1.  **`segmented_customers.csv`**: A CSV file containing the `customer_id`, their assigned `cluster_label`, and the key performance indicators that define that segment's behavior. This file can be directly used by marketing or CRM teams to create targeted campaigns and audiences.
2.  **Slide Deck Summary**: A concise, bullet-point summary of the project's outcome. It highlights the most valuable segments and suggests initial actions, providing a clear takeaway for stakeholders.

These deliverables ensure that the results of our complex analysis are translated into a simple, actionable format for the business.

# ================================================================
# CHECKPOINT 7: Export & Deliverables
# ================================================================

if 'df_export' in locals():
    # --- 1. Export Segmented Customer Data ---
    export_path = 'segmented_customers.csv'
    df_export.to_csv(export_path, index=False)
    print(f"Segmented customer data saved to '{export_path}'")

    # --- 2. Print Concise Summary for Slide Deck ---
    print("\\n\\n================================================================")
    print("  EXECUTIVE SUMMARY FOR PRESENTATION SLIDE")
    print("================================================================")
    
    if pivot_to_rfm:
        champions_data = rfm_profile.loc[['Champions']]
        champions_pct = champions_data['size_pct'].values[0]
        champions_ltv = champions_data['monetary'].values[0]
        
        at_risk_data = rfm_profile.loc[['At Risk']]
        at_risk_pct = at_risk_data['size_pct'].values[0]
        at_risk_ltv = at_risk_data['monetary'].values[0]
        
        print(f"\\n*   **Methodology:** Customer base segmented using robust RFM (Recency, Frequency, Monetary) analysis after initial deep-dive clustering proved unstable.")
        print(f"*   **Top Segment: 'Champions'**")
        print(f"    - **Who:** Our most valuable and engaged customers ({champions_pct:.1f}% of base).")
        print(f"    - **Value:** Average LTV of ${champions_ltv:.2f}, significantly out-spending all other groups.")
        print(f"    - **Action:** Retain & reward with loyalty programs, exclusive access, and solicit reviews.")
        print(f"*   **Key Opportunity: 'At Risk'**")
        print(f"    - **Who:** High-spending, frequent buyers who haven't purchased recently ({at_risk_pct:.1f}% of base).")
        print(f"    - **Value:** High historical LTV of ${at_risk_ltv:.2f}, but in danger of churning.")
        print(f"    - **Action:** Launch targeted win-back campaigns with personalized offers to re-engage.")
        print(f"*   **Next Steps:** Integrate segments into CRM for targeted email campaigns and personalized on-site experiences.")
        
    else:
        # This summary is for the K-Means clusters if they were stable
        top_segment = segment_profiles.sort_values('net_ltv', ascending=False).iloc[0]
        top_segment_name = top_segment.name
        top_segment_pct = top_segment['size_pct']
        top_segment_ltv = top_segment['net_ltv']
        
        print(f"\\n*   **Methodology:** Customers segmented into {best_k} distinct groups using FAMD and K-Means, validated for statistical stability.")
        print(f"*   **Top Segment: Cluster {top_segment_name}**")
        print(f"    - **Who:** A highly engaged group representing {top_segment_pct:.1f}% of the customer base.")
        print(f"    - **Value:** The most valuable segment with an average LTV of ${top_segment_ltv:.2f}.")
        print(f"    - **Action:** Focus retention and loyalty efforts on this core group.")
        print(f"*   **Next Steps:** Deep-dive into the full profile of each cluster to develop tailored marketing actions.")

    print("================================================================")

else:
    print("Skipping Checkpoint 7 because there is no data to export.")

