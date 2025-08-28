import pandas as pd

# Load the file created by your clustering run
df_clusters = pd.read_csv("quiz_clusters_with_meta.csv")

# Filter out noise points if desired
df_clusters = df_clusters[df_clusters["cluster"] != -1]

# Categorical fields to summarize
categorical_cols = [
    "quiz_result",
    "first_sku_bucket",
    "primary_goal",
    "acquisition_code_group",
    "bm_pattern",
    "gi_symptom_cat"
]

# Binary symptom flags to summarize
symptom_flags = [c for c in df_clusters.columns if c.startswith("sx_") or c.endswith("_flag")]

# Numeric fields to summarize
numeric_cols = [
    "order_count",
    "days_since_last_order",
    "symptom_count",
    "gut_issue_score",
    "high_stress"
]

profiles = []

for cluster_id, group in df_clusters.groupby("cluster"):
    profile = {"cluster": cluster_id, "n_customers": len(group)}

    # Top category for each categorical column
    for col in categorical_cols:
        if col in group.columns:
            top_value = group[col].value_counts(normalize=True).head(1)
            profile[f"{col}_top"] = top_value.index[0]
            profile[f"{col}_pct"] = round(top_value.iloc[0] * 100, 1)

    # Symptom prevalence
    for col in symptom_flags:
        if col in group.columns:
            profile[f"{col}_pct"] = round(group[col].mean() * 100, 1)

    # Numeric averages
    for col in numeric_cols:
        if col in group.columns:
            profile[f"{col}_mean"] = round(group[col].mean(), 2)

    profiles.append(profile)

# Convert to DataFrame for review
df_profiles = pd.DataFrame(profiles)

# Sort by cluster
df_profiles = df_profiles.sort_values("cluster")

# Save or print
df_profiles.to_csv("cluster_profiles.csv", index=False)
print(df_profiles.head())
