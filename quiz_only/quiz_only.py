import pandas as pd
import sys
from umap import UMAP
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../raw_data_v2.csv', dtype={
    'quiz_result': str,
    'bm_pattern': str,
    'gi_symptom_cat': str,
    'primary_goal': str
}, low_memory=False)


# Clean primary_goal
df['primary_goal'] = df['primary_goal'].replace('-', 'missing')

# Clean bm_pattern unspecified
df['bm_pattern'] = df['bm_pattern'].replace('unspecified', 'missing')

# Clean quiz_result to group variants
quiz_map = {
    'OMNi-BiOTiC BALANCE': 'Balance',
    'OMNi-BiOTiC® BALANCE': 'Balance',
    'OMNi-BiOTiC Stress Release': 'Stress Release',
    'OMNi-BiOTiC® Stress Release': 'Stress Release',
    'OMNi-BiOTiC HETOX': 'Hetox',
    'OMNi-BiOTiC® HETOX': 'Hetox',
    'Omni-Biotic Power': 'Power',
    'OMNi-BiOTiC Panda': 'Panda',
    'OMNi-BiOTiC AB 10': 'AB 10',
    'OMNi-BiOTiC® AB 10': 'AB 10',
    'Gut Health Reset Program': 'Gut Health Reset'
}
df['quiz_result'] = df['quiz_result'].replace(quiz_map)


# Step 1: Filter quiz takers
df_quiz = df[df['quiz_taker'].str.lower() == 'yes'].copy()

# Step 2: influencer codes
influencer_codes = {
    'dave20', 'dave', 'jessica15', 'drwillcole', 'dr.cain15', 'valeria20',
    'skinny', 'blonde', 'blonde20', 'carly15', 'tammy15', 'sweats15'
}
 
df_quiz['acquisition_code_group'] = df_quiz['ancestor_discount_code'].apply(
    lambda x: x if x in influencer_codes else ('other' if pd.notna(x) else 'none')
)

# Step 2: Drop columns with 0 or near-zero variance or missing entirely
drop_cols = [
'recent_abx_flag', 
'email_key', 
'quiz_date',
'ancestor_discount_code',
'affiliate_segment',
'in_third_trimester_flag',
'stress_physical_flag',
'sx_brain_fog'
]  


# Explicitly drop high-variance transactional RFM features to prevent value-based dominance
rfm_exclude = [
    'gross_ltv', 'net_ltv', 'avg_order_value', 'total_cogs',
    'shipping_collected', 'shipping_spend', 'refund_amt'
]
df_quiz.drop(columns=[col for col in rfm_exclude if col in df_quiz.columns], inplace=True)
df_quiz.drop(columns=['first_order_date', 'last_order_date'], inplace=True)
df_quiz.drop(columns=['refund_ratio', 'refund_count'], inplace=True, errors='ignore')
df_quiz.drop(columns=['quiz_taker'], inplace=True)




df_quiz.drop(columns=drop_cols, inplace=True, errors='ignore')

# Step 3: Drop columns with too much missingness (keep columns with >80% coverage among quiz takers)
non_null_threshold = 0.8 * len(df_quiz)
df_quiz = df_quiz.loc[:, df_quiz.notnull().sum() > non_null_threshold]



# Step 4: Identify categorical and numerical columns
categorical_cols = [
    'first_sku',  'quiz_result', 'acquisition_code_group',
    'bm_pattern', 'gi_symptom_cat', 'primary_goal'
]

print(df_quiz.columns.tolist())


# Limit to those that survived filtering
categorical_cols = [col for col in categorical_cols if col in df_quiz.columns]

# Step 5: One-hot encode categorical variables
df_quiz_encoded = pd.get_dummies(df_quiz, columns=categorical_cols, drop_first=True)
customer_ids = df_quiz['customer_id'].copy()
df_quiz_encoded = df_quiz_encoded.drop(columns=['customer_id'])

# Step 6: Fill remaining missing binary flags with 0
binary_cols = [col for col in df_quiz_encoded.columns if col.startswith('is_') or col.endswith('_flag') or col.startswith('sx_')]
df_quiz_encoded[binary_cols] = df_quiz_encoded[binary_cols].fillna(0).astype(int)

# Step 7: Scale numerical columns
from sklearn.preprocessing import StandardScaler

numerical_cols = ['symptom_count', 'high_stress']

# Add new RFM-style columns
behavioral_cols = ['order_count', 'days_since_last_order']
for col in behavioral_cols:
    if col in df_quiz_encoded.columns:
        df_quiz_encoded[col] = df_quiz_encoded[col].fillna(0)
        numerical_cols.append(col)
scaler = StandardScaler()
df_quiz_encoded[numerical_cols] = scaler.fit_transform(df_quiz_encoded[numerical_cols])

reducer = UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(df_quiz_encoded)

# === HDBSCAN: Density-based clustering ===
clusterer = hdbscan.HDBSCAN(min_cluster_size=60)
clusters = clusterer.fit_predict(embedding)
df_clusters = pd.DataFrame({
    'customer_id': customer_ids,
    'cluster': clusters
})

df_quiz = df_quiz.merge(df_clusters, on='customer_id')



# Attach clusters to DataFrame
# df_quiz = df_quiz.merge(df_quiz_encoded[['customer_id', 'cluster']], on='customer_id')
print(df_quiz.groupby('cluster').mean(numeric_only=True))
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df_quiz.groupby('cluster')[col].value_counts(normalize=True))
with open("cluster_summary.txt", "w") as f:
    # Write cluster means for numeric columns
    f.write("=== NUMERIC CLUSTER MEANS ===\n")
    numeric_means = df_quiz.groupby('cluster').mean(numeric_only=True)
    f.write(numeric_means.to_string())
    f.write("\n\n")

    # Write normalized value counts for each categorical column
    for col in categorical_cols:
        f.write(f"\n=== {col.upper()} ===\n")
        val_counts = df_quiz.groupby('cluster')[col].value_counts(normalize=True)
        f.write(val_counts.to_string())
        f.write("\n\n")


# === Plot clusters ===
plt.figure(figsize=(10, 6))
palette = sns.color_palette('tab10', n_colors=len(set(clusters)) + 1)
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=clusters, palette=palette, s=10)
plt.title("UMAP projection with HDBSCAN clusters")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# Output the cleaned dataset shape






