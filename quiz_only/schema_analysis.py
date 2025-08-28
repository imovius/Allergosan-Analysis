import pandas as pd

# Read CSV
df = pd.read_csv('../raw_data_v3.csv')

# Print sample rows to console
print(df.head(3))

# Open file to write schema and value counts
with open("schema_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== Column Names & Types ===\n")
    for col, dtype in df.dtypes.items():
        f.write(f"{col}: {dtype}\n")
    
    f.write("\n=== Value Counts (Top 10 per column) ===\n")
    for col in df.columns:
        f.write(f"\n-- {col} --\n")
        try:
            vc = df[col].value_counts(dropna=False).head(10)
            f.write(vc.to_string())
        except Exception as e:
            f.write(f"Error getting value counts: {e}")
        f.write("\n")

print("Schema and value counts written to schema_summary.txt")
