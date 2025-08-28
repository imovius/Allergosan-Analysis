# quiz_segments_kproto_famd.py
# Python 3.9+

import os
import numpy as np
import pandas as pd
from typing import Optional, Set, Dict
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from prince import FAMD

# -------------------------- Config --------------------------
CSV_PATH = "../raw_data_v3.csv"
OUTDIR = "outputs"
K_CLUSTERS = 6           # tweak 5-7 quickly if needed
INFLUENCER_ONLY = True   # set False to include all quiz takers (recommended for "promo influence" macro view)
SEED = 42

os.makedirs(OUTDIR, exist_ok=True)

# -------------------------- Helpers --------------------------
SKU_TO_BUCKET: Dict[str, str] = {
    "OBSR": "stress", "S-OBSR": "stress",
    "OBAB": "gut_restoration", "3MGHR": "gut_restoration",
    "OBHX": "detox", "S-OBHX": "detox",
    "OBBA": "immune", "OLIM": "immune",
    "OBPO": "metabolism", "OLPL": "metabolism", "S-OBP": "metabolism",
    "OBPA": "prenatal", "S-OBPA": "prenatal",
    "OBCG": "other_ob", "OBCA": "other_ob", "OBCD": "other_ob",
    "OBSC": "accessory",
    "LBUN": "bundle", "IMBUN": "bundle", "DYNDUO": "bundle",
}

QUIZ_NORMALIZE: Dict[str, str] = {
    "OMNi-BiOTiC BALANCE": "Balance",
    "OMNi-BiOTiCA® BALANCE": "Balance",
    "OMNi-BiOTiC Stress Release": "Stress Release",
    "OMNi-BiOTiCA® Stress Release": "Stress Release",
    "OMNi-BiOTiC HETOX": "Hetox",
    "OMNi-BiOTiCA® HETOX": "Hetox",
    "Omni-Biotic Power": "Power",
    "OMNi-BiOTiC Panda": "Panda",
    "OMNi-BiOTiC AB 10": "AB 10",
    "OMNi-BiOTiCA® AB 10": "AB 10",
    "Gut Health Reset Program": "Gut Health Reset",
}

QUIZ_LINE_TO_BUCKET: Dict[str, str] = {
    "Stress Release": "stress",
    "Balance": "immune",
    "Hetox": "detox",
    "AB 10": "gut_restoration",
    "Gut Health Reset": "gut_restoration",
    "Panda": "prenatal",
    "Power": "metabolism",
}

INFLUENCER_CODES = {
    "dave20","dave","jessica15","drwillcole","dr.cain15","valeria20",
    "skinny","blonde","blonde20","carly15","tammy15","sweats15"
}
INFLUENCER_CODES = {c.lower() for c in INFLUENCER_CODES}

def normalize_quiz_result(q: Optional[str]) -> Optional[str]:
    if not isinstance(q, str):
        return None
    q_norm = q.strip()
    if not q_norm:
        return None
    return QUIZ_NORMALIZE.get(q_norm, QUIZ_NORMALIZE.get(q_norm.lower(), q_norm))

def quiz_to_bucket(quiz_result: Optional[str]) -> Optional[str]:
    if not isinstance(quiz_result, str):
        return None
    line = normalize_quiz_result(quiz_result)
    if line is None:
        return None
    return QUIZ_LINE_TO_BUCKET.get(line)

def parse_first_sku_buckets(first_sku: Optional[str]) -> Set[str]:
    buckets: Set[str] = set()
    if not isinstance(first_sku, str) or not first_sku.strip():
        return buckets
    for token in [t.strip() for t in first_sku.split(",")]:
        bucket = SKU_TO_BUCKET.get(token)
        if bucket:
            buckets.add(bucket)
    return buckets

def collapse_sku_buckets(s: Set[str]) -> str:
    if not s:
        return "none"
    return sorted(list(s))[0] if len(s) == 1 else "multi"

def loose_match(q_bucket: Optional[str], sku_buckets: Set[str]) -> int:
    return int(q_bucket in sku_buckets) if q_bucket and sku_buckets else 0

def map_code_group(x: Optional[str]) -> str:
    if isinstance(x, str) and x.strip():
        v = x.strip().lower()
        return v if v in INFLUENCER_CODES else "other"
    return "none"

# -------------------------- Load & Basic Prep --------------------------
df = pd.read_csv(CSV_PATH, low_memory=False)

if "customer_id" not in df.columns:
    raise ValueError("customer_id column is required.")

# Clean
if "primary_goal" in df.columns:
    df["primary_goal"] = df["primary_goal"].replace("-", "missing")
if "bm_pattern" in df.columns:
    df["bm_pattern"] = df["bm_pattern"].replace("unspecified", "missing")
if "quiz_result" in df.columns:
    df["quiz_result"] = df["quiz_result"].apply(normalize_quiz_result)

# Feature engineering on df (so merge-backs keep the columns)
df["quiz_bucket"] = df["quiz_result"].apply(quiz_to_bucket)
df["first_sku_buckets"] = df["first_sku"].apply(parse_first_sku_buckets)
df["first_sku_bucket"] = df["first_sku_buckets"].apply(collapse_sku_buckets)
df["quiz_reco_match"] = df.apply(
    lambda r: loose_match(r.get("quiz_bucket"), r.get("first_sku_buckets")), axis=1
)
df["acquisition_code_group"] = df.get("ancestor_discount_code", np.nan).apply(map_code_group)

# Quiz takers
if "quiz_taker" not in df.columns:
    raise ValueError("quiz_taker column is required.")
df_quiz = df[df["quiz_taker"].astype(str).str.lower() == "yes"].copy()

# Optional: influencer-only
if INFLUENCER_ONLY:
    df_quiz = df_quiz[df_quiz["acquisition_code_group"].ne("none")].copy()

# Drop raw/unneeded columns (keep acquisition_code_group)
drop_now = [
    "acquisition_channel", "first_sku", "first_sku_buckets", "quiz_bucket",
    "recent_abx_flag", "email_key", "quiz_date", "ancestor_discount_code",
    "affiliate_segment", "in_third_trimester_flag", "stress_physical_flag", "sx_brain_fog",
    "gross_ltv","net_ltv","avg_order_value","total_cogs","shipping_collected","shipping_spend","refund_amt",
    "first_order_date","last_order_date","quiz_taker","refund_ratio"
]
df_quiz.drop(columns=[c for c in drop_now if c in df_quiz.columns], inplace=True, errors="ignore")

# Keep columns with >=80% coverage among quiz takers
non_null_threshold = 0.8 * len(df_quiz)
df_quiz = df_quiz.loc[:, df_quiz.notnull().sum() >= non_null_threshold].copy()

# -------------------------- Model Frame (no one-hot) --------------------------
cat_cols = [
    "first_sku_bucket", "quiz_result", "acquisition_code_group",
    "bm_pattern", "gi_symptom_cat", "primary_goal"
]
cat_cols = [c for c in cat_cols if c in df_quiz.columns]

num_cols = [c for c in ["order_count","days_since_last_order","symptom_count","high_stress","gut_issue_score","refund_count"] if c in df_quiz.columns]

# Build modeling df
use_cols = ["customer_id"] + cat_cols + num_cols
df_model = df_quiz[use_cols].copy()

# Fill types cleanly
for c in cat_cols:
    df_model[c] = df_model[c].fillna("missing").astype(str)
for c in num_cols:
    df_model[c] = pd.to_numeric(df_model[c], errors="coerce").fillna(0.0)

# Scale numerics for FAMD and k-prototypes numeric part
scaler = StandardScaler()
X_num = scaler.fit_transform(df_model[num_cols]) if num_cols else np.empty((len(df_model), 0))
X_cat = df_model[cat_cols].astype(str).values if cat_cols else np.empty((len(df_model), 0))

# -------------------------- FAMD (latent factors for slides) --------------------------
# FAMD takes mixed data directly (categoricals as strings)
famd_input = df_model.drop(columns=["customer_id"]).copy()
famd = FAMD(n_components=3, random_state=SEED)
famd_scores = famd.fit_transform(famd_input)           # shape: (n_samples, 3)
famd_cols = [f"FAMD_{i+1}" for i in range(famd_scores.shape[1])]
df_famd = pd.DataFrame(famd_scores, columns=famd_cols, index=df_model.index)
df_famd["customer_id"] = df_model["customer_id"].values

# Numeric correlations to factors (quick, intuitive)
num_contrib = {}
for col in num_cols:
    v = pd.to_numeric(famd_input[col], errors="coerce").fillna(0).values
    num_contrib[col] = [np.corrcoef(v, df_famd[f])[0,1] for f in famd_cols]
num_contrib_df = pd.DataFrame(num_contrib, index=famd_cols).T

# Categorical level means on factors (which levels sit high/low)
cat_contrib_frames = []
for col in cat_cols:
    tmp = pd.concat([df_famd[famd_cols], df_model[[col]]], axis=1)
    tmp = tmp.groupby(col)[famd_cols].mean().reset_index()
    tmp.insert(0, "feature", col)
    cat_contrib_frames.append(tmp)
cat_contrib_df = pd.concat(cat_contrib_frames, ignore_index=True)

# Save factor outputs
num_contrib_df.to_csv(os.path.join(OUTDIR, "famd_numeric_contributions.csv"), index=True)
cat_contrib_df.to_csv(os.path.join(OUTDIR, "famd_categorical_level_means.csv"), index=False)

# -------------------------- k-Prototypes Clustering --------------------------
# Build mixed matrix: categoricals first, then numerics
X = np.concatenate([X_cat, X_num], axis=1)
cat_idx = list(range(len(cat_cols)))  # positions of categorical features in X

kp = KPrototypes(n_jobs=-1, n_clusters=K_CLUSTERS, init='Huang', n_init=5, random_state=SEED)
labels = kp.fit_predict(X, categorical=cat_idx)

df_labels = df_model[["customer_id"]].copy()
df_labels["cluster"] = labels.astype(int)

# Merge with original df for readable outputs
df_out = df_labels.merge(df, on="customer_id", how="left")
df_out.to_csv(os.path.join(OUTDIR, "kproto_clusters_with_meta.csv"), index=False)

# -------------------------- Cluster Profiles + Personas --------------------------
cat_profile_cols = ["quiz_result","first_sku_bucket","primary_goal","bm_pattern","gi_symptom_cat","acquisition_code_group"]
num_profile_cols = ["order_count","days_since_last_order","symptom_count","gut_issue_score","high_stress"]
flag_cols = [c for c in df_out.columns if c.startswith("sx_") or c.endswith("_flag")]

profiles = []
for cid, g in df_out.groupby("cluster"):
    rec = {"cluster": int(cid), "n_customers": int(len(g))}
    # top cats (+%)
    for col in cat_profile_cols:
        if col in g.columns:
            vc = g[col].value_counts(normalize=True)
            if len(vc):
                rec[f"{col}_top"] = vc.index[0]
                rec[f"{col}_pct"] = round(vc.iloc[0]*100, 1)
    # flags prevalence
    for col in flag_cols:
        rec[f"{col}_pct"] = round(pd.to_numeric(g[col], errors="coerce").fillna(0).mean()*100, 1)
    # numeric means
    for col in num_profile_cols:
        if col in g.columns:
            rec[f"{col}_mean"] = round(pd.to_numeric(g[col], errors="coerce").fillna(0).mean(), 2)
    # promo mix (top 3)
    if "acquisition_code_group" in g.columns:
        promo = g["acquisition_code_group"].value_counts(normalize=True).head(3)
        for i, (code, share) in enumerate(promo.items(), start=1):
            rec[f"promo_{i}"] = f"{code}:{round(share*100,1)}%"
    profiles.append(rec)

df_profiles = pd.DataFrame(profiles).sort_values("cluster")
df_profiles.to_csv(os.path.join(OUTDIR, "kproto_cluster_profiles.csv"), index=False)

# Persona blurbs
def _safe(v): return v if isinstance(v, str) else "—"
persona_lines = []
for _, r in df_profiles.iterrows():
    quiz = _safe(r.get("quiz_result_top"))
    sku  = _safe(r.get("first_sku_bucket_top"))
    goal = _safe(r.get("primary_goal_top"))
    gi   = _safe(r.get("gi_symptom_cat_top"))
    bm   = _safe(r.get("bm_pattern_top"))
    p1   = _safe(r.get("promo_1"))
    p2   = _safe(r.get("promo_2"))
    oc   = r.get("order_count_mean", 0)
    dsl  = r.get("days_since_last_order_mean", 0)
    sc   = r.get("symptom_count_mean", 0)
    hs   = r.get("high_stress_mean", 0)

    blurb = (
        f"Cluster {int(r['cluster'])} — {int(r['n_customers'])} customers\n"
        f"- Dominant need: {quiz}; SKU bucket: {sku}; Goal: {goal}\n"
        f"- GI/BM: {gi}, {bm}; Symptoms avg: {sc}; Stress avg: {hs}\n"
        f"- Behavior: orders avg {oc}, days-since-last-order avg {dsl}\n"
        f"- Top promos: {p1} {p2}\n"
    )
    persona_lines.append(blurb)

with open(os.path.join(OUTDIR, "personas.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(persona_lines))

# Save FAMD scores with cluster to plot in your deck (e.g., FAMD_1 vs FAMD_2 colored by cluster)
df_famd.merge(df_labels, on="customer_id", how="left").to_csv(
    os.path.join(OUTDIR, "famd_scores_with_clusters.csv"), index=False
)

print("Wrote:")
print(f" - {os.path.join(OUTDIR, 'kproto_clusters_with_meta.csv')}")
print(f" - {os.path.join(OUTDIR, 'kproto_cluster_profiles.csv')}")
print(f" - {os.path.join(OUTDIR, 'personas.txt')}")
print(f" - {os.path.join(OUTDIR, 'famd_numeric_contributions.csv')}")
print(f" - {os.path.join(OUTDIR, 'famd_categorical_level_means.csv')}")
print(f" - {os.path.join(OUTDIR, 'famd_scores_with_clusters.csv')}")
