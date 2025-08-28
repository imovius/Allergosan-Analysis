# quiz_segments_kproto_famd.py
# Python 3.9+

import os
import re
import numpy as np
import pandas as pd
from typing import Optional, Set, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from prince import FAMD
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -------------------------- Config --------------------------
CSV_PATH = "../raw_data_v3.csv"
OUTDIR = "outputs"
K_CLUSTERS = 6           # tweak 5–7 quickly if needed
INFLUENCER_ONLY = True   # False = all quiz takers
SEED = 42
DOMINANCE_THRESH = 0.98  # for FAMD variance filtering
DEBUG_SUMMARIES = True   # write quick diags

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

def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

SANITIZED_NORMALIZE = {_sanitize(k): v for k, v in QUIZ_NORMALIZE.items()}

def normalize_quiz_result(q: Optional[str]) -> Optional[str]:
    if not isinstance(q, str):
        return None
    q_norm = q.strip()
    if not q_norm:
        return None
    if q_norm in QUIZ_NORMALIZE:
        return QUIZ_NORMALIZE[q_norm]
    s = _sanitize(q_norm)
    if s in SANITIZED_NORMALIZE:
        return SANITIZED_NORMALIZE[s]
    return q_norm

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

def derive_gender(row: pd.Series) -> str:
    g = row.get("gender", None)
    if isinstance(g, str) and g.strip():
        return g.strip().lower()
    im = row.get("is_male", None)
    if pd.notna(im):
        try:
            return "male" if int(im) == 1 else "female"
        except Exception:
            pass
    return "missing"

# -------------------------- Load & Basic Prep --------------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
df["customer_id"] = df["customer_id"].astype(str)   
if "customer_id" not in df.columns:
    raise ValueError("customer_id column is required.")

if "primary_goal" in df.columns:
    df["primary_goal"] = df["primary_goal"].replace("-", "missing")
if "bm_pattern" in df.columns:
    df["bm_pattern"] = df["bm_pattern"].replace("unspecified", "missing")
if "quiz_result" in df.columns:
    if DEBUG_SUMMARIES:
        df["quiz_result"].value_counts(dropna=False).head(20).to_csv(os.path.join(OUTDIR, "dbg_quiz_result_raw_top20.csv"))
    df["quiz_result"] = df["quiz_result"].apply(normalize_quiz_result)
    if DEBUG_SUMMARIES:
        df["quiz_result"].value_counts(dropna=False).head(20).to_csv(os.path.join(OUTDIR, "dbg_quiz_result_norm_top20.csv"))

df["quiz_bucket"] = df["quiz_result"].apply(quiz_to_bucket)
df["first_sku_buckets"] = df["first_sku"].apply(parse_first_sku_buckets)
df["first_sku_bucket"] = df["first_sku_buckets"].apply(collapse_sku_buckets)
df["quiz_reco_match"] = df.apply(lambda r: loose_match(r.get("quiz_bucket"), r.get("first_sku_buckets")), axis=1)
df["acquisition_code_group"] = df.get("ancestor_discount_code", np.nan).apply(map_code_group)
df["gender_cat"] = df.apply(derive_gender, axis=1)

# Quiz takers
if "quiz_taker" not in df.columns:
    raise ValueError("quiz_taker column is required.")
df_quiz = df[df["quiz_taker"].astype(str).str.lower() == "yes"].copy()

# Optional: influencer-only
if INFLUENCER_ONLY:
    df_quiz = df_quiz[df_quiz["acquisition_code_group"].ne("none")].copy()

# Drop raw/unneeded
drop_now = [
    "acquisition_channel", "first_sku", "first_sku_buckets", "quiz_bucket",
    "recent_abx_flag", "email_key", "quiz_date", "ancestor_discount_code",
    "affiliate_segment", "in_third_trimester_flag", "stress_physical_flag", "sx_brain_fog",
    "gross_ltv","net_ltv","avg_order_value","total_cogs","shipping_collected","shipping_spend","refund_amt",
    "first_order_date","last_order_date","quiz_taker","refund_ratio"
]
df_quiz.drop(columns=[c for c in drop_now if c in df_quiz.columns], inplace=True, errors="ignore")



# Coverage filter
COVERAGE = 0.70
non_null_threshold = COVERAGE * len(df_quiz)
df_quiz = df_quiz.loc[:, df_quiz.notnull().sum() >= non_null_threshold].copy()
print(f"[COVERAGE] Rows: {len(df_quiz)}, Cols kept: {list(df_quiz.columns)}")

# -------------------------- Model Frame (no one-hot) --------------------------
cat_cols = [
    "first_sku_bucket", "quiz_result", "acquisition_code_group",
    "bm_pattern", "gi_symptom_cat", "gender_cat"
]
cat_cols = [c for c in cat_cols if c in df_quiz.columns]

num_cols = [c for c in ["order_count","days_since_last_order","symptom_count","high_stress","gut_issue_score","refund_count"] if c in df_quiz.columns]

use_cols = ["customer_id"] + cat_cols + num_cols
df_model = df_quiz[use_cols].copy()
df_model["customer_id"] = df_model["customer_id"].astype(str) 


for c in cat_cols:
    df_model[c] = df_model[c].fillna("missing").astype(str)
for c in num_cols:
    df_model[c] = pd.to_numeric(df_model[c], errors="coerce").fillna(0.0)

print(f"[PRE-FAMD] cat_cols: {cat_cols}")
print(f"[PRE-FAMD] num_cols: {num_cols}")
for c in cat_cols:
    top = df_model[c].value_counts(normalize=True, dropna=False).head(1)
    print(f"  CAT {c}: top share={round(float(top.iloc[0])*100,1)}%")
for c in num_cols:
    print(f"  NUM {c}: std={df_model[c].std():.4f}")

# Scale numerics for k-prototypes numeric part
scaler = StandardScaler()
X_num = scaler.fit_transform(df_model[num_cols]) if num_cols else np.empty((len(df_model), 0))
X_cat = df_model[cat_cols].astype(str).values if cat_cols else np.empty((len(df_model), 0))

# -------------------------- FAMD (with variance filtering) --------------------------
famd_input = df_model.drop(columns=["customer_id"]).copy()
famd_cat = [c for c in famd_input.columns if famd_input[c].dtype == object]
famd_num = [c for c in famd_input.columns if c not in famd_cat]

keep_num, num_drops = [], []
for c in famd_num:
    s = pd.to_numeric(famd_input[c], errors="coerce").fillna(0)
    if s.std(ddof=0) > 1e-8:
        keep_num.append(c)
    else:
        num_drops.append(c)

keep_cat, cat_drops = [], []
for c in famd_cat:
    v = famd_input[c].astype(str).fillna("missing")
    vc = v.value_counts(normalize=True, dropna=False)
    if (vc.size >= 2) and (vc.iloc[0] <= DOMINANCE_THRESH):
        keep_cat.append(c)
    else:
        cat_drops.append(c)

famd_keep = keep_cat + keep_num
if DEBUG_SUMMARIES:
    pd.DataFrame({
        "kept_numeric": [", ".join(keep_num)],
        "dropped_numeric": [", ".join(num_drops)],
        "kept_categorical": [", ".join(keep_cat)],
        "dropped_categorical": [", ".join(cat_drops)],
    }).to_csv(os.path.join(OUTDIR, "famd_drop_report.csv"), index=False)

print(f"[FAMD KEEP] kept_num={keep_num}, dropped_num={num_drops}")
print(f"[FAMD KEEP] kept_cat={keep_cat}, dropped_cat={cat_drops}")

if len(famd_keep) >= 2:
    famd_input_filt = famd_input[famd_keep].copy()
    for c in keep_cat:
        famd_input_filt[c] = famd_input_filt[c].astype("category")

    famd = FAMD(n_components=min(3, len(famd_keep)-1), random_state=SEED)
    famd_scores = famd.fit_transform(famd_input_filt)

    X = famd_scores if isinstance(famd_scores, np.ndarray) else famd_scores.values

    print("Testing silhouette scores for k=2..10")
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = model.fit_predict(X)
        score = silhouette_score(X, clusters)
        print(f"k={k}: silhouette={score:.4f}")


    fs = np.asarray(famd_scores)
    famd_cols = [f"FAMD_{i+1}" for i in range(fs.shape[1])]
    df_famd = pd.DataFrame(fs, columns=famd_cols, index=df_model.index)
    df_famd["customer_id"] = df_model["customer_id"].astype(str).values

    print("[FAMD] shape:", df_famd.shape, "nulls in FAMD_*:",
          int(df_famd.filter(like="FAMD_").isna().sum().sum()))

    df_famd.to_csv(os.path.join(OUTDIR, "famd_scores_raw.csv"), index=False)
else:
    df_famd = pd.DataFrame({"customer_id": df_model["customer_id"].astype(str).values})
    for i in range(1, 4):
        df_famd[f"FAMD_{i}"] = np.nan
    df_famd.to_csv(os.path.join(OUTDIR, "famd_scores_raw.csv"), index=False)

# -------------------------- k-Prototypes Clustering --------------------------
X = np.concatenate([X_cat, X_num], axis=1)
cat_idx = list(range(len(cat_cols)))

kp = KPrototypes(n_jobs=-1, n_clusters=K_CLUSTERS, init='Huang', n_init=5, random_state=SEED)
labels = kp.fit_predict(X, categorical=cat_idx)

df_labels = df_model[["customer_id"]].copy()
df_labels["customer_id"] = df_labels["customer_id"].astype(str)
df_labels["cluster"] = labels.astype(int)
print("[LABELS] rows:", len(df_labels), "unique ids:", df_labels["customer_id"].nunique())

# Merge with original df for readable outputs
df_out = df_labels.merge(df, on="customer_id", how="left")
df_out.to_csv(os.path.join(OUTDIR, "kproto_clusters_with_meta.csv"), index=False)

# -------------------------- Cluster Profiles + Personas --------------------------
cat_profile_cols = [
    "quiz_result","first_sku_bucket",
    "bm_pattern","gi_symptom_cat","acquisition_code_group","gender_cat"
]
num_profile_cols = ["order_count","days_since_last_order","symptom_count","gut_issue_score","high_stress"]
flag_cols = [c for c in df_out.columns if c.startswith("sx_") or c.endswith("_flag")]

profiles = []
for cid, g in df_out.groupby("cluster"):
    rec = {"cluster": int(cid), "n_customers": int(len(g))}
    for col in cat_profile_cols:
        if col in g.columns:
            vc = g[col].value_counts(normalize=True)
            if len(vc):
                rec[f"{col}_top"] = vc.index[0]
                rec[f"{col}_pct"] = round(vc.iloc[0]*100, 1)
    for col in flag_cols:
        rec[f"{col}_pct"] = round(pd.to_numeric(g[col], errors="coerce").fillna(0).mean()*100, 1)
    for col in num_profile_cols:
        if col in g.columns:
            rec[f"{col}_mean"] = round(pd.to_numeric(g[col], errors="coerce").fillna(0).mean(), 2)
    if "quiz_reco_match" in g.columns:
        rec["quiz_reco_match_pct"] = round(pd.to_numeric(g["quiz_reco_match"], errors="coerce").fillna(0).mean()*100, 1)
    if "acquisition_code_group" in g.columns:
        promo = g["acquisition_code_group"].value_counts(normalize=True).head(3)
        for i, (code, share) in enumerate(promo.items(), start=1):
            rec[f"promo_{i}"] = f"{code}:{round(share*100,1)}%"
    profiles.append(rec)

df_profiles = pd.DataFrame(profiles).sort_values("cluster")
df_profiles.to_csv(os.path.join(OUTDIR, "kproto_cluster_profiles.csv"), index=False)

def _safe(v): return v if isinstance(v, str) else "—"
persona_lines = []
for _, r in df_profiles.iterrows():
    sku  = _safe(r.get("first_sku_bucket_top"))
    aff  = _safe(r.get("acquisition_code_group_top"))
    need = _safe(r.get("quiz_result_top"))

    gi   = _safe(r.get("gi_symptom_cat_top"))
    bm   = _safe(r.get("bm_pattern_top"))
    gen  = _safe(r.get("gender_cat_top"))
    oc   = r.get("order_count_mean", 0)
    dsl  = r.get("days_since_last_order_mean", 0)
    sc   = r.get("symptom_count_mean", 0)
    qmr  = r.get("quiz_reco_match_pct", 0)
    p1   = _safe(r.get("promo_1"))
    p2   = _safe(r.get("promo_2"))
    blurb = (
        f"Cluster {int(r['cluster'])} — {int(r['n_customers'])} customers\n"
        f"- SKU group: {sku}; Dominant need: {need}; Gender: {gen}\n"
        f"- GI/BM: {gi}, {bm}; Avg symptoms: {sc}\n"
        f"- Behavior: avg orders {oc}, days since last order {dsl}; Quiz-match: {qmr}%\n"
        f"- Top promos: {p1} {p2}\n"
    )
    persona_lines.append(blurb)

with open(os.path.join(OUTDIR, "personas.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(persona_lines))

# -------------------------- FAMD + cluster merge & save --------------------------
df_famd["customer_id"] = df_famd["customer_id"].astype(str)
df_labels["customer_id"] = df_labels["customer_id"].astype(str)
famd_with_clusters = df_famd.merge(df_labels, on="customer_id", how="inner", validate="one_to_one")
print("[MERGE] famd rows:", len(df_famd),
      "| labels rows:", len(df_labels),
      "| merged rows:", len(famd_with_clusters))
print("[MERGE] any cluster NaN? ->", famd_with_clusters["cluster"].isna().any())

famd_with_clusters.to_csv(os.path.join(OUTDIR, "famd_scores_with_clusters.csv"), index=False)



# 1. Variance explained by FAMD components
try:
    var_exp = famd.explained_inertia_
    print("FAMD variance explained per component:", var_exp)
    print("Cumulative variance explained (first 2):", sum(var_exp[:2]))
except AttributeError:
    print("FAMD variance explained not available — check library version.")

# 2. Silhouette score using transformed data
score = silhouette_score(famd_scores, df_labels['cluster'])
print(f"Silhouette score: {score:.3f} (closer to 1 = better, ~0 = weak separation)")

# 3. 3D scatter plot of first three components
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    famd_scores[:, 0], famd_scores[:, 1], famd_scores[:, 2],
    c=df_labels['cluster'], cmap='tab10', s=15
)
ax.set_xlabel('FAMD 1')
ax.set_ylabel('FAMD 2')
ax.set_zlabel('FAMD 3')
ax.set_title('FAMD components 1-3 (colored by cluster)')
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()




print("Wrote:")
print(f" - {os.path.join(OUTDIR, 'kproto_clusters_with_meta.csv')}")
print(f" - {os.path.join(OUTDIR, 'kproto_cluster_profiles.csv')}")
print(f" - {os.path.join(OUTDIR, 'personas.txt')}")
print(f" - {os.path.join(OUTDIR, 'famd_scores_raw.csv')}")
print(f" - {os.path.join(OUTDIR, 'famd_scores_with_clusters.csv')}")
if DEBUG_SUMMARIES:
    print(f" - {os.path.join(OUTDIR, 'famd_drop_report.csv')}")
    print(f" - {os.path.join(OUTDIR, 'dbg_quiz_result_raw_top20.csv')}")
    print(f" - {os.path.join(OUTDIR, 'dbg_quiz_result_norm_top20.csv')}")
