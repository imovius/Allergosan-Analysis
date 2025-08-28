# quiz_segments_famd_kmeans_experiment.py
# Python 3.9+
# End-to-end: data prep -> FAMD (variance-targeted) -> (A) sweep OR (B) final clustering -> outputs

import os, re, sys
import numpy as np
import pandas as pd
from typing import Optional, Set, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Try to import HDBSCAN (optional)
try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

# Your FAMD implementation (from prince). Assumes `from prince import FAMD` works in your env.
from prince import FAMD

# ======================= CONFIG =======================
CSV_PATH = "../raw_data_v3.csv"
OUTDIR   = "outputs"
SEED     = 42

# Data filters
INFLUENCER_ONLY   = False   # True = promo purchasers only; False = all quiz takers
COVERAGE          = 0.70    # keep cols with >= 70% non-null among quiz takers
DOMINANCE_THRESH  = 0.98    # drop categorical features dominated by one level

# Run mode
RUN_SWEEP = True            # True = run sweep and write leaderboard CSV, then exit
VAR_TARGETS = [0.70, 0.75, 0.80, 0.85, 0.90]  # cumulative variance targets for sweep

# If RUN_SWEEP=False, choose a final model:
FINAL_MODEL  = "kmeans"     # "kmeans" | "gmm" | "hdbscan"
FINAL_PARAMS = {"k": 4}     # k for kmeans/gmm; for hdbscan: {"min_cluster_size": 30}
FINAL_VAR_TARGET = 0.85     # cumulative variance to keep
# ======================================================
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(SEED)

# -------------------------- Helpers --------------------------
SKU_TO_BUCKET: Dict[str, str] = {
    "OBSR":"stress","S-OBSR":"stress",
    "OBAB":"gut_restoration","3MGHR":"gut_restoration",
    "OBHX":"detox","S-OBHX":"detox",
    "OBBA":"immune","OLIM":"immune",
    "OBPO":"metabolism","OLPL":"metabolism","S-OBP":"metabolism",
    "OBPA":"prenatal","S-OBPA":"prenatal",
    "OBCG":"other_ob","OBCA":"other_ob","OBCD":"other_ob",
    "OBSC":"accessory",
    "LBUN":"bundle","IMBUN":"bundle","DYNDUO":"bundle",
}

QUIZ_NORMALIZE: Dict[str, str] = {
    "OMNi-BiOTiC BALANCE":"Balance",
    "OMNi-BiOTiCA® BALANCE":"Balance",
    "OMNi-BiOTiC Stress Release":"Stress Release",
    "OMNi-BiOTiCA® Stress Release":"Stress Release",
    "OMNi-BiOTiC HETOX":"Hetox",
    "OMNi-BiOTiCA® HETOX":"Hetox",
    "Omni-Biotic Power":"Power",
    "OMNi-BiOTiC Panda":"Panda",
    "OMNi-BiOTiC AB 10":"AB 10",
    "OMNi-BiOTiCA® AB 10":"AB 10",
    "Gut Health Reset Program":"Gut Health Reset",
}

QUIZ_LINE_TO_BUCKET: Dict[str, str] = {
    "Stress Release":"stress",
    "Balance":"immune",
    "Hetox":"detox",
    "AB 10":"gut_restoration",
    "Gut Health Reset":"gut_restoration",
    "Panda":"prenatal",
    "Power":"metabolism",
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
    if not isinstance(q, str): return None
    qn = q.strip()
    if not qn: return None
    if qn in QUIZ_NORMALIZE: return QUIZ_NORMALIZE[qn]
    s = _sanitize(qn)
    return SANITIZED_NORMALIZE.get(s, qn)

def quiz_to_bucket(quiz_result: Optional[str]) -> Optional[str]:
    if not isinstance(quiz_result, str): return None
    line = normalize_quiz_result(quiz_result)
    if line is None: return None
    return QUIZ_LINE_TO_BUCKET.get(line)

def parse_first_sku_buckets(first_sku: Optional[str]) -> Set[str]:
    buckets: Set[str] = set()
    if not isinstance(first_sku, str) or not first_sku.strip(): return buckets
    for token in [t.strip() for t in first_sku.split(",")]:
        b = SKU_TO_BUCKET.get(token)
        if b: buckets.add(b)
    return buckets

def collapse_sku_buckets(s: Set[str]) -> str:
    if not s: return "none"
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
    if isinstance(g, str) and g.strip(): return g.strip().lower()
    im = row.get("is_male", None)
    if pd.notna(im):
        try: return "male" if int(im) == 1 else "female"
        except Exception: pass
    return "missing"

# -------------------------- Load & Prepare --------------------------
df = pd.read_csv(CSV_PATH, low_memory=False)
if "customer_id" not in df.columns:
    raise ValueError("customer_id column is required.")

# clean + normalize
df["primary_goal"] = df.get("primary_goal", pd.Series(dtype=object)).replace("-", "missing")
df["bm_pattern"] = df.get("bm_pattern", pd.Series(dtype=object)).replace("unspecified", "missing")
df["quiz_result"] = df.get("quiz_result", pd.Series(dtype=object)).apply(normalize_quiz_result)

# engineered fields on full df (so they exist for merging back)
df["quiz_bucket"] = df["quiz_result"].apply(quiz_to_bucket)
df["first_sku_buckets"] = df.get("first_sku", pd.Series(dtype=object)).apply(parse_first_sku_buckets)
df["first_sku_bucket"] = df["first_sku_buckets"].apply(collapse_sku_buckets)

df["quiz_reco_match"] = df.apply(
    lambda r: loose_match(r.get("quiz_bucket"), r.get("first_sku_buckets")), axis=1
)
df["acquisition_code_group"] = df.get("ancestor_discount_code", np.nan).apply(map_code_group)
df["gender_cat"] = df.apply(derive_gender, axis=1)

# quiz takers filter (and optional influencers only)
if "quiz_taker" not in df.columns:
    raise ValueError("quiz_taker column is required.")
df_quiz = df[df["quiz_taker"].astype(str).str.lower() == "yes"].copy()
if INFLUENCER_ONLY:
    df_quiz = df_quiz[df_quiz["acquisition_code_group"].ne("none")].copy()

# drop fields we won’t model directly
drop_now = [
    "acquisition_channel","first_sku","first_sku_buckets","quiz_bucket",
    "recent_abx_flag","email_key","quiz_date","ancestor_discount_code",
    "affiliate_segment","in_third_trimester_flag","stress_physical_flag","sx_brain_fog",
    "gross_ltv","net_ltv","avg_order_value","total_cogs","shipping_collected","shipping_spend","refund_amt",
    "first_order_date","last_order_date","quiz_taker","refund_ratio","first_sku_bucket"
]
df_quiz.drop(columns=[c for c in drop_now if c in df_quiz.columns], inplace=True, errors="ignore")

# coverage filter
non_null_threshold = COVERAGE * len(df_quiz)
df_quiz = df_quiz.loc[:, df_quiz.notnull().sum() >= non_null_threshold].copy()
print(f"[COVERAGE] rows={len(df_quiz)}, kept_cols={len(df_quiz.columns)}")

# -------------------------- Model frame (lean) --------------------------
cat_cols = [c for c in [
    "quiz_result","acquisition_code_group",
    "bm_pattern","gi_symptom_cat","primary_goal","gender_cat"
] if c in df_quiz.columns]

num_cols = [c for c in [
    "order_count","days_since_last_order","symptom_count",
    "high_stress","gut_issue_score","refund_count","quiz_reco_match"
] if c in df_quiz.columns]

use_cols = ["customer_id"] + cat_cols + num_cols
df_model = df_quiz[use_cols].copy()

for c in cat_cols:
    df_model[c] = df_model[c].fillna("missing").astype(str)
for c in num_cols:
    df_model[c] = pd.to_numeric(df_model[c], errors="coerce").fillna(0.0)

# -------------------------- FAMD inputs with filters --------------------------
famd_input = df_model.drop(columns=["customer_id"]).copy()
famd_cat = [c for c in famd_input.columns if famd_input[c].dtype == object]
famd_num = [c for c in famd_input.columns if c not in famd_cat]

keep_num = [c for c in famd_num if pd.to_numeric(famd_input[c], errors="coerce").fillna(0).std(ddof=0) > 1e-8]
keep_cat = []
for c in famd_cat:
    v = famd_input[c].astype(str).fillna("missing")
    vc = v.value_counts(normalize=True, dropna=False)
    if (vc.size >= 2) and (vc.iloc[0] <= DOMINANCE_THRESH):
        keep_cat.append(c)

famd_keep = keep_cat + keep_num
print(f"[FAMD KEEP] num={keep_num}, cat={keep_cat}")
if len(famd_keep) < 2:
    raise RuntimeError("Not enough variable features for FAMD. Loosen COVERAGE/DOMINANCE or include more fields.")

famd_input_filt = famd_input[famd_keep].copy()
for c in keep_cat:
    famd_input_filt[c] = famd_input_filt[c].astype("category")

# -------------------------- FAMD utilities --------------------------
def famd_scores_for_target(df_catnum: pd.DataFrame, target: float, n_max: int = 20) -> Tuple[np.ndarray, int, float]:
    """Fit FAMD with many comps, keep enough to reach `target` cumulative variance.
       Returns X (scores), n_keep, cumvar."""
    famd_full = FAMD(n_components=min(n_max, max(2, df_catnum.shape[1]-1)), random_state=SEED)
    famd_full.fit(df_catnum)
    # Explained fraction from eigenvalues_ (present in your FAMD class)
    eigs = np.asarray(famd_full.eigenvalues_, dtype=float)
    expl = eigs / eigs.sum()
    cum  = np.cumsum(expl)
    n_keep = int(np.searchsorted(cum, target) + 1)
    n_keep = max(2, min(n_keep, len(expl)))
    # Scores (row coordinates)
    scores_df = famd_full.row_coordinates(df_catnum)  # DataFrame
    X_all = scores_df.values if hasattr(scores_df, "values") else np.asarray(scores_df)
    X = X_all[:, :n_keep]
    return X, n_keep, float(cum[n_keep-1])

def plot_variance_for_target(expl: np.ndarray, outdir: str):
    cum = np.cumsum(expl)
    np.save(os.path.join(outdir, "famd_explained_inertia.npy"), expl)
    plt.figure(figsize=(7,4))
    plt.bar(np.arange(1, len(expl)+1), expl * 100.0)
    plt.xlabel("FAMD component"); plt.ylabel("Variance explained (%)")
    plt.title("FAMD variance explained (scree)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "famd_variance_scree.png"), dpi=150)
    plt.close()
    plt.figure(figsize=(7,4))
    plt.plot(np.arange(1, len(cum)+1), cum * 100.0, marker="o")
    plt.xlabel("FAMD component"); plt.ylabel("Cumulative variance (%)")
    plt.title("FAMD cumulative variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "famd_variance_cumulative.png"), dpi=150)
    plt.close()

# -------------------------- Sweep mode --------------------------
if RUN_SWEEP:
    print("\n[SWEEP] Running variance-target × algorithm sweep...")
    rows = []
    for vt in VAR_TARGETS:
        X, n_keep, cumvar = famd_scores_for_target(famd_input_filt, vt, n_max=20)

        # KMeans
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
            labels = km.fit_predict(X)
            s = silhouette_score(X, labels)
            rows.append({"var_target": vt, "n_keep": n_keep, "cumvar": cumvar,
                         "model": "kmeans", "params": f"k={k}", "coverage": 1.0, "silhouette": s})

        # GMM
        for k in range(2, 11):
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED,
                                  n_init=2, init_params="kmeans")
            labels = gmm.fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            s = silhouette_score(X, labels)
            rows.append({"var_target": vt, "n_keep": n_keep, "cumvar": cumvar,
                         "model": "gmm", "params": f"k={k}", "coverage": 1.0, "silhouette": s})

        # HDBSCAN (optional)
        if HAS_HDBSCAN:
            for mcs in (20, 30, 40):
                clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=None)
                labels = clusterer.fit_predict(X)
                mask = labels != -1
                labs = labels[mask]
                if mask.sum() < 10 or len(np.unique(labs)) < 2:
                    continue
                s = silhouette_score(X[mask], labs)
                cov = float(mask.mean())
                rows.append({"var_target": vt, "n_keep": n_keep, "cumvar": cumvar,
                             "model": "hdbscan", "params": f"min_cluster_size={mcs}",
                             "coverage": cov, "silhouette": s})

    sweep_df = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
    sweep_path = os.path.join(OUTDIR, "cluster_model_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print("\n=== Top configs by silhouette ===")
    print(sweep_df.head(10).to_string(index=False))
    print(f"\n[SWEEP] Wrote results to: {sweep_path}")
    sys.exit(0)

# -------------------------- Final mode (no sweep) --------------------------
# 1) Build factor space for FINAL_VAR_TARGET
X, n_keep, cumvar = famd_scores_for_target(famd_input_filt, FINAL_VAR_TARGET, n_max=20)
print(f"[FINAL] FAMD kept n={n_keep} comps ({cumvar*100:.1f}% variance) for clustering")

# Also save variance plots once for the report
# (Re-fit once with many comps to get expl fractions)
famd_full = FAMD(n_components=min(20, max(2, famd_input_filt.shape[1]-1)), random_state=SEED)
famd_full.fit(famd_input_filt)
eigs = np.asarray(famd_full.eigenvalues_, dtype=float)
expl = eigs / eigs.sum()
plot_variance_for_target(expl, OUTDIR)

# 2) Cluster
if FINAL_MODEL.lower() == "kmeans":
    k = int(FINAL_PARAMS.get("k", 4))
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    clusters = km.fit_predict(X)
    sil = silhouette_score(X, clusters)
    print(f"[FINAL] KMeans k={k} silhouette={sil:.4f}")

elif FINAL_MODEL.lower() == "gmm":
    k = int(FINAL_PARAMS.get("k", 4))
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED, n_init=2, init_params="kmeans")
    clusters = gmm.fit_predict(X)
    if len(np.unique(clusters)) < 2:
        raise RuntimeError("GMM produced a single cluster; adjust k.")
    sil = silhouette_score(X, clusters)
    print(f"[FINAL] GMM k={k} silhouette={sil:.4f}")

elif FINAL_MODEL.lower() == "hdbscan":
    if not HAS_HDBSCAN:
        raise RuntimeError("HDBSCAN not installed. pip install hdbscan")
    mcs = int(FINAL_PARAMS.get("min_cluster_size", 30))
    hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=None)
    clusters = hdb.fit_predict(X)
    mask = clusters != -1
    if mask.sum() < 10 or len(np.unique(clusters[mask])) < 2:
        raise RuntimeError("HDBSCAN found no stable clusters; adjust min_cluster_size.")
    sil = silhouette_score(X[mask], clusters[mask])
    coverage = float(mask.mean())
    print(f"[FINAL] HDBSCAN min_cluster_size={mcs} silhouette={sil:.4f} (coverage={coverage:.2f})")
else:
    raise ValueError("FINAL_MODEL must be one of: 'kmeans', 'gmm', 'hdbscan'")

# 3) Scatter (first two FAMD axes from the kept X)
plt.figure(figsize=(10,6))
plt.scatter(X[:,0], X[:,1], c=clusters, s=10, cmap="tab10")
plt.title("FAMD (colored by final clusters)")
plt.xlabel("FAMD 1"); plt.ylabel("FAMD 2")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "famd_kmeans_scatter.png"), dpi=150)
plt.close()

# 4) Merge labels + outputs (same as your pipeline)
df_labels = df_model[["customer_id"]].copy()
df_labels["cluster"] = clusters.astype(int)
df_out = df_labels.merge(df, on="customer_id", how="left")
df_out.to_csv(os.path.join(OUTDIR, "kmeans_clusters_with_meta.csv"), index=False)

# 5) Profiles + Personas
cat_profile_cols = [
    "quiz_result","bm_pattern","gi_symptom_cat",
    "acquisition_code_group","gender_cat"
]
num_profile_cols = ["order_count","days_since_last_order","symptom_count","gut_issue_score","high_stress","quiz_reco_match"]
flag_cols = [c for c in df_out.columns if c.startswith("sx_") or c.endswith("_flag")]

profiles = []
for cid, g in df_out.groupby("cluster"):
    rec = {"cluster": int(cid), "n_customers": int(len(g))}
    # top categories (+%)
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
    # quiz recommendation match
    if "quiz_reco_match" in g.columns:
        rec["quiz_reco_match_pct"] = round(pd.to_numeric(g["quiz_reco_match"], errors="coerce").fillna(0).mean()*100, 1)
    # promo mix (top 3)
    if "acquisition_code_group" in g.columns:
        promo = g["acquisition_code_group"].value_counts(normalize=True).head(3)
        for i, (code, share) in enumerate(promo.items(), start=1):
            rec[f"promo_{i}"] = f"{code}:{round(share*100,1)}%"
    profiles.append(rec)

df_profiles = pd.DataFrame(profiles).sort_values("cluster")
df_profiles.to_csv(os.path.join(OUTDIR, "kmeans_cluster_profiles.csv"), index=False)

def _safe(v): return v if isinstance(v, str) else "—"
persona_lines = []
for _, r in df_profiles.iterrows():
    quiz = _safe(r.get("quiz_result_top"))
    gi   = _safe(r.get("gi_symptom_cat_top"))
    bm   = _safe(r.get("bm_pattern_top"))
    gen  = _safe(r.get("gender_cat_top"))
    oc   = r.get("order_count_mean", 0)
    dsl  = r.get("days_since_last_order_mean", 0)
    sc   = r.get("symptom_count_mean", 0)
    qmr  = r.get("quiz_reco_match_pct", 0)
    p1   = _safe(r.get("promo_1")); p2 = _safe(r.get("promo_2"))

    blurb = (
        f"Cluster {int(r['cluster'])} — {int(r['n_customers'])} customers\n"
        f"- Quiz result: {quiz}; Gender: {gen}\n"
        f"- GI/BM: {gi}, {bm}; Avg symptoms: {sc}\n"
        f"- Behavior: avg orders {oc}, days since last order {dsl}; Quiz-match: {qmr}%\n"
        f"- Top promos: {p1} {p2}\n"
    )
    persona_lines.append(blurb)

with open(os.path.join(OUTDIR, "personas.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(persona_lines))

print("\nWrote:")
print(f" - {os.path.join(OUTDIR, 'kmeans_clusters_with_meta.csv')}")
print(f" - {os.path.join(OUTDIR, 'kmeans_cluster_profiles.csv')}")
print(f" - {os.path.join(OUTDIR, 'personas.txt')}")
print(f" - {os.path.join(OUTDIR, 'famd_kmeans_scatter.png')}")
print(f" - {os.path.join(OUTDIR, 'famd_variance_scree.png')}")
print(f" - {os.path.join(OUTDIR, 'famd_variance_cumulative.png')}")
