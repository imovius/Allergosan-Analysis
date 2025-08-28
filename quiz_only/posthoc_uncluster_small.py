# posthoc_uncluster_small.py
# Drop tiny clusters into an "Unclustered" (-1) bucket and regenerate personas/profiles.
# Run AFTER two_pass_personas_value.py so the label files exist.

import os, re
import numpy as np
import pandas as pd

CSV_PATH = "../raw_data_v3.csv"
OUTDIR   = "outputs"
SEED     = 42

# ---- settings ----
MIN_SIZE_PCT = 0.05   # clusters smaller than 5% of sample -> unclustered
PASSES = [
    # (tag, expected_cluster_col_in_labels_csv, labels_filename)
    ("A_withLTV",  "cluster_A_withLTV",  "A_withLTV_labels.csv"),
    ("B_noLTV",    "cluster_B_noLTV",    "B_noLTV_labels.csv"),
]

# -------- helpers (match your main pipeline) --------
SKU_TO_BUCKET = {
    "OBSR":"stress","S-OBSR":"stress",
    "OBAB":"gut_restoration","3MGHR":"gut_restoration",
    "OBHX":"detox","S-OBHX":"detox",
    "OBBA":"immune","OLIM":"immune",
    "OBPO":"metabolism","OLPL":"metabolism","S-OBP":"metabolism",
    "OBPA":"prenatal","S-OBPA":"prenatal",
    "OBCG":"other_ob","OBCA":"other_ob","OBCD":"other_ob",
    "OBSC":"accessory","LBUN":"bundle","IMBUN":"bundle","DYNDUO":"bundle",
}
QUIZ_NORMALIZE = {
    "OMNi-BiOTiC BALANCE":"Balance","OMNi-BiOTiCA® BALANCE":"Balance",
    "OMNi-BiOTiC Stress Release":"Stress Release","OMNi-BiOTiCA® Stress Release":"Stress Release",
    "OMNi-BiOTiC HETOX":"Hetox","OMNi-BiOTiCA® HETOX":"Hetox",
    "Omni-Biotic Power":"Power","OMNi-BiOTiC Panda":"Panda",
    "OMNi-BiOTiC AB 10":"AB 10","OMNi-BiOTiCA® AB 10":"AB 10",
    "Gut Health Reset Program":"Gut Health Reset",
}
QUIZ_LINE_TO_BUCKET = {
    "Stress Release":"stress","Balance":"immune","Hetox":"detox",
    "AB 10":"gut_restoration","Gut Health Reset":"gut_restoration",
    "Panda":"prenatal","Power":"metabolism",
}
INFLUENCER_CODES = {c.lower() for c in {
    "dave20","dave","jessica15","drwillcole","dr.cain15","valeria20",
    "skinny","blonde","blonde20","carly15","tammy15","sweats15"
}}

def normalize_quiz_result(q):
    if not isinstance(q, str): return None
    q = q.strip()
    return QUIZ_NORMALIZE.get(q, q) if q else None

def quiz_to_bucket(quiz_result):
    if not isinstance(quiz_result, str): return None
    line = normalize_quiz_result(quiz_result)
    return QUIZ_LINE_TO_BUCKET.get(line) if line else None

def parse_first_sku_buckets(first_sku):
    s = set()
    if isinstance(first_sku, str) and first_sku.strip():
        for tok in [t.strip() for t in first_sku.split(",")]:
            b = SKU_TO_BUCKET.get(tok)
            if b: s.add(b)
    return s

def collapse_sku_buckets(s):
    if not s: return "none"
    return sorted(list(s))[0] if len(s)==1 else "multi"

def loose_match(q_bucket, sku_buckets):
    return int(q_bucket in sku_buckets) if q_bucket and sku_buckets else 0

def map_code_group(x):
    if isinstance(x, str) and x.strip():
        v = x.strip().lower()
        return v if v in INFLUENCER_CODES else "other"
    return "none"

def derive_gender(row):
    g = row.get("gender", None)
    if isinstance(g, str) and g.strip(): return g.strip().lower()
    im = row.get("is_male", None)
    try:
        return "male" if int(im)==1 else "female"
    except Exception:
        return "missing"

# -------- load & prep base df (quiz takers only; same as main) --------
df = pd.read_csv(CSV_PATH, low_memory=False)
if "customer_id" not in df.columns:
    raise ValueError("customer_id column is required")

df["primary_goal"] = df.get("primary_goal", pd.Series(dtype=object)).replace("-", "missing")
df["bm_pattern"] = df.get("bm_pattern", pd.Series(dtype=object)).replace("unspecified", "missing")
df["quiz_result"] = df.get("quiz_result", pd.Series(dtype=object)).apply(normalize_quiz_result)
df["quiz_bucket"] = df["quiz_result"].apply(quiz_to_bucket)
df["first_sku_buckets"] = df.get("first_sku", pd.Series(dtype=object)).apply(parse_first_sku_buckets)
df["first_sku_bucket"] = df["first_sku_buckets"].apply(collapse_sku_buckets)
df["quiz_reco_match"] = df.apply(lambda r: loose_match(r.get("quiz_bucket"), r.get("first_sku_buckets")), axis=1)
df["acquisition_code_group"] = df.get("ancestor_discount_code", np.nan).apply(map_code_group)
df["gender_cat"] = df.apply(derive_gender, axis=1)

if "quiz_taker" not in df.columns:
    raise ValueError("quiz_taker column is required")
df_quiz = df[df["quiz_taker"].astype(str).str.lower()=="yes"].copy()

# columns used in profiles
CAT_PROFILE = ["quiz_result","bm_pattern","gi_symptom_cat","acquisition_code_group","gender_cat","first_sku_bucket"]
NUM_PROFILE = ["order_count","days_since_last_order","symptom_count","gut_issue_score","high_stress","refund_count","quiz_reco_match","net_ltv","avg_order_value","gross_ltv"]

for c in CAT_PROFILE:
    if c in df_quiz.columns:
        df_quiz[c] = df_quiz[c].fillna("missing").astype(str)
for c in NUM_PROFILE:
    if c in df_quiz.columns:
        df_quiz[c] = pd.to_numeric(df_quiz[c], errors="coerce")

os.makedirs(OUTDIR, exist_ok=True)

def relabel_small_and_reprofile(tag: str, expected_cluster_col: str, label_file: str):
    labels_path = os.path.join(OUTDIR, label_file)
    if not os.path.exists(labels_path):
        print(f"[WARN] Missing labels for {tag}: {labels_path}")
        return

    df_labels = pd.read_csv(labels_path)

    # ---- auto-detect the cluster column (use expected first) ----
    cluster_col = None
    if expected_cluster_col and expected_cluster_col in df_labels.columns:
        cluster_col = expected_cluster_col
    else:
        for cand in ["cluster", f"cluster_{tag}"]:
            if cand in df_labels.columns:
                cluster_col = cand
                break
        if cluster_col is None:
            fallback = [c for c in df_labels.columns if c.startswith("cluster")]
            if fallback:
                cluster_col = fallback[0]

    if cluster_col is None or "customer_id" not in df_labels.columns:
        raise ValueError(
            f"{label_file} must include 'customer_id' and a cluster column "
            f"(got columns: {list(df_labels.columns)})"
        )

    # normalize to 'cluster' for downstream
    if cluster_col != "cluster":
        df_labels = df_labels.rename(columns={cluster_col: "cluster"})

    n = len(df_labels)
    min_size = max(1, int(np.floor(MIN_SIZE_PCT * n)))
    sizes = df_labels["cluster"].value_counts()
    small = sizes[sizes < min_size].index.tolist()

    df_labels["cluster_final"] = df_labels["cluster"].apply(lambda c: -1 if c in small else int(c))
    df_labels.to_csv(os.path.join(OUTDIR, f"{tag}_labels_pruned.csv"), index=False)

    # merge back to data
    merged = df_labels.merge(df_quiz, on="customer_id", how="left")

    # build profiles (exclude -1 from per-cluster %/means but report size)
    profiles = []
    valid = merged[merged["cluster_final"]!=-1].copy()

    for cid, g in valid.groupby("cluster_final"):
        rec = {"cluster": int(cid), "n_customers": int(len(g))}
        # top categoricals
        for col in CAT_PROFILE:
            if col in g.columns:
                vc = g[col].value_counts(normalize=True)
                if len(vc):
                    rec[f"{col}_top"] = vc.index[0]; rec[f"{col}_pct"] = round(vc.iloc[0]*100,1)
        # numerics
        for col in NUM_PROFILE:
            if col in g.columns:
                s = pd.to_numeric(g[col], errors="coerce")
                rec[f"{col}_mean"] = round(s.mean(skipna=True),2)
                rec[f"{col}_median"] = round(s.median(skipna=True),2)
        # quiz match
        if "quiz_reco_match" in g.columns:
            rec["quiz_reco_match_pct"] = round(pd.to_numeric(g["quiz_reco_match"], errors="coerce").fillna(0).mean()*100,1)
        # promos
        if "acquisition_code_group" in g.columns:
            promo = g["acquisition_code_group"].value_counts(normalize=True).head(3)
            for i,(code,share) in enumerate(promo.items(), start=1):
                rec[f"promo_{i}"] = f"{code}:{round(share*100,1)}%"
        profiles.append(rec)

    df_profiles = pd.DataFrame(profiles).sort_values("cluster")
    df_profiles.to_csv(os.path.join(OUTDIR, f"{tag}_cluster_profiles_clean.csv"), index=False)

    # value by cluster (mean LTV etc.)
    keep = ["cluster","n_customers","net_ltv_mean","net_ltv_median","avg_order_value_mean","avg_order_value_median"]
    df_value = df_profiles[keep].copy()
    df_value.to_csv(os.path.join(OUTDIR, f"{tag}_value_by_cluster_clean.csv"), index=False)

    # personas (skip rich text for -1)
    def _safe(v): return v if isinstance(v, str) else "—"
    lines = []
    for _, r in df_profiles.iterrows():
        lines.append(
            f"{tag} — Cluster {int(r['cluster'])} — {int(r['n_customers'])} customers\n"
            f"- Quiz result: {_safe(r.get('quiz_result_top'))}; Gender: {_safe(r.get('gender_cat_top'))}; First-SKU: {_safe(r.get('first_sku_bucket_top'))}\n"
            f"- GI/BM: {_safe(r.get('gi_symptom_cat_top'))}, {_safe(r.get('bm_pattern_top'))}; Avg symptoms: {r.get('symptom_count_mean','—')}\n"
            f"- Behavior: avg orders {r.get('order_count_mean','—')}, days since last order {r.get('days_since_last_order_mean','—')}; Quiz-match: {r.get('quiz_reco_match_pct','—')}%\n"
            f"- VALUE: mean LTV ${r.get('net_ltv_mean','—')}, mean AOV ${r.get('avg_order_value_mean','—')}\n"
            f"- Top promos: {_safe(r.get('promo_1'))} {_safe(r.get('promo_2'))}\n"
        )

    # add unclustered summary
    un = merged[merged["cluster_final"]==-1]
    if len(un):
        pct_un = 100.0 * len(un) / n
        lines.append(f"{tag} — Unclustered (-1) — {len(un)} customers ({pct_un:.1f}% of sample)")

    with open(os.path.join(OUTDIR, f"{tag}_personas_clean.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # bar plot of mean LTV (incl. unclustered as -1 if exists)
    plot_df = pd.concat([
        valid.assign(cluster=valid["cluster_final"]),
        un.assign(cluster=-1)
    ], ignore_index=True)

    plot_df["net_ltv"] = pd.to_numeric(plot_df["net_ltv"], errors="coerce")
    plot_data = plot_df.groupby("cluster")["net_ltv"].mean().reset_index()

    # Use matplotlib directly to avoid environment surprises
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,4))
    plt.bar(plot_data["cluster"].astype(str), plot_data["net_ltv"])
    plt.title(f"{tag}: Mean LTV by cluster (clean)")
    plt.xlabel("Cluster")
    plt.ylabel("Mean net LTV")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"{tag}_mean_ltv_by_cluster_clean.png"), dpi=150)
    plt.close()

    print(f"[{tag}] small clusters -> unclustered: {small}  (min_size={min_size})")
    print(f"[{tag}] wrote: *_clean.csv/txt/png")

# ---- run for both passes ----
for tag, expected_col, file in PASSES:
    relabel_small_and_reprofile(tag, expected_col, file)

print("\nDone. Cleaned personas/profiles/value files written to outputs/.")
