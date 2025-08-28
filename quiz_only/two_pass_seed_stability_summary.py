# two_pass_personas_value_with_stability.py
# Personas + Value, two-pass clustering (with-LTV vs without-LTV) using FAMD -> KMeans
# PLUS: Within-pass seed stability check using ARI
# Python 3.9+

import os, re, sys
import numpy as np
import pandas as pd
from typing import Optional, Set, Dict, Tuple, List
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from prince import FAMD  # assumes your env's FAMD with eigenvalues_ & column_contributions_

# ======================= CONFIG =======================
CSV_PATH = "../raw_data_v3.csv"
OUTDIR   = "outputs"
SEED     = 42         # modeling seed (original run)
NEW_SEED = 123        # stability seed (rerun for ARI)

# Data filters
INFLUENCER_ONLY   = False   # True: promo purchasers only; False: all quiz takers
COVERAGE          = 0.60    # keep columns with >=60% non-null (looser to bring back features)
DOMINANCE_THRESH  = 0.995   # allow dominated categoricals to pass (e.g., first_sku_bucket)

# FAMD (both passes)
EXPLAINED_TARGET  = 0.85    # keep ~85% cumulative variance
N_MAX_COMPONENTS  = 20      # upper bound for components to learn
PRUNE_FRACTION    = 0.15    # drop bottom 15% by total contribution (first n_keep comps)
MIN_CONTRIB_PCT   = 0.40    # also drop any <0.4% contribution

# Clustering
K_RANGE           = range(2, 11)  # candidate k values
FORCE_K_A         = None          # force k for Pass A (with LTV), or None
FORCE_K_B         = None          # force k for Pass B (behavior only), or None

DPI = 150
# ======================================================
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(SEED)

# -------------------------- Helpers --------------------------
def _sanitize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

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
    "Stress Release":"stress","Balance":"immune","Hetox":"detox",
    "AB 10":"gut_restoration","Gut Health Reset":"gut_restoration",
    "Panda":"prenatal","Power":"metabolism",
}
INFLUENCER_CODES = {c.lower() for c in {
    "dave20","dave","jessica15","drwillcole","dr.cain15","valeria20",
    "skinny","blonde","blonde20","carly15","tammy15","sweats15"
}}

def normalize_quiz_result(q: Optional[str]) -> Optional[str]:
    if not isinstance(q, str): return None
    qn = q.strip()
    if not qn: return None
    return QUIZ_NORMALIZE.get(qn, qn)

def quiz_to_bucket(quiz_result: Optional[str]) -> Optional[str]:
    if not isinstance(quiz_result, str): return None
    line = normalize_quiz_result(quiz_result)
    return QUIZ_LINE_TO_BUCKET.get(line) if line else None

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

# Normalize / engineer
df["primary_goal"] = df.get("primary_goal", pd.Series(dtype=object)).replace("-", "missing")
df["bm_pattern"] = df.get("bm_pattern", pd.Series(dtype=object)).replace("unspecified", "missing")
df["quiz_result"] = df.get("quiz_result", pd.Series(dtype=object)).apply(normalize_quiz_result)
df["quiz_bucket"] = df["quiz_result"].apply(quiz_to_bucket)
df["first_sku_buckets"] = df.get("first_sku", pd.Series(dtype=object)).apply(parse_first_sku_buckets)
df["first_sku_bucket"] = df["first_sku_buckets"].apply(collapse_sku_buckets)
df["quiz_reco_match"] = df.apply(lambda r: loose_match(r.get("quiz_bucket"), r.get("first_sku_buckets")), axis=1)
df["acquisition_code_group"] = df.get("ancestor_discount_code", np.nan).apply(map_code_group)
df["gender_cat"] = df.apply(derive_gender, axis=1)

# Filter to quiz takers (+ influencers if desired)
if "quiz_taker" not in df.columns:
    raise ValueError("quiz_taker column is required.")
df_quiz = df[df["quiz_taker"].astype(str).str.lower() == "yes"].copy()
if INFLUENCER_ONLY:
    df_quiz = df_quiz[df_quiz["acquisition_code_group"].ne("none")].copy()

# Drop only hard leakage/IDs; keep value fields for profiling & Pass A features
drop_now = [
    "email_key","quiz_date","first_order_date","last_order_date",
    "ancestor_discount_code","quiz_taker","first_sku","first_sku_buckets",
    "total_cogs","shipping_collected","shipping_spend","refund_amt","refund_ratio",
    "recent_abx_flag","affiliate_segment","avg_order_value","gross_ltv"
]
df_quiz.drop(columns=[c for c in drop_now if c in df_quiz.columns], inplace=True, errors="ignore")

# Coverage filter (looser)
non_null_threshold = COVERAGE * len(df_quiz)
df_quiz = df_quiz.loc[:, df_quiz.notnull().sum() >= non_null_threshold].copy()
print(f"[COVERAGE] rows={len(df_quiz)}, kept_cols={len(df_quiz.columns)}")

# -------------------------- Common feature definitions --------------------------
BASE_CAT = [
    "quiz_result","acquisition_code_group","bm_pattern","gi_symptom_cat",
    "primary_goal","gender_cat","first_sku_bucket"
]
BASE_NUM = [
    "order_count","days_since_last_order","symptom_count","gut_issue_score",
    "high_stress","refund_count","quiz_reco_match"
]
VALUE_COLS = [c for c in ["net_ltv","avg_order_value","gross_ltv"] if c in df_quiz.columns]

for c in BASE_CAT:
    if c in df_quiz.columns:
        df_quiz[c] = df_quiz[c].fillna("missing").astype(str)
for c in BASE_NUM:
    if c in df_quiz.columns:
        df_quiz[c] = pd.to_numeric(df_quiz[c], errors="coerce").fillna(0.0)

# -------------------------- Utilities --------------------------
def famd_fit_scores(df_catnum: pd.DataFrame, explained_target: float, n_max: int, seed: int):
    famd_full = FAMD(n_components=min(n_max, max(2, df_catnum.shape[1]-1)), random_state=seed)
    famd_full.fit(df_catnum)
    eigs = np.asarray(famd_full.eigenvalues_, dtype=float)
    expl = eigs / eigs.sum()
    cum  = np.cumsum(expl)
    n_keep = int(np.searchsorted(cum, explained_target) + 1)
    n_keep = max(2, min(n_keep, len(expl)))
    scores = famd_full.row_coordinates(df_catnum)
    X_all = scores.values if hasattr(scores, "values") else np.asarray(scores)
    X = X_all[:, :n_keep]
    return famd_full, X, n_keep, expl

def famd_contrib_prune(famd_model: FAMD, n_keep: int, famd_vars: List[str], frac: float, min_pct: float) -> List[str]:
    contrib = famd_model.column_contributions_.copy()
    contrib_first = contrib.iloc[:, :n_keep]
    totals = (contrib_first.sum(axis=1))
    totals_pct = totals / totals.sum() * 100.0
    rank = pd.DataFrame({"variable": totals_pct.index, "total_contrib_pct": totals_pct.values}).sort_values("total_contrib_pct", ascending=False)
    rank.to_csv(os.path.join(OUTDIR, "famd_feature_contributions_tmp.csv"), index=False)
    n_prune = int(np.floor(frac * len(rank)))
    to_drop = set(rank.tail(n_prune)["variable"].tolist()) | set(rank[rank["total_contrib_pct"] < min_pct]["variable"].tolist())
    keep_vars = [v for v in famd_vars if v not in to_drop]
    pd.DataFrame({"dropped_feature": sorted(list(to_drop))}).to_csv(os.path.join(OUTDIR, "pruned_features_tmp.csv"), index=False)
    return keep_vars

def choose_k_by_silhouette(X: np.ndarray, k_range=range(2,11), seed=42, label=""):
    sil_curve = []
    best_k, best_s = None, -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=30)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        sil_curve.append((k, s))
        if s > best_s:
            best_k, best_s = k, s
        print(f"[{label}] k={k} silhouette={s:.4f}")
    return best_k, best_s, sil_curve

def plot_sil_curve(sil_curve, path, title):
    ks = [k for k,_ in sil_curve]; ss = [s for _,s in sil_curve]
    pd.DataFrame(sil_curve, columns=["k","silhouette"]).to_csv(path.replace(".png",".csv"), index=False)
    plt.figure(figsize=(7,4)); plt.plot(ks, ss, marker="o")
    plt.xlabel("k"); plt.ylabel("Silhouette"); plt.title(title)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=DPI); plt.close()


def save_famd_explained(tag: str, stage: int, expl: np.ndarray, n_keep: int):
    """
    Save explained-variance vector, cumulative table, and a scree plot.
    stage = 1 (pre-prune) or 2 (post-prune).
    """
    base = os.path.join(OUTDIR, f"{tag}_famd{stage}")

    # 1) raw npy
    np.save(f"{base}_explained.npy", expl)

    # 2) CSV with cumulative + retained flag
    cum = np.cumsum(expl)
    df_expl = pd.DataFrame({
        "component": np.arange(1, len(expl)+1),
        "explained_var": expl,
        "cumulative_var": cum,
        "retained": [i <= n_keep for i in range(1, len(expl)+1)]
    })
    df_expl.to_csv(f"{base}_explained.csv", index=False)

    # 3) scree & cumulative plot
    plt.figure(figsize=(8,4))
    plt.plot(df_expl["component"], df_expl["explained_var"], marker="o", label="Explained var")
    plt.plot(df_expl["component"], df_expl["cumulative_var"], marker="o", label="Cumulative")
    if n_keep is not None:
        plt.axvline(n_keep, linestyle="--", alpha=0.7, label=f"n_keep={n_keep}")
    plt.xlabel("Component")
    plt.ylabel("Proportion of variance")
    plt.title(f"{tag} — FAMD{stage} variance explained")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base}_explained.png", dpi=DPI)
    plt.close()


def cluster_and_profile(tag: str, df_base: pd.DataFrame, famd_vars_cat: List[str], famd_vars_num: List[str],
                        include_value_in_famd: bool, force_k: Optional[int]):
    # Build famd input
    cat_cols = [c for c in famd_vars_cat if c in df_base.columns]
    num_cols = [c for c in famd_vars_num if c in df_base.columns]
    famd_vars = cat_cols + num_cols
    famd_input = df_base[famd_vars].copy()

    # dominance filter
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
    famd_input_filt = famd_input[famd_keep].copy()
    for c in keep_cat:
        famd_input_filt[c] = famd_input_filt[c].astype("category")

    # FAMD fit #1
    famd1, X1, n_keep1, expl1 = famd_fit_scores(famd_input_filt, EXPLAINED_TARGET, N_MAX_COMPONENTS, SEED)
    cumvar1 = float(np.cumsum(expl1)[n_keep1-1]); print(f"[{tag}] FAMD#1 kept {n_keep1} comps ({cumvar1*100:.1f}%)")
    save_famd_explained(tag, 1, expl1, n_keep1)
    # prune
    keep_vars = famd_contrib_prune(famd1, n_keep1, famd_keep, PRUNE_FRACTION, MIN_CONTRIB_PCT)
    if len(keep_vars) < 2: keep_vars = famd_keep

    # FAMD fit #2 (pruned)
    famd_input_pruned = famd_input[keep_vars].copy()
    for c in keep_cat:
        if c in famd_input_pruned.columns:
            famd_input_pruned[c] = famd_input_pruned[c].astype("category")
    famd2, X2, n_keep2, expl2 = famd_fit_scores(famd_input_pruned, EXPLAINED_TARGET, N_MAX_COMPONENTS, SEED)
    cumvar2 = float(np.cumsum(expl2)[n_keep2-1]); print(f"[{tag}] FAMD#2 kept {n_keep2} comps ({cumvar2*100:.1f}%)")
    save_famd_explained(tag, 2, expl2, n_keep2)   # <— add this

    # choose k
    forced = force_k if isinstance(force_k, int) else None
    if forced is None:
        best_k, best_s, sil_curve = choose_k_by_silhouette(X2, K_RANGE, SEED, label=tag)
        chosen_k = best_k
    else:
        km = KMeans(n_clusters=forced, random_state=SEED, n_init=30)
        labs = km.fit_predict(X2)
        best_s = silhouette_score(X2, labs)
        sil_curve = [(k, np.nan) for k in K_RANGE]
        chosen_k = forced
        print(f"[{tag}] FORCED k={chosen_k} silhouette={best_s:.4f}")

    plot_sil_curve(sil_curve, os.path.join(OUTDIR, f"{tag}_silhouette.png"),
                   f"{tag}: Silhouette by k (FAMD pruned)")

    # final cluster labels (original run)
    kmeans = KMeans(n_clusters=chosen_k, random_state=SEED, n_init=50)
    clusters = kmeans.fit_predict(X2)

    # SAVE FAMD scores and labels for stability checks
    np.save(os.path.join(OUTDIR, f"{tag}_famd_scores.npy"), X2)
    df_labels = df_base[["customer_id"]].copy()
    df_labels[f"cluster_{tag}"] = clusters.astype(int)
    df_labels.to_csv(os.path.join(OUTDIR, f"{tag}_labels.csv"), index=False)

    # scatter of first two components
    plt.figure(figsize=(10,6))
    plt.scatter(X2[:,0], X2[:,1], c=clusters, s=10, cmap="tab10")
    plt.title(f"{tag}: FAMD (pruned) — KMeans k={chosen_k}")
    plt.xlabel("FAMD 1"); plt.ylabel("FAMD 2")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, f"{tag}_famd_scatter.png"), dpi=DPI); plt.close()

    # build profiles (include value overlays regardless of whether LTV was used)
    df_join = df_labels.merge(df_quiz, on="customer_id", how="left")

    cat_profile_cols = [
        "quiz_result","bm_pattern","gi_symptom_cat",
        "acquisition_code_group","gender_cat","first_sku_bucket"
    ]
    num_profile_cols = [
        "order_count","days_since_last_order","symptom_count","gut_issue_score",
        "high_stress","refund_count","quiz_reco_match",
        # value overlays
        "net_ltv","avg_order_value","gross_ltv"
    ]

    profiles = []
    for cid, g in df_join.groupby(f"cluster_{tag}"):
        rec = {"cluster": int(cid), "n_customers": int(len(g))}
        for col in cat_profile_cols:
            if col in g.columns:
                vc = g[col].value_counts(normalize=True)
                if len(vc):
                    rec[f"{col}_top"] = vc.index[0]
                    rec[f"{col}_pct"] = round(vc.iloc[0]*100, 1)
        for col in num_profile_cols:
            if col in g.columns:
                series = pd.to_numeric(g[col], errors="coerce")
                rec[f"{col}_mean"] = round(series.fillna(0).mean(), 2)
                rec[f"{col}_median"] = round(series.median(), 2)
        if "quiz_reco_match" in g.columns:
            rec["quiz_reco_match_pct"] = round(pd.to_numeric(g["quiz_reco_match"], errors="coerce").fillna(0).mean()*100, 1)
        if "acquisition_code_group" in g.columns:
            promo = g["acquisition_code_group"].value_counts(normalize=True).head(3)
            for i, (code, share) in enumerate(promo.items(), start=1):
                rec[f"promo_{i}"] = f"{code}:{round(share*100,1)}%"
        profiles.append(rec)

    df_profiles = pd.DataFrame(profiles).sort_values("cluster")
    df_profiles.to_csv(os.path.join(OUTDIR, f"{tag}_cluster_profiles.csv"), index=False)

    # personas file
    def _safe(v): return v if isinstance(v, str) else "—"
    persona_lines = []
    for _, r in df_profiles.iterrows():
        blurb = (
            f"{tag} — Cluster {int(r['cluster'])} — {int(r['n_customers'])} customers\n"
            f"- Quiz result: {_safe(r.get('quiz_result_top'))}; Gender: {_safe(r.get('gender_cat_top'))}; First-SKU: {_safe(r.get('first_sku_bucket_top'))}\n"
            f"- GI/BM: {_safe(r.get('gi_symptom_cat_top'))}, {_safe(r.get('bm_pattern_top'))}; Avg symptoms: {r.get('symptom_count_mean',0)}\n"
            f"- Behavior: avg orders {r.get('order_count_mean',0)}, days since last order {r.get('days_since_last_order_mean',0)}; Quiz-match: {r.get('quiz_reco_match_pct',0)}%\n"
            f"- VALUE: mean LTV ${r.get('net_ltv_mean','—')}, mean AOV ${r.get('avg_order_value_mean','—')}\n"
            f"- Top promos: {_safe(r.get('promo_1'))} {_safe(r.get('promo_2'))}\n"
        )
        persona_lines.append(blurb)
    with open(os.path.join(OUTDIR, f"{tag}_personas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(persona_lines))

    return {
        "tag": tag,
        "chosen_k": chosen_k,
        "best_s": best_s,
        "n_keep": n_keep2,
        "cumvar": float(np.cumsum(expl2)[n_keep2-1]),
        "labels_path": os.path.join(OUTDIR, f"{tag}_labels.csv"),
        "scores_path": os.path.join(OUTDIR, f"{tag}_famd_scores.npy"),
        "profiles": df_profiles,
    }

# -------------------------- Build base modeling frame --------------------------
keep_cols = ["customer_id"] + BASE_CAT + BASE_NUM + [c for c in VALUE_COLS if c in df_quiz.columns]
df_model_base = df_quiz[keep_cols].copy()

# -------------------------- PASS A: WITH LTV in FAMD --------------------------
print("\n=== PASS A: WITH LTV in clustering ===")
passA_cat = BASE_CAT[:]
passA_num = BASE_NUM[:] + [c for c in VALUE_COLS if c in df_model_base.columns]
resA = cluster_and_profile("A_withLTV", df_model_base, passA_cat, passA_num, True, FORCE_K_A)

# -------------------------- PASS B: WITHOUT LTV in FAMD -----------------------
print("\n=== PASS B: WITHOUT LTV in clustering (value overlay only) ===")
passB_cat = BASE_CAT[:]
passB_num = BASE_NUM[:]
resB = cluster_and_profile("B_noLTV", df_model_base, passB_cat, passB_num, False, FORCE_K_B)

# -------------------------- Comparison: value by cluster across passes --------
def value_summary(df_profiles, tag):
    keep = ["cluster","n_customers","net_ltv_mean","net_ltv_median",]
    return df_profiles[keep].copy().assign(tag=tag)

valA = value_summary(resA["profiles"], "A_withLTV")
valB = value_summary(resB["profiles"], "B_noLTV")
valA.to_csv(os.path.join(OUTDIR, "A_value_by_cluster.csv"), index=False)
valB.to_csv(os.path.join(OUTDIR, "B_value_by_cluster.csv"), index=False)

def bar_value_plot(dfv, title, path):
    plt.figure(figsize=(8,4))
    plt.bar(dfv["cluster"].astype(str), pd.to_numeric(dfv["net_ltv_mean"], errors="coerce").fillna(0))
    plt.xlabel("Cluster"); plt.ylabel("Mean net LTV"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=DPI); plt.close()

bar_value_plot(valA.sort_values("cluster"), "Pass A (with LTV): mean net LTV by cluster", os.path.join(OUTDIR, "A_mean_ltv_by_cluster.png"))
bar_value_plot(valB.sort_values("cluster"), "Pass B (no LTV): mean net LTV by cluster", os.path.join(OUTDIR, "B_mean_ltv_by_cluster.png"))

with open(os.path.join(OUTDIR, "two_pass_summary.txt"), "w", encoding="utf-8") as f:
    f.write(
        f"PASS A (with LTV): k={resA['chosen_k']}, silhouette={resA['best_s']:.4f}, FAMD comps kept={resA['n_keep']} ({resA['cumvar']*100:.1f}% variance)\n"
        f"PASS B (no LTV):  k={resB['chosen_k']}, silhouette={resB['best_s']:.4f}, FAMD comps kept={resB['n_keep']} ({resB['cumvar']*100:.1f}% variance)\n"
        "Files: *_cluster_profiles.csv, *_personas.txt, *_silhouette.png, *_famd_scatter.png, A_value_by_cluster.csv, B_value_by_cluster.csv\n"
    )

# -------------------------- WITHIN-PASS SEED STABILITY (THIS IS THE ASK) -----
def seed_stability(tag: str, new_seed: int = NEW_SEED):
    scores_path = os.path.join(OUTDIR, f"{tag}_famd_scores.npy")
    labels_path = os.path.join(OUTDIR, f"{tag}_labels.csv")
    if not (os.path.exists(scores_path) and os.path.exists(labels_path)):
        print(f"[{tag}] Missing scores or labels. Skipping stability.")
        return None

    X = np.load(scores_path)
    df_labels = pd.read_csv(labels_path)
    cluster_col = f"cluster_{tag}"
    if cluster_col not in df_labels.columns:
        # fallback to the unique cluster_* column
        cand = [c for c in df_labels.columns if c.startswith("cluster_")]
        if len(cand) == 1:
            cluster_col = cand[0]
        else:
            raise ValueError(f"{labels_path} must include '{cluster_col}'")

    original_labels = df_labels[cluster_col].values.astype(int)
    n_clusters = len(np.unique(original_labels))

    km_new = KMeans(n_clusters=n_clusters, random_state=new_seed, n_init=50)
    new_labels = km_new.fit_predict(X)
    ari = adjusted_rand_score(original_labels, new_labels)

    # size deltas
    orig_sizes = pd.Series(original_labels).value_counts().sort_index()
    new_sizes  = pd.Series(new_labels).value_counts().sort_index()
    size_df = pd.DataFrame({"orig_size": orig_sizes, "new_size": new_sizes})
    size_df["delta"] = size_df["new_size"] - size_df["orig_size"]
    size_df.to_csv(os.path.join(OUTDIR, f"{tag}_stability_sizes.csv"))

    comp = pd.DataFrame({
        "customer_id": df_labels["customer_id"],
        "cluster_original": original_labels,
        "cluster_newseed": new_labels
    })
    comp.to_csv(os.path.join(OUTDIR, f"{tag}_labels_stability.csv"), index=False)

    print(f"[{tag}] ARI (seed {SEED} vs {new_seed}) = {ari:.3f}")
    return {"tag": tag, "ari": float(ari), "n_clusters": int(n_clusters), "n_samples": int(X.shape[0])}

stabA = seed_stability("A_withLTV", NEW_SEED)
stabB = seed_stability("B_noLTV", NEW_SEED)

stab_rows = [r for r in [stabA, stabB] if r is not None]
if stab_rows:
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(os.path.join(OUTDIR, "two_pass_seed_stability_summary.csv"), index=False)
    print("\nSeed stability summary (ARI):")
    print(stab_df.to_string(index=False))

print("\nWrote:")
print(f" - {os.path.join(OUTDIR, 'A_withLTV_cluster_profiles.csv')}")
print(f" - {os.path.join(OUTDIR, 'B_noLTV_cluster_profiles.csv')}")
print(f" - {os.path.join(OUTDIR, 'A_withLTV_personas.txt')}")
print(f" - {os.path.join(OUTDIR, 'B_noLTV_personas.txt')}")
print(f" - {os.path.join(OUTDIR, 'A_withLTV_silhouette.png')}, {os.path.join(OUTDIR, 'B_noLTV_silhouette.png')}")
print(f" - {os.path.join(OUTDIR, 'A_withLTV_famd_scatter.png')}, {os.path.join(OUTDIR, 'B_noLTV_famd_scatter.png')}")
print(f" - {os.path.join(OUTDIR, 'A_value_by_cluster.csv')}, {os.path.join(OUTDIR, 'B_value_by_cluster.csv')}")
print(f" - {os.path.join(OUTDIR, 'A_mean_ltv_by_cluster.png')}, {os.path.join(OUTDIR, 'B_mean_ltv_by_cluster.png')}")
print(f" - {os.path.join(OUTDIR, 'two_pass_summary.txt')}")
print(f" - {os.path.join(OUTDIR, 'two_pass_seed_stability_summary.csv')}")
print(f" - {os.path.join(OUTDIR, 'A_withLTV_labels.csv')}, {os.path.join(OUTDIR, 'B_noLTV_labels.csv')}")
print(f" - {os.path.join(OUTDIR, 'A_withLTV_famd_scores.npy')}, {os.path.join(OUTDIR, 'B_noLTV_famd_scores.npy')}")
print(f" - {os.path.join(OUTDIR, 'A_withLTV_labels_stability.csv')}, {os.path.join(OUTDIR, 'B_noLTV_labels_stability.csv')}")
print(f" - {os.path.join(OUTDIR, 'A_withLTV_stability_sizes.csv')}, {os.path.join(OUTDIR, 'B_noLTV_stability_sizes.csv')}")
