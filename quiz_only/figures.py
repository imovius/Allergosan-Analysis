# make_segmentation_figures.py
# Creates: silhouette comparison, FAMD variance + loadings (optional),
# side-by-side cluster scatter, cluster profile bars, and SKU distribution.

import os, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIG (EDIT ME) =====================
OUTDIR = "report_figures"
os.makedirs(OUTDIR, exist_ok=True)

# 1) Stage 1 (RFM-heavy) silhouette/R^2 pulled from your screenshot
stage1_k = np.array([3,4,5,6,7])
stage1_sil = np.array([0.327,0.352,0.343,0.311,0.315])

# 2) Final quiz+promo run (your recent printout)
final_k = np.array([2,3,4,5,6,7,8,9,10])
final_sil = np.array([0.3411,0.4953,0.6173,0.5063,0.4385,0.4337,0.4370,0.3630,0.3763])

# 3) Files from your pipeline
#    - Row level with cluster + first_sku_bucket
ROW_CSV   = "outputs/kmeans_clusters_with_meta.csv"        # put correct path
#    - Cluster profiles with quiz_reco_match_pct, promo_1..3, symptoms, etc.
PROF_CSV  = "outputs/kmeans_cluster_profiles.csv"          # put correct path
#    - Optional side-by-side scatters (already rendered by your script)
SCATTER_EARLY = None       # if you have it; else leave None
SCATTER_FINAL = "outputs/famd_kmeans_scatter.png"          # your latest image
# ===========================================================

def line_with_secondary_y(x, y_main, y2, title, fname):
    fig, ax1 = plt.subplots(figsize=(7,4.2))
    ax1.plot(x, y_main, marker="o")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Silhouette")
    ax1.set_title(title)
    if y2 is not None:
        ax2 = ax1.twinx()
        ax2.plot(x, y2, marker="s", linestyle="--")
        ax2.set_ylabel("R² (variance explained)")
        ax2.grid(False)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=180)
    plt.close(fig)

def single_line(x, y, title, fname, ylab="Silhouette"):
    fig, ax = plt.subplots(figsize=(7,4.2))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=180)
    plt.close(fig)

def side_by_side_images(img_left, img_right, titles, fname):
    if (img_left is None) or (not os.path.exists(img_left)) or (img_right is None) or (not os.path.exists(img_right)):
        return
    import matplotlib.image as mpimg
    L = mpimg.imread(img_left)
    R = mpimg.imread(img_right)
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].imshow(L); axes[0].axis("off"); axes[0].set_title(titles[0])
    axes[1].imshow(R); axes[1].axis("off"); axes[1].set_title(titles[1])
    fig.suptitle("Cluster Separation (FAMD space) — Before vs After", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=180)
    plt.close(fig)

def stacked_pct_bars_from_rowlevel(row_csv, cat_col="first_sku_bucket", cluster_col="cluster", title="", fname=""):
    df = pd.read_csv(row_csv)
    tab = (df.groupby([cluster_col, cat_col]).size()
             .reset_index(name="n"))
    totals = tab.groupby(cluster_col)["n"].transform("sum")
    tab["pct"] = tab["n"] / totals * 100
    pivot = tab.pivot(index=cluster_col, columns=cat_col, values="pct").fillna(0)
    pivot = pivot[pivot.sum().sort_values(ascending=False).index]  # order columns by volume

    fig, ax = plt.subplots(figsize=(10,5))
    bottom = np.zeros(len(pivot))
    for col in pivot.columns:
        ax.bar(pivot.index.astype(str), pivot[col].values, bottom=bottom, label=col)
        bottom += pivot[col].values
    ax.set_ylabel("% of customers")
    ax.set_xlabel("Cluster")
    ax.set_title(title or f"{cat_col} distribution by cluster")
    ax.legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0.)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname or f"{cat_col}_by_cluster.png"), dpi=180)
    plt.close(fig)

def cluster_profile_bars(prof_csv, metric_cols, title, fname, ylabel):
    prof = pd.read_csv(prof_csv)
    prof = prof.sort_values("cluster")
    fig, ax = plt.subplots(figsize=(10,5))
    width = 0.8 / len(metric_cols)
    centers = np.arange(len(prof))
    for i, col in enumerate(metric_cols):
        ax.bar(centers + i*width, prof[col].values, width=width, label=col)
    ax.set_xticks(centers + width*(len(metric_cols)-1)/2)
    ax.set_xticklabels(prof["cluster"].astype(str))
    ax.set_xlabel("Cluster")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, fname), dpi=180)
    plt.close(fig)

# ------------------- RUN -------------------
# 1) Stage 1 curve (silhouette + R²)
single_line(stage1_k, stage1_sil,
    "Clustering Quality — Stage 1 (RFM-heavy behavioral segmentation)",
    "fig1_stage1_silhouette_r2.png")

# 2) Final run silhouette curve
single_line(final_k, final_sil,
    "Clustering Quality — Final Quiz + Promo segmentation",
    "fig2_final_silhouette.png")

# 3) Side-by-side separation image (if you have both)
side_by_side_images(
    SCATTER_EARLY, SCATTER_FINAL,
    ["Early quiz run (with first_sku_bucket)", "Final quiz run (with quiz_reco_match)"],
    "fig3_scatter_before_after.png"
)

# 4) SKU distribution (justifies dropping it from clustering)
if os.path.exists(ROW_CSV):
    stacked_pct_bars_from_rowlevel(
        ROW_CSV,
        cat_col="first_sku_bucket",
        cluster_col="cluster",
        title="First SKU Bucket — distribution by cluster (for justification)",
        fname="fig4_first_sku_bucket_by_cluster.png"
    )

# 5) Marketing-useful differences from profiles
if os.path.exists(PROF_CSV):
    # 5a) Quiz recommendation match %
    cluster_profile_bars(
        PROF_CSV,
        ["quiz_reco_match_pct"],
        "Quiz recommendation match by cluster",
        "fig5_quiz_match_by_cluster.png",
        "% of customers"
    )
    # 5b) Promo code skew: promo_1..promo_3 are text; build % columns if present
    prof = pd.read_csv(PROF_CSV)
    promo_pct_cols = [c for c in prof.columns if c.startswith("promo_") and c.endswith("_pct")]
    if promo_pct_cols:
        cluster_profile_bars(
            PROF_CSV,
            promo_pct_cols[:3],
            "Top promo code share by cluster",
            "fig6_promo_skew_by_cluster.png",
            "% of customers"
        )
    # 5c) Symptom prevalence examples (edit to the ones you have)
    symptom_cols = [c for c in prof.columns if c.startswith("sx_") and c.endswith("_pct")]
    take = [c for c in symptom_cols if any(k in c for k in ["bloating","constipation","diarrhea"])]
    take = take[:3] if take else symptom_cols[:3]
    if take:
        cluster_profile_bars(
            PROF_CSV,
            take,
            "Key symptom prevalence by cluster",
            "fig7_symptoms_by_cluster.png",
            "% of customers"
        )

# 6) (Optional) FAMD variance + loadings
# If you can export arrays from your modeling script, save them and plot here.
# Example expectations:
#   np.save("famd_explained_inertia.npy", explained_inertia_)  # shape (n_components,)
#   pd.DataFrame(loadings).to_csv("famd_loadings.csv", index=False)  # columns: component, feature, loading
if os.path.exists("famd_explained_inertia.npy"):
    var = np.load("famd_explained_inertia.npy")
    k = min(10, len(var))
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(np.arange(1,k+1), np.array(var[:k])*100.0)
    ax.set_xlabel("FAMD component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("FAMD variance explained (first 10 components)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig8_famd_variance.png"), dpi=180)
    plt.close(fig)

if os.path.exists("famd_loadings.csv"):
    load = pd.read_csv("famd_loadings.csv")  # expects columns: component, feature, loading
    top1 = (load[load["component"]==1]
                .assign(abs_load=lambda d: d["loading"].abs())
                .nlargest(12, "abs_load"))
    top2 = (load[load["component"]==2]
                .assign(abs_load=lambda d: d["loading"].abs())
                .nlargest(12, "abs_load"))
    def plot_load(df, comp, fname):
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(df["feature"], df["abs_load"])
        ax.invert_yaxis()
        ax.set_xlabel("|loading|")
        ax.set_title(f"Top contributors to FAMD component {comp}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, fname), dpi=180)
        plt.close(fig)
    plot_load(top1, 1, "fig9_famd_loadings_c1.png")
    plot_load(top2, 2, "fig10_famd_loadings_c2.png")

print(f"Done. Figures saved to: {OUTDIR}")
