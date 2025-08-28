# Configuration constants extracted from the original script
import os

# ======================= CONFIG =======================
CSV_PATH = "../raw_data_v3.csv"
FAMD_DIR = "../FAMD_output"      # Core FAMD interpretation files
CLUSTER_DIR = "../quiz_only/outputs"  # Clustering/profiling files
SEED = 42

# Ensure output directories exist
os.makedirs(FAMD_DIR, exist_ok=True)
os.makedirs(CLUSTER_DIR, exist_ok=True)

# Data filters
INFLUENCER_ONLY = False   # True: promo purchasers only; False: all quiz takers
COVERAGE = 0.60    # keep columns with >=60% non-null (looser to bring back features)
DOMINANCE_THRESH = 0.995   # allow dominated categoricals to pass (e.g., first_sku_bucket)

# FAMD (both passes)
EXPLAINED_TARGET = 0.85    # keep ~85% cumulative variance
N_MAX_COMPONENTS = 20      # upper bound for components to learn
PRUNE_FRACTION = 0.15    # drop bottom 15% by total contribution (first n_keep comps)
MIN_CONTRIB_PCT = 0.40    # also drop any <0.4% contribution

# Clustering
K_RANGE = range(2, 11)  # candidate k values
FORCE_K_A = None          # force k for Pass A (with LTV), or None
FORCE_K_B = None          # force k for Pass B (behavior only), or None

DPI = 150

# Feature definitions
BASE_CAT = [
    "quiz_result", "acquisition_code_group", "bm_pattern", "gi_symptom_cat",
    "primary_goal", "gender_cat", "first_sku_bucket"
]

BASE_NUM = [
    "order_count", "days_since_last_order", "symptom_count", "gut_issue_score",
    "high_stress", "refund_count", "quiz_reco_match"
]

# VALUE FIELDS
VALUE_COLS = ["net_ltv", "avg_order_value", "gross_ltv"]
