# Data loading, cleaning, and feature engineering
import re
import numpy as np
import pandas as pd
from typing import Optional, Set, Dict

try:
    from .config import *
except ImportError:
    from config import *

# -------------------------- Lookup Tables --------------------------
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

# -------------------------- Helper Functions --------------------------
def _sanitize(s: str) -> str:
    """Sanitize string for safe file names."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def normalize_quiz_result(q: Optional[str]) -> Optional[str]:
    """Normalize quiz result names."""
    if not isinstance(q, str): 
        return None
    qn = q.strip()
    if not qn: 
        return None
    return QUIZ_NORMALIZE.get(qn, qn)

def quiz_to_bucket(quiz_result: Optional[str]) -> Optional[str]:
    """Map quiz result to product bucket."""
    if not isinstance(quiz_result, str): 
        return None
    line = normalize_quiz_result(quiz_result)
    return QUIZ_LINE_TO_BUCKET.get(line) if line else None

def parse_first_sku_buckets(first_sku: Optional[str]) -> Set[str]:
    """Parse first SKU into product buckets."""
    buckets: Set[str] = set()
    if not isinstance(first_sku, str) or not first_sku.strip(): 
        return buckets
    for token in [t.strip() for t in first_sku.split(",")]:
        b = SKU_TO_BUCKET.get(token)
        if b: 
            buckets.add(b)
    return buckets

def collapse_sku_buckets(s: Set[str]) -> str:
    """Collapse SKU buckets to single category."""
    if not s: 
        return "none"
    return sorted(list(s))[0] if len(s) == 1 else "multi"

def loose_match(q_bucket: Optional[str], sku_buckets: Set[str]) -> int:
    """Check if quiz recommendation matches purchased bucket."""
    return int(q_bucket in sku_buckets) if q_bucket and sku_buckets else 0

def map_code_group(x: Optional[str]) -> str:
    """Map discount codes to influencer groups."""
    if isinstance(x, str) and x.strip():
        v = x.strip().lower()
        return v if v in INFLUENCER_CODES else "other"
    return "none"

def derive_gender(row: pd.Series) -> str:
    """Derive gender from available fields."""
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

# -------------------------- Main Data Loading Function --------------------------
def load_and_clean_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load and clean the raw data, applying all feature engineering.
    
    Returns:
        pd.DataFrame: Cleaned dataframe ready for analysis
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    if "customer_id" not in df.columns:
        raise ValueError("customer_id column is required.")
    
    print(f"Raw data shape: {df.shape}")
    
    # Normalize / engineer features
    df["primary_goal"] = df.get("primary_goal", pd.Series(dtype=object)).replace("-", "missing")
    df["bm_pattern"] = df.get("bm_pattern", pd.Series(dtype=object)).replace("unspecified", "missing")
    df["quiz_result"] = df.get("quiz_result", pd.Series(dtype=object)).apply(normalize_quiz_result)
    df["quiz_bucket"] = df["quiz_result"].apply(quiz_to_bucket)
    df["first_sku_buckets"] = df.get("first_sku", pd.Series(dtype=object)).apply(parse_first_sku_buckets)
    df["first_sku_bucket"] = df["first_sku_buckets"].apply(collapse_sku_buckets)
    df["quiz_reco_match"] = df.apply(lambda r: loose_match(r.get("quiz_bucket"), r.get("first_sku_buckets")), axis=1)
    df["acquisition_code_group"] = df.get("ancestor_discount_code", np.nan).apply(map_code_group)
    df["gender_cat"] = df.apply(derive_gender, axis=1)
    
    print("Feature engineering completed.")
    return df

def filter_to_quiz_takers(df: pd.DataFrame, influencer_only: bool = INFLUENCER_ONLY) -> pd.DataFrame:
    """
    Filter dataframe to quiz takers and optionally influencer customers.
    
    Args:
        df: Input dataframe
        influencer_only: If True, include only influencer customers
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if "quiz_taker" not in df.columns:
        raise ValueError("quiz_taker column is required.")
    
    df_quiz = df[df["quiz_taker"].astype(str).str.lower() == "yes"].copy()
    print(f"Quiz takers: {len(df_quiz):,}")
    
    if influencer_only:
        df_quiz = df_quiz[df_quiz["acquisition_code_group"].ne("none")].copy()
        print(f"After influencer filter: {len(df_quiz):,}")
    
    return df_quiz

def apply_coverage_filter(df: pd.DataFrame, coverage_threshold: float = COVERAGE) -> pd.DataFrame:
    """
    Remove columns with insufficient data coverage.
    
    Args:
        df: Input dataframe
        coverage_threshold: Minimum fraction of non-null values required
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    print(f"Applying coverage filter (>= {coverage_threshold*100:.0f}% non-null)...")
    
    non_null_threshold = coverage_threshold * len(df)
    df_filtered = df.loc[:, df.notnull().sum() >= non_null_threshold].copy()
    
    print(f"Columns before: {len(df.columns)}, after: {len(df_filtered.columns)}")
    print(f"Rows: {len(df_filtered):,}")
    
    return df_filtered

def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that would cause data leakage in modeling.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with leakage columns removed
    """
    drop_cols = [
        "email_key","quiz_date","first_order_date","last_order_date",
        "ancestor_discount_code","quiz_taker","first_sku","first_sku_buckets",
        "total_cogs","shipping_collected","shippingspend","refund_amt","refund_ratio",
        "recent_abx_flag","affiliate_segment",
    ]
    
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")
    
    print(f"Removed {len(cols_to_drop)} leakage columns")
    return df_clean

def prepare_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure categorical and numeric features have correct data types.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with proper data types
    """
    df_typed = df.copy()
    
    # Set categorical columns
    for c in BASE_CAT:
        if c in df_typed.columns:
            df_typed[c] = df_typed[c].fillna("missing").astype(str)
    
    # Set numeric columns  
    for c in BASE_NUM:
        if c in df_typed.columns:
            df_typed[c] = pd.to_numeric(df_typed[c], errors="coerce").fillna(0.0)
    
    # Ensure value columns are numeric
    value_cols_present = [c for c in VALUE_COLS if c in df_typed.columns]
    for c in value_cols_present:
        df_typed[c] = pd.to_numeric(df_typed[c], errors="coerce").fillna(0.0)
    
    print(f"Prepared {len(BASE_CAT)} categorical and {len(BASE_NUM)} numeric features")
    print(f"Value columns available: {value_cols_present}")
    
    return df_typed

def get_modeling_dataframe() -> pd.DataFrame:
    """
    Main function to load, clean, and prepare data for modeling.
    
    Returns:
        pd.DataFrame: Clean dataframe ready for FAMD analysis
    """
    # Load and engineer features
    df = load_and_clean_data()
    
    # Filter to quiz takers
    df_quiz = filter_to_quiz_takers(df)
    
    # Remove leakage columns
    df_quiz = remove_leakage_columns(df_quiz)
    
    # Apply coverage filter
    df_quiz = apply_coverage_filter(df_quiz)
    
    # Prepare feature types
    df_quiz = prepare_feature_types(df_quiz)
    
    # Keep only essential columns for modeling
    keep_cols = ["customer_id"] + BASE_CAT + BASE_NUM
    value_cols_present = [c for c in VALUE_COLS if c in df_quiz.columns]
    keep_cols.extend(value_cols_present)
    
    df_model = df_quiz[keep_cols].copy()
    
    print(f"Final modeling dataframe: {df_model.shape}")
    print(f"Columns: {list(df_model.columns)}")
    
    return df_model

if __name__ == "__main__":
    # Test the data cleaning pipeline
    df_clean = get_modeling_dataframe()
    print("\nData cleaning completed successfully!")
    print(f"Shape: {df_clean.shape}")
    print(f"Memory usage: {df_clean.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
