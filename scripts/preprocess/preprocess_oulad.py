#!/usr/bin/env python3
"""
Script Name   : preprocess_oulad.py
Description   : Preprocessing pipeline for the Open University Learning Analytics
                Dataset (OULAD). Transforms raw OULAD CSV files into a unified,
                machine-learning-ready format compatible with both classical
                models (e.g., XGBoost) and deep learning models (e.g., TabNet).

                The script performs:
                - Feature engineering from VLE and assessment data
                - Leakage-aware aggregation (explicit exclusion of exams)
                - Harmonization of features across datasets
                - Separation of features, targets, and sensitive attributes
                - Encoding, scaling, and metadata generation for TabNet
Input Data    : data/raw/oulad/
                - studentInfo.csv
                - studentVle.csv
                - studentAssessment.csv
                - assessments.csv

Outputs       : data/processed/oulad/
                - X.csv                  (feature matrix)
                - y.csv                  (binary target variable)
                - sensitive.csv          (protected attributes)
                - feature_metadata.json  (TabNet metadata)

Preprocessors : models/preprocessors/oulad/
                - scaler.pkl
                - labelenc_<col>.pkl (if applicable)
                - encoders.pkl

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybsersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119

Created On    : 2025-Dec-10

License       : MIT License (recommended for research reproducibility)
Citation      : If you use this code in academic work, please cite the
                corresponding publication or acknowledge the author.

Notes        :
- Designed for fairness-aware learning pipelines.
- Sensitive attributes are strictly excluded from X.
- Exam scores are excluded to prevent target leakage.
- Deterministic and reproducible assuming fixed random seeds downstream.

Dependencies :
- Python >= 3.9
- pandas, numpy, scikit-learn

"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Config & Constants
# ---------------------------
RAW_DIR = Path("data/raw/oulad")
OUT_DIR = Path("data/processed/oulad")
PP_DIR = Path("models/preprocessors/oulad")

# Strict Ordinal Mapping for Education (Low -> High)
EDUCATION_MAP = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4
}

# ---------------------------
# Helpers
# ---------------------------

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PP_DIR.mkdir(parents=True, exist_ok=True)

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)

def imd_to_ses_quintile(imd: str) -> int:
    """
    Parse IMD_Band into quintiles (0..4).
    Unknown -> -1 (to be handled later or treated as category).
    """
    if pd.isna(imd):
        return -1
    s = str(imd).strip()
    if "unknown" in s.lower():
        return -1
    
    # Extract "0" from "0-10%"
    try:
        left = s.split("-")[0].replace("%", "").strip()
        decile_start = int(left)  # 0, 10, ... 90
        # Map 0-20 -> 0, 20-40 -> 1, etc.
        return int(np.clip(decile_start // 20, 0, 4))
    except Exception:
        return -1

def final_result_to_target(fr: str) -> float:
    if pd.isna(fr):
        return np.nan
    s = str(fr).strip().lower()
    if s in {"pass", "distinction"}:
        return 1.0
    if s in {"fail", "withdrawn"}:
        return 0.0
    return np.nan

def weighted_assessment_signal(
    student_assess: pd.DataFrame, assess: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates weighted average of assignments, STRICTLY EXCLUDING EXAMS 
    to prevent target leakage.
    """
    sa = student_assess.copy()
    a = assess.copy()

    # CRITICAL: Filter out Exams
    a = a[a["assessment_type"] != "Exam"].copy()

    # Merge weights
    merged = sa.merge(
        a[["id_assessment", "code_module", "code_presentation", "weight"]],
        on="id_assessment",
        how="inner"  # Inner join drops assessments that were Exams (since we filtered 'a')
    )

    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")

    def agg_fn(g: pd.DataFrame) -> pd.Series:
        scores = g["score"]
        weights = g["weight"]
        
        # If valid weights exist
        if weights.sum() > 0 and scores.notna().any():
            ws = np.nansum(scores.values * weights.values)
            wsum = np.nansum(weights.values * (~scores.isna()).values.astype(float))
            val = ws / wsum if wsum > 0 else np.nan
        else:
            # Fallback to mean if no weights (some courses are 100% exam)
            val = scores.mean()
        return pd.Series({"assessment_signal": val})

    out = (
        merged.groupby(["id_student", "code_module", "code_presentation"], dropna=False)
        .apply(agg_fn)
        .reset_index()
    )
    return out

def vle_engagement(student_vle: pd.DataFrame) -> pd.DataFrame:
    """Sum of clicks per student/module."""
    sv = student_vle.copy()
    sv["sum_click"] = pd.to_numeric(sv.get("sum_click"), errors="coerce")
    out = (
        sv.groupby(["id_student", "code_module", "code_presentation"], dropna=False)["sum_click"]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"sum_click": "engagement_level"})
    )
    return out

def fit_scaler(
    X: pd.DataFrame, numeric_cols: List[str], pp_dir: Path
) -> Tuple[pd.DataFrame, StandardScaler]:
    X_out = X.copy()
    scaler = StandardScaler()
    if numeric_cols:
        X_out[numeric_cols] = scaler.fit_transform(X_out[numeric_cols].values)
    
    with open(pp_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return X_out, scaler

def build_tabnet_metadata(
    numerical_cols: List[str],
    categorical_cols: List[str],
    cat_dims: Dict[str, int],
    final_col_order: List[str]
) -> Dict:
    cat_idxs = [final_col_order.index(c) for c in categorical_cols]
    cat_dims_list = [cat_dims[c] for c in categorical_cols]
    return {
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims_list,
    }

# ---------------------------
# Main Pipeline
# ---------------------------

def main() -> None:
    ensure_dirs()
    print("[OULAD] Loading raw data...")
    
    # Load Data
    s_info = read_csv_required(RAW_DIR / "studentInfo.csv")
    s_vle = read_csv_required(RAW_DIR / "studentVle.csv")
    s_assess = read_csv_required(RAW_DIR / "studentAssessment.csv")
    assess = read_csv_required(RAW_DIR / "assessments.csv")

    # Feature Engineering
    print("[OULAD] Engineering features...")
    eng = vle_engagement(s_vle)
    assess_sig = weighted_assessment_signal(s_assess, assess)

    # Merge
    df = s_info.merge(
        eng, on=["id_student", "code_module", "code_presentation"], how="left"
    ).merge(
        assess_sig, on=["id_student", "code_module", "code_presentation"], how="left"
    )

    # ---------------------------
    # Harmonization & Mapping
    # ---------------------------
    
    # 1. Target (y)
    df["y"] = df["final_result"].map(final_result_to_target)
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    # 2. Sensitive Attributes
    # Gender (Normalize)
    df["gender"] = df["gender"].astype(str).str.strip().map({"M": "0", "F": "1"}).fillna("Unknown")
    
    # SES Quintile (0-4)
    df["ses_quintile"] = df["imd_band"].map(imd_to_ses_quintile)
    # Filter unknowns for SES if needed, or keep as -1
    
    # Age Band
    df["age_band"] = df["age_band"].astype(str)
    
    # Disability (Unified: 1=Yes, 0=No)
    df["disability"] = df["disability"].map({"Y": 1, "N": 0}).fillna(0).astype(int)

    # 3. Features (X)
    # Prior Education (Ordinal Map)
    df["prior_education"] = df["highest_education"].map(EDUCATION_MAP).fillna(-1).astype(int)
    
    # Engagement & Assessment (Numeric)
    df["engagement_level"] = pd.to_numeric(df["engagement_level"], errors="coerce").fillna(0)
    # Impute missing assessment with Median (conservative) or -1? 
    # Standard practice: Median imputation
    median_assess = df["assessment_signal"].median()
    df["assessment_signal"] = df["assessment_signal"].fillna(median_assess)

    # ---------------------------
    # Splits
    # ---------------------------
    
    # Sensitive DataFrame (Include disability for OULAD specific audit)
    sensitive_cols = ["gender", "ses_quintile", "age_band", "disability"]
    sensitive = df[sensitive_cols].copy()

    # X DataFrame (Feature Matrix)
    # NOTE: prior_education is IN X as a feature, but NOT in sensitive (for debiasing purposes)
    # unless you want to debias on education too.
    feature_cols = ["prior_education", "engagement_level", "assessment_signal"]
    X = df[feature_cols].copy()

    # ---------------------------
    # Encoding & Scaling
    # ---------------------------
    
    # Categorical: prior_education is already Ordinal Ints (0-4). 
    # We treat it as categorical for TabNet Embeddings? 
    # Yes, typically embeddings work better for ordinal categories than treating them as linear scalars.
    
    cat_cols = ["prior_education"]
    num_cols = ["engagement_level", "assessment_signal"]
    
    # Shift prior_education to be positive for Embedding Index (if -1 exists)
    # If -1 (Unknown), make it a distinct class (e.g. 5)
    X["prior_education"] = X["prior_education"].replace(-1, 5)
    cat_dims = {"prior_education": 6} # 0..5

    # Scale Numericals
    X_final, scaler = fit_scaler(X, num_cols, PP_DIR)

    # ---------------------------
    # Save
    # ---------------------------
    
    # Metadata
    meta = build_tabnet_metadata(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        cat_dims=cat_dims,
        final_col_order=list(X_final.columns)
    )

    print(f"[OULAD] Saving artifacts to {OUT_DIR}...")
    X_final.to_csv(OUT_DIR / "X.csv", index=False)
    df[["y"]].to_csv(OUT_DIR / "y.csv", index=False)
    sensitive.to_csv(OUT_DIR / "sensitive.csv", index=False)
    
    with open(OUT_DIR / "feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[OULAD] Done.")

if __name__ == "__main__":
    main()
