#!/usr/bin/env python3
"""
===============================================================================
Script Name   : preprocess_uci.py
Description   : Preprocessing pipeline for the UCI Student Performance Dataset.
                Converts raw UCI student records into a unified, machine-
                learning-ready format compatible with classical ML models
                (e.g., XGBoost) and deep learning models (e.g., TabNet).

                This script is explicitly designed to align semantically and
                structurally with the OULAD preprocessing pipeline to enable
                cross-dataset learning, fairness auditing, and comparative
                analysis.

Key Features  :
                - Ordinal-safe handling of parental education (Medu)
                - SES approximation using parental occupation tiers
                - Domain-aware age binning with documented limitations
                - Robust parsing across UCI CSV format variants
                - Strict feature / target / sensitive attribute separation
                - TabNet-compatible metadata generation

Input Data    : data/raw/uci/
                - student-mat.csv (preferred)
                - student-por.csv (fallback)
                - or any compatible UCI Student Performance CSV

Outputs       : data/processed/uci/
                - X.csv                  (feature matrix)
                - y.csv                  (binary target variable)
                - sensitive.csv          (protected attributes)
                - feature_metadata.json  (TabNet metadata)

Preprocessors : models/preprocessors/uci/
                - scaler.pkl

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybsersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119

Created On    : 2025-Dec-10

License       : MIT License (recommended for reproducible research)
Citation      : If this code is used in academic work, please cite the
                corresponding publication or acknowledge the author.

Design Notes :
- Medu is preserved as an explicit ordinal integer (0–4) to avoid semantic loss.
- SES quintiles are approximated using parental occupation tiers following
  established UCI dataset conventions (Cortez et al.).
- Age bands are retained at higher granularity due to domain mismatch with
  OULAD; fairness auditing scripts may disable cross-dataset age comparison.
- Engagement is inversely approximated via absences to maintain directional
  consistency with VLE-based engagement in OULAD.
- Sensitive attributes are strictly excluded from X.

Dependencies :
- Python >= 3.9
- pandas, numpy, scikit-learn

===============================================================================
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------
# Config & Constants
# ---------------------------
RAW_DIR = Path("data/raw/uci")
OUT_DIR = Path("data/processed/uci")
PP_DIR = Path("models/preprocessors/uci")

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PP_DIR.mkdir(parents=True, exist_ok=True)

def find_uci_file() -> Path:
    # Prefer Math (more standard for SES/demographic analysis) over Portuguese
    preferred = [RAW_DIR / "student-mat.csv", RAW_DIR / "student-por.csv"]
    for p in preferred:
        if p.exists():
            return p
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR}")
    return csvs[0]

def job_to_tier(job: str) -> float:
    """
    Map job string to approximate SES tier (0=Low, 1=Mid, 2=High).
    Based on Cortez et al. hierarchies.
    """
    if pd.isna(job):
        return np.nan
    s = str(job).strip().lower()
    if s in {"at_home", "athome", "other"}:
        return 0.0
    if s == "services":
        return 1.0
    if s in {"health", "teacher"}:
        return 2.0
    return np.nan

def make_ses_quintile(mjob: str, fjob: str) -> str:
    """
    Combine Mother and Father job to approximate an SES Quintile (0-4).
    Logic: Mean Tier (0-2) * 2 => Scale (0-4).
    """
    mt = job_to_tier(mjob)
    ft = job_to_tier(fjob)
    vals = [v for v in [mt, ft] if not np.isnan(v)]
    if not vals:
        return "-1" # Unknown
    
    mean_tier = float(np.mean(vals))  # 0.0 to 2.0
    # Map 0.0->0, 2.0->4
    q = int(np.clip(round(mean_tier * 2.0), 0, 4))
    return str(q)

def age_to_band(age_val) -> str:
    """Bucket numeric age to match OULAD bands."""
    if pd.isna(age_val):
        return "Unknown"
    try:
        a = int(age_val)
    except Exception:
        return "Unknown"
    
    # OULAD bands are broadly <35, 35-55, >55. 
    # UCI is 15-22. They will ALL fall into "<35" (or "<=16", "17-18" etc for internal granularity).
    # To conform to Harmonization Map for Cross-Dataset comparison, we must map to OULAD schema?
    # NO: The Map says "Bin UCI ... if distinct bins < 2, set fairness_eligible=False".
    # We will keep granular bins here for internal analysis, but the audit script 
    # will likely skip this for cross-dataset comparison due to domain shift.
    if a <= 16:
        return "<=16"
    if 17 <= a <= 18:
        return "17-18"
    if 19 <= a <= 20:
        return "19-20"
    return ">=21"

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

def main() -> None:
    ensure_dirs()
    uci_path = find_uci_file()
    print(f"[UCI] Processing {uci_path}...")

    # Load with flexible separator (some UCI versions use ;)
    try:
        df = pd.read_csv(uci_path, sep=";")
        if "Mjob" not in df.columns: # Try comma if semi-colon failed to parse cols
            df = pd.read_csv(uci_path, sep=",")
    except Exception:
        df = pd.read_csv(uci_path, sep=None, engine="python")

    # 1. Harmonization
    # SES
    df["ses_quintile"] = [make_ses_quintile(m, f) for m, f in zip(df["Mjob"], df["Fjob"])]
    
    # Prior Education (Medu is 0=None, 4=Higher). 
    # Fits OULAD (0=No Formal, 4=PostGrad) conceptually.
    df["prior_education"] = pd.to_numeric(df["Medu"], errors="coerce").fillna(0).astype(int)
    
    # Gender (Normalize to 0=M, 1=F to match OULAD)
    df["gender"] = df["sex"].map({"F": "1", "M": "0"}).fillna("Unknown")
    
    # Age
    df["age_band"] = df["age"].map(age_to_band)

    # Engagement (-absences)
    df["engagement_level"] = -pd.to_numeric(df["absences"], errors="coerce").fillna(0)

    # Assessment (Mean of G1, G2)
    # G1, G2 are 0-20.
    df["assessment_signal"] = (pd.to_numeric(df["G1"]) + pd.to_numeric(df["G2"])) / 2.0
    df["assessment_signal"] = df["assessment_signal"].fillna(df["assessment_signal"].median())

    # Target (G3 >= 10)
    g3 = pd.to_numeric(df["G3"], errors="coerce")
    df = df.dropna(subset=["G3"]).copy()
    df["y"] = (g3 >= 10).astype(int)

    # 2. Splits Setup
    sensitive_cols = ["gender", "ses_quintile", "age_band"]
    sensitive = df[sensitive_cols].copy()
    
    # X - Feature Matrix
    # prior_education is categorical (Ordinal Integer)
    feature_cols = ["prior_education", "engagement_level", "assessment_signal"]
    X = df[feature_cols].copy()

    # 3. Encoding & Scaling
    # Categorical: prior_education is already int 0..4
    cat_cols = ["prior_education"]
    num_cols = ["engagement_level", "assessment_signal"]
    
    # Ensure prior_education is strictly 0..N for TabNet embedding
    # Medu is 0..4. No mapping needed, just validation.
    cat_dims = {"prior_education": 5} # 0,1,2,3,4

    # Scale Continuous
    X_final, scaler = fit_scaler(X, num_cols, PP_DIR)

    # 4. Save
    meta = build_tabnet_metadata(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        cat_dims=cat_dims,
        final_col_order=list(X_final.columns)
    )

    X_final.to_csv(OUT_DIR / "X.csv", index=False)
    df[["y"]].to_csv(OUT_DIR / "y.csv", index=False)
    sensitive.to_csv(OUT_DIR / "sensitive.csv", index=False)
    
    with open(OUT_DIR / "feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[UCI] Done. Saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
