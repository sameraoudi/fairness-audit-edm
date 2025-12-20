#!/usr/bin/env python3
"""
===============================================================================
Script Name   : preprocess_xapi.py
Description   : Preprocessing pipeline for the xAPI-Edu-Data dataset.
                Transforms raw xAPI educational interaction data into a
                unified, machine-learning-ready format compatible with both
                classical ML models (e.g., XGBoost) and deep learning models
                (e.g., TabNet).

                This script is designed to align structurally and semantically
                with the OULAD and UCI preprocessing pipelines to support
                cross-dataset learning, benchmarking, and fairness analysis.

Key Features  :
                - Schema-aligned gender normalization (M/F → 0/1)
                - Robust handling of inconsistent column naming conventions
                - Conservative treatment of missing SES and age attributes
                - Strict separation of features, targets, and sensitive data
                - TabNet-compatible feature metadata generation

Input Data    : data/raw/xapi/
                - xapi.csv
                - xAPI-Edu-Data.csv
                - xAPI_Edu_Data.csv
                - or any compatible xAPI-Edu-Data CSV

Outputs       : data/processed/xapi/
                - X.csv                  (feature matrix)
                - y.csv                  (binary target variable)
                - sensitive.csv          (protected attributes)
                - feature_metadata.json  (TabNet metadata)

Preprocessors : models/preprocessors/xapi/
                - scaler.pkl

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119

Created On    : 2025-Dec-10

License       : MIT License (recommended for reproducible research)
Citation      : If this code is used in academic work, please cite the
                corresponding publication or acknowledge the author.

Design Notes :
- Gender is harmonized to match OULAD/UCI encoding for fairness audits.
- Engagement is derived solely from visited resources to maintain
  cross-dataset comparability.
- SES and age attributes are conservatively excluded due to lack of
  reliable proxies in xAPI-Edu-Data.
- Sensitive attributes are never included in X.
- The resulting feature space is intentionally minimal to avoid
  dataset-specific bias leakage.

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
RAW_DIR = Path("data/raw/xapi")
OUT_DIR = Path("data/processed/xapi")
PP_DIR = Path("models/preprocessors/xapi")

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PP_DIR.mkdir(parents=True, exist_ok=True)

def find_xapi_file() -> Path:
    candidates = [
        RAW_DIR / "xapi.csv",
        RAW_DIR / "xAPI-Edu-Data.csv",
        RAW_DIR / "xAPI_Edu_Data.csv",
        RAW_DIR / "xAPI.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR}")
    return csvs[0]

def class_to_target(v: str) -> float:
    """Map Class L->0, M/H->1."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    if s == "L":
        return 0.0
    if s in {"M", "H"}:
        return 1.0
    return np.nan

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
    xapi_path = find_xapi_file()
    print(f"[xAPI] Processing {xapi_path}...")
    
    df = pd.read_csv(xapi_path)

    # 1. Harmonization
    # Gender (Normalize to 0=M, 1=F)
    if "gender" not in df.columns and "Gender" in df.columns:
        df["gender"] = df["Gender"]
    
    # Map raw 'M'/'F' to '0'/'1'
    # Check distinct values first? usually 'M', 'F'
    df["gender"] = df["gender"].astype(str).str.upper().map({"M": "0", "F": "1"}).fillna("Unknown")

    # Engagement
    # Dataset often has 'VisITedResources'
    if "VisITedResources" in df.columns:
        df["engagement_level"] = pd.to_numeric(df["VisITedResources"], errors="coerce")
    elif "VisitedResources" in df.columns:
        df["engagement_level"] = pd.to_numeric(df["VisitedResources"], errors="coerce")
    else:
        # Fallback or Error
        print("[WARNING] xAPI missing VisITedResources. Setting engagement to 0.")
        df["engagement_level"] = 0

    # Target
    if "Class" not in df.columns:
        raise ValueError("xAPI file missing Class column.")
    df["y"] = df["Class"].map(class_to_target)

    # Drop missing target
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    # 2. Splits Setup
    # Sensitive
    # Note: SES and Age are NaN per Conservative Exclusion
    sensitive = pd.DataFrame({
        "gender": df["gender"],
        "ses_quintile": "Unknown",  # Explicit string better than NaN for CSV reading
        "age_band": "Unknown"
    })

    # X - Feature Matrix
    # Only engagement_level is unified
    feature_cols = ["engagement_level"]
    X = df[feature_cols].copy()

    # Impute numericals
    X["engagement_level"] = X["engagement_level"].fillna(X["engagement_level"].median())

    # 3. Encoding & Scaling
    # No categorical features in this unified view
    cat_cols: List[str] = []
    num_cols: List[str] = ["engagement_level"]
    cat_dims: Dict[str, int] = {}

    # Scale numeric
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

    print(f"[xAPI] Done. Saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
