#!/usr/bin/env python3
"""
===============================================================================
Script Name   : verify_oulad.py
Description   : Verification and quality-control (QC) checks for the processed
                OULAD artifacts produced by the preprocessing pipeline.

                The script validates that:
                - Required output files exist and can be loaded
                - X, y, and sensitive files have aligned row counts
                - Sensitive attributes did not leak into the feature matrix (X)
                - The target variable (y) is binary and well-formed
                - Numerical features appear standardized (mean ~ 0, std ~ 1)
                - Ordinal features (e.g., prior_education) preserve integer encoding
                - Sensitive attribute distributions can be inspected for sanity

Input Data    : data/processed/oulad/
                - X.csv
                - y.csv
                - sensitive.csv
                - feature_metadata.json

Outputs       : Console verification log (stdout). No files are modified.

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
- This verifier is intended as a lightweight, deterministic sanity-check step
  for reproducibility and fairness-aware workflows.
- Checks are conservative: failures indicate likely data integrity issues or
  schema drift that should be addressed before training and evaluation.

Dependencies :
- Python >= 3.9
- pandas, numpy

===============================================================================
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("data/processed/oulad")

def verify():
    print("--- 🔍 OULAD Verification Log ---")
    
    # 1. Load Artifacts
    try:
        X = pd.read_csv(DATA_DIR / "X.csv")
        y = pd.read_csv(DATA_DIR / "y.csv")
        sensitive = pd.read_csv(DATA_DIR / "sensitive.csv")
        with open(DATA_DIR / "feature_metadata.json") as f:
            meta = json.load(f)
        print(f"✅ All files exist in {DATA_DIR}")
    except FileNotFoundError as e:
        print(f"❌ FATAL: Missing file - {e}")
        return

    # 2. Check Alignment
    n_X, n_y, n_s = len(X), len(y), len(sensitive)
    if n_X == n_y == n_s:
        print(f"✅ Row counts align: {n_X} rows")
    else:
        print(f"❌ FATAL: Row mismatch! X={n_X}, y={n_y}, sens={n_s}")
        return

    # 3. Check for Target Leakage
    # Ensure no sensitive columns leaked into X
    sens_cols = set(sensitive.columns)
    x_cols = set(X.columns)
    intersect = sens_cols.intersection(x_cols)
    if not intersect:
        print("✅ No sensitive attributes leaked into X")
    else:
        print(f"❌ FATAL: Sensitive Leakage detected in X: {intersect}")

    # 4. Check Target Validity
    y_vals = y.iloc[:, 0].unique()
    if set(y_vals).issubset({0, 1}):
        print(f"✅ Target is binary {set(y_vals)}")
    else:
        print(f"❌ FATAL: Target contains non-binary values: {y_vals}")

    # 5. Check Feature Scaling (StandardScaler)
    # Numerical columns should have mean ~0 and std ~1
    # We grab numericals from metadata
    num_cols = meta["numerical_columns"]
    if num_cols:
        means = X[num_cols].mean()
        stds = X[num_cols].std()
        if (means.abs() < 0.5).all() and (stds.between(0.5, 1.5)).all():
            print("✅ Numerical features appear scaled (Mean ~0, Std ~1)")
        else:
            print("⚠️ WARNING: Suspicious scaling. Check means:\n", means)

    # 6. Check Ordinal Integrity (Prior Education)
    # Should be integers 0, 1, 2, 3, 4 (maybe 5 for unknown)
    if "prior_education" in X.columns:
        edu_vals = sorted(X["prior_education"].unique())
        print(f"ℹ️  Prior Education Levels found: {edu_vals}")
        if len(edu_vals) >= 3 and all(isinstance(v, (int, np.integer)) for v in edu_vals):
             print("✅ Prior Education is ordinal integer encoded")
        else:
             print("❌ FATAL: Prior Education is NOT ordinal integers")

    # 7. Check Sensitive Attributes (Harmonization)
    print("\n--- Sensitive Attribute Distribution ---")
    print(sensitive["gender"].value_counts(normalize=True))
    print(sensitive["ses_quintile"].value_counts(normalize=True).sort_index())
    
    if "0" in sensitive["gender"].values and "1" in sensitive["gender"].values:
        print("✅ Gender normalized to 0/1")
    else:
        print("❌ Gender NOT normalized (expected 0/1)")

if __name__ == "__main__":
    verify()
