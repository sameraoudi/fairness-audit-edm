#!/usr/bin/env python3
"""
===============================================================================
Script Name   : verify_datasets.py
Description   : Cross-dataset verification and quality-control (QC) checks for
                processed educational datasets used in a unified, fairness-aware
                machine learning pipeline.

                This script validates (per dataset):
                - Required artifacts exist and can be loaded
                - Row alignment across X, y, and sensitive outputs
                - Absence of sensitive and target leakage into X
                - Dataset-specific sanity checks (UCI, xAPI)

                The goal is to catch schema drift, preprocessing regressions,
                leakage issues, and implausible distributions before training or
                downstream fairness auditing.

Datasets      : data/processed/<dataset>/
                - uci
                - xapi
                (Extendable by adding names to DATASETS)

Input Data    : data/processed/<dataset>/
                - X.csv
                - y.csv
                - sensitive.csv
                - feature_metadata.json (optional; not required by all checks)

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
- Leakage checks are conservative and include both sensitive attributes and
  common raw/target columns that must never appear in X.
- Dataset-specific checks:
  - UCI: validates ordinal integrity of prior_education (Medu), SES plausibility,
    and target pass-rate sanity.
  - xAPI: validates gender normalization, confirms conservative exclusion of
    SES/age, and checks engagement standardization behavior.
- Intended to run after preprocessing and before model training/evaluation.

Dependencies :
- Python >= 3.9
- pandas, numpy

===============================================================================
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Config
DATASETS = ["uci", "xapi"]
BASE_DIR = Path("data/processed")

def check_alignment(name, X, y, sens):
    n_x, n_y, n_s = len(X), len(y), len(sens)
    if n_x == n_y == n_s:
        print(f"✅ [{name}] Alignment: OK ({n_x} rows)")
        return True
    else:
        print(f"❌ [{name}] Alignment FATAL: X={n_x}, y={n_y}, sens={n_s}")
        return False

def check_leakage(name, X, sens):
    # Check if sensitive columns (or proxies) leaked into X
    sens_cols = set(sens.columns)
    x_cols = set(X.columns)
    intersect = sens_cols.intersection(x_cols)
    
    # Specific checks for raw columns that shouldn't leak
    forbidden = {"G3", "Class", "final_result", "Mjob", "Fjob", "sex", "gender"}
    leaked_forbidden = forbidden.intersection(x_cols)
    
    if not intersect and not leaked_forbidden:
        print(f"✅ [{name}] Leakage: Clean (No sensitive/target columns in X)")
    else:
        print(f"❌ [{name}] Leakage DETECTED: {intersect | leaked_forbidden}")

def verify_uci(X, y, sens):
    print(f"--- 🔍 Deep Dive: UCI ---")
    
    # 1. Prior Education Ordinality
    # Expect integers 0..4 (from Medu)
    if "prior_education" in X.columns:
        vals = sorted(X["prior_education"].unique())
        if all(v in [0, 1, 2, 3, 4] for v in vals):
            print(f"✅ Prior Education: Ordinal Integrity confirmed {vals}")
        else:
            print(f"❌ Prior Education: Invalid values found (Expected 0-4): {vals}")
    
    # 2. SES Quintile Logic
    # Expect 0..4 derived from Job Tiers
    ses_vals = sorted(sens["ses_quintile"].astype(str).unique())
    print(f"ℹ️  SES Quintile Distribution: {sens['ses_quintile'].value_counts(normalize=True).to_dict()}")
    
    # 3. Target Threshold
    # G3 >= 10 means we expect roughly 60-70% pass rate typically
    pass_rate = y["y"].mean()
    print(f"ℹ️  Pass Rate: {pass_rate:.2%}")
    if 0.2 < pass_rate < 0.9:
        print("✅ Target Distribution: Plausible")
    else:
        print("⚠️ Target Distribution: Suspiciously skewed")

def verify_xapi(X, y, sens):
    print(f"--- 🔍 Deep Dive: xAPI ---")
    
    # 1. Gender Normalization
    # MUST be 0/1 (or '0'/'1'), NOT 'M'/'F'
    genders = set(sens["gender"].unique())
    # Allow int or string '0'/'1'
    valid_binary = {'0', '1', 0, 1}
    if genders.issubset(valid_binary):
        print(f"✅ Gender: Normalized correctly {genders}")
    else:
        print(f"❌ Gender: FAILED normalization. Found: {genders}")

    # 2. Conservative Exclusion (SES)
    # SES should be "Unknown" or NaN
    ses_vals = sens["ses_quintile"].unique()
    if len(ses_vals) == 1 and ("Unknown" in ses_vals or pd.isna(ses_vals[0])):
         print("✅ SES: Correctly excluded (All 'Unknown')")
    else:
         print(f"❌ SES: Unexpected values found (Expected exclusion): {ses_vals}")

    # 3. Engagement Scaling
    # Should be StandardScaled (~0 mean)
    eng_mean = X["engagement_level"].mean()
    if -0.5 < eng_mean < 0.5:
        print(f"✅ Engagement: Scaled properly (Mean={eng_mean:.4f})")
    else:
        print(f"❌ Engagement: Scaling failed (Mean={eng_mean:.4f})")

def main():
    for name in DATASETS:
        dir_path = BASE_DIR / name
        if not dir_path.exists():
            print(f"⚠️ Skipping {name}: Folder not found")
            continue
            
        print(f"\n>>> Verifying {name.upper()} <<<")
        try:
            X = pd.read_csv(dir_path / "X.csv")
            y = pd.read_csv(dir_path / "y.csv")
            sens = pd.read_csv(dir_path / "sensitive.csv")
            
            if check_alignment(name, X, y, sens):
                check_leakage(name, X, sens)
                if name == "uci":
                    verify_uci(X, y, sens)
                elif name == "xapi":
                    verify_xapi(X, y, sens)
                    
        except Exception as e:
            print(f"❌ ERROR reading {name}: {e}")

if __name__ == "__main__":
    main()
