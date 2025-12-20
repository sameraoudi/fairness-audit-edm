#!/usr/bin/env python3
"""
===============================================================================
Script Name   : audit_fairness.py
Description   : Fairness audit and differential explainability (XAI) analysis
                for baseline classifiers trained on unified, processed
                educational datasets (OULAD, UCI, xAPI).

                The script:
                - Loads the processed test split (X, y, sensitive) for a dataset
                - Loads the trained baseline model (defaults to XGBoost)
                - Computes group-level performance and selection metrics
                - Quantifies disparities using:
                    * Demographic Parity Difference (DP Diff)
                    * Disparate Impact Ratio (DI Ratio)
                    * Equal Opportunity Difference (EOD)
                - Optionally performs intersectional auditing (Gender × SES)
                - Runs differential SHAP analysis (TreeExplainer) to identify
                  top features driving decisions for reference vs. worst group
                - Saves structured audit results to JSON for reporting

How to Run   :
                python scripts/audit_fairness.py --dataset oulad
                python scripts/audit_fairness.py --dataset uci
                python scripts/audit_fairness.py --dataset xapi

Inputs        :
                configs/fairness_constraints.yaml
                  - min_subgroup_size
                  - bootstrap_samples (reserved for CI workflows if extended)

                data/processed/<dataset>/
                  - X.csv
                  - y.csv
                  - sensitive.csv

                splits/<dataset>/
                  - test_idx.npy

                models/baselines/<dataset>/
                  - XGBoost.pkl (expected)

Outputs       :
                results/
                  - fairness_audit_<dataset>.json

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
- Auditing is performed strictly on the held-out test set indices to avoid
  optimistic bias in fairness reporting.
- Groups with sample size below min_subgroup_size are excluded from disparity
  comparisons (guardrail against unreliable inference).
- Reference group is selected as the group with the highest selection rate to
  support Disparate Impact comparisons (DI Ratio).
- Differential SHAP is computed per group pair using TreeExplainer:
  the top features for the reference group and worst DI group are reported.
- For XGBoost 2.0+, SHAP compatibility is improved by explaining the raw booster
  when available (model.get_booster()).

Dependencies :
- Python >= 3.9
- pandas, numpy, scikit-learn
- pyyaml
- shap
- xgboost

===============================================================================
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
import yaml
import shap
from pathlib import Path
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
with open("configs/fairness_constraints.yaml") as f:
    config = yaml.safe_load(f)
    MIN_N = config.get("min_subgroup_size", 50)
    BOOTSTRAPS = config.get("bootstrap_samples", 1000)

MODELS_DIR = Path("models/baselines")
RESULTS_DIR = Path("results")
SPLITS_DIR = Path("splits")

def load_data_and_splits(dataset):
    # Load Processed Data
    base = Path(f"data/processed/{dataset}")
    X = pd.read_csv(base / "X.csv")
    y = pd.read_csv(base / "y.csv")
    sens = pd.read_csv(base / "sensitive.csv")
    
    # Load Test Indices (We ONLY audit the Test Set)
    test_idx = np.load(SPLITS_DIR / dataset / "test_idx.npy")
    
    return X.iloc[test_idx], y.iloc[test_idx], sens.iloc[test_idx]

def compute_group_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    
    selection_rate = (tp + fp) / n if n > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Equality of Opportunity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / n
    
    return {
        "n": int(n),
        "selection_rate": float(selection_rate),
        "tpr": float(tpr),
        "accuracy": float(accuracy)
    }

def audit_attribute(attr_name, groups, y_true, y_pred):
    """
    Audits a specific attribute (e.g., Gender) by comparing all groups
    against the 'Privileged' (highest selection rate or majority) group.
    """
    unique_groups = groups.unique()
    
    # 1. Calculate base metrics for every group
    group_stats = {}
    for g in unique_groups:
        mask = (groups == g)
        if mask.sum() < MIN_N:
            continue # Skip small groups per guardrail
        
        stats = compute_group_metrics(y_true[mask], y_pred[mask])
        group_stats[str(g)] = stats

    if len(group_stats) < 2:
        return None # Not enough groups to compare

    # 2. Identify Reference Group (Heuristic: Largest group usually, or highest privilege)
    # For this paper, we select the group with the HIGHEST Selection Rate as 'Privileged' reference
    # to measure Disparate Impact against.
    ref_group = max(group_stats, key=lambda k: group_stats[k]['selection_rate'])
    ref_stats = group_stats[ref_group]
    
    audit_results = {
        "attribute": attr_name,
        "reference_group": ref_group,
        "disparities": {}
    }

    # 3. Calculate Disparities
    for g, stats in group_stats.items():
        if g == ref_group:
            continue
            
        # Demographic Parity Difference (Selection Rate A - Selection Rate B)
        dp_diff = ref_stats['selection_rate'] - stats['selection_rate']
        
        # Disparate Impact Ratio (Selection Rate B / Selection Rate A)
        di_ratio = stats['selection_rate'] / ref_stats['selection_rate'] if ref_stats['selection_rate'] > 0 else 0
        
        # Equal Opportunity Difference (Recall A - Recall B)
        # Positive value means Reference group has better Recall (Bias against G)
        eod = ref_stats['tpr'] - stats['tpr']

        audit_results["disparities"][g] = {
            "n": stats["n"],
            "dp_diff": round(dp_diff, 4),
            "di_ratio": round(di_ratio, 4),
            "eod": round(eod, 4)
        }
    
    return audit_results

def run_differential_xai(model, X, sens, attr_col, ref_group, target_group):
    """
    Calculates SHAP values separately for Reference vs Target group.
    Returns the top 3 features that drive decisions for each group.
    """
    # --- FIX START: Handle XGBoost 2.0+ / SHAP mismatch ---
    # XGBoost 2.0+ stores base_score as a list string '[0.5]', crashing SHAP.
    # Passing the raw booster bypasses the sklearn wrapper metadata issue.
    if hasattr(model, "get_booster"):
        model_to_explain = model.get_booster()
    else:
        model_to_explain = model
    # --- FIX END ---

    try:
        explainer = shap.TreeExplainer(model_to_explain)
    except Exception as e:
        print(f"⚠️ SHAP Explainer Failed: {e}. Skipping XAI for this pair.")
        return None
    
    # Filter Data
    X_ref = X[sens[attr_col].astype(str) == str(ref_group)]
    X_tgt = X[sens[attr_col].astype(str) == str(target_group)]
    
    # Subsample if large (SHAP is expensive)
    if len(X_ref) > 200: X_ref = X_ref.sample(200, random_state=42)
    if len(X_tgt) > 200: X_tgt = X_tgt.sample(200, random_state=42)
    
    if len(X_ref) < 10 or len(X_tgt) < 10:
        return None

    # Calculate SHAP values
    # check_additivity=False suppresses errors if XGBoost uses specific approx methods
    shap_ref = explainer.shap_values(X_ref, check_additivity=False)
    shap_tgt = explainer.shap_values(X_tgt, check_additivity=False)
    
    # Handle Output Formats:
    # XGBClassifier (wrapper) returns [Negative, Positive] list for binary.
    # Raw Booster (model_to_explain) returns just [LogOdds] array.
    
    # If it's a list (from wrapper or multiclass), take index 1 (Positive Class)
    if isinstance(shap_ref, list): 
        shap_ref = shap_ref[1]
        shap_tgt = shap_tgt[1]
    
    # If it's a raw booster output, it's already the single array we need.
    
    # Get mean absolute SHAP per feature
    mean_ref = np.abs(shap_ref).mean(axis=0)
    mean_tgt = np.abs(shap_tgt).mean(axis=0)
    
    features = X.columns
    
    # Rank features
    ref_ranks = pd.DataFrame({"feature": features, "importance": mean_ref}).sort_values("importance", ascending=False)
    tgt_ranks = pd.DataFrame({"feature": features, "importance": mean_tgt}).sort_values("importance", ascending=False)
    
    return {
        "ref_top_features": ref_ranks.head(3).to_dict(orient="records"),
        "tgt_top_features": tgt_ranks.head(3).to_dict(orient="records")
    }

def main(dataset):
    print(f"[{dataset}] Starting Fairness Audit...")
    X_test, y_test, sens_test = load_data_and_splits(dataset)
    
    # Load BEST baseline (Defaulting to XGBoost as it's usually strongest)
    model_path = MODELS_DIR / dataset / "XGBoost.pkl"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    preds = model.predict(X_test)
    
    results = {"dataset": dataset, "audits": [], "xai": []}

    # 1. Audit Single Attributes
    for col in sens_test.columns:
        print(f"   -> Auditing {col}...")
        audit = audit_attribute(col, sens_test[col], y_test.values.ravel(), preds)
        if audit:
            results["audits"].append(audit)
            
            # Run XAI for the largest disparity
            # Find group with worst Disparate Impact
            worst_group = min(audit["disparities"], key=lambda k: audit["disparities"][k]["di_ratio"])
            xai_res = run_differential_xai(model, X_test, sens_test, col, audit["reference_group"], worst_group)
            if xai_res:
                xai_res["attribute"] = col
                xai_res["comparison"] = f"{audit['reference_group']} vs {worst_group}"
                results["xai"].append(xai_res)

    # 2. Audit Intersectional (SES x Gender)
    # Only if both exist
    if "gender" in sens_test.columns and "ses_quintile" in sens_test.columns:
        print(f"   -> Auditing Intersection: Gender x SES...")
        inter_group = sens_test["gender"].astype(str) + "_" + sens_test["ses_quintile"].astype(str)
        audit = audit_attribute("Gender_x_SES", inter_group, y_test.values.ravel(), preds)
        if audit:
            results["audits"].append(audit)

    # Save
    out_file = RESULTS_DIR / f"fairness_audit_{dataset}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{dataset}] Audit Complete. Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    main(args.dataset)
