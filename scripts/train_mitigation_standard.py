#!/usr/bin/env python3
"""
===============================================================================
Script Name   : train_mitigation_standard.py
Description   : Standard fairness mitigation training and evaluation for unified,
                processed educational datasets (OULAD, UCI, xAPI).

                This script implements two widely used, publication-standard
                mitigation approaches for binary classification:

                (1) Pre-processing mitigation: Reweighing
                    - Computes inverse-probability sample weights for the
                      intersection of (label y × sensitive group).
                    - Retrains an XGBoost classifier using sample_weight to
                      reduce representation-induced bias.

                (2) Post-processing mitigation: Fairlearn ThresholdOptimizer
                    - Keeps the baseline model frozen (no retraining).
                    - Fits group-specific decision thresholds on a validation set
                      under an equalized_odds constraint.
                    - Applies the learned thresholding policy to the test set.

                The script logs utility metrics (Accuracy, Recall, F1, ROC-AUC
                where applicable) and saves trained mitigation artifacts.

How to Run   :
                python scripts/train_mitigation_standard.py --dataset oulad
                python scripts/train_mitigation_standard.py --dataset uci
                python scripts/train_mitigation_standard.py --dataset xapi

Inputs        :
                configs/fairness_constraints.yaml
                  - random_seed (reproducibility)

                data/processed/<dataset>/
                  - X.csv
                  - y.csv
                  - sensitive.csv

                splits/<dataset>/
                  - train_idx.npy
                  - val_idx.npy
                  - test_idx.npy

                models/baselines/<dataset>/
                  - XGBoost.pkl  (required for ThresholdOptimizer; frozen baseline)

Outputs       :
                models/mitigated_pre/
                  - <dataset>_reweighed.pkl

                models/mitigated_post/
                  - <dataset>_threshold.pkl

                results/
                  - mitigation_standard_performance.csv

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
- Mitigation target attribute defaults to ses_quintile; if unavailable (xAPI),
  the script falls back to gender.
- Evaluation is performed strictly on the held-out test set indices to avoid
  optimistic reporting.
- ThresholdOptimizer produces hard decisions (0/1), therefore ROC-AUC is not
  computed for the post-processing method (reported as N/A).
- The baseline model used for post-processing must already exist; run the
  baseline training step prior to executing this script.

Dependencies :
- Python >= 3.9
- pandas, numpy, scikit-learn
- xgboost
- pyyaml
- fairlearn
- torch
===============================================================================
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from fairlearn.postprocessing import ThresholdOptimizer

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
with open("configs/fairness_constraints.yaml") as f:
    config = yaml.safe_load(f)
    RANDOM_SEED = config.get("random_seed", 42)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
SPLITS_DIR = Path("splits")

MODELS_DIR_PRE = MODELS_DIR / "mitigated_pre"
MODELS_DIR_POST = MODELS_DIR / "mitigated_post"
MODELS_DIR_PRE.mkdir(parents=True, exist_ok=True)
MODELS_DIR_POST.mkdir(parents=True, exist_ok=True)

def load_data(dataset):
    base = Path(f"data/processed/{dataset}")
    X = pd.read_csv(base / "X.csv")
    y = pd.read_csv(base / "y.csv")
    sens = pd.read_csv(base / "sensitive.csv")
    
    # Load splits
    train_idx = np.load(SPLITS_DIR / dataset / "train_idx.npy")
    val_idx = np.load(SPLITS_DIR / dataset / "val_idx.npy")
    test_idx = np.load(SPLITS_DIR / dataset / "test_idx.npy")
    
    return {
        "X_train": X.iloc[train_idx], "y_train": y.iloc[train_idx], "s_train": sens.iloc[train_idx],
        "X_val": X.iloc[val_idx],   "y_val": y.iloc[val_idx],   "s_val": sens.iloc[val_idx],
        "X_test": X.iloc[test_idx], "y_test": y.iloc[test_idx], "s_test": sens.iloc[test_idx]
    }

def compute_sample_weights(y, sensitive_col):
    """
    Pre-processing: Inverse Probability Reweighing.
    Weights samples inversely to their group frequency to balance representation.
    """
    # Create strata: y + sensitive
    strata = y.astype(str) + "_" + sensitive_col.astype(str)
    counts = strata.value_counts()
    n_total = len(y)
    n_strata = len(counts)
    
    # Weight = Total / (n_classes * count_of_stratum)
    weights = strata.map(lambda x: n_total / (n_strata * counts[x]))
    return weights

def evaluate(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4)
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        except:
            metrics["roc_auc"] = 0.0
    return metrics

def run_standard_mitigation(dataset):
    print(f"[{dataset}] Starting Standard Mitigation...")
    d = load_data(dataset)
    
    # We focus on SES Quintile as the primary sensitive attribute for mitigation
    # (based on your audit findings showing SES bias > Gender bias)
    # If SES is missing (xAPI), we fall back to Gender.
    sens_col = "ses_quintile"
    if dataset == "xapi":
        sens_col = "gender"
        print(f"[{dataset}] SES not available, mitigating on Gender.")
    
    s_train = d["s_train"][sens_col].astype(str)
    s_val = d["s_val"][sens_col].astype(str)
    s_test = d["s_test"][sens_col].astype(str)

    results = []

    # ---------------------------------------------------------
    # METHOD 1: Pre-processing (Reweighing)
    # ---------------------------------------------------------
    print(f"   -> Running Pre-processing (Reweighing)...")
    sample_weights = compute_sample_weights(d["y_train"]["y"], s_train)
    
    # Train new XGBoost with weights
    # Note: Using XGBoost 1.7.6 compatible syntax
    model_pre = XGBClassifier(
        random_state=RANDOM_SEED, 
        eval_metric='logloss'
    )
    model_pre.fit(d["X_train"], d["y_train"].values.ravel(), sample_weight=sample_weights)
    
    # Evaluate
    preds_pre = model_pre.predict(d["X_test"])
    probs_pre = model_pre.predict_proba(d["X_test"])[:, 1]
    
    metrics_pre = evaluate(d["y_test"], preds_pre, probs_pre)
    metrics_pre.update({"dataset": dataset, "method": "Reweighing"})
    results.append(metrics_pre)
    
    # Save Model
    with open(MODELS_DIR_PRE / f"{dataset}_reweighed.pkl", "wb") as f:
        pickle.dump(model_pre, f)

    # ---------------------------------------------------------
    # METHOD 2: Post-processing (Threshold Optimizer)
    # ---------------------------------------------------------
    print(f"   -> Running Post-processing (Threshold Optimizer)...")
    
    # Load FROZEN Baseline (Do not retrain!)
    baseline_path = MODELS_DIR / "baselines" / dataset / "XGBoost.pkl"
    if not baseline_path.exists():
        print("❌ Baseline not found! Run Step 5 first.")
        return

    with open(baseline_path, "rb") as f:
        baseline_model = pickle.load(f)

    # Fit ThresholdOptimizer on VALIDATION set
    # Constraint: 'equalized_odds' tries to balance TPR (Recall) and FPR across groups
    post_model = ThresholdOptimizer(
        estimator=baseline_model,
        constraints="equalized_odds", 
        predict_method='predict_proba',
        prefit=True
    )
    
    post_model.fit(d["X_val"], d["y_val"], sensitive_features=s_val)
    
    # Predict on Test
    # Note: ThresholdOptimizer returns predictions (0/1), not probabilities
    preds_post = post_model.predict(d["X_test"], sensitive_features=s_test)
    
    metrics_post = evaluate(d["y_test"], preds_post) # No AUC for hard predictions
    metrics_post["roc_auc"] = "N/A" # Post-processing outputs decisions, not scores
    metrics_post.update({"dataset": dataset, "method": "ThresholdOptimizer"})
    results.append(metrics_post)
    
    # Save Model
    with open(MODELS_DIR_POST / f"{dataset}_threshold.pkl", "wb") as f:
        pickle.dump(post_model, f)

    # ---------------------------------------------------------
    # Log Results
    # ---------------------------------------------------------
    df_res = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "mitigation_standard_performance.csv"
    mode = 'a' if csv_path.exists() else 'w'
    header = not csv_path.exists()
    df_res.to_csv(csv_path, mode=mode, header=header, index=False)
    print(f"[{dataset}] Standard Mitigation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    run_standard_mitigation(args.dataset)
