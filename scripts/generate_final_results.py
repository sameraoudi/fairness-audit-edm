#!/usr/bin/env python3
"""
===============================================================================
Script Name   : generate_final_results.py
Description   : Final results aggregation script for the educational fairness
                study. This script generates the comparative performance and
                fairness metrics used in the paper by evaluating baseline and
                mitigation methods on the held-out test sets across datasets.

                For each dataset (OULAD, UCI, xAPI), the script:
                - Loads the processed test set (X, y, sensitive) using saved
                  test indices to ensure a fixed evaluation protocol
                - Evaluates the following methods (when available):
                    1) Baseline (XGBoost)
                    2) Reweighing (pre-processing mitigation)
                    3) Thresholding (post-processing mitigation; Fairlearn)
                    4) Adversarial Debiasing (deep learning; GRL-based)
                - Computes utility metrics:
                    * Accuracy, Recall, F1
                - Computes fairness metrics (Fairlearn):
                    * Demographic Parity Difference (DP_Diff)
                    * Equalized Odds Difference (EO_Diff)
                - Exports a publication-ready CSV table to results/

How to Run   :
                python scripts/generate_paper_results.py

Inputs        :
                data/processed/<dataset>/
                  - X.csv
                  - y.csv
                  - sensitive.csv

                splits/<dataset>/
                  - test_idx.npy

                models/
                  baselines/<dataset>/XGBoost.pkl
                  mitigated_pre/<dataset>_reweighed.pkl
                  mitigated_post/<dataset>_threshold.pkl
                  mitigated_adversarial/<dataset>_adversarial.pth

Outputs       :
                results/
                  - final_paper_results.csv

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
- Evaluation is performed strictly on the saved test split indices to ensure
  reproducibility and prevent leakage from training/validation.
- The sensitive attribute used for fairness metrics is selected as:
    * ses_quintile for OULAD/UCI (when available)
    * gender for xAPI (SES not available)
- Fairlearn fairness metrics require sensitive_features; values are cast to
  strings to avoid category-type inconsistencies across datasets.
- The adversarial model is re-instantiated with the same architecture used in
  training and then loaded via state_dict. Architecture changes will invalidate
  checkpoint compatibility.
- The output CSV is formatted as a paper-ready table (Dataset × Method).

Dependencies :
- Python >= 3.9
- pandas, numpy, scikit-learn
- fairlearn
- torch
- xgboost

===============================================================================
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# ---------------------------------------------------------
# CONFIG & DEFINITIONS
# ---------------------------------------------------------
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
SPLITS_DIR = Path("splits")

# Re-define FairNet for loading PyTorch weights
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class FairNet(nn.Module):
    def __init__(self, input_dim, n_sensitive_classes):
        super(FairNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Linear(64, 1))
        self.adversary = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_sensitive_classes)
        )

    def forward(self, x, alpha=1.0):
        features = self.encoder(x)
        y_logits = self.classifier(features)
        reverse_features = GradientReversal.apply(features, alpha)
        s_logits = self.adversary(reverse_features)
        return y_logits, s_logits

def load_data(dataset):
    base = Path(f"data/processed/{dataset}")
    X = pd.read_csv(base / "X.csv")
    y = pd.read_csv(base / "y.csv")
    sens = pd.read_csv(base / "sensitive.csv")
    test_idx = np.load(SPLITS_DIR / dataset / "test_idx.npy")
    return X.iloc[test_idx], y.iloc[test_idx], sens.iloc[test_idx]

def get_metrics(y_true, y_pred, sensitive_features):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "DP_Diff": round(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features), 4),
        "EO_Diff": round(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features), 4)
    }

def main():
    print("--- 📊 Generating Final Comparative Results (Step 10) ---")
    datasets = ["oulad", "uci", "xapi"]
    final_results = []

    for ds in datasets:
        print(f"\nProcessing {ds}...")
        X_test, y_test, sens_test = load_data(ds)
        y_true = y_test.values.ravel()
        
        # Determine Sensitive Attribute for Metrics
        sens_col = "ses_quintile" if "ses_quintile" in sens_test.columns and ds != "xapi" else "gender"
        s_data = sens_test[sens_col].astype(str) # For Fairlearn metrics
        print(f"   (Audit Attribute: {sens_col})")

        # -------------------------------------------------
        # 1. Baseline (XGBoost)
        # -------------------------------------------------
        try:
            with open(MODELS_DIR / "baselines" / ds / "XGBoost.pkl", "rb") as f:
                model = pickle.load(f)
            preds = model.predict(X_test)
            m = get_metrics(y_true, preds, s_data)
            m.update({"Dataset": ds, "Method": "Baseline (XGBoost)"})
            final_results.append(m)
            print(f"   ✅ Baseline: Recall={m['Recall']}, EO_Diff={m['EO_Diff']}")
        except Exception as e:
            print(f"   ❌ Baseline failed: {e}")

        # -------------------------------------------------
        # 2. Reweighing (Pre-proc)
        # -------------------------------------------------
        try:
            with open(MODELS_DIR / "mitigated_pre" / f"{ds}_reweighed.pkl", "rb") as f:
                model = pickle.load(f)
            preds = model.predict(X_test)
            m = get_metrics(y_true, preds, s_data)
            m.update({"Dataset": ds, "Method": "Reweighing"})
            final_results.append(m)
            print(f"   ✅ Reweighing: Recall={m['Recall']}, EO_Diff={m['EO_Diff']}")
        except Exception as e:
            print(f"   ⚠️ Reweighing not found/failed")

        # -------------------------------------------------
        # 3. Thresholding (Post-proc)
        # -------------------------------------------------
        try:
            with open(MODELS_DIR / "mitigated_post" / f"{ds}_threshold.pkl", "rb") as f:
                model = pickle.load(f)
            # ThresholdOptimizer requires sensitive_features arg
            preds = model.predict(X_test, sensitive_features=sens_test[sens_col].astype(str))
            m = get_metrics(y_true, preds, s_data)
            m.update({"Dataset": ds, "Method": "Thresholding"})
            final_results.append(m)
            print(f"   ✅ Thresholding: Recall={m['Recall']}, EO_Diff={m['EO_Diff']}")
        except Exception as e:
             print(f"   ⚠️ Thresholding not found/failed: {e}")

        # -------------------------------------------------
        # 4. Adversarial (Deep Learning)
        # -------------------------------------------------
        try:
            # Re-map sensitive classes to match training
            s_classes = sorted(s_data.unique())
            
            # Init Model
            model = FairNet(input_dim=X_test.shape[1], n_sensitive_classes=len(s_classes))
            model.load_state_dict(torch.load(MODELS_DIR / "mitigated_adversarial" / f"{ds}_adversarial.pth"))
            model.eval()
            
            with torch.no_grad():
                logits, _ = model(torch.FloatTensor(X_test.values))
                preds = (torch.sigmoid(logits).numpy().flatten() >= 0.5).astype(int)
                
            m = get_metrics(y_true, preds, s_data)
            m.update({"Dataset": ds, "Method": "Adversarial Debiasing"})
            final_results.append(m)
            print(f"   ✅ Adversarial: Recall={m['Recall']}, EO_Diff={m['EO_Diff']}")
        except Exception as e:
            print(f"   ❌ Adversarial failed: {e}")

    # Save
    df = pd.DataFrame(final_results)
    # Reorder columns
    cols = ["Dataset", "Method", "Accuracy", "Recall", "F1", "DP_Diff", "EO_Diff"]
    df = df[cols]
    df.to_csv(RESULTS_DIR / "final_paper_results.csv", index=False)
    print(f"\n✅ SUCCESS. Final results saved to {RESULTS_DIR / 'final_paper_results.csv'}")
    print(df.to_string())

if __name__ == "__main__":
    main()
