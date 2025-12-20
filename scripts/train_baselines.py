#!/usr/bin/env python3
"""
===============================================================================
Script Name   : train_baselines.py
Description   : Baseline model training and evaluation for unified, processed
                educational datasets in a fairness-aware machine learning
                pipeline.

                The script:
                - Loads processed artifacts (X, y, sensitive) for a selected dataset
                - Creates intersectional stratification keys (y + gender + SES)
                  to preserve subgroup representation across splits
                - Generates reproducible train/validation/test splits (70/15/15)
                - Trains baseline classifiers:
                    * Logistic Regression
                    * Random Forest
                    * XGBoost
                - Saves trained models, split indices, and logs performance metrics

How to Run   :
                python scripts/train_baselines.py --dataset oulad
                python scripts/train_baselines.py --dataset uci
                python scripts/train_baselines.py --dataset xapi

Inputs        :
                configs/fairness_constraints.yaml
                  - random_seed (used to ensure consistent splits)

                data/processed/<dataset>/
                  - X.csv
                  - y.csv
                  - sensitive.csv

Outputs       :
                models/baselines/<dataset>/
                  - LogisticRegression.pkl
                  - RandomForest.pkl
                  - XGBoost.pkl

                splits/<dataset>/
                  - train_idx.npy
                  - val_idx.npy
                  - test_idx.npy

                results/
                  - baseline_performance.csv (appended per run)

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
- Intersectional stratification is used to reduce subgroup under-representation
  in the test set (particularly for fairness evaluation).
- Singleton strata (subgroups with insufficient samples) are handled via a
  conservative fallback to y-only stratification for those rows.
- Class imbalance is mitigated using class_weight='balanced' (where supported).
- Results are appended to a single CSV to support multi-dataset benchmarking.

Dependencies :
- Python >= 3.9
- pandas, numpy, scikit-learn
- xgboost
- pyyaml

===============================================================================
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
# Load seed from global config to ensure consistency
with open("configs/fairness_constraints.yaml") as f:
    config = yaml.safe_load(f)
    RANDOM_SEED = config.get("random_seed", 42)

MODELS_DIR = Path("models/baselines")
RESULTS_DIR = Path("results")
SPLITS_DIR = Path("splits")

def load_processed_data(dataset_name):
    base_path = Path(f"data/processed/{dataset_name}")
    X = pd.read_csv(base_path / "X.csv")
    y = pd.read_csv(base_path / "y.csv")
    sensitive = pd.read_csv(base_path / "sensitive.csv")
    return X, y, sensitive

def create_intersectional_strata(y, sensitive):
    """
    Creates a stratify_key = y + Gender + SES + Age
    to ensure all subgroups are represented in Test set.
    """
    # Convert all to string and concatenate
    strata = y["y"].astype(str) + "_" + \
             sensitive["gender"].astype(str) + "_" + \
             sensitive["ses_quintile"].astype(str)
             # Note: We omit smaller groups if they cause split errors (singletons)
             # But Gender+SES is the critical intersection for this paper.
    return strata

def save_splits(dataset_name, train_idx, val_idx, test_idx):
    out_dir = SPLITS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "val_idx.npy", val_idx)
    np.save(out_dir / "test_idx.npy", test_idx)
    print(f"[{dataset_name}] Saved splits to {out_dir}")

def train_and_evaluate(dataset_name, X, y, sensitive):
    # 1. Create Intersectional Strata
    print(f"[{dataset_name}] Creating intersectional stratified splits...")
    strata = create_intersectional_strata(y, sensitive)
    
    # Check for singleton groups (groups with only 1 member cannot be split)
    counts = strata.value_counts()
    singletons = counts[counts < 2].index
    if len(singletons) > 0:
        print(f"⚠️ Warning: {len(singletons)} singleton groups found. Falling back to y-stratification for these rows.")
        # Fallback: Replace stratum label with just 'y' label for these rare cases
        strata = strata.apply(lambda x: x if x not in singletons else x.split('_')[0])

    # 2. Split (70/15/15)
    # First split: Train (70) vs Temp (30)
    indices = np.arange(len(X))
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices, test_size=0.30, stratify=strata, random_state=RANDOM_SEED
    )
    
    # Re-generate strata for the temp set to split it 50/50 (15/15 overall)
    strata_temp = strata.iloc[idx_temp]
    # Check singletons again in temp
    counts_temp = strata_temp.value_counts()
    singletons_temp = counts_temp[counts_temp < 2].index
    if len(singletons_temp) > 0:
         strata_temp = strata_temp.apply(lambda x: x if x not in singletons_temp else x.split('_')[0])

    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.50, stratify=strata_temp, random_state=RANDOM_SEED
    )

    save_splits(dataset_name, idx_train, idx_val, idx_test)

    # 3. Define Baselines
    # Note: Class_weight='balanced' helps with the imbalance often seen in failure rates
    models = {
        "LogisticRegression": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED)
    }

    # 4. Train & Log
    out_dir = MODELS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    results = []

    print(f"[{dataset_name}] Training baselines...")
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train.values.ravel())
        
        # Save Model
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Predict (Test Set)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else []
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs) if len(probs) > 0 else 0.0
        
        results.append({
            "dataset": dataset_name,
            "model": name,
            "accuracy": round(acc, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4)
        })
        print(f"   -> {name}: AUC={auc:.4f}, Recall={rec:.4f}")

    # Append to global CSV
    res_df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "baseline_performance.csv"
    mode = 'a' if csv_path.exists() else 'w'
    header = not csv_path.exists()
    res_df.to_csv(csv_path, mode=mode, header=header, index=False)
    print(f"[{dataset_name}] Performance logged to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="oulad, uci, or xapi")
    args = parser.parse_args()
    
    X, y, sens = load_processed_data(args.dataset)
    train_and_evaluate(args.dataset, X, y, sens)
