#!/usr/bin/env python3
"""
===============================================================================
Script Name   : train_mitigation_adversarial.py
Description   : Adversarial debiasing (representation learning) mitigation for
                unified, processed educational datasets (OULAD, UCI, xAPI).

                This script trains a neural model that jointly:
                - Predicts the task target (student success)
                - Learns an intermediate representation that is *uninformative*
                  about a chosen sensitive attribute (SES or Gender)

                The mitigation is implemented using a Gradient Reversal Layer
                (GRL), enabling an adversarial game:
                - The classifier minimizes target prediction loss.
                - The adversary tries to predict the sensitive attribute from
                  the shared representation.
                - Through GRL, the encoder is optimized to *confuse* the
                  adversary, reducing sensitive information leakage in the
                  learned representation.

How to Run   :
                python scripts/train_mitigation_adversarial.py --dataset oulad
                python scripts/train_mitigation_adversarial.py --dataset uci
                python scripts/train_mitigation_adversarial.py --dataset xapi

Inputs        :
                configs/fairness_constraints.yaml
                  - random_seed (reproducibility)

                data/processed/<dataset>/
                  - X.csv
                  - y.csv
                  - sensitive.csv

                splits/<dataset>/
                  - train_idx.npy
                  - test_idx.npy

Outputs       :
                models/mitigated_adversarial/
                  - <dataset>_adversarial.pth      (PyTorch state_dict)

                results/
                  - mitigation_adversarial_performance.csv

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
- Sensitive attribute choice:
    * Default: ses_quintile (primary mitigation target)
    * Fallback: gender (used when SES is unavailable, e.g., xAPI)
- The trade-off parameter (ADVERSARY_WEIGHT / lambda) controls the fairness–
  utility balance; higher values typically enforce stronger invariance at
  potential cost to predictive performance.
- The adversary is treated as a multiclass classifier over sensitive groups
  (e.g., SES quintiles). Groups are mapped to integer indices at runtime.
- Evaluation is performed strictly on the held-out test set indices.
- This script saves only the PyTorch weights (state_dict). Reconstruct the
  model architecture consistently when loading for inference.

Dependencies :
- Python >= 3.9
- pandas, numpy, scikit-learn
- pyyaml
- torch

===============================================================================
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
with open("configs/fairness_constraints.yaml") as f:
    config = yaml.safe_load(f)
    RANDOM_SEED = config.get("random_seed", 42)

# Hyperparameters for Adversarial Training
HIDDEN_DIM = 64
ADVERSARY_WEIGHT = 0.5  # Lambda (Trade-off parameter)
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 256

MODELS_DIR = Path("models/mitigated_adversarial")
RESULTS_DIR = Path("results")
SPLITS_DIR = Path("splits")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# PyTorch Models
# ---------------------------------------------------------
class GradientReversal(torch.autograd.Function):
    """
    Flip the sign of the gradient during backpropagation.
    This enables the "Adversarial" game.
    """
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
        # Shared Representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )
        
        # Target Predictor (Standard Task)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1) # Binary Classification
        )
        
        # Adversary (Tries to predict Sensitive Attribute)
        self.adversary = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, n_sensitive_classes) # Multiclass (SES quintiles)
        )

    def forward(self, x, alpha=1.0):
        features = self.encoder(x)
        
        # Main Task Prediction
        y_logits = self.classifier(features)
        
        # Adversarial Branch (Gradient Reversal applied here)
        reverse_features = GradientReversal.apply(features, alpha)
        s_logits = self.adversary(reverse_features)
        
        return y_logits, s_logits

# ---------------------------------------------------------
# Training Logic
# ---------------------------------------------------------
def load_data(dataset):
    base = Path(f"data/processed/{dataset}")
    X = pd.read_csv(base / "X.csv")
    y = pd.read_csv(base / "y.csv")
    sens = pd.read_csv(base / "sensitive.csv")
    
    # Load splits
    train_idx = np.load(SPLITS_DIR / dataset / "train_idx.npy")
    test_idx = np.load(SPLITS_DIR / dataset / "test_idx.npy")
    
    return {
        "X_train": X.iloc[train_idx].values, 
        "y_train": y.iloc[train_idx].values, 
        "s_train": sens.iloc[train_idx],
        "X_test": X.iloc[test_idx].values,   
        "y_test": y.iloc[test_idx].values,   
        "s_test": sens.iloc[test_idx]
    }

def run_adversarial_debiasing(dataset):
    print(f"[{dataset}] Starting Adversarial Debiasing...")
    torch.manual_seed(RANDOM_SEED)
    
    d = load_data(dataset)
    
    # Prepare Sensitive Attribute (Target for Adversary)
    sens_col = "ses_quintile"
    if dataset == "xapi":
        sens_col = "gender" # Fallback
    
    # Convert sensitive to integer class indices
    s_train_raw = d["s_train"][sens_col].astype(str)
    s_classes = sorted(s_train_raw.unique())
    s_map = {v: i for i, v in enumerate(s_classes)}
    
    s_train_indices = s_train_raw.map(s_map).values
    
    # Tensors
    X_train = torch.FloatTensor(d["X_train"])
    y_train = torch.FloatTensor(d["y_train"])
    s_train = torch.LongTensor(s_train_indices)
    
    X_test = torch.FloatTensor(d["X_test"])
    
    # Initialize Model
    model = FairNet(input_dim=X_train.shape[1], n_sensitive_classes=len(s_classes))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    criterion_y = nn.BCEWithLogitsLoss() # For Target
    criterion_s = nn.CrossEntropyLoss()  # For Adversary
    
    # Training Loop
    model.train()
    n_samples = len(X_train)
    
    print(f"   -> Training for {EPOCHS} epochs (Lambda={ADVERSARY_WEIGHT})...")
    for epoch in range(EPOCHS):
        permutation = torch.randperm(n_samples)
        
        for i in range(0, n_samples, BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y, batch_s = X_train[indices], y_train[indices], s_train[indices]
            
            optimizer.zero_grad()
            
            y_logits, s_logits = model(batch_x, alpha=ADVERSARY_WEIGHT)
            
            # Loss = Task Loss + (Adversary Loss is inherently handled by Gradient Reversal layer)
            # We minimize: Loss_y + Loss_s
            # Because of Reversal layer, minimizing Loss_s actually MAXIMIZES the adversary's confusion.
            loss_y = criterion_y(y_logits, batch_y)
            loss_s = criterion_s(s_logits, batch_s)
            
            loss = loss_y + loss_s
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        logits_test, _ = model(X_test)
        probs_test = torch.sigmoid(logits_test).numpy().flatten()
        preds_test = (probs_test >= 0.5).astype(int)
        
    metrics = {
        "dataset": dataset,
        "method": "AdversarialDebiasing",
        "accuracy": round(accuracy_score(d["y_test"], preds_test), 4),
        "recall": round(recall_score(d["y_test"], preds_test), 4),
        "f1": round(f1_score(d["y_test"], preds_test), 4),
        "roc_auc": round(roc_auc_score(d["y_test"], probs_test), 4)
    }
    
    # Log Results
    df_res = pd.DataFrame([metrics])
    csv_path = RESULTS_DIR / "mitigation_adversarial_performance.csv"
    mode = 'a' if csv_path.exists() else 'w'
    header = not csv_path.exists()
    df_res.to_csv(csv_path, mode=mode, header=header, index=False)
    
    # Save Model
    torch.save(model.state_dict(), MODELS_DIR / f"{dataset}_adversarial.pth")
    print(f"[{dataset}] Adversarial Debiasing Complete. AUC={metrics['roc_auc']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    run_adversarial_debiasing(args.dataset)
