# Fairness Audit & Mitigation in Educational Data Mining

**Supplementary Code for the Paper:**
*"Beyond the Black Box: Intersectional Fairness Gerrymandering in Educational AI and a Parsimony-First Governance Framework "*

**Target Journal:** AI and Ethics

---

## Overview

This repository contains the official implementation of the **Fairness Audit & Mitigation Framework** proposed in our study. We investigate "Fairness Gerrymandering" in student success prediction models and evaluate three mitigation strategies across diverse educational datasets.

**Key Findings Reproduced by this Code:**

1. **Detection:** Standard XGBoost baselines encode "Digital Habitus," disproportionately penalizing Low-SES female students.
2. **Mitigation effectiveness:** Threshold Optimization (Post-processing) achieves the lowest EOD (0.173) with high recall (0.891) on OULAD, outperforming Adversarial Debiasing.
3. **Model Collapse Warning:** Both Adversarial Debiasing and Threshold Optimization exhibit degenerate behavior ("Model Collapse") on the small, near-separable UCI dataset (EOD = 1.0 for both). Only Reweighing avoids destabilization.

---

## Repository Structure

```text
fairness-audit-edm/
│
├── configs/
│   └── fairness_constraints.yaml       # Hyperparameters and random seed
│
├── data/
│   ├── raw/                            # Raw data (download separately; see below)
│   └── processed/                      # Processed X.csv, y.csv, sensitive.csv
│       ├── oulad/
│       ├── uci/
│       └── xapi/
│
├── figures/                            # Generated plots (Pareto Frontier, DAGs)
│   ├── figure2_pareto_oulad.png        # Fairness–accuracy Pareto frontier (OULAD)
│   └── fig4_model_collapse.png         # UCI adversarial training dynamics
│
├── models/                             # Saved model artifacts (.pkl, .pth)
│   ├── baselines/
│   ├── mitigated_pre/
│   ├── mitigated_post/
│   ├── mitigated_adversarial/
│   └── mitigated_adversarial_det/      # Deterministically retrained adversarial models
│
├── results/                            # Metrics and audit outputs
│   ├── baseline_performance.csv
│   ├── final_paper_results.csv         # Table 5 (deterministic, seed=42)
│   └── uci_adversarial_epoch_metrics.csv  # Per-epoch dynamics for Fig 4
│
├── scripts/                            # Main execution pipeline
│   ├── preprocess/
│   │   ├── preprocess_oulad.py
│   │   ├── preprocess_uci.py
│   │   └── preprocess_xapi.py
│   ├── audit_fairness.py               # Phase 1: Detect bias
│   ├── train_baselines.py              # Phase 2: Train XGBoost baselines
│   ├── train_mitigation_standard.py    # Phase 3a: Reweighing + ThresholdOptimizer
│   ├── train_mitigation_adversarial.py # Phase 3b: Adversarial FairNet (GRL)
│   ├── generate_final_results.py       # Phase 4: Generate Table 5 (final_paper_results.csv)
│   └── plot_pareto.py                  # Phase 5: Generate Pareto frontier figure
│
├── src/
│   └── causal_model.py                 # Structural Causal Model (DAG) visualization
│
├── splits/                             # Fixed train/val/test index arrays (.npy)
│   ├── oulad/
│   ├── uci/
│   └── xapi/
│
├── requirements.txt
└── README.md
```

---

## 🔒 Reproducibility

All experiments use fixed random seeds (`seed = 42`) throughout, including:
- Model training (XGBoost, adversarial network)
- Data splitting (stratified by Y × Gender × SES)
- ThresholdOptimizer predictions (`random_state=42`)
- Adversarial network initialization (os, random, numpy, torch)

All reported results are fully deterministic and reproducible from this codebase.

---

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

The interactive runner guides you through environment setup, data preparation, and all experimental phases. It detects which steps have already been completed and lets you run phases individually or sequentially.

For advanced usage, individual scripts can still be run directly — see [Reproducing the Experiments](#reproducing-the-experiments) below.

---

## Usage

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

**Note:** Raw data are not included due to licensing and size constraints. Download from the sources below and place in `data/raw/`.

| Dataset | Source |
|---------|--------|
| OULAD | [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset) |
| xAPI-Edu-Data | [UCI ML Repository — Higher Education Students](https://archive.ics.uci.edu/ml/datasets/Higher+Education+Students+Performance+Evaluation+Dataset) |
| UCI Student Performance | [UCI ML Repository — Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance) |

Structure the processed data as:

```text
data/processed/
├── oulad/   { X.csv, y.csv, sensitive.csv }
├── uci/     { X.csv, y.csv, sensitive.csv }
└── xapi/    { X.csv, y.csv, sensitive.csv }
```

---

## Reproducing the Experiments

### Phase 1: Bias Detection

Audit the XGBoost baselines for intersectional fairness violations.

```bash
python scripts/audit_fairness.py --dataset oulad
python scripts/audit_fairness.py --dataset uci
python scripts/audit_fairness.py --dataset xapi
```

### Phase 2: Causal Diagnosis

Generate the Structural Causal Model (DAG) illustrating the "Digital Habitus" mechanism.

```bash
python src/causal_model.py
# Output: figures/causal_dag_v1.png
```

### Phase 3: Mitigation

Train all three mitigation strategies across all three datasets.

**A. Standard Mitigations (Reweighing + ThresholdOptimizer)**

```bash
python scripts/train_mitigation_standard.py --dataset oulad
python scripts/train_mitigation_standard.py --dataset uci
python scripts/train_mitigation_standard.py --dataset xapi
```

**B. Adversarial Debiasing (FairNet with Gradient Reversal Layer)**

```bash
python scripts/train_mitigation_adversarial.py --dataset oulad
python scripts/train_mitigation_adversarial.py --dataset uci
python scripts/train_mitigation_adversarial.py --dataset xapi
```

### Phase 4: Evaluation & Reporting

Generate Table 5 (Comparative Performance of Mitigation Strategies).

```bash
python scripts/generate_final_results.py
# Output: results/final_paper_results.csv
```

### Phase 5: Figures

Generate the Fairness–Accuracy Pareto Frontier (Figure 3).

```bash
python scripts/plot_pareto.py
# Output: figures/figure2_pareto_oulad.png
```

---

## Results Summary (Table 5)

| Dataset | Method                 | Accuracy | Recall | EOD/IEOD | Verdict      |
|---------|------------------------|----------|--------|----------|--------------|
| OULAD   | Baseline (XGBoost)     | 0.829    | 0.853  | 0.246    | Biased       |
| OULAD   | Reweighing (Pre)       | 0.831    | 0.857  | 0.189    | Improved     |
| OULAD   | Thresholding (Post)    | 0.798    | 0.891  | 0.173    | Best         |
| OULAD   | Adversarial (Deep)     | 0.817    | 0.839  | 0.201    | Marginal     |
| UCI     | Baseline (XGBoost)     | 0.950    | 0.950  | 0.250    | Control      |
| UCI     | Reweighing (Pre)       | 0.950    | 0.925  | 0.250    | No change    |
| UCI     | Thresholding (Post)    | 0.783    | 0.800  | 1.000    | Collapsed    |
| UCI     | Adversarial (Deep)     | 0.867    | 0.875  | 1.000    | Collapsed    |
| xAPI    | Baseline (XGBoost)     | 0.764    | 0.811  | 0.438    | Biased       |
| xAPI    | Reweighing (Pre)       | 0.736    | 0.717  | 0.250    | Best         |
| xAPI    | Thresholding (Post)    | 0.833    | 0.887  | 0.375    | Improved     |
| xAPI    | Adversarial (Deep)     | 0.792    | 0.849  | 0.438    | No change    |

EOD/IEOD = Fairlearn `equalized_odds_difference`; sensitive attribute = `ses_quintile` (OULAD/UCI) or `gender` (xAPI).

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Citation

Please cite the software using the generated DOI:

> Aoudi, S. (2025). *Fairness Audit & Mitigation Framework for EDM (v1.0.1)* [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.17999876
