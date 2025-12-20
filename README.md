# Beyond the Black Box: Fairness Audit & Mitigation in Educational Data Mining

**Supplementary Code for the Paper:**
*"Beyond the Black Box: A Comparative Audit of Intersectional Fairness and Adversarial Mitigation in Educational Data Mining"*

**Author:** Dr. Samer Aoudi
**Target Journal:** Computers & Education: Artificial Intelligence (CAEAI)


---

## 📌 Overview
This repository contains the official implementation of the **Fairness Audit & Mitigation Framework** proposed in our study. We investigate "Fairness Gerrymandering" in student success prediction models and evaluate three mitigation strategies across diverse educational datasets.

**Key Findings Reproduced by this Code:**
1.  **Detection:** Standard XGBoost baselines encode "Digital Habitus," disproportionately penalizing Low-SES female students.
2.  **Refutation of Hypothesis 2:** Contrary to common assumptions, **Threshold Optimization (Post-processing)** significantly outperforms **Adversarial Debiasing (Deep Learning)** on large tabular educational data ($N=32k$).
3.  **Generalizability Warning:** Adversarial methods exhibit degenerate behavior ("Model Collapse") on small or high-accuracy datasets (UCI).

---

## 📂 Repository Structure

```text
fairness-audit-edm/
│
├── configs/
│   └── fairness_constraints.yaml       # Hyperparameters for fairness auditing
│
├── data/
│   ├── raw/                            # (Excluded from git)
│   └── processed/                      # Place your processed X.csv, y.csv here
│       ├── oulad/
│       ├── uci/
│       └── xapi/
│
├── figures/                            # Generated plots (Pareto Frontier, DAGs)
│
├── models/                             # Saved model artifacts (.pkl, .pth)
│   ├── baselines/
│   ├── mitigated_pre/
│   ├── mitigated_post/
│   └── mitigated_adversarial/
│
├── results/                            # Audit logs and performance metrics
│   ├── fairness_audit_oulad.json
│   ├── mitigation_standard_performance.csv
│   └── final_paper_results.csv
│
├── scripts/                            # Main execution pipelines
│   ├── audit_fairness.py               # Step 1: Detect Bias (Hypothesis 1)
│   ├── train_mitigation_standard.py    # Step 2: Train Reweighing & Thresholding
│   ├── train_mitigation_adversarial.py # Step 3: Train Adversarial Network
│   ├── generate_paper_results.py       # Step 4: Generate Table 2
│   └── plot_pareto.py                  # Step 5: Generate Figure 2
│
├── src/                                # Helper modules
│   ├── __init__.py
│   └── causal_model.py                 # Structural Causal Model (DAG) visualization
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file

## 🚀 Usage

### 1. Installation
```bash
pip install -r requirements.txt

### 2. Data Preparation
**Note:** This repository contains the processing logic but **not the raw data** due to licensing and size constraints.

1.  **Download the Raw Data:**
    * [OULAD (Open University Learning Analytics Dataset)](https://analyse.kmi.open.ac.uk/open_dataset)
    * [xAPI-Edu-Data](https://archive.ics.uci.edu/ml/datasets/Higher+Education+Students+Performance+Evaluation+Dataset)
    * [UCI Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

2.  **Preprocess & Structure:**
    * Process the data into `X.csv` (features), `y.csv` (target labels), and `sensitive.csv` (demographics).
    * Ensure the directory structure matches the expectation of the scripts:
    ```text
    data/processed/
    ├── oulad/
    │   ├── X.csv
    │   ├── y.csv
    │   └── sensitive.csv
    ├── xapi/
    │   └── ...
    └── uci/
        └── ...
    ```

---

## 🔬 Reproducing the Experiments

The reproduction pipeline consists of four sequential phases, mirroring the methodology described in the paper.

### Phase 1: Detection (Hypothesis 1)
Audit the baseline XGBoost models to detect "Intersectional Masking" (e.g., bias against Low-SES Females).

```bash
# Run fairness audit on the primary dataset
python scripts/audit_fairness.py --dataset oulad

# Output: results/fairness_audit_oulad.json
# (Contains SHAP values, global metrics, and intersectional EOD analysis)

Phase 2: Causal DiagnosisGenerate the Structural Causal Model (DAG) to visualize the "Digital Habitus" mechanism (SES $\rightarrow$ Engagement).
# Generate Figure 1 from the paper
python src/causal_model.py

# Output: figures/causal_dag_v1.png

Phase 3: Mitigation (Hypothesis 2 & 3)
Train and evaluate the three competing mitigation strategies across all datasets.

A. Standard Mitigations (Reweighing & Thresholding) Trains the Reweighed (Pre-processing) model and fits the Threshold Optimizer (Post-processing).
python scripts/train_mitigation_standard.py --dataset oulad
python scripts/train_mitigation_standard.py --dataset xapi
python scripts/train_mitigation_standard.py --dataset uci

B. Adversarial Debiasing (Deep Learning) Trains the Custom Neural Network with Gradient Reversal (In-processing).
python scripts/train_mitigation_adversarial.py --dataset oulad
python scripts/train_mitigation_adversarial.py --dataset xapi
python scripts/train_mitigation_adversarial.py --dataset uci

Phase 4: Evaluation & Reporting
Compile the results into the final tables and figures used in the manuscript.

Generate Table 2 (Comparative Metrics) Aggregates performance (Accuracy, Recall) and fairness (EOD) metrics from all models into a single CSV.
python scripts/generate_paper_results.py


# Output: results/final_paper_results.csv

Generate Figure 2 (Pareto Frontier) Visualizes the trade-off between Recall and Unfairness to identify the Pareto-optimal strategy.
python scripts/plot_pareto.py

# Output: figures/pareto_oulad.png

## 📊 Results Summary

| Dataset | Method                   | Verdict                                       |
|---------|--------------------------|-----------------------------------------------|
| OULAD   | Threshold Optimization   | Best Result (High Recall, Low Bias)           |
| OULAD   | Adversarial Debiasing    | Underperformed (Hypothesis 2 Rejected)        |
| xAPI    | Reweighing               | Best for small/proxy-heavy data               |
| UCI     | Adversarial Debiasing    | Model Collapse (EOD = 1.0)                    |

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
