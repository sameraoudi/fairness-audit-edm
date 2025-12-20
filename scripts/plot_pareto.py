#!/usr/bin/env python3
"""
===============================================================================
Script Name   : plot_pareto.py
Description   : Generates a fairness–utility Pareto plot illustrating the
                trade-off between predictive performance and algorithmic
                fairness for the OULAD dataset.

                Specifically, the plot visualizes:
                - Utility  : Recall (Sensitivity) — higher is better
                - Unfairness: Equal Opportunity Difference (EOD) — lower is better

                Each point corresponds to a modeling or mitigation strategy:
                - Baseline (XGBoost)
                - Reweighing (Pre-processing mitigation)
                - Thresholding (Post-processing mitigation)
                - Adversarial Debiasing (Deep learning)

                The resulting figure is designed to be publication-ready and
                is referenced as Figure 2 in the manuscript, highlighting the
                Pareto frontier between accuracy and fairness.

How to Run   :
                python scripts/plot_pareto.py

Inputs        :
                - Manually specified metrics (derived from final_paper_results.csv)
                  for the OULAD dataset.

Outputs       :
                - figure_2_pareto.png   (high-resolution, 300 DPI)

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
- The bottom-right region of the plot represents the preferred trade-off
  (high recall, low unfairness).
- An “Ideal State” (Recall = 1.0, EOD = 0.0) is shown for conceptual reference.
- Axis limits are chosen to emphasize meaningful differences between methods.
- The plot uses a consistent visual encoding to support interpretability
  in an academic publication.

Dependencies :
- Python >= 3.9
- matplotlib

===============================================================================
"""


import matplotlib.pyplot as plt

# Data from final_paper_results.csv (OULAD)
methods = ['Baseline (XGBoost)', 'Reweighing (Pre)', 'Thresholding (Post)', 'Adversarial (Deep)']
recall = [0.8535, 0.8565, 0.8929, 0.8340]  # Utility (Higher is Better)
eod = [0.2464, 0.1885, 0.1406, 0.2201]    # Unfairness (Lower is Better)

# Plot Setup
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot Points
colors = ['gray', 'blue', 'green', 'red']
markers = ['o', 's', '*', 'D']

for i in range(len(methods)):
    plt.scatter(recall[i], eod[i], color=colors[i], s=200, label=methods[i], marker=markers[i], edgecolors='k')

# Arrows indicating "Better"
plt.annotate('Better (High Recall)', xy=(0.88, 0.26), xytext=(0.90, 0.26),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('Better (Low Bias)', xy=(0.82, 0.15), xytext=(0.82, 0.10),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Ideal Point Annotation
plt.scatter(1.0, 0.0, color='gold', s=300, marker='*', label='Ideal State')
plt.text(0.98, 0.02, "Ideal", fontsize=12)

# Labels and Title
plt.xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
plt.ylabel('Equal Opportunity Difference (Unfairness)', fontsize=12, fontweight='bold')
plt.title('Figure 2: Fairness-Accuracy Pareto Frontier (OULAD Dataset)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)

# Invert Y axis so "Lower" is visually "Better" (Optional, but standard Pareto plots usually maximize both)
# Here we keep standard: Bottom-Right is the goal (Low EOD, High Recall)
plt.xlim(0.80, 0.91)
plt.ylim(0.0, 0.30)

plt.tight_layout()
plt.savefig('figure_2_pareto.png', dpi=300)
plt.show()
