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
import pandas as pd
import numpy as np
from pathlib import Path

# Config
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

def plot_pareto(dataset="oulad"):
    print(f"Generating Pareto Plot for {dataset}...")

    # Data from Table 2 in the paper
    data = {
        'Method': ['Baseline (XGBoost)', 'Reweighing (Pre)', 'Thresholding (Post)', 'Adversarial (Deep)'],
        'Recall': [0.8535, 0.8565, 0.8929, 0.8340],       # X-axis (Higher is better)
        'EOD':    [0.2464, 0.1885, 0.1406, 0.2201],       # Y-axis (Lower is better)
        'Color':  ['#7f7f7f', '#1f77b4', '#2ca02c', '#d62728'], # Grey, Blue, Green, Red
        'Marker': ['o', 's', '*', 'D']                    # Circle, Square, Star, Diamond
    }
    
    # Custom Offsets to prevent label overlap (x_offset, y_offset)
    # Adjusted specifically for OULAD results layout
    offsets = [
        (10, 5),    # Baseline: Push right & up slightly
        (10, -15),  # Reweighing: Push right & down (away from Baseline)
        (-100, 5),  # Thresholding: Push left (it's far right) & up
        (-10, 15)   # Adversarial: Push left & up (away from Reweighing)
    ]

    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Plot Points
    for i in range(len(data['Method'])):
        plt.scatter(
            data['Recall'][i], 
            data['EOD'][i], 
            color=data['Color'][i], 
            s=250, 
            label=data['Method'][i], 
            marker=data['Marker'][i], 
            edgecolors='black',
            linewidth=1.5,
            zorder=3
        )
        
        # 2. Add Smart Annotations
        plt.annotate(
            data['Method'][i], 
            (data['Recall'][i], data['EOD'][i]),
            xytext=offsets[i], 
            textcoords='offset points',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
            arrowprops=dict(arrowstyle="-", color='black', alpha=0.5)
        )

    # 3. Add "Ideal" Arrow
    # Arrow pointing to Bottom-Right (High Recall, Low Unfairness)
    plt.arrow(0.82, 0.26, 0.08, -0.12, head_width=0.01, head_length=0.015, fc='gold', ec='orange', alpha=0.6, width=0.002)
    plt.text(0.86, 0.20, "Better Performance\n& Fairness", color='orange', fontweight='bold', rotation=-35, ha='center')

    # 4. Formatting
    plt.title(f'Figure 2: Fairness-Accuracy Pareto Frontier ({dataset.upper()})', fontsize=15, pad=20)
    plt.xlabel('Recall (Sensitivity) $\\rightarrow$\n(Higher is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('Equal Opportunity Difference (Unfairness) $\\rightarrow$\n(Lower is Better)', fontsize=12, fontweight='bold')
    
    # Set limits with padding to prevent label clipping
    plt.xlim(0.80, 0.92)
    plt.ylim(0.10, 0.30)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save
    out_path = FIGURES_DIR / f"pareto_{dataset}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Pareto plot saved to {out_path}")

if __name__ == "__main__":
    plot_pareto()
