#!/usr/bin/env python3
"""
===============================================================================
Script Name   : causal_model.py
Description   : Causal model definition and visualization for the educational
                fairness study. This script constructs a structural causal
                graph (DAG) capturing hypothesized relationships between
                sensitive attributes (e.g., SES, Gender), mediating/proxy
                variables (e.g., Engagement, Prior Education, Assessment),
                and the outcome of interest (Success).

                The DAG is intended to:
                - Make causal assumptions explicit and auditable
                - Support causal/fairness reasoning beyond correlational metrics
                - Provide a publication-ready figure for reports/manuscripts
                - Integrate empirical signals from fairness audits and
                  differential explainability (e.g., SHAP) into the narrative

How to Run   :
                python scripts/causal_model.py

Inputs        :
                None required (model structure is defined in code).

Outputs       :
                figures/
                  - causal_dag_v1.pdf
                  - causal_dag_v1.png

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
- Nodes are categorized as:
    * sensitive  : SES, Gender
    * mediator   : Prior_Education, Engagement, Assessment (proxy pathways)
    * outcome    : Success
- Edges represent causal assumptions grounded in prior literature and
  supported by observed audit patterns (e.g., reliance on Engagement as a proxy).
- The plotted DAG is a conceptual model: it does not estimate causal effects by
  itself; it supports transparent reasoning and guides subsequent analyses.

Dependencies :
- Python >= 3.9
- networkx
- matplotlib

===============================================================================
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

def define_structural_model():
    """
    Defines the Causal DAG based on Literature + Audit Findings.
    Nodes:
      S: Sensitive Attribute (SES, Gender)
      X: Proxies (Engagement, Prior Edu)
      Y: Outcome (Success)
    """
    G = nx.DiGraph()
    
    # 1. Define Nodes
    G.add_node("SES", type="sensitive")
    G.add_node("Gender", type="sensitive")
    G.add_node("Prior_Education", type="mediator")
    G.add_node("Engagement", type="mediator")
    G.add_node("Assessment", type="mediator")
    G.add_node("Success", type="outcome")
    
    # 2. Define Edges (Assumptions validated by Audit)
    
    # Assumption 1: Socioeconomic Status affects Preparation
    G.add_edge("SES", "Prior_Education")
    
    # Assumption 2: SES affects Digital Engagement (Digital Divide)
    # (Our SHAP analysis showed model over-relies on this link)
    G.add_edge("SES", "Engagement")
    
    # Assumption 3: Gender affects Engagement behaviors (Literature)
    G.add_edge("Gender", "Engagement")
    
    # Assumption 4: Prior Education influences Assessment performance
    G.add_edge("Prior_Education", "Assessment")
    
    # Assumption 5: Engagement influences Assessment (Studying leads to scores)
    G.add_edge("Engagement", "Assessment")
    
    # Direct Causal Links to Outcome
    G.add_edge("Assessment", "Success")
    G.add_edge("Engagement", "Success") # Direct path (participation grade)
    
    return G

def plot_dag(G):
    plt.figure(figsize=(10, 6))
    
    # Layout
    pos = {
        "SES": (-1, 1),
        "Gender": (-1, -1),
        "Prior_Education": (0, 1),
        "Engagement": (0, -1),
        "Assessment": (1, 0),
        "Success": (2, 0)
    }
    
    # Colors
    color_map = []
    for node in G.nodes():
        if G.nodes[node]['type'] == 'sensitive':
            color_map.append('#ffcccc') # Red
        elif G.nodes[node]['type'] == 'outcome':
            color_map.append('#ccffcc') # Green
        else:
            color_map.append('#ccccff') # Blue
            
    nx.draw(G, pos, with_labels=True, node_color=color_map, 
            node_size=3000, font_size=10, font_weight="bold", 
            arrows=True, edge_color="gray")
    
    plt.title("Causal DAG for Educational Fairness\n(Validated by Differential SHAP)")
    plt.savefig(FIGURES_DIR / "causal_dag_v1.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "causal_dag_v1.png", bbox_inches='tight') # For preview
    print(f"✅ DAG Visualization saved to {FIGURES_DIR}")

if __name__ == "__main__":
    dag = define_structural_model()
    plot_dag(dag)
