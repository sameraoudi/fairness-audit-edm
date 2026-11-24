# No Silver Bullet: Dataset-Dependent Fairness in Educational AI and a Governance Framework for Responsible Deployment

**Supplementary code and verification artifacts for the manuscript submitted to *Education Sciences* (MDPI).**

Samer Aoudi\*, Hussain Al-Aqrabi, Nur Siyam, Mohammed Saleh, and Daoud Daoud

\* Corresponding author

Archived release: [https://doi.org/10.5281/zenodo.22974238](https://doi.org/10.5281/zenodo.22974238)

---

## Overview

This repository supports a fairness audit of student-success prediction models on three public educational datasets, together with a governance framework that specifies when a fairness-mitigation method may be deployed.

The work has two distinct parts:

1. **Empirical findings.** Audit baseline models for gender, SES, and intersectional disparities. Then compare three mitigation strategies (Reweighing, Threshold Optimization, Adversarial Debiasing) under protocols suited to each dataset.
2. **Governance framework.** Encode five deployment guards as a formal rule, `Deploy(m)`, and check with the Z3 SMT solver that the rule is internally consistent.

**Prediction target.** The positive class is **Pass** (student success), not "at-risk". So the true-positive rate (TPR) measures how well the model **recognises successful students**, and TPR gaps show which groups' success is under-recognised.

---

## Datasets

Raw data are not redistributed. Download them from the original sources and place them in `data/raw/<dataset>/`.

| Dataset | N | Source | What it tests |
|---------|---|--------|---------------|
| OULAD (Open University Learning Analytics Dataset) | 32,593 student–module records | [research.stem.open.ac.uk/ouanalyse/dataset](https://research.stem.open.ac.uk/ouanalyse/dataset/) | Large-scale audit: gender, SES, and gender × SES disparities; fixed-split mitigation comparison |
| UCI Student Performance | 395 students | [archive.ics.uci.edu/dataset/320](https://archive.ics.uci.edu/dataset/320/student+performance) | Design sensitivity: whether method rankings hold under different outcome thresholds, feature sets, and SES definitions |
| xAPI-Edu-Data | 478 unique records (480 raw; 2 exact duplicates removed) | [kaggle.com/datasets/aljarah/xAPI-Edu-Data](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data) | Small-sample robustness: gender EOD across 20 independent splits |

---

## Empirical Findings

### OULAD

- The baseline shows a **gender-driven TPR gap of about 9 percentage points** (p < 0.001): the model recognises successful female students less often than successful male students.
- The **gender × SES interaction is not significant** (p = 0.642). The data do **not** support a "double disadvantage" claim for low-SES female students beyond the additive gender effect.
- The intersectional equal-opportunity difference (IEOD, TPR over gender × SES) is **0.110**.
  - An earlier version of this repository reported **0.1286** as an intersectional EOD. That was wrong. 0.1286 is a **one-sided pairwise TPR gap** between a single reference subgroup and a single target subgroup. It is not an intersectional EOD.
- On the fixed split, **Threshold Optimization reduced disparity relative to Adversarial Debiasing**.

### UCI Student Performance

- **No catastrophic collapse was observed** in any of the **1,440 evaluations**. Earlier versions of this repository claimed "model collapse" on UCI; that claim is withdrawn.
- The main finding is **design sensitivity**. Which mitigation method ranks best changes with:
  - the pass threshold on G3 (≥ 10, ≥ 11, ≥ 12),
  - whether prior-period grades G1/G2 are available as features,
  - the definition of SES.
- Fairness and utility can come apart. In one condition (G3 ≥ 12, no G1/G2), Thresholding achieved **IEOD = 0.253 but recall = 0.223**: fairness looks better because the model recognises very few successful students. The recall guard in the governance framework (below) exists to reject this kind of configuration.

### xAPI-Edu-Data

- Corrected protocol:
  - **2 exact duplicate rows removed** (480 → 478 unique records),
  - **20 independent stratified 70/15/15 holdout splits** (seeds 42–61),
  - **HistGradientBoosting** baseline (not XGBoost, as stated in earlier versions).
- **No mitigation method significantly improved gender EOD** over the baseline.

### Explainability (SHAP)

SHAP analysis shows **attribution differences**: how feature contributions to predictions differ across groups. These are descriptive attributions of model behaviour. They are **not causal mechanisms**, and they do not show why disparities arise in the underlying educational process.

---

## Governance Framework

### Deployment rule

A mitigation method `m` is authorised for deployment only if all five guards hold:

```
Deploy(m) = (ieod_method   <  ieod_baseline)   # G1  fairness improvement over baseline
          ∧ (recall_method ≥  0.70)            # G2  recall guard
          ∧ (collapse_rate ≤  0.10)            # G3  collapse guard
          ∧ (subgroup_min  ≥  50)              # G4  minimum subgroup support
          ∧ (seed_std      ≤  0.05)            # G5  stability across random seeds
```

### Z3 verification (manuscript Table 6)

[`z3/verify_guardrails.py`](z3/verify_guardrails.py) encodes the rule over real-valued variables and checks:

| Property | Query | Expected |
|----------|-------|----------|
| P1–P5 | For each guard G*i*: `Deploy(m) ∧ ¬G`*i* | UNSAT |
| P6 (combined) | `Deploy(m) ∧ (¬G1 ∨ … ∨ ¬G5)` | UNSAT |
| Weakened model | Recall guard **removed**; IEOD = 0.253, baseline IEOD = 0.412, recall = 0.223 | SAT |

The script also reports the mean solver time for the combined query over 10,000 iterations. This time depends on the machine; expect sub-millisecond values.

**What it proves.**
- The encoded rule has no loophole: no combination of metric values can be authorised while violating any guard (P1–P5), or any combination of guards (P6).
- The weakened-model counterexample shows the **recall guard** is necessary. Without it, the rule would authorise the UCI Thresholding configuration (IEOD = 0.253, recall = 0.223), which looks fairer only because recall has collapsed.

**What it does not prove.**
- Z3 checks the **internal consistency of the encoded deployment rules** only.
- It does not verify trained classifiers, datasets, or the empirical fairness estimates.
- It does not show that the chosen thresholds (0.70, 0.10, 50, 0.05) are the right ones for a given institution.

---

## Repository Structure

```text
fairness-audit-edm/
├── configs/fairness_constraints.yaml   # Hyperparameters and random seed
├── scripts/
│   ├── preprocess/                     # preprocess_{oulad,uci,xapi}.py
│   ├── audit_fairness.py               # Subgroup fairness audit
│   ├── train_baselines.py              # Fixed-split baselines
│   ├── train_mitigation_standard.py    # Reweighing + ThresholdOptimizer
│   ├── train_mitigation_adversarial.py # Adversarial debiasing (gradient reversal)
│   ├── generate_final_results.py       # Fixed-split comparison table
│   ├── plot_pareto.py                  # Fairness–recall Pareto figure (OULAD)
│   └── plot_additional_figures.py
├── src/causal_model.py                 # Conceptual DAG figure (illustrative, not causal evidence)
├── z3/
│   ├── verify_guardrails.py            # Z3 verification of Deploy(m) (Table 6)
│   └── requirements.txt
├── requirements.txt
└── LICENSE
```

---

## How to Reproduce

### Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # experiment pipeline
pip install -r z3/requirements.txt       # Z3 verification (z3-solver)
```

### Running the experiments

Download the raw data (see [Datasets](#datasets)), then:

```bash
# Preprocess
python scripts/preprocess/preprocess_oulad.py
python scripts/preprocess/preprocess_uci.py
python scripts/preprocess/preprocess_xapi.py

# Baselines, audit, mitigation (repeat for --dataset uci / xapi)
python scripts/train_baselines.py --dataset oulad
python scripts/audit_fairness.py --dataset oulad
python scripts/train_mitigation_standard.py --dataset oulad
python scripts/train_mitigation_adversarial.py --dataset oulad

# Fixed-split comparison and figures
python scripts/generate_final_results.py
python scripts/plot_pareto.py
```

> **Scope note.** The scripts above implement the fixed-split pipeline (seed 42), which underlies the OULAD fixed-split comparison. Scripts for two analyses are **not yet included** in this repository and will be added in a later update:
> - the corrected xAPI protocol (deduplication, 20 independent splits, HistGradientBoosting),
> - the UCI design-sensitivity sweep (1,440 evaluations).
>
> Until then, the values reported for those analyses come from the manuscript.

### Running the Z3 verification

```bash
python z3/verify_guardrails.py
```

The script prints the Table 6 results and the counterexample admitted by the weakened model. It exits with status 0 only if every result matches its expected value.

---

## Citation

If you use this code, please cite the manuscript:

> Aoudi, S.; Al-Aqrabi, H.; Siyam, N.; Saleh, M.; Daoud, D. No Silver Bullet: Dataset-Dependent Fairness in Educational AI and a Governance Framework for Responsible Deployment. *Education Sciences* 2026, submitted.

```bibtex
@article{aoudi2026nosilverbullet,
  author  = {Aoudi, Samer and Al-Aqrabi, Hussain and Siyam, Nur and Saleh, Mohammed and Daoud, Daoud},
  title   = {No Silver Bullet: Dataset-Dependent Fairness in Educational {AI} and a Governance Framework for Responsible Deployment},
  journal = {Education Sciences},
  year    = {2026},
  note    = {Submitted}
}
```

and the archived software release:

> Aoudi, S.; Al-Aqrabi, H.; Siyam, N.; Saleh, M.; Daoud, D. *No Silver Bullet: Code and Verification Artifacts* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22974238

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
