#!/usr/bin/env python3
"""
run.py
======
Interactive pipeline runner for:
  "Beyond the Black Box: A Comparative Audit of Intersectional Fairness
   and Adversarial Mitigation in Educational Data Mining"

Usage:
    python run.py

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119

All phases use seed=42 for full reproducibility.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. GLOBAL SEEDS — must be first, before any other imports
# ─────────────────────────────────────────────────────────────────────────────
import os, random
SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
np.random.seed(SEED)

try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 1. STDLIB
# ─────────────────────────────────────────────────────────────────────────────
import sys
import time
import subprocess
import traceback
import importlib
from pathlib import Path
from datetime import datetime

# Anchor working directory to repo root so all relative paths resolve correctly
REPO_ROOT = Path(__file__).parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))   # enables `from scripts import …`

# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTIONAL RICH
# ─────────────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
VERSION  = "1.0"
DATASETS = ["oulad", "uci", "xapi"]
LOG_PATH = REPO_ROOT / "results" / "pipeline_log.txt"

# At least one of these files must exist in data/raw/<ds>/ to count as "raw available"
_RAW_INDICATORS = {
    "oulad": ["studentInfo.csv"],
    "uci":   ["student-mat.csv", "student-por.csv"],
    "xapi":  ["xAPI-Edu-Data.csv", "xapi.csv", "xAPI_Edu_Data.csv"],
}

_DOWNLOAD_URLS = {
    "oulad": "https://analyse.kmi.open.ac.uk/open_dataset",
    "uci":   "https://archive.ics.uci.edu/ml/datasets/Student+Performance",
    "xapi":  "https://archive.ics.uci.edu/ml/datasets/Higher+Education+Students+Performance+Evaluation+Dataset",
}

# requirements.txt package → importable module name
_PKG_IMPORT_MAP = {
    "pandas":       "pandas",
    "numpy":        "numpy",
    "scikit-learn": "sklearn",
    "xgboost":      "xgboost",
    "pyyaml":       "yaml",
    "networkx":     "networkx",
    "matplotlib":   "matplotlib",
    "seaborn":      "seaborn",
    "shap":         "shap",
    "fairlearn":    "fairlearn",
    "torch":        "torch",
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as fh:
        fh.write(f"[{ts}] {msg}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_W = 64   # line width

def _hr(c="─"): print(c * _W)

def _section(title: str) -> None:
    print()
    if RICH:
        console.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        _hr("═")
        print(f"  {title}")
        _hr("═")
    print()

def _ok(msg: str):   print(f"  ✓  {msg}")
def _warn(msg: str): print(f"  ⚠  {msg}")
def _err(msg: str):  print(f"  ✗  {msg}")
def _info(msg: str): print(f"     {msg}")

def _ask(prompt: str, default: str = "") -> str:
    dflt = f" [{default}]" if default else ""
    try:
        ans = input(f"  {prompt}{dflt}: ").strip()
        return ans if ans else default
    except (EOFError, KeyboardInterrupt):
        return default

def _confirm(prompt: str, default: bool = True) -> bool:
    tag = "Y/n" if default else "y/N"
    ans = _ask(f"{prompt} ({tag})", "y" if default else "n").lower()
    return ans in ("y", "yes", "") if default else ans in ("y", "yes")

def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}m"

# ─────────────────────────────────────────────────────────────────────────────
# 6. STATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_state() -> dict:
    """Scan the filesystem and return a dict describing what has been completed."""
    s: dict = {
        "raw_available":      {},
        "preprocessed":       {},
        "baselines_trained":  {},
        "splits_exist":       {},
        "audit_done":         {},
        "mitigations_trained":{},
        "results_generated":  False,
        "figures":            {},
    }
    for ds in DATASETS:
        raw_dir  = REPO_ROOT / "data" / "raw" / ds
        proc_dir = REPO_ROOT / "data" / "processed" / ds

        s["raw_available"][ds] = any((raw_dir / f).exists() for f in _RAW_INDICATORS[ds])
        s["preprocessed"][ds]  = all((proc_dir / f).exists() for f in ["X.csv", "y.csv", "sensitive.csv"])

        bl_dir = REPO_ROOT / "models" / "baselines" / ds
        s["baselines_trained"][ds] = (bl_dir / "XGBoost.pkl").exists()
        s["splits_exist"][ds]      = all(
            (REPO_ROOT / "splits" / ds / f).exists()
            for f in ["train_idx.npy", "val_idx.npy", "test_idx.npy"]
        )
        s["audit_done"][ds] = (REPO_ROOT / "results" / f"fairness_audit_{ds}.json").exists()

        adv_ok = (
            (REPO_ROOT / "models" / "mitigated_adversarial_det" / f"{ds}_adversarial_det.pth").exists()
            or (REPO_ROOT / "models" / "mitigated_adversarial"  / f"{ds}_adversarial.pth").exists()
        )
        s["mitigations_trained"][ds] = all([
            (REPO_ROOT / "models" / "mitigated_pre"  / f"{ds}_reweighed.pkl").exists(),
            (REPO_ROOT / "models" / "mitigated_post" / f"{ds}_threshold.pkl").exists(),
            adv_ok,
        ])

    s["results_generated"] = (REPO_ROOT / "results" / "final_paper_results.csv").exists()

    for fig in [
        "figure2_pareto_oulad.png",
        "fig4_model_collapse.png",
        "causal_dag_v1.png",
        "fig3_intersectional_heatmap.png",
        "fig4_differential_shap.png",
        "fig5_model_collapse.png",
    ]:
        s["figures"][fig] = (REPO_ROOT / "figures" / fig).exists()

    return s

# ─────────────────────────────────────────────────────────────────────────────
# 7. BANNER + MENU
# ─────────────────────────────────────────────────────────────────────────────
def print_banner() -> None:
    lines = [
        "Fairness Audit & Mitigation in Educational Data Mining",
        "",
        '"Beyond the Black Box: A Comparative Audit of Intersectional',
        ' Fairness and Adversarial Mitigation in EDM"',
        "",
        f"Pipeline Runner v{VERSION}  ·  seed={SEED}  ·  Python {sys.version.split()[0]}",
        "Target journal: AI and Ethics",
    ]
    if RICH:
        console.print(Panel("\n".join(lines), style="bold blue", expand=False))
    else:
        _hr("═")
        for ln in lines:
            print(f"  {ln}")
        _hr("═")
    print()


def _phase_tick(s: dict, key: str) -> str:
    """Return ✓ if all datasets have this key True, ◑ if some, · if none."""
    vals = list(s.get(key, {}).values())
    if not vals:
        return "·"
    n = sum(bool(v) for v in vals)
    if n == len(vals): return "✓"
    if n > 0:          return "◑"
    return "·"


def print_menu(state: dict) -> None:
    _hr("═")
    print("  Fairness Audit & Mitigation Pipeline")
    print(f"  seed={SEED}  ·  fully reproducible end-to-end")
    _hr()
    print()
    print("  Pipeline status  (✓ done  ◑ partial  · not started)")
    print(f"    [1] Preprocessing    {_phase_tick(state,'preprocessed')}")
    print(f"    [2] Detection        {_phase_tick(state,'baselines_trained')}")
    print(f"    [3] Diagnosis        {'✓' if state['figures'].get('causal_dag_v1.png') else '·'}")
    print(f"    [4] Mitigation       {_phase_tick(state,'mitigations_trained')}")
    print(f"    [5] Evaluation       {'✓' if state['results_generated'] else '·'}")
    print(f"    [6] Figures          {'✓' if state['figures'].get('figure2_pareto_oulad.png') else '·'}")
    print()
    _hr()
    print()
    print("  Phases:")
    print("    [0]  Pre-flight    dependency check + data availability")
    print("    [1]  Preprocessing raw → processed features (X, y, sensitive)")
    print("    [2]  Detection     train baselines + intersectional fairness audit")
    print("    [3]  Diagnosis     generate causal DAG visualization")
    print("    [4]  Mitigation    reweighing / thresholding / adversarial FairNet")
    print("    [5]  Evaluation    generate Table 5 (final_paper_results.csv)")
    print("    [6]  Figures       Pareto, heatmap, SHAP, collapse dynamics")
    print("    [A]  Run all       execute full pipeline 1 → 6")
    print("    [Q]  Quit")
    print()
    _hr()

# ─────────────────────────────────────────────────────────────────────────────
# 8. PHASE 0: PRE-FLIGHT
# ─────────────────────────────────────────────────────────────────────────────
def _check_pkg(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _data_status(state: dict) -> None:
    print()
    print("  Data availability:")
    _hr()
    print(f"  {'Dataset':<10} {'Raw':<13} {'Processed':<13} Status")
    _hr()
    for ds in DATASETS:
        raw  = "✓ Found   " if state["raw_available"][ds]  else "✗ Missing "
        proc = "✓ Ready   " if state["preprocessed"][ds]   else "✗ Missing "
        if state["preprocessed"][ds]:
            status = "Ready"
        elif state["raw_available"][ds]:
            status = "Needs preprocessing  →  run [1]"
        else:
            status = "Download required"
        print(f"  {ds.upper():<10} {raw:<13} {proc:<13} {status}")
    _hr()


def phase_preflight(state: dict) -> dict:
    _section("Phase 0 — Pre-flight Checks")
    _log("Phase 0: started")

    # Python version
    if sys.version_info < (3, 8):
        _err(f"Python >= 3.8 required (got {sys.version.split()[0]})")
        sys.exit(1)
    _ok(f"Python {sys.version.split()[0]}")

    # Dependencies
    print()
    print("  Checking dependencies...")
    missing = []
    for pkg, imp in _PKG_IMPORT_MAP.items():
        if _check_pkg(imp):
            _ok(f"{pkg}")
        else:
            _err(f"{pkg}  ← not found")
            missing.append(pkg)

    if missing:
        print()
        _warn(f"{len(missing)} package(s) missing: {', '.join(missing)}")
        if _confirm("Install missing dependencies now? (pip install -r requirements.txt)"):
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=str(REPO_ROOT),
            )
            if result.returncode == 0:
                _ok("Installation complete — re-checking...")
                _log("Dependencies installed via pip")
                # re-check
                still = [p for p, i in _PKG_IMPORT_MAP.items() if not _check_pkg(i)]
                if still:
                    _warn(f"Still missing: {', '.join(still)}")
                else:
                    _ok("All dependencies verified")
            else:
                _err("Installation failed — check output above")
        else:
            _warn("Continuing without installing — some phases may fail")
    else:
        print()
        _ok("All required dependencies are installed")

    # Data status
    state = detect_state()
    _data_status(state)

    # Prompt to download if anything is missing
    missing_ds = [ds for ds in DATASETS
                  if not state["raw_available"][ds] and not state["preprocessed"][ds]]
    if missing_ds:
        print()
        _warn("The following datasets need to be downloaded:")
        for ds in missing_ds:
            raw_dir = REPO_ROOT / "data" / "raw" / ds
            print(f"\n  • {ds.upper()}:")
            print(f"    URL:  {_DOWNLOAD_URLS[ds]}")
            print(f"    Place files in: data/raw/{ds}/")
        print()
        resp = _ask("Press Enter to re-check, or [s] to skip to menu", "").lower()
        if resp != "s":
            state = detect_state()
            _data_status(state)

    _log("Phase 0: complete")
    return state

# ─────────────────────────────────────────────────────────────────────────────
# 9. DATASET SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
def _select_datasets(available: list, label: str = "run") -> list:
    if not available:
        _warn("No datasets available for this phase.")
        return []
    print(f"  Datasets ready to {label}: {', '.join(d.upper() for d in available)}")
    resp = _ask("Select datasets (all / oulad,uci,xapi)", "all").lower().strip()
    if resp == "all":
        return list(available)
    chosen = [r.strip() for r in resp.replace(" ", "").split(",") if r.strip() in DATASETS]
    valid  = [ds for ds in chosen if ds in available]
    if not valid:
        _warn("None of the selected datasets are available — using all.")
        return list(available)
    return valid

# ─────────────────────────────────────────────────────────────────────────────
# 10. PHASE 1: PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def phase_preprocessing(state: dict) -> dict:
    _section("Phase 1 — Preprocessing")
    _log("Phase 1: started")
    t0 = time.time()

    print("  Transforms raw CSV files into processed feature matrices.")
    print("  Outputs: data/processed/{dataset}/X.csv  y.csv  sensitive.csv")
    print()

    candidates = [ds for ds in DATASETS if state["raw_available"][ds]]
    if not candidates:
        _err("No raw data found. Download datasets first (Phase 0 shows URLs).")
        return state

    already = [ds for ds in candidates if state["preprocessed"][ds]]
    if already:
        _info(f"Already preprocessed: {', '.join(d.upper() for d in already)}")
        if not _confirm("Re-run preprocessing for already-done datasets?", False):
            candidates = [ds for ds in candidates if not state["preprocessed"][ds]]

    if not candidates:
        _ok("Nothing left to preprocess.")
        return detect_state()

    chosen = _select_datasets(candidates, "preprocess")
    if not chosen:
        return state

    _script_map = {
        "oulad": "scripts/preprocess/preprocess_oulad.py",
        "uci":   "scripts/preprocess/preprocess_uci.py",
        "xapi":  "scripts/preprocess/preprocess_xapi.py",
    }

    for ds in chosen:
        print(f"  [{ds.upper()}] Preprocessing...")
        result = subprocess.run(
            [sys.executable, _script_map[ds]],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            _ok(f"{ds.upper()} done")
            _log(f"Phase 1: {ds} OK")
            # Shape summary
            try:
                import pandas as pd
                proc = REPO_ROOT / "data" / "processed" / ds
                X = pd.read_csv(proc / "X.csv")
                y = pd.read_csv(proc / "y.csv")
                pos = int(y.values.sum())
                _info(f"  X: {X.shape[0]:,} rows × {X.shape[1]} features")
                _info(f"  y: {pos:,} positive ({100*pos/len(y):.1f}%) / {len(y)-pos:,} negative")
            except Exception:
                pass
        else:
            _err(f"{ds.upper()} failed")
            print(result.stderr[-800:] if result.stderr else "(no stderr)")
            _log(f"Phase 1: {ds} FAILED")
        print()

    _info(f"Phase 1 complete in {_elapsed(t0)}")
    _log(f"Phase 1: complete ({_elapsed(t0)})")
    return detect_state()

# ─────────────────────────────────────────────────────────────────────────────
# 11. PHASE 2: DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def phase_detection(state: dict) -> dict:
    _section("Phase 2 — Detection (Hypothesis 1)")
    _log("Phase 2: started")
    t0 = time.time()

    print("  Trains XGBoost / LR / RF baselines with stratified 70/15/15 splits.")
    print("  Then runs an intersectional fairness audit on the XGBoost model,")
    print("  computing EOD across Gender × SES subgroups.")
    print()

    ready = [ds for ds in DATASETS if state["preprocessed"][ds]]
    if not ready:
        _err("No preprocessed data. Run Phase 1 first.")
        return state

    chosen = _select_datasets(ready, "run")
    if not chosen:
        return state

    # Lazy imports — split so a missing audit dependency doesn't block training
    try:
        import scripts.train_baselines as tb_mod
    except Exception as exc:
        _err(f"Cannot import train_baselines: {exc}")
        traceback.print_exc()
        return state

    af_mod = None
    try:
        import scripts.audit_fairness as af_mod
    except ImportError as exc:
        _warn(f"audit_fairness import failed ({exc})")
        _warn("Baseline training will run but fairness audit will be skipped.")
        _warn("Fix: pip install shap  (or run Phase 0 to install all dependencies)")

    for ds in chosen:
        print(f"  ── {ds.upper()} ──")

        # Baselines
        if state["baselines_trained"][ds] and state["splits_exist"][ds]:
            if not _confirm(f"  Baselines for {ds.upper()} already exist. Re-train?", False):
                _info("Skipping baseline training")
            else:
                _run_baselines(tb_mod, ds)
        else:
            _run_baselines(tb_mod, ds)

        # Fairness audit
        if af_mod is None:
            _warn("Fairness audit skipped (shap not installed — run Phase 0 to fix)")
            print()
            continue

        if state["audit_done"][ds]:
            if not _confirm(f"  Audit for {ds.upper()} already exists. Re-run?", False):
                _info("Skipping audit")
                print()
                continue

        print(f"  Running fairness audit for {ds.upper()}...")
        _log(f"Phase 2: audit {ds}")
        try:
            af_mod.main(ds)
            _ok(f"Audit → results/fairness_audit_{ds}.json")
            _log(f"Phase 2: {ds} audit OK")
            _print_audit_summary(ds)
        except Exception as exc:
            _err(f"Audit failed: {exc}")
            _log(f"Phase 2: {ds} audit FAILED: {exc}")
            traceback.print_exc()
        print()

    _info(f"Phase 2 complete in {_elapsed(t0)}")
    _log(f"Phase 2: complete ({_elapsed(t0)})")
    return detect_state()


def _run_baselines(tb_mod, ds: str) -> None:
    _log(f"Phase 2: training baselines for {ds}")
    print(f"  Training baselines for {ds.upper()}...")
    try:
        X, y, sens = tb_mod.load_processed_data(ds)
        tb_mod.train_and_evaluate(ds, X, y, sens)
        _ok(f"Models → models/baselines/{ds}/")
        _log(f"Phase 2: {ds} baselines OK")
        # Print XGBoost metrics from CSV
        try:
            import pandas as pd
            perf = pd.read_csv(REPO_ROOT / "results" / "baseline_performance.csv")
            xgb = perf[(perf["dataset"] == ds) & (perf["model"] == "XGBoost")].iloc[-1]
            _info(f"  XGBoost  acc={xgb['accuracy']:.4f}  "
                  f"recall={xgb['recall']:.4f}  AUC={xgb['roc_auc']:.4f}")
        except Exception:
            pass
    except Exception as exc:
        _err(f"Baseline training failed: {exc}")
        _log(f"Phase 2: {ds} baselines FAILED: {exc}")
        traceback.print_exc()


def _print_audit_summary(ds: str) -> None:
    import json
    path = REPO_ROOT / "results" / f"fairness_audit_{ds}.json"
    if not path.exists():
        return
    try:
        with open(path) as f:
            audit = json.load(f)
        # Walk audit dict looking for eod / dp_diff values
        shown = 0
        for attr_key, attr_data in audit.items():
            if not isinstance(attr_data, dict):
                continue
            eod = attr_data.get("eod") or attr_data.get("EO_Diff")
            if eod is None:
                # look one level deeper
                for sub_key, sub_val in attr_data.items():
                    if isinstance(sub_val, dict):
                        eod = sub_val.get("eod") or sub_val.get("EO_Diff")
                        if eod is not None:
                            break
            if eod is not None:
                flag = "  ⚠ BIAS DETECTED" if abs(float(eod)) > 0.10 else ""
                _info(f"  {attr_key}: EOD={float(eod):.4f}{flag}")
                shown += 1
            if shown >= 4:
                break
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 12. PHASE 3: DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
def phase_diagnosis(state: dict) -> dict:
    _section("Phase 3 — Causal Diagnosis")
    _log("Phase 3: started")
    t0 = time.time()

    print("  Generates the Structural Causal Model (DAG) illustrating the")
    print("  'Digital Habitus' mechanism: SES → Engagement → Success.")
    print("  Note: theory-driven graph, not empirically estimated.")
    print()

    out = REPO_ROOT / "figures" / "causal_dag_v1.png"
    if out.exists() and not _confirm("DAG already exists. Regenerate?", False):
        _ok(f"Existing: figures/causal_dag_v1.png")
        return state

    try:
        import matplotlib
        matplotlib.use("Agg")
        # Use importlib to load from src/ (no __init__.py needed)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "causal_model", REPO_ROOT / "src" / "causal_model.py"
        )
        cm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cm_mod)

        G = cm_mod.define_structural_model()
        cm_mod.plot_dag(G)
        _ok("DAG → figures/causal_dag_v1.png  (and .pdf)")
        _log("Phase 3: DAG OK")
    except Exception as exc:
        _err(f"DAG generation failed: {exc}")
        _log(f"Phase 3: FAILED: {exc}")
        traceback.print_exc()

    _info(f"Phase 3 complete in {_elapsed(t0)}")
    return detect_state()

# ─────────────────────────────────────────────────────────────────────────────
# 13. PHASE 4: MITIGATION
# ─────────────────────────────────────────────────────────────────────────────
def phase_mitigation(state: dict) -> dict:
    _section("Phase 4 — Mitigation (Hypotheses 2 & 3)")
    _log("Phase 4: started")
    t0 = time.time()

    print("  Trains three competing fairness mitigation strategies:")
    print("  (Pre)  Reweighing   — inverse-probability sample weights")
    print("  (Post) Thresholding — group-specific decision thresholds (seed=42)")
    print("  (In)   FairNet      — gradient reversal adversarial network (seed=42)")
    print()

    ready = [ds for ds in DATASETS
             if state["baselines_trained"][ds] and state["splits_exist"][ds]]
    if not ready:
        _err("No trained baselines found. Run Phase 2 (Detection) first.")
        return state

    chosen = _select_datasets(ready, "mitigate")
    if not chosen:
        return state

    try:
        import scripts.train_mitigation_standard    as std_mod
        import scripts.train_mitigation_adversarial as adv_mod
    except Exception as exc:
        _err(f"Import error: {exc}")
        traceback.print_exc()
        return state

    for ds in chosen:
        print(f"  ── {ds.upper()} ──")

        # Standard (Reweighing + Thresholding)
        std_done = (
            (REPO_ROOT / "models" / "mitigated_pre"  / f"{ds}_reweighed.pkl").exists() and
            (REPO_ROOT / "models" / "mitigated_post" / f"{ds}_threshold.pkl").exists()
        )
        if std_done and not _confirm(f"  Standard models for {ds.upper()} exist. Re-train?", False):
            _info("Skipping standard mitigations")
        else:
            print(f"  Standard mitigations (Reweighing + ThresholdOptimizer)...")
            _log(f"Phase 4: {ds} standard")
            try:
                std_mod.run_standard_mitigation(ds)
                _ok(f"Reweighing → models/mitigated_pre/{ds}_reweighed.pkl")
                _ok(f"Thresholding → models/mitigated_post/{ds}_threshold.pkl")
                _log(f"Phase 4: {ds} standard OK")
            except Exception as exc:
                _err(f"Standard mitigation failed: {exc}")
                _log(f"Phase 4: {ds} standard FAILED: {exc}")
                traceback.print_exc()

        # Adversarial
        adv_done = (
            (REPO_ROOT / "models" / "mitigated_adversarial_det" / f"{ds}_adversarial_det.pth").exists()
            or (REPO_ROOT / "models" / "mitigated_adversarial" / f"{ds}_adversarial.pth").exists()
        )
        if adv_done and not _confirm(f"  Adversarial model for {ds.upper()} exists. Re-train?", False):
            _info("Skipping adversarial training")
        else:
            print(f"  Adversarial FairNet (seed={SEED}, epochs=50, λ=0.5)...")
            _log(f"Phase 4: {ds} adversarial")
            try:
                adv_mod.run_adversarial_debiasing(ds)
                _ok(f"FairNet → models/mitigated_adversarial/{ds}_adversarial.pth")
                _log(f"Phase 4: {ds} adversarial OK")
            except Exception as exc:
                _err(f"Adversarial training failed: {exc}")
                _log(f"Phase 4: {ds} adversarial FAILED: {exc}")
                traceback.print_exc()

        print()

    _info(f"Phase 4 complete in {_elapsed(t0)}")
    _log(f"Phase 4: complete ({_elapsed(t0)})")
    return detect_state()

# ─────────────────────────────────────────────────────────────────────────────
# 14. PHASE 5: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def phase_evaluation(state: dict) -> dict:
    _section("Phase 5 — Evaluation (Table 5)")
    _log("Phase 5: started")
    t0 = time.time()

    print("  Evaluates all models on held-out test sets and writes")
    print("  results/final_paper_results.csv — the canonical Table 5.")
    print("  ThresholdOptimizer.predict() uses random_state=42.")
    print()

    # Soft prerequisite warning (non-blocking)
    no_bl = [ds for ds in DATASETS if not state["baselines_trained"][ds] and state["preprocessed"][ds]]
    if no_bl:
        _warn(f"Baselines missing for {', '.join(d.upper() for d in no_bl)} — those rows will fail.")

    if state["results_generated"] and not _confirm("Results CSV already exists. Regenerate?", False):
        _ok("Using existing results/final_paper_results.csv")
        _print_results_table()
        return state

    try:
        import scripts.generate_final_results as gfr_mod
        print("  Running evaluation...")
        _log("Phase 5: calling generate_final_results.main()")
        gfr_mod.main()
        _ok("results/final_paper_results.csv written")
        _log("Phase 5: OK")
        _print_results_table()
    except Exception as exc:
        _err(f"Evaluation failed: {exc}")
        _log(f"Phase 5: FAILED: {exc}")
        traceback.print_exc()

    _info(f"Phase 5 complete in {_elapsed(t0)}")
    _log(f"Phase 5: complete ({_elapsed(t0)})")
    return detect_state()


def _print_results_table() -> None:
    csv_path = REPO_ROOT / "results" / "final_paper_results.csv"
    if not csv_path.exists():
        return
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print()
        w = 62
        _hr()
        print(f"  {'Dataset':<8} {'Method':<26} {'Acc':>5} {'Rec':>5} {'EOD':>5}  Note")
        _hr()
        for _, row in df.iterrows():
            try:
                eod = float(row["EO_Diff"])
            except (ValueError, KeyError):
                eod = float("nan")
            note = ""
            if eod >= 1.0:        note = "⚠ Collapsed"
            elif eod > 0.30:      note = "⚠ Biased"
            elif eod > 0.15:      note = "◑ Improved"
            else:                 note = "✓ Best"
            print(
                f"  {str(row['Dataset']):<8} {str(row['Method']):<26}"
                f" {row['Accuracy']:>5.3f} {row['Recall']:>5.3f}"
                f" {eod:>5.3f}  {note}"
            )
        _hr()
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 15. PHASE 6: FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def phase_figures(state: dict) -> dict:
    _section("Phase 6 — Figures")
    _log("Phase 6: started")
    t0 = time.time()

    print("  Generates all publication figures:")
    print("  • Figure 2 (Pareto)    Recall vs EOD frontier — OULAD")
    print("  • Figure 3 (Heatmap)   Intersectional unfairness — Gender × SES")
    print("  • Figure 4 (SHAP)      Differential feature importance by SES")
    print("  • Figure 5 (Collapse)  Adversarial training dynamics — UCI")
    print()

    import matplotlib
    matplotlib.use("Agg")
    (REPO_ROOT / "figures").mkdir(exist_ok=True)

    generated, skipped, failed = [], [], []

    # ── Figure 2: Pareto ────────────────────────────────────────────────────
    try:
        import scripts.plot_pareto as pp_mod
        pp_mod.plot_pareto(dataset="oulad")
        _ok("Figure 2 (Pareto)  → figures/figure2_pareto_oulad.png")
        generated.append("figure2_pareto_oulad.png")
        _log("Phase 6: Pareto OK")
    except Exception as exc:
        _err(f"Pareto failed: {exc}")
        failed.append("figure2_pareto_oulad.png")
        _log(f"Phase 6: Pareto FAILED: {exc}")

    # ── Figure 3/4/5: Additional ────────────────────────────────────────────
    try:
        import scripts.plot_additional_figures as paf_mod
    except ImportError as exc:
        _warn(f"Could not import plot_additional_figures: {exc}")
        paf_mod = None

    if paf_mod:
        for fn_name, label, out_file in [
            ("plot_intersectional_heatmap", "Figure 3 (Heatmap)",  "fig3_intersectional_heatmap.png"),
            ("plot_differential_shap",      "Figure 4 (SHAP)",     "fig4_differential_shap.png"),
            ("plot_model_collapse",         "Figure 5 (Collapse)", "fig5_model_collapse.png"),
        ]:
            try:
                getattr(paf_mod, fn_name)()
                _ok(f"{label}  → figures/{out_file}")
                generated.append(out_file)
                _log(f"Phase 6: {fn_name} OK")
            except Exception as exc:
                _err(f"{label} failed: {exc}")
                failed.append(out_file)
                _log(f"Phase 6: {fn_name} FAILED: {exc}")

    # ── Note real collapse figure ────────────────────────────────────────────
    real = REPO_ROOT / "figures" / "fig4_model_collapse.png"
    if real.exists():
        _info("Note: figures/fig4_model_collapse.png (real per-epoch data, seed=42) also present")

    # ── Causal DAG ──────────────────────────────────────────────────────────
    if not state["figures"].get("causal_dag_v1.png"):
        _warn("Causal DAG not found — run Phase 3 to generate it.")
    else:
        _ok("Figure 1 (DAG)   → figures/causal_dag_v1.png  ✓ already present")

    print()
    _info(f"Generated: {len(generated)}  Failed: {len(failed)}  Location: figures/")
    _info(f"Phase 6 complete in {_elapsed(t0)}")
    _log(f"Phase 6: complete — generated={generated} failed={failed}")
    return detect_state()

# ─────────────────────────────────────────────────────────────────────────────
# 16. RUN ALL
# ─────────────────────────────────────────────────────────────────────────────
def run_all(state: dict) -> dict:
    _section("Run All (Phases 1 → 6)")
    print("  Runs the full pipeline sequentially.")
    print("  Existing outputs will be preserved — you will be prompted per phase.")
    print()
    if not _confirm("Start full pipeline?", True):
        return state

    _log("Run All: started")
    t_total = time.time()

    phases = [
        ("Preprocessing", phase_preprocessing),
        ("Detection",     phase_detection),
        ("Diagnosis",     phase_diagnosis),
        ("Mitigation",    phase_mitigation),
        ("Evaluation",    phase_evaluation),
        ("Figures",       phase_figures),
    ]

    for name, fn in phases:
        print()
        print(f"  ━━━  {name}  ━━━")
        try:
            state = fn(state)
        except KeyboardInterrupt:
            print()
            _warn("Interrupted — stopping pipeline")
            break
        except Exception as exc:
            _err(f"{name} failed: {exc}")
            traceback.print_exc()
            if not _confirm(f"Continue to next phase after {name} failure?", True):
                break

    print()
    _info(f"Full pipeline finished in {_elapsed(t_total)}")
    _log(f"Run All: complete ({_elapsed(t_total)})")
    return state

# ─────────────────────────────────────────────────────────────────────────────
# 17. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if sys.version_info < (3, 8):
        print(f"Error: Python >= 3.8 required (got {sys.version.split()[0]})")
        sys.exit(1)

    _log(f"run.py started — seed={SEED} — Python {sys.version.split()[0]}")
    print_banner()

    # Quick startup scan
    print("  Scanning environment...")
    state = detect_state()

    # Non-blocking startup summary
    missing_pkgs = [pkg for pkg, imp in _PKG_IMPORT_MAP.items() if not _check_pkg(imp)]
    if missing_pkgs:
        _warn(f"Missing packages: {', '.join(missing_pkgs)}.  Run [0] to install.")
    else:
        _ok("All dependencies present")
    _data_status(state)

    _PHASE_MAP = {
        "0": phase_preflight,
        "1": phase_preprocessing,
        "2": phase_detection,
        "3": phase_diagnosis,
        "4": phase_mitigation,
        "5": phase_evaluation,
        "6": phase_figures,
        "a": run_all,
    }

    while True:
        state = detect_state()
        print_menu(state)

        try:
            choice = input("  Select phase: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice == "q":
            break

        fn = _PHASE_MAP.get(choice)
        if fn is None:
            _warn(f"Unknown option '{choice}'. Choose 0–6, A, or Q.")
            continue

        _log(f"User selected: {choice}")
        try:
            state = fn(state)
        except KeyboardInterrupt:
            print()
            _warn("Interrupted — back to menu")
        except Exception as exc:
            _err(f"Unexpected error: {exc}")
            traceback.print_exc()
            _log(f"Unhandled exception in phase {choice}: {exc}")

    _log("run.py exited cleanly")
    print()
    print("  Goodbye.")
    print()


if __name__ == "__main__":
    main()
