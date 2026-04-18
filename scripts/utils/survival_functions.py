"""
survival_functions.py
=====================
All logic for notebook P2 · Survival Filter.

Functions
---------
load_gene_list      — load DEA + hub gene lists
fetch_tcga_lihc     — download TCGA-LIHC clinical + expression from UCSC Xena
simulate_tcga       — realistic simulated TCGA fallback (no internet needed)
run_survival        — Kaplan-Meier + Cox regression per gene
filter_survivors    — apply significance thresholds
export_survival     — save results CSVs
"""

import io
import warnings
import requests
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

# TCGA-LIHC download URLs (UCSC Xena public hub)
_CLINICAL_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.LIHC.sampleMap%2FLIHC_clinicalMatrix"
)
_EXPR_URL = (
    "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    "TCGA.LIHC.sampleMap%2FHiSeqV2"
)


# ─────────────────────────────────────────────────────────────────────────────
def load_gene_list(dea_path, hub_path=None,
                   padj_thresh=0.05, log2fc_thresh=1.0):
    """
    Load significant DEGs and optionally their hub scores.

    Parameters
    ----------
    dea_path : Path
        Path to dea_results.csv.
    hub_path : Path or None
        Path to hub_genes.csv. If provided, hub_score_map is populated.
    padj_thresh, log2fc_thresh : float
        Significance filters.

    Returns
    -------
    sig : pd.DataFrame
        Filtered DEGs with regulation column.
    gene_list : list
        All significant gene symbols.
    hub_score_map : dict
        gene → hub_score (empty if hub_path is None or missing).
    """
    dea = pd.read_csv(dea_path)
    sig = dea[
        (dea.adj_pvalue < padj_thresh) &
        (dea.log2FC.abs() >= log2fc_thresh)
    ].copy()
    sig["regulation"] = (sig.log2FC > 0).map({True: "up", False: "down"})
    gene_list = sig.gene.dropna().unique().tolist()

    hub_score_map = {}
    if hub_path is not None and hub_path.exists():
        hub_df = pd.read_csv(hub_path)
        if "hub_score" in hub_df.columns:
            hub_score_map = hub_df.set_index("gene")["hub_score"].to_dict()

    print(f"DEGs       : {len(gene_list)}")
    print(f"Hub scores : {len(hub_score_map)} genes loaded")
    return sig, gene_list, hub_score_map


# ─────────────────────────────────────────────────────────────────────────────
def fetch_tcga_lihc():
    """
    Download TCGA-LIHC clinical (overall survival) and RNA-seq expression
    data from the UCSC Xena public hub.

    Returns
    -------
    merged : pd.DataFrame or None
        Wide-format dataframe: patient_id, OS_time, OS_event, <gene cols>.
        None if the download failed.
    is_simulated : bool
        True if the download failed (caller should use simulate_tcga).
    """
    print("Downloading TCGA-LIHC from UCSC Xena...")
    try:
        r = requests.get(_CLINICAL_URL, timeout=30)
        r.raise_for_status()
        clinical = pd.read_csv(io.StringIO(r.text), sep="\t", low_memory=False)
        clinical = clinical.rename(columns={
            "sampleID" : "patient_id",
            "OS.time"  : "OS_time",
            "OS"       : "OS_event",
            "_OS_IND"  : "OS_event",
            "_OS"      : "OS_time",
        })
        if "patient_id" in clinical.columns:
            # Keep only primary tumor samples (barcode position 13–14 == "01")
            clinical = clinical[clinical.patient_id.str[13:15] == "01"].copy()
        clinical = clinical[["patient_id", "OS_time", "OS_event"]].dropna()
        clinical[["OS_time", "OS_event"]] = clinical[
            ["OS_time", "OS_event"]].apply(pd.to_numeric, errors="coerce")
        clinical = clinical.dropna()
        print(f"  Clinical: {len(clinical)} patients")

        r2 = requests.get(_EXPR_URL, timeout=120)
        r2.raise_for_status()
        expr = (pd.read_csv(io.StringIO(r2.text), sep="\t",
                            index_col=0, low_memory=False)
                  .T.reset_index()
                  .rename(columns={"index": "patient_id"}))
        expr["patient_id"]     = expr.patient_id.str[:15]
        clinical["patient_id"] = clinical.patient_id.str[:15]
        merged = clinical.merge(expr, on="patient_id", how="inner")
        print(f"  Merged  : {len(merged)} patients with expression data")
        return merged, False

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print("  → Will use simulated data")
        return None, True


# ─────────────────────────────────────────────────────────────────────────────
def simulate_tcga(gene_list, n=374, random_seed=42):
    """
    Generate realistic TCGA-LIHC-like survival + expression data.
    Used automatically when the real download fails.

    Known protective genes (APOE, ALB) are given better simulated survival.
    Known risk genes (XIST, FTL) are given worse simulated survival.

    Parameters
    ----------
    gene_list : list
        Gene symbols to include as expression columns.
    n : int
        Number of simulated patients.
    random_seed : int

    Returns
    -------
    df : pd.DataFrame
        Columns: patient_id, OS_time, OS_event, <gene_list cols>.
    """
    np.random.seed(random_seed)
    os_time  = np.random.exponential(800, n).clip(30, 3000)
    os_event = np.random.binomial(1, 0.55, n)
    expr = {g: np.random.randn(n) for g in gene_list}

    for g in ["APOE", "ALB"]:
        if g in expr:
            hi = expr[g] > 0
            os_time[hi]  *= np.random.uniform(1.1, 1.4, hi.sum())
            os_event[hi]  = np.random.binomial(1, 0.40, hi.sum())

    for g in ["XIST", "FTL"]:
        if g in expr:
            hi = expr[g] > 0
            os_time[hi]  *= np.random.uniform(0.6, 0.85, hi.sum())
            os_event[hi]  = np.random.binomial(1, 0.70, hi.sum())

    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "OS_time"   : os_time.clip(30, 3000),
        "OS_event"  : os_event.astype(int),
        **expr,
    })


# ─────────────────────────────────────────────────────────────────────────────
def run_survival(gene_list, merged):
    """
    Run Kaplan-Meier log-rank test + Cox regression for every gene.

    Patients are split at the gene's median expression level.
    Cox model uses standardised expression to get interpretable HR.

    Parameters
    ----------
    gene_list : list
        Genes to test (must be columns in merged).
    merged : pd.DataFrame
        TCGA-LIHC or simulated data (patient_id, OS_time, OS_event, genes).

    Returns
    -------
    results : pd.DataFrame
        Columns: gene, logrank_p, cox_p, HR, HR_CI_low, HR_CI_high.
    """
    avail   = [g for g in gene_list if g in merged.columns]
    results = []
    print(f"Testing {len(avail)} genes...")

    for i, gene in enumerate(avail):
        gd = merged[["OS_time", "OS_event", gene]].dropna().copy()
        gd.columns = ["T", "E", "expr"]
        if len(gd) < 20:
            continue
        gd["group"] = np.where(gd.expr >= gd.expr.median(), "High", "Low")
        hi, lo = gd[gd.group == "High"], gd[gd.group == "Low"]
        if len(hi) < 5 or len(lo) < 5:
            continue

        lr = logrank_test(hi.T, lo.T,
                          event_observed_A=hi.E, event_observed_B=lo.E)
        try:
            cd = gd[["T", "E", "expr"]].copy()
            cd["expr"] = (cd.expr - cd.expr.mean()) / (cd.expr.std() + 1e-9)
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cd, duration_col="T", event_col="E", show_progress=False)
            hr   = float(np.exp(cph.params_["expr"]))
            ci_l = float(np.exp(
                cph.confidence_intervals_.loc["expr", "95% lower-bound"]))
            ci_h = float(np.exp(
                cph.confidence_intervals_.loc["expr", "95% upper-bound"]))
            cox_p = float(cph.summary.loc["expr", "p"])
        except Exception:
            hr = ci_l = ci_h = cox_p = np.nan

        results.append({
            "gene"      : gene,
            "logrank_p" : lr.p_value,
            "cox_p"     : cox_p,
            "HR"        : hr,
            "HR_CI_low" : ci_l,
            "HR_CI_high": ci_h,
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(avail)}]")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
def filter_survivors(surv_df, sig,
                     km_p=0.05, cox_p=0.05, hr_min=0.8, hr_max=1.2):
    """
    Merge DEA stats into survival results and apply significance filters.

    A gene passes if:
      logrank_p < km_p  AND  cox_p < cox_p_thresh  AND
      (HR < hr_min  OR  HR > hr_max)

    Parameters
    ----------
    surv_df : pd.DataFrame
        Output of run_survival().
    sig : pd.DataFrame
        Significant DEGs (gene, log2FC, adj_pvalue, regulation).
    km_p, cox_p : float
        P-value thresholds.
    hr_min, hr_max : float
        HR range to exclude (no meaningful prognostic effect).

    Returns
    -------
    surv_df : pd.DataFrame
        Full results with DEA columns merged in.
    filtered : pd.DataFrame
        Genes passing all filters, with prognosis column (protective/risk).
    """
    surv_df = (surv_df
               .merge(sig[["gene", "log2FC", "adj_pvalue", "regulation"]],
                      on="gene", how="left")
               .sort_values("logrank_p")
               .reset_index(drop=True))

    filtered = surv_df[
        (surv_df.logrank_p < km_p) &
        (surv_df.cox_p     < cox_p) &
        ((surv_df.HR < hr_min) | (surv_df.HR > hr_max))
    ].copy()
    filtered["prognosis"] = filtered.HR.apply(
        lambda h: "protective" if pd.notna(h) and h < 1 else "risk")

    print(f"Genes analysed         : {len(surv_df)}")
    print(f"KM significant         : {(surv_df.logrank_p<km_p).sum()}")
    print(f"Passing all filters    : {len(filtered)}")
    print(f"  Protective (HR<1)    : {(filtered.HR<1).sum()}")
    print(f"  Risk (HR>1)          : {(filtered.HR>1).sum()}")
    return surv_df, filtered


# ─────────────────────────────────────────────────────────────────────────────
def export_survival(surv_df, filtered, tables_dir):
    """
    Save survival_results.csv and survival_filtered_genes.csv.

    Parameters
    ----------
    surv_df : pd.DataFrame
        Full per-gene survival stats.
    filtered : pd.DataFrame
        Genes passing all significance filters.
    tables_dir : Path
        Output directory.
    """
    surv_df["prognosis"] = surv_df.HR.apply(
        lambda h: "protective" if pd.notna(h) and h < 1
                  else "risk" if pd.notna(h) else "")
    surv_df.to_csv(tables_dir / "survival_results.csv", index=False)

    filt_cols = ["gene", "logrank_p", "cox_p", "HR", "HR_CI_low", "HR_CI_high",
                 "log2FC", "regulation", "prognosis"]
    filtered[[c for c in filt_cols if c in filtered.columns]].to_csv(
        tables_dir / "survival_filtered_genes.csv", index=False)

    print(f"Saved: survival_results.csv          ({len(surv_df)} genes)")
    print(f"Saved: survival_filtered_genes.csv   ({len(filtered)} genes  →  GNN input)")
