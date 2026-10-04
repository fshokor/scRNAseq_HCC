"""
survival_functions.py
=====================
Optional supporting bulk-tumour survival evidence for notebook P2.

Functions
---------
load_gene_list      — load DEA + hub gene lists
fetch_tcga_lihc     — download TCGA-LIHC clinical + expression from UCSC Xena
simulate_tcga       — explicit demonstration/test data only
run_survival        — Kaplan-Meier + Cox regression per gene
filter_survivors    — flag Cox FDR support with PH diagnostics
export_survival     — save results CSVs
"""

import io
import warnings
import requests
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from statsmodels.stats.multitest import multipletests



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
def _prepare_tcga(clinical, expression):
    """Validate OS endpoints and use one primary-tumour expression row per patient.

    Expression is samples x genes. Multiple primary aliquots are averaged.
    Endpoint conflicts within a patient fail validation rather than selecting
    an arbitrary clinical record. OS_time from these Xena fields is in days.
    """
    id_col = next((c for c in ["sampleID", "patient_id", "sample"] if c in clinical), None)
    if id_col is None:
        raise ValueError("Clinical sample/patient identifier is missing")
    def endpoint(candidates, label):
        present = [c for c in candidates if c in clinical]
        if not present:
            raise ValueError(f"Missing validated {label} endpoint")
        result = pd.to_numeric(clinical[present[0]], errors="coerce")
        for col in present[1:]:
            other = pd.to_numeric(clinical[col], errors="coerce")
            overlap = result.notna() & other.notna()
            if not np.allclose(result[overlap], other[overlap]):
                raise ValueError(f"Conflicting aliases for {label}")
            result = result.fillna(other)
        return result
    c = clinical.copy()
    c["OS_time"] = endpoint(["OS_time", "OS.time", "_OS"], "OS time")
    c["OS_event"] = endpoint(["OS_event", "OS", "_OS_IND"], "OS event")
    identifiers = c[id_col].astype(str)
    is_sample = identifiers.str.len() >= 15
    c = c[~is_sample | (identifiers.str[13:15] == "01")].copy()
    c["patient_id"] = c[id_col].astype(str).str[:12]
    c = c.dropna(subset=["OS_time", "OS_event"])
    if not c.OS_event.isin([0, 1]).all() or (c.OS_time <= 0).any():
        raise ValueError("OS events must be 0/1 and OS times positive")
    conflicts = c.groupby("patient_id")[["OS_time", "OS_event"]].nunique()
    if (conflicts > 1).any().any():
        raise ValueError("Conflicting survival endpoints within a patient")
    c = c.sort_values(id_col).drop_duplicates("patient_id")
    e = expression.copy()
    e.index = e.index.astype(str)
    e = e[e.index.str[13:15] == "01"].apply(pd.to_numeric, errors="coerce")
    if not e.columns.is_unique:
        raise ValueError("Duplicate expression gene identifiers")
    e.index = e.index.str[:12]
    e = e.groupby(level=0).mean()
    # Keep other clinical fields for explicitly selected covariate adjustment.
    overlap = set(c.columns) & set(e.columns)
    if overlap:
        raise ValueError(f"Gene/clinical column collision: {sorted(overlap)}")
    merged = c.merge(e, left_on="patient_id", right_index=True, validate="one_to_one")
    if len(merged) < 20:
        raise ValueError("Fewer than 20 patients with validated endpoints and primary tumour RNA")
    merged.attrs.update(time_unit="days", source="TCGA-LIHC UCSC Xena",
                        expression_aggregation="mean primary-tumour aliquots per patient")
    return merged


def fetch_tcga_lihc():
    """Return real validated data or (None, False). Never simulate on failure."""
    try:
        r = requests.get(_CLINICAL_URL, timeout=60)
        r.raise_for_status()
        clinical = pd.read_csv(io.StringIO(r.text), sep="\t", low_memory=False)
        r = requests.get(_EXPR_URL, timeout=120)
        r.raise_for_status()
        expression = pd.read_csv(io.StringIO(r.text), sep="\t", index_col=0).T
        merged = _prepare_tcga(clinical, expression)
        print(f"Validated TCGA-LIHC: {len(merged)} unique patients")
        return merged, False
    except Exception as exc:
        print(f"Survival evidence unavailable: {exc}")
        return None, False


def simulate_tcga(gene_list, n=374, random_seed=42):
    """Explicit null simulation for demonstrations only; no gene-specific effects.

    Marked simulated so scientific run_survival rejects this object.
    This function never acts as a download fallback.
    """
    rng = np.random.default_rng(random_seed)
    frame = pd.DataFrame({
        "patient_id": [f"DEMO{i:04d}" for i in range(n)],
        "OS_time": rng.exponential(800, n) + 1,
        "OS_event": rng.binomial(1, 0.55, n),
        **{g: rng.normal(size=n) for g in gene_list},
    })
    frame.attrs["is_simulated"] = True
    return frame


def _adjust_p(values):
    values = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index)
    valid = values.notna()
    if valid.any():
        result.loc[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return result


def run_survival(gene_list, merged, covariates=(), min_patients=20, min_events=10):
    """Continuous Cox HR per expression SD; KM is descriptive supporting evidence.

    Covariates must be explicitly supplied validated clinical column names.
    Categorical covariates are dummy encoded. Failed/untestable genes remain
    in the ledger. BH correction is across successfully tested genes within
    this contrast, not across all possible project hypotheses.
    """
    if merged is None:
        raise ValueError("Real survival data are unavailable")
    if merged.attrs.get("is_simulated"):
        raise ValueError("Simulated data cannot enter scientific survival prioritisation")
    if merged.patient_id.duplicated().any():
        raise ValueError("Survival input must contain one row per patient")
    if not merged.OS_event.dropna().isin([0, 1]).all() or (merged.OS_time.dropna() <= 0).any():
        raise ValueError("Invalid survival endpoints")
    for col in covariates:
        if col in {"patient_id", "OS_time", "OS_event"} or col in gene_list:
            raise ValueError(f"Invalid clinical covariate: {col}")
        if col not in merged:
            raise KeyError(f"Missing requested covariate: {col}")
    rows = []
    for gene in dict.fromkeys(gene_list):
        row = dict(gene=gene, logrank_p=np.nan, cox_p=np.nan, HR=np.nan,
                   HR_CI_low=np.nan, HR_CI_high=np.nan, ph_p=np.nan,
                   n_patients=0, n_events=0, status="not_tested", reason="",
                   covariates=" | ".join(covariates), model="adjusted" if covariates else "unadjusted")
        if gene not in merged:
            row["reason"] = "No expression measurement"
            rows.append(row)
            continue
        gd = merged[["OS_time", "OS_event", gene, *covariates]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        gd = gd.rename(columns={"OS_time": "T", "OS_event": "E", gene: "expr"})
        row.update(n_patients=len(gd), n_events=int(gd.E.sum()))
        if len(gd) < min_patients or gd.E.sum() < min_events or gd.expr.std() == 0:
            row["reason"] = "Too few patients/events or constant expression"
            rows.append(row)
            continue
        high, low = gd[gd.expr >= gd.expr.median()], gd[gd.expr < gd.expr.median()]
        if min(len(high), len(low)) >= 5:
            row["logrank_p"] = logrank_test(high["T"], low["T"],
                event_observed_A=high.E, event_observed_B=low.E).p_value
        cd = pd.get_dummies(gd, columns=[c for c in covariates if not pd.api.types.is_numeric_dtype(gd[c])], drop_first=True, dtype=float)
        cd["expr"] = (cd.expr - cd.expr.mean()) / cd.expr.std()
        predictors = [c for c in cd if c not in ["T", "E"]]
        cd = cd.drop(columns=[c for c in predictors if c != "expr" and cd[c].nunique() <= 1])
        if gd.E.sum() < max(min_events, 10 * (len(cd.columns) - 2)):
            row["reason"] = "Insufficient events for requested model complexity"
            rows.append(row)
            continue
        try:
            cph = CoxPHFitter(penalizer=0.0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cph.fit(cd, duration_col="T", event_col="E")
            if any("Convergence" in w.category.__name__ for w in caught):
                raise RuntimeError("Cox convergence warning; model needs review: " +
                                   " | ".join(str(w.message) for w in caught))
            ci = cph.confidence_intervals_.loc["expr"]
            row.update(HR=float(np.exp(cph.params_["expr"])),
                       HR_CI_low=float(np.exp(ci.iloc[0])), HR_CI_high=float(np.exp(ci.iloc[1])),
                       cox_p=float(cph.summary.loc["expr", "p"]), status="completed")
            ph = proportional_hazard_test(cph, cd, time_transform="rank")
            row["ph_p"] = float(ph.summary.loc["expr", "p"])
            row["ph_min_p"] = float(ph.summary.p.min())
        except Exception as exc:
            row.update(status="failed", reason=str(exc))
        rows.append(row)
    columns = ["gene", "logrank_p", "cox_p", "HR", "HR_CI_low", "HR_CI_high", "ph_p", "ph_min_p",
               "n_patients", "n_events", "status", "reason", "covariates", "model"]
    result = pd.DataFrame(rows, columns=columns)
    for field in ["cox_p", "logrank_p"]:
        result[field + "_adj"] = _adjust_p(result[field])
    result["ph_warning"] = result.ph_min_p < 0.05
    return result


def filter_survivors(surv_df, sig, km_p=0.05, cox_p=0.05,
                     hr_min=0.8, hr_max=1.2):
    """Cox BH-FDR support only; KM and arbitrary HR cutoffs are not gates.

    Legacy threshold arguments remain accepted for existing callers. PH
    diagnostics are screening flags; models with missing or flagged checks
    are not labelled supported until separately reviewed.
    """
    merged = surv_df.merge(sig, on="gene", how="left").sort_values("cox_p_adj")
    supported = merged[(merged.status == "completed") & (merged.cox_p_adj < cox_p)
                       & merged.ph_min_p.notna() & ~merged.ph_warning].copy()
    supported["prognosis"] = np.where(supported.HR < 1, "lower_hazard_association", "higher_hazard_association")
    return merged.reset_index(drop=True), supported.reset_index(drop=True)


def export_survival(surv_df, filtered, tables_dir):
    from pathlib import Path
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    surv_df.to_csv(tables_dir / "survival_results.csv", index=False)
    filtered.to_csv(tables_dir / "survival_supported_genes.csv", index=False)
