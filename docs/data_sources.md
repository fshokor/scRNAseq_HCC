# Data sources

All data used in this project are publicly available. This document lists
each source, its access method, and the pipeline step that uses it.

---

## Primary dataset — scRNA-seq

| Field | Details |
|-------|---------|
| Accession | GSE166635 |
| Database | NCBI GEO |
| URL | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166635 |
| Description | Single-cell RNA-seq of HCC tumor (HCC2) and adjacent normal tissue (HCC1) |
| Platform | 10x Genomics Chromium |
| Sequencer | Illumina NovaSeq 6000 |
| Format | MTX (barcodes, features, matrix) |
| Size | ~204 MB compressed |
| Download | `python scripts/data_download.py` |
| Used in | Notebooks 01–04, 06 |

---

## Annotation references

| Resource | Access | Used in |
|----------|--------|---------|
| CellTypist models (`Immune_All_High.pkl`, `Immune_All_Low.pkl`) | Auto-downloaded by `celltypist.models.download_models()` | Notebook 03 |
| ScType marker database (`ScTypeDB_full.xlsx`) | Sourced at runtime from [GitHub](https://github.com/IanevskiAleksandr/sc-type) | Notebook 03 |
| Human Primary Cell Atlas (HPCA) | `celldex::HumanPrimaryCellAtlasData()` via Bioconductor ExperimentHub | Notebook 03 |

---

## Survival data

| Field | Details |
|-------|---------|
| Cohort | TCGA-LIHC (The Cancer Genome Atlas Liver Hepatocellular Carcinoma) |
| Source | UCSC Xena public hub |
| Clinical URL | `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LIHC.sampleMap%2FLIHC_clinicalMatrix` |
| Expression URL | `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LIHC.sampleMap%2FHiSeqV2` |
| Patients | ~374 HCC patients (primary tumor samples only) |
| Variables used | Overall survival time (days), survival event (0/1), gene expression (log2 RSEM) |
| Download | Auto-downloaded in notebook 11 / `scripts/survival_analysis.py` |
| Fallback | Simulated TCGA-like data used if download fails |

---

## PPI network

| Field | Details |
|-------|---------|
| Database | STRING v12 |
| URL | https://string-db.org |
| API | `https://string-db.org/api/json/network` |
| Species | Homo sapiens (taxon 9606) |
| Network type | Functional |
| Minimum score | 400 (medium confidence) |
| Access | REST API, queried in notebook 09 / `scripts/ppi_analysis.py` |

---

## Drug–gene interactions

| Database | Type | URL | Access |
|----------|------|-----|--------|
| DGIdb | GraphQL API | https://dgidb.org/api/graphql | `scripts/utils/api_clients.py` |
| ChEMBL | REST API | https://www.ebi.ac.uk/chembl/api/data | `scripts/utils/api_clients.py` |
| OpenTargets Platform | GraphQL API | https://api.platform.opentargets.org/api/v4/graphql | `scripts/utils/api_clients.py` |
| Curated fallback | Built-in dataset | — | `scripts/utils/api_clients.py::get_curated_fallback()` |

The curated fallback contains 37 literature-based drug-gene interactions for
16 HCC hub genes, compiled from the Wang et al. (2025) paper and supporting
references. It is used automatically when live APIs are inaccessible.

---

## Gene ontology & pathway databases

| Resource | Access | Used in |
|----------|--------|---------|
| GO Biological Process, MF, CC | `clusterProfiler::gseGO` + `org.Hs.eg.db` | Notebook 05 |
| KEGG | `clusterProfiler::gseKEGG` (organism `hsa`) | Notebook 05 |
| DisGeNET, Reactome | Referenced in paper; not queried programmatically | — |

---

## Molecular descriptors (GNN features)

Drug molecular features were derived from the DGI databases listed above.
No external molecular fingerprint databases (PubChem, ChEMBL structures)
are downloaded — only the tabular interaction data from the APIs.

---

## Licences & terms of use

| Resource | Licence / terms |
|----------|-----------------|
| GSE166635 | GEO public access — free for research use |
| TCGA-LIHC | NIH TCGA data access policy — open-access tier |
| STRING | Creative Commons CC BY 4.0 |
| DGIdb | Creative Commons CC0 |
| ChEMBL | Creative Commons CC BY-SA 3.0 |
| OpenTargets | Apache 2.0 |
| CellTypist models | Creative Commons CC BY 4.0 |
| ScType database | MIT licence |
| Human Primary Cell Atlas | Creative Commons CC BY 4.0 |

All data are used strictly for non-commercial research purposes.
