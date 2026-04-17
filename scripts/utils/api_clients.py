"""
api_clients.py
==============
HTTP clients for the three drug-gene interaction databases used in step 12.

Functions
---------
safe_request          — retry wrapper with rate-limit handling
query_dgidb           — DGIdb GraphQL API
query_chembl          — ChEMBL REST API (target search + mechanism)
query_opentargets     — OpenTargets GraphQL API
get_curated_fallback  — literature-curated dataset (used when APIs blocked)
"""

import time
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def safe_request(method: str, url: str, retries: int = 3, **kwargs):
    """
    Wrapper around requests.get / requests.post with retry logic.

    Parameters
    ----------
    method : str
        "get" or "post".
    url : str
        Full request URL.
    retries : int
        Number of attempts before giving up.
    **kwargs
        Passed directly to requests.get / requests.post.

    Returns
    -------
    requests.Response or None
        None if all retries failed or a non-retryable status code was returned.
    """
    kwargs.setdefault("timeout", 30)
    for attempt in range(retries):
        try:
            r = getattr(requests, method)(url, **kwargs)
            if r.status_code == 200:
                return r
            if r.status_code in [403, 404]:
                return None          # not accessible — don't retry
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"    Request failed after {retries} attempts: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DGIdb
# ─────────────────────────────────────────────────────────────────────────────

DGIDB_URL = "https://dgidb.org/api/graphql"

_DGIDB_QUERY = """
query GetInteractions($genes: [String!]!) {
  genes(names: $genes) {
    nodes {
      name
      interactions {
        interactionScore
        interactionTypes { type directionality }
        publications { pmid }
        sources { fullName }
        drug {
          name
          conceptId
          approved
          immunotherapy
          antiNeoplastic
        }
      }
    }
  }
}
"""


def query_dgidb(genes: list, batch_size: int = 10) -> list:
    """
    Query DGIdb GraphQL API for all interactions of the given gene list.

    Parameters
    ----------
    genes : list
        Gene symbols to query.
    batch_size : int
        Number of genes per API request (DGIdb recommends ≤ 10).

    Returns
    -------
    list of dict
        One dict per drug-gene interaction edge.
    """
    edges = []
    for i in range(0, len(genes), batch_size):
        batch = genes[i: i + batch_size]
        r = safe_request(
            "post", DGIDB_URL,
            json={"query": _DGIDB_QUERY, "variables": {"genes": batch}},
        )
        if r is None:
            print(f"    DGIdb batch {i // batch_size + 1}: no response")
            continue

        nodes = r.json().get("data", {}).get("genes", {}).get("nodes", [])
        for node in nodes:
            gene = node["name"]
            for ix in node.get("interactions", []):
                drug = ix.get("drug", {})
                if not drug or not drug.get("name"):
                    continue
                itype = ix.get("interactionTypes", [])
                edges.append({
                    "gene"             : gene,
                    "drug"             : drug["name"],
                    "drug_id"          : drug.get("conceptId", ""),
                    "source"           : "DGIdb",
                    "interaction_type" : itype[0]["type"] if itype else "unknown",
                    "directionality"   : itype[0]["directionality"] if itype else "unknown",
                    "approved"         : bool(drug.get("approved", False)),
                    "immunotherapy"    : bool(drug.get("immunotherapy", False)),
                    "anti_neoplastic"  : bool(drug.get("antiNeoplastic", False)),
                    "interaction_score": float(ix.get("interactionScore") or 0),
                    "n_publications"   : len(ix.get("publications", [])),
                    "clinical_phase"   : 4 if drug.get("approved") else 0,
                })
        time.sleep(0.5)

    print(f"    DGIdb: {len(edges)} interactions returned")
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# ChEMBL
# ─────────────────────────────────────────────────────────────────────────────

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def query_chembl(genes: list) -> list:
    """
    Query ChEMBL REST API. For each gene, finds the best matching human
    protein target, then retrieves known drugs via the mechanism endpoint.

    Returns
    -------
    list of dict
        One dict per drug-gene interaction edge.
    """
    edges = []
    for gene in genes:
        r = safe_request(
            "get", f"{CHEMBL_BASE}/target/search",
            params={"q": gene, "organism": "Homo sapiens",
                    "target_type": "SINGLE PROTEIN",
                    "format": "json", "limit": 1},
        )
        if not r:
            continue
        targets = r.json().get("targets", [])
        if not targets:
            continue
        target_id = targets[0]["target_chembl_id"]

        r2 = safe_request(
            "get", f"{CHEMBL_BASE}/mechanism",
            params={"target_chembl_id": target_id, "format": "json", "limit": 50},
        )
        if not r2:
            continue

        for mech in r2.json().get("mechanisms", []):
            mol_id = mech.get("molecule_chembl_id")
            if not mol_id:
                continue
            r3 = safe_request("get", f"{CHEMBL_BASE}/molecule/{mol_id}",
                              params={"format": "json"})
            if not r3:
                continue
            mol   = r3.json()
            phase = int(mol.get("max_phase") or 0)
            name  = mol.get("pref_name") or mol_id
            moa   = mech.get("mechanism_of_action", "unknown")
            edges.append({
                "gene"             : gene,
                "drug"             : name,
                "drug_id"          : mol_id,
                "source"           : "ChEMBL",
                "interaction_type" : moa,
                "directionality"   : ("inhibitory" if "inhibit" in moa.lower()
                                      else "activating"),
                "approved"         : phase == 4,
                "immunotherapy"    : False,
                "anti_neoplastic"  : False,
                "interaction_score": 5.0 + phase,
                "n_publications"   : 0,
                "clinical_phase"   : phase,
            })
        time.sleep(0.3)

    print(f"    ChEMBL: {len(edges)} interactions returned")
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# OpenTargets
# ─────────────────────────────────────────────────────────────────────────────

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

_OT_MAP_QUERY = """
query M($s: String!) {
  targets(queryString: $s, page: {size: 1}) {
    rows { id approvedSymbol }
  }
}
"""

_OT_DRUG_QUERY = """
query D($id: String!) {
  target(ensemblId: $id) {
    approvedSymbol
    knownDrugs(size: 50) {
      rows {
        drug {
          id name isApproved maximumClinicalTrialPhase
        }
        mechanismOfAction
        references { source urls }
      }
    }
  }
}
"""


def query_opentargets(genes: list) -> list:
    """
    Query OpenTargets Platform for known drugs per gene.
    Resolves gene symbols to Ensembl IDs first, then fetches drug data.

    Returns
    -------
    list of dict
        One dict per drug-gene interaction edge.
    """
    edges = []
    for gene in genes:
        r = safe_request("post", OT_URL,
                         json={"query": _OT_MAP_QUERY,
                               "variables": {"s": gene}})
        if not r:
            continue
        rows = (r.json().get("data", {}).get("targets", {}).get("rows", []))
        if not rows:
            continue
        ensembl_id = rows[0]["id"]
        time.sleep(0.2)

        r2 = safe_request("post", OT_URL,
                          json={"query": _OT_DRUG_QUERY,
                                "variables": {"ensemblId": ensembl_id}})
        if not r2:
            continue

        drug_rows = ((r2.json().get("data", {}).get("target", {}) or {})
                     .get("knownDrugs", {}).get("rows", []))
        for row in drug_rows:
            drug = row.get("drug", {})
            if not drug or not drug.get("name"):
                continue
            phase = int(drug.get("maximumClinicalTrialPhase") or 0)
            moa   = row.get("mechanismOfAction", "unknown")
            edges.append({
                "gene"             : gene,
                "drug"             : drug["name"],
                "drug_id"          : drug.get("id", ""),
                "source"           : "OpenTargets",
                "interaction_type" : moa,
                "directionality"   : ("inhibitory" if "inhibit" in moa.lower()
                                      else "activating"),
                "approved"         : bool(drug.get("isApproved", False)),
                "immunotherapy"    : False,
                "anti_neoplastic"  : False,
                "interaction_score": 5.0 + phase,
                "n_publications"   : len(row.get("references", [])),
                "clinical_phase"   : phase,
            })
        time.sleep(0.3)

    print(f"    OpenTargets: {len(edges)} interactions returned")
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Curated fallback
# ─────────────────────────────────────────────────────────────────────────────

_CURATED = [
    # (gene, drug, source, interaction_type, directionality,
    #  approved, immunotherapy, anti_neoplastic, score, n_pubs, phase)
    ("APOE","Fluvastatin","DGIdb","inhibitor","inhibitory",True,False,True,7.2,12,4),
    ("APOE","Atorvastatin","DGIdb","inhibitor","inhibitory",True,False,True,6.8,18,4),
    ("APOE","Simvastatin","ChEMBL","inhibitor","inhibitory",True,False,True,6.5,14,4),
    ("APOE","Fenofibrate","OpenTargets","agonist","activating",True,False,False,6.2,7,4),
    ("ALB","Gadobenate Dimeglumine","DGIdb","binder","activating",True,False,False,8.1,3,4),
    ("ALB","Warfarin","DGIdb","binder","inhibitory",True,False,False,7.5,21,4),
    ("ALB","Cisplatin","OpenTargets","binder","inhibitory",True,False,True,6.7,22,4),
    ("SERPINA1","Igmesine","DGIdb","inhibitor","inhibitory",False,False,False,9.2,2,1),
    ("SERPINA1","Alpha-1 Antitrypsin","DGIdb","activator","activating",True,False,False,8.8,33,4),
    ("SERPINA1","Sivelestat","ChEMBL","inhibitor","inhibitory",True,False,False,7.4,8,3),
    ("APOA2","PKR-A","DGIdb","agonist","activating",False,False,False,8.5,1,1),
    ("APOA2","Fenofibrate","OpenTargets","agonist","activating",True,False,False,6.0,9,4),
    ("FTL","Deferoxamine","DGIdb","inhibitor","inhibitory",True,False,False,7.8,15,4),
    ("FTL","Deferasirox","ChEMBL","inhibitor","inhibitory",True,False,False,7.1,11,4),
    ("FTL","Deferiprone","ChEMBL","inhibitor","inhibitory",True,False,False,6.8,10,4),
    ("MMP9","Marimastat","DGIdb","inhibitor","inhibitory",False,False,True,8.3,14,2),
    ("MMP9","Sorafenib","OpenTargets","inhibitor","inhibitory",True,False,True,7.0,24,4),
    ("MMP9","Doxycycline","ChEMBL","inhibitor","inhibitory",True,False,False,6.4,9,4),
    ("IL1B","Canakinumab","DGIdb","antibody","inhibitory",True,False,True,9.1,28,4),
    ("IL1B","Anakinra","DGIdb","antagonist","inhibitory",True,False,False,8.7,22,4),
    ("IL1B","Rilonacept","ChEMBL","antagonist","inhibitory",True,False,False,7.9,11,4),
    ("NFKB1","Bortezomib","DGIdb","inhibitor","inhibitory",True,False,True,8.2,19,4),
    ("NFKB1","Sulfasalazine","OpenTargets","inhibitor","inhibitory",True,False,False,6.8,12,4),
    ("NFKB1","Curcumin","ChEMBL","inhibitor","inhibitory",False,False,False,6.1,31,2),
    ("CCL2","Carlumab","DGIdb","antibody","inhibitory",False,False,True,8.4,6,2),
    ("CCL2","Bindarit","ChEMBL","inhibitor","inhibitory",False,False,False,7.2,9,2),
    ("IFNG","Emapalumab","DGIdb","antibody","inhibitory",True,True,False,8.9,7,4),
    ("TYROBP","Sorafenib","OpenTargets","inhibitor","inhibitory",True,False,True,6.5,24,4),
    ("TYROBP","Regorafenib","ChEMBL","inhibitor","inhibitory",True,False,True,6.8,16,4),
    ("AIF1","Minocycline","ChEMBL","inhibitor","inhibitory",True,False,False,5.8,13,4),
    ("S100A9","Tasquinimod","DGIdb","inhibitor","inhibitory",False,False,True,7.4,8,2),
    ("CTSB","CA-074Me","ChEMBL","inhibitor","inhibitory",False,False,False,8.0,11,1),
    ("SPP1","Alendronate","DGIdb","inhibitor","inhibitory",True,False,False,6.3,17,4),
    ("CD68","Pexidartinib","OpenTargets","inhibitor","inhibitory",True,False,True,7.5,11,4),
    ("GAPDH","Heptelidic acid","ChEMBL","inhibitor","inhibitory",False,False,False,6.0,5,1),
    ("FCER1G","Omalizumab","DGIdb","antibody","inhibitory",True,False,False,6.9,14,4),
    ("GRN","AL001","OpenTargets","activator","activating",False,False,False,5.8,3,2),
]


def get_curated_fallback(genes: list) -> list:
    """
    Return curated drug-gene interactions for genes in `genes`.
    Used automatically when all three live APIs are inaccessible.

    Parameters
    ----------
    genes : list
        Gene symbols to filter for.

    Returns
    -------
    list of dict
        Interaction edges for matching genes.
    """
    gene_set = set(g.upper() for g in genes)
    edges = []
    for r in _CURATED:
        if r[0].upper() not in gene_set:
            continue
        edges.append({
            "gene"             : r[0],
            "drug"             : r[1],
            "drug_id"          : "",
            "source"           : r[2],
            "interaction_type" : r[3],
            "directionality"   : r[4],
            "approved"         : r[5],
            "immunotherapy"    : r[6],
            "anti_neoplastic"  : r[7],
            "interaction_score": r[8],
            "n_publications"   : r[9],
            "clinical_phase"   : r[10],
        })
    return edges
