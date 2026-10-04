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
get_curated_fallback  — retired unverified fallback; raises an error
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
            raise RuntimeError(f"DGIdb batch {i // batch_size + 1} failed; partial results discarded")

        payload = r.json()
        if payload.get("errors") or not payload.get("data", {}).get("genes"):
            raise RuntimeError(f"DGIdb GraphQL error: {payload.get('errors', 'missing gene data')}")
        nodes = payload["data"]["genes"].get("nodes", [])
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
                    "interaction_type" : " | ".join(sorted({t.get("type") or "unknown" for t in itype})) or "unknown",
                    "directionality"   : " | ".join(sorted({t.get("directionality") or "unknown" for t in itype})) or "unknown",
                    "approved"         : drug.get("approved"),
                    "immunotherapy"    : drug.get("immunotherapy"),
                    "anti_neoplastic"  : drug.get("antiNeoplastic"),
                    "publication_ids"  : " | ".join(sorted({str(p["pmid"]) for p in ix.get("publications", []) if p.get("pmid")})),
                    "evidence_sources" : " | ".join(sorted({s["fullName"] for s in ix.get("sources", []) if s.get("fullName")})),
                    "interaction_score": float(ix["interactionScore"]) if ix.get("interactionScore") is not None else None,
                    "n_publications"   : len(ix.get("publications", [])),
                    "clinical_phase"   : None,  # DGIdb does not supply clinical trial phase.
                    "phase_scope"      : "not_available",
                    "mechanism_review" : "Requires review of original interaction evidence",
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
                                      else "activating" if "agonist" in moa.lower() or "activat" in moa.lower() else "unknown"),
                "approved"         : phase == 4,
                "immunotherapy"    : False,
                "anti_neoplastic"  : False,
                "interaction_score": None,
                "phase_scope"      : "drug_global_not_HCC_specific",
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
                                      else "activating" if "agonist" in moa.lower() or "activat" in moa.lower() else "unknown"),
                "approved"         : bool(drug.get("isApproved", False)),
                "immunotherapy"    : False,
                "anti_neoplastic"  : False,
                "interaction_score": None,
                "phase_scope"      : "drug_global_not_HCC_specific",
                "reference_urls"   : " | ".join(sorted({url for ref in row.get("references", []) for url in ref.get("urls", [])})),
                "n_publications"   : len(row.get("references", [])),
                "clinical_phase"   : phase,
            })
        time.sleep(0.3)

    print(f"    OpenTargets: {len(edges)} interactions returned")
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Curated fallback
# ─────────────────────────────────────────────────────────────────────────────

def get_curated_fallback(genes: list) -> list:
    """Retired: previous hardcoded records lacked verifiable citations."""
    raise ValueError("Unverified fallback retired; use original database evidence or independently cited manual records")
