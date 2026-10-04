"""
ppi_functions.py
================
All logic for notebook P1 · PPI Network Analysis.

Functions
---------
load_dea            — load & filter DEA results
query_string        — map identifiers and query all pairs of STRING blocks
build_and_score     — build NetworkX graph + compute hub scores
export_ppi          — save hub_genes.csv and Cytoscape files
"""

import time
import requests
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations_with_replacement
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
def load_dea(dea_path, log2fc_thresh=1.0, padj_thresh=0.05):
    """
    Load DEA results CSV and filter to significant DEGs.

    Parameters
    ----------
    dea_path : Path
        Path to dea_results.csv (columns: gene, log2FC, adj_pvalue).
    log2fc_thresh : float
        Minimum absolute log2 fold change.
    padj_thresh : float
        Maximum adjusted p-value.

    Returns
    -------
    sig : pd.DataFrame
        Filtered DEGs with added 'regulation' column (up/down).
    gene_list : list
        Unique gene symbols, used for STRING query.
    """
    dea = pd.read_csv(dea_path)
    required = {"gene", "log2FC", "adj_pvalue"}
    if not required.issubset(dea.columns):
        raise ValueError(f"DE input requires {sorted(required)}")
    if dea.gene.isna().any() or dea.gene.duplicated().any():
        raise ValueError("DE input must contain unique, nonmissing gene identifiers")
    sig = dea[
        (dea.adj_pvalue < padj_thresh) &
        (dea.log2FC.abs() >= log2fc_thresh)
    ].copy()
    sig["regulation"] = (sig.log2FC > 0).map({True: "up", False: "down"})
    gene_list = sig.gene.dropna().unique().tolist()

    print(f"DEGs loaded    : {len(sig)}")
    print(f"  Upregulated  : {(sig.regulation=='up').sum()}")
    print(f"  Downregulated: {(sig.regulation=='down').sum()}")
    print(f"  Unique genes : {len(gene_list)}")
    return sig, gene_list


# ─────────────────────────────────────────────────────────────────────────────
def query_string(gene_list, string_score=400, batch_size=500,
                 api_url="https://version-12-5.string-db.org/api", request_pause=1.0):
    """
    Query the STRING API for protein-protein interactions.

    Map identifiers in blocks, then query unions of every pair of blocks.
    Cross-block associations are preserved. HTTP/schema failures stop the
    analysis rather than returning a partial network. Ambiguous/unmapped
    genes remain in the exported mapping audit and candidate ledger.

    Parameters
    ----------
    gene_list : list
        Gene symbols to query.
    string_score : int
        Minimum combined score (400=medium, 700=high confidence).
    batch_size : int
        Mapping block size; network requests contain at most two blocks.

    Returns
    -------
    edges_df : pd.DataFrame
        Columns: gene_A, gene_B, combined_score.
        Self-loops and duplicates removed.
    """
    # Query every union of two blocks, so cross-block associations are retained.
    # Map to stable protein IDs first and export ambiguous/unmapped identifiers.
    if batch_size < 1 or not 0 <= string_score <= 1000:
        raise ValueError("Invalid batch size or STRING score")
    genes = list(dict.fromkeys(gene_list))
    mapping_rows = []
    for start in range(0, len(genes), batch_size):
        batch = genes[start:start + batch_size]
        r = requests.post(api_url + "/json/get_string_ids", data={
            "identifiers": "\r".join(batch), "species": 9606,
            "echo_query": 1, "caller_identity": "hcc_pipeline"}, timeout=120)
        r.raise_for_status()
        records = r.json()
        if not isinstance(records, list):
            raise RuntimeError("STRING mapping request failed")
        for record in records:
            gene = record.get("queryItem")
            if gene is None:
                gene = batch[int(record["queryIndex"])]
            if gene not in batch or not record.get("stringId"):
                raise RuntimeError("Malformed STRING mapping response")
            mapping_rows.append(dict(gene=gene, string_id=record["stringId"],
                                     preferred_name=record.get("preferredName", "")))
        time.sleep(request_pause)
    mapping = pd.DataFrame(mapping_rows, columns=["gene", "string_id", "preferred_name"]).drop_duplicates()
    mapping["mapping_status"] = "mapped"
    mapping.loc[mapping.gene.duplicated(False), "mapping_status"] = "ambiguous"
    mapping.loc[mapping.string_id.duplicated(False), "mapping_status"] = "many_to_one"
    audit = pd.DataFrame({"gene": genes}).merge(mapping, on="gene", how="left")
    audit["mapping_status"] = audit.mapping_status.fillna("unmapped")
    valid = mapping[mapping.mapping_status == "mapped"]
    id_to_gene = valid.set_index("string_id").gene.to_dict()
    protein_ids = list(id_to_gene)
    blocks = [protein_ids[i:i + batch_size] for i in range(0, len(protein_ids), batch_size)]
    STRING_URL = api_url + "/json/network"
    all_edges  = []

    for left, right in combinations_with_replacement(range(len(blocks)), 2):
        batch = list(dict.fromkeys(blocks[left] + blocks[right]))
        if len(batch) < 2:
            continue
        print(f"  Blocks {left + 1}/{right + 1}: {len(batch)} proteins...", end=" ")
        try:
            r = requests.post(STRING_URL, data={
                "identifiers"    : "\r".join(batch),
                "species"        : 9606,
                "required_score" : string_score,
                "network_type"   : "functional",
                "add_nodes"      : 0,
                "caller_identity": "hcc_pipeline",
            }, timeout=120)
            r.raise_for_status()
            batch_edges = r.json()
            if not isinstance(batch_edges, list):
                raise RuntimeError("STRING returned an error response")
            for edge in batch_edges:
                edge["preferredName_A"] = id_to_gene.get(edge["stringId_A"])
                edge["preferredName_B"] = id_to_gene.get(edge["stringId_B"])
            all_edges.extend(batch_edges)
            print(f"{len(batch_edges)} interactions")
        except Exception as e:
            raise RuntimeError("STRING failed; refusing to rank a partial network") from e
        time.sleep(request_pause)

    if not all_edges:
        edges = pd.DataFrame(columns=["gene_A", "gene_B", "combined_score"])
        edges.attrs.update(mapping_audit=audit, api_url=api_url, network_type="functional")
        return edges

    keep     = {"preferredName_A": "gene_A",
                "preferredName_B": "gene_B",
                "score"          : "combined_score"}
    edges_df = pd.DataFrame(all_edges).rename(columns=keep)[list(keep.values())]
    edges_df = edges_df.dropna(subset=["gene_A", "gene_B"])
    edges_df["combined_score"] = pd.to_numeric(edges_df.combined_score, errors="raise")
    if not edges_df.combined_score.between(0, 1).all():
        raise ValueError("STRING confidence must be between zero and one")
    edges_df = edges_df[edges_df.combined_score >= string_score / 1000].copy()
    edges_df = edges_df[edges_df.gene_A != edges_df.gene_B]
    edges_df["pair"] = edges_df.apply(
        lambda r: tuple(sorted([r.gene_A, r.gene_B])), axis=1)
    edges_df = edges_df.sort_values("combined_score", ascending=False).drop_duplicates("pair").drop(columns="pair")
    edges_df.attrs.update(mapping_audit=audit, api_url=api_url, network_type="functional")
    edges_df["combined_score"] = pd.to_numeric(
        edges_df.combined_score, errors="coerce")

    print(f"\nUnique interactions: {len(edges_df)}")
    return edges_df


# ─────────────────────────────────────────────────────────────────────────────
def build_and_score(sig, edges_df):
    """
    Build a NetworkX PPI graph and compute a composite hub score per gene.

    Hub score = normalized mean of unweighted degree, betweenness and
    closeness, plus confidence-weighted eigenvector centrality computed per
    component and scaled by component size. Isolates remain with zero hub
    scores. Confidence is not a shortest-path distance. This is a heuristic
    within the submitted gene set, not evidence of target causality.

    Parameters
    ----------
    sig : pd.DataFrame
        Significant DEGs (gene, log2FC, adj_pvalue, regulation).
    edges_df : pd.DataFrame
        STRING edge list (gene_A, gene_B, combined_score).

    Returns
    -------
    G : nx.Graph
        PPI graph with node and edge attributes.
    hub_df : pd.DataFrame
        Per-gene centrality + hub score, sorted descending.
    """
    # Build graph
    G = nx.Graph()
    for _, row in sig.iterrows():
        G.add_node(row.gene, log2FC=row.log2FC,
                   adj_pvalue=row.adj_pvalue, regulation=row.regulation)
    for _, row in edges_df.iterrows():
        if row.gene_A in G and row.gene_B in G and row.gene_A != row.gene_B:
            G.add_edge(row.gene_A, row.gene_B,
                       weight=float(row.combined_score))
    isolates = list(nx.isolates(G))
    # Preserve all candidate genes, including isolates, for drug queries.

    # Centrality
    deg_c = nx.degree_centrality(G)
    bet_c = nx.betweenness_centrality(G, weight=None)
    clo_c = nx.closeness_centrality(G)
    eig_c = dict.fromkeys(G, 0.0)
    for nodes in nx.connected_components(G):
        if len(nodes) > 1:
            values = nx.eigenvector_centrality(G.subgraph(nodes), max_iter=5000,
                                             tol=1e-7, weight="weight")
            # Component-wise centrality, scaled by component size.
            eig_c.update({n: v * len(nodes) / len(G) for n, v in values.items()})

    hub_df = pd.DataFrame({
        "gene"       : list(G.nodes()),
        "degree"     : [G.degree(n) for n in G.nodes()],
        "deg_c"      : [deg_c[n]  for n in G.nodes()],
        "betweenness": [bet_c[n]  for n in G.nodes()],
        "closeness"  : [clo_c[n]  for n in G.nodes()],
        "eigenvector": [eig_c[n]  for n in G.nodes()],
    })
    for col in ["deg_c", "betweenness", "closeness", "eigenvector"]:
        mn, mx = hub_df[col].min(), hub_df[col].max()
        hub_df[f"{col}_n"] = (hub_df[col] - mn) / (mx - mn + 1e-9)

    hub_df["hub_score"] = hub_df[
        [c for c in hub_df.columns if c.endswith("_n")]
    ].mean(axis=1)
    hub_df["ppi_connected"] = hub_df.degree > 0
    hub_df.loc[hub_df.gene.isin(isolates), "hub_score"] = 0.0
    hub_df = hub_df.merge(
        sig,
        on="gene", how="left"
    ).sort_values(["hub_score", "gene"], ascending=[False, True]).reset_index(drop=True)

    print(f"Nodes : {G.number_of_nodes()}")
    print(f"Edges : {G.number_of_edges()}")
    print(f"Isolates retained: {len(isolates)}")
    return G, hub_df


# ─────────────────────────────────────────────────────────────────────────────
def export_ppi(hub_df, G, edges_df, tables_dir):
    """
    Save hub_genes.csv and ppi_edges_cytoscape.csv.

    Parameters
    ----------
    hub_df : pd.DataFrame
        Hub gene ranking from build_and_score().
    G : nx.Graph
        PPI graph (used to filter edges to connected nodes only).
    edges_df : pd.DataFrame
        STRING edge list (gene_A, gene_B, combined_score).
    tables_dir : Path
        Output directory.
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    hub_df.to_csv(
        tables_dir / "hub_genes.csv", index=False)
    edges_df.to_csv(tables_dir / "string_edges.csv", index=False)
    if "mapping_audit" in edges_df.attrs:
        edges_df.attrs["mapping_audit"].to_csv(tables_dir / "string_mapping.csv", index=False)

    cyto = edges_df[
        edges_df.gene_A.isin(G.nodes()) & edges_df.gene_B.isin(G.nodes())
    ].copy()
    cyto.rename(columns={"gene_A": "source", "gene_B": "target",
                         "combined_score": "STRING_score"}, inplace=True)
    cyto.to_csv(tables_dir / "ppi_edges_cytoscape.csv", index=False)

    print(f"Saved: hub_genes.csv            ({len(hub_df)} genes)")
    print(f"Saved: ppi_edges_cytoscape.csv  ({len(cyto)} edges)")
    print(f"\nTop 5 hub genes: {hub_df.gene.head().tolist()}")
