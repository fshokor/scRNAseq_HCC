"""
ppi_functions.py
================
All logic for notebook P1 · PPI Network Analysis.

Functions
---------
load_dea            — load & filter DEA results
query_string        — batch query STRING API
build_and_score     — build NetworkX graph + compute hub scores
export_ppi          — save hub_genes.csv and Cytoscape files
"""

import time
import requests
import numpy as np
import pandas as pd
import networkx as nx


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
def query_string(gene_list, string_score=400, batch_size=500):
    """
    Query the STRING API for protein-protein interactions.

    Sends genes in batches to avoid URL length limits.
    Automatically deduplicates and returns a clean edge dataframe.

    Parameters
    ----------
    gene_list : list
        Gene symbols to query.
    string_score : int
        Minimum combined score (400=medium, 700=high confidence).
    batch_size : int
        Genes per API request.

    Returns
    -------
    edges_df : pd.DataFrame
        Columns: gene_A, gene_B, combined_score.
        Self-loops and duplicates removed.
    """
    STRING_URL = "https://string-db.org/api/json/network"
    all_edges  = []

    for i in range(0, len(gene_list), batch_size):
        batch = gene_list[i: i + batch_size]
        print(f"  Batch {i//batch_size+1}: {len(batch)} genes...", end=" ")
        try:
            r = requests.post(STRING_URL, data={
                "identifiers"    : "\r".join(batch),
                "species"        : 9606,
                "required_score" : string_score,
                "network_type"   : "functional",
                "caller_identity": "hcc_pipeline",
            }, timeout=60)
            r.raise_for_status()
            batch_edges = r.json()
            all_edges.extend(batch_edges)
            print(f"→ {len(batch_edges)} interactions")
        except Exception as e:
            print(f"✗ {e}")
        time.sleep(1)

    if not all_edges:
        print("  No interactions returned. Check gene names and internet connection.")
        return pd.DataFrame(columns=["gene_A", "gene_B", "combined_score"])

    keep     = {"preferredName_A": "gene_A",
                "preferredName_B": "gene_B",
                "score"          : "combined_score"}
    edges_df = pd.DataFrame(all_edges).rename(columns=keep)[list(keep.values())]
    edges_df = edges_df[edges_df.gene_A != edges_df.gene_B]
    edges_df["pair"] = edges_df.apply(
        lambda r: tuple(sorted([r.gene_A, r.gene_B])), axis=1)
    edges_df = edges_df.drop_duplicates("pair").drop(columns="pair")
    edges_df["combined_score"] = pd.to_numeric(
        edges_df.combined_score, errors="coerce")

    print(f"\nUnique interactions: {len(edges_df)}")
    return edges_df


# ─────────────────────────────────────────────────────────────────────────────
def build_and_score(sig, edges_df):
    """
    Build a NetworkX PPI graph and compute a composite hub score per gene.

    Hub score = normalised mean of four centrality measures:
    degree, betweenness, closeness, eigenvector.

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
        if row.gene_A in G and row.gene_B in G:
            G.add_edge(row.gene_A, row.gene_B,
                       weight=float(row.combined_score))
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    # Centrality
    deg_c = nx.degree_centrality(G)
    bet_c = nx.betweenness_centrality(G, weight="weight")
    clo_c = nx.closeness_centrality(G)
    eig_c = nx.eigenvector_centrality(G, max_iter=500, weight="weight")

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
    hub_df = hub_df.merge(
        sig[["gene", "log2FC", "adj_pvalue", "regulation"]],
        on="gene", how="left"
    ).sort_values("hub_score", ascending=False).reset_index(drop=True)

    print(f"Nodes : {G.number_of_nodes()}")
    print(f"Edges : {G.number_of_edges()}")
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
    hub_cols = ["gene", "degree", "hub_score", "deg_c", "betweenness",
                "closeness", "eigenvector", "log2FC", "adj_pvalue", "regulation"]
    hub_df[[c for c in hub_cols if c in hub_df.columns]].to_csv(
        tables_dir / "hub_genes.csv", index=False)

    cyto = edges_df[
        edges_df.gene_A.isin(G.nodes()) & edges_df.gene_B.isin(G.nodes())
    ].copy()
    cyto.rename(columns={"gene_A": "source", "gene_B": "target",
                         "combined_score": "STRING_score"}, inplace=True)
    cyto.to_csv(tables_dir / "ppi_edges_cytoscape.csv", index=False)

    print(f"Saved: hub_genes.csv            ({len(hub_df)} genes)")
    print(f"Saved: ppi_edges_cytoscape.csv  ({len(cyto)} edges)")
    print(f"\nTop 5 hub genes: {hub_df.gene.head().tolist()}")
