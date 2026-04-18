"""
utils — shared helpers for the HCC drug discovery pipeline.

Modules
-------
graph_utils   — PPI graph construction + GNN graph building
plot_utils    — all matplotlib figure functions
api_clients   — DGIdb / ChEMBL / OpenTargets HTTP clients
"""

from .graph_utils import (
    build_ppi_graph,
    compute_hub_scores,
    build_gnn_graph,
    edge_tensors,
)
from .plot_utils import (
    plot_ppi_network,
    plot_km_grid,
    plot_cox_forest,
    plot_drug_ranking,
    plot_training_curves,
    plot_model_comparison,
    plot_scatter,
)
from .api_clients import (
    safe_request,
    query_dgidb,
    query_chembl,
    query_opentargets,
    get_curated_fallback,
)
from .scrna_functions import (
    load_samples,
    qc_metrics,
    filter_cells,
    normalize,
    select_hvg,
    save_adata,
    run_pca,
    run_umap,
    run_leiden,
    run_celltypist,
    prep_seurat_object,
    pull_r_col,
    marker_score_clusters,
    majority_vote,
)
from .ppi_analysis import (
    _find_repo_root,
    norm_width,
    norm_alpha,
)
from .gsea_functions import (
    prepare_ranked_list,
    run_gsea_r,
    print_gsea_summary,
)
from .gsea_analysis import (
    _find_repo_root,
    prepare_ranked_list,
    run_gsea_in_r,
    print_summary,
)
from .dea_functions import (
    run_wilcoxon,
    plot_volcano,
    export_dea,
)

__all__ = [
    "build_ppi_graph", "compute_hub_scores", "build_gnn_graph", "edge_tensors",
    "plot_ppi_network", "plot_km_grid", "plot_cox_forest", "plot_drug_ranking",
    "plot_training_curves", "plot_model_comparison", "plot_scatter",
    "safe_request", "query_dgidb", "query_chembl", "query_opentargets",
    "get_curated_fallback",
    "load_samples", "qc_metrics", "filter_cells", "normalize", "select_hvg",
    "save_adata", "run_pca", "run_umap", "run_leiden", "run_celltypist",
    "prep_seurat_object", "pull_r_col", "marker_score_clusters",
    "majority_vote", "_find_repo_root", "norm_width", "norm_alpha",
    "plot_ppi_network", "plot_km_grid", "plot_cox_forest", "plot_drug_ranking",
    "plot_training_curves", "plot_model_comparison", "plot_scatter",
    "prepare_ranked_list", "run_gsea_r", "print_gsea_summary",
    "prepare_ranked_list", "run_gsea_in_r", "print_summary",
    "build_ppi_graph", "compute_hub_scores", "build_gnn_graph", "edge_tensors",
    "run_wilcoxon", "plot_volcano", "export_dea",
    "safe_request", "query_dgidb", "query_chembl", "query_opentargets",
    "get_curated_fallback",
]
