"""Focused tests of sample contrasts; no downloads or changes to project data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "utils" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dea = load_module("dea_functions")
gsea = load_module("gsea_functions")


def fixture():
    rng = np.random.default_rng(7)
    counts = rng.poisson(3, size=(40, 24))
    # Type A has both samples; B and C each lack a comparison sample.
    samples = ["normal"] * 10 + ["tumor"] * 10 + ["tumor"] * 10 + ["normal"] * 10
    types = ["A"] * 20 + ["B"] * 10 + ["C"] * 10
    counts[np.array(samples) == "tumor", 0] += 40
    counts[np.array(samples) == "normal", 1] += 40
    counts[:, -1] = 0
    logcounts = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10000)
    data = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"sample": samples, "manual_celltype": types,
                          "cluster": ["0"] * 20 + ["1"] * 10 + ["2"] * 10},
                         index=[f"cell{i}" for i in range(40)]),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(24)]),
    )
    data.layers["counts"] = data.X.copy()
    data.layers["logcounts"] = sparse.csr_matrix(logcounts)
    data.uns["log1p"] = {"base": None}
    data.uns["rank_genes_groups"] = {"original": True}
    return data


class SampleContrastTests(unittest.TestCase):
    def test_direction_all_genes_and_preserved_object(self):
        data = fixture()
        original = data.X.copy()
        sig, full = dea.run_wilcoxon(data, group="tumor", reference="normal")
        indexed = full.set_index("gene")
        self.assertGreater(indexed.loc["gene0", "log2FC"], 0)
        self.assertGreater(indexed.loc["gene0", "scores"], 0)
        self.assertLess(indexed.loc["gene1", "scores"], 0)
        self.assertEqual(len(full), 23)  # all-zero gene omitted, no top-N cap
        self.assertTrue({"pct_nz_group", "pct_nz_reference"}.issubset(full.columns))
        self.assertEqual((data.X != original).nnz, 0)
        self.assertEqual(data.uns["rank_genes_groups"], {"original": True})

    def test_wrapper_skips_absent_groups_and_isolates_exports(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = dea.run_sample_contrasts(
                data, root / "processed", root / "figures",
                group="tumor", reference="normal", min_cells=5,
            )
            self.assertEqual(set(analysis["results"]), {"pooled", "celltype_A"})
            summary = analysis["summary_df"].set_index("contrast")
            self.assertEqual(summary.loc["celltype_B", "status"], "skipped")
            self.assertEqual(summary.loc["celltype_C", "status"], "skipped")
            self.assertTrue((root / "processed/pooled/de_all_genes.csv").exists())
            self.assertFalse((root / "processed/dea_results.csv").exists())
            excluded = dea.run_sample_contrasts(
                data, root / "sensitivity", root / "sensitivity_figures",
                group="tumor", reference="normal", min_cells=5,
                exclude_clusters=("0",), cluster_col="cluster",
            )
            self.assertEqual(set(excluded["results"]), {"pooled"})
            self.assertEqual(excluded["provenance"]["excluded_clusters"], ["0"])

    def test_full_ranking_keeps_nonsignificant_and_both_directions(self):
        full = pd.DataFrame({"gene": ["A", "B", "C"], "scores": [-2, 0, 3],
                             "adj_pvalue": [0.8, 1, 0.01]})
        with tempfile.TemporaryDirectory() as temporary:
            ranked = gsea.prepare_ranked_list(full, Path(temporary))
            self.assertEqual(ranked["gene"].tolist(), ["C", "B", "A"])
            self.assertEqual(len(ranked), 3)
            self.assertTrue((Path(temporary) / "ranked_genes.tsv").exists())
            with self.assertRaises(ValueError):
                gsea.prepare_ranked_list(pd.concat([full, full]), Path(temporary))

    def test_failure_status_prevents_false_completion(self):
        class BrokenR:
            globalenv = {}
            def r(self, script):
                raise RuntimeError("fixture failure")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = {"results": {"pooled": {
                "all_genes": pd.DataFrame({"gene": ["A", "B"], "scores": [-1, 1]}),
                "proc_dir": root / "proc", "figures_dir": root / "figures",
            }}}
            summary = gsea.run_contrast_gsea(BrokenR(), analysis, root / "tables")
            self.assertEqual(summary.iloc[0]["status"], "failed")
            self.assertEqual(analysis["results"]["pooled"]["gsea_status"], "failed")
            status = json.loads((root / "tables/pooled/gsea_status.json").read_text())
            self.assertEqual(status["status"], "failed")

    def test_leading_edge_symbols_not_statistics(self):
        table = pd.DataFrame({"ID": ["GO:test"], "Description": ["lipid metabolism"],
                              "NES": [-2.0], "p.adjust": [0.01],
                              "leading_edge": ["tags=20%, list=10%, signal=25%"],
                              "core_enrichment": ["1/2"],
                              "core_enrichment_symbols": ["ALB/APOE"]})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table.to_csv(root / "gsea_go_bp.csv", index=False)
            result = gsea.query_gene_pathways("ALB", root)
            self.assertEqual(len(result), 1)
            summary, _ = gsea.generate_pathway_summary_table(root, ontologies=("GO-BP",))
            self.assertIn("ALB", summary.iloc[0]["Key genes"])
            self.assertNotIn("tags=", summary.iloc[0]["Key genes"])


if __name__ == "__main__":
    unittest.main()
