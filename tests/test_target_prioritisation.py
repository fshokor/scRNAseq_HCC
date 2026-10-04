"""Regression checks for scientific failure modes; all HTTP is mocked."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# The package initializer eagerly loads scRNA/GNN dependencies unrelated to
# these checks. Import the real target modules without that heavyweight setup.
package = types.ModuleType("utils")
package.__path__ = [str(ROOT / "scripts/utils")]
sys.modules.setdefault("utils", package)
from utils import ppi_functions as ppi, dgi_functions as dgi
from utils import survival_functions as survival, api_clients as api

W = dict(interaction=0.65, publications=0, phase=0, approved=0.15, hub=0.20)


class Response:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self.payload


def string_response(url, data, **kwargs):
    identifiers = data["identifiers"].split("\r")
    if url.endswith("get_string_ids"):
        return Response([dict(queryItem=g, stringId="9606." + g, preferredName=g) for g in identifiers if g != "UNMAPPED"])
    pairs = [("A", "C", 0.9), ("C", "D", 0.5)]
    return Response([dict(stringId_A="9606." + a, stringId_B="9606." + b, score=s)
        for a, b, s in pairs if "9606." + a in identifiers and "9606." + b in identifiers])


def candidates(genes=("A", "B", "C", "D", "UNMAPPED")):
    return pd.DataFrame(dict(gene=genes, log2FC=[2.] * len(genes),
        adj_pvalue=[0.01] * len(genes), regulation=["up"] * len(genes),
        contrast=["pooled"] * len(genes), cell_type=["All cells"] * len(genes)))


def drug_records():
    return [dict(gene="A", drug="Drug one", drug_id="id:1", source="DGIdb",
                 interaction_type="inhibitor", directionality="inhibitory",
                 approved=True, interaction_score=3, clinical_phase=None,
                 publication_ids="1 | 2", evidence_sources="DB1", n_publications=2),
            dict(gene="A", drug="Drug one alias", drug_id="id:1", source="DGIdb",
                 interaction_type="binder", directionality="unknown",
                 approved=True, interaction_score=4, clinical_phase=None,
                 publication_ids="2 | 3", evidence_sources="DB2", n_publications=2),
            dict(gene="C", drug="Drug two", drug_id="id:2", source="DGIdb",
                 interaction_type="agonist", approved=False, interaction_score=2)]


class TargetChecks(unittest.TestCase):
    def test_cross_batch_edges_mapping_and_isolates(self):
        with patch.object(ppi.requests, "post", side_effect=string_response):
            edges = ppi.query_string(candidates().gene.tolist(), batch_size=2, request_pause=0)
        self.assertEqual(len(edges), 2)
        self.assertIn("unmapped", edges.attrs["mapping_audit"].mapping_status.tolist())
        graph, hubs = ppi.build_and_score(candidates(), edges)
        self.assertEqual(set(graph), set(candidates().gene))
        self.assertEqual(hubs.set_index("gene").loc["B", "hub_score"], 0)
        self.assertGreater(hubs.set_index("gene").loc["C", "betweenness"], 0)

    def test_network_failure_cannot_return_partial_edges(self):
        def response(url, data, **kwargs):
            if url.endswith("network"):
                raise requests.ConnectionError("offline")
            return string_response(url, data, **kwargs)
        with patch.object(ppi.requests, "post", side_effect=response):
            with self.assertRaises(RuntimeError):
                ppi.query_string(["A", "C"], request_pause=0)

    def test_empty_network_and_single_isolate(self):
        graph, hubs = ppi.build_and_score(candidates(("A",)), pd.DataFrame(columns=["gene_A", "gene_B", "combined_score"]))
        self.assertEqual(len(graph), 1)
        self.assertEqual(hubs.hub_score.iloc[0], 0)
        from utils.plot_utils import plot_ppi_network
        fig, _ = plot_ppi_network(graph, hubs)
        plt.close(fig)

    def test_dgidb_preserves_citations_types_and_unknown_phase(self):
        payload = dict(data=dict(genes=dict(nodes=[dict(name="A", interactions=[dict(
            interactionScore=3, interactionTypes=[dict(type="inhibitor", directionality="inhibitory"),
                dict(type="binder", directionality=None)], publications=[dict(pmid=1), dict(pmid=2)],
            sources=[dict(fullName="Original DB")], drug=dict(name="Drug", conceptId="X", approved=True))])])))
        with patch.object(api, "safe_request", return_value=Response(payload)), patch.object(api.time, "sleep"):
            records = api.query_dgidb(["A"])
        self.assertEqual(records[0]["publication_ids"], "1 | 2")
        self.assertIn("binder", records[0]["interaction_type"])
        self.assertIsNone(records[0]["clinical_phase"])
        with patch.object(api, "safe_request", return_value=Response({"errors": [{"message": "bad"}]})):
            with self.assertRaises(RuntimeError):
                api.query_dgidb(["A"])

    def test_drug_dedup_merges_evidence_and_hub_weight_is_not_shrunk(self):
        frame = dgi.build_dgi_dataframe(drug_records(), {"A": 1, "C": 0}, W)
        self.assertEqual(len(frame), 2)
        a = frame.set_index("gene").loc["A"]
        self.assertEqual(a.publication_ids, "1 | 2 | 3")
        self.assertEqual(a.n_publications, 3)
        self.assertIn("binder", a.interaction_type)
        self.assertTrue(pd.isna(a.clinical_phase))
        self.assertAlmostEqual(a.composite_score, 1)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            encoded = dgi.build_gnn_edge_list(frame, {"A": 1}, Path(directory))
            self.assertEqual(encoded.loc[0, "clinical_phase_missing"], 1)
            self.assertEqual(encoded.loc[0, "type_binder"], 1)

    def test_drug_api_failure_and_empty_export(self):
        with patch.object(api, "query_dgidb", side_effect=RuntimeError("offline")):
            records, sources, status = dgi.collect_interactions(["A"], return_status=True)
        self.assertEqual(records, [])
        self.assertEqual(sources, [])
        self.assertEqual(status["DGIdb"]["status"], "failed")
        with self.assertRaises(ValueError):
            dgi.collect_interactions(["A"], use_curated=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            empty = dgi.build_dgi_dataframe([], {}, W)
            dgi.build_gnn_edge_list(empty, {}, Path(directory))

    def test_survival_download_failure_never_simulates(self):
        with patch.object(survival.requests, "get", side_effect=requests.ConnectionError("offline")), patch.object(survival, "simulate_tcga") as simulate:
            merged, is_sim = survival.fetch_tcga_lihc()
        self.assertIsNone(merged)
        self.assertFalse(is_sim)
        simulate.assert_not_called()
        with self.assertRaises(ValueError):
            survival.run_survival(["A"], survival.simulate_tcga(["A"]))

    def test_patient_merge_and_conflicting_endpoints(self):
        ids = [f"TCGA-AA-{i:04d}-01A" for i in range(24)]
        clinical = pd.DataFrame(dict(sampleID=ids, OS=[1] * 24, **{"OS.time": [100] * 24}))
        expression = pd.DataFrame(dict(A=np.arange(24)), index=ids)
        extra = pd.DataFrame(dict(A=[10]), index=[ids[0][:-1] + "B"])
        merged = survival._prepare_tcga(clinical, pd.concat([expression, extra]))
        self.assertEqual(len(merged), 24)
        self.assertEqual(merged.A.iloc[0], 5)
        conflict = clinical.iloc[[0]].copy()
        conflict["OS.time"] = 200
        with self.assertRaises(ValueError):
            survival._prepare_tcga(pd.concat([clinical, conflict]), expression)

    def test_continuous_cox_fdr_adjustment_and_covariates(self):
        rng = np.random.default_rng(3)
        n = 160
        x = rng.normal(size=n)
        data = pd.DataFrame(dict(patient_id=[f"P{i}" for i in range(n)],
            OS_time=rng.exponential(500 * np.exp(-0.6 * x)) + 1,
            OS_event=rng.binomial(1, 0.8, n), A=x, C=rng.normal(size=n),
            Constant=np.ones(n), age=rng.uniform(40, 80, n)))
        frame = survival.run_survival(["A", "C", "Constant", "Missing"], data, covariates=["age"])
        self.assertEqual(frame.set_index("gene").loc["A", "status"], "completed")
        self.assertEqual(frame.set_index("gene").loc["Missing", "status"], "not_tested")
        valid = frame.cox_p.notna()
        self.assertTrue((frame.loc[valid, "cox_p_adj"] >= frame.loc[valid, "cox_p"]).all())
        self.assertTrue(frame.model.eq("adjusted").all())
        # An expression HR close to 1 can still have FDR support; raw KM p is not a gate.
        supported = frame.iloc[[0]].copy()
        supported["HR"], supported["cox_p_adj"], supported["logrank_p"] = 1.1, 0.01, 0.8
        supported["ph_warning"], supported["ph_min_p"] = False, 0.4
        _, subset = survival.filter_survivors(supported, candidates(("A",)))
        self.assertEqual(len(subset), 1)
        duplicate = pd.concat([data, data.iloc[[0]]])
        with self.assertRaises(ValueError):
            survival.run_survival(["A"], duplicate)

    def test_notebook_cells_and_complete_offline_run(self):
        nb = json.loads((ROOT / "notebooks/02_target_prioritisation.ipynb").read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                code = "".join(line for line in cell["source"] if not line.startswith("%"))
                compile(code, f"cell_{i}", "exec")
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            base = Path(directory)
            paths = {name: base / name for name in ["PROC_DIR", "TABLES_DIR", "FIGURES_DIR", "REPORTS_DIR"]}
            de = paths["PROC_DIR"] / "sample_contrasts"
            (de / "pooled").mkdir(parents=True)
            candidates().to_csv(de / "pooled/de_all_genes.csv", index=False)
            pd.DataFrame([dict(contrast="pooled", cell_type="All cells", n_tumor=50,
                n_adjacent=50, status="completed", small_or_imbalanced=False)]).to_csv(de / "contrast_summary.csv", index=False)
            (de / "provenance.json").write_text(json.dumps(dict(group="tumor", reference="normal")))
            scope = {**paths, "Path": Path, "display": lambda *args: None}
            for i in [3, 4]:
                code = "".join(line for line in nb["cells"][i]["source"] if not line.startswith("%"))
                exec(code, scope)
            scope["RUN_SURVIVAL"] = False
            with patch.object(requests, "post", side_effect=string_response), patch.object(api, "query_dgidb", return_value=drug_records()), patch.object(ppi.time, "sleep"), patch.object(plt, "show"):
                for i in range(6, 27):
                    if nb["cells"][i]["cell_type"] == "code":
                        exec("".join(nb["cells"][i]["source"]), scope)
            table_dir = paths["TABLES_DIR"] / "target_prioritisation/pooled"
            exported = pd.read_csv(table_dir / "dgi_edges_gnn.csv")
            self.assertTrue(exported.contrast.eq("pooled").all())
            self.assertTrue((table_dir / "score_weight_sensitivity.csv").exists())
            report = scope["report_path"].read_text(encoding="utf-8")
            self.assertIn("Status: disabled", report)
            self.assertIn("Original evidence", exported.mechanism_review.iloc[0])
            self.assertEqual(json.loads((table_dir / "provenance.json").read_text())["status"], "completed")
            # Run again with unavailable TCGA and drug API; no stale scientific
            # figures or drug evidence should survive the new failed-source run.
            exec("".join(nb["cells"][4]["source"]), scope)
            with patch.object(requests, "post", side_effect=string_response), patch.object(api, "query_dgidb", side_effect=RuntimeError("offline")), patch.object(survival, "fetch_tcga_lihc", return_value=(None, False)), patch.object(ppi.time, "sleep"), patch.object(plt, "show"):
                for i in range(6, 27):
                    if nb["cells"][i]["cell_type"] == "code":
                        exec("".join(nb["cells"][i]["source"]), scope)
            self.assertTrue(pd.read_csv(table_dir / "dgi_edges_gnn.csv").empty)
            self.assertEqual(scope["survival_status"], "unavailable")
            self.assertFalse((scope["TARGET_FIGURES"] / "dgi_summary_dashboard.png").exists())
            self.assertIn("Status: unavailable", scope["report_path"].read_text(encoding="utf-8"))
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
