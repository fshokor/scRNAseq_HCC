"""Offline checks of notebook 03's data boundaries and executable workflow."""
import importlib.util
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evidence_gnn_test', ROOT/'scripts/utils/evidence_gnn_functions.py')
gnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gnn)
torch.set_num_threads(1)


def fixture():
    rows = []
    for g in range(6):
        for d in range(10):
            interaction = ((g*7+d*3) % 19)/19
            rows.append(dict(gene=f'G{g}', drug='G0' if d == 0 else f'Drug{d}',
                             drug_id=f'dgidb:{d}', contrast='pooled',
                             hub_score=g/6, log2FC=g/3, approved=d % 2,
                             approved_missing=False, clinical_phase=0,
                             clinical_phase_missing=True, interaction_score=interaction,
                             n_publications=g+d, source_DGIdb=1,
                             composite_score=.65*interaction+.15*(d % 2)+.2*g/6))
    return pd.DataFrame(rows)


class EvidenceRegressionTests(unittest.TestCase):
    def test_typed_identity_and_pair_feature_placement(self):
        data = gnn.prepare_data(fixture())
        self.assertIn('gene:G0', data['node_keys'])
        self.assertIn('drug:id:dgidb:0', data['node_keys'])
        self.assertFalse(any('interaction_score' in n for n in data['node_feature_names']))
        self.assertIn('interaction_score_value', data['pair_feature_names'])
        self.assertEqual(data['edge_index'].shape[1], 120)
        self.assertEqual(len(set(data['edges'].pair_id)), 60)
        self.assertFalse(any('composite_score' in n or n.startswith('score_')
                             for n in data['node_feature_names']+data['pair_feature_names']))

    def test_missingness_and_scaler_training_boundary(self):
        frame = fixture()
        data = gnn.prepare_data(frame)
        j = data['pair_feature_names'].index('interaction_score_value')
        self.assertAlmostEqual(data['scalers']['pair'].mean_[j], frame.iloc[data['train']].interaction_score.mean())
        x = data['scalers']['node'].inverse_transform(data['x'].numpy())
        j = data['node_feature_names'].index('drug_clinical_phase_missing')
        for i, key in enumerate(data['node_keys']):
            if key.startswith('drug:'):
                self.assertAlmostEqual(x[i, j], 1, places=5)
        frame.loc[0, 'approved'] = 1  # same ID now has conflicting known property
        conflict = gnn.prepare_data(frame)['node_feature_audit']
        self.assertTrue(conflict.feature.eq('approved').any())

    def test_input_validation(self):
        frame = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            gnn.prepare_data(pd.concat([frame, frame.iloc[[0]]]))
        frame.loc[0, 'composite_score'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Targets'):
            gnn.prepare_data(frame)

    def test_candidate_aliases_never_cross_label_splits(self):
        frame = fixture()
        aliases = frame.iloc[:15].copy()
        aliases.drug_id = aliases.drug_id+'-other_namespace'
        frame = pd.concat([frame, aliases], ignore_index=True)
        data = gnn.prepare_data(frame)
        edges = data['edges']
        self.assertTrue(edges.groupby('label_split_group').label_split.nunique().eq(1).all())
        self.assertTrue(edges.groupby(['gene', 'drug']).label_split.nunique().eq(1).all())
        self.assertTrue(data['drug_identity_audit'].n_ids_in_alias_group.gt(1).any())
        # Conservative splitting does not silently merge unconfirmed entities.
        self.assertEqual(len(edges), len(frame))

    def test_heldout_targets_do_not_change_features(self):
        first = gnn.prepare_data(fixture())
        altered = fixture()
        altered.loc[first['test'], 'composite_score'] = .99
        second = gnn.prepare_data(altered)
        for key in ['x', 'pair', 'edge_index', 'src', 'dst']:
            self.assertTrue(torch.equal(first[key], second[key]))

    def test_training_baselines_exports_and_checkpoint(self):
        data = gnn.prepare_data(fixture())
        result = gnn.run_experiment(data, seeds=(3, 4), hidden=8, embed=4, dropout=0,
                                    epochs=3, patience=2)
        mean_val = result['validation'].groupby('model').validation_mse.mean()
        self.assertEqual(result['selected'], mean_val.loc[['GCN', 'GAT', 'GraphSAGE']].idxmin())
        self.assertEqual(set(result['test_metrics'].model),
                         {result['selected'], 'MLP', 'No_graph', 'Ridge', 'Training_mean'})
        self.assertEqual(len(result['stability']), 1)
        self.assertTrue(result['ranking'].gnn_score.between(0, 1).all())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            prov = dict(contrast='pooled', task=gnn.SCOPE)
            gnn.export_experiment(result, data, path/'tables', path/'models', prov)
            report = gnn.generate_outputs(result, data, path/'figures', path/'reports', prov)
            self.assertIn('not independent drug activity', report.read_text(encoding='utf-8'))
            saved = pd.read_csv(path/'tables/gnn_drug_ranking.csv')
            self.assertTrue({'drug_id', 'pair_id', 'clinical_phase_missing', 'contrast', 'label_split'} <= set(saved))
            self.assertTrue((path/'tables/composite_drug_ranking.csv').exists())
            self.assertTrue((path/'tables/biological_review_queue.csv').exists())
            preferred = result['validation_comparison'].sort_values('validation_mse').iloc[0].model
            self.assertEqual(preferred, result['recommended_approximator'])
            run = result['runs'][result['selected']][0]
            checkpoint = torch.load(path/'models'/f'{result["selected"]}_seed_3.pt', weights_only=False)
            model = gnn.EvidenceModel(**checkpoint['config'])
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            with torch.no_grad():
                np.testing.assert_allclose(model(data).numpy(), gnn.predict(run, data))

    def test_selection_is_independent_of_test_labels(self):
        data = gnn.prepare_data(fixture())
        settings = dict(seeds=(7,), hidden=8, embed=4, dropout=0, epochs=2, patience=2)
        first = gnn.run_experiment(data, **settings)
        data['y'][data['test']] = .99
        second = gnn.run_experiment(data, **settings)
        self.assertEqual(first['selected'], second['selected'])
        pd.testing.assert_frame_equal(first['validation'], second['validation'])

    def test_notebook_code_compiles_and_outputs_cleared(self):
        nb = json.loads((ROOT/'notebooks/03_gnn_drug_ranking.ipynb').read_text(encoding='utf-8'))
        for index, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell_{index}', 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])

    def test_notebook_executes_end_to_end_on_offline_input(self):
        nb = json.loads((ROOT/'notebooks/03_gnn_drug_ranking.ipynb').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            helper = root/'scripts/utils/evidence_gnn_functions.py'
            helper.parent.mkdir(parents=True)
            shutil.copyfile(ROOT/'scripts/utils/evidence_gnn_functions.py', helper)
            shutil.copyfile(ROOT/'scripts/utils/drug_identity_functions.py', helper.with_name('drug_identity_functions.py'))
            inputs = root/'results/tables/target_prioritisation/pooled'
            inputs.mkdir(parents=True)
            fixture().to_csv(inputs/'dgi_edges_gnn.csv', index=False)
            import hashlib
            (inputs/'provenance.json').write_text(json.dumps(dict(status='completed', contrast='pooled',
                identity_handling='conservative alias grouping',
                dgi_edges_sha256=hashlib.sha256((inputs/'dgi_edges_gnn.csv').read_bytes()).hexdigest())))
            namespace = {'display': lambda *args: None}
            with patch('pathlib.Path.cwd', return_value=root):
                for cell in nb['cells']:
                    if cell['cell_type'] != 'code':
                        continue
                    source = ''.join(cell['source'])
                    exec(compile(source, '<notebook>', 'exec'), namespace)
                    if 'TRAINING' in source:
                        namespace['TRAINING'].update(epochs=2, patience=2, hidden=8, embed=4)
                        namespace['INITIALIZATION_SEEDS'] = (42,)
            output = root/'results/tables/evidence_score_regression/pooled'
            self.assertEqual(json.loads((output/'provenance.json').read_text())['status'], 'completed')
            self.assertTrue((root/'results/reports/evidence_score_regression/pooled/03_gnn_drug_ranking_report.html').exists())


if __name__ == '__main__':
    unittest.main()
