"""Known-association evidence-score regression; not response or link prediction.

Held-out labels remain on a known, unweighted graph. Pair metadata is available
at inference. The target is constructed from some of these same ingredients:
test performance measures score distillation, not biological validation.
"""
import copy
import base64
import hashlib
import html
import json
import importlib.util
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

DRUG_FEATURES = ['approved', 'immunotherapy', 'anti_neoplastic', 'clinical_phase']
GENE_FEATURES = ['hub_score', 'log2FC']
PAIR_FEATURES = ['interaction_score', 'n_publications', 'source_DGIdb',
                 'source_ChEMBL', 'source_OpenTargets', 'type_inhibitor',
                 'type_agonist', 'type_antagonist', 'type_antibody',
                 'type_binder', 'type_activator']
SCOPE = 'Regression of constructed evidence scores for known associations'

_identity_spec = importlib.util.spec_from_file_location(
    'hcc_drug_identity', Path(__file__).with_name('drug_identity_functions.py'))
identity = importlib.util.module_from_spec(_identity_spec)
_identity_spec.loader.exec_module(identity)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def known_values(df, column):
    values = pd.to_numeric(df.get(column, pd.Series(np.nan, index=df.index)), errors='coerce')
    flag = column + '_missing'
    if flag in df:
        missing = df[flag].astype(str).str.lower().isin(['true', '1', '1.0'])
        values = values.mask(missing)
    return values.replace([np.inf, -np.inf], np.nan)


def prepare_data(edges, split_seed=42):
    edges = edges.copy().reset_index(drop=True)
    for col in ['gene', 'drug', 'composite_score']:
        if col not in edges:
            raise ValueError(f'Missing required column: {col}')
    if len(edges) < 20:
        raise ValueError('At least 20 unique associations are needed for this split.')
    for col in ['gene', 'drug']:
        if edges[col].isna().any() or edges[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'Empty {col} identity.')
    if 'contrast' in edges and edges.contrast.nunique(dropna=False) != 1:
        raise ValueError('Use exactly one contrast per experiment.')
    y = pd.to_numeric(edges.composite_score, errors='raise').to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any() or (y > 1).any() or np.var(y) == 0:
        raise ValueError('Targets must be finite, variable evidence scores in [0, 1].')
    edges['gene_key'] = 'gene:' + edges.gene.astype(str).str.strip()
    ids = edges.get('drug_id', pd.Series('', index=edges.index)).fillna('').astype(str).str.strip()
    edges['drug_key'] = np.where(ids.ne(''), 'drug:id:' + ids,
                                'drug:name:' + edges.drug.astype(str).str.strip().str.casefold())
    edges['drug_identity_fallback'] = ids.eq('')
    edges['pair_id'] = [hashlib.sha256((g+'\0'+d).encode()).hexdigest()
                        for g, d in zip(edges.gene_key, edges.drug_key)]
    if edges.pair_id.duplicated().any():
        raise ValueError('Duplicate gene/drug identities; deduplicate evidence upstream.')
    groups, identity_audit = identity.alias_groups(edges)
    edges['drug_alias_group'] = groups
    edges['label_split_group'] = [hashlib.sha256((gene+'\0'+group).encode()).hexdigest()
                                  for gene, group in zip(edges.gene_key, groups)]
    split_groups = edges.label_split_group.to_numpy()
    if len(set(split_groups)) < 20:
        raise ValueError('At least 20 distinct gene/candidate-alias groups are needed.')
    splitter = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=split_seed)
    train, hold = next(splitter.split(edges, groups=split_groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=.50, random_state=split_seed)
    a, b = next(splitter.split(hold, groups=split_groups[hold]))
    val, test = hold[a], hold[b]
    keys = sorted(set(edges.gene_key) | set(edges.drug_key))
    lookup = {key: i for i, key in enumerate(keys)}
    src = edges.gene_key.map(lookup).to_numpy()
    dst = edges.drug_key.map(lookup).to_numpy()
    node_cols = [f'{kind}_{col}_{suffix}' for kind, cols in [('drug', DRUG_FEATURES), ('gene', GENE_FEATURES)]
                 for col in cols for suffix in ['value', 'missing', 'conflict']] + ['is_gene', 'is_drug']
    x = np.zeros((len(keys), len(node_cols)), dtype=np.float32)
    audit = []
    for kind, keycol, cols, offset in [('drug', 'drug_key', DRUG_FEATURES, 0),
                                     ('gene', 'gene_key', GENE_FEATURES, 3*len(DRUG_FEATURES))]:
        for key, rows in edges.groupby(keycol, sort=False):
            i = lookup[key]
            x[i, -2 if kind == 'gene' else -1] = 1
            for j, col in enumerate(cols):
                values = known_values(rows, col).dropna().unique()
                conflict = len(values) > 1 and not np.allclose(values, values[0])
                if kind == 'gene' and conflict:
                    raise ValueError(f'Inconsistent {col} for {key} within contrast.')
                missing = len(values) == 0 or conflict
                x[i, offset+3*j:offset+3*j+3] = [0 if missing else values[0], missing, conflict]
                if conflict:
                    audit.append({'node_key': key, 'feature': col, 'status': 'conflict_set_unknown'})
    pair_cols = [f'{col}_{suffix}' for col in PAIR_FEATURES for suffix in ['value', 'missing']]
    pair = np.column_stack([v for col in PAIR_FEATURES for v in
                            [known_values(edges, col).fillna(0), known_values(edges, col).isna()]]).astype(np.float32)
    train_nodes = np.unique(np.r_[src[train], dst[train]])
    node_scaler, pair_scaler = StandardScaler(), StandardScaler()
    x = node_scaler.fit(x[train_nodes]).transform(x).astype(np.float32)
    pair = pair_scaler.fit(pair[train]).transform(pair).astype(np.float32)
    edge_index = np.array([np.r_[src, dst], np.r_[dst, src]])
    edges['label_split'] = ''
    for name, indices in [('train', train), ('validation', val), ('test', test)]:
        edges.loc[indices, 'label_split'] = name
    edges['drug_seen_in_training'] = edges.drug_key.isin(edges.iloc[train].drug_key)
    edges['drug_alias_seen_in_training'] = edges.drug_alias_group.isin(edges.iloc[train].drug_alias_group)
    return dict(edges=edges, x=torch.tensor(x), pair=torch.tensor(pair),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                src=torch.tensor(src), dst=torch.tensor(dst), y=torch.tensor(y, dtype=torch.float32),
                train=train, val=val, test=test, node_keys=keys,
                node_feature_names=node_cols, pair_feature_names=pair_cols,
                scalers={'node': node_scaler, 'pair': pair_scaler},
                node_feature_audit=pd.DataFrame(audit, columns=['node_key', 'feature', 'status']),
                drug_identity_audit=identity_audit,
                adjacency_scope='All known unweighted associations, including held-out label pairs')


class EvidenceModel(nn.Module):
    def __init__(self, node_dim, pair_dim, architecture='GCN', hidden=64, embed=32, dropout=.2):
        super().__init__()
        self.architecture = architecture
        self.config = dict(node_dim=node_dim, pair_dim=pair_dim, architecture=architecture,
                           hidden=hidden, embed=embed, dropout=dropout)
        constructors = {'GCN': GCNConv, 'GAT': GATConv, 'GraphSAGE': SAGEConv}
        conv = constructors.get(architecture)
        self.first = conv(node_dim, hidden) if conv else nn.Linear(node_dim, hidden)
        self.second = conv(hidden, embed) if conv else nn.Linear(hidden, embed)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2*embed+pair_dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1), nn.Sigmoid())

    def encode(self, x, adjacency):
        if self.architecture == 'MLP':
            return self.second(self.drop(torch.relu(self.first(x))))
        return self.second(self.drop(torch.relu(self.first(x, adjacency))), adjacency)

    def forward(self, data, no_graph=False):
        adjacency = data['edge_index'][:, :0] if no_graph else data['edge_index']
        z = self.encode(data['x'], adjacency)
        return self.head(torch.cat([z[data['src']], z[data['dst']], data['pair']], dim=1)).flatten()


def fit_model(data, architecture, seed, *, hidden=64, embed=32, dropout=.2,
              lr=.003, weight_decay=1e-4, epochs=300, patience=40, no_graph=False):
    seed_everything(seed)
    model = EvidenceModel(data['x'].shape[1], data['pair'].shape[1], architecture,
                          hidden, embed, dropout).to(data['x'].device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, state, stale = float('inf'), None, 0
    history = {'train_loss': [], 'val_loss': []}
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(data, no_graph)
        loss = nn.functional.mse_loss(pred[data['train']], data['y'][data['train']])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss.')
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            value = nn.functional.mse_loss(model(data, no_graph)[data['val']], data['y'][data['val']]).item()
        if not np.isfinite(value):
            raise RuntimeError('Non-finite validation loss.')
        history['train_loss'].append(loss.item())
        history['val_loss'].append(value)
        if value < best:
            best, state, stale = value, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if state is None:
        raise ValueError('Training needs at least one epoch.')
    model.load_state_dict(state)
    model.eval()
    return dict(model=model, seed=seed, validation_mse=best, history=history, no_graph=no_graph)


def predict(run, data):
    with torch.no_grad():
        return run['model'](data, run['no_graph']).cpu().numpy()


def metrics(y, pred):
    return dict(mse=float(mean_squared_error(y, pred)), mae=float(mean_absolute_error(y, pred)),
                r2=float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else float('nan'))


def run_experiment(data, seeds=(42, 43, 44), **training):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct initialization seeds.')
    runs, records = {}, []
    for name in ['GCN', 'GAT', 'GraphSAGE', 'MLP']:
        runs[name] = []
        for seed in seeds:
            run = fit_model(data, name, seed, **training)
            runs[name].append(run)
            records.append(dict(model=name, seed=seed, validation_mse=run['validation_mse']))
        print(f'{name}: mean validation MSE = {np.mean([r["validation_mse"] for r in runs[name]]):.6f}', flush=True)
    validation = pd.DataFrame(records)
    means = validation.groupby('model').validation_mse.mean()
    selected = means.loc[['GCN', 'GAT', 'GraphSAGE']].idxmin()
    # Selection is frozen before any test labels are evaluated.
    print(f'Validation-selected GNN: {selected}', flush=True)
    runs['No_graph'] = [fit_model(data, selected, seed, no_graph=True, **training) for seed in seeds]
    x, pair = data['x'].cpu().numpy(), data['pair'].cpu().numpy()
    features = np.column_stack([x[data['src'].cpu()], x[data['dst'].cpu()], pair])
    y = data['y'].cpu().numpy()
    ridge = Ridge(alpha=1.).fit(features[data['train']], y[data['train']])
    predictions = {name: np.mean([predict(r, data) for r in runs[name]], axis=0)
                   for name in [selected, 'MLP', 'No_graph']}
    predictions['Ridge'] = np.clip(ridge.predict(features), 0, 1)
    predictions['Training_mean'] = np.full(len(y), np.mean(y[data['train']]))
    comparison = pd.DataFrame([dict(model=name, validation_mse=metrics(y[data['val']], pred[data['val']])['mse'])
                               for name, pred in predictions.items()])
    recommended = comparison.loc[comparison.validation_mse.idxmin(), 'model']
    # All learned comparator choices above use validation; test is read only here.
    table = pd.DataFrame([dict(model=name, **metrics(y[data['test']], pred[data['test']]))
                          for name, pred in predictions.items()])
    seed_preds = np.array([predict(r, data) for r in runs[selected]])
    ranking = data['edges'].copy()
    ranking['gnn_score'] = predictions[selected]
    ranking['initialization_score_sd'] = seed_preds.std(axis=0)
    ranking['delta_from_constructed_score'] = ranking.gnn_score - ranking.composite_score
    ranking = ranking.sort_values('gnn_score', ascending=False, kind='stable').reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(1, len(ranking)+1))
    primary = data['edges'].sort_values('composite_score', ascending=False, kind='stable').copy()
    primary.insert(0, 'evidence_rank', np.arange(1, len(primary)+1))
    review = primary.head(100).copy()
    for field in ['direct_target_evidence', 'desired_action', 'hcc_evidence_reference',
                  'celltype_support', 'review_decision', 'review_notes']:
        review[field] = ''
    review['biological_review_status'] = 'pending; database association alone is insufficient'
    bands = pd.cut(y[data['test']], bins=[-.00001, .1, .25, .5, 1.00001],
                   labels=['0–0.1', '0.1–0.25', '0.25–0.5', '0.5–1'])
    band_metrics = []
    for band in bands.categories:
        indices = data['test'][np.asarray(bands == band)]
        if len(indices):
            for name, pred in predictions.items():
                band_metrics.append(dict(score_band=str(band), n_pairs=len(indices), model=name,
                                         **metrics(y[indices], pred[indices])))
    stability = []
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            k = min(20, len(y))
            a, b = seed_preds[i], seed_preds[j]
            stability.append(dict(seed_a=seeds[i], seed_b=seeds[j],
                spearman=pd.Series(a).corr(pd.Series(b), method='spearman'),
                top20_overlap=len(set(np.argsort(a)[-k:]) & set(np.argsort(b)[-k:]))/k))
    return dict(selected=selected, runs=runs, validation=validation, test_metrics=table,
                predictions=predictions, ranking=ranking, ridge=ridge, seed_predictions=seed_preds,
                validation_comparison=comparison, recommended_approximator=recommended,
                primary_ranking=primary, biological_review_queue=review, score_band_metrics=pd.DataFrame(band_metrics),
                stability=pd.DataFrame(stability, columns=['seed_a', 'seed_b', 'spearman', 'top20_overlap']))


def export_experiment(result, data, tables_dir, models_dir, provenance):
    tables_dir, models_dir = Path(tables_dir), Path(models_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [('gnn_drug_ranking', result['ranking']), ('validation_metrics', result['validation']),
                         ('test_metrics', result['test_metrics']), ('ranking_stability', result['stability']),
                         ('node_feature_audit', data['node_feature_audit']),
                         ('drug_identity_audit', data['drug_identity_audit']),
                         ('validation_comparison', result['validation_comparison']),
                         ('composite_drug_ranking', result['primary_ranking']),
                         ('biological_review_queue', result['biological_review_queue']),
                         ('test_score_band_metrics', result['score_band_metrics'])]:
        frame.to_csv(tables_dir/f'{name}.csv', index=False)
    data['edges'][['pair_id', 'gene_key', 'drug_key', 'drug_alias_group', 'label_split_group',
                   'label_split', 'drug_seen_in_training', 'drug_alias_seen_in_training']].to_csv(tables_dir/'label_splits.csv', index=False)
    result['ranking'].assign(absolute_score_error=lambda f: f.delta_from_constructed_score.abs()).nlargest(
        50, 'absolute_score_error').to_csv(tables_dir/'largest_score_disagreements.csv', index=False)
    pd.DataFrame(result['seed_predictions'].T, columns=[f'seed_{r["seed"]}' for r in result['runs'][result['selected']]]).assign(pair_id=data['edges'].pair_id).to_csv(tables_dir/'initialization_predictions.csv', index=False)
    for name in [result['selected'], 'MLP', 'No_graph']:
        for run in result['runs'][name]:
            torch.save(dict(state_dict=run['model'].state_dict(), config=run['model'].config,
                            no_graph=run['no_graph'], seed=run['seed'], provenance=provenance),
                       models_dir/f'{name}_seed_{run["seed"]}.pt')
    with (models_dir/'preprocessing.pkl').open('wb') as handle:
        pickle.dump(dict(scalers=data['scalers'], ridge=result['ridge'],
                         node_feature_names=data['node_feature_names'], pair_feature_names=data['pair_feature_names']), handle)
    torch.save({k: data[k].cpu() for k in ['x', 'pair', 'edge_index', 'src', 'dst', 'y']}, models_dir/'graph_tensors.pt')
    (models_dir/'node_keys.json').write_text(json.dumps(data['node_keys'], indent=2), encoding='utf-8')


def generate_outputs(result, data, figures_dir, reports_dir, provenance):
    import matplotlib.pyplot as plt
    figures_dir, reports_dir = Path(figures_dir), Path(reports_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ['GCN', 'GAT', 'GraphSAGE', 'MLP']:
        run = result['runs'][name][0]
        axes[0].plot(run['history']['val_loss'], label=name)
    axes[0].set(xlabel='Epoch', ylabel='Validation MSE', title='First initialization; selection uses all seeds')
    axes[0].legend()
    test = data['test']
    y = data['y'].cpu().numpy()[test]
    axes[1].scatter(y, result['predictions'][result['selected']][test], s=8, alpha=.5)
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set(xlabel='Constructed evidence score', ylabel='Predicted evidence score', title='Held-out labels, known graph')
    fig.tight_layout()
    fig.savefig(figures_dir/'evidence_regression.png', dpi=150)
    plt.close(fig)
    top = result['ranking'].head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    positions = np.arange(len(top))
    ax.barh(positions, top.gnn_score.to_numpy()[::-1])
    ax.set_yticks(positions, [f'{r.rank}. {r.drug} / {r.gene}' for r in top.itertuples()][::-1])
    ax.set(xlabel='Predicted evidence score', title='Known drug–gene associations; not predicted efficacy')
    fig.tight_layout()
    fig.savefig(figures_dir/'evidence_ranking.png', dpi=150)
    plt.close(fig)
    message = ('The target is a rule-derived composite score, with overlapping feature ingredients. '
               'These test metrics assess reproduction of that score, not independent drug activity, '
               'binding affinity, sensitivity, or clinical efficacy. All known unweighted associations '
               'are visible in the graph; only labels are held out. This is not a novel-link evaluation. '
               'One sample per tissue group also limits biological generalization of upstream evidence. '
               'Initialization spread and top-20 overlap are computational stability checks, not confidence intervals. '
               'The original composite ranking remains the transparent reference; retain a GNN only if its added value is defensible.')
    text = f'<!doctype html><meta charset="utf-8"><title>Evidence score regression</title><h1>{SCOPE}</h1>'
    text += f'<p>{html.escape(message)}</p><p>Validation-selected architecture: {result["selected"]}.</p>'
    text += f'<p>Validation-preferred learned score approximator: {result["recommended_approximator"]}. '
    text += 'The original composite score remains the primary evidence ranking, regardless of learned model choice. '
    text += 'Matched names or IDs were grouped conservatively before label splitting; this is not proof of chemical equivalence.</p>'
    text += '<h2>Validation comparison including simpler baselines</h2>'+result['validation_comparison'].to_html(index=False)
    text += '<h2>Validation selection</h2>'+result['validation'].to_html(index=False)
    text += '<h2>Test comparison after selection</h2>'+result['test_metrics'].to_html(index=False)
    text += '<h2>Initialization stability</h2>'+result['stability'].to_html(index=False)
    for filename in ['evidence_regression.png', 'evidence_ranking.png']:
        encoded = base64.b64encode((figures_dir/filename).read_bytes()).decode('ascii')
        text += f'<p><img style="max-width:100%" src="data:image/png;base64,{encoded}" alt="{filename}"></p>'
    text += '<h2>Primary composite evidence ranking</h2>'+result['primary_ranking'].head(20).to_html(index=False)
    text += '<h2>Experimental GNN approximation ranking</h2>'+top.to_html(index=False)
    text += '<h2>Test performance by constructed-score band</h2>'+result['score_band_metrics'].to_html(index=False)
    text += '<h2>Provenance</h2><pre>'+html.escape(json.dumps(provenance, indent=2))+'</pre>'
    path = reports_dir/'03_gnn_drug_ranking_report.html'
    path.write_text(text, encoding='utf-8')
    return path
