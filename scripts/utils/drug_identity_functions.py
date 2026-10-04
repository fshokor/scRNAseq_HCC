"""Conservative drug identity review and grouping; names do not prove equivalence."""
import hashlib
import pandas as pd


def alias_groups(frame):
    """Connect exact normalized names or identical IDs for conservative splitting.

    These are candidate alias groups, not chemical equivalence assertions. Salts,
    punctuation and formulations are intentionally not stripped from names.
    """
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    names = frame.drug.astype(str).map(lambda s: ' '.join(s.strip().casefold().split()))
    ids = frame.get('drug_id', pd.Series('', index=frame.index)).fillna('').astype(str).str.strip()
    aliases = frame.get('drug_alias_names', frame.drug).fillna('')
    for name, identifier, alias_text in zip(names, ids, aliases):
        find('name:'+name)
        if identifier:
            union('name:'+name, 'id:'+identifier)
        for alias in str(alias_text).split(' | '):
            alias = ' '.join(alias.strip().casefold().split())
            if alias:
                union('name:'+name, 'name:'+alias)
    members = {}
    for key in list(parents):
        members.setdefault(find(key), []).append(key)
    group_ids = {key: 'alias:'+hashlib.sha256('\0'.join(sorted(values)).encode()).hexdigest()
                 for key, values in members.items()}
    groups = pd.Series([group_ids[find('name:'+name)] for name in names], index=frame.index)
    audit = frame[['drug', 'drug_id']].copy() if 'drug_id' in frame else frame[['drug']].assign(drug_id='')
    audit['drug_alias_group'] = groups
    audit = audit.drop_duplicates().sort_values(['drug_alias_group', 'drug', 'drug_id'])
    counts = audit.assign(drug_id=audit.drug_id.replace('', pd.NA)).groupby('drug_alias_group').drug_id.nunique()
    audit['n_ids_in_alias_group'] = audit.drug_alias_group.map(counts)
    audit['identity_review_status'] = audit.n_ids_in_alias_group.map(
        lambda n: 'candidate_aliases_require_review' if n > 1 else
                  'single_database_identity' if n == 1 else 'name_only_requires_review')
    return groups, audit.reset_index(drop=True)


def confirmed_identity_map(mapping):
    """Only explicit confirmed mappings may collapse different database IDs."""
    if mapping is None:
        return {}
    required = {'drug_id', 'canonical_drug_id', 'confirmed', 'evidence_reference'}
    if not required <= set(mapping):
        raise ValueError(f'Identity map requires columns {sorted(required)}')
    frame = mapping.copy()
    if frame.empty:
        return {}
    confirmed = frame.confirmed.astype(str).str.lower().isin(['true', '1', '1.0'])
    if not confirmed.all():
        raise ValueError('Identity mapping rows must be explicitly confirmed.')
    for col in ['drug_id', 'canonical_drug_id', 'evidence_reference']:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'Identity mapping has missing {col}.')
        frame[col] = frame[col].astype(str).str.strip()
    if frame.groupby('drug_id').canonical_drug_id.nunique().gt(1).any():
        raise ValueError('Conflicting canonical drug identities.')
    result = frame.drop_duplicates('drug_id').set_index('drug_id').canonical_drug_id.to_dict()
    # Require final canonical IDs, rather than order-dependent chains or cycles.
    if any(value in result and result[value] != value for value in result.values()):
        raise ValueError('Use final canonical IDs; mapping chains/cycles are unsupported.')
    return result
