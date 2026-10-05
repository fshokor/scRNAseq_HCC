from pathlib import Path
root=Path('/home/shoko/OmicSage')
p=root/'pipeline/modules/scripts/spatial/checkpoint_cache.py';s=p.read_text()
old='''def require_predecessor(step, cfg, output):
    parent, source = predecessor(step,cfg,output)
    if parent and not valid(parent,predecessor(parent,cfg,source)[1],cfg,source):
        raise ValueError(f'Checkpoint {parent} is stale or unverified. Run from ingest to rebuild the dependency chain.')'''
new='''def first_invalid(step, input_path, cfg, output):
    """Report the earliest failed dependency, not just the requested parent."""
    output = Path(output)
    parent, source = predecessor(step, cfg, output)
    if parent:
        failure = first_invalid(parent, predecessor(parent, cfg, source)[1], cfg, source)
        if failure:
            return failure
    if not output.exists():
        return step, f"missing checkpoint {output}"
    if not manifest_path(output).exists():
        return step, f"missing validation manifest for {output}"
    try:
        saved = json.loads(manifest_path(output).read_text())
        if saved['output_sha256'] != digest(output):
            return step, f"checkpoint checksum changed: {output}; it may be incomplete or overwritten"
        if saved['signature'] != signature(step, input_path, cfg):
            return step, f"inputs, parameters or code changed for {output}"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return step, f"cannot verify {output}: {exc}"
    return None


def require_predecessor(step, cfg, output):
    parent, source = predecessor(step,cfg,output)
    failure = first_invalid(parent, predecessor(parent,cfg,source)[1], cfg, source) if parent else None
    if failure:
        failed_step, reason = failure
        raise ValueError(f"Cannot run {step}: {reason}. Rebuild from '{failed_step}' through '{parent}'.")


def atomic_write_h5ad(adata, output):
    """Finish HDF5 serialization before replacing the previous checkpoint."""
    import os
    import tempfile
    output = Path(output)
    fd, name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp.h5ad', dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        adata.write_h5ad(temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)'''
assert old in s;s=s.replace(old,new);p.write_text(s)
p=root/'run_spatial_pipeline.py';s=p.read_text();s=s.replace('record as cache_record, require_predecessor','record as cache_record, require_predecessor, atomic_write_h5ad');s=s.replace('adata.write_h5ad(out_path)','atomic_write_h5ad(adata, out_path)');p.write_text(s)
print('Atomic checkpoint writes and precise dependency errors applied.')
