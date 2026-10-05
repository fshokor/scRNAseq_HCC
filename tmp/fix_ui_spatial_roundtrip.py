from pathlib import Path
root=Path('/home/shoko/OmicSage')
p=root/'ui/config_builder.py';s=p.read_text()
needle='    # None is meaningful here: omitting it would restore the runner\'s 20% default.'
assert needle in s
replacement='''    # Preserve imported parameters not represented by widgets, including explicit
    # nulls. Dropping these changes cache fingerprints and silently loses options.
    import copy
    for step, block in cfg["spatial"].items():
        if isinstance(block, dict):
            inherited = copy.deepcopy(step_params.get(step, {}))
            inherited.update(block)
            cfg["spatial"][step] = inherited

'''+needle
s=s.replace(needle,replacement)
# Export what the user edited rather than silently restoring hardcoded defaults.
start=s.index('def build_spatial(');end=s.index('\n# ── Helpers',start);block=s[start:end]
block=block.replace('"normalize_total": True,','"normalize_total": reduce_p.get("normalize_total", True),').replace('"log1p":           True,','"log1p":           reduce_p.get("log1p", True),').replace('"flavor":          "seurat",','"flavor":          reduce_p.get("flavor", "seurat"),').replace('"run_svg":       True,','"run_svg":       cluster_p.get("run_svg", True),').replace('"svg_n_genes":   3000,','"svg_n_genes":   cluster_p.get("svg_n_genes", 3000),').replace('"annotation_map": None,','"annotation_map": cluster_p.get("annotation_map"),')
s=s[:start]+block+s[end:];p.write_text(s)
print('Spatial export preserves hidden parameters and explicit nulls.')
