from pathlib import Path
root=Path('/home/shoko/OmicSage')
p=root/'ui/_pages/p2_configure.py';s=p.read_text()
old='''            p["max_mt_pct"] = _slider_num("Max MT%", 1.0, 60.0,
                                           p.get("max_mt_pct", 20.0), 0.5,
                                           f"{k}_mt", fmt="%.1f")'''
assert old in s
new='''            mt_enabled = st.toggle(
                "Filter spots by mitochondrial %",
                value=p.get("max_mt_pct", 20.0) is not None,
                key=f"{k}_mt_enabled",
                help="When off, mitochondrial percentages are measured but not used to remove spots.",
            )
            if mt_enabled:
                previous = p.get("max_mt_pct")
                p["max_mt_pct"] = _slider_num("Max MT%", 0.0, 100.0,
                                             20.0 if previous is None else previous, 0.5,
                                             f"{k}_mt", fmt="%.1f")
            else:
                p["max_mt_pct"] = None
                st.caption("MT percentages will be measured; mitochondrial filtering is disabled.")'''
s=s.replace(old,new);p.write_text(s)
p=root/'ui/config_builder.py';s=p.read_text()
needle='    cfg["spatial"] = {k: v for k, v in cfg["spatial"].items() if v is not None}'
assert needle in s
s=s.replace(needle,'    # None is meaningful here: omitting it would restore the runner\'s 20% default.\n    cfg["spatial"]["qc"]["max_mt_pct"] = qc_p.get("max_mt_pct", 20.0)\n'+needle)
p.write_text(s)
print('Spatial MT toggle and null-preserving config serialization fixed.')
