"""Merge results/run*/results.json into results/results.json (later runs fill stages that failed earlier)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
merged = {}
for p in sorted((root / "results").glob("run*/results.json")):
    for k, v in json.load(open(p)).items():
        if not isinstance(v, dict):
            continue
        if k not in merged or ("error" in merged[k] and "error" not in v):
            merged[k] = dict(v, source_run=p.parent.name)
(root / "results" / "results.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
print({k: ("ok" if "error" not in v else "FAILED") + " " + v["source_run"] for k, v in merged.items()})
