"""Repeat the fixed AMR document that exposed stochastic S2Match scores."""

import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "parser_repeatability30"
manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
parsed = json.loads((out / "parsed_amrs_run1.json").read_text(encoding="utf-8"))
positions = [i for i, row in enumerate(manifest) if int(row["source_index"]) == 23]
if len(positions) != 5 or any(parsed[p] != parsed[positions[0]] for p in positions[1:]):
    raise RuntimeError("Expected five identical AMR parses for source 23")
(out / "s2match_repeat100_parsed.json").write_text(
    json.dumps([parsed[positions[0]]] * 100, indent=2), encoding="utf-8"
)
print({"source_index": 23, "fixed_amr_repeats": 100})
