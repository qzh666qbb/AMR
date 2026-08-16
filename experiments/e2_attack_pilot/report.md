# E2 AMR Attack Pilot

## Scope

Fifty paragraphs above the predeclared 1% FPR threshold in the formal baseline
were selected. DeepSeek generated three same-budget candidate families. Every
candidate was reparsed with the official BART AMR parser and rescored with the
fixed SWAN detector.

The 1% FPR escape threshold was fixed at z = 2.33 from human calibration
scores. Valid ASR additionally required preservation of all detected numbers,
source named entities, negation polarity, and a length ratio between 0.45 and
2.0. These are surface/structural filters, not a substitute for human quality
evaluation.

## Results

| Candidate family | Quality pass | Escape at 1% FPR | Valid ASR | Mean z | Mean z drop |
|---|---:|---:|---:|---:|---:|
| Plain paraphrase | 39/50 | 6/50 (12%) | 5/50 (10%) | 3.229 | 0.573 |
| Sentence boundary | 42/50 | 20/50 (40%) | 17/50 (34%) | 2.814 | 0.988 |
| AMR-guided | 40/50 | 16/50 (32%) | 12/50 (24%) | 2.802 | 1.001 |

## Interpretation

Both sentence-boundary and AMR-guided candidates reduce detector scores more
than ordinary paraphrase. Sentence-boundary changes currently have the highest
pilot Valid ASR. AMR guidance is promising but does not yet demonstrate an
advantage over this simpler baseline. This is a pilot result, not a claim about
the 250-paragraph main experiment.

The quality filter is conservative and automated. At least 100 paired examples
still require blinded human evaluation before a final semantic-preservation
claim.

## Evidence

- candidates: candidates.json
- reparsed candidates: datasets/*/parsed_amrs.json
- raw detector outputs: detection/*/results.csv
- validated paired rows: validated_results.json
- runners: scripts/generate_e2_candidates.py and scripts/validate_e2_results.py

## Decision

E2 passes the feasibility gate: at least one meaning-constrained candidate
family lowers detection in a stable pilot. E3 should compare
sentence-boundary and AMR-guided search on all 250 paragraphs under equal
candidate budgets, including a no-AMR adaptive-search control.
