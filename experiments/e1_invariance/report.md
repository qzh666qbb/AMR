# E1 AMR Invariance Audit

## Verdict

`pass_with_serialization_caveat`

The graph-level negative controls support the expected invariance. Direct AMR
serialization changes must not be counted as text attacks because the S2MATCH
string parser is sensitive to layout in ways that do not reflect graph
semantics.

## Protocol

- source: first 100 valid sentence AMRs from the formal 250-paragraph run
- seed: `20260725`
- controls: variable renaming, triple reordering, whitespace formatting, and
  canonical variable normalization
- checks: pairwise S2MATCH and maximum S2MATCH against the full 50-template bank

## Results

| Control | Exact invariance | Mean pair S2MATCH | Maximum bank-score change |
|---|---:|---:|---:|
| Variable renaming | 100/100 | 1.000 | numerical zero |
| Canonical normalization | 100/100 | 1.000 | numerical zero |
| Triple reordering | 98/100 | 0.997 | 0.157 |
| Single-line whitespace layout | 0/100 | 0.632 | 0.444 |

The two triple-order exceptions and all single-line exceptions occur after
PENMAN re-serialization. They identify a string-parser/layout artifact, not a
meaning-preserving text transformation.

## Gate decision

Proceed to E2 under these restrictions:

1. attacks take text as input and produce text as output;
2. every candidate is parsed again by the official BART parser;
3. detector scores are computed only from that fresh parser output;
4. direct AMR variable, triple-order, or whitespace manipulation is excluded
   from attack success;
5. serialization/parser failures are reported separately.

Raw evidence is in `results.json`; the reproducible runner is
`scripts/run_e1_invariance_audit.py`.
