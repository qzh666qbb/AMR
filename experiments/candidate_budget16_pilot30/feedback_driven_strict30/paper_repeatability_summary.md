# Repeatability Summary

This note collects the two repeatability results that are most useful for a paper.

## Full candidate set: 120 candidates, 10 repeats

| Metric | Value |
|---|---:|
| Total candidates | 120 |
| Paragraphs | 30 |
| Repeats | 10 |
| Mean candidate escape rate | 0.2167 |
| Candidates that ever escaped | 26 |
| Candidates that always escaped | 26 |
| Candidates with mixed threshold outcomes | 0 |
| Paragraphs with at least one escaping candidate | 11 |
| Paragraph-level mean ASR | 0.3667 |

Key point: no candidate crossed the `z <= 2.33` threshold differently across repeats. Some scores moved inside the safe side of the threshold, but the binary detection decision stayed fixed for all 120 candidates.

Paragraphs with stable success across all 10 repeats:

| Paragraph index | Success frequency |
|---|---:|
| 4 | 10/10 |
| 5 | 10/10 |
| 6 | 10/10 |
| 10 | 10/10 |
| 11 | 10/10 |
| 13 | 10/10 |
| 14 | 10/10 |
| 16 | 10/10 |
| 20 | 10/10 |
| 23 | 10/10 |
| 26 | 10/10 |

## Strict-valid subset: 16 manually verified candidates, 50 repeats

| Metric | Value |
|---|---:|
| Verified candidates | 16 |
| Verified paragraphs | 9 |
| Repeats | 50 |
| Mean paragraph ASR | 1.0000 |
| Paragraphs stable across all repeats | 9/9 |
| Candidates with mixed threshold outcomes | 0 |

The strict-valid subset is fully stable at the paragraph level. Candidate 23 shows score-level jitter between `1.1547` and `2.3094`, but both remain below the escape threshold.

## Interpretation

The detection pipeline is not score-deterministic, but the paper-relevant binary conclusion is stable on this dataset:

1. Score values can move because S2Match uses random hill-climbing.
2. The `z <= 2.33` decision did not flip for any of the 120 candidates in the 10-repeat full audit.
3. The 9 strict-valid paragraphs also remained stable across 50 repeats.

Artifacts:

- `full_repeatability_10.json`
- `full_repeatability_10.md`
- `full_repeatability_10.png`
- `repeatability_50.json`
- `repeatability_50.md`
