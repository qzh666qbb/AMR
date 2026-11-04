# Final Paper Bundle

This is the strongest consolidated result from the current experiment set.

| Quantity | Value |
|---|---:|
| Strict Valid ASR | 0.300 |
| 95% CI for Strict Valid ASR | [0.147, 0.494] |
| Strict successful paragraphs | 9/30 |
| Strict candidate rate | 0.133 |
| 95% CI for strict candidate rate | [0.078, 0.207] |
| Threshold plateau | 0.65 to 0.80 |
| Plateau strict Valid ASR | 0.300 |
| Full 50-repeat candidate flips | 0 |
| Full 50-repeat stable candidates | 120/120 |
| Full 50-repeat paragraph ASR | 0.367 |

## Core claim

The strict result is 30.0% at the default threshold 0.65, remains flat through 0.80, and is stable under 50 repeated S2Match recomputations on the full 120-candidate set.

## Artifacts

- `threshold_sensitivity.png`
- `full_repeatability_50.png`
- `paper_final_bundle.json`