# Threshold Sensitivity Summary

The strict Valid ASR is stable once the sentence-level S2Match threshold reaches 0.65.

## Main result

| Threshold | Strict Valid ASR |
|---|---:|
| 0.50 | 0.033 |
| 0.55 | 0.133 |
| 0.60 | 0.267 |
| 0.62 | 0.267 |
| 0.64 | 0.267 |
| 0.65 | 0.300 |
| 0.70 | 0.300 |
| 0.75 | 0.300 |
| 0.80 | 0.300 |

## Interpretation

- The strict Valid ASR rises as the threshold tightens from 0.50 to 0.65.
- At the default threshold 0.65, the strict Valid ASR is 30.0%.
- From 0.65 through 0.80, the strict Valid ASR stays flat at 30.0%.
- Candidate-level escape rate and paragraph-level ASR continue to grow with looser thresholds, so the detection metric remains threshold-sensitive even though the final strict conclusion is stable above 0.65.

## Artifacts

- `threshold_sensitivity.json`
- `threshold_sensitivity.md`
- `threshold_sensitivity.png`
