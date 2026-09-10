# Baseline comparison under the same detector

Escape threshold: 2.33

| Method | n | Mean z | Escape rate | 95% CI | AUROC | Notes |
|---|---:|---:|---:|---:|---:|---|
| Baseline pilot 30x5 | 30 | 3.473 | 13.3% | [0.038, 0.307] | 0.998 | Existing 30-prompt pilot under the same detector |
| Baseline formal 250x5 | 250 | 3.439 | 10.8% | [0.072, 0.153] | 0.985 | Larger formal baseline, different sample size |
| Proposed method | 30 | - | 36.7% | [0.147, 0.494] | - | Quality-checked strict ASR + repeatability |

## Interpretation

- The pilot baseline is the cleanest apples-to-apples external reference because it shares the same 30-prompt scale.
- The formal 250x5 run gives a larger-sample background estimate under the same detector.
- The proposed method is reported with quality-checked strict ASR; its detector-level repeatability is summarized separately.

## Strict-method specifics

- Strict Valid ASR: 30.0%
- 95% CI: [0.147, 0.494]
- Threshold plateau: 0.65 to 0.80
- Plateau Strict Valid ASR: 30.0%
