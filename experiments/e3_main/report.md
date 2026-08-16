# E3 RealNews Main Attack Comparison

The experiment uses 223 formal-baseline-positive paragraphs, three equal-budget
DeepSeek candidate families, the same BART parser, the same 50-template bank,
and the fixed SWAN detector.

| Family | AUC | TPR@1% | TPR@5% | Mean z | Mean z drop | Valid ASR |
|---|---:|---:|---:|---:|---:|---:|
| Plain paraphrase | 0.980 | 0.870 | 0.928 | 3.181 | 0.569 | 6.7% (15/223) |
| Sentence boundary | 0.983 | 0.610 | 0.825 | 2.607 | 1.144 | 30.9% (69/223) |
| AMR-guided | 0.977 | 0.637 | 0.874 | 2.686 | 1.064 | 21.1% (47/223) |

Valid ASR requires the fixed 1% FPR escape threshold plus preservation of
numbers, named entities, negation, and length constraints. It is an automated
conservative measure; human evaluation remains required.

## Conclusion

Sentence-boundary transformation is the strongest attack in this comparison.
AMR-guided rewriting is effective relative to ordinary paraphrase, but does not
outperform the simpler sentence-boundary baseline. The current evidence supports
a narrower contribution around paragraph aggregation and sentence-boundary
robustness, unless a later AMR-specific search or ablation isolates additional
gains.
