# E3 Statistical Addendum

Bootstrap intervals use 10,000 paired resamples with seed 20260725. Wilcoxon
tests are paired on paragraph z-score drops.

| Family | Mean z drop | 95% bootstrap CI | Paired p-value vs baseline |
|---|---:|---:|---:|
| Plain paraphrase | 0.569 | [0.437, 0.714] | 4.85e-15 |
| Sentence boundary | 1.144 | [0.985, 1.307] | 9.64e-26 |
| AMR-guided | 1.064 | [0.919, 1.218] | 3.52e-25 |

Both structured attack families significantly reduce scores relative to their
own baseline. Sentence boundary has a mean advantage of 0.574 over plain
paraphrase (p = 7.88e-08); AMR-guided has a mean advantage of 0.495
(p = 2.90e-09). The direct sentence-boundary vs AMR-guided comparison is not
significant (mean difference 0.079, p = 0.299), so the data support comparable
performance rather than a definitive winner.

These tests concern detector scores, not semantic validity. Human evaluation
is still required for the final Valid ASR claim.
