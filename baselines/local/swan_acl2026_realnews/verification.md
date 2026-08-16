# SWAN Baseline Verification

## Verdict

- no-attack DeepSeek variant: `verified_close`
- full paper reproduction: `partially_verified`
- date: 2026-07-23

The official detector contract, RealNews sample size, AMR parser, S2MATCH
threshold, null rate, and 50-template bank were retained. The generation model
is a documented DeepSeek replacement for the unavailable paper model, so this
is not an exact model-equivalent reproduction. Official paraphrase comparisons
remain outside this verification.

## Formal result

| Metric | Formal run | Paper no-attack | Difference |
|---|---:|---:|---:|
| AUC | 98.5% | 99.1% | -0.6 pp |
| TPR @ 1% FPR | 89.2% | 91.6% | -2.4 pp |
| TPR @ 5% FPR | 94.4% | 97.6% | -3.2 pp |

The AUC is within the predeclared three-percentage-point tolerance. Low-FPR
tail performance is modestly lower but preserves the expected strong
separation.

## Generation and score diagnostics

- paragraphs: 250 machine and 250 human
- target sentences: 1,250
- accepted injections: 1,010 (80.8%)
- attempts per target: mean 22.35, median 17, maximum 50
- machine z-score: mean 3.439; 242/250 above zero
- human z-score: mean -0.834; 20/250 above zero

The principal caveat is generation efficiency and injection reliability, not
detector operation. Paragraph aggregation preserves strong detection even when
some sentence-level injections fail.

## Durable evidence

- generation metadata: `runs/formal_250x5/sampled_data.json`
- generation checkpoint: `runs/formal_250x5/generation_checkpoint.json`
- parsed machine AMRs: `runs/formal_250x5/parsed_amrs.json`
- detector metrics: `runs/formal_250x5/detection/results.csv`
- z-scores: `runs/formal_250x5/detection/machine_z_scores.npy` and
  `runs/formal_250x5/detection/human_z_scores.npy`

## Transition

The no-attack baseline is trustworthy enough to serve as the control for the
AMR invariance audit and the first 50-paragraph D1-A attack pilot. Claims about
paper-wide robustness remain provisional until ordinary paraphrase and
AMR-guided attack comparisons are available.
