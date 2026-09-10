# Paper outline

Working title:

`AMR-Structured Evasion of Semantic Watermarking with Feedback-Driven Candidate Search`

## 1. Abstract

- Problem statement
- Method overview
- Main strict result and stability claim
- Why the result matters

## 2. Introduction

- Semantic watermark detection on AMR representations is brittle under structure-aware rewrites
- Existing rewriting baselines improve either raw escape or fluency, but not both
- Our contribution: feedback-driven candidate search, strict validity filtering, and robustness analysis

## 3. Method

3.1 Threat model and notation

3.2 Candidate generation

3.3 Detector and strict-valid criterion

3.4 Robustness metrics

## 4. Experimental setup

- Data
- Detector
- Candidate budgets
- Thresholds
- Evaluation metrics

## 5. Results

5.1 Main results

5.2 Baseline comparison

5.3 Prompt-component ablation

5.4 Threshold sensitivity and repeatability

5.5 Failure analysis

## 6. Discussion

- What the result proves
- What the result does not prove
- Best observed vs. most robust configuration

## 7. Limitations and reproducibility

- Automatic vs. manual quality validation
- Seed, thresholds, scripts, and artifacts

## 8. Conclusion

- Summary of contributions and takeaways

## Planned figures

- Fig. 1: pipeline schematic
- Fig. 2: frontier of observed success rates
- Fig. 3: threshold sensitivity and 50-repeat stability
- Fig. 4: prompt-component ablation and failure/quality tradeoff

## Planned tables

- Table 1: main results and confidence intervals
- Table 2: baseline comparison under the same detector
- Table 3: prompt-component ablation
- Table 4: robustness and repeatability summary
- Table 5: failure taxonomy / quality analysis

