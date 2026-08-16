# DeepSeek 30-Prompt Pilot

## Contract

- Date: 2026-07-22
- Model: `deepseek-v4-flash`
- Mode: non-thinking
- Dataset: first 30 C4 RealNews validation prompts
- Output: five target sentences per prompt
- Injection bank: official 50 templates
- Injection threshold: 0.75
- Detection threshold: 0.65
- Null rate: 0.25
- Seed: 20260722
- Candidate strategy: parser-in-the-loop feedback, maximum 50 trials per sentence

## Generation

- Target sentences: 150
- Accepted injections: 126 (84.0%)
- Total candidate requests: 3298
- Mean attempts per target: 21.99
- Median attempts: 16.5
- 95th percentile: 50
- Paragraphs with all five accepted: 18/30
- Paragraphs with at least four accepted: 24/30
- Placeholder leaks: one rejected fallback candidate; fixed after this pilot
- Wall time: 1:21:53

## Detection

- Machine/human paragraphs: 30/30
- AUC: 0.998
- TPR at 1% FPR: 0.867
- TPR at 5% FPR: 0.933
- Mean machine z-score: 3.473
- Mean human z-score: -0.795
- Positive z-score: machine 30/30; human 3/30

## Verdict

`pilot_pass_with_generation_caveat`

The detector line is strong enough to proceed to a checkpointed formal baseline. The DeepSeek generator is not equivalent to the paper's retired Claude model: 16% of target sentences missed the injection threshold and candidate cost is high. All downstream claims must report both paragraph detection and sentence-level injection acceptance.
