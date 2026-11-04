# Best observed frontier

This consolidates the strongest results already present in the repository.

## Highest observed strict-valid result

- K = 1
- Valid ASR = 33.3%
- Escape rate = 100.0%
- Surface-valid rate = 96.7%
- Semantic-valid rate = 33.3%

## Most stability-backed strict result

- Strict Valid ASR = 30.0%
- 95% CI = [0.147, 0.494]
- Threshold plateau = 0.65 to 0.80
- 50-repeat candidate flips = 0
- 50-repeat stable candidates = 120/120

## Best raw detector escape

- Strategy = boundary_only
- Escape rate = 93.3%
- Mean z drop = 3.005

## Best quality-weighted prompt component escape

- Strategy = boundary_only
- Escape rate = 93.3%
- Mean z drop = 3.005

## Strongest external baseline

- Baseline pilot 30x5
- Escape rate = 13.3%
- Mean z = 3.473

## Interpretation

- If you optimize purely for the highest strict-valid ASR already observed, K=1 is currently the leader at 33.3%.
- If you optimize for paper robustness, the feedback-driven strict result at 30.0% is stronger because it has a threshold plateau and 50-repeat stability.
- If you optimize for raw detector escape, boundary-only is the strongest prompt primitive at 93.3%, but quality must be accounted for separately.
