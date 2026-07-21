# SWAN Baseline Reproduction Plan

> Windows compatibility alias for the baseline-stage `PLAN.md` contract. The quest-level research map remains in `plan.md`.

## 1. Map link

- parent map node: SWAN baseline reproduction
- node objective: obtain a trustworthy SWAN detection baseline for downstream attack comparisons
- node deliverable: reproducible environment, fixed source revision, detector outputs, metric contract, and verification report
- success transition: AMR invariance audit and 50-paragraph attack pilot
- failure transition: bounded repair or explicit downgrade decision

## 2. Core contract

- route: reproduce from the official SWAN paper and repository
- baseline id: `swan_acl2026_realnews`
- primary variant: official 50-template bank, five generated sentences per RealNews prompt
- source paper: https://arxiv.org/abs/2605.04305
- source repo: https://github.com/amazon-science/SWAN
- source revision: to be pinned immediately after source checkout
- task: paragraph-level discrimination of SWAN-watermarked and human text
- data/split: first 250 examples from the C4 RealNews subset, matching the official repository
- detector contract:
  - official amrlib BART-large parser
  - S2MATCH sentence score against the private bank
  - sentence threshold `0.65`
  - null hit rate `lambda=0.25`
  - paragraph-level z-test
- metrics:
  - AUC, TPR@1% FPR, TPR@5% FPR
  - sentence hit rate, max-bank S2MATCH, paragraph z-score
- paper references:
  - no attack: AUC 99.1, TPR@1% 91.6, TPR@5% 97.6
  - Pegasus: 98.1 / 81.2 / 92.8
  - Parrot: 97.5 / 82.0 / 92.4
  - Claude paraphrase: 98.3 / 86.0 / 95.2

## 3. Resource and environment plan

- OS: Windows 10 build 26200
- local GPU: NVIDIA GeForce RTX 4060, 8GB GDDR6 (user-reported)
- current GPU diagnostic: `nvidia-smi` unavailable; driver/CUDA visibility unverified
- environment manager: `uv 0.11.6`
- Python: create a workspace-local Python 3.11 environment; do not use installed Python 3.13/3.14
- local workloads:
  - amrlib BART-large parser with FP16 and batch size 1 initially
  - T5 AMR-to-text with one model loaded at a time
  - S2MATCH/GloVe and evaluation on CPU
- API workloads:
  - DeepSeek for generation and attack-candidate production through its OpenAI-compatible API
  - pin formal model id, parameters, and call date
  - API key only in `DEEPSEEK_API_KEY`; never persist it
- API deviation:
  - the paper's retired Bedrock model versions cannot be reproduced exactly
  - DeepSeek-generated results are a current-model reproduction variant, not an exact numerical replication

## 4. Execution path

1. Verify NVIDIA driver visibility and PyTorch CUDA smoke feasibility.
2. Check out the official SWAN source and pin its commit.
3. Create `.venv` with Python 3.11 through `uv`.
4. Install dependencies; obtain S2MATCH, GloVe, amrlib parser/generator, pre-built banks, and RealNews data.
5. Run a 10-prompt CPU/GPU detector smoke test before any full generation.
6. Reproduce detection mechanics and no-attack metrics.
7. Adapt the LLM provider boundary to DeepSeek without changing rejection sampling or detector logic.
8. Run 30–50 prompts as a generation/detection pilot.
9. Run the full 250-prompt baseline and official paraphrase baselines.
10. Write verification verdict and canonical metric contract.

Exact command lines for steps 5–9 will be frozen after the checked-out source entrypoints are inspected; no non-existent DeepSeek CLI option is assumed in advance.

## 5. Outputs

- baseline root: `baselines/local/swan_acl2026_realnews/`
- source: `third_party/SWAN/`
- environment record: `baselines/local/swan_acl2026_realnews/setup.md`
- raw logs/results: `baselines/local/swan_acl2026_realnews/runs/`
- verification: `baselines/local/swan_acl2026_realnews/verification.md`
- canonical metric contract: `baselines/local/swan_acl2026_realnews/json/metric_contract.json`

## 6. Acceptance and downgrade rules

- `verified_match`: official code/data/model-equivalent route and paper-facing metrics reproduced within predeclared tolerances;
- `verified_close`: detector contract is exact and trends are stable, with modest deviations attributable to stochasticity or supported model-version differences;
- `partially_verified`: pipeline and metrics are operational, but DeepSeek generation or platform differences prevent clean numerical comparison; caveat must accompany all later results;
- `failed`: parser/detector, dataset, or metric contract cannot be made trustworthy.

Provisional numerical tolerance for `verified_close`: no-attack AUC within 3 percentage points and paraphrase AUC within 5 percentage points of the paper, with the same qualitative robustness ordering. Any tolerance revision must occur before seeing the full 250-sample result and be recorded here.

## 7. Risks and fallback

- missing NVIDIA driver/CUDA: perform a tiny CPU smoke only, then repair GPU visibility before the full 1,250-sentence parse;
- 8GB OOM: batch size 1, FP16, load parser and generator separately, cache parsed AMRs;
- Python/package incompatibility: stay on Python 3.11 in a local `uv` environment;
- Windows symlink/model layout issues: use explicit model paths or a workspace-local directory/junction; do not rely on undocumented global state;
- model-version drift: keep exact SWAN detector, label DeepSeek as a new generation variant, and avoid claiming exact paper replication;
- unavailable RealNews/model downloads: record the failed source and request an approved mirror rather than silently changing datasets.

## 8. Revision log

| Date | Change | Reason | Impact |
|---|---|---|---|
| 2026-07-21 | Selected hybrid RTX 4060 + DeepSeek route | User confirmed hardware and API access | Exact Bedrock generation is replaced by a documented current-model variant |
| 2026-07-21 | Used `analysis_plan.md` compatibility alias | Windows paths are case-insensitive, so `plan.md` and `PLAN.md` cannot coexist | Preserves separate research-map and baseline-contract layers |
