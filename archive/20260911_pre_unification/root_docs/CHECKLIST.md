# SWAN Baseline Checklist

## Identity

- parent map node: SWAN baseline reproduction
- baseline id: `swan_acl2026_realnews`
- route: reproduce
- owner stage: baseline

## In progress

- [x] Implement and run the 223-paragraph E3 main attack comparison.

## Next

- [ ] Run official paraphrase comparisons as a paper-wide reproduction extension.

## Core gate

- [x] Baseline identity and reproduce route are explicit.
- [x] Dataset, detector, and metric contracts are explicit.
- [x] `analysis_plan.md` records outputs, acceptance, risks, and fallback (Windows compatibility alias for `PLAN.md`).
- [x] Smoke test completed once with durable evidence.
- [x] Full no-attack validation decision completed with durable evidence.
- [x] Expected no-attack result files and metrics verified.
- [x] No-attack baseline accepted as `verified_close`; full paper reproduction marked `partially_verified`.

## Known blockers/dependencies

- [ ] DeepSeek key was supplied in chat; rotate it after the requested experiment.

## Done

- [x] User selected D1-A, D2-A, D3-A, D4-C, and D5-A.
- [x] Local tools checked: `uv` and Git available; installed Python versions are unsuitable for the target environment.
- [x] Hardware/API workload split recorded.
- [x] NVIDIA driver and CUDA runtime verified with a GPU tensor operation.
- [x] Workspace CPython 3.11.14 environment and official Python dependencies installed.
- [x] SWAN pinned at `2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a`.
- [x] S2MATCH pinned at `711a231d3600139662fe048460e10d773ff8e214` and SWAN patch applied.
- [x] GloVe 6B 100d installed; S2MATCH smoke passed.
- [x] Official BART-large AMR parser installed; single-sentence GPU parse passed at 1615 MiB peak allocated VRAM.
- [x] Official T5 AMR-to-text generator installed; generation smoke passed at 862 MiB peak allocated VRAM.
- [x] C4 RealNews saved: 250 validation prompts and 5000 human references.
- [x] Ten-document local detector smoke passed; 47 sentences scored against the 50-template bank.
- [x] DeepSeek V4 Flash non-thinking request path passed an offline mocked-response test.
- [x] DeepSeek credential smoke passed against the live API.
- [x] Ten-prompt, five-sentence API smoke passed: 45/50 accepted; detector AUC 1.0 on 10+10 paragraphs.
- [x] Thirty-prompt pilot completed: 126/150 accepted; AUC 0.998; TPR@1% 0.867; TPR@5% 0.933.
- [x] Formal 250-prompt generation completed: 1010/1250 accepted (80.8%).
- [x] Formal no-attack detection completed: AUC 0.985; TPR@1% 0.892; TPR@5% 0.944.
- [x] Verification report written to `baselines/local/swan_acl2026_realnews/verification.md`.
- [x] E1 invariance audit completed on 100 AMRs: graph controls passed with a documented serialization caveat.
- [x] E2 50-paragraph attack pilot completed: sentence-boundary Valid ASR 34%; AMR-guided Valid ASR 24%.
- [x] E2 candidates reparsed and rescored with the official detector; validated paired results saved.
- [x] E3 223-paragraph main comparison completed for all three candidate families.
- [x] E3 paired bootstrap intervals and Wilcoxon tests completed.
- [x] E4 strict same-budget ablation completed on 50 paired paragraphs.
- [x] AI-assisted quality evaluation completed and independently confirmed by the user.
- [ ] Transfer across template banks or parsers.
- [ ] Lightweight defense experiment.
- [ ] Final paper figures and evidence consolidation.
