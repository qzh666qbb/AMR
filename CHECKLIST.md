# SWAN Baseline Checklist

## Identity

- parent map node: SWAN baseline reproduction
- baseline id: `swan_acl2026_realnews`
- route: reproduce
- owner stage: baseline

## In progress

- [ ] Verify NVIDIA driver/CUDA visibility; `nvidia-smi` is currently unavailable.

## Next

- [ ] Obtain approval for source/dependency/model/data downloads.
- [ ] Check out and pin the official SWAN source revision.
- [ ] Create a workspace-local Python 3.11 environment with `uv`.
- [ ] Inspect official entrypoints and freeze exact smoke commands.
- [ ] Run a 10-prompt detector smoke test.
- [ ] Run a 30–50-prompt generation/detection pilot with the DeepSeek adapter.
- [ ] Run the full 250-prompt baseline and official paraphrase comparisons.
- [ ] Write verification and decide `verified_match`, `verified_close`, `partially_verified`, or `failed`.
- [ ] Transition to the AMR invariance audit and attack pilot.

## Core gate

- [x] Baseline identity and reproduce route are explicit.
- [x] Dataset, detector, and metric contracts are explicit.
- [x] `analysis_plan.md` records outputs, acceptance, risks, and fallback (Windows compatibility alias for `PLAN.md`).
- [ ] Smoke test completed once with durable evidence.
- [ ] Full validation decision completed with durable evidence.
- [ ] Expected result files and metrics verified.
- [ ] Baseline accepted, downgraded, or blocked with a durable note.

## Known blockers/dependencies

- [ ] NVIDIA driver/CUDA visibility is not yet established.
- [ ] Python 3.11 and project dependencies are not yet installed.
- [ ] Official repository, models, GloVe, and RealNews data are not yet downloaded.
- [ ] DeepSeek API credentials have not been tested; no credential should be pasted into chat or committed.

## Done

- [x] User selected D1-A, D2-A, D3-A, D4-C, and D5-A.
- [x] Local tools checked: `uv` and Git available; installed Python versions are unsuitable for the target environment.
- [x] Hardware/API workload split recorded.
