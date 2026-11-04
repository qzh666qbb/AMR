# Sentence-Boundary Watermark Research State

Updated: 2026-11-02

> **2026-11-02 全项补实验已授权并执行：** 用户在W01–W08、X01–X04、
> S01–S02缺口审计后明确要求“都补一补吧”。当前有运行中的迁移实验和
> 持久GPU后继队列；下文旧的no-active/no-successor状态均按各历史日期理解，
> 不覆盖最新授权。冻结Stop/Halt不追认成功，不把资源中断解释为方法无效。
> W01现已完成24对×6条件144记录并通过最终双审计；保留读出12对上均值
> 融合AUROC 0.910–0.938，复杂融合无稳定额外增益，正式FPR/TPR未确认。
> W02 pilot完整0/8，492候选仅3接受，终态Stop；这是写入配方失败，
> 不能当作正常长度攻击结果。W03后续50来源生成已结束，自动合格0/50；
> 2616调用仅5槽接受，现2500行评分已完成；Null对照0/200、path0各条件3–4/200，
> 无合格正例不能验证path0正例权衡。原字典复述读回320句完成，
> 读出符号BA32.29%（模型辅助映射）、唯一完整序列0/12；
> AMR soft排序54.41%低于同候选embedding74.16%，不能宣称可靠消息恢复。
> W07新提示432/432生成及
> 账单已验证、读回仍待计算。正常长度W02主实验、W05全三臂链路与匹配LLM
> Null、独立Bank/二元反馈/多Bank/多解析器均已实现并部署排队，不能视作效果确认。
> 新独立真人证据仍0；SemStamp原缺9项中6公共资产已恢复同SHA，三份密封输入
> 仍缺，尚未开始后继认证。最新范围、逐项状态及证据见
> [任务清单](任务清单.md)和[执行账本](experiments/swan_research_completion_20261102/PLAN.md)。

> **2026-10-14 V3 INVALID-EDGE POLICY / P1B NO-GO:** The v1 No-Go and
> stopped v2 partial remain immutable. v2 stopped before decision at 144/1000
> calibration rows when `FORT SILL, Okla.,` produced a structurally invalid
> recovered graph; development stayed 0/210 and no RESULT/AUDIT exists. v3
> preserves the frozen v2 topology and exact count-state DP, but excludes
> selected-beam ERROR/CRITICAL, graph-invalid, and roundtrip-invalid edges
> before DP. A disconnected lattice returns abstain and remains in the full
> denominator; missing recovery or AMR evidence still stops the run.
>
> The separately frozen v3 score-blind P0 is Go: all five punctuation forms
> recover 600/600 original boundaries, all 5150 pressure rows retain complete
> topology paths, positive/null span budgets are 1.4793x/2.9582x, and the
> synthetic invalid-high-score and all-invalid-path checks pass with zero
> parser/scorer/GPU/network calls. Local tests pass 50/50; the remote run had
> 50 tests, zero failures, and one environment-dependent skip. P1b used a new
> output directory, SQLite cache, lock, and Supervisor, started from
> calibration 0/1000, and retained all 1000 calibration plus 210 development
> rows. Baseline detection was 20/30, below the frozen 24/30 gate, so the
> decision is No-Go. All-five-variant conditional survival was 18/20, all
> seven invalid candidate edges were excluded before DP, selected invalid
> edges were zero, and both remote and local no-model audits passed. This is
> development evidence, not independent FPR or formal TPR. No further rule
> repair on these assets and no P2 preparation are authorized. See
> `experiments/swan_elastic_v3_development1/CLOSEOUT.md`.

> **2026-10-14 COMPLETED ENGINEERING EVALUATION / NO-GO:** The user authorized planning
> and executing follow-up SWAN elastic-detector validation. Original formal
> Null evaluation completed with 6/1000 false positives on 100–220-word human
> windows. The existing short development set yielded 28/30 baseline and
> 27/30 boundary-attack detections, with 27/28 conditional survival, but used
> global fallback outside the populated calibration length buckets.
> These are exploratory detection results, not validated short-text TPR at
> 1% actual FPR. The frozen `swan_elastic_length_boundary_audited2` completed:
> 1000 short calibration documents, 1000 independent test documents, and
> five fixed punctuation variants on the test documents and 30 existing
> positives. It preserves the original detector and calibration, extends only
> the short-length calibration bucket, and audits condition-specific FPR.
> The initial run stopped after 192 calibration rows because upstream amrlib
> recovery logs lacked Span bindings. The successor adds those bindings;
> 108 fixed Span outputs and 16 fixed document scores passed equivalence checks.
> All 7214 rows and both audits completed; 22163 cache/graph bindings passed.
> The old calibration produced short-text FPR 168/1000 (16.8%). The extension
> reduced it to 10/1000 (1.0%), with a one-sided 95% upper bound of 1.6903%;
> strict 1% support failed. Existing positives were detected at 26/30 and
> historical boundary attacks at 17/30, with 17/26 conditional survival.
> Comma/newline/space variants each yielded 0/30 detection; survival across
> all five variants was 0/26. Both semicolon variants yielded 25/30 detection.
> Comma/newline/space variants leave only one candidate Span for 29/30 positives
> in each condition. Full denominators were retained. There were 24 selected-beam
> ERROR Spans; statistical/stress and joint engineering gates both failed.
> GPU and finalizer jobs exited normally; there is no active GPU experiment.
> See `experiments/swan_elastic_length_boundary_audited2/CLOSEOUT.md` and `任务清单.md`.
> Earlier no-active-experiment statements below are historical. This task
> does not reopen failed generation/main/certificate runs or upgrade
> CL04–CL08, CL12, or CR08.

> **2026-10-06 UNIFIED-SUBMISSION CORRECTION / current authority:** The
> project has one TMLR submission, not a SWAN paper plus a separate
> certificate paper. The SWAN audit and the SemStamp/SimMark
> certified-resegmentation sequence are two evidence streams under the shared
> question of sentence boundaries as a watermark interface. Their
> system-specific claims remain non-transferable: certificate evidence does
> not complete SWAN CL04--CL08, and SWAN behavior does not provide a
> SemStamp/SimMark certified rate. The authoritative scope is
> `experiments/swan_branch_c_final_literature_refresh_20261002/UNIFIED_SUBMISSION_SCOPE.md`.
>
> **2026-09-18 ERRATUM:** `任务清单.md` is the sole
> execution plan. `TERMINOLOGY.md` fixes B, R(s), T(R(s)), and D(g1,g2).
> The SWAN evidence stream is an end-to-end audit of segmentation dependence.
> Normal-length count-neutral evidence, blinded quality, and condition-specific
> FPR remain uncompleted. Historical protocols below confer no execution authority.
>
> P0 preserved 39,457 file hashes and 82 original document copies.
> P1/P2 corrections and machine-readable endpoint gates are in
> `experiments/swan_reframing1/`. No new model, parser, S2Match, or detector
> observation was produced during these corrections.
>
> The old length-confirmation3 and the 15-unit A/B protocol are inactive.
> New generation stops on first reaching 100 words (upper bound 220), uses
> at least eight original detection units, preserves official fallback, and
> does not require 16 accepted slots or a 7--12-word sentence gate.
> New paid execution requires the concrete phase budget under checklist §18.10.
>
> `swan_count_neutral_final1` has completed and failed its eight-source pilot
> under the currently frozen labels. It is archived in Branch C; main,
> transformation, parsing, matching, and detection remain blocked.
> A post-closure read-only checklist audit accidentally exposed portions of
> `sealed/main_sources.json`. No main computation ran, but the allocation is
> now permanently treated as observed/contaminated. See decision log D007.
>
> The separate `swan_source_anchored1` experiment completed its frozen
> eight-source pilot without reopening final1. All 1,891 paid calls are
> terminal, with zero uncertain/invalid calls and USD 0.4919673 peak-price
> accounting. Five documents passed the structural checks, below the required
> 6/8, so the evidence-derived gate entered Branch C before quality review.
> No review packet, transform, detector result, control cohort, or main
> authorization was created. Full hash/accounting replay passed under manifest
> SHA256 `3fa20cc2b23a6ed0d7faafd5929e1d378f15346b6966a8af10d60c8f92cb6427`.
> Frozen AMR dependencies also emitted parser diagnostics during generation
> and replay, which is an independent audit stop under `STOP_RULES.md`.
>
> A later, separate engineering sequence established a narrow positive
> prerequisite without reopening either failed route. The target-47-only
> `swan_coherent_injection_pilot2` produced ten unique strict acceptances with
> no fallback. A disclosed post-hoc event-log adjudication passed two of three
> same-backend blind model reviews, and
> `swan_coherent_baseline_cached1` detected all 10/10 units
> (`z=5.477225575051661`). Cached and official S2Match values were identical
> on three frozen validation pairs. The document is only 53 words, uses one
> favorably selected template, lacks independent human review and empirical
> FPR calibration, and therefore does not satisfy formal Gate A/B or authorize
> the boundary experiment.
>
> The next isolated engineering step,
> `swan_normal_length_coherent_pilot1`, extended this construction to 126 words
> and twenty exact units: 20/20 strict acceptances, zero fallback, and two
> passing detector-blind same-backend event-log reviews. Full Bank50 detection
> was 20/20 green (`z=7.745966692414833`), with cached/official S2Match values
> identical on three frozen validation pairs. This closes the technical
> normal-length prerequisite for one target-47-only document, but it still
> lacks source anchoring, independent human review, a formal cohort, and
> empirical FPR calibration. Formal Gate A/B and the boundary experiment
> remain blocked.

> **2026-10-02 EVIDENCE-STREAM RECONCILIATION, superseded for publication
> structure on 2026-10-06:** The `certified_resegmentation_wp2_*` sequence is
> independent of the SWAN confirmatory endpoints. It supplies no endpoint for
> CL04--CL08, but it is included in the same unified manuscript. Its v18 B1 readiness
> stopped before certificate scoring because the frozen pair-capacity and
> replay-budget gates failed; it produced no threshold, FPR, TPR, witness, or
> certified-rate result. No v19, exact successor, cheap-bound successor, or
> paid run is authorized. Evidence-only consolidation and the pre-submission
> audit are complete: immediate main-conference submission No-Go and unified
> TMLR manuscript preparation Go. The single
> reserved literature refresh has been consumed, TMLR is the selected
> peer-review target, and the original eight-page SWAN-only draft has been
> superseded by an anonymous unified manuscript. There is no active experiment;
> external submission
> requires author, affiliation, conflict, ethics, funding, and artifact-license
> confirmation. See
> `experiments/research_direction_reconciliation_20261002/DECISION.md`.

`CURRENT_RESULTS.md` is the historical evidence ledger; experiment directories
remain the reproducibility source. The sections below describe historical
milestones unless explicitly identified as current.

The unified manuscript scope is
`experiments/swan_branch_c_final_literature_refresh_20261002/UNIFIED_SUBMISSION_SCOPE.md`.
The SWAN component and stop-state mapping remain
`experiments/swan_reframing1/BRANCH_C_SYSTEM_AUDIT.md` and
`experiments/swan_reframing1/BRANCH_C_CLOSURE.md`. The revised literature
position is `experiments/literature_novelty_audit1/BRANCH_C_UPDATE.md`.
The submission-readiness decision and English review packet are in
`experiments/swan_branch_c_submission_readiness_20261002/`.
The final literature refresh, venue decision, and compiled TMLR draft are in
`experiments/swan_branch_c_final_literature_refresh_20261002/`.
The versioned reconciliation audit is
`experiments/swan_reframing1/audit_v2.json`: it preserves the legacy audit's
single expected `ok=false` difference, proves that removing the fixed
SUPERSEDED notice byte-restores the P0 report SHA256, and verifies the current
Branch C authority, all 82 preserved originals, and CL01--CL16. It makes no
new model, parser, S2Match, or detector observation.

## Unified Core Question

How do sentence boundaries enter the operational contract of sentence-level
semantic watermarks, and can robustness be computed exactly over a declared
fixed-token, fixed-unit-count boundary-relocation set?

The SWAN component asks whether a normal-length, quality-qualified,
count-neutral boundary change alters the detector and its real parser-produced
candidate graphs; that endpoint remains open. The certificate component gives
exact algorithms for stated local detector classes and records the
SemStamp/SimMark system-validation boundary. The project does not claim a new
generic boundary attack, an attack on "AMR semantics," a completed
segmentation-invariant defense, or a real certified rate.

## Evidence That Survives The Reset

1. The historical 250-document study establishes a practical SWAN attack
   surface: fixed `K=4` Valid ASR was `191/250` (76.4%), and adaptive rescue
   reached `218/250` (87.2%). These results do not establish the final causal
   mechanism or normal-length generality.
2. On an independent 30-document short-text set, boundary-only rewriting
   reached `29/30` no-feedback Valid ASR. In the strict punctuation control,
   the public threshold escape count changed from `1/30` for the original
   five units to `30/30` after merging and returned to `1/30` when the saved
   five boundaries were supplied again. This is strong evidence for
   short-text segmentation dependence.
3. In the saved 108-target scorer audit, median target S2Match F1 changed from
   `0.8197` on original component graphs to `0.4502` after synthetic
   deterministic composition and `0.5200` after merged-text reparsing. This
   quantifies S2Match sensitivity to the constructed graph scope. It does not
   establish the causal mechanism of the natural-text attack, and it does not
   resolve whether parser changes are required for real merged text.
4. AliMark and SwordStamp already cover generic sentence splitting/merging,
   unit-count effects, and resegmentation-aware detection. The remaining
   defensible contribution is the system-specific SWAN causal audit.
5. A detector-blind count-neutral construction audit found that `0/30` frozen
   short documents contain a conservative natural split point, although
   `23/30` contain at least one related adjacent merge. It stopped before AMR
   parsing and S2Match. This is a cohort-feasibility negative result, not a
   detector result and not evidence against count-neutral effects on normal
   prose.

## Questions Still Open

- Whether count-neutral boundary exchange lowers SWAN detection on
  normal-length, quality-qualified watermarked text.
- Whether exact boundary restoration reverses that loss under independently
  calibrated condition-specific thresholds.
- Whether the actual parser-produced candidate template graphs and their
  fixed-bank matches change on that same final quality set.
- Whether a matched-length, matched-generator comparison supports a strong
  SWAN-specificity claim. Existing KGW/SynthID results are supporting only.

The final1 and source-anchored routes are closed. These questions remain
scientifically open, but they do not authorize retuning either failed pilot or
running either sealed main cohort. The source-anchored route satisfied the
new-identity, unseen-source, preregistration, budget, and credential-rotation
requirements, but failed its structural gate before the two-person quality
review. The certified-resegmentation branch does not answer these questions:
it evaluates a different detector-level worst-case certification problem on
SemStamp/SimMark and contains no SWAN count-neutral paired endpoint.

## Claim-to-Evidence Decision

`CLAIM_EVIDENCE_MATRIX.md` is the active decision record. Existing evidence
supports a narrow short-text SWAN segmentation case study and a separate
synthetic S2Match implementation diagnostic. The final1 pilot did not produce
a quality-qualified cohort and therefore entered Branch C before any
count-neutral detector comparison. It does not establish an AMR-specific
natural-text failure mechanism or Branch B.

## Terminated Partial Confirmation Run

`experiments/swan_final_confirmation1/` is preserved and terminated as an
incomplete feasibility run.

- 250 watermarked sources and 500 human controls were frozen.
- 66 complete eight-slot baselines and 558 slot checkpoints were produced.
- 27/558 slots met the target; per-document accepted-slot counts were
  0 (48 documents), 1 (12), 2 (5), and 3 (1).
- 0/66 baselines met the frozen minimum of five accepted slots.
- 5,510/5,510 recorded API calls are complete; no request is in flight.

This is not a Gate A result because the fixed 250 allocations were not
completed. It must not be reported as either experimental success or formal
Gate A failure. Reaching Gate A would require at least 104 of the remaining
184 allocations to become eligible after 0 of the first 66 complete
allocations met even the five-accepted-slot prerequisite. The run will not be
resumed; any future study requires a separately justified preregistration and
must not tune this protocol from the observed checkpoints.

## Positive-Asset Audit

`experiments/swan_positive_asset_audit1/` freezes the public and local asset
inventory. At official SWAN commit
`2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a`, the checked 36-file tree,
releases, tags, full public git history, paper links, and targeted dataset
searches contain no pre-generated watermarked-text release. This is a bounded
statement about the checked public locations, not universal nonexistence.

No existing local candidate satisfies all four saved-evidence gates:
traceable SWAN injection, document independence, 100--220 words with at least
8 sentence units, and detector-blind human quality. The closest quality asset
has 14/30 automated-quality C4 windows but no injection; the closest traceable
cohort has 30 independent SWAN baselines but only 17--46 words. Six
normal-length oracle batches are cross-topic statistical containers. In the
terminated run, 56/66 complete baselines had normal form, but 0/66 met its
minimum injection prerequisite and no quality review ran.

A separate collection protocol is frozen for 5 pilot plus 30 disjoint main
allocations beginning at C4 train shard 8. It uses the exact public injection
path and the repository's current Claude Sonnet 4.5 default, explicitly not
the paper's retired Claude 3.5 model. Pilot data cannot tune or enter main,
source replacement is forbidden, quality is frozen before detection, and
original detector response is an outcome rather than an inclusion gate.
Execution remains unauthorized because it requires a paid API.

`experiments/swan_positive_collection1/` now freezes the source side of that
protocol. A fresh scan considered 29,160 JSON/JSONL artifacts, parsed 13,732
identity-bearing files, and retained 18,346 normalized text identities plus
1,293 URL identities; the terminated final run is explicitly included. The
first 35 eligible records were C4 shard 8 rows 0--34, so the first 5 are frozen
as pilot and the next 30 as main. A supplemental scan found no prior shard-8
locator and no collision in 1,327 identity-like values from 70 CSV files.

The original pre-execution manifest hashed 54 files, but a provenance
correction found that its `third_party/SWAN` root is not byte-identical to the
claimed official commit. Five of the 36 tracked paths differ:
`detection/parse_machine_text.py`, both injection files, `utils/amr_utils.py`,
and `utils/bedrock_utils.py`. The v1 `runtime.json`,
`pre_execution_manifest.json`, and `audit.json` are preserved but superseded
for execution provenance.

`runtime_v2.json` now points to a separate clean checkout with exact HEAD
`2ca0c207...`, tree `2195cce...`, and all 36 official blob IDs and SHA-256
hashes frozen. The source allocation remains valid because
`utils/text_utils.py`, the only upstream code used by allocation, is
byte-identical. Independent v2 audit again recovers 5 pilot plus 30 main.

The pre-execution defects are now resolved by
`protocol_amendment_v3.md` and `execution_contract_v3.json`. A dedicated
`.venv-swan-positive` matches all 16 official package pins and freezes 99
installed distributions. A clean amr-metric-suite checkout at
`711a231d...` is combined mechanically with the official SWAN S2Match patch;
the parser/tokenizer, GloVe, NLTK, and spaCy model are bound and hashed.

The external runner preserves the official prompt, parse, normalization,
matching, threshold, and retry logic while adding deterministic JSON-order
bank sampling, isolated slot seeds, deterministic S2Match RNG handling,
single-attempt transport journaling, and explicit acceptance/fallback
evidence. For Claude Sonnet 4.5, `top_p=0.9` remains the declared parameter but
the effective request records `top_p=null`, matching official behavior.
Bedrock's top-level `InvokeModel` response does not return `modelId`, so the
old returned-identity claim is replaced by frozen inference-profile metadata,
requested profile ID, AWS request ID, and provider response metadata.

Independent v3 audit verifies local execution readiness, but remote execution
remains unauthorized. The one prior public C4 request is unchanged; AWS
control-plane, model-probe, generation, parser, S2Match, detector, and attack
counts remain zero. No watermarked positive exists yet.

A separately frozen, non-equivalent local-backend preflight used
Qwen2.5-0.5B-Instruct on the first slot of each of the five pilot sources.
All 50 local generations completed and received one parser and S2Match call.
No candidate reached the 0.75 injection threshold: 0/5 slots were accepted,
the maximum candidate score was 0.5911, and the preregistered >=3/5 gate
failed. Independent audit therefore fixes `stop_local_qwen_route`. These are
backend-feasibility results, not detector or attack results. DeepSeek and
Bedrock calls remain zero.

A separate, non-equivalent DeepSeek-V4.1-Flash preflight completed on the same
five first slots. A user-directed credential-risk override was recorded
without falsely claiming that the exposed credential had been rotated. The
run used one probe and 41 generation calls; all 42 API journals are complete,
with conservative peak-price cost $0.0062358 under the $0.08/51-call ceiling.
Only 1/5 slots was accepted, so the preregistered >=3/5 gate failed and
`stop_deepseek_route` is fixed. No detector or attack was run.

## Independent DeepSeek Confirmation: Pilot Stopped

`experiments/swan_positive_deepseek_confirmation1/` is a new experiment, not
a continuation or retuning of the failed five-source preflight. Its
preregistered five-document pilot is now terminal:

- source allocation used previously unseen C4 realnewslike train shard 9,
  with a fresh full-history identity snapshot;
- 5 pilot and 30 main sources were allocated together before model output;
- each document requires 16/16 accepted slots at S2Match >=0.75, 100--220
  words, exactly 16 sentence units, no AMR leakage, and two independent blind
  quality passes;
- each slot has a fixed 128-trial ceiling, 26 templates, and at most 5 retries
  per template; fallback never passes;
- pilot and main have separate immutable authorization files, call ceilings,
  and USD ceilings; main remains blocked until at least one pilot document
  passes the complete quality gate;
- every transmitted DeepSeek request is journaled before transport, and an
  uncertain request blocks continuation.

"Tuning after observing results" is now explicitly defined as designing a
separate future experiment with a new name and wholly unseen sources. It does
not permit changing this experiment's prompt, model, source order, template
count, retry count, threshold, length gate, or quality rule and resuming on
the same data.

`experiments/swan_count_neutral_confirmation1/` is the prepared downstream
protocol. For each quality-qualified 16-sentence main document it selects two
nonadjacent syntactically valid boundaries by a detector-blind hash. A changes
only boundary a from period to semicolon; B changes only boundary b. Both have
15 detector units and identical case-sensitive lexical sequence, order, and
punctuation multiset, so A to B is exactly one split plus one merge. The
runner also preregisters 250 calibration and 250 test controls from unseen C4
train shard 10, a common <=1% calibration-FPR threshold, independent test FPR,
and deterministic composition versus merged-text-reparse precision/recall/F1
decomposition.

Shard 9 rows 0--34 remain frozen as 5 pilot plus 30 main sources. The pilot
used one probe plus 904 generation calls; all 905 journals are complete, with
421,527 prompt tokens, 15,840 completion tokens, and conservative peak-price
cost $0.1454661 under the $17.49/10,241-call authorization.

One of five documents completed 16/16 accepted slots. It had 69 words and
exactly 16 sentence units, so it failed the frozen 100--220 word gate. The
other four documents exhausted slot 0 or slot 1. Thus injection completion
was 1/5, structural eligibility was 0/5, and the exported blind packet is
empty. No reviewer judgments were fabricated. The pilot continuation gate
failed before quality review, main remains unauthorized with zero calls, and
the downstream count-neutral confirmation remains blocked. No detector,
attack, shard-10 control, or mechanism run occurred.

## Length-Controlled DeepSeek Follow-up: Operationally Stopped

`experiments/swan_positive_deepseek_length_confirmation2/` is a separately
preregistered follow-up motivated by the closed pilot's 69-word failure. It
does not reuse or relabel any shard-9 output. Before seeing new model output,
the experiment froze an appended 7--12-word prompt instruction and a
candidate gate requiring 7--12 lexical words, exactly one NLTK sentence unit,
terminal punctuation, and then S2Match >=0.75.

All 35 shard-11 sources were allocated before generation as 5 pilot plus 30
main. The complete history snapshot found zero text or URL identity overlap.
The frozen pilot ceiling is 10,241 calls and $18.19 including contingency
(conservative bound $14.551215).

The pilot stopped after a journaled request returned `ReadTimeout`.
There are 607 call journals: 606 complete and one
`failed_or_uncertain`. The complete calls contain 312,462 prompt tokens and
8,962 completion tokens. Complete-call peak-price cost is $0.104493; charging
the uncertain request at its full reserve gives $0.1055448 authorization
accounting.

The 606 attempt traces comprise 605 complete candidates and one uncertain
request with no response or candidate AMR. Among the complete candidates, 359
passed the 7--12-word gate and four slots were accepted, but no document
completed 16/16 slots. Three documents are terminal injection failures, the
fourth is censored by the uncertain call, and the fifth was not started. This
is an operationally incomplete pilot, not a five-document structural-gate
estimate. The uncertain request is not retried, main remains unauthorized,
and no detector, attack, or count-neutral stage ran.

## Retrospective Length-Gate Diagnostic

`experiments/swan_length_gate_tradeoff1/` rescored all 605 available
candidate AMRs offline with the original per-attempt seeds. It reports 605
unique candidate-score observations and made zero model, parser, network,
detector, or attack calls. Deterministic implementation reruns are not treated
as new samples. All 359 historical scores reproduced exactly.

Four of 246 gate-rejected candidates had F1 >=0.75. They occur in two observed
slot prefixes: without applying the gate to the already observed prefix, the
first threshold pass would have appeared at a six-word candidate in each
slot. This is direct evidence that the strict per-slot length gate reduced
candidate and observed-prefix yield. It is not a document-level
counterfactual because accepting either candidate would have changed all
later generation.

The gate is not the sole or dominant candidate-level explanation:
355/359 gate-eligible candidates still failed S2Match, and 242/246 rejected
candidates also scored below threshold. Overlong candidates had median graph
size 29 triples, precision 0.2804, recall 0.6471, and F1 0.3889, versus 19
triples, precision 0.3636, recall 0.6667, and F1 0.4762 in the 7--12-word
group. Candidate-minus-target graph size correlated with precision at
`rho=-0.6722`, recall at `rho=0.1053`, and F1 at `rho=-0.5250`. These
descriptive results are consistent with whole-graph precision dilution for
overlong candidates. They do not isolate a prompt effect, compare models, or
show that generator choice is irrelevant.

## Resilient Length Confirmation: Historical Freeze, Now Inactive

`experiments/swan_positive_deepseek_length_confirmation3/` was prepared as a
prospective experiment on C4 realnewslike train shard 12. It preserved
all length-confirmation2 generation and quality parameters. The sole design
change is operational: an uncertain request is never retried, its full
reserve is charged, the current document becomes an operational failure in
the denominator, and later preallocated documents continue.

The complete history snapshot contains 19,407 normalized text identities and
1,398 URL identities. All 35 newly allocated identities are unique and have
zero historical overlap. Network-free tests and both prior-runner regressions
pass 36/36. The frozen pilot ceiling is 10,241 calls and $18.29 including
contingency (conservative bound $14.628629). Current DeepSeek calls,
generation attempts, and phase authorizations are all zero. Main remains
separately blocked.

## final1 Normal-Length Pilot: Branch C

`experiments/swan_count_neutral_final1/` completed all eight preregistered
pilot allocations with 3,467 fully terminal API calls and conservative
peak-price accounting of $0.5782413. Six documents passed the frozen
structural checks; two failed because their generated sentence segmentation
did not satisfy the required detection-unit structure.

The currently frozen baseline labels pass 0/6 documents. Consequently, the
transform and pair cohorts are empty, the recomputed pilot gate is
`passed=false`, and main access is blocked. No count-neutral transform,
neural AMR parse, S2Match evaluation, detector run, or confirmatory attack
test was performed. This is a generation-feasibility negative result, not a
detector or attack-effect result.

The auxiliary report discloses that the A/B labels were produced by language-
model-assisted review and do not satisfy the protocol's requirement for two
independent human reviewers. The result is therefore retained only as a
frozen-label stopping decision; it is not relabeled as formal human blinded
quality evidence.

The post-run closure audit replays generation provenance, the journal chain,
quality binding, empty transform/pair artifacts, and the failed gate. A
second no-compute recovery pass matched all five closure outputs byte for
byte. It recorded zero new API, parser, S2Match, or detector calls. The route
is archived under Branch C and must not be retuned on the same sources.

## Source-Anchored Pilot: Branch C

`experiments/swan_source_anchored1/` completed all eight allocations with
1,891 terminal calls and USD 0.4919673 peak-price accounting. Five documents
completed ten slots and passed structure. Three terminated after a slot had
no surface-qualified candidate, leaving 6, 2, and 2 detection units. The 60
completed slots all used the frozen fallback after 30 attempts; none reached
the 0.75 S2Match acceptance threshold.

The structural gate failed at 5/8. Quality review is unperformed rather than
failed, and no transforms, detector endpoints, controls, or main run exist.
Post-run hash and accounting verification passed with zero uncertain/invalid
calls. Parser diagnostics observed during execution and replay also trigger
the protocol's independent audit stop. The binding conclusion is
`experiments/swan_source_anchored1/PILOT_CONCLUSION.md`.

The no-compute closure under `pilot_closure/` independently reconnects 1,890
attempts, 1,891 responses, 60 completed fallback slots, three failed slots,
and all eight terminal documents. It checked 3,925 files and froze stable
aggregate SHA256
`6ea3550ab5dafead68abeb23334944499c2a869266ac6cb9ec8e43c9b50b404b`;
a second verify-only run reproduced all five outputs byte for byte. The audit
made zero API, neural parser, S2Match, or detector calls. It deliberately
retains `parser_warning_stop_resolved=false` because stderr diagnostics were
not persisted by the original runner.

## Archived Directions

The following are historical evidence or negative results, not active work:

- Plan E prompt expansion, best-of-k ASR optimization, and adaptive rescue;
- microfact, span compiler, dependency pruning, and natural-window searches;
- local multi-instance, template-specificity, and relation-motif defenses;
- unmatched KGW/SynthID transfer as a strong causal specificity claim;
- the superseded strict-30 paper and pre-unification plans.

Raw artifacts remain available for reproducibility. They should be opened only
when a current claim requires their exact evidence.

## Superseded Work Rule (Historical Context)

No API experiment is active. The old five-source Qwen2.5-0.5B and
DeepSeek-V4.1-Flash preflight routes are closed and must not be retuned on
their observed sources. The independent DeepSeek confirmation pilot is also
closed after its preregistered structural gate failed at 0/5. Its 30 main
sources remain frozen but unauthorized and must not be run.

The exposed old DeepSeek credential remains excluded from artifacts and
should be revoked. The official Claude path remains an alternative locally
ready route, not an active run. The frozen 30-document short cohort remains
exhausted for its prior conservative count-neutral construction: its `0/30`
coverage cannot be changed by post-hoc rule relaxation. The new A/B protocol
cannot begin because no independent main quality decision exists. The
terminated DeepSeek pilot must not be resumed or retuned on shard 9, and
historical attack optimization and failed defense branches remain closed.

The current work rule is the erratum at the top and `任务清单.md`. In
particular, the old A/B protocol and Claude alternative are not active next
steps. The mathematical N=1 maximum is sqrt(3), below 2.33; the official direct
AMR utility instead returns NaN for N<2, while its preparsed entry computes
the formula for N=1. This interface distinction is required in future reports.
