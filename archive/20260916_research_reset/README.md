# Research Reset Archive Index

> **Superseded notice (2026-09-18):** This index describes the 2026-09-16
> reset and is no longer the active workspace authority. Use
> `../../README.md` and `../../任务清单.md`. In particular, deterministic AMR
> composition is now limited to an S2Match implementation diagnostic and must
> not be used as a natural-text causal counterfactual.

Archive date: 2026-09-16

This index records what was removed from the active research context. It does
not delete raw data or alter historical results.

## Active Authority

1. `../../RESEARCH_STATE.md`: current question, surviving evidence, open
   questions, and restart rule.
2. `../../CURRENT_RESULTS.md`: detailed historical evidence ledger.
3. Individual experiment `protocol`, `summary`, `audit`, and raw checkpoint
   files: reproducibility sources.

## Confirmed Core Evidence

- `experiments/planE_e2_joint250/`: historical practical attack effectiveness.
- `experiments/confirmation_three_arm30/`: independent short-text comparison.
- `experiments/confirmation_mechanism30/`: punctuation-only and exact-boundary
  restoration evidence.
- `experiments/swan_amr_composition_decomposition1/` and
  `experiments/swan_amr_precision_recall1/`: saved graph decomposition.
- `experiments/literature_novelty_audit1/`: novelty boundary and nearest work.

These artifacts may support current claims, subject to the limitations in
`RESEARCH_STATE.md`.

## Supporting Or Exploratory Evidence

- `experiments/segmentation_intervention/`
- `experiments/sira_adapted30/`
- `experiments/cross_watermark_transfer30/`
- `experiments/e5_transfer/`
- `experiments/feedback2_diagnostics/`
- `experiments/swan_sentence_growth1/`
- `experiments/swan_generator_comparison1/`
- `experiments/grounded_generation3/`
- `experiments/grounded_injection5/`
- `experiments/swan_known_positive1/`
- `experiments/swan_oracle_composition1/`
- `experiments/swan_partial_coverage_power1/`

These results can motivate limitations or protocol choices. They are not
independent final confirmation.

## Closed Negative Branches

- `experiments/swan_atomic_fact_preflight1/`
- `experiments/swan_microfact_generation1/`
- `experiments/swan_microfact_sequential2/`
- `experiments/swan_bank_scale_pilot1/`
- `experiments/swan_bank100_null_calibration1/`
- `experiments/swan_bank100_microfact_null1/`
- `experiments/swan_constrained_microfact1/`
- `experiments/swan_deepseek_span_compiler1/`
- `experiments/swan_dependency_pruner1/`
- `experiments/swan_natural_mixed_windows1/`
- `experiments/swan_natural_narrative_windows2/`
- `experiments/swan_c4_train_narrative_windows3/`
- `experiments/swan_quality_first_windows4/`
- `experiments/swan_local_multi_instance1/`
- `experiments/swan_local_multi_instance2/`
- `experiments/swan_template_specificity1/`
- `experiments/swan_relation_motif1/`

Do not reopen these branches by loosening their gates or increasing search
budgets on the same development assets.

## Superseded Narratives And Code

`../20260911_pre_unification/` contains the old strict-30 paper, phased plans,
reports, presentation material, and scripts tied to superseded narratives.
The corresponding Git deletions are intentional archival work and must not be
reverted.

## Terminated Frozen Run

`experiments/swan_final_confirmation1/` is transparently terminated as an
incomplete feasibility run, not an experimental result. It has 66 complete
baselines, 558 completed slot checkpoints, 27 accepted slots, zero
mechanically eligible baselines, and 5,510 complete API records. The
registered 250-source endpoint was not reached, so Gate A was not evaluated.
See its `PAUSED.md` and `../../CLAIM_EVIDENCE_MATRIX.md`.
