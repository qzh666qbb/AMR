# Quality and Failure Analysis

This is an AI-assisted pre-analysis of 100 pairs, not a substitute for
independent human blind annotation.

The user independently reviewed the assessment and confirmed that the
judgments match their evaluation. This is recorded as user-confirmed expert
review; it remains distinct from a multi-rater blind annotation study.

## Quality summary

| Method | Mean quality | Quality >= 4 | Major-error rate |
|---|---:|---:|---:|
| Plain paraphrase | 4.43 | 79.4% | 0/34 (0%) |
| Sentence boundary | 4.24 | 72.7% | 1/33 (3.0%) |
| AMR guided | 3.63 | 57.6% | 3/33 (9.1%) |

Among high-quality (mean score >= 4) samples, the observed 1% FPR escape rates
were 8.8%, 27.3%, and 24.2% respectively. This supports sentence-boundary
rewriting as the best quality/evasion compromise in this sample.

## Failure taxonomy

The recurring failure modes are:

1. **Repetition / template echo** — repeated names, instructions, or vague
   phrases; present across all methods and especially common in the generated
   baseline itself.
2. **Fragmentation** — incomplete clauses, awkward sentence boundaries, or
   ungrammatical fragments; more common in sentence-boundary and AMR-guided
   outputs.
3. **Vagueness / information loss** — specific content replaced with
   “something”, “thing”, or generic roles.
4. **Unsupported causal or relational change** — a candidate changes a state
   into a causal claim or changes who did what.
5. **Parser-shaped unnatural wording** — passive or nominal structures that
   appear chosen to alter AMR structure but reduce readability.

The most serious examples are AMR-guided cases 15, 75, and 87 in the AI
evaluation file: subject/object inversion, an unsupported causal relation, and
semantically nonsensical phrasing. Sentence-boundary case 95 is a severe
repetition/incoherence failure.

## Decision

Do not count a detector escape as a final Valid ASR unless it passes the
quality checks. Human review should prioritize all mean-score < 3.5 cases,
all major-error cases, and a random sample of passing cases. The current
automated results are suitable for prioritization and error discovery, not for
the final human-quality claim.
