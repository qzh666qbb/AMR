# Related Work

> SUPERSEDED / ERRATUM 2026-09-18: this historical framing is inactive.
> SWAN reports empirical robustness on RealNews and selected paraphrasers;
> those results are not merely an untested assumption. Neither semantic
> watermark superiority nor real-text parser/matcher causality is established
> by the old draft. Current claims: root CLAIM_EVIDENCE_MATRIX.md.

## AMR as a semantic representation

Abstract Meaning Representation (AMR) is a rooted semantic graph formalism designed to abstract away from surface syntax while retaining predicate-argument structure, named entities, and other semantically salient relations. The original AMR sembanking paper established the representation as a practical semantic annotation layer for sentence-level meaning capture and generation research.

For this paper, the key point is not only that AMR is expressive, but that it is also normalized enough to support detector design. That makes AMR a natural target for structure-aware perturbation: surface changes may leave the detector unchanged, but structured changes can alter the graph alignment logic that the detector depends on.

## AMR evaluation metrics

AMR research relies heavily on graph-matching metrics. Smatch is the canonical evaluation metric for AMR graphs; it estimates the maximum overlap between two semantic graphs by aligning variables and counting matched triples. Its appeal is simplicity, but its hard matching behavior means that small structural differences can produce large score shifts.

S2Match softens this picture by allowing graded similarity between concept nodes and aiming for a more semantically tolerant comparison. In the context of this paper, S2Match is important for a different reason: it exposes the detector to a score landscape that can vary under repeated recomputation, so stability analysis is necessary. The repeatability experiments in this paper show that score jitter can exist even when binary escape decisions remain stable.

## Semantic watermarking with AMR and beyond

SWAN is the most direct prior work for this paper. It encodes watermark signatures directly into AMR structure, then detects them using an off-the-shelf AMR parser and a one-proportion z-test. The central claim of SWAN is paraphrase robustness: if meaning is preserved, the semantic signature is expected to persist.

The broader watermarking literature makes the same move at different semantic granularities. SemStamp injects a sentence-level semantic watermark using locality-sensitive hashing over sentence embeddings, while later work has shown that paraphrase robustness can still be reverse-engineered or otherwise stressed by stronger attacks. These results reinforce a common lesson: semantic watermarking is more robust than token-level watermarking, but it is not immune to adversarial rewriting.

That design also creates an attack surface. If a rewriting method can preserve enough quality while changing the AMR parse or the distribution of AMR matches, then the detector can be weakened without necessarily introducing obvious token-level artifacts. The current paper studies exactly that setting: candidate rewrites are ranked by detector feedback, then filtered by conservative validity constraints and analyzed for threshold sensitivity and stability.

## Positioning of this paper

This paper differs from prior AMR work in three ways.

First, it treats the AMR detector as a target system rather than only as an evaluation object. Second, it separates raw detector escape from strict valid ASR, so high escape rates are not conflated with usable attacks. Third, it adds robustness analysis---threshold plateaus and repeated recomputation---to distinguish a genuine method from a one-off favorable score.

The contribution is not a new AMR metric or a new parser. It is an empirical attack and robustness study showing that AMR-level watermarking can be weakened by feedback-driven rewrites, but that robust claims require stricter validity accounting than raw escape rates alone.
