from __future__ import annotations

import time

from .lattice_builder import LatticeBuilder
from .mock_scorer import MockScorer
from .viterbi_aligner import ViterbiAligner


def run_poc_test() -> None:
    target_sentences = [
        "The president addressed the nation today.",
        "He spoke about the upcoming economic reforms.",
        "Citizens watched the broadcast closely.",
        "The reaction was generally positive.",
    ]
    attacked_text = (
        "The president addressed the nation today and spoke about the upcoming "
        "economic reforms. Citizens watched; the broadcast was followed "
        "closely. The reaction was generally positive."
    )

    builder = LatticeBuilder(max_boundary_hops=3)
    scorer = MockScorer()
    aligner = ViterbiAligner()

    started = time.monotonic()
    lattice = builder.build_lattice(attacked_text)
    result = aligner.align(lattice, target_sentences, scorer)
    elapsed_ms = (time.monotonic() - started) * 1000

    print("=== Phase 1 elastic-alignment PoC ===")
    print(
        f"Lattice: {lattice.node_count} nodes, {len(lattice.edges)} edges; "
        f"score={result.normalized_score:.4f}; elapsed={elapsed_ms:.2f} ms"
    )
    for step in result.steps:
        edge_text = step.edge.span_text if step.edge else "-"
        print(
            f"{step.operation:13s} target={step.target_index!s:>4s} "
            f"score={step.match_score:.3f} span={edge_text}"
        )


if __name__ == "__main__":
    run_poc_test()
