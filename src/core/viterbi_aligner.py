from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .models import AlignmentResult, AlignmentStep, Lattice, LatticeEdge


class ViterbiAligner:
    """Elastic DAG alignment with bounded insertion/deletion penalties."""

    def __init__(
        self,
        insertion_penalty: float = 0.35,
        deletion_penalty: float = 0.35,
    ) -> None:
        if insertion_penalty < 0 or deletion_penalty < 0:
            raise ValueError("Alignment penalties must be non-negative")
        self.insertion_penalty = insertion_penalty
        self.deletion_penalty = deletion_penalty

    def align(
        self,
        lattice: Lattice,
        target_sentences: Sequence[str],
        scorer,
    ) -> AlignmentResult:
        if lattice.node_count == 0:
            raise ValueError("Lattice has no nodes")

        if hasattr(scorer, "prepare"):
            scorer.prepare(
                [edge.span_text for edge in lattice.edges]
                + list(target_sentences)
            )

        outgoing: Dict[int, List[LatticeEdge]] = {
            index: [] for index in range(lattice.node_count)
        }
        for edge in lattice.edges:
            outgoing[edge.start].append(edge)

        target_count = len(target_sentences)
        negative_infinity = float("-inf")
        dp = [
            [negative_infinity] * (target_count + 1)
            for _ in range(lattice.node_count)
        ]
        back: List[List[Optional[Tuple[int, int, AlignmentStep]]]] = [
            [None] * (target_count + 1)
            for _ in range(lattice.node_count)
        ]
        dp[0][0] = 0.0

        def relax(
            end_node: int,
            end_target: int,
            value: float,
            previous_node: int,
            previous_target: int,
            step: AlignmentStep,
        ) -> None:
            if value > dp[end_node][end_target]:
                dp[end_node][end_target] = value
                back[end_node][end_target] = (
                    previous_node,
                    previous_target,
                    step,
                )

        for node in range(lattice.node_count):
            for target_index in range(target_count + 1):
                current = dp[node][target_index]
                if current == negative_infinity:
                    continue

                if target_index < target_count:
                    relax(
                        node,
                        target_index + 1,
                        current - self.deletion_penalty,
                        node,
                        target_index,
                        AlignmentStep(
                            operation="delete_target",
                            edge=None,
                            target_index=target_index,
                            match_score=0.0,
                        ),
                    )

                for edge in outgoing[node]:
                    relax(
                        edge.end,
                        target_index,
                        current - self.insertion_penalty,
                        node,
                        target_index,
                        AlignmentStep(
                            operation="insert_span",
                            edge=edge,
                            target_index=None,
                            match_score=0.0,
                        ),
                    )
                    if target_index < target_count:
                        match_score = float(
                            scorer.score(
                                edge.span_text,
                                target_sentences[target_index],
                            )
                        )
                        relax(
                            edge.end,
                            target_index + 1,
                            current + match_score,
                            node,
                            target_index,
                            AlignmentStep(
                                operation="match",
                                edge=edge,
                                target_index=target_index,
                                match_score=match_score,
                            ),
                        )

        final_node = lattice.node_count - 1
        raw_score = dp[final_node][target_count]
        if raw_score == negative_infinity:
            return AlignmentResult(
                raw_score=raw_score,
                normalized_score=raw_score,
                steps=(),
            )

        steps: List[AlignmentStep] = []
        node = final_node
        target_index = target_count
        while node != 0 or target_index != 0:
            pointer = back[node][target_index]
            if pointer is None:
                raise RuntimeError("Alignment backtrace is incomplete")
            previous_node, previous_target, step = pointer
            steps.append(step)
            node, target_index = previous_node, previous_target
        steps.reverse()

        denominator = max(target_count, 1)
        return AlignmentResult(
            raw_score=raw_score,
            normalized_score=raw_score / denominator,
            steps=tuple(steps),
        )
