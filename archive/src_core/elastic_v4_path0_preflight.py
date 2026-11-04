"""Executable P0 checks for the two-arm v4 path0 development runner."""
from __future__ import annotations

import math
import random

from .bank_scorer import template_features
from .elastic_v3 import ElasticBoundaryProposerV3, SwanAMRBankScorerV3
from .elastic_v4 import HopPenalizedOptimizer, SwanBankLatticeDetectorV4
from .elastic_v4_path0 import AllBankScorer, CheckedHopOptimizer, EmptyPathZeroScorer
from .models import BankMatch, Boundary, Lattice, LatticeEdge, DetectionResult
from .null_calibration import EmpiricalNullCalibrator


def run_p0(bank: str) -> dict:
    control = AllBankScorer(None, bank)
    old = SwanAMRBankScorerV3(None, bank)
    candidate = EmptyPathZeroScorer(None, bank)
    assert control.fingerprint == old.fingerprint
    assert candidate.fingerprint != control.fingerprint
    fixtures = (
        "(a / act-01 :ARG0 (b / boy))",
        "(a / act-01 :ARG0 (b / boy :mod (c / clever)))",
        "(a / act-01 :ARG0 (b / boy) :ARG1 (c / car))",
        "(a / boy)",
    ) + control.templates
    comparisons = 0
    for graph in fixtures:
        features = template_features(graph)
        assert control.score_features(features) == old.score_features(features)
        for index, (left, right) in enumerate(zip(
            control.all_matches(features), candidate.all_matches(features)
        )):
            assert left.concept_score == right.concept_score
            assert left.relation_score == right.relation_score
            empty_pair = not features.paths and not control._bank_features[index].paths
            assert right.path_score == (0.0 if empty_pair else left.path_score)
            assert abs(left.score - right.score - (0.25 if empty_pair else 0)) < 1e-12
            comparisons += 1
    builder = ElasticBoundaryProposerV3()
    detectors = {
        "control": SwanBankLatticeDetectorV4(builder, control),
        "path0": SwanBankLatticeDetectorV4(builder, candidate),
    }
    assert detectors["control"].fingerprint != detectors["path0"].fingerprint

    rng = random.Random(19019)
    optimizer = CheckedHopOptimizer()
    oracle_cases = 0
    for size in range(2, 8):
        edges = tuple(
            LatticeEdge(start, end, f"{start}:{end}", end - start)
            for start in range(size)
            for end in range(start + 1, min(size, start + 4))
        )
        lattice = Lattice("p0", tuple(Boundary(i, i, "p0") for i in range(size)), edges)
        for _ in range(40):
            matches = tuple(BankMatch(i, rng.random(), 0, 0, 0) for i in range(len(edges)))
            valid = tuple(rng.random() > .25 for _ in edges)
            paths = []

            def visit(node, utility, count):
                if node == size - 1:
                    paths.append(utility / math.sqrt(count))
                    return
                for edge, match, allowed in zip(edges, matches, valid):
                    if edge.start == node and allowed:
                        penalty = .18 if edge.boundary_hops >= 2 else 0
                        visit(edge.end, utility + match.score - .65 - penalty, count + 1)

            visit(0, 0, 0)
            result = optimizer.optimize(lattice, matches, valid)
            assert result == HopPenalizedOptimizer(.65, .75).optimize(lattice, matches, valid)
            if paths:
                assert not result.abstained
                assert abs(result.statistic - max(paths)) < 1e-12
                allowed_edges = {edge for edge, allowed in zip(edges, valid) if allowed}
                assert all(unit.edge in allowed_edges for unit in result.units)
            else:
                assert result.abstained and result.statistic is None
            oracle_cases += 1
    edge = LatticeEdge(0, 1, "p0", 1)
    lattice = Lattice("p0", (Boundary(0, 0, "p0"), Boundary(1, 2, "p0")), (edge,))
    for score in (math.nan, math.inf, -math.inf):
        try:
            optimizer.optimize(lattice, (BankMatch(0, score, 0, 0, 0),), (True,))
        except ValueError:
            pass
        else:
            raise AssertionError("P0 failed to reject a non-finite score")
    try:
        optimizer.optimize(lattice, (), (True,))
    except ValueError:
        pass
    else:
        raise AssertionError("P0 failed cardinality contract")
    calibrator = EmpiricalNullCalibrator.fit(
        [("alpha", DetectionResult(0, 0, 1, 0, ()))],
        detectors["control"].fingerprint, control.fingerprint, control.bank_sha256, "p0-parser",
    )
    try:
        calibrator.assert_compatible(
            detectors["path0"].fingerprint, candidate.fingerprint, "p0-parser"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("P0 accepted the opposite arm calibration")
    return {
        "status": "pass",
        "all_bank_component_comparisons": comparisons,
        "dp_oracle_cases": oracle_cases,
        "control_original_identity": True,
        "different_candidate_identity": True,
        "nonfinite_rejection": True,
        "invalid_edge_and_abstain_contract": True,
        "calibration_cross_arm_rejection": True,
    }
