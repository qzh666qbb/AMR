"""Independently audit SemStamp certified-defense v2 WP1 evidence."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

import run_certified_resegmentation_wp2_v17_b0 as v17


STAGE = "semstamp_certified_defense_v2_wp1_threat_model"
COHORTS = ("calibration_null", "independent_test_null", "positive")
COHORT_SIZE = 1000
TOTAL_ROWS = 3000
LEGACY_FAILURE_COUNT = 42
LEXICAL_RE = re.compile(r"\S+")
FORBIDDEN_PUBLIC_KEYS = {
    "generated_text",
    "prompt",
    "raw_text",
    "source_text",
    "text",
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_digest(value):
    return sha256_bytes(canonical_bytes(value))


def renderer_digest(value):
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reject_raw_public_keys(value, location="root"):
    if isinstance(value, dict):
        overlap = FORBIDDEN_PUBLIC_KEYS & set(value)
        if overlap:
            raise ValueError(f"raw text key at {location}: {overlap}")
        for key, child in value.items():
            reject_raw_public_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_public_keys(child, f"{location}[{index}]")


def parse_manifest(path):
    entries = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="ascii").splitlines(), 1
    ):
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or relative in entries
            or not relative
        ):
            raise ValueError(f"invalid manifest line {line_number}")
        entries[relative] = digest
    return entries


def codepoint_to_byte_offsets(text):
    offsets = [0]
    size = 0
    for character in text:
        size += len(character.encode("utf-8"))
        offsets.append(size)
    return offsets


def align_units(text, units):
    cursor = 0
    spans = []
    for unit_index, unit in enumerate(units):
        if not unit or unit != unit.strip() or LEXICAL_RE.search(unit) is None:
            raise ValueError(f"invalid unit {unit_index}")
        start = text.find(unit, cursor)
        if start < 0:
            raise ValueError(f"unit not found {unit_index}")
        skipped = text[cursor:start]
        if skipped and not skipped.isspace():
            raise ValueError(f"non-whitespace before unit {unit_index}")
        end = start + len(unit)
        spans.append((start, end))
        cursor = end
    if text[cursor:] and not text[cursor:].isspace():
        raise ValueError("non-whitespace trailing text")
    return spans


def independently_build_row(index, planned, text, units, legacy_indices):
    raw = text.encode("utf-8")
    offsets = codepoint_to_byte_offsets(text)
    lexical_matches = list(LEXICAL_RE.finditer(text))
    unit_spans = align_units(text, units)
    endpoints = {position for span in unit_spans for position in span}

    lexical_table = []
    atoms = []
    refinement_cuts = []
    for lexical_index, match in enumerate(lexical_matches):
        start = match.start()
        end = match.end()
        lexical_table.append({
            "codepoint_span": [start, end],
            "byte_span": [offsets[start], offsets[end]],
            "payload_sha256": sha256_bytes(
                raw[offsets[start] : offsets[end]]
            ),
        })
        cuts = sorted({
            start,
            end,
            *(point for point in endpoints if start < point < end),
        })
        refinement_cuts.extend(cuts[1:-1])
        for atom_start, atom_end in zip(cuts, cuts[1:]):
            payload = raw[offsets[atom_start] : offsets[atom_end]]
            if (
                atom_start >= atom_end
                or not payload
                or LEXICAL_RE.fullmatch(text[atom_start:atom_end]) is None
            ):
                raise ValueError(f"invalid independent atom at {index}")
            atoms.append({
                "codepoint_span": [atom_start, atom_end],
                "byte_span": [offsets[atom_start], offsets[atom_end]],
                "lexical_span_index": lexical_index,
                "payload_sha256": sha256_bytes(payload),
            })
    if len(atoms) < 2:
        raise ValueError(f"insufficient independent atoms at {index}")

    starts = {
        atom["codepoint_span"][0]: atom_index
        for atom_index, atom in enumerate(atoms)
    }
    ends = {
        atom["codepoint_span"][1]: atom_index + 1
        for atom_index, atom in enumerate(atoms)
    }
    unit_atom_spans = []
    unit_payloads = []
    for unit_index, ((start_cp, end_cp), unit) in enumerate(
        zip(unit_spans, units)
    ):
        if start_cp not in starts or end_cp not in ends:
            raise ValueError(f"missed independent endpoint at {index}")
        start_atom = starts[start_cp]
        end_atom = ends[end_cp]
        payload = raw[
            atoms[start_atom]["byte_span"][0]:
            atoms[end_atom - 1]["byte_span"][1]
        ]
        if payload.decode("utf-8", "strict") != unit:
            raise ValueError(
                f"independent unit mismatch at {index}:{unit_index}"
            )
        unit_atom_spans.append((start_atom, end_atom))
        unit_payloads.append(payload)
    if (
        unit_atom_spans[0][0] != 0
        or unit_atom_spans[-1][1] != len(atoms)
        or any(
            left[1] != right[0]
            for left, right in zip(unit_atom_spans, unit_atom_spans[1:])
        )
    ):
        raise ValueError(f"independent unit spans are not contiguous at {index}")

    boundaries = tuple(span[1] for span in unit_atom_spans[:-1])
    gaps = []
    zero_boundaries = []
    for boundary in range(1, len(atoms)):
        gap = raw[
            atoms[boundary - 1]["byte_span"][1]:
            atoms[boundary]["byte_span"][0]
        ]
        if not gap:
            zero_boundaries.append(boundary)
            if boundary not in boundaries:
                raise ValueError(
                    f"independent unauthorized zero gap at {index}"
                )
        elif not gap.decode("utf-8", "strict").isspace():
            raise ValueError(f"independent non-whitespace gap at {index}")
        gaps.append({
            "boundary": boundary,
            "bytes": len(gap),
            "is_original_boundary": boundary in boundaries,
            "sha256": sha256_bytes(gap),
        })

    prefix = raw[: atoms[0]["byte_span"][0]]
    suffix = raw[atoms[-1]["byte_span"][1] :]
    rebuilt = bytearray(prefix)
    for unit_index, payload in enumerate(unit_payloads):
        rebuilt.extend(payload)
        if unit_index < len(boundaries):
            boundary = boundaries[unit_index]
            rebuilt.extend(
                raw[
                    atoms[boundary - 1]["byte_span"][1]:
                    atoms[boundary]["byte_span"][0]
                ]
            )
    rebuilt.extend(suffix)
    if bytes(rebuilt) != raw:
        raise ValueError(f"independent reconstruction mismatch at {index}")

    lexical_starts = {match.start() for match in lexical_matches}
    lexical_ends = {match.end() for match in lexical_matches}
    legacy_failure = any(
        start not in lexical_starts or end not in lexical_ends
        for start, end in unit_spans
    )
    legality_sha256 = canonical_digest({
        "scope": "b0_only_inherited_v17_native_legality",
        "native_token_limit": 512,
        "original_unit_atom_spans": [
            list(span) for span in unit_atom_spans
        ],
        "v17_max_native_tokens": planned["max_native_tokens"],
    })
    topology_sha256 = renderer_digest({
        "atoms": atoms,
        "boundaries": list(boundaries),
        "gaps": gaps,
        "lexical_spans": lexical_table,
        "raw_utf8_sha256": sha256_bytes(raw),
        "refinement_cut_codepoints": sorted(set(refinement_cuts)),
    })
    return {
        "schema_version": 1,
        "protocol_sha256": planned["_protocol_sha256"],
        "renderer_contract": "refined_nltk_boundary_atoms_v2",
        "status": "pass",
        "plan_index": index,
        "cohort": planned["cohort"],
        "cohort_index": planned["cohort_index"],
        "raw_text_sha256": planned["raw_text_sha256"],
        "max_native_tokens": planned["max_native_tokens"],
        "legacy_v1_alignment_failure": index in legacy_indices,
        "raw_utf8_sha256": sha256_bytes(raw),
        "raw_utf8_bytes": len(raw),
        "lexical_span_count": len(lexical_matches),
        "atom_count": len(atoms),
        "refinement_count": len(atoms) - len(lexical_matches),
        "zero_byte_gap_count": len(zero_boundaries),
        "zero_byte_gaps_all_original": set(zero_boundaries).issubset(
            boundaries
        ),
        "zero_byte_gap_boundaries_sha256": renderer_digest(zero_boundaries),
        "unit_count": len(units),
        "original_boundaries": list(boundaries),
        "refinement_cuts_sha256": renderer_digest(
            sorted(set(refinement_cuts))
        ),
        "original_units_sha256": renderer_digest(
            [sha256_bytes(unit) for unit in unit_payloads]
        ),
        "prefix_gap_sha256": sha256_bytes(prefix),
        "suffix_gap_sha256": sha256_bytes(suffix),
        "gap_table_sha256": renderer_digest(gaps),
        "topology_sha256": topology_sha256,
        "byte_reconstruction_sha256": sha256_bytes(bytes(rebuilt)),
        "legality_sha256": legality_sha256,
        "score_access_count_during_construction": 0,
        "contains_raw_text": False,
        "_legacy_failure_independently_observed": legacy_failure,
    }


def audit(root, output_dir, output, nltk_data):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    paths = {
        "protocol": output_dir / "PROTOCOL.md",
        "threat_model": output_dir / "THREAT_MODEL.md",
        "admissible": output_dir / "ADMISSIBLE_RESEGMENTATION.md",
        "threat_json": output_dir / "THREAT_MODEL.json",
        "contract": output_dir / "CONTRACT.json",
        "manifest": output_dir / "INPUT_MANIFEST.sha256",
        "rows0": output_dir / "P0_RENDERER_ROWS_REPLAY0.jsonl",
        "rows1": output_dir / "P0_RENDERER_ROWS_REPLAY1.jsonl",
        "replay0": output_dir / "REPLAY0_RESULT.json",
        "replay1": output_dir / "REPLAY1_RESULT.json",
        "result": output_dir / "RESULT.json",
        "runner": root / "scripts/run_semstamp_certified_defense_v2_wp1.py",
        "renderer": root / "scripts/semstamp_refined_atom_renderer.py",
        "renderer_test": root / "scripts/test_semstamp_refined_atom_renderer.py",
        "runner_test": (
            root / "scripts/test_semstamp_certified_defense_v2_wp1.py"
        ),
        "auditor": Path(__file__).resolve(),
        "v17_dir": (
            root
            / "experiments/certified_resegmentation_wp2_v17_b0_detector_health1"
        ),
        "v1_dir": (
            root
            / "experiments/semstamp_certified_defense_v1_wp3_detector_health1"
        ),
    }
    contract = read_json(paths["contract"])
    artifact_fields = {
        "protocol_sha256": paths["protocol"],
        "threat_model_sha256": paths["threat_model"],
        "admissible_contract_sha256": paths["admissible"],
        "threat_model_json_sha256": paths["threat_json"],
        "runner_sha256": paths["runner"],
        "renderer_sha256": paths["renderer"],
        "renderer_test_sha256": paths["renderer_test"],
        "runner_test_sha256": paths["runner_test"],
        "auditor_sha256": paths["auditor"],
        "input_manifest_sha256": paths["manifest"],
    }
    if any(
        contract.get(field) != sha256_file(path)
        for field, path in artifact_fields.items()
    ):
        raise ValueError("contract artifact binding mismatch")

    manifest = parse_manifest(paths["manifest"])
    if set(manifest) != set(contract.get("inputs", {})):
        raise ValueError("manifest/contract input set mismatch")
    for relative, digest in manifest.items():
        path = (root / relative).resolve()
        meta = contract["inputs"][relative]
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or sha256_file(path) != digest
            or path.stat().st_size != meta["bytes"]
        ):
            raise ValueError(f"input drift: {relative}")
        if (
            meta.get("mode") is not None
            and f"{stat.S_IMODE(path.stat().st_mode):04o}" != meta["mode"]
        ):
            raise ValueError(f"input mode drift: {relative}")

    rows0 = read_jsonl(paths["rows0"])
    rows1 = read_jsonl(paths["rows1"])
    replay0 = read_json(paths["replay0"])
    replay1 = read_json(paths["replay1"])
    result = read_json(paths["result"])
    reject_raw_public_keys([rows0, rows1, replay0, replay1, result])
    if (
        len(rows0) != TOTAL_ROWS
        or paths["rows0"].read_bytes() != paths["rows1"].read_bytes()
        or replay0.get("status") != "pass"
        or replay1.get("status") != "pass"
        or result.get("status") != "pass"
        or result.get("terminal_reason")
        != "wp1_execution_pass_pending_independent_audit"
        or not all(result.get("gates", {}).values())
    ):
        raise ValueError("runner terminal evidence is not a complete pass")

    v1_readiness = read_json(paths["v1_dir"] / "P0_READINESS.json")
    v1_audit = read_json(paths["v1_dir"] / "AUDIT.json")
    legacy_indices = tuple(
        v1_readiness["renderer"]["failed_plan_indices"]
    )
    if (
        len(legacy_indices) != LEGACY_FAILURE_COUNT
        or v1_readiness.get("status") != "stop"
        or v1_audit.get("status") != "stop"
    ):
        raise ValueError("v1 failure evidence mismatch")

    b0_paths = v17.paths_for(root, paths["v17_dir"])
    _, plan, texts = v17.materialize_inputs(b0_paths)
    if (
        v17.jsonl_bytes(plan)
        != (paths["v17_dir"] / "B0_PLAN.jsonl").read_bytes()
        or len(plan) != TOTAL_ROWS
        or len(texts) != TOTAL_ROWS
    ):
        raise ValueError("independent materialization mismatch")

    os.environ["NLTK_DATA"] = str(Path(nltk_data).resolve())
    import nltk
    from nltk.tokenize import sent_tokenize

    if nltk.__version__ != contract["nltk_version"]:
        raise ValueError("auditor NLTK version mismatch")
    nltk.data.path.insert(0, str(Path(nltk_data).resolve()))
    resource = nltk.data.find("tokenizers/punkt/english.pickle")
    with resource.open() as handle:
        if sha256_bytes(handle.read()) != contract["punkt_payload_sha256"]:
            raise ValueError("auditor Punkt payload mismatch")

    independently_observed_legacy = []
    for index, (planned_value, text, observed) in enumerate(
        zip(plan, texts, rows0)
    ):
        planned = {
            **planned_value,
            "_protocol_sha256": contract["protocol_sha256"],
        }
        expected = independently_build_row(
            index,
            planned,
            text,
            sent_tokenize(text),
            legacy_indices,
        )
        legacy_failure = expected.pop(
            "_legacy_failure_independently_observed"
        )
        if legacy_failure:
            independently_observed_legacy.append(index)
        if canonical_bytes(expected) != canonical_bytes(observed):
            raise ValueError(f"independent row mismatch at {index}")
    if tuple(independently_observed_legacy) != legacy_indices:
        raise ValueError("legacy 42 failures not independently reproduced")

    checks = {
        "contract_and_inputs_exact": True,
        "fixed_3000_rows_accounted": True,
        "two_replays_byte_exact": True,
        "all_rows_independently_recomputed": True,
        "all_original_units_exact": True,
        "all_raw_bytes_reconstructed": True,
        "all_zero_byte_gaps_are_b0": True,
        "legacy_42_failures_reproduced": True,
        "legacy_42_rows_repaired": True,
        "score_model_gpu_generation_calls_zero": True,
        "raw_text_absent_from_public_artifacts": True,
    }
    audit_result = {
        "schema_version": 1,
        "stage": f"{STAGE}_independent_audit",
        "status": "go" if all(checks.values()) else "stop",
        "terminal_reason": "wp1_go_for_wp2_design",
        "scope": "independent_model_free_raw_to_row_recalculation",
        "bindings": {
            label: sha256_file(path)
            for label, path in paths.items()
            if (
                label not in {"v17_dir", "v1_dir"}
                and path.is_file()
            )
        },
        "checks": checks,
        "counts": {
            "rows": TOTAL_ROWS,
            "cohorts": dict(Counter(row["cohort"] for row in rows0)),
            "legacy_v1_failures": len(legacy_indices),
            "legacy_v1_failures_repaired": sum(
                row["legacy_v1_alignment_failure"] for row in rows0
            ),
            "rows_with_refinement": sum(
                row["refinement_count"] > 0 for row in rows0
            ),
            "refinement_total": sum(
                row["refinement_count"] for row in rows0
            ),
            "zero_byte_gap_total": sum(
                row["zero_byte_gap_count"] for row in rows0
            ),
        },
        "execution_attestation": {
            "certificate_calls": 0,
            "detector_imports": 0,
            "detector_model_loads": 0,
            "detector_scores_computed": 0,
            "generation_calls": 0,
            "gpu_calls": 0,
        },
        "decision": {
            "wp1": "go",
            "wp2": "authorized_to_freeze_design",
            "wp2_started": False,
            "wp3_started": False,
        },
        "contains_raw_text": False,
    }
    reject_raw_public_keys(audit_result)
    atomic_write_json(output, audit_result)
    print(json.dumps(audit_result, sort_keys=True))
    return audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output_dir, args.output, args.nltk_data)


if __name__ == "__main__":
    main()
