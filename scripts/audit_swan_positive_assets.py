"""Offline audit of public and local SWAN positive-text assets."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/swan_positive_asset_audit1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def write_frozen_json(path, payload):
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Frozen artifact differs: {path}")
    path.write_text(text)


def write_frozen_text(path, text):
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Frozen artifact differs: {path}")
    path.write_text(text)


def shasum_listing_digest(paths):
    lines = "".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n"
        for path in sorted(paths)
    )
    return hashlib.sha256(lines.encode()).hexdigest()


def verify_inputs(manifest):
    checked = {}
    for relative, expected in manifest["local_evidence"].items():
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Input hash changed: {relative}")
        checked[relative] = actual

    final = manifest["final_baseline_collection"]
    paths = sorted((ROOT / final["directory"]).glob("*.json"))
    if len(paths) != final["file_count"]:
        raise ValueError("Final baseline file count changed")
    if shasum_listing_digest(paths) != final["shasum_listing_sha256"]:
        raise ValueError("Final baseline collection changed")
    return checked


def verify_official_snapshot(manifest):
    snapshot = manifest["official_snapshot"]
    paths = snapshot["tracked_paths"]
    if (
        snapshot["main_commit"]
        != "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
        or snapshot["recursive_tree_truncated"]
        or len(paths) != snapshot["tracked_file_count"]
        or len(paths) != len(set(paths))
        or paths != sorted(paths)
        or snapshot["release_count"] != 0
        or snapshot["tag_count"] != 0
        or snapshot["repository_has_downloads"]
        or len(snapshot["reachable_commits"])
        != snapshot["reachable_commit_count"]
    ):
        raise ValueError("Official snapshot metadata is inconsistent")

    generated_names = {
        "sampled_data.json",
        "parsed_amrs.json",
        "z_scores.csv",
    }
    generated_paths = [
        path
        for path in paths
        if path.startswith("output/")
        or Path(path).name in generated_names
        or "watermarked" in Path(path).name.lower()
    ]
    json_paths = [path for path in paths if path.endswith(".json")]
    if (
        generated_paths
        or json_paths != snapshot["data_bearing_paths"]
        or "injection/watermark_generation.py" not in paths
        or "utils/load_c4_realnews_data.py" not in paths
    ):
        raise ValueError("Official tree classification changed")

    return {
        "main_commit": snapshot["main_commit"],
        "tracked_files": len(paths),
        "data_bearing_files": len(json_paths),
        "generated_watermarked_text_files": len(generated_paths),
        "releases": snapshot["release_count"],
        "tags": snapshot["tag_count"],
        "reachable_commits": snapshot["reachable_commit_count"],
        "result": snapshot["result"],
    }


def build_inventory(manifest):
    short = load_json(
        ROOT / "experiments/confirmation_three_arm30/final_analysis.json"
    )
    growth = load_json(
        ROOT / "experiments/swan_sentence_growth1/summary.json"
    )
    comparison = load_json(
        ROOT / "experiments/swan_generator_comparison1/summary.json"
    )
    grounded = load_json(
        ROOT / "experiments/grounded_injection5/summary.json"
    )
    quality = load_json(
        ROOT / "experiments/swan_quality_first_windows4/summary.json"
    )
    quality_manifest = load_json(
        ROOT / "experiments/swan_quality_first_windows4/manifest.json"
    )
    oracle = load_json(
        ROOT / "experiments/swan_oracle_composition1/summary.json"
    )
    final_sources = load_json(
        ROOT / "experiments/swan_final_confirmation1/source_manifest.json"
    )
    known = load_json(
        ROOT / "experiments/swan_known_positive1/summary.json"
    )

    growth_normal = sum(
        100 <= row["words"] <= 220 and row["detection"]["n"] >= 8
        for row in growth["prefixes"].values()
    )
    growth_quality = sum(
        bool(row["feasibility_gate"])
        for row in growth["prefixes"].values()
    )

    watermarked_arms = [
        comparison["arms"][name] for name in ("upstream", "local_adapted")
    ]
    comparison_normal = sum(
        100 <= row["words"] <= 220 and row["detection"]["n"] >= 8
        for row in watermarked_arms
    )
    comparison_quality = sum(
        bool(row["gates"]["source_quality_pass"])
        for row in comparison["arms"].values()
    )

    grounded_rows = [
        row for row in grounded["records"] if row["arm"] == "swan_grounded"
    ]
    grounded_length = sum(
        100 <= row["words"] <= 220 for row in grounded_rows
    )
    grounded_normal = sum(
        100 <= row["words"] <= 220 and row["detection"]["n"] >= 8
        for row in grounded_rows
    )

    quality_windows = {
        row["prompt_id"]: row for row in quality["documents"]
    }
    manifest_windows = {
        row["prompt_id"]: row for row in quality_manifest["sources"]
    }
    if set(quality_windows) != set(manifest_windows):
        raise ValueError("Quality-window source identities changed")
    quality_normal = sum(
        100 <= row["words"] <= 220
        and len(manifest_windows[prompt_id]["window"]["sentences"]) >= 8
        for prompt_id, row in quality_windows.items()
    )

    oracle_normal = sum(
        100 <= row["words"] <= 220 and row["original_units"] >= 8
        for row in oracle["groups"]
    )

    final_baselines = [
        load_json(path)
        for path in sorted(
            (ROOT / "experiments/swan_final_confirmation1/baselines").glob(
                "*.json"
            )
        )
    ]
    final_slots = [
        load_json(path)
        for path in sorted(
            (
                ROOT
                / "experiments/swan_final_confirmation1/baseline_slots"
            ).glob("*.json")
        )
    ]
    final_normal = sum(
        row["mechanical"]["checks"]["word_range"]
        and row["mechanical"]["checks"]["minimum_detector_units"]
        for row in final_baselines
    )
    final_injection = sum(
        row["mechanical"]["accepted_slots"] >= 5
        for row in final_baselines
    )

    inventory = {
        "official_public_release": {
            "candidate_units": 0,
            "independent_documents": 0,
            "normal_form_documents": 0,
            "blind_quality_pass_documents": 0,
            "joint_qualified_documents": 0,
        },
        "confirmation_three_arm30": {
            "candidate_units": short["baseline_words"]["n"],
            "independent_documents": short["baseline_words"]["n"],
            "normal_form_documents": sum(
                (
                    100 <= row["text"].count(" ") + 1 <= 220
                    and len(row["sentences"]) >= 8
                )
                for row in (
                    load_json(path)
                    for path in sorted(
                        (
                            ROOT
                            / "experiments/confirmation_three_arm30/baselines"
                        ).glob("*.json")
                    )
                )
            ),
            "blind_quality_pass_documents": None,
            "joint_qualified_documents": 0,
            "all_five_slots_accepted": short[
                "all_five_sentences_accepted"
            ],
            "original_detected": (
                short["baseline_words"]["n"]
                - len(short["initially_undetected_ids"])
            ),
            "word_min": short["baseline_words"]["min"],
            "word_max": short["baseline_words"]["max"],
            "word_mean": short["baseline_words"]["mean"],
        },
        "swan_sentence_growth1": {
            "candidate_units": len(growth["prefixes"]),
            "independent_documents": growth["independent_documents"],
            "normal_form_documents": growth_normal,
            "saved_automated_quality_pass_documents": growth_quality,
            "blind_quality_pass_documents": None,
            "joint_qualified_documents": 0,
            "nested_prefixes": True,
        },
        "swan_generator_comparison1": {
            "candidate_units": len(comparison["arms"]),
            "independent_documents": comparison["independent_prompts"],
            "normal_form_watermarked_documents": comparison_normal,
            "saved_automated_quality_pass_documents": comparison_quality,
            "blind_quality_pass_documents": None,
            "joint_qualified_documents": 0,
        },
        "grounded_injection5": {
            "candidate_units": len(grounded_rows),
            "independent_documents": grounded["allocated_documents"],
            "length_only_documents": grounded_length,
            "normal_form_documents": grounded_normal,
            "saved_automated_quality_pass_documents": sum(
                bool(row["source_pass"]) for row in grounded_rows
            ),
            "blind_quality_pass_documents": None,
            "fully_injected_documents": sum(
                bool(row["full_injection"]) for row in grounded_rows
            ),
            "original_detected": sum(
                bool(row["detected"]) for row in grounded_rows
            ),
            "joint_qualified_documents": grounded[
                "joint_usable_watermarked"
            ],
        },
        "swan_quality_first_windows4": {
            "candidate_units": len(quality["documents"]),
            "independent_documents": quality["selected_documents"],
            "normal_form_documents": quality_normal,
            "saved_automated_quality_pass_documents": quality[
                "quality_pass"
            ],
            "blind_quality_pass_documents": None,
            "traceable_injected_documents": 0,
            "joint_qualified_documents": 0,
        },
        "swan_oracle_composition1": {
            "candidate_units": len(oracle["groups"]),
            "independent_documents": 0,
            "normal_form_documents": oracle_normal,
            "saved_quality_pass_documents": 0,
            "blind_quality_pass_documents": None,
            "accepted_slots": sum(
                row["accepted_slots"] for row in oracle["groups"]
            ),
            "total_slots": sum(
                row["original_units"] for row in oracle["groups"]
            ),
            "original_detected": sum(
                bool(row["conditions"]["original"]["detected"])
                for row in oracle["groups"]
            ),
            "joint_qualified_documents": 0,
        },
        "swan_final_confirmation1": {
            "allocated_documents": len(final_sources["sources"]),
            "candidate_units": len(final_baselines),
            "independent_documents": len(final_baselines),
            "normal_form_documents": final_normal,
            "blind_quality_pass_documents": None,
            "minimum_injection_evidence_documents": final_injection,
            "accepted_slots": sum(
                bool(row["accepted"]) for row in final_slots
            ),
            "slot_checkpoints": len(final_slots),
            "joint_qualified_documents": 0,
        },
        "swan_known_positive1": {
            "candidate_units": known["parser"]["examples"],
            "independent_documents": 0,
            "normal_form_documents": 0,
            "blind_quality_pass_documents": None,
            "joint_qualified_documents": 0,
        },
    }
    if inventory != manifest["expected_inventory"]:
        raise ValueError("Reconstructed local inventory differs")
    return inventory


def validate_collection_contract(manifest):
    collection = manifest["new_collection"]
    allocation = collection["allocation"]
    generation = collection["generation"]
    qualification = collection["qualification"]
    checks = {
        "execution_locked": collection["execution_authorized"] is False,
        "not_old_run_continuation": (
            collection["not_continuation_of"]
            == "swan_final_confirmation1"
        ),
        "new_disjoint_shard": (
            collection["dataset"]["start_train_shard"] == 8
        ),
        "pilot_main_separate": (
            allocation["pilot"] == 5
            and allocation["main"] == 30
            and allocation["freeze_all_before_generation"]
            and not allocation["replacement_allowed"]
            and not allocation["pilot_in_main"]
        ),
        "upstream_commit_fixed": (
            generation["swan_commit"]
            == manifest["official_snapshot"]["main_commit"]
        ),
        "non_paper_model_declared": (
            generation["paper_equivalent_model"] is False
        ),
        "fixed_injection_threshold": (
            generation["acceptance_s2match"] == 0.75
        ),
        "normal_form_fixed": (
            qualification["word_min"] == 100
            and qualification["word_max"] == 220
            and qualification["segmenter_units"] == 16
        ),
        "all_slots_must_be_accepted": (
            qualification["complete_single_sentence_slots"]
            == qualification["accepted_slots"]
            == generation["sentence_slots"]
        ),
        "detector_not_selection_gate": (
            qualification["original_detector_is_inclusion_gate"] is False
        ),
        "pilot_cannot_tune": (
            collection["pilot_parameter_tuning_allowed"] is False
        ),
        "attack_not_authorized": collection["attack_authorized"] is False,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError(f"Collection contract failed: {failed}")
    return checks


def build_summary(manifest, output=OUTPUT):
    official = verify_official_snapshot(manifest)
    inventory = build_inventory(manifest)
    contract = validate_collection_contract(manifest)
    joint_total = sum(
        row["joint_qualified_documents"] for row in inventory.values()
    )
    if joint_total != 0:
        raise ValueError("Existing qualifying asset count is no longer zero")
    return {
        "status": "complete_offline_asset_audit",
        "public_asset_result": official,
        "inventory": inventory,
        "joint_qualifying_existing_documents": joint_total,
        "decision": "no_existing_asset_meets_joint_gate",
        "next_collection": {
            "status": manifest["new_collection"]["status"],
            "execution_authorized": False,
            "pilot_allocations": manifest["new_collection"]["allocation"][
                "pilot"
            ],
            "main_allocations": manifest["new_collection"]["allocation"][
                "main"
            ],
            "source_start_shard": manifest["new_collection"]["dataset"][
                "start_train_shard"
            ],
            "model": manifest["new_collection"]["generation"][
                "requested_model"
            ],
            "paper_equivalent_model": False,
            "contract_checks": contract,
        },
        "resource_counts": {
            "network_calls": 0,
            "api_generation_calls": 0,
            "model_loads": 0,
            "parser_calls": 0,
            "s2match_calls": 0,
            "detector_calls": 0,
            "attack_calls": 0,
        },
        "manifest_sha256": sha256(output / "manifest.json"),
        "protocol_sha256": sha256(output / "protocol.md"),
    }


def render_report(summary):
    return f"""# SWAN positive-asset availability audit

Date: 2026-09-16. Status: complete offline audit.

## Decision

No checked public or local asset meets the joint gate for a traceable SWAN
injection, an independent document, normal form, and detector-blind human
quality. The number of qualifying existing documents is
**{summary["joint_qualifying_existing_documents"]}**.

This is a bounded absence statement. It applies to the public locations and
local candidate families frozen in `manifest.json`; it does not assert that no
public or private copy exists elsewhere.

## Public release audit

The fixed official snapshot is commit
`{summary["public_asset_result"]["main_commit"]}`. Its recursive tree contains
{summary["public_asset_result"]["tracked_files"]} tracked files and
{summary["public_asset_result"]["data_bearing_files"]} JSON data files, all of
which are AMR banks or AMR/template artifacts. It contains
{summary["public_asset_result"]["generated_watermarked_text_files"]}
pre-generated watermarked-text files. The checked repository has
{summary["public_asset_result"]["releases"]} releases and
{summary["public_asset_result"]["tags"]} tags.

The README treats C4 REALNEWS as prompts and human references, then directs
users to run the injector and create `output/watermarked` locally. The paper
links the code repository but no separate pre-generated watermarked-text
asset was found in the checked paper, GitHub, HuggingFace, or Zenodo locations.

## Local four-gate inventory

| Candidate family | Independent | Normal form | Injection evidence | Quality evidence | Joint |
|---|---:|---:|---|---|---:|
| Official public release | 0 | 0 | no released text | not applicable | 0 |
| Short three-arm baselines | 30 | 0/30 | 21/30 all five slots accepted; 29/30 detected | baseline blind-human quality not assessed | 0 |
| Sentence-growth prefixes | 1 document | 1/3 prefixes | 19/20 accepted in the only normal-form prefix | 0/3 automated pass; no blind-human labels | 0 |
| Generator comparison | 1 prompt | 0 watermarked arms | upstream 0/5; local 3/5 accepted | 0/3 automated pass; no blind-human labels | 0 |
| Grounded injection | 5 | 1/5 | 0/5 fully injected; 0/5 detected | 0/5 automated pass; no blind-human labels | 0 |
| Quality-first windows | 30 | 30/30 | injection not run | 14/30 automated pass; no blind-human labels | 0 |
| Oracle compositions | 0 | 6/6 containers | 137/150 component slots; 6/6 original detected | known repetitive cross-topic containers | 0 |
| Terminated final run | 66 complete | 56/66 | 0/66 reached five accepted slots | not assessed | 0 |
| Bank known-positive examples | 0 documents | 0/3 | parser/matcher examples, not generated documents | not assessed | 0 |

The closest length-and-quality asset is the 14 quality-first C4 windows, but
none was watermarked. The closest traceable watermarked asset is the short
30-document cohort, but every document is only 17--46 words. The six
normal-length oracle batches concatenate unrelated short documents and remain
mechanics diagnostics rather than independent natural documents. The
terminated final run is not reused: 56/66 complete baselines had normal form,
but 0/66 met its minimum injection prerequisite and no quality review ran.

## Frozen next collection

The new protocol allocates 5 pilot and 30 main sources from C4 train shard 8
or later, after a fresh global identity exclusion. All 35 identities are
frozen before generation, pilot and main are disjoint, and no replacement is
allowed.

It uses the exact public SWAN injection path with the repository's current
Claude Sonnet 4.5 default. This is explicitly not the retired Claude 3.5
Sonnet v2 used in the paper. A qualifying baseline must have 16/16 accepted
sentence slots, 100--220 words, exactly 16 detector-independent sentence
units, and two blind human quality passes. Original detector response is an
outcome, never a selection criterion.

Execution is **not authorized** because this fixed backend requires a paid
remote API. No generation, model, parser, S2Match, detector, or attack call was
made by this audit. Any future execution requires explicit budget
authorization plus a pre-execution identity/runtime hash manifest.

## Reproducibility

- Protocol: `protocol.md`
- Frozen evidence and exact URLs: `manifest.json`
- Machine-readable result: `summary.json`
- Independent offline check: `audit.json`
- Audit implementation: `../../scripts/audit_swan_positive_assets.py`
"""


def audit(output=OUTPUT):
    manifest_path = output / "manifest.json"
    protocol_path = output / "protocol.md"
    if not manifest_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("Protocol or manifest is missing")
    manifest = load_json(manifest_path)
    if sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Protocol hash changed")

    inputs = verify_inputs(manifest)
    summary = build_summary(manifest, output)
    report = render_report(summary)
    write_frozen_json(output / "summary.json", summary)
    write_frozen_text(output / "report.md", report)

    checks = {
        "input_hashes_match": True,
        "official_snapshot_consistent": True,
        "official_tree_has_no_generated_text": (
            summary["public_asset_result"][
                "generated_watermarked_text_files"
            ]
            == 0
        ),
        "local_inventory_reconstructed": (
            summary["inventory"] == manifest["expected_inventory"]
        ),
        "joint_qualifying_documents_zero": (
            summary["joint_qualifying_existing_documents"] == 0
        ),
        "new_protocol_distinct_and_locked": all(
            summary["next_collection"]["contract_checks"].values()
        ),
        "no_new_compute": all(
            value == 0 for value in summary["resource_counts"].values()
        ),
    }
    if not all(checks.values()):
        raise ValueError("Audit checks failed")
    audit_payload = {
        "status": "passed",
        "checks": checks,
        "inputs_checked": inputs,
        "artifacts": {
            "protocol.md": sha256(protocol_path),
            "manifest.json": sha256(manifest_path),
            "summary.json": sha256(output / "summary.json"),
            "report.md": sha256(output / "report.md"),
        },
        "audit_script_sha256": sha256(Path(__file__)),
        "new_network_calls": 0,
        "new_api_generation_calls": 0,
        "new_model_loads": 0,
        "new_parser_calls": 0,
        "new_s2match_calls": 0,
        "new_detector_calls": 0,
        "new_attack_calls": 0,
    }
    write_frozen_json(output / "audit.json", audit_payload)
    return audit_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
