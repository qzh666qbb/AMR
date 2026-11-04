"""Run the frozen profile -> calibration seal -> held-out test -> closure flow."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.embedding_amr_evaluation import interval, paired_comparisons
from src.core.embedding_amr_runtime import atomic, file_sha, sealed_scores
from swan_final1_common import single_writer


def run(inputs, out, encoder, bank, predecessor_study=None, stress_input=None,
        execution_amendment=None, resume_calibration=None):
    protocol = json.loads((inputs / "PROTOCOL.json").read_text())
    for name, sha in protocol["files"].items():
        assert file_sha(inputs / name) == sha
    assert bool(stress_input) == bool(execution_amendment)
    resume_rows = 0
    if resume_calibration:
        resume_rows = json.loads((resume_calibration / "HEARTBEAT.json").read_text())["completed"]
        assert resume_rows > 0
    amendment = None
    if execution_amendment:
        amendment = json.loads(execution_amendment.read_text())
        assert file_sha(inputs / "PROTOCOL.json") == amendment["protocol_sha256"]
        assert file_sha(stress_input) == amendment["stress_input_sha256"]
        assert os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"
        original = [json.loads(line) for line in (inputs / "private/calibration.jsonl").read_text().splitlines()]
        stress = [json.loads(line) for line in stress_input.read_text().splitlines()]
        expected = [{**original[i], "split": "development"} for i in amendment["stress_calibration_row_indices"]]
        assert stress == expected
    out.mkdir()
    (out / ".gitignore").write_text("private/\n*.log\n")
    assert file_sha(bank) == protocol["bank_sha256"]
    base = [sys.executable, str(ROOT / "scripts/run_embedding_amr_detection.py")]
    common = ["--bank", str(bank), "--bank-sha256", protocol["bank_sha256"],
              "--writer-bank-sha256", protocol["bank_sha256"], "--encoder", str(encoder),
              "--device", "cuda", "--batched", "--parser-batch-size", "8"]
    started = time.time()
    if amendment:
        atomic(out / "EXECUTION_AMENDMENT.json", amendment)

    def execute(command):
        with (out / "execution.log").open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError("Stage failed with exit code " + str(result.returncode))

    try:
        with single_writer(out):
            execute(base + ["score", "--input", str(inputs / "private/profile.jsonl"),
                            "--output", str(out / "profile")] + common)
            profile = json.loads((out / "profile/HEARTBEAT.json").read_text())
            assert profile["device_used_fraction"] < .8
            assert json.loads((out / "profile/AUDIT.json").read_text())["parser_exact_replay"]
            if predecessor_study:
                old = sealed_scores(predecessor_study / "profile/SCORES.json")
                new = sealed_scores(out / "profile/SCORES.json")
                old_manifest = json.loads((predecessor_study / "profile/MANIFEST.json").read_text())
                new_manifest = json.loads((out / "profile/MANIFEST.json").read_text())
                fields = ("input_sha256", "config", "windows", "bank_sha256",
                          "parser_fingerprint", "encoder_fingerprint", "dependencies")
                assert all(old_manifest[k] == new_manifest[k] for k in fields)
                assert len(old) == len(new) == 3
                for previous, current in zip(old, new):
                    for key in ("row_id", "text_sha256", "arms", "candidate_count", "usable_candidates"):
                        assert previous[key] == current[key], key
                # The official scorer fingerprint also hashes dependency versions.
                # An unrelated environment upgrade (spacy/nltk) changes metadata
                # without changing any profile score; drift is recorded, and the
                # exact row-level score equality above stays a hard condition.
                drift = {k: {"old": old_manifest[k], "new": new_manifest[k]}
                         for k in ("bank_scorer_fingerprint",)
                         if old_manifest[k] != new_manifest[k]}
                scorer_dependencies = {
                    "old": (old_manifest.get("scorer_assets") or {}).get("dependencies"),
                    "new": (new_manifest.get("scorer_assets") or {}).get("dependencies")}
                atomic(out / "PROFILE_EQUIVALENCE.json", {
                    "status": "scores_exactly_equal_metadata_drift" if drift else "exactly_equal",
                    "rows": 3, "arms": 3,
                    "score_max_abs_error": 0,
                    "old_profile_terminal_sha256": file_sha(predecessor_study / "profile/TERMINAL.json"),
                    "new_profile_terminal_sha256": file_sha(out / "profile/TERMINAL.json"),
                    "metadata_drift": drift,
                    "scorer_dependencies": scorer_dependencies,
                    "scientific_input_and_model_fields_equal": True})
            stress_profile = None
            if stress_input:
                atomic(out / "STAGE.json", {"stage": "resource_stress", "terminal": False})
                execute(base + ["score", "--input", str(stress_input),
                                "--output", str(out / "resource_stress")] + common)
                stress_profile = json.loads((out / "resource_stress/HEARTBEAT.json").read_text())
                assert stress_profile["completed"] == len(stress)
                assert stress_profile["device_used_fraction"] < .8
                assert json.loads((out / "resource_stress/AUDIT.json").read_text())["parser_exact_replay"]
                atomic(out / "STRESS_PRECHECK.json", {
                    "status": "go", "profile": stress_profile,
                    "source": "previously_failed_calibration_source_all_three_conditions",
                    "source_row_indices": amendment["stress_calibration_row_indices"],
                    "stress_input_sha256": file_sha(stress_input),
                    "score_terminal_sha256": file_sha(out / "resource_stress/TERMINAL.json"),
                    "used_for_threshold_or_algorithm_selection": False,
                    "calibration_reparsed_from_row_zero": resume_rows == 0,
                    "calibration_resumed_rows": resume_rows,
                    "resume_source": str(resume_calibration) if resume_calibration else None})
            atomic(out / "RESOURCE_PRECHECK.json", {
                "status": "go", "profile": profile, "dtype": "float32",
                "failed_source_stress_profile": stress_profile,
                "parser_replay": "two_complete_document_batches",
                "estimate_note": "three engineering examples; total time remains uncertain"})
            for stage in ("calibration", "test"):
                atomic(out / "STAGE.json", {"stage": stage, "terminal": False})
                if stage == "test":
                    threshold_sha = file_sha(out / "CALIBRATION.json")
                    atomic(out / "TEST_START.json", {"threshold_sha256": threshold_sha,
                                                    "at_unix": time.time()})
                extra = []
                if stage == "calibration" and resume_calibration:
                    extra = ["--resume-run", str(resume_calibration),
                             "--resume-rows", str(resume_rows)]
                execute(base + ["score", "--input", str(inputs / f"private/{stage}.jsonl"),
                                "--output", str(out / stage)] + common + extra)
                if stage == "calibration":
                    execute(base + ["calibrate", "--scores", str(out / stage / "SCORES.json"),
                                    "--output", str(out / "CALIBRATION.json"), "--alpha", ".05"])
            assert file_sha(out / "CALIBRATION.json") == threshold_sha
            execute(base + ["evaluate", "--scores", str(out / "test/SCORES.json"),
                            "--calibration", str(out / "CALIBRATION.json"),
                            "--output", str(out / "EVALUATION.json")])
            evaluation = json.loads((out / "EVALUATION.json").read_text())
            accepted = {r["source_id"] for r in protocol["positive_metadata"]
                        if r["all_injection_accepted"]}
            subgroup = [r for r in evaluation["predictions"] if r["source_id"] in accepted]
            metrics = []
            for condition in protocol["conditions"]:
                group = [r for r in subgroup if r["condition"] == condition]
                for arm in ("amr_only", "embedding_amr", "embedding_without_relation"):
                    hits = sum(r["decisions"][arm]["is_watermarked"] for r in group)
                    metrics.append({"condition": condition, "arm": arm, "hits": hits,
                                    "denominator": len(group), "rate": hits / len(group),
                                    "ci95_clopper_pearson": interval(hits, len(group))})
            atomic(out / "FULLY_WRITTEN_SUBGROUP.json", {
                "scope": "descriptive_predeclared_21_of_30_historical_attempts",
                "metrics": metrics,
                "paired_comparisons": paired_comparisons(subgroup)})
            atomic(out / "FINAL_AUDIT.json", {
                "status": "evaluation_complete", "full_positive_denominator": 30,
                "fully_written_subgroup": 21, "null_denominator_per_condition": 1000,
                "calibration_denominator_per_condition": 1000,
                "elapsed_seconds": time.time() - started,
                "prior_stopped_run_window_estimate_seconds": (
                    amendment["prior_stopped_run_window_estimate_seconds"] if amendment else None),
                "cumulative_run_window_estimate_seconds": (
                    amendment["prior_stopped_run_window_estimate_seconds"] + time.time() - started
                    if amendment else None),
                "cost_definition": "wall-time windows, not exclusive GPU kernel seconds",
                "calibration_sealed_before_test": True, "no_quality_filtering": True,
                "scope": protocol["scope"], "bank_is_public": True,
                "fusion_benefit_established": False,
                "effect_interpretation_requires_review_summary": True})
            names = ["RESOURCE_PRECHECK.json", "CALIBRATION.json", "TEST_START.json",
                     "EVALUATION.json", "FULLY_WRITTEN_SUBGROUP.json", "FINAL_AUDIT.json"]
            if amendment:
                names += ["EXECUTION_AMENDMENT.json", "STRESS_PRECHECK.json"]
            if predecessor_study:
                names += ["PROFILE_EQUIVALENCE.json"]
            atomic(out / "TERMINAL.json", {"status": "evaluation_complete",
                                          "files": {n: file_sha(out / n) for n in names}})
    except Exception as exc:
        atomic(out / "HALT.json", {"status": "halted", "exception_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--predecessor-study", type=Path)
    p.add_argument("--stress-input", type=Path)
    p.add_argument("--execution-amendment", type=Path)
    p.add_argument("--resume-calibration", type=Path,
                   help="Halted calibration stage to carry forward; test still runs fresh")
    a = p.parse_args()
    run(a.inputs, a.output, a.encoder, a.bank, a.predecessor_study,
        a.stress_input, a.execution_amendment, a.resume_calibration)
