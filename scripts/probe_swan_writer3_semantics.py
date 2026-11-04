"""Bounded synthetic semantic guard validation; labels are hidden from the judge."""
import json
import os
from pathlib import Path
import sys

ROOT = Path("/workspace/swan-elastic-reconstruct")
PARENT = ROOT / "normal_writer_repair3_full2"
OUT = ROOT / "normal_writer_repair3_guard1"
code = PARENT / "private/code"
sys.path[:0] = [str(code), str(code / "scripts"),
               str(ROOT / "atomic_scoring_dependencies1/overlay"),
               str(ROOT / "independent_banks2_runtime/private/python_env")]
os.environ["NLTK_DATA"] = str(code / ".nltk_data")
import swan_writer_repair3 as w
import deploy_swan_writer_repair2 as deployment
from swan_final1_common import freeze, load, sha256, single_writer

CASES = [
    {"id": "music_synonyms", "expected": False, "focus": "novel",
     "source": "A family is remembering a singer.", "premise": "The family listens to the singer's music.",
     "fact": "The family plays an album.", "previous_sentences": ["The family has been playing the record."],
     "candidate": "The family has been playing the album."},
    {"id": "workspace_synonyms", "expected": False, "focus": "novel",
     "source": "A company renovates its office.", "premise": "An office renovation.",
     "fact": "The firm renovates its workspace.", "previous_sentences": ["The company is renovating its office."],
     "candidate": "The firm has been renovating its workspace."},
    {"id": "repair_synonyms", "expected": False, "focus": "novel",
     "source": "A team prepares a court.", "premise": "The team fixes a damaged net.",
     "fact": "The team fixes the net.", "previous_sentences": ["The team has been repairing the net."],
     "candidate": "The team has been fixing the net."},
    {"id": "correction_restatement", "expected": False, "focus": "novel",
     "source": "A broadcaster corrects a video.", "premise": "An error in a clip is corrected.",
     "fact": "The clip contains a correction note.",
     "previous_sentences": ["The broadcaster attached a correction note to the clip."],
     "candidate": "The clip now includes the broadcaster's correction note."},
    {"id": "same_object_new_action", "expected": True, "focus": "novel",
     "source": "A company reviews and then signs a contract.", "premise": "A contract reaches agreement.",
     "fact": "The company signs the contract after reviewing it.",
     "previous_sentences": ["The company finished reviewing the contract yesterday."],
     "candidate": "The company has now signed the contract."},
    {"id": "same_topic_new_fact", "expected": True, "focus": "novel",
     "source": "A club prepares a concert.", "premise": "The club arranges a concert.",
     "fact": "The club sells tickets.", "previous_sentences": ["The club has booked a hall for the concert."],
     "candidate": "The club has now started selling tickets."},
    {"id": "fact_substitution", "expected": False, "focus": "meaning",
     "source": "A worker requests a promotion.", "premise": "A worker builds a promotion case.",
     "fact": "She submits a promotion request.", "previous_sentences": ["Maya prepared her promotion request."],
     "candidate": "She has been organizing the office files."},
    {"id": "physical_impossibility", "expected": False, "focus": "plausible",
     "source": "An ice hockey team checks a solid rubber puck.", "premise": "The team handles a solid rubber puck.",
     "fact": "The team stitches the solid rubber puck.",
     "previous_sentences": ["The team found damage on its solid rubber puck."],
     "candidate": "The team has been stitching the puck."},
]


def main():
    deployment.verify(PARENT)
    terminal = load(PARENT / "TERMINAL.json")
    for field, name in (("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json")):
        assert terminal[field] == sha256(PARENT / name)
    for name, expected in load(PARENT / "AUDIT.json")["files"].items():
        assert sha256(PARENT / name) == expected
    parent_result = load(PARENT / "RESULT.json")
    OUT.mkdir(exist_ok=True)
    freeze(OUT / "CASES.json", CASES)
    freeze(OUT / "FREEZE.json", {
        "stage": "synthetic_semantic_guard_probe", "cases": 8, "human": False,
        "parent_budget": {"calls": parent_result["budget"]["cumulative_calls"],
                          "exposure_nanodollars": parent_result["budget"]["cumulative_exposure_nanodollars"]},
        "caps": {"new_calls": 8, "new_nanodollars": 100_000_000},
        "parent_freeze_sha256": sha256(PARENT / "FREEZE.json"),
        "parent_result_sha256": sha256(PARENT / "RESULT.json"),
        "case_sha256": sha256(OUT / "CASES.json"),
        "source_sha256": sha256(Path(__file__)),
        "labels_sent_to_judge": False, "provider_scores_visible": False,
        "distributional_quality_or_robustness_claim": False,
    })
    with single_writer(OUT):
        if (OUT / "TERMINAL.json").exists():
            raise RuntimeError("Probe already completed")
        if (OUT / "HALT.json").exists():
            raise RuntimeError("Permanent probe halt")
        manifest = load(ROOT / "v4_confirmation1_runtime/experiments/swan_v4_independent_confirmation1/private/queue/MANIFEST.json")
        keys = [j["argv"][j["argv"].index("--key-file") + 1] for j in manifest["jobs"] if "--key-file" in j["argv"]]
        assert len(keys) == 1
        key = w.normal.pilot.credential.read_credential(Path(keys[0]))
        journal = w.Journal(OUT, lambda body: w.budget.transport(body, key))
        judgments = []
        for case in CASES:
            packet = {k: v for k, v in case.items() if k not in ("id", "expected", "focus")}
            review = w.model_review(journal, case["id"], w.SENTENCE_REVIEW, packet)
            freeze(OUT / "private/reviews" / (case["id"] + ".json"), review)
            decision = w.review_pass(review["value"],
                ("meaning", "novel", "topic", "plausible", "coherent"), "duplicate_of")
            judgments.append({"id": case["id"], "expected": case["expected"], "accepted": decision,
                              "decision_correct": decision == case["expected"],
                              "focus": case["focus"], "focus_value": review["value"].get(case["focus"]),
                              "review": review["value"]})
        result = {"status": "GUARD_PROBE_COMPLETE", "cases": 8,
                  "correct": sum(j["decision_correct"] for j in judgments),
                  "judgments": judgments, "budget": journal.accounting(),
                  "independent_human": False, "distributional_claim": False}
        freeze(OUT / "RESULT.json", result)
        files = [p for p in (OUT / "private").rglob("*") if p.is_file()]
        files += [OUT / "RESULT.json", OUT / "CASES.json", OUT / "FREEZE.json"]
        freeze(OUT / "AUDIT.json", {"files": {str(p.relative_to(OUT)): sha256(p) for p in sorted(files)}})
        freeze(OUT / "TERMINAL.json", {"result_sha256": sha256(OUT / "RESULT.json"),
                                       "audit_sha256": sha256(OUT / "AUDIT.json")})
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if OUT.exists() and not (OUT / "TERMINAL.json").exists():
            freeze(OUT / "HALT.json", {"reason_type": type(exc).__name__, "automatic_restart": False})
        raise
