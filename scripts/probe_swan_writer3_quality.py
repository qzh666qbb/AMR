"""Four unwatermarked editing baselines: preserve facts, merge overlap, vary prose."""
import json
import os
from pathlib import Path
import sys

ROOT = Path("/workspace/swan-elastic-reconstruct")
WRITER = ROOT / "normal_writer_repair3_full2"
PARENT = ROOT / "normal_writer_repair3_guard1"
OUT = ROOT / "normal_writer_repair3_quality1"
code = WRITER / "private/code"
sys.path[:0] = [str(code), str(code / "scripts"),
               str(ROOT / "atomic_scoring_dependencies1/overlay"),
               str(ROOT / "independent_banks2_runtime/private/python_env")]
os.environ["NLTK_DATA"] = str(code / ".nltk_data")
import swan_writer_repair3 as w
import deploy_swan_writer_repair2 as deployment
from swan_final1_common import freeze, load, sha256, single_writer

INSTRUCTION = """Edit these planned facts into a natural fictional continuation.
Keep the central topic and the planned substantive events; do not introduce new facts.
Use 100-220 English words in 8-14 sentences with varied rhythm and openings.
Combine related facts into sentences, use appropriate transitions and clear referents.
Avoid repetitive inventories or sixteen identical subject-verb-object frames.
Semantic deduplication: if two facts repeat or entail the same substantive information,
express that information ONCE and assign both fact IDs to that sentence.
Every original fact ID 1-16 must appear exactly once across the fact_ids lists.
Combining distinct related facts in a sentence is also allowed, but do not incorrectly
label them duplicates. Use merged_duplicates only for actual redundant information.
Return JSON only: {"sentences":[{"text":"A complete sentence.","fact_ids":[1,2]},...],
"merged_duplicates":[[11,13]],"notes":[]}. merged_duplicates may be [].
Do not optimize for any watermark. This is an unwatermarked quality baseline.
"""
REVIEW = w.FINAL_REVIEW + """
Additionally compare the passage against the original planned facts. Add boolean
meaning: true only if every substantive event is preserved, without contradictory
new details. Redundant facts may be expressed just once. You may judge the source
as fictional continuation, not verified news. Include any failures in issues.
"""


def verify_terminal(folder):
    terminal = load(folder / "TERMINAL.json")
    for field, name in (("result_sha256", "RESULT.json"), ("audit_sha256", "AUDIT.json")):
        assert terminal[field] == sha256(folder / name)
    for name, expected in load(folder / "AUDIT.json")["files"].items():
        assert sha256(folder / name) == expected


def validate(value, tokenize):
    sentences = value.get("sentences")
    if not isinstance(sentences, list) or not 8 <= len(sentences) <= 14:
        raise ValueError("sentence_count")
    ids, texts = [], []
    for row in sentences:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("sentence_schema")
        text = row["text"]
        if tokenize(text) != [text] or not text.endswith("."):
            raise ValueError("sentence_boundary")
        if not isinstance(row.get("fact_ids"), list) or not row["fact_ids"]:
            raise ValueError("fact_mapping")
        if any(type(i) is not int for i in row["fact_ids"]):
            raise ValueError("noninteger_fact_id")
        ids.extend(row["fact_ids"])
        texts.append(text)
    if sorted(ids) != list(range(1, 17)):
        raise ValueError("fact_coverage")
    if tokenize(" ".join(texts)) != texts:
        raise ValueError("document_boundaries")
    if not 100 <= len(w.normal.WORD.findall(" ".join(texts))) <= 220:
        raise ValueError("word_count")
    groups = value.get("merged_duplicates")
    if not isinstance(groups, list):
        raise ValueError("duplicate_group_schema")
    for group in groups:
        if (not isinstance(group, list) or len(group) < 2
                or any(type(i) is not int or not 1 <= i <= 16 for i in group)
                or len(set(group)) != len(group)
                or not any(set(group).issubset(row["fact_ids"]) for row in sentences)):
            raise ValueError("duplicate_group_mapping")
    return texts


def main():
    deployment.verify(WRITER)
    verify_terminal(WRITER)
    verify_terminal(PARENT)
    parent_result = load(PARENT / "RESULT.json")
    sources = load(WRITER / "private/SOURCES.json")
    OUT.mkdir(exist_ok=True)
    freeze(OUT / "FREEZE.json", {
        "stage": "unwatermarked_document_editing_baseline", "sources": [s["allocation_id"] for s in sources],
        "parent_budget": {"calls": parent_result["budget"]["cumulative_calls"],
                          "exposure_nanodollars": parent_result["budget"]["cumulative_exposure_nanodollars"]},
        "caps": {"new_calls": 8, "new_nanodollars": 100_000_000},
        "parent_result_sha256": sha256(PARENT / "RESULT.json"),
        "writer_freeze_sha256": sha256(WRITER / "FREEZE.json"),
        "plans": {s["allocation_id"]: sha256(WRITER / "private/plans" / (s["allocation_id"] + ".json")) for s in sources},
        "source_sha256": sha256(Path(__file__)),
        "prose_contract": "100-220 words, 8-14 sentences, all16 fact IDs, duplicate claims expressed once",
        "watermark_written": False, "watermark_scored": False, "independent_human": False,
        "old_sentence_gate_relaxed_for_quality_baseline_only": True,
        "automatic_scale_up": False,
    })
    with single_writer(OUT):
        if (OUT / "TERMINAL.json").exists() or (OUT / "HALT.json").exists():
            raise RuntimeError("Terminal quality probe; do not rerun")
        manifest = load(ROOT / "v4_confirmation1_runtime/experiments/swan_v4_independent_confirmation1/private/queue/MANIFEST.json")
        keys = [j["argv"][j["argv"].index("--key-file") + 1] for j in manifest["jobs"] if "--key-file" in j["argv"]]
        assert len(keys) == 1
        key = w.normal.pilot.credential.read_credential(Path(keys[0]))
        journal = w.Journal(OUT, lambda body: w.budget.transport(body, key))
        tokenizer = w.normal.tokenizer()
        records = []
        for source in sources:
            sid = source["allocation_id"]
            plan = load(WRITER / "private/plans" / (sid + ".json"))["plan"]
            facts = [{"id": i + 1, "fact": s["fact"], "draft": s["sentence"]}
                     for i, s in enumerate(plan["slots"])]
            packet = {"source": source["generation_prompt"], "premise": plan["premise"], "facts": facts}
            response = journal.request(sid + "-quality", INSTRUCTION + json.dumps(packet), w.SYSTEM, 2400)
            value, texts, failure, review = None, [], None, None
            try:
                value = w.decode(response)
                texts = validate(value, tokenizer)
            except (TypeError, ValueError) as exc:
                failure = str(exc)
            if not failure:
                review = w.model_review(journal, sid + "-review", REVIEW,
                                       {**packet, "passage": texts, "mapping": value})
            passed = bool(review and w.review_pass(review["value"],
                ("meaning", "topic", "plausible", "novel", "coherent", "natural"), "duplicate_pairs"))
            record = {"source_id": sid, "edited": value, "failure": failure, "review": review,
                      "model_quality_pass": passed, "response_sha256": response["call_sha256"],
                      "words": len(w.normal.WORD.findall(" ".join(texts))),
                      "sentence_count": len(texts), "watermarked": False}
            freeze(OUT / "private/documents" / (sid + ".json"), record)
            records.append(record)
        result = {"status": "QUALITY_BASELINE_COMPLETE", "documents": 4,
                  "valid_drafts": sum(r["failure"] is None for r in records),
                  "model_quality_pass": sum(r["model_quality_pass"] for r in records),
                  "budget": journal.accounting(), "watermarked_documents": 0,
                  "formal_TPR": None, "formal_FPR": None, "independent_human": False}
        freeze(OUT / "RESULT.json", result)
        files = [p for p in (OUT / "private").rglob("*") if p.is_file()]
        files += [OUT / "RESULT.json", OUT / "FREEZE.json"]
        freeze(OUT / "AUDIT.json", {"files": {str(p.relative_to(OUT)): sha256(p) for p in sorted(files)}})
        freeze(OUT / "TERMINAL.json", {"result_sha256": sha256(OUT / "RESULT.json"),
                                       "audit_sha256": sha256(OUT / "AUDIT.json")})
        print(json.dumps(result, ensure_ascii=False, indent=2))


def self_test():
    tokenize = lambda text: [s.strip() + "." for s in text.split(".") if s.strip()]
    text = "The committee reviewed the report and recorded several useful changes for the next meeting."
    valid = {"sentences": [{"text": text, "fact_ids": [i * 2 + 1, i * 2 + 2]}
                          for i in range(8)], "merged_duplicates": []}
    assert len(validate(valid, tokenize)) == 8
    for mode in ("missing_fact", "duplicate_mapping", "too_few_sentences"):
        altered = json.loads(json.dumps(valid))
        if mode == "missing_fact":
            altered["sentences"][-1]["fact_ids"] = [15, 15]
        elif mode == "duplicate_mapping":
            altered["merged_duplicates"] = [[1, 16]]
        else:
            altered["sentences"] = altered["sentences"][:7]
        try:
            validate(altered, tokenize)
        except ValueError:
            continue
        raise AssertionError(mode)
    print(json.dumps({"quality_contract_checks": 4, "passed": True, "provider_calls": 0}))


if __name__ == "__main__":
    try:
        self_test() if "--self-test" in sys.argv else main()
    except Exception as exc:
        if OUT.exists() and not (OUT / "TERMINAL.json").exists():
            freeze(OUT / "HALT.json", {"reason_type": type(exc).__name__, "automatic_restart": False})
        raise
