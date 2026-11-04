"""Freeze the complete historical short benchmark and separated natural nulls."""
import argparse
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.embedding_amr import digest
from src.core.embedding_amr_runtime import atomic, file_sha
from run_embedding_amr_detection import input_rows

BANK = "f0a88c1af73d12078586d10ef68804c12ed5b79cabc07c9f6c2a34bb084ef5a4"
CONDITIONS = ("clean", "joined", "split")


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def edited(text):
    joined = re.sub(r"[.!?;]+(?=\s|$)", "", text).strip() + "."
    words = joined.split()
    split = " ".join(w + ("." if (i + 1) % 6 == 0 and not w.endswith((".", "!", "?", ";"))
                         else "") for i, w in enumerate(words))
    result = {"clean": text, "joined": joined, "split": split}
    tokens = lambda t: re.findall(r"\w+(?:['’\-]\w+)*", t)
    assert all(tokens(t) == tokens(text) for t in result.values())
    return result


def prepare(out):
    if out.exists():
        raise ValueError("New output directory required")
    old = ROOT / "experiments/confirmation_three_arm30"
    runtime = json.loads((old / "runtime.json").read_text())
    preparation = json.loads((old / "preparation.json").read_text())
    prompts = {p["prompt_id"]: p for p in preparation["prompts"]}
    assert runtime["dependencies"]["third_party/SWAN/amr_bank/banks/amr_bank_50.json"] == BANK
    seals = {str(p.relative_to(ROOT)): file_sha(p)
             for p in (old / "runtime.json", old / "preparation.json")}
    rows = {"calibration": [], "test": []}
    identities = {}
    positive_metadata, review_packets = [], []
    for split, name in (("calibration", "calibration"), ("test", "null")):
        parent = ROOT / "experiments/swan_v4_independent_confirmation1"
        manifest = json.loads((parent / "INPUT_MANIFEST.json").read_text())
        path = parent / manifest["outputs"][name]["path"]
        assert file_sha(path) == manifest["outputs"][name]["sha256"]
        seals[str(path.relative_to(ROOT))] = file_sha(path)
        values = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(values) == 1000
        ids, hashes, urls = set(), set(), set()
        for value in values:
            assert digest(value["text"]) == value["window_sha256"]
            assert digest(canonical(value["text"])) == value["normalized_window_sha256"]
            sid = value["allocation_id"]
            assert sid not in ids
            ids.add(sid)
            hashes.update(value[k] for k in ("source_sha256", "normalized_source_sha256",
                                             "window_sha256", "normalized_window_sha256"))
            urls.add(value["url_sha256"])
            for condition, text in edited(value["text"]).items():
                rows[split].append(dict(row_id=sid + "::" + condition, source_id=sid,
                    source_sha256=value["source_sha256"], condition=condition,
                    kind="human", split=split, text=text))
        identities[split] = {"ids": ids, "hashes": hashes, "urls": urls}
    overlap = {k: len(identities["calibration"][k] & identities["test"][k])
               for k in identities["test"]}
    assert not any(overlap.values()), overlap
    baselines = sorted((old / "baselines").glob("*.json"))
    assert len(baselines) == 30
    for path in baselines:
        value = json.loads(path.read_text())
        prompt = prompts[value["prompt_id"]]
        assert value["complete"] and len(value["sentences"]) == 5
        assert digest(value["text"]) == value["text_sha256"]
        assert value["runtime_sha256"] == digest(runtime)
        assert value["prompt_sha256"] == prompt["generation_prompt_sha256"]
        assert all(s["runtime_sha256"] == digest(runtime) for s in value["sentences"])
        assert value["text"] == " ".join(s["text"] for s in value["sentences"])
        accepted = sum(s["accepted"] for s in value["sentences"])
        assert value["all_injection_accepted"] == (accepted == 5)
        identity = {prompt["raw_sha256"], prompt["normalized_sha256"],
                    digest(prompt["text"]), digest(canonical(prompt["text"])),
                    digest(value["text"]), digest(canonical(value["text"]))}
        for group in identities.values():
            assert not identity & group["hashes"]
            assert digest(prompt["url"]) not in group["urls"]
        sid = "benchmark30-" + value["prompt_id"]
        positive_metadata.append({"source_id": sid, "accepted_slots": accepted,
                                  "all_injection_accepted": accepted == 5,
                                  "writer_bank_sha256": BANK,
                                  "baseline_sha256": file_sha(path)})
        for condition, text in edited(value["text"]).items():
            rows["test"].append(dict(row_id=sid + "::" + condition, source_id=sid,
                source_sha256=digest(prompt["text"]), condition=condition,
                kind="watermarked", split="test", text=text))
            review_packets.append({"row_id": sid + "::" + condition, "condition": condition,
                                   "source": value["text"], "candidate": text})
        seals[str(path.relative_to(ROOT))] = file_sha(path)
    assert sum(p["all_injection_accepted"] for p in positive_metadata) == 21
    out.mkdir()
    private = out / "private"
    private.mkdir(mode=0o700)
    (out / ".gitignore").write_text("private/\n*.log\n")
    for split, values in rows.items():
        path = private / (split + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values))
        input_rows(path)
    atomic(private / "review_packets.json", review_packets)
    # Non-efficacy engineering profile. It never enters calibration/test.
    examples = [
        "Alice lent a book to Bob. Bob read the book. Alice visited the library.",
        "The driver stopped the bus. The passengers left safely. A mechanic checked the engine.",
        "The council approved the plan. Residents discussed the budget. Workers repaired the road.",
    ]
    (private / "profile.jsonl").write_text("".join(json.dumps({
        "row_id": f"profile-{i}", "source_id": f"profile-{i}", "source_sha256": digest(t),
        "condition": "clean", "kind": "synthetic", "split": "development", "text": t
    }) + "\n" for i, t in enumerate(examples)))
    protocol = {
        "version": "embedding-amr-short-benchmark/v1",
        "scope": "exploratory_re_evaluation_of_historical_30_short_writing_attempts",
        "conditions": CONDITIONS, "alpha": .05,
        "calibration_sources": 1000, "test_null_sources": 1000, "positive_attempts": 30,
        "fully_written_subgroup": 21, "bank_sha256": BANK, "bank_is_public": True,
        "candidate_budget": 24, "top_k": 4, "min_similarity": .35,
        "parser_batch_size": 8, "dtype": "float32",
        "edits": "remove terminal punctuation before whitespace; split joined every six tokens",
        "denominator": "all_attempts_including_partial_writes_and_abstentions",
        "reviews": {"type": "provider_model", "rows": 90, "filtering": False,
                    "blind_to": ["Bank", "scores", "decisions", "writing_acceptance"],
                    "checks": ["meaning", "natural", "coherent", "plausible"],
                    "interpretation": "model_judgment_not_external_fact_verification"},
        "calibration_sealed_before_test_scoring": True,
        "comparisons": "two_fixed_controls_exact_McNemar_Holm",
        "confirmatory_positive_claim": False,
        "no_old_cache": True, "no_score_filtering": True,
        "preparation_code_sha256": file_sha(__file__),
        "source_seals": seals, "cross_split_overlap": overlap,
        "positive_metadata": positive_metadata,
        "files": {str(p.relative_to(out)): file_sha(p) for p in sorted(private.iterdir())},
    }
    atomic(out / "PROTOCOL.json", protocol)
    atomic(out / "INPUT_TERMINAL.json", {"status": "inputs_frozen",
                                       "protocol_sha256": file_sha(out / "PROTOCOL.json")})
    print(json.dumps({"status": "inputs_frozen", "rows": {s: len(v) for s, v in rows.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
