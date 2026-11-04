"""Post-hoc diagnostics for the failed bank100 microfact null construction."""

import argparse
import json
from pathlib import Path
import statistics

from run_segmentation_intervention import ROOT, write_json
from run_swan_atomic_fact_preflight import configure_environment
from run_swan_bank100_microfact_null import (
    decode_facts,
    structure_metrics,
)

OUTPUT = ROOT / "experiments/swan_bank100_microfact_null1"


def minimally_repair_missing_array_close(content):
    content = content.strip()
    try:
        return decode_facts(content), False
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    if (
        not content.startswith('{"facts":[')
        or not content.endswith("}")
        or content.endswith("]}")
    ):
        return None, False
    candidate = content[:-1] + "]}"
    try:
        return decode_facts(candidate), True
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None, False


def diagnose(folder):
    import nltk

    rows = []
    review_by_id = {
        path.stem: json.loads(path.read_text())
        for path in (folder / "reviews").glob("*.json")
    }
    for path in sorted((folder / "generated").glob("*.json")):
        saved = json.loads(path.read_text())
        facts, repaired = minimally_repair_missing_array_close(
            saved["response"]["content"]
        )
        metrics = structure_metrics(facts, nltk.sent_tokenize)
        review = review_by_id[saved["prompt_id"]]
        rows.append({
            "prompt_id": saved["prompt_id"],
            "formal_decoded": saved["structure"]["decoded"],
            "minimal_missing_bracket_repair": repaired,
            "diagnostic_decoded": facts is not None,
            "diagnostic_structure": metrics,
            "formal_gates": review["gates"],
            "formal_quality_pass": review["quality_pass"],
        })
    recovered = [row for row in rows if row["diagnostic_decoded"]]
    fact_words = [
        words
        for row in recovered
        for words in row["diagnostic_structure"]["fact_words"]
    ]
    checks = (
        list(recovered[0]["diagnostic_structure"]["checks"])
        if recovered
        else []
    )
    result = {
        "status": "exploratory_post_hoc",
        "formal_labels_changed": False,
        "documents": len(rows),
        "strict_json_decoded": sum(row["formal_decoded"] for row in rows),
        "minimal_missing_bracket_repairs": sum(
            row["minimal_missing_bracket_repair"] for row in rows
        ),
        "diagnostic_decoded": len(recovered),
        "formal_structure_pass": sum(
            row["formal_gates"]["structure"] for row in rows
        ),
        "diagnostic_structure_pass": sum(
            row["diagnostic_structure"]["pass"] for row in recovered
        ),
        "diagnostic_document_check_passes": {
            check: sum(
                row["diagnostic_structure"]["checks"].get(check, False)
                for row in recovered
            )
            for check in checks
        },
        "diagnostic_fact_words": {
            "facts": len(fact_words),
            "within_7_10": sum(7 <= words <= 10 for words in fact_words),
            "over_10": sum(words > 10 for words in fact_words),
            "under_7": sum(words < 7 for words in fact_words),
            "mean": statistics.mean(fact_words) if fact_words else None,
            "median": statistics.median(fact_words) if fact_words else None,
            "min": min(fact_words, default=None),
            "max": max(fact_words, default=None),
        },
        "formal_review_passes": {
            gate: sum(row["formal_gates"][gate] for row in rows)
            for gate in ("style", "facts", "atomicity")
        },
        "rows": rows,
        "interpretation": (
            "The one-character repairs are diagnostic copies only. They show whether "
            "strict JSON closure was the sole blocker; they do not change formal "
            "structure, quality, parser, detector, or continuation labels."
        ),
    }
    write_json(folder / "generation_failure_diagnostic.json", result)
    write_report(folder, result)
    return result


def write_report(folder, result):
    words = result["diagnostic_fact_words"]
    checks = result["diagnostic_document_check_passes"]
    lines = [
        "# Post-hoc microfact generation failure diagnostic",
        "",
        "This analysis does not alter any frozen label.",
        "",
        f"- Strict JSON decoded: **{result['strict_json_decoded']}/"
        f"{result['documents']}**.",
        f"- One-character missing-bracket diagnostic repairs: "
        f"**{result['minimal_missing_bracket_repairs']}/"
        f"{result['documents']}**.",
        f"- Diagnostic copies decoded: **{result['diagnostic_decoded']}/"
        f"{result['documents']}**.",
        f"- Formal structure pass: **{result['formal_structure_pass']}/"
        f"{result['documents']}**.",
        f"- Diagnostic structure pass after minimal copies: "
        f"**{result['diagnostic_structure_pass']}/"
        f"{result['documents']}**.",
        f"- Individual facts within 7--10 words: **{words['within_7_10']}/"
        f"{words['facts']}**; over 10 words: **{words['over_10']}**.",
        f"- Document fact-word-bound pass after diagnostic repair: "
        f"**{checks.get('fact_word_bounds', 0)}/{result['documents']}**.",
        f"- Formal review passes: style "
        f"**{result['formal_review_passes']['style']}/{result['documents']}**, "
        f"facts **{result['formal_review_passes']['facts']}/{result['documents']}**, "
        f"atomicity **{result['formal_review_passes']['atomicity']}/"
        f"{result['documents']}**.",
        "",
        "The failure is not only a missing JSON bracket: no document satisfies all "
        "short-sentence bounds even after the minimal diagnostic repair.",
        "",
    ]
    (folder / "generation_failure_diagnostic.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    configure_environment()
    result = diagnose(args.output)
    print(json.dumps({
        key: result[key]
        for key in (
            "status",
            "documents",
            "strict_json_decoded",
            "minimal_missing_bracket_repairs",
            "diagnostic_decoded",
            "formal_structure_pass",
            "diagnostic_structure_pass",
            "diagnostic_fact_words",
            "formal_review_passes",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
