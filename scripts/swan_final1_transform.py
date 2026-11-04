"""Detector-blind final1 boundary intervention; imports no AMR/scoring code."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")
LEAK_RE = re.compile(
    r"(?:template\s+AMR|Abstract Meaning Representation|:ARG\d|"
    r"\(\s*[a-z]\d*\s*/|\b(?:NE|N|X)\b)"
)
ABBREVIATION_END = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|Inc|Ltd)|\b[A-Z]|"
    r"(?:\b[A-Za-z]\.)+[A-Za-z])\.$", re.I
)
PROBLEM_PREFIX = re.compile(
    r"^(?:and|or|but|nor|yet|so|because|although|though|if|when|while|"
    r"whereas|unless|until|since|that|which|who|whose)\b", re.I
)
PRONOUN_PREFIX = re.compile(
    r"^(?:He|She|It|They|We|I|This|That|These|Those|His|Her|Their|Its)\b"
)


def digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_tokens(text):
    return WORD_RE.findall(text)


@lru_cache(maxsize=1)
def _punkt():
    # No network download, AMR model, or statistical detection.
    import nltk
    from nltk.tokenize.punkt import PunktTokenizer
    nltk.data.path.insert(0, str(ROOT / ".nltk_data"))
    return PunktTokenizer("english")


def sentence_spans(text):
    return [list(span) for span in _punkt().span_tokenize(text)]


def unit_texts(text, spans):
    return [text[start:end] for start, end in spans]


def document_structure(slot_texts):
    """Acceptances/fallback flags deliberately absent from this interface."""
    text = " ".join(slot_texts)
    spans = sentence_spans(text)
    words = len(word_tokens(text))
    each_one = bool(slot_texts) and all(
        unit_texts(s, sentence_spans(s)) == [s]
        and bool(re.search(r'[.!?]["\')\]]*$', s))
        for s in slot_texts
    )
    checks = {
        "word_count": words, "num_detection_units": len(spans),
        "word_range_pass": 100 <= words <= 220,
        "unit_count_pass": 8 <= len(spans) <= 24,
        "each_slot_one_complete_unit": each_one,
        "slot_document_segmentation_equal": unit_texts(text, spans) == slot_texts,
        "amr_or_placeholder_leak": bool(LEAK_RE.search(text)),
    }
    checks["structural_pass"] = (
        checks["word_range_pass"] and checks["unit_count_pass"]
        and each_one and checks["slot_document_segmentation_equal"]
        and not checks["amr_or_placeholder_leak"]
    )
    return {"text": text, "segmentation_spans": spans, **checks}


def stop_document(slot_texts):
    """Stop immediately on first crossing; qualification is a separate step."""
    if len(word_tokens(" ".join(slot_texts))) >= 100:
        return "first_100_words"
    if len(slot_texts) >= 24:
        return "maximum_slots"
    return None


class FrozenSyntax:
    """Loaded only for authorized execution, never in pure synthetic tests."""
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

    @lru_cache(maxsize=8192)
    def __call__(self, text):
        doc = self.nlp(text)
        roots = [t for t in doc if t.dep_ == "ROOT"]
        root = roots[0] if len(roots) == 1 else None
        subjects = [t for t in doc if t.dep_ in {"nsubj", "nsubjpass", "csubj"}]
        # Subject must belong to the finite root clause, not only a subordinate.
        root_cluster = {root.i} if root is not None else set()
        if root is not None:
            root_cluster.update(t.i for t in root.children if t.dep_ in {"aux", "auxpass", "cop"})
        finite = [t for t in doc if t.i in root_cluster and
                  (t.tag_ in {"VBD", "VBP", "VBZ", "MD"} or
                   "Fin" in t.morph.get("VerbForm"))]
        complete = bool(root is not None and finite and
                        any(t.head.i == root.i for t in subjects))
        return {
            "complete": complete,
            "capitalization_valid_mid_sentence": bool(doc and (
                doc[0].pos_ == "PROPN" or doc[0].text == "I")),
            "content_lemmas": sorted({t.lemma_.casefold() for t in doc
                                     if t.pos_ in {"NOUN", "PROPN", "VERB"}
                                     and not t.is_stop}),
        }


def punctuation(text):
    return Counter(c for c in text if unicodedata.category(c).startswith("P"))


def unsupported(text):
    if any(c in text for c in '\n\r\t"“”‘()[]{}'):
        return "quotes_brackets_or_layout"
    if re.search(r"(?:^|[.!?]\s+)(?:[-*•]|\d+[.)])\s", text):
        return "list"
    for i, c in enumerate(text):
        if c in {"'", "\u2019"}:
            if not (i > 0 and i + 1 < len(text) and
                    text[i - 1].isalnum() and text[i + 1].isalnum()):
                return "quotation_apostrophe"
        if unicodedata.category(c).startswith("P") and c not in ".,;:!?'-\u2019":
            return "unsupported_unicode_punctuation"
    if re.search(r"[.!?]{2,}", text):
        return "ambiguous_terminal_punctuation"
    return None


def _span_record(text, segment):
    spans = segment(text)
    previous = 0
    for start, end in spans:
        if not previous <= start < end <= len(text) or text[previous:start].strip():
            raise ValueError("Invalid segmentation offsets")
        previous = end
    if text[previous:].strip():
        raise ValueError("Uncovered text")
    return {
        "text": text, "document_sha256": text_hash(text),
        "word_tokens_sha256": digest(word_tokens(text)),
        "segmentation_spans": spans, "num_detection_units": len(spans),
        "unit_texts": unit_texts(text, spans),
    }


def _candidates(text, spans, syntax):
    merges, splits = [], []
    for index, (start, end) in enumerate(spans):
        sentence = text[start:end]
        if index + 1 < len(spans) and sentence.endswith(".") \
                and not ABBREVIATION_END.search(sentence):
            right = text[slice(*spans[index + 1])]
            left_info, right_info = syntax(sentence), syntax(right)
            shared = sorted(set(left_info["content_lemmas"]) &
                            set(right_info["content_lemmas"]))
            linked = bool(shared or PRONOUN_PREFIX.match(right))
            if left_info["complete"] and right_info["complete"] and linked \
                    and right_info["capitalization_valid_mid_sentence"]:
                merges.append({"unit_index": index, "position": end - 1,
                               "shared_lemmas": shared,
                               "pronoun_link": bool(PRONOUN_PREFIX.match(right))})
        for match in re.finditer(r"[,;](?= +[A-Z0-9])", sentence):
            position = start + match.start()
            left = text[start:position].strip() + "."
            right = text[position + 1:end].strip()
            if PROBLEM_PREFIX.match(left) or PROBLEM_PREFIX.match(right):
                continue
            if syntax(left)["complete"] and syntax(right)["complete"]:
                splits.append({"unit_index": index, "position": position,
                               "original_punctuation": text[position]})
    return merges, splits


def _edits(text, merge, split=None):
    edits = [{"position": merge["position"], "before": ".", "after": ";",
              "kind": "merge"}]
    if split is not None:
        edits.append({"position": split["position"],
                      "before": text[split["position"]], "after": ".",
                      "kind": "split"})
    return sorted(edits, key=lambda row: row["position"])


def apply_edits(text, edits, reverse=False):
    positions = [row["position"] for row in edits]
    if len(set(positions)) != len(positions):
        raise ValueError("Overlapping edits")
    output = list(text)
    for row in edits:
        before, after = row["before"], row["after"]
        if len(before) != 1 or len(after) != 1:
            raise ValueError("Only character substitutions are permitted")
        if reverse:
            before, after = after, before
        if output[row["position"]] != before:
            raise ValueError("Edit provenance does not match text")
        output[row["position"]] = after
    return "".join(output)


def restore_original(c2, original_spans, original_hash):
    restored = apply_edits(c2["text"], c2["punctuation_edits"], reverse=True)
    if text_hash(restored) != original_hash:
        raise ValueError("Exact original restoration failed")
    return {"visible_text": c2["text"], "restored_text": restored,
            "segmentation_spans": original_spans,
            "unit_texts": unit_texts(restored, original_spans),
            "num_detection_units": len(original_spans),
            "oracle_uses_original_spans_and_edit_record": True,
            "document_sha256": original_hash}


def transform(text, syntax, segment=sentence_spans):
    """Choose once from linguistic candidates, then validate once; no retries."""
    c0 = _span_record(text, segment)
    result = {"schema_version": 1, "algorithm": "final1_v1",
              "input_sha256": text_hash(text), "conditions": {"C0": c0},
              "failures": {}, "detector_feedback": False}
    issue = unsupported(text)
    if issue or c0["num_detection_units"] < 3:
        result["failures"] = {c: issue or "too_few_units" for c in ("C1", "C2", "C3", "C4")}
        return result
    merges, splits = _candidates(text, c0["segmentation_spans"], syntax)
    tuples = [(split, merge) for split in splits for merge in merges
              if split["unit_index"] not in {merge["unit_index"], merge["unit_index"] + 1}]
    tuples.sort(key=lambda pair: (pair[0]["position"], pair[1]["position"]))
    result["candidate_counts"] = {"merges": len(merges), "splits": len(splits),
                                  "eligible_pairs": len(tuples)}
    if not tuples:
        result["failures"] = {c: "no_eligible_swap" for c in ("C1", "C2", "C3", "C4")}
        return result
    split, merge = tuples[0]
    result["selected_pair"] = {"merge": merge, "split": split}
    alternatives = [m for s, m in tuples if s == split and m != merge]
    plans = {"C1": _edits(text, merge), "C2": _edits(text, merge, split)}
    if alternatives:
        plans["C3"] = _edits(text, alternatives[0], split)
        result["selected_control_merge"] = alternatives[0]
    else:
        result["failures"]["C3"] = "no_alternative_merge"
    for condition, edits in plans.items():
        changed = apply_edits(text, edits)
        row = {**_span_record(changed, segment), "punctuation_edits": edits}
        expected_n = c0["num_detection_units"] - (condition == "C1")
        if word_tokens(changed) != word_tokens(text) or len(changed) != len(text):
            raise AssertionError("Case-sensitive word or offset invariant failed")
        if row["num_detection_units"] != expected_n:
            result["failures"][condition] = "selected_pair_segmentation_failed_no_retry"
            continue
        # Same N is insufficient: verify every intended boundary by original offset.
        expected_ends = {end for _, end in c0["segmentation_spans"]}
        expected_ends.remove(next(e["position"] + 1 for e in edits if e["kind"] == "merge"))
        if condition != "C1":
            expected_ends.add(split["position"] + 1)
        if {end for _, end in row["segmentation_spans"]} != expected_ends:
            result["failures"][condition] = "unexpected_boundary_positions_no_retry"
            continue
        result["conditions"][condition] = row
    conditions = result["conditions"]
    if "C2" in conditions and "C3" in conditions:
        if punctuation(conditions["C2"]["text"]) != punctuation(conditions["C3"]["text"]):
            raise AssertionError("C2/C3 punctuation multiset differs")
    if "C2" in conditions:
        conditions["C4"] = restore_original(
            conditions["C2"], c0["segmentation_spans"], c0["document_sha256"])
        if conditions["C4"]["unit_texts"] != c0["unit_texts"]:
            raise AssertionError("C4 units differ from C0")
    else:
        result["failures"]["C4"] = "C2_failed"
    return result
