"""Run a local dependency-path pruner over frozen DeepSeek span plans."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re

from feedback2_runtime import Heartbeat, claim
from prepare_confirmation_experiment import normalized_hash
from run_confirmation_experiment import (
    CallJournal,
    digest,
    freeze,
    request_body,
)
from run_segmentation_intervention import ROOT, sha, text_sha, write_json
from run_swan_atomic_fact_preflight import MODEL_DIR, TOKENIZER_DIR, VECTORS_PATH
from run_swan_bank100_microfact_null import BANK_PATH, parse_quality_machine
from run_swan_constrained_microfact import (
    MAX_PRIOR_JACCARD,
    content_words,
    decode_grounding_review,
    document_structure,
    jaccard,
    quality_gates,
    style_body,
)
from run_swan_deepseek_span_compiler import (
    canonical,
    grounding_body,
    ordered_spans,
    span_boundary_ok,
)
from run_swan_microfact_sequential import graph_document
from run_swan_sentence_growth import decode_baseline_review
from supervise_confirmation_experiment import read_credential


OUTPUT = ROOT / "experiments/swan_dependency_pruner1"
INPUT = ROOT / "experiments/swan_deepseek_span_compiler1"
DOCUMENTS = 5
FACTS = 15
MIN_WORDS = 7
MAX_WORDS = 10
MIN_JOINT_DOCUMENTS = 3
API_CAP = 11
UNRESOLVED = {
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "it",
    "its",
    "she",
    "that",
    "their",
    "theirs",
    "them",
    "these",
    "they",
    "this",
    "those",
    "we",
    "you",
    "your",
    "yours",
}
CONDITIONAL_STARTS = ("if ", "unless ", "although ", "while ", "when ")
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj"}
COMPLEMENT_DEPS = {
    "dobj",
    "obj",
    "iobj",
    "attr",
    "oprd",
    "dative",
    "pobj",
    "acomp",
    "xcomp",
    "ccomp",
}
PATH_CHILD_DEPS = {
    "aux",
    "auxpass",
    "neg",
    "prt",
}
HEAD_MODIFIER_DEPS = {
    "compound",
    "flat",
    "nummod",
    "poss",
    "case",
}
OPTIONAL_PRIORITY = {
    "compound": 0,
    "flat": 0,
    "nummod": 0,
    "poss": 0,
    "case": 0,
    "det": 1,
    "amod": 2,
    "advmod": 3,
}
MODALS = {
    "can",
    "could",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
}


def configure_environment():
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("GLOVE_VECTORS_PATH", str(VECTORS_PATH))


def lexical_first(text):
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return words[0] if words else None


def base_plan_metrics(plan, source):
    subject = canonical(plan["subject"])
    predicate = canonical(plan["predicate"])
    object_text = canonical(plan["object"])
    evidence = canonical(plan["evidence"])
    fields = [subject, predicate, object_text]
    locations = ordered_spans(evidence, fields)
    subject_start = locations[0][0] if locations else None
    first_comma = evidence.find(",")
    conditional_prefix = evidence.lower().startswith(CONDITIONAL_STARTS)
    conditional_subject = bool(
        conditional_prefix
        and first_comma >= 0
        and subject_start is not None
        and subject_start < first_comma
    )
    checks = {
        "evidence_in_source": evidence in source,
        "ordered_nonoverlapping_spans": locations is not None,
        "span_punctuation": all(
            not re.search(r"[.?!\n\r]", field) for field in fields
        ),
        "span_boundaries": all(span_boundary_ok(field) for field in fields),
        "resolved_subject": lexical_first(subject) not in UNRESOLVED,
        "not_question_evidence": not evidence.rstrip().endswith("?"),
        "not_initial_conditional_clause": not conditional_subject,
    }
    body = f"{subject} {predicate} {object_text}"
    return {
        "body": body,
        "body_sha256": text_sha(body),
        "evidence": evidence,
        "span_locations": locations,
        "checks": checks,
        "pass": all(checks.values()),
    }


def path_to_root(token, root):
    path = [token]
    current = token
    seen = {token.i}
    while current.i != root.i:
        current = current.head
        if current.i in seen:
            return None
        seen.add(current.i)
        path.append(current)
    return path


def whitespace_units(text):
    return [
        {
            "index": index,
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for index, match in enumerate(re.finditer(r"\S+", text))
    ]


def token_unit(token, units):
    for unit in units:
        if token.idx < unit["end"] and token.idx + len(token.text) > unit["start"]:
            return unit["index"]
    raise ValueError("spaCy token did not align to whitespace unit")


def descendants_with_deps(token, deps):
    retained = []
    stack = list(token.children)
    while stack:
        child = stack.pop()
        if child.dep_ in deps:
            retained.append(child)
            stack.extend(child.children)
    return retained


def connected_to_root(token, root):
    return path_to_root(token, root) is not None


def clean_unit(text):
    return text.strip("()[]{}\"'\u201c\u201d")


def has_negation(doc):
    return any(
        token.dep_ == "neg"
        or token.lower_ in {"not", "never", "no", "n't", "n\u2019t"}
        for token in doc
    )


def validate_pruned(text, original_doc, nlp):
    words = text.split()
    final_doc = nlp(text)
    sentences = list(final_doc.sents)
    root = sentences[0].root if len(sentences) == 1 else None
    subjects = [
        token for token in final_doc
        if token.dep_ in SUBJECT_DEPS
    ]
    complements = [
        token for token in final_doc
        if token.dep_ in COMPLEMENT_DEPS
    ]
    original_modals = {
        token.lower_ for token in original_doc
        if token.lower_ in MODALS
    }
    final_modals = {
        token.lower_ for token in final_doc
        if token.lower_ in MODALS
    }
    checks = {
        "word_bounds": MIN_WORDS <= len(words) <= MAX_WORDS,
        "one_terminal_period": (
            text.endswith(".")
            and not re.search(r"[.?!]", text[:-1])
        ),
        "one_sentence": len(sentences) == 1,
        "verbal_root": bool(root and root.pos_ in {"VERB", "AUX"}),
        "explicit_subject": bool(subjects),
        "resolved_subject": lexical_first(text) not in UNRESOLVED,
        "object_or_complement": bool(complements),
        "negation_preserved": (
            not has_negation(original_doc) or has_negation(final_doc)
        ),
        "modals_preserved": original_modals <= final_modals,
    }
    return {
        "words": len(words),
        "root": root.text if root else None,
        "subject_tokens": [token.text for token in subjects],
        "complement_tokens": [token.text for token in complements],
        "original_modals": sorted(original_modals),
        "final_modals": sorted(final_modals),
        "checks": checks,
        "pass": all(checks.values()),
    }


def prune_body(body, nlp):
    units = whitespace_units(body)
    original_doc = nlp(body)
    sentences = list(original_doc.sents)
    if len(sentences) != 1:
        return {
            "text": None,
            "reason": "compiled_input_not_one_sentence",
            "selected_units": [],
            "validation": None,
        }
    root = sentences[0].root
    if root.pos_ not in {"VERB", "AUX"}:
        return {
            "text": None,
            "reason": "compiled_input_has_no_verbal_root",
            "selected_units": [],
            "validation": None,
        }
    if MIN_WORDS <= len(units) <= MAX_WORDS:
        text = " ".join(clean_unit(unit["text"]) for unit in units) + "."
        validation = validate_pruned(text, original_doc, nlp)
        return {
            "text": text,
            "text_sha256": text_sha(text),
            "reason": None if validation["pass"] else "passthrough_validation_failed",
            "mode": "passthrough",
            "root_token": root.text,
            "subject_head": None,
            "complement_head": None,
            "mandatory_token_indices": [],
            "selected_units": [unit["index"] for unit in units],
            "validation": validation,
        }
    subjects = [
        token for token in original_doc
        if token.dep_ in SUBJECT_DEPS and connected_to_root(token, root)
    ]
    complements = [
        token for token in original_doc
        if token.dep_ in COMPLEMENT_DEPS and connected_to_root(token, root)
    ]
    if not subjects or not complements:
        return {
            "text": None,
            "reason": "compiled_input_missing_subject_or_complement",
            "selected_units": [],
            "validation": None,
        }
    subject = min(
        subjects,
        key=lambda token: (
            0 if token.i < root.i else 1,
            len(path_to_root(token, root) or []),
            token.i,
        ),
    )
    complement = min(
        complements,
        key=lambda token: (
            0
            if token.dep_
            in {"dobj", "obj", "iobj", "attr", "oprd", "dative", "acomp"}
            and token.pos_ not in {"NUM", "PRON"}
            else 1
            if token.dep_ == "pobj"
            else 2
            if token.dep_
            in {"dobj", "obj", "iobj", "attr", "oprd", "dative", "acomp"}
            else 3,
            0 if token.i > root.i else 1,
            len(path_to_root(token, root) or []),
            token.i,
        ),
    )
    mandatory_tokens = set()
    for token in (subject, complement):
        path = path_to_root(token, root)
        if path is None:
            return {
                "text": None,
                "reason": "dependency_path_not_connected",
                "selected_units": [],
                "validation": None,
            }
        mandatory_tokens.update(item.i for item in path)
    for token in original_doc:
        if token.dep_ in PATH_CHILD_DEPS and connected_to_root(token, root):
            mandatory_tokens.add(token.i)
    dependency_path_tokens = {
        original_doc[index] for index in mandatory_tokens
    }
    for path_token in list(dependency_path_tokens):
        if path_token.pos_ not in {"VERB", "AUX"}:
            continue
        for child in path_token.children:
            if child.dep_ in SUBJECT_DEPS:
                mandatory_tokens.add(child.i)
                mandatory_tokens.update(
                    token.i
                    for token in descendants_with_deps(
                        child,
                        HEAD_MODIFIER_DEPS,
                    )
                )
    for head in (subject, complement):
        mandatory_tokens.update(
            token.i
            for token in descendants_with_deps(head, HEAD_MODIFIER_DEPS)
        )
    for child in subject.children:
        if child.dep_ != "conj":
            continue
        mandatory_tokens.add(child.i)
        mandatory_tokens.update(
            token.i
            for token in descendants_with_deps(
                child,
                HEAD_MODIFIER_DEPS | {"cc", "conj"},
            )
        )
        mandatory_tokens.update(
            sibling.i for sibling in child.children
            if sibling.dep_ == "cc"
        )
    mandatory_units = {
        token_unit(original_doc[index], units)
        for index in mandatory_tokens
        if not original_doc[index].is_punct
    }
    if len(mandatory_units) > MAX_WORDS:
        return {
            "text": None,
            "reason": "mandatory_path_exceeds_word_budget",
            "selected_units": sorted(mandatory_units),
            "validation": None,
        }
    selected_units = set(mandatory_units)
    optional_groups = []
    group_seen = set()

    def add_group(priority, tokens):
        group = frozenset(
            token_unit(token, units)
            for token in tokens
            if not token.is_punct
        )
        group = group - selected_units
        if not group or group in group_seen:
            return
        group_seen.add(group)
        optional_groups.append(
            (
                priority,
                min(group),
                group,
            )
        )

    for head in (subject, complement):
        for child in head.children:
            if child.dep_ not in OPTIONAL_PRIORITY:
                continue
            tokens = [child]
            tokens.extend(
                descendants_with_deps(child, HEAD_MODIFIER_DEPS)
            )
            add_group(OPTIONAL_PRIORITY[child.dep_], tokens)
    for path_token in {
        original_doc[index] for index in mandatory_tokens
    }:
        for child in path_token.children:
            if child.dep_ == "advmod" and not any(
                ancestor.dep_ in {"advcl", "relcl"}
                for ancestor in child.ancestors
            ):
                add_group(3, [child])
            if child.dep_ == "prep":
                group = [child]
                for descendant in child.subtree:
                    if (
                        descendant.dep_
                        in COMPLEMENT_DEPS
                        | HEAD_MODIFIER_DEPS
                        | {"det", "amod"}
                    ):
                        group.append(descendant)
                add_group(4, group)
            if child.dep_ in {"xcomp", "ccomp"}:
                group = [child]
                for descendant in child.subtree:
                    if (
                        descendant.dep_
                        in SUBJECT_DEPS
                        | COMPLEMENT_DEPS
                        | PATH_CHILD_DEPS
                        | HEAD_MODIFIER_DEPS
                        | {"det", "amod"}
                    ):
                        group.append(descendant)
                add_group(5, group)
    for _, _, group in sorted(optional_groups):
        if len(selected_units) >= MIN_WORDS:
            break
        if len(selected_units | group) <= MAX_WORDS:
            selected_units.update(group)
    if len(selected_units) < MIN_WORDS:
        return {
            "text": None,
            "reason": "insufficient_safe_dependency_units",
            "selected_units": sorted(selected_units),
            "validation": None,
        }
    selected_units = set(sorted(selected_units)[:MAX_WORDS])
    output_units = []
    for index in sorted(selected_units):
        cleaned = clean_unit(units[index]["text"])
        if cleaned:
            output_units.append(cleaned)
    text = " ".join(output_units) + "."
    validation = validate_pruned(text, original_doc, nlp)
    return {
        "text": text,
        "text_sha256": text_sha(text),
        "reason": None if validation["pass"] else "pruned_validation_failed",
        "root_token": root.text,
        "subject_head": subject.text,
        "complement_head": complement.text,
        "mandatory_token_indices": sorted(mandatory_tokens),
        "selected_units": sorted(selected_units),
        "validation": validation,
    }


def prune_plan(plan, source, previous, nlp):
    base = base_plan_metrics(plan, source)
    pruning = (
        prune_body(base["body"], nlp)
        if base["pass"]
        else {
            "text": None,
            "reason": "base_plan_rejected",
            "selected_units": [],
            "validation": None,
        }
    )
    text = pruning["text"]
    unique = bool(
        text
        and normalized_hash(text) not in {
            normalized_hash(fact) for fact in previous
        }
    )
    maximum_jaccard = (
        max(
            (
                jaccard(content_words(text), content_words(fact))
                for fact in previous
            ),
            default=0.0,
        )
        if text
        else 1.0
    )
    checks = {
        "base_plan": base["pass"],
        "dependency_prune": bool(
            pruning["validation"] and pruning["validation"]["pass"]
        ),
        "normalized_unique": unique,
        "prior_jaccard": maximum_jaccard <= MAX_PRIOR_JACCARD,
    }
    return {
        "base": base,
        "pruning": pruning,
        "maximum_prior_jaccard": maximum_jaccard,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def input_batches(source):
    paths = sorted((INPUT / "plans").glob(f"{source['prompt_id']}-*.json"))
    if len(paths) != 2:
        raise ValueError("Expected two frozen planning batch artifacts")
    return [json.loads(path.read_text()) for path in paths]


def process_source(source, nlp):
    selected = []
    candidates = []
    for batch_row in input_batches(source):
        for index, plan in enumerate(batch_row["plans"] or []):
            metrics = prune_plan(plan, source["source"], selected, nlp)
            selected_now = bool(
                len(selected) < FACTS and metrics["eligible"]
            )
            candidates.append(
                {
                    "batch": batch_row["batch"],
                    "index": index,
                    "plan": plan,
                    "metrics": metrics,
                    "selected": selected_now,
                    "selection_reason": (
                        "eligible_in_fixed_order"
                        if selected_now
                        else "quota_already_filled"
                        if metrics["eligible"]
                        else "pruner_rejected"
                    ),
                }
            )
            if selected_now:
                selected.append(metrics["pruning"]["text"])
    return {"selected": selected, "candidates": candidates}


def build_document(source, nlp, tokenize):
    processed = process_source(source, nlp)
    facts = processed["selected"]
    structure = document_structure(facts, tokenize)
    passage = " ".join(facts)
    return {
        "prompt_id": source["prompt_id"],
        "source_sha256": source["source_sha256"],
        "input_plan_sha256": [
            sha(path)
            for path in sorted(
                (INPUT / "plans").glob(f"{source['prompt_id']}-*.json")
            )
        ],
        "input_plans": len(processed["candidates"]),
        "prunable_plans": sum(
            row["metrics"]["eligible"]
            for row in processed["candidates"]
        ),
        "candidates": processed["candidates"],
        "facts": facts,
        "passage": passage,
        "passage_sha256": text_sha(passage),
        "structure": structure,
        "selection": {
            "selected_facts": len(facts),
            "required_facts": FACTS,
            "pass": bool(len(facts) == FACTS and structure["pass"]),
        },
    }


def package_versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "amrlib",
            "en_core_web_sm",
            "nltk",
            "numpy",
            "penman",
            "requests",
            "spacy",
            "torch",
            "transformers",
        )
    }


def dependency_paths():
    paths = [
        Path(__file__),
        ROOT / "scripts/audit_swan_dependency_pruner.py",
        OUTPUT / "protocol.md",
        INPUT / "manifest.json",
        INPUT / "summary.json",
        INPUT / "audit.json",
        INPUT / "recovery_verification.json",
        INPUT / "implementation_observations.json",
        ROOT / "scripts/feedback2_runtime.py",
        ROOT / "scripts/prepare_confirmation_experiment.py",
        ROOT / "scripts/run_confirmation_experiment.py",
        ROOT / "scripts/run_segmentation_intervention.py",
        ROOT / "scripts/run_swan_atomic_fact_preflight.py",
        ROOT / "scripts/run_swan_bank100_microfact_null.py",
        ROOT / "scripts/run_swan_constrained_microfact.py",
        ROOT / "scripts/run_swan_deepseek_span_compiler.py",
        ROOT / "scripts/run_swan_known_positive_audit.py",
        ROOT / "scripts/run_swan_microfact_sequential.py",
        ROOT / "scripts/run_swan_sentence_growth.py",
        ROOT / "scripts/supervise_confirmation_experiment.py",
        BANK_PATH,
        ROOT / "third_party/SWAN/utils/amr_utils.py",
        ROOT / "third_party/SWAN/utils/s2match_patch.py",
        MODEL_DIR / "amrlib_meta.json",
        MODEL_DIR / "model_parse_xfm_bart_large.json",
        MODEL_DIR / "config.json",
        MODEL_DIR / "pytorch_model.bin",
        TOKENIZER_DIR / "tokenizer.json",
        TOKENIZER_DIR / "tokenizer_config.json",
        VECTORS_PATH,
    ]
    paths.extend(sorted((INPUT / "plans").glob("*.json")))
    paths.extend(
        sorted((ROOT / ".nltk_data/tokenizers/punkt_tab/english").glob("*"))
    )
    return paths


def dependencies():
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in dependency_paths()
    }


def sources():
    manifest = json.loads((INPUT / "manifest.json").read_text())
    rows = manifest["sources"]
    if len(rows) != DOCUMENTS:
        raise ValueError("Expected five paired development sources")
    return rows


def prepare(folder=OUTPUT):
    configure_environment()
    path = folder / "manifest.json"
    if path.exists():
        return check(folder)
    manifest = {
        "status": "frozen_before_dependency_pruning",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Post-hoc development preflight over frozen observed plans; "
            "not an independent validation"
        ),
        "sources": sources(),
        "protocol": {
            "workflow": "swan_dependency_path_pruner_preflight",
            "documents": DOCUMENTS,
            "facts_per_document": FACTS,
            "input": {
                "strictly_decoded_plans": 270,
                "new_semantic_planning_calls": 0,
            },
            "pruner": {
                "spacy_model": "en_core_web_sm",
                "fact_word_range": [MIN_WORDS, MAX_WORDS],
                "preserve_dependency_paths": True,
                "preserve_negation_and_modals": True,
                "explicit_second_person_rejection": True,
                "fixed_first_eligible_selection": True,
                "fallback": False,
                "maximum_prior_content_jaccard": MAX_PRIOR_JACCARD,
                "amr_or_detector_inputs": False,
            },
            "quality": {
                "review_model": "deepseek-v4-flash",
                "style": "all dimensions >=4 and no severe problem",
                "facts": (
                    "faithfulness >=4, no unsupported claim, "
                    "contradiction, duplicate, or issue"
                ),
                "atomicity": "atomicity and self-containment >=4",
            },
            "graph_scale": {
                "bank_size": 100,
                "bank_sha256": sha(BANK_PATH),
                "sentence": (
                    "6..20 transformed triples and at least 10 "
                    "count-compatible bank100 templates at U>=0.75"
                ),
                "document": (
                    "15 parsed facts, at least 10 scale-compatible, "
                    "median transformed triples 8..16"
                ),
                "s2match_optimization": False,
            },
            "continuation_gate": {
                "minimum_joint_documents": MIN_JOINT_DOCUMENTS,
                "denominator": DOCUMENTS,
            },
            "api_cap": API_CAP,
            "failure_policy": (
                "No new plan, retry, fallback, source replacement, or "
                "detector-conditioned filtering"
            ),
            "no_watermark_injection_detection_or_attack": True,
        },
        "dependencies": dependencies(),
        "packages": package_versions(),
    }
    freeze(path, manifest)
    return manifest


def check(folder=OUTPUT):
    configure_environment()
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol"]["workflow"] != "swan_dependency_path_pruner_preflight":
        raise ValueError("Unexpected workflow")
    if manifest["sources"] != sources():
        raise ValueError("Paired source cohort changed")
    for relative, expected in manifest["dependencies"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen dependency changed: {relative}")
    if package_versions() != manifest["packages"]:
        raise ValueError("Frozen package versions changed")
    return manifest


class Journal(CallJournal):
    def __init__(self, folder, key, heartbeat):
        super().__init__(folder, key, API_CAP)
        self.prior = len(list((folder / "calls").glob("*.json")))
        self.heartbeat = heartbeat

    def request(self, scope, body):
        ordinal = self.counters.get(scope, 0)
        path = self.folder / "calls" / f"{scope}-{ordinal:03d}.json"
        if not path.exists() and self.prior + self.sent >= API_CAP:
            raise RuntimeError("Frozen cumulative API cap reached")
        self.heartbeat.update(
            api_scope=scope,
            api_attempts=self.prior + self.sent,
        )
        return super().request(scope, body)


def generated_path(folder, prompt_id):
    return folder / "generated" / f"{prompt_id}.json"


def construct_all(folder, manifest, heartbeat):
    import nltk
    import spacy

    heartbeat.update(stage="loading_dependency_parser")
    nlp = spacy.load("en_core_web_sm")
    for index, source in enumerate(manifest["sources"]):
        document = build_document(source, nlp, nltk.sent_tokenize)
        freeze(generated_path(folder, source["prompt_id"]), document)
        heartbeat.update(
            stage="dependency_pruning",
            completed_documents=index + 1,
            total_documents=DOCUMENTS,
            prompt_id=source["prompt_id"],
        )
        print(
            json.dumps(
                {
                    "stage": "dependency_pruning",
                    "prompt_id": source["prompt_id"],
                    "input_plans": document["input_plans"],
                    "prunable_plans": document["prunable_plans"],
                    "selected_facts": len(document["facts"]),
                    "words": document["structure"]["words"],
                    "construction_pass": document["selection"]["pass"],
                }
            ),
            flush=True,
        )


def review_all(folder, manifest, api, runtime, heartbeat):
    for index, source in enumerate(manifest["sources"]):
        generated = json.loads(
            generated_path(folder, source["prompt_id"]).read_text()
        )
        style_request = style_body(generated["passage"])
        source_request = grounding_body(
            source["source"],
            generated["facts"],
        )
        style_response = api.request(
            f"style-{source['prompt_id']}",
            style_request,
        )
        source_response = api.request(
            f"facts-{source['prompt_id']}",
            source_request,
        )
        errors = {}
        try:
            style = decode_baseline_review(style_response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            style = None
            errors["style"] = type(exc).__name__
        try:
            grounding = decode_grounding_review(source_response["content"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            grounding = None
            errors["grounding"] = type(exc).__name__
        gates = quality_gates(generated, style, grounding)
        result = {
            "prompt_id": source["prompt_id"],
            "source_sha256": source["source_sha256"],
            "passage_sha256": generated["passage_sha256"],
            "runtime_sha256": digest(runtime),
            "style_request_sha256": digest(style_request),
            "grounding_request_sha256": digest(source_request),
            "style_response": style_response,
            "grounding_response": source_response,
            "style": style,
            "grounding": grounding,
            "decode_errors": errors,
            "gates": gates,
            "quality_pass": all(gates.values()),
        }
        freeze(
            folder / "reviews" / f"{source['prompt_id']}.json",
            result,
        )
        heartbeat.update(
            stage="quality_review",
            completed_documents=index + 1,
            total_documents=DOCUMENTS,
        )
        print(
            json.dumps(
                {
                    "stage": "quality_review",
                    "prompt_id": source["prompt_id"],
                    "gates": gates,
                    "quality_pass": result["quality_pass"],
                }
            ),
            flush=True,
        )


def summarize(folder=OUTPUT):
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = []
    for source in manifest["sources"]:
        generated_file = generated_path(folder, source["prompt_id"])
        review_file = folder / "reviews" / f"{source['prompt_id']}.json"
        if not generated_file.exists() or not review_file.exists():
            continue
        generated = json.loads(generated_file.read_text())
        review = json.loads(review_file.read_text())
        parser_file = folder / "parser" / f"{source['prompt_id']}.json"
        graph = (
            graph_document(json.loads(parser_file.read_text()))
            if parser_file.exists()
            else None
        )
        rows.append(
            {
                "prompt_id": source["prompt_id"],
                "input_plans": generated["input_plans"],
                "prunable_plans": generated["prunable_plans"],
                "selected_facts": len(generated["facts"]),
                "words": generated["structure"]["words"],
                "gates": review["gates"],
                "quality_pass": review["quality_pass"],
                "graph": graph,
                "joint_pass": bool(
                    review["quality_pass"] and graph and graph["pass"]
                ),
            }
        )
    quality_pass = sum(row["quality_pass"] for row in rows)
    parser_documents = sum(row["graph"] is not None for row in rows)
    complete = (
        len(rows) == DOCUMENTS
        and parser_documents == quality_pass
    )
    joint = sum(row["joint_pass"] for row in rows)
    result = {
        "status": "complete" if complete else "incomplete",
        "allocated_documents": DOCUMENTS,
        "input_plans": sum(row["input_plans"] for row in rows),
        "prunable_plans": sum(row["prunable_plans"] for row in rows),
        "gate_pass_counts": {
            key: sum(row["gates"][key] for row in rows)
            for key in (
                "structure",
                "selection",
                "style",
                "facts",
                "atomicity",
            )
        },
        "quality_pass": quality_pass,
        "graph_pass": sum(
            bool(row["graph"] and row["graph"]["pass"]) for row in rows
        ),
        "joint_pass": joint,
        "documents": rows,
        "continuation_gate": bool(
            complete and joint >= MIN_JOINT_DOCUMENTS
        ),
        "decision": (
            "dependency_pruner_ready_for_independent_validation"
            if complete and joint >= MIN_JOINT_DOCUMENTS
            else "stop_dependency_pruner_route"
            if complete
            else "pending"
        ),
        "api_attempts": len(list((folder / "calls").glob("*.json"))),
        "neural_parser_sentences": sum(
            len(json.loads(path.read_text())["sentences"])
            for path in (folder / "parser").glob("*.json")
        ),
        "new_semantic_planning_calls": 0,
        "new_s2match_calls": 0,
        "new_watermark_injection_detection_or_attack_calls": 0,
        "manifest_sha256": sha(folder / "manifest.json"),
    }
    write_json(folder / "summary.json", result)
    if complete:
        write_report(folder, result)
    return result


def write_report(folder, result):
    lines = [
        "# SWAN dependency-path microfact pruner preflight",
        "",
        f"Quality-valid documents: **{result['quality_pass']}/{DOCUMENTS}**; "
        f"quality plus graph-scale pass: **{result['joint_pass']}/{DOCUMENTS}**.",
        "",
        "| Document | Input plans | Prunable | Selected | Words | Construction | "
        "Style | Facts | Atomic | Graph | Joint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["documents"]:
        graph = row["graph"]
        lines.append(
            f"| {row['prompt_id']} | {row['input_plans']} | "
            f"{row['prunable_plans']} | {row['selected_facts']} | "
            f"{row['words']} | "
            f"{'yes' if row['gates']['selection'] else 'no'} | "
            f"{'yes' if row['gates']['style'] else 'no'} | "
            f"{'yes' if row['gates']['facts'] else 'no'} | "
            f"{'yes' if row['gates']['atomicity'] else 'no'} | "
            f"{'yes' if graph and graph['pass'] else 'no'} | "
            f"{'yes' if row['joint_pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"- Frozen input plans: **{result['input_plans']}**; dependency-"
            f"prunable plans: **{result['prunable_plans']}**.",
            f"- Review API calls: **{result['api_attempts']}**; new semantic "
            f"planning calls: **0**; local AMR parser sentences: "
            f"**{result['neural_parser_sentences']}**; S2Match calls: **0**.",
            "- No watermark injection, detector evaluation, attack, TPR, FPR, "
            "or ASR was run.",
            f"- Decision: `{result['decision']}`.",
            "",
        ]
    )
    (folder / "report.md").write_text("\n".join(lines))


def runtime_path(folder):
    return folder / "runtime.json"


def run(folder, key):
    manifest = check(folder)
    with Heartbeat(folder) as heartbeat:
        generated_complete = all(
            generated_path(folder, source["prompt_id"]).exists()
            for source in manifest["sources"]
        )
        if not generated_complete:
            construct_all(folder, manifest, heartbeat)
        api = Journal(folder, key, heartbeat)
        probe = api.request(
            "probe",
            request_body("Reply exactly OK.", 0, 16),
        )
        if probe["finish_reason"] != "stop" or probe["content"].strip() != "OK":
            raise ValueError("Unexpected connectivity response")
        api.model_returned = probe["model"]
        runtime = freeze(
            runtime_path(folder),
            {
                "manifest_sha256": sha(folder / "manifest.json"),
                "model_returned": probe["model"],
                "probe_id": probe["id"],
            },
        )
        review_all(folder, manifest, api, runtime, heartbeat)
        parser_calls = parse_quality_machine(
            folder,
            {"machine_sources": manifest["sources"]},
            runtime,
            heartbeat,
        )
        heartbeat.update(
            stage="complete",
            new_api_calls=api.sent,
            new_semantic_planning_calls=0,
            new_neural_parser_calls=parser_calls,
            new_s2match_calls=0,
        )
    return summarize(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    configure_environment()
    with claim(args.output / ".runner.lock"):
        if args.stage == "prepare":
            manifest = prepare(args.output)
            result = {
                "status": "prepared",
                "sources": len(manifest["sources"]),
                "input_plans": manifest["protocol"]["input"][
                    "strictly_decoded_plans"
                ],
                "api_cap": manifest["protocol"]["api_cap"],
                "manifest_sha256": sha(args.output / "manifest.json"),
            }
        elif args.stage == "run":
            key = (
                read_credential(args.key_file)
                if args.key_file
                else os.environ.get("DEEPSEEK_API_KEY", "")
            )
            try:
                result = run(args.output, key)
            finally:
                if (args.output / "manifest.json").exists():
                    print(json.dumps(summarize(args.output), indent=2), flush=True)
        else:
            result = summarize(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
