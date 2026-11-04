"""Development-only normal-length writer with full-Bank acceptance.

All attempts and failures remain in the denominator. Frozen runtime copies of
this module are used; original cohorts, caches and ledgers are never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import run_swan_normal_completion1 as normal
import swan_final1_budget as budget
from swan_final1_common import freeze, load, sha256, single_writer
from semantic_fact_raw_amr import audit_raw, replay_signature
from run_swan_independent_calibration1 import Matcher
from run_swan_independent_banks1 import Normalizer

STAGE_CALL_CAP = 3200
STAGE_MONEY_CAP = 3_000_000_000
PARENT_CALL_CAP = 160000
PARENT_MONEY_CAP = 40_000_000_000
SLOTS = 16
ATTEMPTS = 50
THRESHOLD = .75


def proposal_prompt(context):
    """Explicitly a new proposal distribution, bounded to a simple semantic shape."""
    return (
        "Continue the general topic below with one new fictional but plausible detail. "
        "Write exactly one natural English sentence of 7 to 9 words ending with a period. "
        "Use this grammatical shape: The [collective noun] has been [transitive verb-ing] the [noun]. "
        "Use has been or have been to reach seven words; add NO adjectives or modifiers. "
        "Choose a relevant subject from team, group, company, firm, family, committee, organization. "
        "Choose the subject that fits this topic: a team for sport, a company for business, "
        "a family for personal matters. Do not default to a committee. "
        "Use a specific object naturally connected to the topic, preferably a tangible thing "
        "such as a ball, file, letter, house, book, or relevant document. "
        "Avoid nominalizations denoting something someone proposes or states; do not use "
        "proposal, statement, or decision as the object. "
        "Use a bare noun for the object, without possessives, adjective phrases or noun compounds. "
        "Describe one subject acting on one object. Avoid subordinate clauses, prepositional "
        "phrases, adverbs, quotations, names, dates, lists and explanations. "
        "Avoid occupational nouns such as coach, manager or archivist. "
        "The action and object must specifically fit the given topic. "
        "Do not repeat a previous sentence or fact. Do not output a generic administrative sentence. "
        "Examples: The team has been practicing the routine. "
        "The company has been printing the newsletter. "
        "Output only the sentence.\nTopic and previous sentences:\n" + context
    )


def sentence_valid(text, stop_reason, tokenize):
    return (stop_reason == "stop" and 7 <= len(normal.WORD.findall(text)) <= 12
            and tokenize(text) == [text] and text.endswith(".") and not text.endswith(".."))


def accept_scores(scores, valid):
    """Retain every Bank score; no data-dependent change to the Bank or cutoff."""
    import math
    if len(scores) != 50 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise budget.GlobalStop("Invalid complete Bank score vector")
    return bool(valid and max(scores) >= THRESHOLD)


class Journal(budget.Journal):
    def __init__(self, out, send):
        f = load(out / "FREEZE.json")
        self.inherited = f["parent_budget"]
        caps = f.get("caps", {})
        calls = min(STAGE_CALL_CAP, caps.get("new_calls", STAGE_CALL_CAP))
        money = min(STAGE_MONEY_CAP, caps.get("new_nanodollars", STAGE_MONEY_CAP))
        budget.CAPS = {"writer2": (calls, money)}
        super().__init__(out, "writer2", calls, send, sha256(out / "FREEZE.json"))
        if self.uncertain:
            raise budget.GlobalStop("Unresolved request; preserve reserve and never redispatch")

    def _parse(self, payload, reserve):
        if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
            raise budget.GlobalStop("Provider JSON must contain an object and usage object")
        return super()._parse(payload, reserve)

    def capacity(self, calls=1):
        local = super().capacity(calls)
        if local:
            return local
        totals = super().totals()
        if totals["calls"] + self.inherited["calls"] + calls > PARENT_CALL_CAP:
            return "cumulative_parent_call_cap"
        if (totals["exposure_nanodollars"] + self.inherited["exposure_nanodollars"]
                + calls * budget.MAX_RESERVE > PARENT_MONEY_CAP):
            return "cumulative_parent_money_cap"
        return None

    def accounting(self):
        local = super().totals()
        return {"new_version": local, "inherited": self.inherited,
                "cumulative_calls": local["calls"] + self.inherited["calls"],
                "cumulative_exposure_nanodollars": local["exposure_nanodollars"]
                + self.inherited["exposure_nanodollars"]}

    def request(self, call_id, user_text, system_text, max_tokens=50, temperature=.7, top_p=.9):
        if call_id in self.calls:
            call = self.calls[call_id]
            if call["terminal"] is None or call["terminal"]["kind"] != "complete":
                raise budget.GlobalStop("Incomplete request is non-replayable")
            if call["reserve"]["request"] != budget.body_for(
                    user_text, system_text, max_tokens, temperature, top_p):
                raise budget.GlobalStop("Completed request binding changed")
            value = self._parse(load(self.folder / "responses" / (call_id + ".json")),
                                call["reserve"]["reserve"])
            return {**value, "call_sha256": call["terminal"]["response_sha256"]}
        try:
            return super().request(call_id, user_text, system_text, max_tokens, temperature, top_p)
        except budget.DocumentFailure as exc:
            if str(exc) != "empty_model_output":
                raise
            return self.request(call_id, user_text, system_text, max_tokens, temperature, top_p)


def run(out, assets, bank_environment, send, verify):
    """One frozen four-document feasibility cohort; no automatic expansion."""
    with single_writer(out):
        f = verify()
        if (out / "HALT.json").exists():
            raise budget.GlobalStop("Frozen writer HALT is permanent")
        if (out / "TERMINAL.json").exists():
            terminal = load(out / "TERMINAL.json")
            for name in ("RESULT", "AUDIT"):
                if terminal[name.lower() + "_sha256"] != sha256(out / (name + ".json")):
                    raise budget.GlobalStop("Writer terminal changed")
            for name, expected in load(out / "AUDIT.json")["files"].items():
                if sha256(out / name) != expected:
                    raise budget.GlobalStop("Writer output changed")
            return
        journal = Journal(out, send)
        sources = load(out / "private/SOURCES.json")
        bank = [r["normalized_template_amr"] for r in load(out / "private/amr_bank_50.json")]
        tokenize = normal.tokenizer()
        swan = normal.pilot.swan
        prompts = swan.official_function_module_for_test({})
        probe = f["stage"] == "proposal_probe"
        slot_count, attempt_count = (2, 1) if probe else (SLOTS, ATTEMPTS)
        parser = normal.rt.Parser(assets, out, f)
        normalizer = Normalizer(bank_environment)
        matcher = Matcher(out, assets)
        probes = ("The scientist carefully examined the new sample.",
                  "The young student quietly opened the heavy door.")
        checks = []
        for text in probes:
            pair = [{**parser.fresh(text), "case_id": normal.rt.digest(text)} for _ in range(2)]
            if replay_signature(pair[0]) != replay_signature(pair[1]):
                raise budget.GlobalStop("Parser replay differs")
            if not audit_raw(pair[0], {"case_id": normal.rt.digest(text),
                                     "input_text_sha256": normal.rt.digest(text)},
                             sha256(out / "FREEZE.json"))["structurally_eligible"]:
                raise budget.GlobalStop("Parser precheck excluded")
            checks.append({"text_sha256": normal.rt.digest(text), "double_replay_equal": True})
        self_scores = [matcher.fresh(bank[0], bank[0]) for _ in range(2)]
        if self_scores != [1., 1.]:
            raise budget.GlobalStop("Matcher replay differs")
        freeze(out / "GPU_PRECHECK.json", {"dtype": "float32", "probes": checks,
                                          "self_match_scores": self_scores})
        documents = []
        for source in sources:
            source_id = source["allocation_id"]
            docpath = out / "private/documents" / (source_id + ".json")
            sentences, accepted_events, failure = [], [], None
            for slot in range(slot_count):
                # Bank entry 21 is only the documented proposal shape; acceptance
                # still scores all 50 entries, without selecting a new Bank.
                target = bank[21]
                context = source["generation_prompt"] + (" " + " ".join(sentences) if not probe else "")
                for attempt in range(attempt_count):
                    cid = source_id + f"-writer2-slot-{slot:02d}-attempt-{attempt:02d}"
                    prompt = proposal_prompt(context)
                    response = journal.request(cid, prompt, prompts.build_system_prompt())
                    text = response["text"]
                    path = out / "private/attempts" / (cid + ".json")
                    binding = {"call_id": cid, "response_sha256": response["call_sha256"],
                               "text_sha256": normal.rt.digest(text), "target": target,
                               "freeze_sha256": sha256(out / "FREEZE.json")}
                    if path.exists():
                        event = load(path)
                        if any(event[k] != v for k, v in binding.items()):
                            raise budget.GlobalStop("Attempt replay binding changed")
                        raw_path = event.get("raw_path")
                        if raw_path and sha256(out / raw_path) != event["raw_sha256"]:
                            raise budget.GlobalStop("Attempt raw graph changed")
                    else:
                        event = dict(binding, text=text, accepted=False, rejection=None,
                                     bank_scores=None, template=None)
                        if not probe and text in sentences:
                            event["rejection"] = "duplicate_sentence"
                        elif not sentence_valid(text, response["stop_reason"], tokenize):
                            event["rejection"] = "sentence_gate"
                        else:
                            raw = {**parser.fresh(text), "case_id": normal.rt.digest(text)}
                            rawpath = out / "private/raw_graphs" / (cid + ".json")
                            freeze(rawpath, raw)
                            event.update(raw_path=str(rawpath.relative_to(out)),
                                         raw_sha256=sha256(rawpath))
                            check = audit_raw(raw, {"case_id": normal.rt.digest(text),
                                                   "input_text_sha256": normal.rt.digest(text)},
                                              sha256(out / "FREEZE.json"))
                            if not check["structurally_eligible"]:
                                event["rejection"] = "raw_parser_excluded"
                            else:
                                normalized = normalizer(raw["amr"])
                                if not normalized["eligible"]:
                                    event["rejection"] = "normalizer_excluded"
                                else:
                                    scores = [matcher.pair(normalized["template"], t) for t in bank]
                                    event.update(template=normalized["template"], bank_scores=scores,
                                                 selected_target_score=scores[bank.index(target)],
                                                 accepted=accept_scores(scores, True))
                                    if not event["accepted"]:
                                        event["rejection"] = "no_bank_match"
                        freeze(path, event)
                    normal.rt.atomic(out / "HEARTBEAT.json", {
                        "phase": "writer_feasibility", "documents_complete": len(documents),
                        "total_documents": len(sources), "slot": slot, "attempt": attempt,
                        "accepted_current_document": len(sentences), "budget": journal.accounting(),
                        "at": normal.rt.now()})
                    if event["accepted"]:
                        sentences.append(text)
                        accepted_events.append({"path": str(path.relative_to(out)), "sha256": sha256(path)})
                        break
                else:
                    failure = "slot_exhausted"
                    if not probe:
                        break
            text = " ".join(sentences)
            eligible = (not probe and failure is None and len(sentences) == SLOTS and tokenize(text) == sentences
                        and 100 <= len(normal.WORD.findall(text)) <= 220)
            readback = []
            if eligible:
                for i, sentence in enumerate(sentences):
                    raw = {**parser.fresh(sentence), "case_id": normal.rt.digest(sentence)}
                    rawpath = out / "private/readback" / (source_id + f"-{i:02d}.json")
                    # Exclude elapsed time from the immutable replay file.
                    raw["seconds"] = None
                    freeze(rawpath, raw)
                    check = audit_raw(raw, {"case_id": normal.rt.digest(sentence),
                                            "input_text_sha256": normal.rt.digest(sentence)},
                                      sha256(out / "FREEZE.json"))
                    if not check["structurally_eligible"]:
                        raise budget.GlobalStop("Fresh complete-document readback excluded")
                    normalized = normalizer(raw["amr"])
                    if not normalized["eligible"]:
                        raise budget.GlobalStop("Fresh readback normalization excluded")
                    scores = [matcher.fresh(normalized["template"], t) for t in bank]
                    written = load(out / accepted_events[i]["path"])
                    if (not accept_scores(scores, True) or normalized["template"] != written["template"]
                            or max(abs(a - b) for a, b in zip(scores, written["bank_scores"])) > 1e-12):
                        raise budget.GlobalStop("Writer and independent fresh readback differ")
                    readback.append({"path": str(rawpath.relative_to(out)), "sha256": sha256(rawpath),
                                     "max_bank_score": max(scores), "scores_replayed": True})
            record = {"source_id": source_id, "text": text, "sentences": sentences,
                      "accepted_events": accepted_events, "failure": failure, "eligible": eligible,
                      "fresh_readback": readback,
                      "words": len(normal.WORD.findall(text)), "freeze_sha256": sha256(out / "FREEZE.json")}
            freeze(docpath, record)
            documents.append(record)
        matcher.db.close()
        verify()
        result = {"status": "PROPOSAL_PROBE_COMPLETE" if probe else "FEASIBILITY_COMPLETE",
                  "stage": f["stage"], "documents": len(documents),
                  "eligible": sum(d["eligible"] for d in documents),
                  "accepted_sentences": sum(len(d["sentences"]) for d in documents),
                  "fresh_readback_sentences": sum(len(d["fresh_readback"]) for d in documents),
                  "budget": journal.accounting(), "confirmation_sources_used": 0,
                  "old_0_of_200_replaced": False, "formal_TPR": None, "formal_FPR": None,
                  "model_semantic_review_pending": True, "automatic_scale_up": False}
        freeze(out / "RESULT.json", result)
        files = [p for sub in ("attempts", "raw_graphs", "readback", "documents", "writer2")
                 for p in (out / "private" / sub).rglob("*") if p.is_file()]
        files += [out / "RESULT.json", out / "GPU_PRECHECK.json"]
        freeze(out / "AUDIT.json", {"freeze_sha256": sha256(out / "FREEZE.json"),
                                    "files": {str(p.relative_to(out)): sha256(p) for p in sorted(files)}})
        freeze(out / "TERMINAL.json", {"status": result["status"],
                                       "result_sha256": sha256(out / "RESULT.json"),
                                       "audit_sha256": sha256(out / "AUDIT.json")})
