"""CUDA-only cycle2 pilot, sealed blind predictions, and post-seal evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import warnings

ROOT = Path("/workspace/swan-elastic-reconstruct")
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def inventory(directory):
    return [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(directory.rglob("*")) if p.is_file()]


def assets(out):
    from huggingface_hub import snapshot_download
    import importlib.metadata as md
    import spacy
    private = out / "private"
    private.mkdir(parents=True, exist_ok=True)
    encoder = private / "minilm"
    snapshot_download("sentence-transformers/all-MiniLM-L6-v2", revision=REVISION,
                      local_dir=str(encoder), allow_patterns=[
                          "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json",
                          "1_Pooling/config.json", "modules.json"])
    # Snapshot download metadata is neither inference input nor redistributed.
    import shutil
    shutil.rmtree(encoder / ".cache", ignore_errors=True)
    if not (encoder / "model.safetensors").exists():
        raise RuntimeError("Pinned encoder weights missing")
    spacy.load("en_core_web_sm").to_disk(private / "spacy_model")
    runtime_files = ROOT / "third_party/swan_s2match_runtime_711a231d"
    manifest = {"minilm_revision": REVISION, "minilm": inventory(encoder),
                "spacy": inventory(private / "spacy_model"),
                "glove_sha256": sha(ROOT / "vectors/glove.6B.100d.txt"),
                "packages": {k: md.version(k) for k in
                             ["torch", "amrlib", "transformers", "spacy", "penman", "en-core-web-sm"]}}
    atomic(out / "ASSET_MANIFEST.json", manifest)


def verify(out):
    frozen = read(out / "FREEZE.json")
    for rel, expected in frozen["sources"].items():
        if sha(out / "private/code" / rel) != expected:
            raise RuntimeError("Frozen source SHA mismatch")
    for directory, field in [("minilm", "minilm"), ("spacy_model", "spacy")]:
        for item in frozen["assets"][field]:
            if sha(out / "private" / directory / item["path"]) != item["sha256"]:
                raise RuntimeError("Model SHA mismatch")
    for file in ["BANK.json", "propbank_verbs.json"]:
        if sha(out / "private" / file) != frozen["private_sha256"][file]:
            raise RuntimeError("Frozen dictionary SHA mismatch")
    if sha(ROOT / "vectors/glove.6B.100d.txt") != frozen["assets"]["glove_sha256"]:
        raise RuntimeError("GloVe SHA mismatch")
    if frozen.get("completion_only"):
        origin = frozen["prediction_origin"]
        if sha(out / "PREDICTION_SEAL.json") != frozen["inherited_seal_sha256"]:
            raise RuntimeError("Inherited prediction seal SHA mismatch")
        if sha(out / "private/PROTOTYPES.json") != origin["prototypes_sha256"]:
            raise RuntimeError("Inherited prototype SHA mismatch")
        if sha(out / "SYMBOL_MAP.json") != frozen["symbol_map_sha256"]:
            raise RuntimeError("Inherited symbol map SHA mismatch")
        for item in origin["documents"]:
            if sha(out / "private/documents" / item["name"]) != item["sha256"]:
                raise RuntimeError("Inherited document SHA mismatch")
        for key, expected in read(out / "PREDICTION_SEAL.json")["sha256"].items():
            path = out / "private/predictions" / (key + ".json")
            if sha(path) != expected:
                raise RuntimeError("Inherited blind prediction SHA mismatch")
            if read(path)["freeze_sha256"] != origin["source_freeze_sha256"]:
                raise RuntimeError("Inherited prediction source identity mismatch")
    return frozen


def runtime(out, frozen):
    import torch
    if sys.platform != "linux" or not torch.cuda.is_available():
        raise RuntimeError("Linux NVIDIA CUDA is required")
    code = out / "private/code"
    sys.path.insert(0, str(code / "src/core"))
    sys.path.insert(0, str(code / "scripts"))
    from run_multiscale_transfer import Parser as BaseParser, Scorer as BaseScorer
    import spacy
    torch.manual_seed(frozen["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    class Parser(BaseParser):
        def fresh(self, text):
            n = len(self.model.tokenizer(text, truncation=False)["input_ids"])
            if not text or n >= self.model.max_sent_len:
                return {"valid": False, "amr": None, "amr_sha256": None,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "freeze_sha256": self.freeze_sha, "native_limit_or_empty": True}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                record = super().fresh(text)
            if caught:
                # Python warnings have no deserializer span identity.
                atomic(out / "private/PYTHON_WARNINGS.json", [
                    {"category": w.category.__name__, "message": str(w.message)} for w in caught])
                raise RuntimeError("Unbound Python inference warning")
            accepted = [a for a in record["attempts"] if a["accepted"]]
            selected = accepted[0]["beam_index"] if len(accepted) == 1 else None
            record["valid"] = bool(record["valid"] and selected == 0 and
                                   not record["events"] and not record["decode_events"])
            return record

    class Scorer(BaseScorer):
        def scores(self, amr, kind="official"):
            if amr is None:
                return [None] * len(self.bank)
            result = super().scores(amr, kind)
            return result if len(result) == len(self.bank) else [None] * len(self.bank)

    # Recognized library deprecation during model loading is recorded, not scored.
    with warnings.catch_warnings(record=True) as load_warnings:
        warnings.simplefilter("always")
        parser = Parser(ROOT, out, frozen)
    unexpected = [w for w in load_warnings if not (
        w.category is FutureWarning and "torch.load" in str(w.message) and
        "weights_only=False" in str(w.message))]
    atomic(out / "private/LOAD_LIBRARY_WARNINGS.json", [
        {"category": w.category.__name__, "message": str(w.message)} for w in load_warnings])
    if unexpected:
        raise RuntimeError("Unexpected unbound load warning")
    scorer = Scorer(ROOT, out)
    return parser, scorer, spacy.load(out / "private/spacy_model")


class Encoder:
    def __init__(self, out):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        path = out / "private/minilm"
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).float().to("cuda:0").eval()
        self.limit = min(self.model.config.max_position_embeddings,
                         read(path / "sentence_bert_config.json")["max_seq_length"])
        self.cache = {}
        if not all(p.dtype == torch.float32 for p in self.model.parameters()):
            raise RuntimeError("Embedding FP32 required")

    def fresh(self, text):
        torch = self.torch
        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")
        if not text or tokens["input_ids"].shape[1] > self.limit:
            return None
        with torch.inference_mode():
            tokens = {k: v.to("cuda:0") for k, v in tokens.items()}
            output = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            vector = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
        return vector[0].cpu().numpy().astype("float64")

    def encode(self, text):
        if text not in self.cache:
            self.cache[text] = self.fresh(text)
        return self.cache[text]


def memory():
    import torch
    free, total = torch.cuda.mem_get_info()
    ratio = (total - free) / total
    if ratio >= .8:
        raise RuntimeError("Global GPU VRAM gate")
    return ratio


def progress(out, phase, **values):
    atomic(out / "heartbeat.json", {"phase": phase, "at": datetime.now(timezone.utc).isoformat(), **values})


def pilot(out, frozen):
    import numpy as np
    import penman
    from penman.models.amr import model as AMR_MODEL
    sys.path.insert(0, str(out / "private/code/src/core"))
    from semantic_cycle2 import event_view, main_verb
    started = time.monotonic()
    parser, scorer, nlp = runtime(out, frozen)
    encoder = Encoder(out)
    assignments = read(out.parent / "pilot/ASSIGNMENTS.json")
    replay, rows, peak = [], [], memory()
    for i in [0, 1, 2]:
        doc = read(out / "private/documents" / f"d{i:02d}.json")
        for slot in assignments[i]["slots"]:
            s = slot["slot"]
            text = doc["arms"]["positive"]["slots"][s]
            first = parser.fresh(text)
            if i == 0 and s < 3:
                second = parser.fresh(text)
                same = (first["amr_sha256"] == second["amr_sha256"] and
                        first["valid"] == second["valid"] and
                        first.get("events") == second.get("events"))
                def fresh_scores(graph):
                    if graph is None:
                        return [None] * len(scorer.bank)
                    try:
                        normalized = scorer.normalize(graph, "official")
                    except Exception:
                        return [None] * len(scorer.bank)
                    return [scorer.pair(normalized, t) for t in scorer.bank]
                scores = [fresh_scores(r["amr"] if r["valid"] else None) for r in [first, second]]
                def delta(a, b):
                    if a is None or b is None:
                        return 0.0 if a is None and b is None else math.inf
                    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                score_delta = max(delta(a, b) for a, b in zip(*scores))
                verb = main_verb(text, nlp)
                graphs = [event_view(r["amr"], verb["lemma"]) if r["valid"] and verb else None
                          for r in [first, second]]
                event_scores = [fresh_scores(g) for g in graphs]
                event_delta = max(delta(a, b) for a, b in zip(*event_scores))
                emb_delta = delta(encoder.fresh(text), encoder.fresh(text))
                replay.append({"carrier": s, "parser_exact": same, "score_max_error": score_delta,
                               "event_score_max_error": event_delta, "embedding_max_error": emb_delta})
                if not same or max(score_delta, event_delta, emb_delta) > 1e-6:
                    raise RuntimeError("Numerical replay gate")
            observed = {"unknown_nodes": 0, "name_sizes": [], "imperative": False, "polite": False}
            if first["valid"]:
                graph = penman.decode(first["amr"], model=AMR_MODEL)
                concepts = {a: c for a, b, c in graph.triples if b == ":instance"}
                names = {c for a, b, c in graph.triples if b == ":name"}
                observed = {
                    "unknown_nodes": sum(c == "amr-unknown" for c in concepts.values()),
                    "name_sizes": sorted(sum(a == n and bool(re.fullmatch(r":op\d+", b))
                                             for a, b, c in graph.triples) for n in names),
                    "imperative": any(b == ":mode" and c == "imperative" for a, b, c in graph.triples),
                    "polite": any(b == ":polite" and c == "+" for a, b, c in graph.triples)}
            required = slot["markers"]
            passed = (first["valid"] and bool(text) and
                      observed["unknown_nodes"] >= required["unknown_nodes"] and
                      all(observed["name_sizes"].count(x) >= required["name_sizes"].count(x)
                          for x in required["name_sizes"]) and
                      (not required["imperative"] or observed["imperative"]) and
                      (not required["polite"] or observed["polite"]))
            record_path = out / "private/parses" / (hashlib.sha256(text.encode()).hexdigest() + ".json")
            atomic(record_path, first)
            rows.append({"doc_id": i, "slot": s, "strict_valid": first["valid"],
                         "markers_pass": bool(passed), "required": required, "observed": observed})
            peak = max(peak, memory())
            progress(out, "pilot", completed=len(rows), total=24, elapsed_seconds=time.monotonic() - started)
    atomic(out / "CUDA_PRECHECK.json", {"replay": replay, "all_errors_le1e6": True,
                                       "peak_global_vram_ratio": peak, "precision": "FP32",
                                       "assets_sha_verified": True})
    successes = sum(r["markers_pass"] for r in rows)
    atomic(out / "G1E.json", {"successes": successes, "denominator": 24, "minimum": 20,
                              "Go": successes >= 20, "rows": rows,
                              "elapsed_seconds": time.monotonic() - started,
                              "formal_human_G1": "NOT_EVALUATED"})


def dataset(out, nlp):
    from semantic_sequence import shared_candidates
    from semantic_cycle2 import constructed_text
    rows, sources = [], []
    for path in sorted((out / "private/documents").glob("d*.json")):
        doc = read(path)
        if doc["split"] == "development":
            for s, text in enumerate(doc["arms"]["positive"]["slots"]):
                sources.append({"doc_id": doc["doc_id"], "slot": s, "text": text})
        for arm in ["positive", "unwatermarked"]:
            conditions = ["clean", "paraphrase", "split", "merge"] if arm == "positive" else ["clean", "paraphrase"]
            for condition in conditions:
                data = doc["arms"][arm]
                text = data["paraphrase"] if condition == "paraphrase" else constructed_text(data["slots"], condition)[0]
                rows.append({"doc_id": doc["doc_id"], "split": doc["split"], "arm": arm,
                             "condition": condition, "text": text, "candidates": shared_candidates(text, nlp)})
    return rows, sources


def raw_windows(observation, parser, scorer, encoder, prototypes):
    from semantic_cycle2 import event_view
    import numpy as np
    values = {"W": [], "E1": [], "C": []}
    for window in observation["windows"]:
        text = window["text"]
        amr = parser.parse(text)
        values["W"].append(scorer.scores(amr))
        subgraph = event_view(amr, observation["lemma"]) if amr else None
        values["E1"].append(scorer.scores(subgraph))
        vector = encoder.encode(text)
        values["C"].append((np.asarray(prototypes) @ vector).tolist() if vector is not None else None)
    return values


def blind(out, frozen):
    import numpy as np
    started = time.monotonic()
    parser, scorer, nlp = runtime(out, frozen)
    encoder = Encoder(out)
    rows, sources = dataset(out, nlp)
    mapping = read(out / "SYMBOL_MAP.json")
    # Labels derive from the predeclared public cyclic scheduling function only
    # for development prototype fitting; no targets enter candidate predictions.
    def scheduled_symbol(doc_id, slot):
        import random
        ids = []
        for symbol in "ABCD":
            members = sorted(mapping["symbols"][symbol]["templates"])
            ids.extend([members[2 * (doc_id % 12) % 6], members[(2 * (doc_id % 12) + 1) % 6]])
        random.Random(frozen["seed"] + 1000 + doc_id).shuffle(ids)
        return "ABCD".index(mapping["template_to_symbol"][str(ids[slot])])
    proto_rows = [[] for _ in range(4)]
    for row in sources:
        vector = encoder.encode(row["text"])
        if vector is None:
            raise RuntimeError("Unscreened prototype missing; no class/sample selection")
        proto_rows[scheduled_symbol(row["doc_id"], row["slot"])].append(vector)
    prototypes = []
    for group in proto_rows:
        value = np.mean(group, axis=0)
        prototypes.append((value / np.linalg.norm(value)).tolist())
    atomic(out / "private/PROTOTYPES.json", prototypes)
    windows = sorted({w["text"] for r in rows for o in r["candidates"]["observations"] for w in o["windows"]})
    occurrences = sum(len(o["windows"]) for r in rows for o in r["candidates"]["observations"])
    if not windows:
        raise RuntimeError("No calibration workload")
    measured = []
    for text in windows[:min(8, len(windows))]:
        before = time.monotonic()
        obs = {"lemma": next((t.lemma_ for t in nlp(text) if t.pos_ == "VERB"), ""),
               "windows": [{"text": text}, {"text": text}]}
        raw_windows(obs, parser, scorer, encoder, prototypes)
        measured.append(time.monotonic() - before)
        memory()
    count_docs = len(list((out / "private/documents").glob("d*.json")))
    # Reserve every planned known-position window and full-sentence diagnostic,
    # regardless of later reference coverage. No condition is dropped for budget.
    reserve = count_docs * 8 * 2 * 2 + count_docs * 8
    seconds_per = max(np.mean(measured), np.percentile(measured, 75))
    previous = read(out / "G1E.json")["elapsed_seconds"]
    projected = (occurrences + reserve) * seconds_per * frozen["safety_factor"] + previous
    resource = {"actual_unique_blind_windows": len(windows), "reserved_known_and_sentence_windows": reserve,
                "blind_window_occurrences": occurrences,
                "measured_seconds": measured, "projected_gpu_seconds": projected,
                "gpu_cap_seconds": frozen["gpu_budget_seconds"], "Go": bool(projected <= frozen["gpu_budget_seconds"]),
                "all_conditions_retained": True, "safety_factor": frozen["safety_factor"]}
    atomic(out / "RESOURCE_PRECHECK.json", resource)
    if not resource["Go"]:
        raise RuntimeError("Frozen resource projection gate")
    completed, seal = 0, {}
    for row in rows:
        key = f"d{row['doc_id']:02d}-{row['arm']}-{row['condition']}"
        path = out / "private/predictions" / f"{key}.json"
        if not path.exists():
            observations = []
            for obs in row["candidates"]["observations"]:
                observations.append({**obs, "raw_scores": raw_windows(obs, parser, scorer, encoder, prototypes)})
                memory()
                if time.monotonic() - started + previous > frozen["gpu_budget_seconds"]:
                    raise RuntimeError("Actual GPU time gate")
            atomic(path, {**row, "observations": observations, "freeze_sha256": sha(out / "FREEZE.json")})
        seal[key] = sha(path)
        completed += 1
        progress(out, "blind_predictions", completed=completed, total=len(rows),
                 elapsed_seconds=time.monotonic() - started)
    atomic(out / "PREDICTION_SEAL.json", {"sha256": seal, "versions": len(rows),
                                         "reference_labels_loaded": False,
                                         "elapsed_seconds": time.monotonic() - started})


def references(out, row, nlp, assignments):
    from semantic_cycle2 import constructed_text, main_verb
    doc = read(out / "private/documents" / f"d{row['doc_id']:02d}.json")
    result = [{"slot": s["slot"], "symbol": "ABCD".index(s["symbol"]), "span": None}
              for s in assignments[row["doc_id"]]["slots"]]
    if row["arm"] != "positive":
        return []
    if row["condition"] != "paraphrase":
        text, spans = constructed_text(doc["arms"]["positive"]["slots"], row["condition"])
        for item, source, (a, b) in zip(result, doc["arms"]["positive"]["slots"], spans):
            source = source.strip()
            anchor = main_verb(source, nlp)
            if anchor:
                # Re-use source token occurrence, never choose a new root after editing.
                before = list(re.finditer(r"\S+", source))
                after = list(re.finditer(r"\S+", text[a:b]))
                index = next(j for j, t in enumerate(before)
                             if t.start() <= anchor["span"][0] < t.end())
                if len(before) == len(after):
                    offset = anchor["span"][0] - before[index].start()
                    start = a + after[index].start() + offset
                    item.update(span=[start, start + anchor["span"][1] - anchor["span"][0]],
                                lemma=anchor["lemma"])
        return result
    labels = read(out / "private/references" / f"d{row['doc_id']:02d}.json").get("events", [])
    text = row["text"]
    for item in result:
        candidates = [x for x in labels if x.get("slot") == item["slot"]]
        if len(candidates) != 1:
            continue
        label = candidates[0]
        phrase, predicate = label.get("phrase"), label.get("predicate")
        if label.get("status") != "located" or not isinstance(phrase, str) or not phrase:
            continue
        if text.count(phrase) != 1 or not isinstance(predicate, str) or not predicate:
            continue
        tokens = [t for t in nlp(phrase) if t.pos_ == "VERB" and t.text == predicate]
        if len(tokens) != 1:
            continue
        token, base = tokens[0], text.index(phrase)
        item.update(span=[base + token.idx, base + token.idx + len(token.text)], lemma=token.lemma_)
    return result


def evaluate(out, frozen):
    import numpy as np
    sys.path.insert(0, str(out / "private/code/src/core"))
    from semantic_sequence import centered_window
    from semantic_cycle2 import weights, identity, diagnostic, match_positions, main_verb
    started = time.monotonic()
    seal = read(out / "PREDICTION_SEAL.json")
    for key, expected in seal["sha256"].items():
        if sha(out / "private/predictions" / f"{key}.json") != expected:
            raise RuntimeError("Prediction seal mismatch")
    parser, scorer, nlp = runtime(out, frozen)
    encoder = Encoder(out)
    prototypes = read(out / "private/PROTOTYPES.json")
    assignments = read(out / "private/ASSIGNMENTS.json")
    bank_ids = sorted({s["template_id"] for d in assignments for s in d["slots"]})
    mapping = read(out / "SYMBOL_MAP.json")
    groups = [[j for j, t in enumerate(bank_ids) if mapping["template_to_symbol"][str(t)] == c] for c in "ABCD"]
    rows = [read(out / "private/predictions" / f"{key}.json") for key in sorted(seal["sha256"])]
    for row in rows:
        row["refs"] = references(out, row, nlp, assignments)
    selected = {}
    for representation in ["W", "E1", "C"]:
        dev = [r for r in rows if r["split"] == "development" and r["arm"] == "positive"]
        # All four development conditions enter temperature selection; missing=uniform.
        choices = []
        for temperature in frozen["temperatures"]:
            losses = []
            for row in dev:
                pairs = dict((r, p) for p, r in match_positions(row["observations"], row["refs"]))
                for j, ref in enumerate(row["refs"]):
                    p = pairs.get(j)
                    q = weights(row["observations"][p]["raw_scores"][representation], representation,
                                groups, temperature) if p is not None else None
                    losses.append(-math.log(max(q[ref["symbol"]] if q else .25, 1e-9)))
            choices.append((float(np.mean(losses)), temperature))
        temperature = min(choices)[1]
        threshold_choices = []
        for threshold in frozen["retain_thresholds"]:
            values = []
            for row in dev:
                if row["condition"] != "paraphrase":
                    continue
                preds = [identity(weights(o["raw_scores"][representation], representation, groups, temperature),
                                  threshold) for o in row["observations"]]
                values.append(diagnostic(row["observations"], row["refs"], preds)["identity_f1"])
            threshold_choices.append((-float(np.mean(values)), threshold))
        threshold = min(threshold_choices)[1]
        selected[representation] = {"temperature": temperature, "threshold": threshold}
    def row_result(row, rep):
        params = selected[rep]
        preds = [identity(weights(o["raw_scores"][rep], rep, groups, params["temperature"]),
                          params["threshold"]) for o in row["observations"]]
        return diagnostic(row["observations"], row["refs"], preds)
    ranking = []
    for rep in ["C", "W", "E1"]:
        means = []
        for condition in ["paraphrase", "clean"]:
            values = [row_result(r, rep)["identity_f1"] for r in rows
                      if r["split"] == "development" and r["arm"] == "positive" and r["condition"] == condition]
            means.append(float(np.mean(values)))
        ranking.append((-means[0], -means[1], ["C", "W", "E1"].index(rep), rep))
    selected["R_star"] = min(ranking)[-1]
    atomic(out / "SELECTION.json", {"selection": selected, "readout_used_for_selection": False})
    output = []
    prior_seconds = read(out / "G1E.json")["elapsed_seconds"] + seal["elapsed_seconds"]
    for row in rows:
        result = {k: row[k] for k in ["doc_id", "split", "arm", "condition"]}
        result["candidate_status"] = row["candidates"]["status"]
        result["surface"] = {"words": len(row["text"].split()), "characters": len(row["text"]),
                             "question_marks": row["text"].count("?"),
                             "inert_paraphrase": (row["condition"] == "paraphrase" and row["text"] ==
                                 read(out / "private/documents" / f"d{row['doc_id']:02d}.json")["arms"][row["arm"]]["text"])}
        result["observed_anchor_count"] = row["candidates"]["observed_anchor_count"]
        result["diagnostics"] = {rep: row_result(row, rep) for rep in ["W", "E1", "C"]}
        if row["arm"] == "positive" and row["condition"] in ["clean", "paraphrase"]:
            known = {rep: [] for rep in ["W", "E1", "C"]}
            tokens = list(re.finditer(r"\S+", row["text"]))
            for ref in row["refs"]:
                raw = None
                if ref.get("span"):
                    obs = {"lemma": ref["lemma"], "windows": [centered_window(
                        row["text"], tokens, ref["span"][0], width) for width in [6, 12]]}
                    raw = raw_windows(obs, parser, scorer, encoder, prototypes)
                for rep in known:
                    q = weights(raw[rep], rep, groups, selected[rep]["temperature"]) if raw else None
                    known[rep].append({"true": ref["symbol"], "predicted": identity(q, selected[rep]["threshold"])})
                memory()
            result["known_position"] = known
        if row["arm"] == "positive" and row["condition"] == "clean":
            full = []
            doc = read(out / "private/documents" / f"d{row['doc_id']:02d}.json")
            for s, text in enumerate(doc["arms"]["positive"]["slots"]):
                score = scorer.scores(parser.parse(text))
                from semantic_sequence import symbol_weights
                q = symbol_weights(score, groups, selected["W"]["temperature"])
                full.append({"true": row["refs"][s]["symbol"],
                             "predicted": identity(q.tolist() if q is not None else None, selected["W"]["threshold"])})
            result["whole_sentence_W"] = full
        output.append(result)
        progress(out, "post_seal_evaluation", completed=len(output), total=len(rows))
        if time.monotonic() - started + prior_seconds > frozen["gpu_budget_seconds"]:
            raise RuntimeError("Actual cumulative GPU budget")
    summaries = []
    for split in ["development", "readout"]:
        for condition in ["clean", "paraphrase", "split", "merge"]:
            subset = [r for r in output if r["split"] == split and r["arm"] == "positive" and r["condition"] == condition]
            if not subset:
                continue
            for rep in ["W", "E1", "C"]:
                ds = [r["diagnostics"][rep] for r in subset]
                item = {"split": split, "condition": condition, "representation": rep, "pairs": len(subset),
                        "mean_identity_f1": float(np.mean([d["identity_f1"] for d in ds])),
                        "mean_localization_f1": float(np.mean([d["localization_f1"] for d in ds])),
                        "mean_identity_precision": float(np.mean([d["identity_precision"] for d in ds])),
                        "mean_identity_recall": float(np.mean([d["identity_recall"] for d in ds])),
                        "erasures": sum(d["erasures"] for d in ds),
                        "planned": sum(d["planned"] for d in ds),
                        "observations": sum(d["observations"] for d in ds),
                        "per_class": [{"planned": sum(d["per_class"][c]["planned"] for d in ds),
                                       "correct": sum(d["per_class"][c]["correct"] for d in ds)} for c in range(4)],
                        "reference_coverage": sum(d["reference_coverage"] for d in ds) / sum(d["planned"] for d in ds)}
                known = [x for r in subset for x in r.get("known_position", {}).get(rep, [])]
                if known:
                    recalls = [sum(x["predicted"] == c for x in known if x["true"] == c) /
                               sum(x["true"] == c for x in known) for c in range(4)]
                    item["known_position_BA"] = float(np.mean(recalls))
                    item["known_position_confusion_including_erasure"] = [
                        [sum(x["true"] == c and x["predicted"] == predicted for x in known)
                         for predicted in [0, 1, 2, 3, None]] for c in range(4)]
                if split == "readout":
                    rng = np.random.default_rng(frozen["seed"])
                    indices = rng.integers(0, len(subset), (5000, len(subset)))
                    for metric in ["identity_f1", "localization_f1"]:
                        values = np.asarray([d[metric] for d in ds])
                        item[metric + "_CI95"] = np.quantile(values[indices].mean(1), [.025, .975]).tolist()
                    if known:
                        doc_ba = []
                        for r in subset:
                            values = r["known_position"][rep]
                            doc_ba.append(np.mean([sum(x["predicted"] == c for x in values if x["true"] == c) /
                                sum(x["true"] == c for x in values) for c in range(4)]))
                        item["known_position_BA_CI95"] = np.quantile(
                            np.asarray(doc_ba)[indices].mean(1), [.025, .975]).tolist()
                summaries.append(item)
    contrasts = []
    test = [r for r in output if r["split"] == "readout" and r["arm"] == "positive" and r["condition"] == "paraphrase"]
    if test:
        rng = np.random.default_rng(frozen["seed"])
        for alternative in ["E1", "C"]:
            differences = np.asarray([r["diagnostics"][alternative]["identity_f1"] -
                                      r["diagnostics"]["W"]["identity_f1"] for r in test])
            boot = differences[rng.integers(0, len(test), (5000, len(test)))].mean(1)
            contrasts.append({"contrast": alternative + "-W", "condition": "paraphrase",
                              "unit": "prompt_pair", "n": len(test), "mean": float(differences.mean()),
                              "CI95": np.quantile(boot, [.025, .975]).tolist()})
    atomic(out / "RESULT.json", {"scope": "constructed/model-assisted exploratory diagnostics",
                                 "formal_G1_G2_G3": "NOT_EVALUATED", "G4": "NOT_RUN",
                                 "summaries": summaries, "paired_contrasts": contrasts, "rows": output,
                                 "elapsed_seconds": time.monotonic() - started,
                                 "evaluation_freeze_sha256": sha(out / "FREEZE.json"),
                                 "prediction_source_freeze_sha256": frozen.get("prediction_origin", {}).get(
                                     "source_freeze_sha256", sha(out / "FREEZE.json")),
                                 "prediction_seal_sha256": sha(out / "PREDICTION_SEAL.json")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["assets", "pilot", "blind", "evaluate"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    os.umask(0o077)
    try:
        if args.action == "assets":
            assets(args.out)
        else:
            if (args.out / "HALT.json").exists():
                raise RuntimeError("Permanent halt")
            frozen = verify(args.out)
            {"pilot": pilot, "blind": blind, "evaluate": evaluate}[args.action](args.out, frozen)
    except BaseException as exc:
        if not (args.out / "HALT.json").exists():
            atomic(args.out / "HALT.json", {"permanent": True, "reason_class": type(exc).__name__,
                                           "reason": str(exc) if isinstance(exc, RuntimeError) else "execution_failure",
                                           "phase": args.action})
        raise


if __name__ == "__main__":
    main()
