"""Frozen, development-only target/actual AMR diagnostics; no detector scores."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def isolate(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError("Required source functions not found")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace


def prepare(root, out, inventory_path):
    import importlib.metadata as md
    import spacy
    from nltk.corpus import propbank

    if (out / "FREEZE.json").exists():
        raise RuntimeError("Existing freeze cannot be overwritten")
    run = root / "experiments/semantic_sequence_stage1_run2"
    assignments = sorted(
        [a for a in lines(run / "private/ASSIGNMENTS.jsonl")
         if a["split"] == "development" and a["half_index"] in (0, 1, 2)],
        key=lambda a: a["half_index"])
    assert len(assignments) == 3
    inventory = json.loads(inventory_path.read_text())
    splitter = spacy.blank("en")
    splitter.add_pipe("sentencizer")
    sealed = {r["carrier_id"]: r for r in lines(
        run / "private/annotation/coordinator_only/stage2_templates_sealed.jsonl")}
    labels = lines(run / "private/ai_review1/EVALUATOR_AI_LABELS.jsonl")
    catalog = {r["graph_id"]: r for r in lines(
        run / "private/ai_review1/TARGET_GRAPH_CATALOG.jsonl")}
    inputs, carriers, docids = [], [], []
    for assignment in assignments:
        docid = assignment["doc_id"]
        docids.append(docid)
        doc = json.loads((run / f"private/documents/d{docid:02d}.json").read_text())
        positive = doc["arms"]["positive"]
        assert digest(positive["paraphrase"]) == positive["paraphrase_sha256"]
        para_units = []
        for index, span in enumerate(splitter(positive["paraphrase"]).sents):
            text = span.text.strip()
            start = span.start_char + len(span.text) - len(span.text.lstrip())
            end = start + len(text)
            item = {"input_id": f"d{docid:02d}-paraphrase-u{index:02d}",
                    "doc_id": docid, "condition": "paraphrase",
                    "span": [start, end], "text": text, "text_sha256": digest(text)}
            para_units.append(item)
            inputs.append(item)
        for slot in assignment["slots"]:
            index = slot["slot"]
            label = next(r for r in labels
                         if r["doc_id"] == docid and r["planned_carrier_slot"] == index)
            source = sealed[label["carrier_id"]]
            assert source["source_text"] == positive["text"]
            start, end = source["source_carrier_span_construction_only"]
            text = positive["text"][start:end]
            assert text == positive["slots"][index]["text"]
            target = catalog[label["graph_id"]]
            assert digest(target["target_amr"]) == label["target_graph_sha256"]
            item = {"input_id": f"d{docid:02d}-clean-c{index}",
                    "doc_id": docid, "condition": "clean",
                    "span": [start, end], "text": text, "text_sha256": digest(text)}
            inputs.append(item)
            span = label["paraphrase"].get("evidence_span")
            associated = [u["input_id"] for u in para_units if span
                          and max(span[0], u["span"][0]) < min(span[1], u["span"][1])]
            carriers.append({
                "carrier_id": label["carrier_id"], "doc_id": docid, "slot": index,
                "template": slot["intended_template"], "state": slot["intended_state"],
                "target_amr": target["target_amr"],
                "target_sha256": label["target_graph_sha256"],
                "clean_input": item["input_id"], "paraphrase_inputs": associated,
                "mapping_span_AI_only": span, "prior_AI": {
                    condition: label[condition] for condition in ("clean", "paraphrase")}})
    assert len({c["template"] for c in carriers}) == len(carriers) == 24
    private = out / "private"
    private.mkdir(parents=True, exist_ok=True)
    atomic(private / "INPUTS.json", inputs)
    atomic(private / "CARRIERS.json", carriers)
    atomic(private / "propbank_verbs.json",
           sorted({instance.roleset.split(".")[0] for instance in propbank.instances()}))
    normalizer_model = spacy.load("en_core_web_sm")
    model_files = sorted(Path(normalizer_model.path).rglob("*"))
    normalizer_assets = [{"path": str(f.relative_to(normalizer_model.path)),
                         "sha256": sha(f)} for f in model_files if f.is_file()]
    atomic(out / "NORMALIZER_ASSETS.json", normalizer_assets)
    source_paths = [
        Path(__file__), root / "src/core/parser_recovery_audit.py",
        root / "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py",
        root / "src/core/bank_scorer.py", root / "scripts/semantic_sequence_stage1_common.py"]
    frozen = {
        "schema": "semantic-sequence-target-alignment/v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "authorization": "User requested actual target/AMR alignment review",
        "scope": "Fixed development diagnostics, not original G1 or blind validation",
        "doc_ids": docids, "half_indices": [0, 1, 2], "carriers": len(carriers),
        "inputs": len(inputs), "template_coverage": sorted(c["template"] for c in carriers),
        "selection": "First three development half_index values; no score selection",
        "segmentation": "spaCy blank(en)+default sentencizer on complete paraphrase; clean carriers unchanged",
        "mapping": "All sentence units overlapping previously sealed AI evidence_span; no best-score selection; unknown retained",
        "parser": {"device": "cuda:0", "dtype": "float32", "batch_size": 1,
                   "num_beams": 4, "max_in_len": 1024, "max_out_len": 1024,
                   "input_overflow": "record then halt, never truncate",
                   "unbound_warning": "permanent halt; do not resume manifest",
                   "inference_assets": inventory["files"]},
        "normalizer": {
            "official": "AST-isolated original classify_concept+generate_template_amr",
            "project": "AST-isolated bank_scorer._concept_class, quoted literals to NE",
            "versions": {n: md.version(n) for n in
                         ("spacy", "nltk", "en-core-web-sm", "penman", "networkx")},
            "spacy_assets_sha256": sha(out / "NORMALIZER_ASSETS.json"),
            "propbank_verbs_sha256": sha(private / "propbank_verbs.json")},
        "comparison": "AMR inverse-normalized, injective target subgraph monomorphism, exact placeholder/role/literal and reentrancy; extra observed structure allowed. Rooted and anywhere reported. This is stricter than fuzzy S2Match and is not a detector.",
        "feature_diagnostic": "Frozen dictionary feature signature equality only, no class ranking or readout tuning",
        "source_sha256": {str(f.relative_to(root)): sha(f) for f in source_paths},
        "input_sha256": sha(private / "INPUTS.json"),
        "carrier_sha256": sha(private / "CARRIERS.json"),
        "original_generation_manifest_sha256": sha(run / "MANIFEST_GENERATION.sha256"),
        "sealed_AI_labels_sha256": sha(run / "private/ai_review1/EVALUATOR_AI_LABELS.jsonl"),
        "no_generation_no_detector_no_new_threshold": True}
    atomic(out / "FREEZE.json", frozen)
    print(json.dumps({"prepared": True, "docs": docids, "inputs": len(inputs),
                      "carriers": len(carriers), "freeze_sha256": sha(out / "FREEZE.json")}))


def recovery_namespace(source):
    namespace = dict(globals())
    namespace.update(AUDIT_VERSION="amrlib-recovery-audit:v1",
                     _LOCK=threading.RLock(), text_sha256=lambda t: digest(
                         re.sub(r"\s+", " ", t).strip()))
    return isolate(source, ["bytes_sha", "RecoveryHandler", "capture_recovery_logs",
                            "parse_with_recovery_audit"], namespace)


def parse(root, out):
    import fcntl
    import inspect
    import importlib.metadata as md
    import torch
    import amrlib
    import amrlib.models.parse_xfm.inference as inference_module
    import penman
    from penman.models.amr import model as amr_model

    out = out.resolve()
    private = out / "private"
    private.mkdir(parents=True, exist_ok=True)
    lock = (private / "parse.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    halt_path = out / "HALT.json"
    if halt_path.exists():
        raise RuntimeError("Permanent halt: same manifest must not resume")
    freeze = json.loads((out / "FREEZE.json").read_text())
    def halt(reason):
        if not halt_path.exists():
            atomic(halt_path, {"reason": reason, "at": datetime.now(timezone.utc).isoformat(),
                               "freeze_sha256": sha(out / "FREEZE.json")})
    try:
        assert sha(private / "INPUTS.json") == freeze["input_sha256"]
        for asset in freeze["parser"]["inference_assets"]:
            assert sha(root / asset["path"]) == asset["sha256"]
        assert sha(Path(__file__)) == freeze["source_sha256"][
            "scripts/audit_semantic_sequence_alignment.py"]
        recovery_source = out / "parser_recovery_audit.py"
        assert sha(recovery_source) == freeze["source_sha256"][
            "src/core/parser_recovery_audit.py"]
        assert torch.cuda.is_available()
        torch.manual_seed(20261023)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        namespace = recovery_namespace(recovery_source)
        load_events = []
        with namespace["capture_recovery_logs"](
                {"phase": "model_load", "span_index": None, "beam_index": None}, load_events):
            model = amrlib.load_stog_model(
                model_dir=str(root / "models/model_parse_xfm_bart_large-v0_1_0"),
                tok_name_or_path=str(root / "models/bart-large-tokenizer"),
                device="cuda:0", batch_size=1, num_beams=4)
        if load_events:
            atomic(private / "LOAD_DIAGNOSTICS.json", load_events)
            raise RuntimeError("Unbound parser diagnostic during load")
        assert all(p.dtype == torch.float32 for p in model.model.parameters())
        assert model.max_sent_len == model.max_graph_len == 1024
        inputs = json.loads((private / "INPUTS.json").read_text())
        token_lengths = {r["input_id"]: len(model.tokenizer(
            r["text"], truncation=False)["input_ids"]) for r in inputs}
        atomic(out / "TOKEN_LENGTHS.json", token_lengths)
        if any(n >= model.max_sent_len for n in token_lengths.values()):
            raise RuntimeError("Input reaches native tokenizer limit; no truncation authorized")
        source_files = [Path(inspect.getfile(inference_module)),
                        Path(inspect.getfile(inference_module.PenmanDeSerializer))]
        runtime = {
            "python": sys.version, "versions": {n: md.version(n) for n in
                ("torch", "amrlib", "transformers", "penman", "spacy")},
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "inference_sources": [{"filename": f.name, "sha256": sha(f)} for f in source_files],
            "dtype": str(next(model.model.parameters()).dtype), "tf32": False,
            "max_input_tokens": max(token_lengths.values()),
            "freeze_sha256": sha(out / "FREEZE.json")}
        atomic(out / "RUNTIME.json", runtime)
        records_dir = private / "PARSES"
        records_dir.mkdir(exist_ok=True)
        completed, cached, total_seconds = 0, 0, 0.0
        original_generate = model.model.generate
        lengths = []
        def monitored_generate(*args, **kwargs):
            outputs = original_generate(*args, **kwargs)
            eos = model.tokenizer.eos_token_id
            pad = model.tokenizer.pad_token_id
            for ids in outputs:
                values = ids.tolist()
                while values and values[-1] == pad:
                    values.pop()
                lengths.append({"length": len(values),
                                "ends_eos": bool(values) and values[-1] == eos})
            return outputs
        model.model.generate = monitored_generate
        for row in inputs:
            destination = records_dir / (row["input_id"] + ".json")
            if destination.exists():
                prior = json.loads(destination.read_text())
                assert prior["freeze_sha256"] == runtime["freeze_sha256"]
                assert prior["text_sha256"] == row["text_sha256"]
                assert digest(prior["amr"]) == prior["amr_sha256"]
                cached += 1
            else:
                lengths.clear()
                start = time.monotonic()
                outputs, audits = namespace["parse_with_recovery_audit"](
                    model, [row["text"]], inference_module)
                elapsed = time.monotonic() - start
                # Decode diagnostics are explicitly bound to this single input.
                decode_events = []
                state = {"phase": "roundtrip", "span_index": 0, "beam_index": None}
                with namespace["capture_recovery_logs"](state, decode_events):
                    graph = penman.decode(outputs[0], model=amr_model)
                    recovered = penman.decode(penman.encode(graph, model=amr_model),
                                              model=amr_model)
                valid = (graph.top is not None and bool(graph.instances())
                         and graph.top in graph.variables()
                         and graph.top == recovered.top
                         and sorted(graph.triples) == sorted(recovered.triples))
                clipped_output = any(n["length"] >= model.max_graph_len
                                     and not n["ends_eos"] for n in lengths)
                result = {**row, "amr": outputs[0], "amr_sha256": digest(outputs[0]),
                          "recovery_audit": audits[0], "decode_events": decode_events,
                          "graph_roundtrip_valid": valid,
                          "output_beam_lengths": lengths.copy(),
                          "output_limit_hit": clipped_output,
                          "elapsed_seconds": elapsed, "token_length": token_lengths[row["input_id"]],
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                          "freeze_sha256": runtime["freeze_sha256"]}
                atomic(destination, result)
                total_seconds += elapsed
            completed += 1
            atomic(out / "heartbeat.json", {
                "completed": completed, "total": len(inputs), "cached": cached,
                "last_input_id": row["input_id"], "at": datetime.now(timezone.utc).isoformat()})
        atomic(out / "PARSE_COMPLETE.json", {
            "completed": completed, "cached": cached,
            "inference_seconds_this_invocation": total_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "freeze_sha256": runtime["freeze_sha256"]})
        print(json.dumps({"completed": completed, "cached": cached, "seconds": total_seconds}))
    except BaseException as exc:
        halt(type(exc).__name__ + ": " + str(exc))
        raise


def normalization_namespace(root, out):
    import penman
    import spacy
    namespace = dict(penman=penman, re=re, logger=logging.getLogger("alignment-normalizer"),
                     verb_pattern=re.compile(r"(.*)-\d+$"),
                     verb_lemmas=set(json.loads((out / "private/propbank_verbs.json").read_text())),
                     nlp=spacy.load("en_core_web_sm"))
    isolate(root / "third_party/SWAN_official_2ca0c207_clean/utils/amr_utils.py",
            ["classify_concept", "generate_template_amr"], namespace)
    project = dict(re=re, _TEMPLATE_CONCEPTS={"ADJ", "LOC", "N", "NE", "NUM", "V", "X"},
                   _NUMERIC_CONCEPTS={"date-entity", "monetary-quantity", "ordinal-entity",
                                      "percentage-entity", "temporal-quantity"},
                   _LOCATION_CONCEPTS={"city", "continent", "country", "province", "state"},
                   _SENSE_RE=re.compile(r".+-\d+$"))
    isolate(root / "src/core/bank_scorer.py", ["_concept_class"], project)
    def project_normalize(amr):
        graph = penman.decode(amr)
        named = {s for s, r, t in graph.triples if r == ":name"}
        triples = []
        for s, r, t in graph.triples:
            if r == ":instance":
                t = project["_concept_class"](t, s in named)
            elif t.startswith('"') and t.endswith('"'):
                t = '"NE"'
            triples.append((s, r, t))
        return penman.encode(penman.Graph(triples))
    return namespace["generate_template_amr"], project_normalize


def structure(amr):
    import networkx as nx
    import penman
    from penman.models.amr import model
    graph = penman.decode(amr, model=model)
    network = nx.MultiDiGraph()
    concepts = {s: t for s, r, t in graph.triples if r == ":instance"}
    for variable, concept in concepts.items():
        network.add_node(variable, concept=concept, kind="variable")
    for index, (s, r, t) in enumerate(graph.triples):
        if r in (":instance", ":wiki"):
            continue
        if t not in concepts:
            variable = f"__literal_{index}"
            network.add_node(variable, concept=t, kind="literal")
            t = variable
        network.add_edge(s, t, role=r)
    network.graph["top"] = graph.top
    return network


def covers(observed, target, rooted=False):
    from networkx.algorithms.isomorphism import MultiDiGraphMatcher
    def node_match(left, right):
        return left["concept"] == right["concept"] and left["kind"] == right["kind"]
    def edge_match(left, right):
        actual = Counter(v["role"] for v in left.values())
        expected = Counter(v["role"] for v in right.values())
        return all(actual[r] >= n for r, n in expected.items())
    if rooted:
        # Make root an explicit constraint, avoiding enumeration of non-root mappings.
        observed, target = observed.copy(), target.copy()
        for graph in (observed, target):
            for node in graph:
                graph.nodes[node]["is_root"] = node == graph.graph["top"]
        original_node_match = node_match
        node_match = lambda l, r: original_node_match(l, r) and l["is_root"] == r["is_root"]
    matcher = MultiDiGraphMatcher(observed, target, node_match=node_match,
                                 edge_match=edge_match)
    return matcher.subgraph_is_monomorphic()


def local_constraints(network, root):
    concept = network.nodes[root]["concept"]
    args, names, literals, nested = [], [], [], []
    for s, t, data in network.edges(data=True):
        role = data["role"]
        if s == root and re.fullmatch(r":ARG\d+", role):
            args.append([role, network.nodes[t]["concept"]])
            if network.nodes[t]["concept"] == "V":
                nested.append(role)
        if role == ":name":
            ops = sorted(d["role"] for _, _, d in network.out_edges(t, data=True)
                         if re.fullmatch(r":op\d+", d["role"]))
            names.append(ops)
        if s == root and network.nodes[t]["kind"] == "literal":
            literals.append([role, network.nodes[t]["concept"]])
    indegree = [n for n in network if network.nodes[n]["kind"] == "variable"
                and network.in_degree(n) > 1]
    return {"root_class": concept, "root_arguments": sorted(args),
            "root_nested_predicate_roles": sorted(nested),
            "root_literals": sorted(literals),
            "name_operand_sets": sorted(names), "reentrant_variable_count": len(indegree)}


def compare(root, out):
    import importlib.util
    import penman
    from penman.models.amr import model as amr_model

    freeze = json.loads((out / "FREEZE.json").read_text())
    assert sha(out / "private/INPUTS.json") == freeze["input_sha256"]
    assert sha(out / "private/CARRIERS.json") == freeze["carrier_sha256"]
    for path, expected in freeze["source_sha256"].items():
        assert sha(root / path) == expected
    assert not (out / "HALT.json").exists()
    assert (out / "PARSE_COMPLETE.json").exists()
    official, project = normalization_namespace(root, out)
    spec = importlib.util.spec_from_file_location(
        "alignment_features", root / "scripts/semantic_sequence_stage1_common.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    parses = {p.stem: json.loads(p.read_text()) for p in
              (out / "private/PARSES").glob("*.json")}
    carriers = json.loads((out / "private/CARRIERS.json").read_text())
    detailed, comparisons = [], []
    for carrier in carriers:
        target = structure(carrier["target_amr"])
        target_features = helpers.structural_features(carrier["target_amr"])
        for condition in ("clean", "paraphrase"):
            ids = ([carrier["clean_input"]] if condition == "clean"
                   else carrier["paraphrase_inputs"])
            units = []
            for input_id in ids:
                parsed = parses[input_id]
                raw = "\n".join(l for l in parsed["amr"].splitlines()
                                if not l.lstrip().startswith("#"))
                normalized = {kind: function(raw) for kind, function in
                              (("official", official), ("project", project))}
                graphs = {kind: structure(amr) for kind, amr in normalized.items()}
                eligible = (parsed["graph_roundtrip_valid"] and not parsed["output_limit_hit"]
                            and not parsed["decode_events"]
                            and not parsed["recovery_audit"]["events"])
                # Retain raw/recovered results, but do not count warned edges as clean evidence.
                results = {}
                for kind, graph in graphs.items():
                    results[kind] = {
                        "full_target_anywhere": covers(graph, target),
                        "full_target_at_root": covers(graph, target, rooted=True),
                        "frozen_feature_signature_equal": (
                            helpers.structural_features(normalized[kind]) == target_features),
                        "observed_constraints": local_constraints(graph, graph.graph["top"])}
                raw_graph = penman.decode(raw, model=amr_model)
                classified = {s: {"concept": t, "official": None, "project": None}
                              for s, r, t in raw_graph.triples if r == ":instance"}
                for kind, value in normalized.items():
                    for s, r, t in penman.decode(value).triples:
                        if r == ":instance":
                            classified[s][kind] = t
                differences = [v for v in classified.values() if v["official"] != v["project"]]
                units.append({"input_id": input_id, "eligible": eligible,
                              "normalization_differences": differences,
                              "normalizations": normalized, "results": results})
            public = {
                "carrier_id": carrier["carrier_id"], "doc_id": carrier["doc_id"],
                "slot": carrier["slot"], "template": carrier["template"],
                "state": carrier["state"], "condition": condition,
                "prior_AI_realized": carrier["prior_AI"][condition]["realized"],
                "mapped_units": len(units), "eligible_units": sum(u["eligible"] for u in units),
                "target_constraints": local_constraints(target, target.graph["top"])}
            for kind in ("official", "project"):
                for measure in ("full_target_anywhere", "full_target_at_root",
                                "frozen_feature_signature_equal"):
                    public[f"{kind}_{measure}"] = any(
                        u["eligible"] and u["results"][kind][measure] for u in units)
            public["normalization_difference_nodes"] = sum(
                len(u["normalization_differences"]) for u in units)
            detailed.append({**public, "target_amr": carrier["target_amr"],
                             "prior_AI": carrier["prior_AI"][condition], "units": units})
            comparisons.append(public)
    atomic(out / "private/ALIGNMENT_DETAILS.json", detailed)
    atomic(out / "ALIGNMENT_ROWS.json", comparisons)
    summaries = {}
    for condition in ("clean", "paraphrase"):
        rows = [r for r in comparisons if r["condition"] == condition]
        summaries[condition] = {
            "planned_carriers": len(rows), "mapped_carriers": sum(r["mapped_units"] > 0 for r in rows),
            "prior_AI_labels": dict(Counter(r["prior_AI_realized"] for r in rows)),
            "official_full_target_anywhere": sum(r["official_full_target_anywhere"] for r in rows),
            "official_full_target_at_root": sum(r["official_full_target_at_root"] for r in rows),
            "project_full_target_anywhere": sum(r["project_full_target_anywhere"] for r in rows),
            "official_frozen_feature_signature_equal": sum(
                r["official_frozen_feature_signature_equal"] for r in rows),
            "project_frozen_feature_signature_equal": sum(
                r["project_frozen_feature_signature_equal"] for r in rows),
            "prior_AI_no_but_actual_full_target": sum(
                r["prior_AI_realized"] == "no" and r["official_full_target_anywhere"] for r in rows)}
    unique = list(parses.values())
    summary = {
        "scope": freeze["scope"], "doc_ids": freeze["doc_ids"],
        "unique_parser_inputs": len(unique), "conditions": summaries,
        "recovery": {
            "inputs_with_bound_events": sum(bool(p["recovery_audit"]["events"]) for p in unique),
            "warning_events": sum(e["level"] == "WARNING" for p in unique
                                  for e in p["recovery_audit"]["events"]),
            "error_events": sum(e["level"] in ("ERROR", "CRITICAL") for p in unique
                                for e in p["recovery_audit"]["events"]),
            "fallback_beams": sum(p["recovery_audit"]["selected_beam"] > 0 for p in unique),
            "invalid_roundtrip": sum(not p["graph_roundtrip_valid"] for p in unique),
            "output_limit_hits": sum(p["output_limit_hit"] for p in unique),
            "unbound_events": 0},
        "new_generation_calls": 0, "detector_calls": 0, "S2Match_calls": 0,
        "inference_seconds": sum(p["elapsed_seconds"] for p in unique),
        "peak_reserved_bytes": max(p["peak_reserved_bytes"] for p in unique),
        "freeze_sha256": sha(out / "FREEZE.json"),
        "original_G1": "NOT_EVALUATED", "no_independent_parser_gold": True}
    atomic(out / "ALIGNMENT_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("command", choices=("prepare", "parse", "compare"))
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--inventory", type=Path)
    args = cli.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.out, args.inventory)
    elif args.command == "parse":
        parse(args.root, args.out)
    else:
        compare(args.root, args.out)


if __name__ == "__main__":
    main()
