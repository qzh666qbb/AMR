"""Freeze graph-only development readout, compare references, package human review."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.semantic_fact_layers import FIELDS, compare_fact_layers
from src.core.semantic_graph_fact_reader import VERSION, read_graph_facts, compare_development

DEFAULT_OUT = ROOT / "experiments/semantic_fact_layers_v4_readout1"
V3 = ROOT / "experiments/semantic_fact_layers_v3_parse1"
V2 = ROOT / "experiments/semantic_fact_carrier_pilot2"
CODE = ["src/core/semantic_graph_fact_reader.py", "src/core/semantic_fact_carrier.py",
        "src/core/semantic_fact_layers.py", "scripts/run_semantic_fact_graph_readout.py",
        "scripts/collect_semantic_graph_fact_reviews.py"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_map(folder, files):
    for name, expected in files.items():
        if sha(folder / name) != expected:
            raise RuntimeError("Sealed source changed: " + str(folder / name))
    return len(files)


def source_checks():
    maps = {
        str(V3.relative_to(ROOT)): load(V3 / "SHA256.json"),
        str((V2 / "private_review/model_review").relative_to(ROOT)):
            load(V2 / "private_review/model_review/ANNOTATION_SEAL.json")["files"],
        str((V2 / "private_review/fact_layers").relative_to(ROOT)):
            load(V2 / "private_review/fact_layers/INPUT_SEAL.json")["files"],
    }
    counts = {name: verify_map(ROOT / name, mapping) for name, mapping in maps.items()}
    return maps, counts


def prepare(out):
    import penman
    if (out / "FREEZE.json").exists():
        raise RuntimeError("Existing freeze: use run/pack or a new version")
    maps, counts = source_checks()
    original = load(V3 / "private_review/ROWS.json")
    layers = {r["case_id"]: r for r in load(V3 / "private_review/LAYER_TABLE_WITH_RAW_AMR.json")}
    inputs = {r["case_id"]: r for r in load(V2 / "private_review/fact_layers/PARSER_INPUTS_NO_TARGET.json")["rows"]}
    if len(original) != 192 or len(layers) != 192 or set(inputs) != set(layers):
        raise RuntimeError("192-row identity mismatch")
    if any(l["human_text_confirmed"] or l["independent_revision_attestation"] for l in layers.values()):
        raise RuntimeError("New human annotations require a separately bound evidence version")
    keys_path = V2 / "private_review/admin/KEYS.json"
    plan_path = Path.home() / ".cache/amr/semantic_fact_carrier_pilot2/PLAN.json"
    if sha(plan_path) != load(V2 / "FREEZE.json")["private_plan_sha256"]:
        raise RuntimeError("Frozen private plan drift")
    keys = load(keys_path)["1"]["facts"]
    plan = {e["row"]["item_id"]: e["row"] for e in load(plan_path)}
    graph_inputs, canonical, admin = [], [], []
    for row in original:
        case_id = row["case_id"]
        layer, parser_input = layers[case_id], inputs[case_id]
        literal = row["original_penman"]
        if digest(literal) != row["original_penman_sha256"] or digest(parser_input["input_text"]) != row["input_text_sha256"]:
            raise RuntimeError("Original literal SHA differs")
        contract = layer["public_roles_object"]
        identity = keys[layer["model_fact_item_id"]]["identity"]
        planned = plan[identity["item"]]
        if identity["condition"] != row["condition"] or planned["public_contract"] != contract:
            raise RuntimeError("Plan identity/contract differs")
        graph_only = "\n".join(l for l in literal.splitlines() if not l.lstrip().startswith("#"))
        graph_inputs.append({"case_id": case_id, "original_penman": literal, "contract": contract})
        canonical.append({
            "case_id": case_id, "condition": row["condition"], "text": parser_input["input_text"],
            "input_text_sha256": row["input_text_sha256"],
            "original_penman": literal, "original_penman_sha256": row["original_penman_sha256"],
            "graph_only": graph_only, "graph_only_sha256": digest(graph_only),
            "contract": contract, "contract_sha256": digest(json.dumps(contract, sort_keys=True)),
            "structurally_eligible": row["raw_parse"]["structurally_eligible"],
            "normalized_triples": row["graph_inventory"]["triples"],
        })
        admin.append({"case_id": case_id, "planned_symbol": planned["intended_symbol"],
                      "model_fact_item_id": layer["model_fact_item_id"]})
    out.mkdir(parents=True, exist_ok=True)
    private = out / "private_review"
    private.mkdir(mode=0o700)
    for name in CODE:
        dst = private / "frozen_code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
    dump(private / "GRAPH_INPUTS_NO_TARGET.json", graph_inputs)
    dump(private / "FORMAL_CANONICAL_INPUTS.json", canonical)
    dump(private / "ADMIN_PLANNED_SYMBOLS.json", admin)
    shutil.copy2(V2 / "private_review/model_review/facts.json", private / "SEALED_MODEL_FACTS.json")
    tests = Path("/tmp/test_semantic_graph_fact_reader.py")
    if tests.exists():
        shutil.copy2(tests, private / "frozen_code/test_semantic_graph_fact_reader.py")
    collector_tests = Path("/tmp/test_semantic_graph_fact_review_collector.py")
    if collector_tests.exists():
        shutil.copy2(collector_tests, private / "frozen_code/test_semantic_graph_fact_review_collector.py")
    shutil.copy2(ROOT / "tests/test_semantic_fact_layers.py", private / "frozen_code/test_semantic_fact_layers.py")
    freeze = {
        "version": "semantic-fact-layers/v4-development-readout1", "at": datetime.now(timezone.utc).isoformat(),
        "authorization": "应补图事实读出与分类，先做开发对照，然后帮我确认正式归因",
        "reader_version": VERSION, "rows": 192, "original": 96, "paraphrase": 96,
        "source_maps": maps, "verified_sealed_file_counts": counts,
        "source_manifest_sha256": sha(V3 / "SHA256.json"),
        "graph_inputs_sha256": sha(private / "GRAPH_INPUTS_NO_TARGET.json"),
        "model_reference_sha256": sha(private / "SEALED_MODEL_FACTS.json"),
        "admin_plans_sha256": sha(private / "ADMIN_PLANNED_SYMBOLS.json"),
        "canonical_sha256": sha(private / "FORMAL_CANONICAL_INPUTS.json"),
        "protocol_sha256": sha(out / "PROTOCOL.md"), "code_sha256": {n: sha(ROOT / n) for n in CODE},
        "reader_test_sha256": sha(tests) if tests.exists() else None,
        "collector_test_sha256": sha(collector_tests) if collector_tests.exists() else None,
        "formal_layers_test_sha256": sha(ROOT / "tests/test_semantic_fact_layers.py"),
        "penman_version": penman.__version__,
        "graph_reader_receives_only_original_penman_and_public_contract": True,
        "readout_all_rows_before_reference_join": True, "model_references_are_not_human_gold": True,
        "same_batch_development_only": True, "structural_warning_rows_primary_erasures": 20,
        "no_new_parser_provider_detector_embedding_or_message_decode_calls": True,
    }
    dump(out / "FREEZE.json", freeze)
    print(json.dumps({"status": "FROZEN_BEFORE_BATCH", "rows": 192, "freeze_sha256": sha(out / "FREEZE.json")}))


def check_freeze(out):
    f = load(out / "FREEZE.json")
    for folder, mapping in f["source_maps"].items():
        verify_map(ROOT / folder, mapping)
    verify_map(ROOT, f["code_sha256"])
    if sha(out / "PROTOCOL.md") != f["protocol_sha256"]:
        raise RuntimeError("Protocol drift")
    for name, key in (("GRAPH_INPUTS_NO_TARGET.json", "graph_inputs_sha256"),
                      ("FORMAL_CANONICAL_INPUTS.json", "canonical_sha256"),
                      ("ADMIN_PLANNED_SYMBOLS.json", "admin_plans_sha256"),
                      ("SEALED_MODEL_FACTS.json", "model_reference_sha256")):
        if sha(out / "private_review" / name) != f[key]:
            raise RuntimeError("Frozen input drift:" + name)
    return f


def summarize(rows, primary=True):
    key = "primary_prediction" if primary else "graph_only_prediction"
    confusion = {symbol: {p: 0 for p in "ABCD"} | {"ERASURE": 0} for symbol in "ABCD"}
    for row in rows:
        dev = row["development"]
        confusion[dev["text_reference_symbol"]][dev[key] or "ERASURE"] += 1
    correct = sum(r["development"][key] == r["development"]["text_reference_symbol"] for r in rows)
    return {"rows": len(rows), "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
            "erasures": sum(r["development"][key] is None for r in rows),
            "wrong_non_erased": sum(r["development"][key] is not None and r["development"][key] !=
                                   r["development"]["text_reference_symbol"] for r in rows),
            "confusion": confusion}


def run(out):
    freeze = check_freeze(out)
    private = out / "private_review"
    if (out / "RESULT.json").exists() or (private / "GRAPH_ONLY_READOUTS.json").exists():
        raise RuntimeError("Existing batch outputs; use a new version for another run")
    inputs = load(private / "GRAPH_INPUTS_NO_TARGET.json")
    readouts = []
    for row in inputs:
        result = read_graph_facts(row["original_penman"], row["contract"])
        readouts.append({"case_id": row["case_id"], "graph_readback": result})
    # Persist graph-only predictions before even loading text reference labels.
    dump(private / "GRAPH_ONLY_READOUTS.json", readouts)
    facts = {r["item_id"]: r for r in load(private / "SEALED_MODEL_FACTS.json")}
    admin = {r["case_id"]: r for r in load(private / "ADMIN_PLANNED_SYMBOLS.json")}
    canonical = {r["case_id"]: r for r in load(private / "FORMAL_CANONICAL_INPUTS.json")}
    outputs = []
    for row in readouts:
        case_id = row["case_id"]
        source = canonical[case_id]
        reference = facts[admin[case_id]["model_fact_item_id"]]
        if reference["text"] != source["text"]:
            raise RuntimeError("Reference text does not bind to the literal source")
        readback = row["graph_readback"]
        development = compare_development(text_reference=reference, graph_readback=readback,
                                          public_contract=source["contract"],
                                          structurally_eligible=source["structurally_eligible"])
        formal = compare_fact_layers(
            contract=source["contract"], planned_symbol=admin[case_id]["planned_symbol"],
            text_facts=reference, raw_graph_facts=readback["facts"],
            raw_prediction=development["primary_prediction"],
            text_human_confirmed=False, raw_graph_facts_independent_confirmed=False,
            revision_independent_confirmed=False)
        outputs.append({
            "case_id": case_id, "condition": source["condition"],
            "input_text_sha256": source["input_text_sha256"],
            "original_penman_sha256": source["original_penman_sha256"],
            "structurally_eligible": source["structurally_eligible"],
            "graph_readback": readback, "development": development, "formal": formal,
            "field_agreement_with_model_reference": {
                field: " ".join(str(readback["facts"][field]).split()).casefold() ==
                       " ".join(str(reference[field]).split()).casefold() for field in FIELDS},
        })
    dump(private / "ROWS.json", outputs)
    columns = ["case_id", "condition", "structurally_eligible", *FIELDS, "graph_only_prediction",
               "primary_prediction", "text_reference_symbol", "primary_symbol_correct",
               "graph_facts_match_model_text_reference", "candidate_failure_layers",
               "formal_status", "graph_evidence_json", "binding_and_state_issues_json"]
    with (private / "ROWS.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in outputs:
            dev, reader = row["development"], row["graph_readback"]
            writer.writerow({
                "case_id": row["case_id"], "condition": row["condition"],
                "structurally_eligible": row["structurally_eligible"],
                **{k: reader["facts"][k] for k in FIELDS},
                **{k: dev[k] for k in columns if k in dev and k != "candidate_failure_layers"},
                "candidate_failure_layers": json.dumps(dev["candidate_failure_layers"], ensure_ascii=False),
                "formal_status": row["formal"]["status"],
                "graph_evidence_json": json.dumps(reader["evidence_triples"], ensure_ascii=False),
                "binding_and_state_issues_json": json.dumps(
                    [{"node": e["node"], "concept": e["concept"], "issues": e["binding_issues"],
                      "blockers": e["blockers"], "negative": e["negative"], "not_yet": e["not_yet"]}
                     for e in reader["events"]], ensure_ascii=False),
            })
    result = {
        "status": "DEVELOPMENT_COMPARISON_COMPLETE_FORMAL_UNCONFIRMED",
        "rows": len(outputs), "actual_graph_fact_readout_calls": len(readouts),
        "development_comparisons_complete": len(outputs),
        "structurally_eligible": sum(r["structurally_eligible"] for r in outputs),
        "structural_exclusions": sum(not r["structurally_eligible"] for r in outputs),
        "primary": summarize(outputs), "graph_only_secondary": summarize(outputs, False),
        "by_condition": {c: {"primary": summarize([r for r in outputs if r["condition"] == c]),
                             "graph_only_secondary": summarize([r for r in outputs if r["condition"] == c], False)}
                         for c in ("original", "paraphrase")},
        "field_agreement_model_reference_all_192": {
            f: sum(r["field_agreement_with_model_reference"][f] for r in outputs) for f in FIELDS},
        "known_graph_fact_fields": {
            f: sum(r["graph_readback"]["facts"][f] != "unknown" for r in outputs) for f in FIELDS},
        "candidate_failure_counts": dict(Counter(
            candidate for r in outputs for candidate in r["development"]["candidate_failure_layers"])),
        "event_binding_issue_occurrences_not_row_counts": dict(Counter(
            issue for r in outputs for e in r["graph_readback"]["events"] for issue in e["binding_issues"])),
        "formal_status_counts": dict(Counter(r["formal"]["status"] for r in outputs)),
        "formal_confirmed": 0, "no_independent_human_gold": True,
        "revised_graph_readout_calls": 0, "blind_message_recovery_claimed": False,
        "new_parser_provider_detector_embedding_or_message_decode_calls": 0,
        "rows_sha256": sha(private / "ROWS.json"),
        "graph_only_outputs_persisted_before_reference_join_sha256": sha(private / "GRAPH_ONLY_READOUTS.json"),
        "freeze_sha256": sha(out / "FREEZE.json"),
    }
    dump(out / "RESULT.json", result)
    print(json.dumps({k: result[k] for k in ("status", "rows", "primary", "graph_only_secondary",
                                            "field_agreement_model_reference_all_192", "formal_confirmed")},
                     ensure_ascii=False))


def blank_answer(source, phase):
    answer = {k: source[k] for k in ("case_id", "input_text_sha256", "original_penman_sha256", "contract_sha256")}
    answer.update(reviewed=False, facts={**dict.fromkeys(FIELDS, ""), "readability": "",
                                        "explicit_status": "", "contradiction": None},
                  evidence={f: {"text_spans": [], "triples": [], "uncertainty_reason": ""} for f in FIELDS})
    if phase == "revision":
        answer.update(revised_penman="", revision_text_evidence=[], revision_note="")
    return answer


def pack(out):
    freeze = check_freeze(out)
    private = out / "private_review"
    if (out / "REVIEW_PREPARATION.json").exists():
        raise RuntimeError("Review packages already sealed")
    sources = load(private / "FORMAL_CANONICAL_INPUTS.json")
    instructions = """# 独立真人评审

每人192个案例，每案完成原图、文字、修订图三个阶段。所有答案为空。
不得使用模型代填、预期ABCD、模型事实或另一位评审的答案。

1. 先做 raw_graph：仅打开 raw_graph_task.json，只按原图三元组读出四项事实，
   不打开 text_task/revision_task。不为 he/she/it 猜姓名。填完保存原图答案并记录
   raw_graph_completed_at，此后不能因文字答案反向修改原图事实。
2. 再做 text：打开 text_task.json，按文字实际意思读出发送者、接收者、物品、
   transfer_status。完成后记录 text_completed_at。planned_not_occurred 必须有
   明确确定计划和尚未发生两项证据；completed 必须有明确收取/完成事实。
3. 最后 revision：按真实文字独立修订原图，填完整 PENMAN、修订图事实及
   三元组证据，说明每项修订依据。原图正确时可提交原图，但仍说明无需修订。

每项事实必须给证据。文字 evidence.text_spans 使用 Python 字符索引：
start含、end不含，quote必须与原文字面相同；Unicode标点算一个字符。
图 evidence.triples 使用任务中的 normalized_triples，逆向角色已按AMR模型归一。
unknown需填 uncertainty_reason，并给检查过的相关三元组/文字跨度。
所有字段必须明确填写，readability为clear/unclear，explicit_status为yes/no/unknown，
contradiction为true/false；状态为completed/planned_not_occurred/unknown。

确认自己实际完成后设每条 reviewed=true。reviewer.json填写唯一匿名代码、
带时区的完成时间及真实声明。机器只能检查身份代码不同、声明与证据一致，
不能证明真人身份或自动判断语义蕴含。第三位真人负责分歧裁决，不能简单投票。
20条原解析warning保留评审，但主要符号指标仍作为排除/擦除，完整分母192。
"""
    packet_hashes = {}
    for reviewer in (1, 2):
        folder = private / f"reviewer{reviewer}"
        folder.mkdir(mode=0o700)
        meta = {"reviewer_code": "", "completed_at": "", "raw_graph_completed_at": "",
                "text_completed_at": "", "human_attestation": False,
                "independent_attestation": False, "no_model_or_target_answers_used": False,
                "raw_graph_read_without_text_answers": False}
        dump(folder / "reviewer.json", meta)
        (folder / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
        for phase in ("raw_graph", "text", "revision"):
            tasks = []
            for source in sources:
                task = {k: source[k] for k in ("case_id", "input_text_sha256",
                                              "original_penman_sha256", "contract", "contract_sha256")}
                if phase in ("raw_graph", "revision"):
                    task.update(graph_only=source["graph_only"], graph_only_sha256=source["graph_only_sha256"],
                                normalized_triples=source["normalized_triples"],
                                structurally_eligible=source["structurally_eligible"])
                if phase in ("text", "revision"):
                    task["text"] = source["text"]
                tasks.append(task)
            dump(folder / (phase + "_task.json"), tasks)
            dump(folder / (phase + "_answers.json"), [blank_answer(s, phase) for s in sources])
        immutable = {p.name: sha(p) for p in folder.glob("*_task.json")}
        dump(folder / "INPUT_SHA256.json", immutable)
        zip_path = private / f"reviewer{reviewer}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.iterdir()):
                archive.write(path, arcname=path.name)
        zip_path.chmod(0o600)
        packet_hashes[str(reviewer)] = sha(zip_path)
    start = """# 图事实正式归因评审入口

开发比较已完成；正式确认需两名独立真人提交原图、文字和修订图答案。
每人192案例 × 3阶段 = 576行。先只看图，再看文字，最后独立修订图。
reviewer1.zip/reviewer2.zip 已准备；原图任务不含文字，全部任务不含目标或模型答案。
请按包内 INSTRUCTIONS.md 填写三个 *_answers.json 和 reviewer.json。

回收命令（在项目根目录执行，使用新的输出目录）：

```bash
python3 scripts/collect_semantic_graph_fact_reviews.py \\
  --experiment experiments/semantic_fact_layers_v4_readout1 \\
  --reviewer1 /path/to/completed/reviewer1 \\
  --reviewer2 /path/to/completed/reviewer2 \\
  --out experiments/semantic_fact_layers_v4_readout1/private_review/human_collection1 \\
  --dependency-path /tmp/amr_fact_readout_deps
```

依赖为 penman==1.3.1；已安装时可省 dependency-path。原代码及输入按FREEZE核验。
若有分歧，回收目录会产生 DISAGREEMENTS.json 与第三人空白模板；
独立第三人完成后加 --adjudicator 并另选新输出目录重跑。
拒绝、unknown、缺证据和分歧保留在192条分母中，不计已确认。
"""
    (private / "START.md").write_text(start, encoding="utf-8")
    dump(out / "REVIEW_PREPARATION.json", {
        "status": "INDEPENDENT_HUMAN_PACKETS_READY_NOT_COMPLETED",
        "cases_per_reviewer": 192, "answer_rows_per_reviewer": 576,
        "human_completed": 0, "canonical_sha256": freeze["canonical_sha256"],
        "packet_sha256": packet_hashes, "target_or_model_labels_in_packets": False,
        "original_graph_phase_contains_text": False,
    })
    print(json.dumps({"status": "REVIEW_PACKETS_READY", "cases_per_reviewer": 192}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "run", "pack"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dependency-path", type=Path)
    args = ap.parse_args()
    if args.dependency_path:
        sys.path.insert(0, str(args.dependency_path))
    {"prepare": prepare, "run": run, "pack": pack}[args.action](args.out)
