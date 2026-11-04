"""Public-only cycle1 package, byte manifest and private-content scan.

Builds from a fixed file allowlist. --verify-only reads a zip using only Python
standard-library dependencies and never executes any code from the archive.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/semantic_sequence_cycle1"
PRIVATE = Path.home() / ".cache/amr/semantic_sequence_cycle1"
RESULT_FILES = (
    "PROTOCOL.md", "FINAL_REPORT.md", "FOLLOWUP.md", "REPRODUCTION.md",
    "FREEZE.json", "PARSER_FREEZE.json", "PERMUTATIONS.json",
    "GENERATION_COMPLETE.json", "WRITE_READ_RESULTS.json",
    "NORMALIZATION_CORRECTION.json", "FAILURE_DIAGNOSIS.json",
    "CUDA_PRECHECK.json", "PARSE_COMPLETE.json", "INDEPENDENT_CHECKS.json",
    "SUPPLEMENTARY_AUDIT.json", "ENGINEERING_CHECKS.json", "ANNOTATION_STATUS.json",
    "STATUS.json", "PROCESS_CLOSURE.json", "CLOSEOUT.json", "SHA256SUMS.json",
    "DEPENDENCIES.json", "EXTERNAL_ASSET_REQUIREMENTS.json",
    "requirements-engineering.txt", "requirements-audit.txt",
)
SOURCE_FILES = (
    "src/core/semantic_sequence.py", "src/core/parser_recovery_audit.py",
    "src/core/bank_scorer.py", "scripts/run_semantic_sequence_cycle.py",
    "scripts/audit_semantic_sequence_cycle.py",
    "scripts/audit_semantic_sequence_alignment.py",
    "scripts/test_semantic_sequence_cycle.py",
    "scripts/check_semantic_sequence_cycle.py",
    "scripts/diagnose_semantic_sequence_cycle.py",
    "scripts/check_semantic_sequence_closeout.py",
    "scripts/package_semantic_sequence_closeout.py",
)
ROOT_DOCS = ("README.md", "任务清单.md", "experiments/README.md",
             "CLAIM_EVIDENCE_MATRIX.md")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def validate_zip(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate zip member"
        for name in names:
            item = PurePosixPath(name)
            assert not item.is_absolute() and ".." not in item.parts
            assert not name.endswith("/") and not (archive.getinfo(name).external_attr >> 16) & 0o170000 == 0o120000
        manifest = json.loads(archive.read("MANIFEST.sha256.json"))
        assert set(names) == set(manifest["files"]) | {"MANIFEST.sha256.json"}
        for name, expected in manifest["files"].items():
            assert digest(archive.read(name)) == expected, "Archive SHA mismatch"
        assert archive.testzip() is None, "Archive CRC mismatch"
        return {"passed": True, "member_count": len(names),
                "hashed_members": len(manifest["files"]),
                "zip_sha256": digest(path.read_bytes()),
                "uncompressed_bytes": sum(archive.getinfo(name).file_size for name in names)}


def private_phrases():
    """Exact sealed text/graph matches and long fragments; never print contents."""
    values = set()
    text_keys = {"text", "paraphrase", "amr", "target", "target_amr", "normalized",
                 "source_text", "carrier_text", "prompt"}

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in text_keys and isinstance(item, str) and len(item) >= 24:
                    values.add(item.encode())
                elif key == "slots" and isinstance(item, list):
                    values.update(x.encode() for x in item if isinstance(x, str) and len(x) >= 24)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    files = [PRIVATE / "PROMPTS.json", PRIVATE / "TARGETS.json",
             PRIVATE / "CORRECTED_NORMALIZATION_DETAILS.json",
             *(PRIVATE / "documents").glob("*.json"),
             *(PRIVATE / "PARSES").glob("*.json"),
             *(PRIVATE / "calls").glob("*.response.json")]
    for path in files:
        assert path.exists(), "Private scan prerequisite missing"
        visit(read_json(path))
    # Include long prose fragments to detect partial leaks rather than full texts only.
    for value in tuple(values):
        if not value.lstrip().startswith(b"(") and b"\n" not in value:
            words = value.split()
            if len(words) >= 12:
                for i in range(len(words) - 11):
                    fragment = b" ".join(words[i:i + 12])
                    if len(fragment) >= 50:
                        values.add(fragment)
    return values


def privacy_check(payload):
    forbidden_names = {"private", "calls", "PARSES", "annotation", "models",
                       "third_party", ".secrets", "SYMBOL_MAP.json", "remote_access.py"}
    patterns = [
        rb"\bsk-[A-Za-z0-9_-]{20,}\b", rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        b"/" + b"Users/", b"/" + b"root/", b"/" + b"workspace/",
        rb"\b(?:ssh|https?)://[^/\s]*:[0-9]{2,5}",
        rb"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    ]
    sealed = private_phrases()
    for name, data in payload.items():
        assert not forbidden_names.intersection(PurePosixPath(name).parts), "Excluded payload member"
        matching = [index for index, pattern in enumerate(patterns) if re.search(pattern, data)]
        assert not matching, f"Sensitive pattern categories {matching} in allowlisted member {name}"
        assert not any(value in data for value in sealed), f"Sealed source/generated content in member {name}"
    return {"passed": True, "allowlist_only": True,
            "private_exact_and_fragment_patterns": len(sealed),
            "credential_path_endpoint_pattern_count": len(patterns),
            "checked_members": len(payload),
            "sealed_text_graph_provider_payloads_included": False,
            "external_asset_payloads_included": False,
            "scope": "Allowlist + observed private-content and pattern scan; not a formal universal privacy proof"}


def build():
    os.umask(0o077)
    assert read_json(OUT / "STATUS.json")["cycle_closed"]
    assert read_json(OUT / "PROCESS_CLOSURE.json")["closed"]
    supplement = read_json(OUT / "SUPPLEMENTARY_AUDIT.json")
    assert supplement["passed"] and read_json(OUT / "ENGINEERING_CHECKS.json")["passed"]
    for name, expected in supplement["saved_evidence_sha256"].items():
        assert digest((OUT / name).read_bytes()) == expected, "Saved evidence changed"
    for name, expected in supplement["historical_reference_sha256"].items():
        assert digest((ROOT / name).read_bytes()) == expected, "Historical reference changed"
    freeze = read_json(OUT / "FREEZE.json")
    for name, expected in freeze["source_sha256"].items():
        assert digest((ROOT / name).read_bytes()) == expected, "Frozen source changed"
    snapshots = OUT / "source_snapshots"
    snapshots.mkdir(exist_ok=True)
    parser = read_json(OUT / "PARSER_FREEZE.json")
    snapshot_paths = []
    for name, expected in parser["snapshots_sha256"].items():
        source = PRIVATE / "snapshots" / name
        assert digest(source.read_bytes()) == expected, "Frozen parser snapshot changed"
        destination = snapshots / name
        shutil.copyfile(source, destination)
        snapshot_paths.append(destination)
    sources = [OUT / name for name in RESULT_FILES if name != "SHA256SUMS.json"]
    sources += [ROOT / name for name in SOURCE_FILES + ROOT_DOCS]
    sources += snapshot_paths
    original_files = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sources}
    for path in (OUT / "SYMBOL_MAP.json", OUT / "heartbeat.json", OUT / "parse_heartbeat.json"):
        original_files[str(path.relative_to(ROOT))] = digest(path.read_bytes())
    atomic(OUT / "SHA256SUMS.json", {
        "schema": "semantic-sequence-cycle1-closeout-sha/v1",
        "files": original_files,
        "excludes": ["SHA256SUMS.json", "public_reproducibility.zip", "PACKAGE_VALIDATION.json"],
        "external_asset_payloads_excluded": True,
        "external_asset_sha": {"parser": read_json(OUT / "PARSER_FREEZE.json")["parser_assets"],
                               "dictionary": freeze["symbol_map_sha256"]}})
    payload = {str((OUT / name).relative_to(ROOT)): (OUT / name).read_bytes()
               for name in RESULT_FILES}
    payload.update({name: (ROOT / name).read_bytes() for name in SOURCE_FILES})
    payload.update({str(path.relative_to(ROOT)): path.read_bytes() for path in snapshot_paths})
    payload["requirements-engineering.txt"] = (OUT / "requirements-engineering.txt").read_bytes()
    payload["requirements-audit.txt"] = (OUT / "requirements-audit.txt").read_bytes()
    payload["PACKAGE_README.md"] = (
        "# 语义序列 cycle1 公开复现材料\n\n"
        "固定写入读回 pilot：16/24，低于冻结 20/24，No-Go。\n"
        "归一化修正完整覆盖为1/24；G1e不受影响。\n"
        "四阶段实证验证未完成，包内工程测试不代表实际顺序增益。\n\n"
        "报告和命令见 experiments/semantic_sequence_cycle1/FINAL_REPORT.md 与 REPRODUCTION.md。\n"
        "仅包含聚合统计、协议、SHA、自有代码与版本信息；私有正文、图、回执、访问配置、"
        "权重和第三方数据未再分发。\n\n"
        "解压根目录运行 python scripts/test_semantic_sequence_cycle.py。\n"
        "工程依赖见 requirements-engineering.txt；完整保存产物审计需要有权限的原私有材料。\n"
    ).encode()
    privacy = privacy_check(payload)
    payload["MANIFEST.sha256.json"] = encoded({
        "schema": "semantic-sequence-cycle1-public-archive/v1",
        "files": {name: digest(data) for name, data in sorted(payload.items())},
        "scope": "Manifest hashes every member except itself"})
    destination = OUT / "public_reproducibility.zip"
    temp = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 24, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    validated = validate_zip(temp)
    with tempfile.TemporaryDirectory(prefix="semantic-sequence-public-replay-") as directory:
        with zipfile.ZipFile(temp) as archive:
            archive.extractall(directory)
        replay = subprocess.run(
            [sys.executable, "scripts/test_semantic_sequence_cycle.py"],
            cwd=directory, text=True, capture_output=True)
        assert replay.returncode == 0, "Public package engineering replay failed"
        assert "Ran 7 tests" in replay.stderr and replay.stderr.rstrip().endswith("OK")
        replay_result = {"passed": True, "test_methods": 7,
                         "requires_private_data": False,
                         "stdout_sha256": digest(replay.stdout.encode()),
                         "stderr_sha256": digest(replay.stderr.encode())}
    temp.replace(destination)
    for name, expected in original_files.items():
        assert digest((ROOT / name).read_bytes()) == expected, "Workspace SHA mismatch"
    link_count = 0
    omitted = set()
    for name in RESULT_FILES:
        if not name.endswith(".md"):
            continue
        for target in re.findall(r"\]\(([^)]+)\)", (OUT / name).read_text()):
            if "://" in target or target.startswith("#"):
                continue
            path = (OUT / target.split("#")[0]).resolve()
            assert path.exists(), f"Missing local evidence link in {name}"
            link_count += 1
            if str(path.relative_to(ROOT)) not in payload:
                omitted.add(target)
    atomic(OUT / "PACKAGE_VALIDATION.json", {
        "at": datetime.now(timezone.utc).isoformat(), **validated,
        "privacy": privacy, "package_bytes": destination.stat().st_size,
        "public_package_engineering_replay": replay_result,
        "workspace_sha256_manifest_passed": True,
        "local_document_links_checked": link_count,
        "historical_references_intentionally_not_bundled": sorted(omitted),
        "manifest_sha256": digest(payload["MANIFEST.sha256.json"]),
        "source_manifest_sha256": digest((OUT / "SHA256SUMS.json").read_bytes()),
        "packager_sha256": digest(Path(__file__).read_bytes()),
        "uploaded_or_published": False})
    print(json.dumps({**validated, "privacy_passed": privacy["passed"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(validate_zip(args.verify_only)))
    else:
        build()
