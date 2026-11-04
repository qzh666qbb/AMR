"""Preserve original freeze; bind missing package imports in a deployment supplement."""
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_swan_independent_banks1 as run
import run_multiscale_transfer as rt

out = run.OUT
f = run.verify(out)
original = rt.sha(out / "FREEZE.json")
supplement = out / "RUNTIME_DEPENDENCY_COMPLETION.json"
if supplement.exists():
    raise RuntimeError("Dependency completion already sealed")
added = {}
for src in sorted((ROOT / "src/core").glob("*.py")):
    relative = str(src.relative_to(ROOT))
    destination = out / "private/code" / relative
    if destination.exists():
        if rt.sha(destination) != rt.sha(src):
            raise RuntimeError("Existing frozen code differs: " + relative)
        continue
    shutil.copyfile(src, destination)
    added[relative] = rt.sha(destination)
run.immutable(out / "DEPLOYMENT_ATTEMPT1.json", {
    "status": "PREEXECUTION_IMPORT_FAILURE", "parser_calls": 0, "detector_calls": 0,
    "error": "ModuleNotFoundError: No module named 'src.core.amr_extractor'",
    "original_freeze_sha256": original, "old_remote_directory_retained": True,
    "interpretation": "Packaging dependency failure; no method performance evidence",
})
run.immutable(supplement, {
    "version": "preexecution-runtime-dependency-completion/v1", "at": rt.now(),
    "original_freeze_sha256": original, "added_code_sha256": added,
    "existing_frozen_files_modified": False, "scientific_input_or_model_changed": False,
    "original_parser_calls": 0, "reason": "src.core package __init__ imports missing sibling modules",
    "deployment": "new runtime directory; queue verifies all original and added modules",
    "preparation_script_sha256": rt.sha(Path(__file__)),
})
run.verify(out)
archive = Path("/tmp/swan-independent-banks1-runtime2-20261102.tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    for p in sorted(out.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.name != "source.lock":
            tar.add(p, arcname=str(p.relative_to(out)), recursive=False)
print(json.dumps({"original_freeze_sha256": original, "additional_modules": len(added),
                  "supplement_sha256": rt.sha(supplement), "archive_sha256": rt.sha(archive),
                  "archive_bytes": archive.stat().st_size}))
