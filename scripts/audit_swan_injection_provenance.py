"""Compare local injection logic to the pinned upstream source without executing it."""

import ast
from datetime import datetime, timezone
import difflib
import json

import requests

from run_segmentation_intervention import ROOT, sha, text_sha, write_json

COMMIT = "2ca0c2071c8d0c6bb3739b2b2bcfb34cfe63eb2a"
URL = f"https://raw.githubusercontent.com/amazon-science/SWAN/{COMMIT}/injection/injection_amr_utils.py"
DESTINATION = ROOT / "experiments/swan_sentence_growth1/injection_provenance.json"


def functions(source):
    return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def compare(upstream, local):
    left, right = functions(upstream), functions(local)
    result = {}
    for name in ("build_system_prompt", "build_user_prompt", "amr_instruction_completion"):
        before, after = ast.get_source_segment(upstream, left[name]), ast.get_source_segment(local, right[name])
        result[name] = {
            "ast_equal": ast.dump(left[name]) == ast.dump(right[name]),
            "upstream_source": before, "local_source": after,
            "diff": "\n".join(difflib.unified_diff(
                before.splitlines(), after.splitlines(), fromfile=f"upstream/{name}", tofile=f"local/{name}", lineterm="")),
        }
    return result


def main():
    local_path = ROOT / "third_party/SWAN/injection/injection_amr_utils.py"
    if DESTINATION.exists():
        saved = json.loads(DESTINATION.read_text())
        if saved["local_sha256"] != sha(local_path) or saved["upstream_url"] != URL:
            raise ValueError("Provenance evidence changed")
    else:
        response = requests.get(URL, timeout=(20, 60))
        response.raise_for_status()
        source = response.text
        saved = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "upstream_commit": COMMIT, "upstream_url": URL,
            "upstream_sha256": text_sha(source), "local_sha256": sha(local_path),
            "upstream_source": source, "comparison": compare(source, local_path.read_text()),
            "scope": "Pinned source comparison, not an executed upstream reproduction",
        }
        write_json(DESTINATION, saved)
    print(json.dumps({
        "upstream_commit": COMMIT, "upstream_sha256": saved["upstream_sha256"],
        "local_sha256": saved["local_sha256"],
        "function_ast_equal": {k: v["ast_equal"] for k, v in saved["comparison"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
