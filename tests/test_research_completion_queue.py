"""Subprocess tests: consumers record launch only after verified dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import run_research_completion_queue as queue


WORKER = '''
import hashlib, json, pathlib, sys
out, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
count = out / "launches"
n = int(count.read_text()) + 1 if count.exists() else 1
count.write_text(str(n))
if mode in ("fail", "pause", "resource"):
    sys.exit({"fail": 1, "pause": 76, "resource": 75}[mode])
if mode == "retry" and n == 1:
    sys.exit(75)
if mode == "empty":
    sys.exit(0)
def write(name, value):
    p = out / name
    p.write_text(json.dumps(value))
    return hashlib.sha256(p.read_bytes()).hexdigest()
r = write("RESULT.json", {"status": "COMPLETE"})
s = write("SEAL.json", {"result_sha256": r})
if mode != "missing_terminal":
    write("TERMINAL.json", {"status": "BAD" if mode == "bad_status" else "COMPLETE",
                           "result_sha256": r, "seal_sha256": s})
'''


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.worker = self.root / "worker.py"
        self.worker.write_text(WORKER)

    def job(self, name, mode="success", requires=()):
        out = self.root / name
        out.mkdir()
        return {
            "id": name, "argv": [sys.executable, str(self.worker), str(out), mode],
            "cwd": str(self.root), "code_sha256": {str(self.worker): queue.sha(self.worker)},
            "requires": list(requires),
            "completion": {
                "terminal": {"path": str(out / "TERMINAL.json"),
                             "equals": {"status": "COMPLETE"},
                             "sha256_fields": {"result_sha256": "result", "seal_sha256": "seal"}},
                "result": {"path": str(out / "RESULT.json"), "equals": {"status": "COMPLETE"}},
                "seal": {"path": str(out / "SEAL.json"),
                         "sha256_fields": {"result_sha256": "result"}},
            },
        }

    def manifest(self, name, jobs):
        out = self.root / name
        out.mkdir()
        path = out / "MANIFEST.json"
        path.write_text(json.dumps({"schema": queue.VERSION, "jobs": jobs,
                                    "resource_retry": {"max_retries": 1, "delay_seconds": 0}}))
        return path

    def run_queue(self, manifest):
        run = subprocess.run([sys.executable, queue.__file__, str(manifest)],
                             capture_output=True, text=True, timeout=15)
        return run.returncode, queue.read(manifest.parent / "QUEUE_STATUS.json")

    def external(self, manifest, name):
        return {"manifest": str(manifest), "manifest_sha256": queue.sha(manifest), "job_id": name}

    def test_failed_parent_blocks_child_but_independent_job_runs(self):
        path = self.manifest("q", [
            self.job("parent", "fail"), self.job("child", requires=[{"job_id": "parent"}]),
            self.job("independent")])
        rc, status = self.run_queue(path)
        self.assertEqual(rc, 1)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["jobs"], {"parent": "failed", "child": "blocked",
                                          "independent": "success"})
        self.assertFalse((self.root / "child/launches").exists())

    def test_zero_exit_requires_valid_complete_artifacts(self):
        for mode in ("empty", "missing_terminal", "bad_status"):
            with self.subTest(mode=mode):
                path = self.manifest("q_" + mode, [self.job(mode, mode)])
                rc, status = self.run_queue(path)
                self.assertEqual(rc, 1)
                self.assertEqual(status["jobs"][mode], "failed")
                self.assertEqual(queue.read(queue.receipt_path(path, mode))["returncode"], 0)

    def test_success_dependencies_and_resume_do_not_relaunch(self):
        path = self.manifest("q", [self.job("parent"),
                                  self.job("child", requires=[{"job_id": "parent"}])])
        self.assertEqual(self.run_queue(path)[0], 0)
        before = queue.sha(queue.receipt_path(path, "parent"))
        self.assertEqual(self.run_queue(path)[0], 0)
        self.assertEqual(queue.sha(queue.receipt_path(path, "parent")), before)
        for name in ("parent", "child"):
            self.assertEqual((self.root / name / "launches").read_text(), "1")

    def test_tampered_success_invalidates_resume_and_blocks_external_child(self):
        parent = self.manifest("p", [self.job("producer")])
        self.assertEqual(self.run_queue(parent)[0], 0)
        (self.root / "producer/RESULT.json").write_text('{"status": "COMPLETE", "changed": true}')
        rc, status = self.run_queue(parent)
        self.assertEqual((rc, status["jobs"]["producer"]), (1, "invalidated"))
        child = self.manifest("c", [self.job("consumer",
                                           requires=[self.external(parent, "producer")])])
        self.assertEqual(self.run_queue(child)[1]["jobs"]["consumer"], "blocked")
        self.assertFalse((self.root / "consumer/launches").exists())

    def test_external_missing_failed_stale_and_wrong_receipt_block(self):
        for variant in ("missing", "failed", "stale", "receipt"):
            with self.subTest(variant=variant):
                job_id = "p_" + variant
                parent = self.manifest("qp_" + variant,
                                       [self.job(job_id, "fail" if variant == "failed" else "success")])
                dep = self.external(parent, job_id)
                if variant != "missing":
                    self.run_queue(parent)
                if variant == "stale":
                    parent.write_text(parent.read_text() + "\n")
                if variant == "receipt":
                    dep["terminal_sha256"] = "0" * 64
                child_id = "c_" + variant
                child = self.manifest("qc_" + variant, [self.job(child_id, requires=[dep])])
                self.assertEqual(self.run_queue(child)[1]["jobs"][child_id], "blocked")
                self.assertFalse((self.root / child_id / "launches").exists())

    def test_successful_external_dependency_hashes_are_bound(self):
        parent = self.manifest("p", [self.job("producer")])
        self.assertEqual(self.run_queue(parent)[0], 0)
        child = self.manifest("c", [self.job("consumer",
                                           requires=[self.external(parent, "producer")])])
        self.assertEqual(self.run_queue(child)[0], 0)
        record = queue.read(queue.receipt_path(child, "consumer"))
        self.assertEqual(record["dependencies"][0]["terminal_sha256"],
                         queue.sha(queue.receipt_path(parent, "producer")))

    def test_pause76_blocks_dependents_without_retrying_or_blocking_independents(self):
        path = self.manifest("q", [self.job("paused", "pause"),
                                  self.job("child", requires=[{"job_id": "paused"}]),
                                  self.job("independent")])
        rc, status = self.run_queue(path)
        self.assertEqual((rc, status["status"]), (76, "paused"))
        self.assertEqual(status["jobs"]["child"], "blocked")
        self.assertEqual(status["jobs"]["independent"], "success")
        self.assertEqual((self.root / "paused/launches").read_text(), "1")

    def test_resource75_retries_then_succeeds_or_pauses(self):
        path = self.manifest("q", [self.job("retry", "retry"), self.job("resource", "resource")])
        rc, status = self.run_queue(path)
        self.assertEqual(rc, 76)
        self.assertEqual(status["jobs"], {"retry": "success", "resource": "paused"})
        for name in ("retry", "resource"):
            self.assertEqual((self.root / name / "launches").read_text(), "2")

    def test_stale_artifact_and_interrupted_start_prevent_execution(self):
        for name in ("stale", "interrupted"):
            job = self.job(name)
            path = self.manifest("q_" + name, [job])
            if name == "stale":
                (self.root / name / "RESULT.json").write_text("{}")
            else:
                (path.parent / (name + "_STARTED.json")).write_text("{}")
            self.assertEqual(self.run_queue(path)[1]["jobs"][name], "failed")
            self.assertFalse((self.root / name / "launches").exists())

    def test_code_change_prevents_launch(self):
        path = self.manifest("q", [self.job("job")])
        self.worker.write_text(WORKER + "\n")
        self.assertEqual(self.run_queue(path)[1]["jobs"]["job"], "failed")
        self.assertFalse((self.root / "job/launches").exists())

    def test_old_schema_and_forward_dependencies_rejected(self):
        path = self.manifest("q", [self.job("job")])
        plan = queue.read(path)
        plan["jobs"][0]["requires"] = [{"job_id": "not_yet_run"}]
        path.write_text(json.dumps(plan))
        with self.assertRaises(queue.EvidenceError):
            queue.load_plan(path)
        path.write_text(json.dumps({"predecessor_lock": "/tmp/legacy.lock", "jobs": []}))
        with self.assertRaises(queue.EvidenceError):
            queue.load_plan(path)


if __name__ == "__main__":
    unittest.main()
