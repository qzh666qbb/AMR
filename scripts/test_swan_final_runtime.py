"""No-network tests for final SWAN allocation and baseline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import allocate_swan_final_sources as allocation
import run_swan_final_baselines as baselines
from run_segmentation_intervention import text_sha, write_json


def sentence(prefix: str, words: int = 13) -> str:
    body = " ".join(f"{prefix}{index}" for index in range(words - 1))
    return f"{body} finished."


def control_text(prefix: str = "news") -> str:
    return " ".join(sentence(f"{prefix}{index}") for index in range(8))


def tokenize(text: str) -> list[str]:
    return [
        item.strip() + "."
        for item in text.split(".")
        if item.strip()
    ]


class PriorIdentityTest(unittest.TestCase):
    def test_snapshot_collects_declared_text_hash_and_url(self):
        with tempfile.TemporaryDirectory(dir=allocation.ROOT) as directory:
            path = Path(directory) / "source_manifest.json"
            declared = "a" * 64
            write_json(
                path,
                {
                    "source": "A prior source document.",
                    "normalized_source_sha256": declared,
                    "url": "https://example.test/prior",
                    "candidate": "not keyed as source text",
                },
            )
            result = allocation.build_prior_identity_snapshot([path])
        self.assertIn(
            allocation.normalized_hash("A prior source document."),
            result["normalized_text_sha256"],
        )
        self.assertIn(declared, result["normalized_text_sha256"])
        self.assertIn(
            allocation.normalized_url_hash(
                "https://example.test/prior"
            ),
            result["url_sha256"],
        )
        self.assertEqual(result["counts"]["artifacts"], 1)


class HumanControlFilterTest(unittest.TestCase):
    def test_accepts_first_normal_length_eight_sentence_prefix(self):
        result = allocation.human_control_passage(
            control_text(),
            tokenize,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["units"], 8)
        self.assertGreaterEqual(result["words"], 100)
        self.assertLessEqual(result["words"], 220)
        self.assertTrue(all(result["checks"].values()))

    def test_rejects_list_marker_and_duplicate_sentence(self):
        listed = "- " + control_text()
        self.assertIsNone(
            allocation.human_control_passage(listed, tokenize)
        )
        repeated = " ".join([sentence("same")] * 8)
        self.assertIsNone(
            allocation.human_control_passage(repeated, tokenize)
        )


class AllocationOrderTest(unittest.TestCase):
    def test_watermarked_source_scan_excludes_prior_and_preserves_order(self):
        rows = [
            {"text": "Prior source document.", "url": "https://old"},
            {"text": "", "url": ""},
            {"text": "Fresh source alpha with enough words.", "url": "https://a"},
            {"text": "Fresh source beta with enough words.", "url": "https://b"},
        ]

        def stream(index):
            self.assertEqual(index, allocation.SOURCE_START_SHARD)
            yield from rows

        with (
            patch.object(allocation, "SOURCE_COUNT", 2),
            patch.object(allocation, "stream_train_shard", side_effect=stream),
        ):
            selected, screening, _seen_text, _seen_url = (
                allocation._scan_watermarked_sources(
                    {allocation.normalized_hash(rows[0]["text"])},
                    set(),
                )
            )
        self.assertEqual(
            [row["shard_row"] for row in selected],
            [2, 3],
        )
        self.assertEqual(screening["empty_rows"], 1)
        self.assertEqual(screening["prior_or_duplicate_rows"], 1)

    def test_human_controls_start_at_explicit_later_shard(self):
        rows = [
            {
                "text": control_text(f"row{index}"),
                "url": f"https://example.test/{index}",
            }
            for index in range(3)
        ]

        def stream(index):
            self.assertEqual(index, 9)
            yield from rows

        with (
            patch.object(allocation, "HUMAN_COUNT", 3),
            patch.object(allocation, "HUMAN_CALIBRATION_COUNT", 2),
            patch.object(allocation, "stream_train_shard", side_effect=stream),
        ):
            selected, screening = allocation._scan_human_controls(
                9,
                set(),
                set(),
                tokenize,
            )
        self.assertEqual([row["shard_index"] for row in selected], [9] * 3)
        self.assertEqual(
            [row["split"] for row in selected],
            ["calibration", "calibration", "test"],
        )
        self.assertEqual(screening["selected"], 3)


class FakeUtils:
    @staticmethod
    def compute_s2match_score(amr, target):
        del amr, target
        return 0.8


class FakeDetector:
    def __init__(self):
        self.bank = [f"(t / template-{index})" for index in range(50)]
        self.utils = FakeUtils()

    @staticmethod
    def parse(text):
        return [{"text": text, "amr": "(x / accepted)"}]


class RejectingUtils:
    @staticmethod
    def compute_s2match_score(amr, target):
        del amr, target
        return 0.1


class RejectingDetector(FakeDetector):
    def __init__(self):
        super().__init__()
        self.utils = RejectingUtils()


class FakeAPI:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = []

    def request(self, call_id, body):
        self.calls.append((call_id, body))
        content = next(self.contents)
        return {
            "status": "complete",
            "call_sha256": text_sha(call_id),
            "response": {
                "id": call_id,
                "model": "deepseek-flash",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
                "content": content,
                "finish_reason": "stop",
            },
        }


class BaselineCheckpointTest(unittest.TestCase):
    def source(self):
        return {
            "source_id": "wm-000",
            "generation_prompt": "A source prompt.",
        }

    def test_slot_accepts_first_matching_candidate_and_checkpoints(self):
        detector = FakeDetector()
        api = FakeAPI(["A complete generated sentence."])
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(
                baselines,
                "generation_prompt",
                return_value=("system", "user"),
            ):
                result = baselines.run_slot(
                    folder,
                    self.source(),
                    0,
                    [],
                    detector,
                    api,
                    detector.bank,
                    [{}, {}, {}],
                )
                replay = baselines.run_slot(
                    folder,
                    self.source(),
                    0,
                    [],
                    detector,
                    api,
                    detector.bank,
                    [{}, {}, {}],
                )
        self.assertEqual(result, replay)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(len(api.calls), 1)

    def test_baseline_gate_requires_all_eight_single_sentence_slots(self):
        rows = [
            {
                "selected_text": sentence(f"slot{index}", 13),
                "selected_surface_pass": True,
                "accepted": index < 5,
            }
            for index in range(8)
        ]
        text, result = baselines.baseline_gates(rows)
        self.assertTrue(text)
        self.assertTrue(result["mechanically_eligible"])
        rows[-1]["selected_surface_pass"] = False
        self.assertFalse(
            baselines.baseline_gates(rows)[1]["mechanically_eligible"]
        )

    def test_correction_feedback_resets_for_second_template(self):
        corrections = []

        def prompt(_target, _examples, _context, correction=None):
            corrections.append(correction)
            return "system", f"user-{len(corrections)}"

        detector = RejectingDetector()
        api = FakeAPI(
            [f"A complete generated sentence {index}." for index in range(10)]
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                baselines,
                "generation_prompt",
                side_effect=prompt,
            ):
                result = baselines.run_slot(
                    Path(directory),
                    self.source(),
                    0,
                    [],
                    detector,
                    api,
                    detector.bank,
                    [{}, {}, {}],
                )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["attempts"], 10)
        self.assertIsNone(corrections[0])
        self.assertTrue(all(corrections[index] for index in range(1, 5)))
        self.assertIsNone(corrections[5])
        self.assertTrue(all(corrections[index] for index in range(6, 10)))


class CallJournalTest(unittest.TestCase):
    def test_complete_call_replays_without_network_or_new_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            body = {"model": baselines.MODEL, "messages": []}
            path = folder / "calls/example.json"
            write_json(
                path,
                {
                    "call_id": "example",
                    "request_sha256": baselines.digest(body),
                    "status": "complete",
                    "response": {
                        "model": "deepseek-flash",
                        "content": "OK",
                    },
                },
            )
            journal = baselines.PaidCallJournal(
                folder,
                "secret",
                0,
                "deepseek-flash",
            )
            result = journal.request("example", body)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(journal.new_calls, 0)

    def test_uncertain_call_is_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            body = {"model": baselines.MODEL, "messages": []}
            path = folder / "calls/example.json"
            write_json(
                path,
                {
                    "call_id": "example",
                    "request_sha256": baselines.digest(body),
                    "status": "failed_or_uncertain",
                },
            )
            journal = baselines.PaidCallJournal(folder, "secret", 1)
            result = journal.request("example", body)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(journal.new_calls, 0)

    def test_global_cap_is_checked_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_json(folder / "calls/prior.json", {"status": "complete"})
            journal = baselines.PaidCallJournal(folder, "secret", 1)
            with patch.object(baselines, "REMOTE_CALL_CAP", 1):
                with self.assertRaisesRegex(RuntimeError, "global"):
                    journal.request(
                        "new",
                        {"model": baselines.MODEL, "messages": []},
                    )


class BlindPacketTest(unittest.TestCase):
    def test_packet_excludes_generation_and_detection_metadata(self):
        sources = [
            {
                "source_id": f"wm-{index:03d}",
                "generation_prompt": f"Prompt {index}.",
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for source in sources:
                write_json(
                    folder / "baselines" / f'{source["source_id"]}.json',
                    {
                        "source_id": source["source_id"],
                        "text": f'Candidate {source["source_id"]}.',
                        "text_sha256": text_sha(
                            f'Candidate {source["source_id"]}.'
                        ),
                        "mechanical": {
                            "mechanically_eligible": True,
                        },
                    },
                )
            with patch.object(
                baselines,
                "check_allocations",
                return_value=({"sources": sources}, {}),
            ):
                packet = baselines.create_quality_packet(folder)
        encoded = json.dumps(packet["rows"])
        for forbidden in (
            "source_id",
            "target_score",
            "acceptance",
            "parser_output",
            "S2Match",
            "green",
            '"z"',
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(len(packet["rows"]), 2)


class CostEstimateTest(unittest.TestCase):
    def test_estimate_uses_full_frozen_request_cap(self):
        result = baselines.make_cost_estimate(0.006, 0.30, 1.20)
        self.assertEqual(result["request_budget"]["total"], 20417)
        self.assertEqual(result["currency"], "USD")
        self.assertGreater(
            result["estimated_usd"][
                "full_cap_observed_max_per_field"
            ],
            result["estimated_usd"]["full_cap_mean_usage"],
        )


if __name__ == "__main__":
    unittest.main()
