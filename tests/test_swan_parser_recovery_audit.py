import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import penman
from penman.models.amr import model as amr_model

from src.core.parser_recovery_audit import (
    capture_recovery_logs, extract_amr_graph, parse_with_recovery_audit,
)
from scripts.run_multiscale_transfer import Parser, digest


class ParserRecoveryAuditTests(unittest.TestCase):
    def fake_parser(self, text, graph="(x / test)\n  ", tamper=False, warning=False,
                    fallback=False, clipped=False):
        class Deserializer:
            def __init__(self, raw):
                self.graph = None if raw == "bad" else graph
                if warning:
                    logging.getLogger("amrlib.fake").error("bound test error")

            def get_graph_string(self):
                return self.graph

        module = SimpleNamespace(PenmanDeSerializer=Deserializer)
        parser = Parser.__new__(Parser)

        class Model:
            ret_raw_gen = False
            num_ret_seq = 2
            max_sent_len = 1024

            def tokenizer(self, text, **kwargs):
                return {"input_ids": [1, 2, 3]}

            def parse_sents(self, texts):
                outputs = []
                for text in texts:
                    if fallback:
                        module.PenmanDeSerializer("bad")
                    body = module.PenmanDeSerializer("good").get_graph_string()
                    outputs.append("# ::snt " + text + "\n" +
                                   (body.replace("test", "other") if tamper else body))
                parser.beam_lengths.append({"length": 1024 if clipped else 12,
                                            "eos": not clipped})
                return outputs

        parser.model = Model()
        parser.inference = module
        parser.capture = capture_recovery_logs
        parser.extract_graph = extract_amr_graph
        parser.penman, parser.amr_model = penman, amr_model
        parser.out, parser.capacity_freeze = Path(tempfile.gettempdir()), {}
        parser.freeze_sha, parser.beam_lengths = "fixture", []
        parser.calls, parser.cache_hits = 0, 0
        return parser

    def test_exact_metadata_graph_bytes_and_downstream_roundtrip(self):
        for text in ("single", "first\nsecond", "first\r\nsecond", "first\rsecond",
                     "first\n# ::snt forged\n(z / fake)", " trailing \n\t",
                     "unicode\u2028separator"):
            with self.subTest(text=text):
                parser = self.fake_parser(text)
                record = parser.fresh(text)
                self.assertTrue(record["valid"])
                self.assertEqual(record["amr"], "(x / test)\n  ")
                self.assertEqual(record["amr_sha256"], record["attempts"][0]["graph_sha256"])
                self.assertEqual(record["upstream_output_sha256"],
                                 digest("# ::snt " + text + "\n(x / test)\n  "))
                outputs, audit = parse_with_recovery_audit(parser.model, [text], parser.inference)
                self.assertEqual(outputs[0], "# ::snt " + text + "\n(x / test)\n  ")
                self.assertEqual(audit[0]["selected_beam"], 0)

    def test_real_graph_mismatch_still_halts_both_paths(self):
        parser = self.fake_parser("first\nsecond", tamper=True)
        with self.assertRaisesRegex(RuntimeError, "differs from accepted beam"):
            parser.fresh("first\nsecond")
        with self.assertRaisesRegex(RuntimeError, "differs from returned AMR"):
            parse_with_recovery_audit(parser.model, ["first\nsecond"], parser.inference)

    def test_metadata_mismatch_rejected_and_bare_graph_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "metadata differs"):
            extract_amr_graph("# ::snt wrong\n(x / test)", "right")
        self.assertEqual(extract_amr_graph("(x / test)\r\n", "right"), "(x / test)\r\n")

    def test_recovery_and_clipping_eligibility_preserved(self):
        for options in ({"warning": True}, {"clipped": True}):
            parser = self.fake_parser("first\nsecond", **options)
            self.assertFalse(parser.fresh("first\nsecond")["valid"])
        parser = self.fake_parser("first\nsecond", fallback=True)
        record = parser.fresh("first\nsecond")
        self.assertEqual([a["beam_index"] for a in record["attempts"] if a["accepted"]], [1])
        from scripts.semantic_fact_raw_amr import audit_raw
        record["case_id"] = "test"
        audit = audit_raw(record, {"case_id": "test", "input_text_sha256": record["text_sha256"]},
                          "fixture")
        self.assertFalse(audit["structurally_eligible"])

    def test_legacy_cache_cannot_silently_resume(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            parser = self.fake_parser("first")
            parser.out = Path(directory)
            path = parser.out / "private/parses" / (digest("first") + ".json")
            path.parent.mkdir(parents=True)
            record = parser.fresh("first")
            del record["parser_output_version"]
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, "new frozen runtime"):
                parser.parse("first")

    def test_failed_beam_and_recovery_events_bind_to_correct_span(self):
        class Deserializer:
            def __init__(self, raw):
                if raw == "bad":
                    logging.getLogger("amrlib.fake").error("Recovered failed candidate")
                    self.graph = None
                else:
                    logging.getLogger("amrlib.fake").warning("Recoverable layout event")
                    self.graph = "(x / test)"

            def get_graph_string(self):
                return self.graph

        module = SimpleNamespace(PenmanDeSerializer=Deserializer)

        class Model:
            ret_raw_gen = False
            num_ret_seq = 2

            def parse_sents(self, texts):
                outputs = []
                for i, text in enumerate(texts):
                    for beam in (["bad", "good"] if i == 0 else ["good"]):
                        graph = module.PenmanDeSerializer(beam).get_graph_string()
                        if graph is not None:
                            outputs.append("# ::snt " + text + "\n" + graph)
                            break
                return outputs

        outputs, audit = parse_with_recovery_audit(Model(), ["first", "second"], module)
        self.assertEqual(outputs, ("# ::snt first\n(x / test)", "# ::snt second\n(x / test)"))
        self.assertEqual(audit[0]["selected_beam"], 1)
        self.assertEqual(audit[1]["selected_beam"], 0)
        self.assertEqual(audit[0]["events"][0]["level"], "ERROR")
        self.assertTrue(all(e["span_index"] == 0 for e in audit[0]["events"]))
        self.assertIs(module.PenmanDeSerializer, Deserializer)

    def test_unbound_recovery_log_rejects_audit(self):
        class Deserializer:
            def __init__(self, raw):
                pass

            def get_graph_string(self):
                return "(x / test)"

        module = SimpleNamespace(PenmanDeSerializer=Deserializer)

        class Model:
            ret_raw_gen = False
            num_ret_seq = 1

            def parse_sents(self, texts):
                logging.getLogger("amrlib.fake").warning("Unbound parser diagnostic")
                graph = module.PenmanDeSerializer("good").get_graph_string()
                return ["# ::snt first\n" + graph]

        with self.assertRaises(RuntimeError):
            parse_with_recovery_audit(Model(), ["first"], module)
        self.assertIs(module.PenmanDeSerializer, Deserializer)

    def test_all_failed_beams_require_complete_bound_erasure(self):
        class Deserializer:
            def __init__(self, raw):
                self.raw = raw

            def get_graph_string(self):
                return None if self.raw == "bad" else "(x / test)"

        module = SimpleNamespace(PenmanDeSerializer=Deserializer)

        class Model:
            ret_raw_gen = False
            num_ret_seq = 2

            def parse_sents(self, texts):
                for _ in range(2):
                    module.PenmanDeSerializer("bad").get_graph_string()
                graph = module.PenmanDeSerializer("good").get_graph_string()
                return [None, "# ::snt second\n" + graph]

        with self.assertRaisesRegex(RuntimeError, "no unique accepted beam"):
            parse_with_recovery_audit(Model(), ["first", "second"], module)
        outputs, audit = parse_with_recovery_audit(
            Model(), ["first", "second"], module, allow_all_failed=True)
        self.assertIsNone(outputs[0])
        self.assertTrue(audit[0]["all_beams_failed"])
        self.assertIsNone(audit[0]["selected_beam"])
        self.assertEqual(audit[1]["selected_beam"], 0)
        self.assertEqual(audit[1]["attempts"][0]["span_index"], 1)


if __name__ == "__main__":
    unittest.main()
