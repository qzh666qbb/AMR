"""Local FP32 encoder and fresh audited AMR parsing for the fusion prototype."""
from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import warnings

from .amr_extractor import AmrlibBackend, decode_amr_with_audit
from .embedding_amr import GraphObservation, digest, windows
from .parser_recovery_audit import extract_amr_graph, parse_with_recovery_audit


def file_sha(path):
    import hashlib
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def asset_hashes(folder):
    folder = Path(folder)
    result = {str(p.relative_to(folder)): file_sha(p)
              for p in sorted(folder.rglob("*")) if p.is_file()}
    if not result:
        raise ValueError("Empty model asset directory")
    return result


def atomic(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    # Callers create and own the output directory; never append to old runs.
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
        name = f.name
    os.replace(name, path)


def sealed_scores(path):
    """Read only complete fresh-run scores, bound to their code/Bank manifest."""
    path = Path(path)
    run = path.parent
    terminal = json.loads((run / "TERMINAL.json").read_text())
    if path.name != "SCORES.json" or terminal.get("status") != "score_complete":
        raise ValueError("A completed sealed scoring run is required")
    for name, key in (("MANIFEST.json", "manifest_sha256"), ("SCORES.json", "scores_sha256"),
                      ("AUDIT.json", "audit_sha256"), ("OBSERVATION_SEAL.json", "observations_sha256")):
        if file_sha(run / name) != terminal[key]:
            raise ValueError("Score terminal seal mismatch")
    manifest = json.loads((run / "MANIFEST.json").read_text())
    audit = json.loads((run / "AUDIT.json").read_text())
    rows = json.loads(path.read_text())
    if manifest.get("replay_parent_manifest_sha256"):
        raise ValueError("Replay is not fresh calibration/test evidence")
    if (len(rows) != manifest["row_count"] or len(rows) != audit["rows_scored"]
            or len(rows) != audit["rows_expected"]
            or not audit["detector_exact_replay"]
            or any(r["detector_fingerprint"] != manifest["detector_fingerprint"] for r in rows)):
        raise ValueError("Score manifest/audit binding mismatch")
    if (any(r["kind"] == "watermarked" for r in rows)
            and manifest.get("writer_bank_sha256") != manifest["bank_sha256"]):
        raise ValueError("Writer and detector Bank binding mismatch")
    return rows


class LocalEncoder:
    def __init__(self, path, device="cpu"):
        import torch
        from sentence_transformers import SentenceTransformer
        self.assets = asset_hashes(path)
        torch.set_num_threads(2)
        torch.manual_seed(20261104)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.model = SentenceTransformer(str(path), device=device, local_files_only=True, token=False)
        self.model.eval()
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("Encoder must use FP32")
        self.fingerprint = digest({
            "assets": self.assets, "dtype": "float32", "device": device,
            "max_sequence_length": self.model.max_seq_length,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "torch": torch.__version__,
        })
        self.memo = {}
        self.max_replay_error = 0.0

    def fits(self, text):
        return len(self.model.tokenizer(text, truncation=False)["input_ids"]) <= self.model.max_seq_length

    def encode(self, texts):
        import numpy as np
        missing = list(dict.fromkeys(t for t in texts if t not in self.memo))
        if any(not self.fits(t) for t in missing):
            raise ValueError("Encoder context exceeded; no truncation")
        if missing:
            vectors = self.model.encode(missing, batch_size=32, normalize_embeddings=True,
                                        show_progress_bar=False)
            replay = self.model.encode(missing[:1], batch_size=32, normalize_embeddings=True,
                                       show_progress_bar=False)
            error = float(np.max(np.abs(vectors[:1] - replay)))
            if not np.isfinite(vectors).all() or error > 1e-6:
                raise RuntimeError("Encoder finite/deterministic replay failed")
            self.max_replay_error = max(self.max_replay_error, error)
            self.memo.update(zip(missing, vectors))
        return np.stack([self.memo[t] for t in texts])


class FreshParser:
    """Independent exact-input audit; no historical cache import.

    Failed or warning-bearing parses become explicit observations. Unbound
    diagnostics fail the run. Model context limits are checked before inference.
    """
    def __init__(self, model_dir, tokenizer_dir, output, device="cpu", batch_size=4):
        import torch
        self.output = Path(output)
        self.model_assets = asset_hashes(model_dir)
        self.tokenizer_assets = asset_hashes(tokenizer_dir)
        self.backend = AmrlibBackend(str(model_dir), str(tokenizer_dir),
                                    device=device, batch_size=batch_size)
        self.model = self.backend._load()
        self.model.model.eval()
        if any(p.dtype != torch.float32 for p in self.model.model.parameters()):
            raise ValueError("Parser must use FP32")
        torch.manual_seed(20261104)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.fingerprint = digest({
            "model": self.model_assets, "tokenizer": self.tokenizer_assets,
            "device": device, "dtype": "float32", "seed": 20261104,
            "num_ret_seq": self.model.num_ret_seq, "batch_size": batch_size,
            "native_input_limit": self.model.max_sent_len,
            "amrlib": importlib.metadata.version("amrlib"),
            "transformers": importlib.metadata.version("transformers"),
            "torch": torch.__version__, "audit": "all-bound-diagnostics-excluded",
        })
        self.memo = {}
        self.precheck_passed = False

    def _parse(self, text):
        import amrlib.models.parse_xfm.inference as module
        tokens = len(self.model.tokenizer(text, truncation=False)["input_ids"])
        if tokens >= self.model.max_sent_len:
            return GraphObservation(digest(text), None, self.fingerprint, False,
                                    ("parser_native_input_limit",)), {}
        # Single input keeps malformed output/diagnostics bound to one span.
        # Upstream audit errors remain fatal; no bare-parser fallback is used.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outputs, records = parse_with_recovery_audit(self.model, [text], module)
        if caught:
            raise RuntimeError("Unbound Python warning in parser")
        raw = extract_amr_graph(outputs[0], text)
        _, penman_warnings, valid, roundtrip = decode_amr_with_audit(raw)
        audit = records[0]
        reasons = []
        if not valid or not roundtrip:
            reasons.append("invalid_graph_or_roundtrip")
        if penman_warnings or audit["events"]:
            reasons.append("parser_or_penman_diagnostic")
        if audit["selected_beam"] != 0:
            reasons.append("nonzero_selected_beam")
        observation = GraphObservation(digest(text), raw, self.fingerprint, not reasons, tuple(reasons))
        return observation, audit

    def observe(self, text, config, window_config):
        candidates, _ = windows(text, config, window_config)
        for candidate in candidates:
            fragment = candidate["text"]
            key = digest(fragment)
            if key in self.memo:
                continue
            observation, audit = self._parse(fragment)
            if not self.precheck_passed and observation.amr:
                replay, replay_audit = self._parse(fragment)
                if observation != replay or audit != replay_audit:
                    raise RuntimeError("Fresh parser exact replay failed")
                self.precheck_passed = True
            atomic(self.output / (key + ".json"), {"observation": asdict(observation), "audit": audit})
            self.memo[key] = observation
        return [self.memo[digest(c["text"])] for c in candidates]


class SealedObservationProvider:
    """Replay this runner's own observations only, with a complete SHA seal."""
    def __init__(self, run):
        run = Path(run)
        terminal = json.loads((run / "TERMINAL.json").read_text())
        if terminal["status"] != "score_complete":
            raise ValueError("Replay requires a completed scoring run")
        for name, key in (("MANIFEST.json", "manifest_sha256"), ("OBSERVATION_SEAL.json", "observations_sha256"),
                          ("SCORES.json", "scores_sha256"), ("AUDIT.json", "audit_sha256")):
            if file_sha(run / name) != terminal[key]:
                raise ValueError("Replay terminal seal mismatch")
        manifest = json.loads((run / "MANIFEST.json").read_text())
        seal = json.loads((run / "OBSERVATION_SEAL.json").read_text())
        self.fingerprint = manifest["parser_fingerprint"]
        self.records = {}
        for name, sha in seal.items():
            path = (run / name).resolve()
            if not path.is_relative_to(run.resolve()) or file_sha(path) != sha:
                raise ValueError("Observation seal mismatch")
            item = json.loads(path.read_text())["observation"]
            item["reasons"] = tuple(item["reasons"])
            observation = GraphObservation(**item)
            self.records[observation.input_sha256] = observation
        self.precheck_passed = True

    def observe(self, text, config, window_config):
        candidates, _ = windows(text, config, window_config)
        try:
            return [self.records[digest(c["text"])] for c in candidates]
        except KeyError as exc:
            raise ValueError("Sealed replay miss; no parser fallback") from exc
