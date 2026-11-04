"""开发版结构恢复器；未校准时只输出统计量，不给水印判定。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

from .bank_scorer import _multiset_f1, template_features
from .multiview_alignment import AlignmentConfig, align
from .multiview_views import extract_events
from .sqlite_cache import normalize_text, text_sha256


VERSION = "swan-multiview:development-v1"


@dataclass(frozen=True)
class DetectorConfig:
    score_center: float = 0.65
    warning_policy: str = "exclude_selected"
    alignment_mode: str = "predicted"
    weighting: str = "rule"

    def __post_init__(self):
        if (not 0 <= self.score_center <= 1
                or self.warning_policy not in ("exclude_selected", "include_repaired")
                or self.alignment_mode not in ("predicted", "position", "semantic")
                or self.weighting not in ("rule", "fixed")):
            raise ValueError("检测参数无效")


def full_bank_scores(scorer, amr):
    features = template_features(amr)
    scores = []
    for index, reference in enumerate(scorer._bank_features):
        components = (
            _multiset_f1(features.concepts, reference.concepts),
            _multiset_f1(features.relations, reference.relations),
            _multiset_f1(features.paths, reference.paths))
        score = sum(weight * value for weight, value in zip(
            (scorer.concept_weight, scorer.relation_weight, scorer.path_weight), components))
        scores.append({"template_index": index, "score": score,
                       "concept_score": components[0], "relation_score": components[1],
                       "path_score": components[2]})
    # 检查保存的全分数与旧scorer口径一致。
    best = max(scores, key=lambda value: (value["score"], -value["template_index"]))
    original = scorer.score_amr(amr)
    if best["template_index"] != original.template_index or abs(best["score"] - original.score) > 1e-12:
        raise RuntimeError("完整Bank分数与原scorer不一致")
    return scores


class MultiviewDetector:
    def __init__(self, scorer, audited_backend, encoder, surface_analyzer,
                 config=DetectorConfig(), alignment_config=AlignmentConfig()):
        self.scorer, self.backend, self.encoder = scorer, audited_backend, encoder
        self.surface = surface_analyzer
        self.config, self.alignment_config = config, alignment_config
        payload = {
            "version": VERSION, "config": asdict(config),
            "alignment_config": asdict(alignment_config),
            "bank_scorer": scorer.fingerprint, "parser": audited_backend.fingerprint,
            "encoder": encoder.fingerprint, "surface": surface_analyzer.fingerprint,
        }
        self.fingerprint = VERSION + ":" + hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def detect(self, text: str, candidates: list, records: dict) -> dict:
        observations, events = [], []
        for index, candidate in enumerate(candidates):
            if (text[candidate["start_char"]:candidate["end_char"]] != candidate["text"]
                    or text_sha256(candidate["text"]) != candidate["span_sha256"]):
                raise ValueError("候选偏移或文本哈希不一致")
            normalized = normalize_text(candidate["text"])
            if normalized not in records:
                raise RuntimeError("缺少候选缓存；不能静默调用解析器")
            record = records[normalized]
            self.backend.assert_record(record)
            audit = self.backend.records[record.text_sha256]
            selected_events = [event for event in audit["events"]
                               if event["beam_index"] in (None, audit["selected_beam"])]
            reasons = []
            if not record.graph_valid or not record.roundtrip_valid:
                reasons.append("invalid_graph_or_roundtrip")
            if record.penman_warnings:
                reasons.append("penman_warning")
            if any(event["level"] in ("ERROR", "CRITICAL") for event in selected_events):
                reasons.append("selected_beam_error")
            if (self.config.warning_policy == "exclude_selected"
                    and any(event["level"] == "WARNING" for event in selected_events)):
                reasons.append("selected_beam_warning")
            scores = full_bank_scores(self.scorer, record.amr) if (
                record.graph_valid and record.roundtrip_valid) else []
            observation = {
                "candidate_index": index, **candidate, "amr": record.amr,
                "amr_sha256": audit["amr_output_sha256"],
                "graph_body_sha256": hashlib.sha256("\n".join(
                    line for line in record.amr.splitlines() if not line.startswith("#")
                ).encode()).hexdigest(),
                "parser_fingerprint": record.parser_fingerprint,
                "graph_valid": record.graph_valid, "roundtrip_valid": record.roundtrip_valid,
                "selected_beam": audit["selected_beam"], "parser_events": audit["events"],
                "penman_warnings": list(record.penman_warnings), "usable": not reasons,
                "unusable_reasons": reasons, "bank_scores": scores,
                "top2": sorted(scores, key=lambda row: (-row["score"], row["template_index"]))[:2],
            }
            extracted = extract_events(record.amr, index) if not reasons else []
            observation["event_ids"] = [event.event_id for event in extracted]
            observations.append(observation)
            events.extend(extracted)
        surfaces = self.surface.analyze_many([candidate["text"] for candidate in candidates])
        text_vectors = self.encoder.encode([candidate["text"] for candidate in candidates])
        event_vectors = self.encoder.encode([event.description for event in events])
        alignment = align(candidates, events, text_vectors, event_vectors, surfaces,
                          self.alignment_config, self.config.alignment_mode)
        # 一个片段可定位多个事件；水印分数仍取实际事件所属的原图。
        # 不把跨窗口关系拼成图，也不把同一张图计多次。
        by_graph = {}
        for edge in alignment["selected"]:
            index = edge["event_candidate_index"]
            observation = observations[index]
            group_key = observation["graph_body_sha256"]
            by_graph.setdefault(group_key, []).append(edge)
        evidence = []
        for graph_sha, edges in sorted(by_graph.items()):
            selected = max(edges, key=lambda edge: (edge["rule_weight"], -edge["event_candidate_index"]))
            observation = observations[selected["event_candidate_index"]]
            weight = 1.0 if self.config.weighting == "fixed" else selected["rule_weight"]
            score = observation["top2"][0]["score"]
            evidence.append({
                "graph_body_sha256": graph_sha, "amr_sha256": observation["amr_sha256"],
                "candidate_index": observation["candidate_index"],
                "group_ids": sorted({edge["group_id"] for edge in edges}),
                "support_candidate_indices": sorted({edge["candidate_index"] for edge in edges}),
                "weight": weight, "bank_score": score,
                "template_index": observation["top2"][0]["template_index"],
                "utility": weight * (score - self.config.score_center),
            })
        raw = sum(item["utility"] for item in evidence)
        denominator = math.sqrt(sum(item["weight"] ** 2 for item in evidence))
        statistic = raw / denominator if denominator else 0.0
        return {
            "version": VERSION, "fingerprint": self.fingerprint,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "config": asdict(self.config), "candidate_count": len(candidates),
            "usable_candidates": sum(row["usable"] for row in observations),
            "observations": observations, "surface_observations": surfaces,
            "events": [event.to_dict() for event in events], "alignment": alignment,
            "evidence": evidence, "statistic": statistic, "raw_utility": raw,
            "evidence_count": len(evidence), "is_watermarked": None, "p_value": None,
            "status": "uncalibrated_development_diagnostic",
            "learned_reliability": False, "new_fpr_tpr_measured": False,
        }
