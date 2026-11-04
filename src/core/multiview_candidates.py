"""候选只来自待测文本；排序不读取模板分数或原文标签。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from .sqlite_cache import text_sha256


@dataclass(frozen=True)
class WindowConfig:
    lengths: tuple = (12, 24, 48)
    stride: int = 6
    max_words: int = 80
    max_candidates: int = 128

    def __post_init__(self):
        if (not self.lengths or any(n < 1 for n in self.lengths)
                or min(self.stride, self.max_words, self.max_candidates) < 1):
            raise ValueError("候选参数必须为正")


def candidate_windows(text: str, config: WindowConfig = WindowConfig()) -> dict:
    tokens = list(re.finditer(r"\S+", text))
    spans = {}

    def add(left, right, kind):
        fragment = text[left:right]
        trimmed = fragment.strip()
        if not trimmed or len(trimmed.split()) > config.max_words:
            return
        left += len(fragment) - len(fragment.lstrip())
        right = left + len(trimmed)
        spans.setdefault((left, right), set()).add(kind)

    add(0, len(text), "whole_text")
    start = 0
    for boundary in re.finditer(r"[.!?;]\s+|\n+", text):
        add(start, boundary.start() + 1, "surface_boundary")
        start = boundary.end()
    add(start, len(text), "surface_boundary")
    for size in config.lengths:
        if not tokens:
            continue
        starts = set(range(0, max(1, len(tokens) - size + 1), config.stride))
        starts.add(max(0, len(tokens) - size))
        for offset in sorted(starts):
            end = min(len(tokens), offset + size)
            add(tokens[offset].start(), tokens[end - 1].end(), f"word_window_{size}")
    ordered = sorted(spans, key=lambda key: (
        0 if "whole_text" in spans[key] else 1 if "surface_boundary" in spans[key] else 2,
        key[0], key[1],
    ))
    total = len(ordered)
    ordered = ordered[:config.max_candidates]
    candidates = [{
        "start_char": left, "end_char": right, "text": text[left:right],
        "span_sha256": text_sha256(text[left:right]),
        "word_count": len(text[left:right].split()),
        "kinds": sorted(spans[(left, right)]),
    } for left, right in ordered]
    return {"candidates": candidates, "enumerated_count": total,
            "truncated_count": total - len(candidates)}


def select_budget(candidates: list, budget: int = 0) -> list:
    """长度层内按位置交错取样；0表示全候选开发，不等于正式24预算。"""
    if budget < 0:
        raise ValueError("预算不能为负")
    if not budget or len(candidates) <= budget:
        return list(candidates)
    strata = {}
    for index, candidate in enumerate(candidates):
        words = candidate["word_count"]
        layer = 0 if words <= 12 else 1 if words <= 24 else 2 if words <= 48 else 3
        strata.setdefault(layer, []).append(index)
    queues = []
    for layer in sorted(strata):
        indices = sorted(strata[layer], key=lambda i: (
            candidates[i]["start_char"], candidates[i]["end_char"]))
        # 依次取中点、两端、剩余子区间中点，避免集中在文章开头。
        queue, intervals = [], [(0, len(indices))]
        while intervals:
            left, right = intervals.pop(0)
            if left >= right:
                continue
            middle = (left + right) // 2
            queue.append(indices[middle])
            intervals.extend([(left, middle), (middle + 1, right)])
        queues.append(queue)
    selected = []
    while len(selected) < budget:
        for queue in queues:
            if queue and len(selected) < budget:
                selected.append(queue.pop(0))
    return [candidates[i] for i in sorted(selected)]


def config_dict(config: WindowConfig = WindowConfig()) -> dict:
    value = asdict(config)
    value["lengths"] = list(value["lengths"])
    return value
