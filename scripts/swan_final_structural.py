"""Frozen structural transformations for the final SWAN segmentation audit."""

from __future__ import annotations

import math
import random
import re


_ALPHA_TOKEN = re.compile(r"[a-z0-9']+")
_ALPHA_RE = re.compile(r"[A-Za-z]")
_CLOSING_RE = re.compile(r"(['\"\)\]]*)$")
_END_PUNCT_RE = re.compile(r"[.!?]+(?=(?:['\"\)\]]*)$)")
_TERMINAL = ".!?"
_TERMINAL_RE = re.compile(r"([.!?]+)([\"')\]]*)$")


def sentences(text: str) -> list[str]:
    """Use the same NLTK sentence backend as the SWAN detector."""
    import nltk

    return [item.strip() for item in nltk.sent_tokenize(text) if item.strip()]


def lexical_tokens(text: str) -> list[str]:
    return _ALPHA_TOKEN.findall(text.casefold())


def _without_terminal(text: str) -> str:
    return _TERMINAL_RE.sub(r"\2", text.rstrip())


def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


def _upper_first(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _terminate(text: str) -> str:
    stripped = text.rstrip()
    if not stripped:
        return stripped
    return stripped if stripped[-1] in _TERMINAL else stripped + "."


def _strip_terminal_punctuation(text: str) -> str:
    return _END_PUNCT_RE.sub("", text)


def _add_period(text: str) -> str:
    stripped = _strip_terminal_punctuation(text)
    match = _CLOSING_RE.search(stripped)
    position = match.start() if match else len(stripped)
    return f"{stripped[:position]}.{stripped[position:]}"


def _upper_first_alpha(text: str) -> str:
    return _ALPHA_RE.sub(lambda match: match.group(0).upper(), text, count=1)


def _boundary_priority(count: int) -> list[int]:
    """Return a deterministic, nested, approximately even removal order."""
    order: list[int] = []
    queue = [(0, count)]
    while queue:
        start, stop = queue.pop(0)
        if start >= stop:
            continue
        middle = (start + stop) // 2
        order.append(middle)
        queue.append((start, middle))
        queue.append((middle + 1, stop))
    return order


def strict_semicolon_merge(text: str, fraction: float) -> str:
    """Replace a nested fraction of sentence boundaries with semicolons."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    source = sentences(text)
    if len(source) < 2 or fraction == 0:
        return text
    boundary_count = len(source) - 1
    remove_count = int(fraction * boundary_count + 0.5)
    removed = set(_boundary_priority(boundary_count)[:remove_count])
    parts: list[str] = []
    for index, sentence in enumerate(source):
        if index < boundary_count and index in removed:
            parts.append(_without_terminal(sentence) + ";")
        else:
            parts.append(sentence)
    result = " ".join(parts)
    if lexical_tokens(result) != lexical_tokens(text):
        raise ValueError("Strict merge changed lexical tokens")
    return result


def sword_merge_adjacent(
    text: str,
    ratio: float,
    rng: random.Random,
) -> str:
    """Minimal adaptation of SwordStamp's pinned MIT-licensed atom."""
    source = sentences(text)
    output: list[str] = []
    index = 0
    while index < len(source):
        if index + 1 < len(source) and rng.random() < ratio:
            head = source[index].rstrip().rstrip(_TERMINAL).rstrip()
            output.append(f"{head}, {_lower_first(source[index + 1])}")
            index += 2
        else:
            output.append(source[index])
            index += 1
    return " ".join(output)


def _evenly_spaced_indices(size: int, count: int) -> list[int]:
    if count < 0 or count > size:
        raise ValueError(f"count must be in [0, {size}], got {count}")
    if count == 0:
        return []
    return [
        min(size - 1, int((index + 0.5) * size / count))
        for index in range(count)
    ]


def _render_boundary_exchange(
    tokens: list[str],
    old_boundaries: list[int],
    new_boundaries: list[int],
) -> str:
    output = list(tokens)
    old, new = set(old_boundaries), set(new_boundaries)
    for boundary in old - new:
        output[boundary - 1] = _strip_terminal_punctuation(
            output[boundary - 1]
        )
    for boundary in new - old:
        output[boundary - 1] = _add_period(output[boundary - 1])
    for start in [0, *sorted(new)]:
        if start < len(output):
            output[start] = _upper_first_alpha(output[start])
    return " ".join(output)


def sword_boundary_exchange(
    text: str,
    ratio: float,
    rng: random.Random,
) -> str:
    """Minimal adaptation of SwordStamp's pinned MIT-licensed atom."""
    del rng
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    source = sentences(text)
    if len(source) < 2:
        return text
    tokenized = [sentence.split() for sentence in source]
    tokens = [token for sentence in tokenized for token in sentence]
    old_boundaries: list[int] = []
    cursor = 0
    for sentence in tokenized[:-1]:
        cursor += len(sentence)
        old_boundaries.append(cursor)
    exchange_count = min(
        len(old_boundaries),
        int(ratio * len(old_boundaries) + 0.5),
    )
    if exchange_count == 0:
        return text
    new_boundaries = list(old_boundaries)
    starts = [0, *old_boundaries]
    selected = _evenly_spaced_indices(
        len(old_boundaries),
        exchange_count,
    )
    for index in selected:
        sentence_length = len(tokenized[index])
        if sentence_length >= 2:
            new_boundaries[index] = starts[index] + sentence_length // 2
    return _render_boundary_exchange(tokens, old_boundaries, new_boundaries)


def sword_uniform_rechunk(
    text: str,
    words_per_chunk: int,
    rng: random.Random,
) -> str:
    """Minimal adaptation of SwordStamp's pinned MIT-licensed atom."""
    del rng
    if words_per_chunk < 1:
        raise ValueError(
            f"words_per_chunk must be >= 1, got {words_per_chunk}"
        )
    words = [
        word.rstrip(_TERMINAL)
        for word in " ".join(sentences(text)).split()
    ]
    words = [word for word in words if word]
    if not words:
        return text
    chunks = [
        " ".join(words[index:index + words_per_chunk])
        for index in range(0, len(words), words_per_chunk)
    ]
    return " ".join(_upper_first(_terminate(chunk)) for chunk in chunks)


def restored_original_spans(original: str, strict_merged: str) -> list[str]:
    """Split a strict merge at saved source boundaries without changing text."""
    source = sentences(original)
    if len(sentences(strict_merged)) != 1:
        raise ValueError("Expected a fully merged one-sentence surface")
    spans = [
        (
            _without_terminal(sentence) + ";"
            if index < len(source) - 1
            else sentence
        )
        for index, sentence in enumerate(source)
    ]
    if " ".join(spans) != strict_merged:
        raise ValueError("Restored spans do not reconstruct strict-merge text")
    return spans


def apply_condition(text: str, condition: str, seed: int) -> dict:
    """Apply one preregistered condition and verify its lexical invariant."""
    source_n = len(sentences(text))
    rng = random.Random(seed)
    if condition == "C0":
        transformed = text
        supplied_spans = None
    elif condition == "C1":
        transformed = sword_merge_adjacent(text, 1.0, rng)
        supplied_spans = None
    elif condition == "C2":
        transformed = sword_boundary_exchange(text, 1.0, rng)
        supplied_spans = None
    elif condition == "C3":
        transformed = sword_uniform_rechunk(text, 16, rng)
        supplied_spans = None
    elif condition in {"C4a", "C4b", "C4c", "C4d"}:
        fraction = {
            "C4a": 0.25,
            "C4b": 0.50,
            "C4c": 0.75,
            "C4d": 1.00,
        }[condition]
        transformed = strict_semicolon_merge(text, fraction)
        supplied_spans = None
    elif condition == "C5":
        transformed = strict_semicolon_merge(text, 1.0)
        supplied_spans = restored_original_spans(text, transformed)
    else:
        raise ValueError(f"Unknown frozen condition: {condition}")
    if lexical_tokens(transformed) != lexical_tokens(text):
        raise ValueError(f"{condition} changed lexical tokens")
    transformed_n = (
        len(supplied_spans)
        if supplied_spans is not None
        else len(sentences(transformed))
    )
    if condition == "C2" and transformed_n != source_n:
        raise ValueError("C2 did not preserve the detector unit count")
    return {
        "condition": condition,
        "text": transformed,
        "source_units": source_n,
        "surface_units": len(sentences(transformed)),
        "detector_units": transformed_n,
        "supplied_spans": supplied_spans,
        "lexical_tokens_preserved": True,
    }
