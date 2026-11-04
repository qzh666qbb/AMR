"""Frozen writer assignments and fixed-pool attacks, without target observations."""
import hashlib
import math
import random
import re

ARMS = ("a", "b", "joint", "pooled")
SLOTS = 16
ATTEMPTS = 50
BUDGETS = (1, 4, 16)
WRITER_PARAMETERS = {
    "model": "deepseek-flash", "temperature": .7, "top_p": .9,
    "max_tokens": 50, "thinking": {"type": "disabled"},
}
ATTACK_PARAMETERS = {
    "model": "deepseek-flash", "temperature": .9, "top_p": 1,
    "max_tokens": 1600, "thinking": {"type": "disabled"},
}
STYLES = (
    "Use plain, direct prose.",
    "Use fluent explanatory prose.",
    "Use a concise news-report style.",
    "Use a natural narrative style.",
)
OPERATIONS = (
    "Rephrase vocabulary and clauses while keeping the original sentence count.",
    "Merge related sentences where natural, keeping all facts and their relations.",
    "Split compound statements where natural, keeping all facts and their relations.",
    "Reorganize clause boundaries and phrasing while keeping the logical sequence.",
)


def seed(*parts):
    return int(hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest(), 16)


def order(sources):
    result = []
    for source in sources:
        for slot in range(SLOTS):
            arms = list(ARMS)
            random.Random(seed("endpoint1/order", source["source_id"], slot)).shuffle(arms)
            result.extend({"source_id": source["source_id"], "arm": arm, "slot": slot}
                          for arm in arms)
    return result


def template_schedule(banks, source_id, arm, slot):
    if arm not in ARMS or not 0 <= slot < SLOTS:
        raise ValueError("Unknown writer arm or slot")
    key = ("a" if slot % 2 == 0 else "b") if arm == "joint" else arm
    values = banks["a"] + banks["b"] if key == "pooled" else banks[key]
    if len(values) != (100 if key == "pooled" else 50):
        raise ValueError("Bank capacity changed")
    # Ten draws, five attempts each, with replacement as in the official writer.
    # Shared uniforms pair single A/B and pooled draws without using scores.
    rng = random.Random(seed("endpoint1/template", source_id, slot))
    return [(key, values[rng.randrange(len(values))]) for _ in range(10)]


def attack_messages(text, index):
    if not 0 <= index < 16:
        raise ValueError("Unknown fixed candidate")
    return [
        {"role": "system", "content": "Rewrite English text faithfully. Output only the rewritten text."},
        {"role": "user", "content":
         "Preserve every stated fact, participant, number, negation, attribution, and causal "
         "or temporal relation. Add no facts and omit no information. Keep natural English "
         "and a similar total length. " + STYLES[index // 4] + " " + OPERATIONS[index % 4]
         + "\n\nTEXT\n" + text},
    ]


def accepted(text, finish_reason, score, tokenizer):
    # A declared engineering addition shared by all four arms.
    return bool(finish_reason == "stop" and text.strip()
                and len(tokenizer.tokenize(text)) == 1
                and re.search(r'[.!?]["\')\]]*$', text.strip())
                and score is not None and score >= .75)


def client_score(value):
    """Map detector abstention to a fixed public lower sentinel; preserve raw None."""
    if value is None:
        return -1e9
    if type(value) not in (int, float) or not math.isfinite(value) or value <= -1e9:
        raise ValueError("Invalid detector score")
    return value


def client_threshold(threshold):
    return 1e9 if threshold["disabled"] else client_score(threshold["threshold"])


def rate(k, n):
    from swan_independent_calibration1_core import binomial_interval
    return {"numerator": k, "denominator": n, "rate": k / n if n else None,
            "CP95": binomial_interval(k, n) if n else None}
