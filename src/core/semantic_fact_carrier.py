"""Public fact alphabet; classification never reads intended symbols."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ALPHABET_VERSION = "transfer-direction-status:v1"
SYMBOLS = {
    "A": ("role_one", "role_two", "completed"),
    "B": ("role_two", "role_one", "completed"),
    "C": ("role_one", "role_two", "planned_not_occurred"),
    "D": ("role_two", "role_one", "planned_not_occurred"),
}

# Development scenes only; final generalization must use held-out families.
SCENES = [
    ("preparing sample labels at a field geology station", "Mina", "Theo", "label box"),
    ("arranging printed maps at a canoe club", "Ravi", "Elena", "map folder"),
    ("setting up a microscope demonstration", "Jules", "Nora", "lens case"),
    ("organizing costumes for a school play", "Samir", "Ava", "costume bag"),
    ("checking tools at a bicycle workshop", "Lena", "Omar", "tool pouch"),
    ("assembling supplies for a roof garden", "Nia", "Felix", "seed packet"),
    ("sorting recordings for a local radio archive", "Ivo", "Maya", "recording drive"),
    ("preparing ceramics for a neighborhood exhibition", "Esme", "Leon", "glaze jar"),
    ("setting up a birdwatching information desk", "Hana", "Miles", "guidebook"),
    ("organizing a volunteer beach survey", "Ada", "Yusuf", "survey clipboard"),
    ("checking materials for a weaving lesson", "Sofia", "Arun", "thread spool"),
    ("preparing a small planetarium session", "Priya", "Noel", "star chart"),
    ("arranging equipment for a community choir", "Clara", "Dario", "music folder"),
    ("cataloguing objects at a transport museum", "Owen", "Leila", "catalogue notebook"),
    ("preparing markers for an orienteering course", "Tessa", "Kenji", "marker bundle"),
    ("organizing props for a tabletop photography class", "Anya", "Hugo", "reflector"),
    ("setting up an insect observation session", "Sara", "Emil", "specimen tray"),
    ("packing materials for a printmaking demonstration", "Zara", "Finn", "ink roller"),
    ("preparing a community weather display", "Celia", "Andre", "sensor case"),
    ("checking equipment for a school robotics display", "Mira", "Leo", "battery pack"),
    ("arranging supplies for a tea tasting session", "Dina", "Pavel", "sample tin"),
    ("organizing materials for a model railway club", "Nolan", "Iris", "track gauge"),
    ("preparing a history walk through a harbor", "Rosa", "Tariq", "route booklet"),
    ("sorting supplies for a botanical drawing class", "Vera", "Ben", "pencil case"),
]


def plan():
    """Four independently generated facts per scene, with stable public roles."""
    rows = []
    for scene_id, (scene, one, two, item) in enumerate(SCENES):
        for symbol, (sender, recipient, status) in SYMBOLS.items():
            roles = {"role_one": one, "role_two": two}
            rows.append({
                "item_id": f"f{scene_id:02d}{symbol}",
                "scene_id": scene_id,
                "scene": scene,
                "public_contract": {"role_one": one, "role_two": two, "object": item},
                "intended_symbol": symbol,
                "intended_facts": {
                    "sender": roles[sender], "recipient": roles[recipient],
                    "object": item, "transfer_status": status,
                },
            })
    return rows


@dataclass(frozen=True)
class Readback:
    symbol: str | None
    reason: str


def classify_facts(facts: Mapping, contract: Mapping) -> Readback:
    """Map independently read facts into the public alphabet, including erasure."""
    if facts.get("contradiction") is True:
        return Readback(None, "contradictory_facts")
    fields = ("sender", "recipient", "object", "transfer_status")
    if any(not isinstance(facts.get(k), str) or facts[k].strip() in ("", "unknown")
           for k in fields):
        return Readback(None, "missing_or_unknown_fact")
    normalize = lambda s: " ".join(s.split()).casefold()
    one, two = normalize(contract["role_one"]), normalize(contract["role_two"])
    if one == two:
        return Readback(None, "ambiguous_public_roles")
    sender, recipient = normalize(facts["sender"]), normalize(facts["recipient"])
    if normalize(facts["object"]) != normalize(contract["object"]):
        return Readback(None, "wrong_or_unresolved_object")
    direction = (
        ("role_one", "role_two") if (sender, recipient) == (one, two)
        else ("role_two", "role_one") if (sender, recipient) == (two, one)
        else None
    )
    if direction is None:
        return Readback(None, "wrong_or_unresolved_roles")
    state = facts["transfer_status"]
    for symbol, target in SYMBOLS.items():
        if target == (*direction, state):
            return Readback(symbol, "complete_contract_facts")
    return Readback(None, "state_not_in_contract")


def validate_evidence(text: str, evidence) -> bool:
    if not isinstance(evidence, list) or not evidence:
        return False
    for span in evidence:
        if not isinstance(span, dict):
            return False
        a, b = span.get("start"), span.get("end")
        if (type(a) is not int or type(b) is not int
                or not 0 <= a < b <= len(text)
                or not span.get("quote") or text[a:b] != span["quote"]):
            return False
    return True


def generation_messages(row):
    f = row["intended_facts"]
    state = (
        "The transfer has actually been completed: explicitly assert completion "
        "and that the recipient has received this object."
        if f["transfer_status"] == "completed" else
        "There is a definite plan to transfer this object, and the transfer has "
        "explicitly not yet occurred. Assert BOTH the plan and non-occurrence. "
        "A desire, possibility, promise or future tense by itself is insufficient."
    )
    return [
        {"role": "system", "content":
         "Write natural fictional English for a controlled research study. "
         "Return only prose, with no heading, list, symbols or explanation."},
        {"role": "user", "content":
         f"Scene: {row['scene']}.\nParticipants: {row['public_contract']['role_one']} "
         f"and {row['public_contract']['role_two']}.\nWrite a self-contained account "
         "of 2-3 sentences, preferably 35-70 words. Include brief natural background "
         "activity, but only one transfer event involving the specified object.\n"
         f"Sender: {f['sender']}; recipient: {f['recipient']}; object: {f['object']}.\n"
         f"{state}\nPreserve who sends to whom; do not substitute a different object "
         "or swap participants. State the transfer status explicitly, not only with "
         "past/future tense. Use varied natural phrasing. Do not mention graphs, "
         "codes, watermarks, letters or the research contract."},
    ]


def paraphrase_messages(text):
    return [
        {"role": "system", "content":
         "Rewrite fictional English without changing its stated facts. "
         "Return only the rewritten prose."},
        {"role": "user", "content":
         "Paraphrase the account below using different wording or sentence structure. "
         "Keep all participants, the object, transfer direction, explicit completion "
         "or explicit plan plus non-occurrence, and the order of events. Keep names "
         "unchanged and do not invent, remove or reverse a transfer. Prefer 35-70 "
         "words. Do not explain your rewrite.\nACCOUNT:\n" + text},
    ]
