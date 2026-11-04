"""Carrier repair v2. Facts and erasures remain independent of planned symbols."""
from __future__ import annotations

import semantic_fact_carrier as base

ALPHABET_VERSION = "transfer-direction-status:v2-explicit-narration"
SYMBOLS = base.SYMBOLS
SCENES = base.SCENES
classify_facts = base.classify_facts
validate_evidence = base.validate_evidence


def plan():
    rows = base.plan()
    for row in rows:
        row["item_id"] = "v2-" + row["item_id"]
    return rows


def generation_messages(row):
    facts = row["intended_facts"]
    state = (
        "Directly narrate that the sender has handed this object to the recipient "
        "and that the recipient has received it. State receipt as an actual fact, "
        "not something somebody says, believes, expects, or promises."
        if facts["transfer_status"] == "completed" else
        "The narrator must directly assert a definite, CURRENT decision or plan by "
        "the sender to hand this object to the recipient, AND directly assert that "
        "this handover has NOT happened yet. Both assertions must be outside speech, "
        "belief, confirmation, quotation, or any reported proposition. "
        "A past abandoned plan, setting an object aside for somebody, reserving it "
        "for their use, saying one will give it, or finding it still nearby does "
        "not meet these requirements. Do not replace the explicit plan with such "
        "implications. Do not rely on future/past tense alone. "
        "A safe form is '<sender> has decided to hand <object> to <recipient>. "
        "<sender> has not handed it over yet.' Use natural variations that retain "
        "these two narrator-level assertions; do not add a second transfer."
    )
    return [
        {"role": "system", "content":
         "Write natural fictional English for a controlled research study. "
         "Return only prose, without headings, lists, codes, or explanations."},
        {"role": "user", "content":
         f"Scene: {row['scene']}.\n"
         f"Participants: {row['public_contract']['role_one']} and "
         f"{row['public_contract']['role_two']}.\n"
         f"Sender: {facts['sender']}; recipient: {facts['recipient']}; "
         f"object: {facts['object']}.\n"
         "Write 2-3 sentences, preferably 35-70 words, with brief natural background "
         "activity and exactly one actual or planned transfer of this object. "
         "Include an explicit simultaneous relation between two background activities "
         "using 'while', 'at the same time', or another unambiguous expression. "
         "If you include an additional before/after relation, state it clearly.\n"
         f"{state}\n"
         "Keep names, object identity and transfer direction exact. Avoid ambiguity "
         "from pronouns, gratuitous repetition, or inconsistent narrative time. "
         "Do not mention graphs, watermarks, symbols, or research requirements."},
    ]


def paraphrase_messages(text):
    return [
        {"role": "system", "content":
         "Paraphrase fictional English while preserving its explicitly stated facts. "
         "Return only the rewritten prose."},
        {"role": "user", "content":
         "Rewrite the account using different wording or sentence structure, "
         "preferably 35-70 words. Keep names, object identity, sender and recipient "
         "unchanged. Preserve direct narrator assertions of actual receipt, or "
         "CURRENT definite transfer plan plus explicit nonoccurrence. Do not move "
         "assertions into reported speech, promises, beliefs, or merely implied "
         "object allocation. Do not add missing facts or repair an ambiguous source. "
         "Preserve EVERY explicit simultaneous, before, after, and conditional "
         "temporal relation, including background activities. An explicit 'while' "
         "relation must stay explicitly simultaneous, not just become an 'and' list. "
         "Keep the transfer state unchanged and do not add, remove, or reverse a "
         "transfer. Do not explain your rewrite.\nACCOUNT:\n" + text},
    ]
