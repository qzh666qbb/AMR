import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import penman

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/planE_e2_joint250"
KEY = (ROOT / "1.txt").read_text(encoding="utf-8").strip()
ROWS = json.load((EXP / "scored_candidates.json").open(encoding="utf-8"))
PARSED = json.load((EXP / "parsed.json").open(encoding="utf-8"))
OUT = EXP / "quality_aware_reviews.json"


def surface_gate(row):
    source, attack = row["source"], row["attack"]
    numbers = set(re.findall(r"\b\d[\d,.:/%-]*\b", source))
    candidate_numbers = set(re.findall(r"\b\d[\d,.:/%-]*\b", attack))
    negation = re.compile(r"\b(?:no|not|never|without|cannot|n't)\b", re.I)
    ratio = len(attack.split()) / max(1, len(source.split()))
    graphs = PARSED[row["position"]]
    amr_ok = bool(graphs)
    for graph in graphs:
        try:
            penman.decode(graph)
        except Exception:
            amr_ok = False
    return {
        "length_ok": 0.7 <= ratio <= 1.6,
        "numbers_ok": numbers.issubset(candidate_numbers),
        "negation_ok": bool(negation.search(source)) == bool(negation.search(attack)),
        "amr_ok": amr_ok,
    }


def review_group(index):
    group = [row for row in ROWS if row["index"] == index]
    source = group[0]["source"]
    candidates = "\n\n".join(f"CANDIDATE {row['candidate']}:\n{row['attack']}" for row in group)
    prompt = (
        "Evaluate four rewrites against SOURCE. For each candidate score semantic preservation 1-5 and mark "
        "major_error true only for changed entities, numbers, polarity, modality, attribution, events, or relations. "
        "Return JSON only as an array of four objects with keys candidate, score, major_error, reason.\n\n"
        f"SOURCE:\n{source}\n\n{candidates}"
    )
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": "Bearer " + KEY},
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 700,
                "thinking": {"type": "disabled"},
            },
            timeout=240,
        )
        text = response.json()["choices"][0]["message"]["content"]
        reviews = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        mapped = {int(item["candidate"]): item for item in reviews}
    except Exception as exc:
        mapped = {row["candidate"]: {"candidate": row["candidate"], "score": 0, "major_error": True, "reason": type(exc).__name__} for row in group}
    return {"index": index, "reviews": mapped}


with ThreadPoolExecutor(max_workers=10) as pool:
    reviewed_groups = list(pool.map(review_group, range(250)))
review_map = {(item["index"], int(candidate)): review for item in reviewed_groups for candidate, review in item["reviews"].items()}
scored = []
for row in ROWS:
    gate = surface_gate(row)
    review = review_map.get((row["index"], row["candidate"]), {"score": 0, "major_error": True, "reason": "missing"})
    valid = all(gate.values()) and review.get("score", 0) >= 4 and not review.get("major_error", True)
    scored.append({**row, "surface": gate, "review": review, "quality_valid": valid})
selected = []
for index in range(250):
    valid = [row for row in scored if row["index"] == index and row["quality_valid"]]
    if valid:
        selected.append(min(valid, key=lambda row: row["z_score"]))
json.dump(scored, OUT.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(selected, (EXP / "quality_aware_selected.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
low = sum(row["z_score"] <= 2.33 for row in selected)
print(f"quality_valid_paragraphs={len(selected)}/250")
print(f"quality_constrained_success={low}/250")
print(f"success_within_valid={low}/{len(selected)}")
