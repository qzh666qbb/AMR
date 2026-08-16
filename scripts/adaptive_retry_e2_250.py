import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/planE_e2_joint250"
KEY = (ROOT / "1.txt").read_text(encoding="utf-8").strip()
rows = json.load((EXP / "quality_aware_reviews.json").open(encoding="utf-8"))
selected = {r["index"]: r for r in json.load((EXP / "quality_aware_selected.json").open(encoding="utf-8"))}
failed = []
for i in range(250):
    if i not in selected or selected[i]["z_score"] > 2.33:
        failed.append(i)
out = EXP / "adaptive_retry"
out.mkdir(parents=True, exist_ok=True)
checkpoint = out / "candidates.partial.jsonl"
existing = []
if checkpoint.exists():
    existing = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines() if line.strip()]
done = {(r["index"], r["candidate"]) for r in existing}
source_by_index = {r["index"]: r["source"] for r in rows}
failed_reason = {i: ("quality repair" if i not in selected else "detector evasion repair") for i in failed}


def generate(job):
    i, k = job
    instruction = failed_reason[i]
    if instruction == "quality repair":
        focus = "Prioritize exact preservation of every entity, number, date, negation, modality, attribution, and event relation. Do not omit or reverse any fact."
    else:
        focus = "Preserve all facts exactly, but strongly change predicate packaging, sentence boundaries, voice, nominalization, clause attachment, and reference structure to induce a different AMR parse."
    prompt = f"""Rewrite this complete paragraph for a robustness audit. {focus} Keep similar length and natural English. Return only the rewritten paragraph.

SOURCE:
{source_by_index[i]}

This is adaptive candidate {k}; use a substantially different structure from ordinary paraphrasing."""
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": "Bearer " + KEY},
            json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 1400, "thinking": {"type": "disabled"}},
            timeout=240,
        )
        attack = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        attack = ""
        print("ERR", i, k, type(exc).__name__, flush=True)
    return {"index": i, "candidate": k, "source": source_by_index[i], "attack": attack}


jobs = [(i, k) for i in failed for k in range(4) if (i, k) not in done]
with ThreadPoolExecutor(max_workers=12) as pool:
    futures = [pool.submit(generate, job) for job in jobs]
    for n, future in enumerate(as_completed(futures), 1):
        result = future.result()
        existing.append(result)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        if n % 20 == 0 or n == len(jobs):
            print(f"adaptive {n}/{len(jobs)}", flush=True)
json.dump({"failed_indices": failed, "rows": sorted(existing, key=lambda r: (r["index"], r["candidate"]))}, (out / "candidates.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"saved {len(existing)} adaptive candidates for {len(failed)} paragraphs")
