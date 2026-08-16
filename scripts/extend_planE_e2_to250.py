import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
KEY = (ROOT / "1.txt").read_text(encoding="utf-8").strip()
SOURCE = ROOT / "experiments/planE_e2_joint100/candidates.json"
OUT = ROOT / "experiments/planE_e2_joint250"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT / "candidates.partial.jsonl"
DATA = load_from_disk(str(ROOT / "baselines/local/swan_acl2026_realnews/runs/formal_250x5"))
PROMPT = """Rewrite this entire paragraph into one meaning-preserving candidate for a robustness audit. Preserve every entity, number, date, polarity, modality, attribution, and event. Change syntax broadly (sentence boundary, voice, nominalization, clause attachment, reference). Keep similar length. Return only the rewritten paragraph.\n\nSOURCE:\n{}"""


def generate(index, candidate):
    prompt = PROMPT.format(DATA["text"][index]) + f"\nCandidate variant: {candidate}"
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": "Bearer " + KEY},
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1400,
                "thinking": {"type": "disabled"},
            },
            timeout=240,
        )
        attack = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        attack = ""
        print("ERR", index, candidate, type(exc).__name__, flush=True)
    return {"index": index, "candidate": candidate, "source": DATA["text"][index], "attack": attack}


existing = json.load(SOURCE.open(encoding="utf-8"))["rows"]
generated = []
if CHECKPOINT.exists():
    generated = [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines() if line.strip()]
done = {(row["index"], row["candidate"]) for row in generated}
jobs = [(i, k) for i in range(100, 250) for k in range(4) if (i, k) not in done]
with ThreadPoolExecutor(max_workers=12) as pool:
    futures = [pool.submit(generate, i, k) for i, k in jobs]
    for n, future in enumerate(as_completed(futures), 1):
        row = future.result()
        generated.append(row)
        with CHECKPOINT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if n % 25 == 0 or n == len(jobs):
            print(f"new {n}/{len(jobs)}", flush=True)
rows = sorted(existing + generated, key=lambda row: (row["index"], row["candidate"]))
json.dump({"n": 250, "k": 4, "rows": rows}, (OUT / "candidates.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved", len(rows), flush=True)
