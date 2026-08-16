import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/SWAN"))
from utils.bedrock_utils import BedrockManager

SRC = ROOT / "experiments/e3_main/human_eval_sample_100.csv"
OUT = ROOT / "experiments/e3_main/human_eval_ai_assisted_100.csv"
SYSTEM = "Return only JSON with integer scores 1-5 for meaning, factual, fluency, coherence, boolean major_error, and a short note. Do not reward watermark evasion."


def main():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
    done = list(csv.DictReader(OUT.open(encoding="utf-8-sig"))) if OUT.exists() else []
    manager = BedrockManager("us-east-1", "deepseek-v4-flash")
    for row in rows[len(done):]:
        prompt = (
            "SOURCE:\n" + row["source_text"] + "\n\nCANDIDATE:\n" + row["candidate_text"]
            + "\n\nScore meaning preservation, factual consistency, fluency, and coherence from 1 (bad) to 5 (excellent). Set major_error true for a material fact, entity, number, polarity, or claim change."
        )
        raw = manager.generate(prompt, SYSTEM, max_tokens=300, temperature=0.0, top_p=1.0)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise RuntimeError("judge returned non-JSON")
        score = json.loads(match.group(0))
        row.update({
            "meaning_preserved_1to5": score["meaning"],
            "factual_consistency_1to5": score["factual"],
            "fluency_1to5": score["fluency"],
            "coherent_1to5": score["coherence"],
            "major_error": str(score["major_error"]).lower(),
            "notes": score.get("note", ""),
        })
        done.append(row)
        with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=done[0].keys())
            writer.writeheader()
            writer.writerows(done)
        print(f"{len(done)}/100", flush=True)


if __name__ == "__main__":
    main()
