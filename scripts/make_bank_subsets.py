import json
import random
from pathlib import Path

root = Path(__file__).resolve().parents[1]
swan = root / 'third_party/SWAN'
src = swan / 'amr_bank/banks/amr_bank_50.json'
out = root / 'experiments/e5_transfer/banks'
out.mkdir(parents=True, exist_ok=True)
data = json.load(src.open(encoding='utf-8'))
for name, items in [('first25', data[:25]), ('random25', random.Random(20260725).sample(data, 25))]:
    json.dump(items, open(out / f'{name}.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(name, len(items))
