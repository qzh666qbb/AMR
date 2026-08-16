import json
from pathlib import Path
from datasets import Dataset

root = Path(__file__).resolve().parents[1]
src = root / 'experiments/e4_ablation/candidates.json'
out = root / 'experiments/e4_ablation/datasets'
payload = json.load(src.open(encoding='utf-8'))
out.mkdir(parents=True, exist_ok=True)
for family in ['no_amr', 'amr_guided']:
    dataset = Dataset.from_dict({'text': [r['candidates'][family] for r in payload['records']], 'source_index': [r['source_index'] for r in payload['records']]})
    dataset.save_to_disk(str(out / family))
    print(family, len(dataset))
