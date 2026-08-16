import json
from pathlib import Path
from datasets import Dataset
root=Path(__file__).resolve().parents[1]; rows=json.load(open(root/'experiments/api_natural_batch3/candidates.json',encoding='utf-8')); out=root/'experiments/api_natural_batch3'; Dataset.from_dict({'text':[r['attack'] for r in rows]}).save_to_disk(str(out/'dataset'))
