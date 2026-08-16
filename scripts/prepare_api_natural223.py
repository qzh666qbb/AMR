import json
from pathlib import Path
from datasets import Dataset
root=Path(__file__).resolve().parents[1]; rows=[json.loads(x) for x in open(root/'experiments/api_natural223/results.jsonl',encoding='utf-8') if json.loads(x).get('ok')]; out=root/'experiments/api_natural223';Dataset.from_dict({'text':[r['attack'] for r in rows]}).save_to_disk(str(out/'dataset'));json.dump(rows,open(out/'records.json','w',encoding='utf-8'),ensure_ascii=False,indent=2);print(len(rows))
