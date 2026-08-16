import json
from pathlib import Path
from datasets import Dataset
root=Path(__file__).resolve().parents[1]; p=root/'experiments/planE_e2_joint250/adaptive_retry'; rows=json.load(open(p/'candidates.json',encoding='utf-8'))['rows']; Dataset.from_dict({'text':[r['attack'] for r in rows]}).save_to_disk(str(p/'dataset')); json.dump(rows,open(p/'rows.json','w',encoding='utf-8'),ensure_ascii=False,indent=2); print(len(rows))
