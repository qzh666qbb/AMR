import json, os
from pathlib import Path
from datasets import Dataset
root=Path(__file__).resolve().parents[1]; n=os.environ.get('E2_N','30'); p=root/f'experiments/planE_e2_joint{n}'; rows=json.load(open(p/'candidates.json',encoding='utf-8'))['rows']; rows=[r for r in rows if r['attack'].strip()]
Dataset.from_dict({'text':[r['attack'] for r in rows]}).save_to_disk(str(p/'dataset'))
json.dump(rows,open(p/'rows.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(len(rows))
