import json
from pathlib import Path
from datasets import Dataset
root=Path(__file__).resolve().parents[1];rows=[json.loads(x) for x in open(root/'experiments/planE_e1_roundtrip/results.jsonl',encoding='utf-8')]
for lang in ('zh','de'):
 out=root/f'experiments/planE_e1_roundtrip/{lang}';out.mkdir(parents=True,exist_ok=True); rr=[r for r in rows if r['lang']==lang and r['ok']];Dataset.from_dict({'text':[r['attack'] for r in rr]}).save_to_disk(str(out/'dataset'));json.dump(rr,open(out/'records.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
