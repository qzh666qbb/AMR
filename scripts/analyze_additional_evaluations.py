import csv, json, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; E3=ROOT/'experiments/e3_main'
rows=list(csv.DictReader((E3/'human_eval_ai_assisted_100.csv').open(encoding='utf-8-sig')))
candidate_payload=json.load((E3/'candidates.json').open(encoding='utf-8'))
record_by_index={r['source_index']:i for i,r in enumerate(candidate_payload['records'])}
families={'Method-1':'plain_paraphrase','Method-2':'sentence_boundary','Method-3':'amr_guided'}
out={'n':len(rows),'families':{}}
for method,family in families.items():
 x=[r for r in rows if r['candidate_method_blinded']==method]
 zall=np.load(E3/'detection'/family/'machine_z_scores.npy')
 z=np.array([zall[(int(r['pair_id'])-1)//3] for r in x])
 scores=np.array([np.mean([float(r[k]) for k in ['meaning_preserved_1to5','factual_consistency_1to5','fluency_1to5','coherent_1to5']]) for r in x])
 out['families'][family]={'n':len(x),'quality_mean':float(scores.mean()),'quality_ge4_rate':float(np.mean(scores>=4)),'quality_ge4_escape_rate':float(np.mean(z<=2.33)),'major_error_rate':float(np.mean([r['major_error']=='true' for r in x])),'error_keywords':{}}
 for r in x:
  note=r['notes'].lower()
  for label,terms in {'repetition':['repeat','repet'],'fragment':['fragment','incomplete','awkward'],'entity':['entity','name','person'],'omission':['omit','missing','left out'],'hallucination':['add','invent','unsupported']}.items():
   if any(t in note for t in terms): out['families'][family]['error_keywords'][label]=out['families'][family]['error_keywords'].get(label,0)+1
json.dump(out,(E3/'additional_evaluations.json').open('w',encoding='utf-8'),indent=2)
for k,v in out['families'].items(): print(k,v)
