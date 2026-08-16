import json,re
from pathlib import Path
import numpy as np,spacy
nlp=spacy.load('en_core_web_sm'); root=Path(__file__).resolve().parents[1]; rows=[]
for folder in ['api_natural5','api_natural_batch2','api_natural_batch3','api_natural_batch4','api_natural_batch5','api_natural_batch6']:
 data=json.load(open(root/'experiments'/folder/'candidates.json',encoding='utf-8')); z=np.load(root/'experiments'/folder/'detection/machine_z_scores.npy')
 for i,r in enumerate(data):
  s,a=r['text'],r['attack']; nums=lambda t:set(re.findall(r'\b\d[\d,.:/%-]*\b',t)); neg=lambda t:bool(re.search(r"\b(?:no|not|never|without|n't)\b",t,re.I)); ent=lambda t:{(e.text,e.label_) for e in nlp(t).ents}
  c={'numbers':nums(s)<=nums(a),'entities':ent(s)<=ent(a),'negation':neg(s)==neg(a),'length':.7<=len(a)/max(1,len(s))<=1.3,'nonempty':bool(a.strip())}; rows.append({'folder':folder,'index':r['sentence'] if 'sentence' in r else r['paragraph'],'z':float(z[i]),'pass':all(c.values()),'checks':c,'source':s,'attack':a})
json.dump(rows,open(root/'experiments/api_natural30_quality.json','w',encoding='utf-8'),ensure_ascii=False,indent=2); print({'n':len(rows),'quality_pass':sum(x['pass'] for x in rows),'joint':sum(x['pass'] and x['z']<=2.33 for x in rows)})
