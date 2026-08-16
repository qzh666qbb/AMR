import json,re,requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import penman,numpy as np
root=Path(__file__).resolve().parents[1]; base=root/'experiments/planE_e2_joint250'; p=base/'adaptive_retry'; key=(root/'1.txt').read_text().strip(); rows=json.load(open(p/'rows.json',encoding='utf-8')); z=np.load(p/'detection/machine_z_scores.npy'); parsed=json.load(open(p/'parsed.json',encoding='utf-8'))
for r,s in zip(rows,z): r['z_score']=float(s)
def gates(r,i):
 s,a=r['source'],r['attack']; nums=set(re.findall(r'\b\d[\d,.:/%-]*\b',s)); an=set(re.findall(r'\b\d[\d,.:/%-]*\b',a)); neg=lambda x:bool(re.search(r"\b(?:no|not|never|without|cannot|n't)\b",x,re.I)); ratio=len(a.split())/max(1,len(s.split())); amr=bool(parsed[i])
 for g in parsed[i]:
  try: penman.decode(g)
  except: amr=False
 return {'length_ok':.7<=ratio<=1.6,'numbers_ok':nums.issubset(an),'negation_ok':neg(s)==neg(a),'amr_ok':amr}
groups={i:[r for r in rows if r['index']==i] for i in sorted(set(r['index'] for r in rows))}
def review(item):
 i,gs=item; q='Evaluate candidates against SOURCE. Return JSON array of four objects {"candidate":number,"score":1-5,"major_error":boolean,"reason":"short"}. Major error means changed entity, number, polarity, modality, attribution, event, or relation.\nSOURCE:\n'+gs[0]['source']+'\n'+ '\n'.join(f'\nCANDIDATE {r["candidate"]}:\n{r["attack"]}' for r in gs)
 try:
  j=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':q}],'temperature':0,'max_tokens':700,'thinking':{'type':'disabled'}},timeout=240).json(); arr=json.loads(re.search(r'\[.*\]',j['choices'][0]['message']['content'],re.S).group(0)); mp={int(x['candidate']):x for x in arr}
 except Exception as e: mp={r['candidate']:{'score':0,'major_error':True,'reason':type(e).__name__} for r in gs}
 return i,mp
with ThreadPoolExecutor(max_workers=10) as ex: reviews=dict(ex.map(review,groups.items()))
valid=[]
for i,gs in groups.items():
 for pos,r in enumerate(gs):
  r['surface']=gates(r,pos); r['review']=reviews[i].get(r['candidate'],{'score':0,'major_error':True}); r['quality_valid']=all(r['surface'].values()) and r['review'].get('score',0)>=4 and not r['review'].get('major_error',True)
  if r['quality_valid']: valid.append(r)
json.dump(rows,open(p/'evaluated.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
best={}
for r in valid: best[r['index']]=min([x for x in valid if x['index']==r['index']],key=lambda x:x['z_score'])
json.dump(list(best.values()),open(p/'quality_valid_best.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('valid',len(valid),'paragraphs',len(best),'low',sum(r['z_score']<=2.33 for r in best.values()))
