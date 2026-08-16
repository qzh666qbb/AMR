import json,requests
from pathlib import Path
from datasets import load_from_disk
root=Path(__file__).resolve().parents[1]; key=(root/'1.txt').read_text().strip(); ds=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5')); ids=[3,6,10,13,20]; rows=[]
for i in ids:
 t=ds['text'][i]; p='Rewrite this entire paragraph with varied syntax (passive voice, nominalization, relative clauses), changing as many complete sentences as safely possible. Preserve sentence count, every entity, number, date, negation, event, and fact. Return only the rewritten paragraph:\n\n'+t
 try:
  j=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':p}],'temperature':0.1,'max_tokens':1200,'thinking':{'type':'disabled'}},timeout=180).json(); out=j['choices'][0]['message']['content'].strip()
 except Exception as e: out=''; print('error',i,type(e).__name__)
 rows.append({'index':i,'source':t,'attack':out});print(i,bool(out))
out=root/'experiments/api_joint_paragraph5';out.mkdir(parents=True,exist_ok=True);json.dump(rows,open(out/'candidates.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
