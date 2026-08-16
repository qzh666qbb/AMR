import json,time
from pathlib import Path
import requests
from datasets import load_from_disk
root=Path(__file__).resolve().parents[1]; key=(root/'1.txt').read_text().strip(); ds=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5')); texts=ds['text'][:30]
rows=[]
for i,t in enumerate(texts):
 prompt='Rewrite the following paragraph in natural English using varied syntax (active/passive alternation, nominalization, and relative clauses) while preserving every fact, entity, number, date, negation, and sentence count. Return only the rewritten paragraph.\\n\\n'+t
 try:
  r=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':prompt}],'temperature':0.2},timeout=120); r.raise_for_status(); out=r.json()['choices'][0]['message']['content'].strip()
 except Exception as e: out=''; print('error',i,type(e).__name__)
 rows.append({'index':i,'source':t,'attack':out}); print(i,bool(out)); time.sleep(.2)
outdir=root/'experiments/api_syntax30';outdir.mkdir(parents=True,exist_ok=True);json.dump(rows,open(outdir/'candidates.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
