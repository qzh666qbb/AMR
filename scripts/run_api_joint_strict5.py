import json,re,requests
from pathlib import Path
from datasets import load_from_disk
root=Path(__file__).resolve().parents[1];key=(root/'1.txt').read_text().strip();ds=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'));ids=[45,49,51,65,75];rows=[]
for i in ids:
 t=ds['text'][i]; ents=re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b',t); nums=re.findall(r'\b\d[\d,.:/%-]*\b',t); n=len(re.findall(r'[.!?]',t)); p=f'Rewrite this {n}-sentence paragraph. Preserve EXACTLY these entities: {ents}; numbers: {nums}; all negation, events, roles, and facts. Change syntax only (passive, nominalization, relative clauses). Keep length within 0.8-1.2x. Return only the rewritten paragraph.\n\n{t}'
 try:
  j=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':p}],'temperature':0.1,'max_tokens':1200,'thinking':{'type':'disabled'}},timeout=180).json();out=j['choices'][0]['message']['content'].strip()
 except Exception:out=''
 rows.append({'index':i,'source':t,'attack':out});print(i,bool(out))
o=root/'experiments/api_joint_strict5';o.mkdir(parents=True,exist_ok=True);json.dump(rows,open(o/'candidates.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
