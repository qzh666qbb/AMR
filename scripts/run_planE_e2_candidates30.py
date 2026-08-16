import json, re, requests
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_from_disk

ROOT=Path(__file__).resolve().parents[1]
KEY=(ROOT/'1.txt').read_text(encoding='utf-8').strip()
N=int(os.environ.get('E2_N','30')); K=4
OUT=ROOT/f'experiments/planE_e2_joint{N}'; OUT.mkdir(parents=True,exist_ok=True)
ds=load_from_disk(str(ROOT/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))
prompt_tpl='''Rewrite this entire paragraph into one meaning-preserving candidate for a robustness audit. Preserve every entity, number, date, polarity, modality, attribution, and event. Change syntax broadly (sentence boundary, voice, nominalization, clause attachment, reference). Keep similar length. Return only the rewritten paragraph.\n\nSOURCE:\n{}'''
def one(i,k):
 p=prompt_tpl.format(ds['text'][i])+'\nCandidate variant: '+str(k)
 try:
  r=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+KEY},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':p}],'temperature':0.7,'max_tokens':1400,'thinking':{'type':'disabled'}},timeout=240)
  out=r.json()['choices'][0]['message']['content'].strip()
 except Exception as e: out=''; print('ERR',i,k,type(e).__name__,flush=True)
 return {'index':i,'candidate':k,'source':ds['text'][i],'attack':out}
rows=[]
with ThreadPoolExecutor(max_workers=12) as ex:
 futs=[ex.submit(one,i,k) for i in range(N) for k in range(K)]
 for n,f in enumerate(as_completed(futs),1): rows.append(f.result()); print(f'{n}/{N*K}',flush=True)
json.dump({'n':N,'k':K,'rows':sorted(rows,key=lambda x:(x['index'],x['candidate']))},open(OUT/'candidates.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('saved',OUT/'candidates.json')
