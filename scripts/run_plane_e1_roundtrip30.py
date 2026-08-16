import json,time,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datasets import load_from_disk
root=Path(__file__).resolve().parents[1]; key=(root/'1.txt').read_text().strip(); texts=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))['text'][:30]; out=root/'experiments/planE_e1_roundtrip';out.mkdir(parents=True,exist_ok=True); cp=out/'results.jsonl'; done={}
if cp.exists():
 for line in cp.read_text(encoding='utf-8').splitlines():
  try:
   r=json.loads(line);done[(r['index'],r['lang'])]=r
  except: pass
def ask(prompt):
 j=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':prompt}],'temperature':0.1,'max_tokens':1400,'thinking':{'type':'disabled'}},timeout=180);j.raise_for_status();return j.json()['choices'][0]['message']['content'].strip()
def call(job):
 i,lang=job; t=texts[i]; mid='Chinese' if lang=='zh' else 'German';
 for k in range(3):
  try:
   m=ask(f'Translate this paragraph into natural {mid}. Preserve every fact, entity, number, date, negation, and sentence count. Return only the translation.\n\n{t}')
   back=ask(f'Translate this {mid} paragraph back into natural English. Preserve every fact, entity, number, date, negation, and sentence count. Return only the English paragraph.\n\n{m}')
   return {'index':i,'lang':lang,'source':t,'intermediate':m,'attack':back,'ok':True}
  except Exception as e:
   if k==2:return {'index':i,'lang':lang,'source':t,'intermediate':'','attack':'','ok':False,'error':type(e).__name__}
   time.sleep(2**k)
jobs=[(i,l) for i in range(30) for l in ('zh','de') if (i,l) not in done];print({'total':60,'already':len(done),'todo':len(jobs)})
with ThreadPoolExecutor(max_workers=12) as ex,cp.open('a',encoding='utf-8') as f:
 for n,fu in enumerate(as_completed([ex.submit(call,j) for j in jobs]),1):
  r=fu.result();f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
  if n%10==0:print('completed',n)
