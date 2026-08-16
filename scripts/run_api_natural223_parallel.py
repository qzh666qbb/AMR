import json,time,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datasets import load_from_disk
root=Path(__file__).resolve().parents[1]; key=(root/'1.txt').read_text().strip(); cand=json.load(open(root/'experiments/natural_sentence_candidates.json',encoding='utf-8')); out=root/'experiments/api_natural223';out.mkdir(parents=True,exist_ok=True); cp=out/'results.jsonl'; done={}
if cp.exists():
 for line in cp.read_text(encoding='utf-8').splitlines():
  try:
   r=json.loads(line);done[r['id']]=r
  except: pass
def call(item):
 i=item['paragraph']*10000+item['sentence']; p='Rewrite only this sentence naturally using varied syntax. Preserve every fact, entity, number, date, negation and meaning; return only one sentence:\n'+item['text']
 for k in range(3):
  try:
   j=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':p}],'temperature':0.1,'max_tokens':180,'thinking':{'type':'disabled'}},timeout=120);j.raise_for_status(); a=j.json()['choices'][0]['message']['content'].strip();
   if a:return {'id':i,**item,'attack':a,'ok':True}
  except Exception as e:
   if k==2:return {'id':i,**item,'attack':'','ok':False,'error':type(e).__name__}
  time.sleep(2**k)
todo=[x for x in cand if x['paragraph']*10000+x['sentence'] not in done]; print({'total':len(cand),'already':len(done),'todo':len(todo)})
with ThreadPoolExecutor(max_workers=12) as ex, cp.open('a',encoding='utf-8') as f:
 for n,fu in enumerate(as_completed([ex.submit(call,x) for x in todo]),1):
  r=fu.result();f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
  if n%10==0:print('completed',n)
