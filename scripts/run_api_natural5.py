import json,requests
from pathlib import Path
root=Path(__file__).resolve().parents[1]; key=(root/'1.txt').read_text().strip(); cand=json.load(open(root/'experiments/natural_sentence_candidates.json',encoding='utf-8'))[:5]; out=[]
for r in cand:
 p='Rewrite this sentence in a natural passive, nominalized, or relative-clause form. Preserve every fact, entity, number, date, negation, and meaning. Return only one sentence:\n'+r['text']
 try:
  j=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':p}],'temperature':0.1,'max_tokens':180,'thinking':{'type':'disabled'}},timeout=120).json(); r['attack']=j['choices'][0]['message']['content'].strip()
 except Exception as e: r['attack']=''; r['error']=type(e).__name__
 out.append(r)
d=root/'experiments/api_natural5';d.mkdir(parents=True,exist_ok=True);json.dump(out,open(d/'candidates.json','w',encoding='utf-8'),ensure_ascii=False,indent=2);print([(r['paragraph'],bool(r.get('attack'))) for r in out])
