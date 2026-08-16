import json,requests,re,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parents[1]; n=os.environ.get('E2_N','30'); exp_name=os.environ.get('REVIEW_EXPERIMENT',f'planE_e2_joint{n}'); p=root/'experiments'/exp_name; key=(root/'1.txt').read_text().strip(); rows=json.load(open(p/'best_by_paragraph.json',encoding='utf-8'))
def one(r):
 q='Evaluate whether ATTACK preserves the meaning of SOURCE. Score 1-5 (5=fully preserved), and mark major_error true only for changed entities, numbers, polarity, events, or relations. Return JSON only: {"score":number,"major_error":boolean,"reason":"short"}.\nSOURCE:\n'+r['source']+'\nATTACK:\n'+r['attack']
 try:
  j=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':q}],'temperature':0.0,'max_tokens':180,'thinking':{'type':'disabled'}},timeout=180).json(); t=j['choices'][0]['message']['content']; v=json.loads(re.search(r'\{.*\}',t,re.S).group(0))
 except Exception as e: v={'score':0,'major_error':True,'reason':type(e).__name__}
 return {**r,'review':v}
with ThreadPoolExecutor(max_workers=10) as ex: out=list(ex.map(one,rows))
json.dump(out,open(p/'llm_review.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('valid>=4',sum(x['review'].get('score',0)>=4 and not x['review'].get('major_error',True) for x in out),'major',sum(x['review'].get('major_error',True) for x in out))
