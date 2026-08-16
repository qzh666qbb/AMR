import json,re,requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
root=Path(__file__).resolve().parents[1]; base=root/'experiments/planE_e2_joint250'; key=(root/'1.txt').read_text().strip()
base_rows={r['index']:r for r in json.load(open(base/'quality_aware_selected.json',encoding='utf-8'))}
for r in json.load(open(base/'adaptive_retry/quality_valid_best.json',encoding='utf-8')):
 if r['index'] not in base_rows or r['z_score']<base_rows[r['index']]['z_score']: base_rows[r['index']]=r
rows=[base_rows.get(i) for i in range(250)]
def one(item):
 i,r=item
 if not r: return {'index':i,'valid':False,'score':0,'reason':'no quality-passing candidate'}
 q='''You are a conservative independent reviewer. Compare ATTACK to SOURCE. Reject unless every explicit fact, entity identity, event participant, relation, number, date, polarity, modality and attribution is preserved. Reject additions, omissions, vague substitutions (something/unspecified), entity ambiguity, and awkward text that no longer communicates the source precisely. Return JSON only: {"valid":true/false,"score":0-5,"fact":true/false,"entity":true/false,"relation":true/false,"polarity":true/false,"readable":true/false,"reason":"concise"}.\nSOURCE:\n'''+r['source']+'\nATTACK:\n'+r['attack']
 try:
  j=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':q}],'temperature':0,'max_tokens':300,'thinking':{'type':'disabled'}},timeout=240).json(); obj=json.loads(re.search(r'\{.*\}',j['choices'][0]['message']['content'],re.S).group(0)); return {'index':i,'candidate':r['candidate'],'z_score':r['z_score'],**obj}
 except Exception as e: return {'index':i,'valid':False,'score':0,'reason':type(e).__name__}
with ThreadPoolExecutor(max_workers=10) as ex: out=list(ex.map(one,enumerate(rows)))
json.dump(out,open(base/'conservative_review250.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
valid=[x for x in out if x.get('valid') and all(x.get(k,False) for k in ['fact','entity','relation','polarity','readable'])]
print('conservative_valid',len(valid),'/250','low',sum(x.get('z_score',99)<=2.33 for x in valid))
