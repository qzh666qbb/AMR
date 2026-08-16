import json,requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk
root=Path(__file__).resolve().parents[1]; key=(root/'1.txt').read_text().strip(); out=root/'experiments/planE_e3_combo30'; out.mkdir(parents=True,exist_ok=True); ds=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))
def one(x):
 i,k=x; q='Create one strong but meaning-preserving paragraph rewrite. Simulate cross-lingual round-trip paraphrasing, then alter sentence boundaries, voice, nominalization, clause attachment, and reference structure to induce a different AMR parse. Preserve every entity, number, date, polarity, modality, attribution, and event. Similar length. Return only the paragraph.\nSOURCE:\n'+ds['text'][i]+f'\nVariant {k}'
 try:
  j=requests.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+key},json={'model':'deepseek-v4-flash','messages':[{'role':'user','content':q}],'temperature':0.8,'max_tokens':1400,'thinking':{'type':'disabled'}},timeout=240).json(); a=j['choices'][0]['message']['content'].strip()
 except Exception: a=''
 return {'index':i,'candidate':k,'source':ds['text'][i],'attack':a}
with ThreadPoolExecutor(max_workers=12) as ex: rows=list(ex.map(one,[(i,k) for i in range(30) for k in range(4)]))
json.dump({'rows':rows},open(out/'candidates.json','w',encoding='utf-8'),ensure_ascii=False,indent=2); print(len(rows))
