import json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SWAN=ROOT/'third_party/SWAN'; sys.path.insert(0,str(SWAN))
from utils.bedrock_utils import BedrockManager
RUN=ROOT/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'
OUT=ROOT/'experiments/e4_ablation/candidates.json'
def parse(s):
 m=re.search(r'\{.*\}',s,re.S)
 if not m: raise ValueError('no json')
 x=json.loads(m.group(0))
 if set(x)!={'no_amr','amr_guided'}: raise ValueError('bad schema')
 return x
def main():
 data=json.load(open(ROOT/'experiments/e3_main/candidates.json',encoding='utf-8'))
 parsed=json.load(open(RUN/'parsed_amrs.json',encoding='utf-8'))
 done=json.load(open(OUT,encoding='utf-8'))['records'] if OUT.exists() else []
 manager=BedrockManager('us-east-1','deepseek-v4-flash'); OUT.parent.mkdir(parents=True,exist_ok=True)
 for i,r in enumerate(data['records'][:50]):
  if any(x['source_index']==r['source_index'] for x in done): continue
  prompt='''Return JSON with keys no_amr and amr_guided. Rewrite SOURCE into two candidates with the same number of sentences and approximately the same length. Preserve every fact, entity, number, date, polarity and attribution. no_amr must be an ordinary paraphrase without seeing AMR. amr_guided may use the supplied AMR to alter predicate packaging and clause structure, but must not split or merge sentences. SOURCE:\\n'''+r['source']+'\\nAMR:\\n'+'\\n'.join(parsed[r['source_index']])
  raw=manager.generate(prompt,'Return only valid JSON.',max_tokens=1200,temperature=.7,top_p=.9)
  x=parse(raw); done.append({'source_index':r['source_index'],'source':r['source'],'candidates':x})
  json.dump({'complete':False,'records':done},open(OUT,'w',encoding='utf-8'),indent=2,ensure_ascii=False); print(len(done),flush=True)
 json.dump({'complete':True,'records':done},open(OUT,'w',encoding='utf-8'),indent=2,ensure_ascii=False)
if __name__=='__main__': main()
