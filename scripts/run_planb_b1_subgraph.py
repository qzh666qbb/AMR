import json
from pathlib import Path
import nltk, amrlib, penman, torch
from datasets import load_from_disk, Dataset
root=Path(__file__).resolve().parents[1]
ds=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))
paras=ds['text'][:10]; groups=[nltk.sent_tokenize(p) for p in paras]; flat=[s for g in groups for s in g]
stog=amrlib.load_stog_model(device='cuda'); raw=[stog.parse_sents([s])[0] for s in flat]; del stog; torch.cuda.empty_cache()
transformed=[]; changed=[]
for a in raw:
 g=penman.decode(a); triples=list(g.triples); groups_by={}
 for i,(src,role,tgt) in enumerate(triples):
  if role != ':instance': groups_by.setdefault(src,[]).append(i)
 hit=False
 for inds in groups_by.values():
  if len(inds)>=2:
   vals=[triples[i] for i in inds]; vals.reverse()
   for i,val in zip(inds,vals): triples[i]=val
   hit=True; break
 transformed.append(penman.encode(penman.Graph(triples,top=g.top))); changed.append(hit)
gtos=amrlib.load_gtos_model(device='cuda'); gen=gtos.generate(transformed); gen=gen[0] if isinstance(gen,tuple) else gen; del gtos; torch.cuda.empty_cache()
attack=[]; cur=0
for g in groups: attack.append(' '.join(gen[cur:cur+len(g)])); cur+=len(g)
out=root/'experiments/planB_b1_subgraph'; out.mkdir(parents=True,exist_ok=True)
Dataset.from_dict({'text':attack}).save_to_disk(str(out/'dataset'))
json.dump({'paragraphs':10,'source_sentences':len(flat),'changed_sentences':sum(changed),'texts':attack},open(out/'subgraph_transform.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print({'paragraphs':10,'source_sentences':len(flat),'changed_sentences':sum(changed)})
