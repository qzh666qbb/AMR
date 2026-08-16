import json
from pathlib import Path
import nltk, amrlib, penman, torch
from datasets import load_from_disk, Dataset
root=Path(__file__).resolve().parents[1]
ds=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))
paras=ds['text'][:30]; groups=[nltk.sent_tokenize(p) for p in paras]; flat=[s for g in groups for s in g]
stog=amrlib.load_stog_model(device='cuda'); raw=[stog.parse_sents([s])[0] for s in flat]; del stog; torch.cuda.empty_cache()
trans=[]; rec=[]
for a in raw:
 g=penman.decode(a); ts=list(g.triples); inds=[i for i,(s,r,t) in enumerate(ts) if r==':mod' and isinstance(t,str) and t in {x for x,y,z in ts if y==':instance'}]
 hit=False
 if len(inds)>=2:
  i,j=inds[:2]; si,ri,ti=ts[i]; sj,rj,tj=ts[j]; ci=next((z for x,y,z in ts if x==ti and y==':instance'),None); cj=next((z for x,y,z in ts if x==tj and y==':instance'),None)
  if ci and cj:
   ts[i]=(si,ri,tj); ts[j]=(sj,rj,ti); hit=True
 trans.append(penman.encode(penman.Graph(ts,top=g.top))); rec.append(hit)
gtos=amrlib.load_gtos_model(device='cuda'); gen=gtos.generate(trans); gen=gen[0] if isinstance(gen,tuple) else gen; del gtos; torch.cuda.empty_cache()
attack=[]; cur=0
for g in groups: attack.append(' '.join(gen[cur:cur+len(g)])); cur+=len(g)
out=root/'experiments/planC_modifier_swap30'; out.mkdir(parents=True,exist_ok=True)
Dataset.from_dict({'text':attack}).save_to_disk(str(out/'dataset'))
json.dump({'paragraphs':30,'source_sentences':len(flat),'changed_sentences':sum(rec),'records':rec},open(out/'transform.json','w'),indent=2)
print({'paragraphs':30,'source_sentences':len(flat),'changed_sentences':sum(rec)})
