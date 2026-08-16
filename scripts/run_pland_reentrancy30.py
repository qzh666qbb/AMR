import json
from pathlib import Path
import nltk, amrlib, penman, torch
from datasets import load_from_disk, Dataset
root=Path(__file__).resolve().parents[1]; ds=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))
paras=ds['text'][:30]; groups=[nltk.sent_tokenize(p) for p in paras]; flat=[s for g in groups for s in g]
stog=amrlib.load_stog_model(device='cuda'); raw=[stog.parse_sents([s])[0] for s in flat]; del stog; torch.cuda.empty_cache()
trans=[]; rec=[]
for a in raw:
 g=penman.decode(a); ts=list(g.triples); indeg={}
 for s,r,t in ts:
  if r!=':instance' and isinstance(t,str) and any(x==t and y==':instance' for x,y,z in ts): indeg.setdefault(t,[]).append((s,r))
 hit=False
 for v,edges in indeg.items():
  concept=next((t for s,r,t in ts if s==v and r==':instance'), '')
  if len(edges)>=2 and not any(k in concept for k in ('name','date','quantity','temporal','negative')):
   src,role=edges[-1]; nv=v+'z'
   if not any(s==nv for s,r,t in ts):
    ts=[(nv,':instance',concept) if (s==v and r==':instance') else (s,r,nv) if (s==src and r==role and t==v) else (s,r,t) for s,r,t in ts]; hit=True
   break
 try: encoded=penman.encode(penman.Graph(ts,top=g.top)) if hit else a
 except Exception: encoded=a; hit=False
 trans.append(encoded); rec.append(hit)
gtos=amrlib.load_gtos_model(device='cuda'); gen=gtos.generate(trans); gen=gen[0] if isinstance(gen,tuple) else gen; del gtos; torch.cuda.empty_cache()
attack=[];cur=0
for g in groups: attack.append(' '.join(gen[cur:cur+len(g)]));cur+=len(g)
out=root/'experiments/planD_reentrancy30';out.mkdir(parents=True,exist_ok=True);Dataset.from_dict({'text':attack}).save_to_disk(str(out/'dataset'));json.dump({'paragraphs':30,'source_sentences':len(flat),'changed_sentences':sum(rec)},open(out/'transform.json','w'),indent=2);print({'paragraphs':30,'source_sentences':len(flat),'changed_sentences':sum(rec)})
