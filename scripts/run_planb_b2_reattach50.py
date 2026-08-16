import json
from pathlib import Path
import nltk, amrlib, penman, torch
from datasets import load_from_disk, Dataset

root=Path(__file__).resolve().parents[1]
ds=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))
paragraphs=ds['text'][:50]; groups=[nltk.sent_tokenize(p) for p in paragraphs]; flat=[s for g in groups for s in g]
stog=amrlib.load_stog_model(device='cuda'); raw=[stog.parse_sents([s])[0] for s in flat]; del stog; torch.cuda.empty_cache()
trans=[]; records=[]
for a in raw:
 g=penman.decode(a); ts=list(g.triples); vars={s for s,r,t in ts if r==':instance'}; hit=False; old=None
 for i,(s,r,t) in enumerate(ts):
  if r==':mod' and s!=g.top and s in vars:
   ts[i]=(g.top,r,t); hit=True; old=s; break
 trans.append(penman.encode(penman.Graph(ts,top=g.top))); records.append({'changed':hit,'old_parent':old,'new_parent':g.top if hit else None})
gtos=amrlib.load_gtos_model(device='cuda'); gen=gtos.generate(trans); gen=gen[0] if isinstance(gen,tuple) else gen; del gtos; torch.cuda.empty_cache()
attack=[]; cur=0
for g in groups: attack.append(' '.join(gen[cur:cur+len(g)])); cur+=len(g)
out=root/'experiments/planB_b2_reattach50'; out.mkdir(parents=True,exist_ok=True)
Dataset.from_dict({'text':attack}).save_to_disk(str(out/'dataset'))
json.dump({'paragraphs':50,'source_sentences':len(flat),'changed_sentences':sum(x['changed'] for x in records),'records':records},open(out/'transform.json','w'),indent=2)
print({'paragraphs':50,'source_sentences':len(flat),'changed_sentences':sum(x['changed'] for x in records)})
