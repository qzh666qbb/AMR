import json
from pathlib import Path
import nltk, amrlib, penman, torch
from datasets import load_from_disk, Dataset
root=Path(__file__).resolve().parents[1]
ds=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))
paras=ds['text'][:50]; groups=[nltk.sent_tokenize(p) for p in paras]; flat=[s for g in groups for s in g]
mapping={'person':'individual','organization':'institution','company':'business','city':'municipality','event':'occurrence','report':'publication'}
stog=amrlib.load_stog_model(device='cuda'); raw=[stog.parse_sents([s])[0] for s in flat]; del stog; torch.cuda.empty_cache()
trans=[]; rec=[]
for a in raw:
 g=penman.decode(a); ts=[]; node=False; reatt=False; old=None
 for s,r,t in g.triples:
  if r==':instance' and t in mapping: ts.append((s,r,mapping[t])); node=True
  else: ts.append((s,r,t))
 vars={s for s,r,t in ts if r==':instance'}
 for i,(s,r,t) in enumerate(ts):
  if r==':mod' and s!=g.top and s in vars: ts[i]=(g.top,r,t); reatt=True; old=s; break
 trans.append(penman.encode(penman.Graph(ts,top=g.top))); rec.append({'node':node,'reattach':reatt,'old_parent':old})
gtos=amrlib.load_gtos_model(device='cuda'); gen=gtos.generate(trans); gen=gen[0] if isinstance(gen,tuple) else gen; del gtos; torch.cuda.empty_cache()
attack=[]; cur=0
for g in groups: attack.append(' '.join(gen[cur:cur+len(g)])); cur+=len(g)
out=root/'experiments/planB_b3_combo50'; out.mkdir(parents=True,exist_ok=True)
Dataset.from_dict({'text':attack}).save_to_disk(str(out/'dataset'))
json.dump({'paragraphs':50,'source_sentences':len(flat),'node_changed':sum(x['node'] for x in rec),'reattach_changed':sum(x['reattach'] for x in rec),'records':rec},open(out/'transform.json','w'),indent=2)
print({'paragraphs':50,'source_sentences':len(flat),'node_changed':sum(x['node'] for x in rec),'reattach_changed':sum(x['reattach'] for x in rec)})
