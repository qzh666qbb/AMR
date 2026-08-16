import json
from pathlib import Path
from datasets import load_from_disk,Dataset
import nltk
root=Path(__file__).resolve().parents[1]; base=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))['text'][:250]; rows=[]
for folder in ['api_natural5','api_natural_batch2','api_natural_batch3','api_natural_batch4','api_natural_batch5','api_natural_batch6']:
 rows += json.load(open(root/'experiments'/folder/'candidates.json',encoding='utf-8'))
groups={};
for r in rows: groups.setdefault(r['paragraph'],{})[r['sentence']]=r['attack']
out=[]
for p,rep in sorted(groups.items()):
 ss=nltk.sent_tokenize(base[p]); out.append(' '.join(rep.get(i,s) for i,s in enumerate(ss)))
d=root/'experiments/api_natural_paragraph30';d.mkdir(parents=True,exist_ok=True);Dataset.from_dict({'text':out}).save_to_disk(str(d/'dataset'));json.dump({'paragraphs':len(out),'source_indices':sorted(groups)},open(d/'meta.json','w'),indent=2);print(len(out))
