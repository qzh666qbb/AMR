import json
from pathlib import Path
from datasets import load_from_disk,Dataset
import nltk
root=Path(__file__).resolve().parents[1]; base=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))['text']; rec=json.load(open(root/'experiments/api_natural223/records.json',encoding='utf-8')); q=json.load(open(root/'experiments/api_natural223/quality.json')); groups={}
for r,qq in zip(rec,q):
 if qq['pass']: groups.setdefault(r['paragraph'],{})[r['sentence']]=r['attack']
out=[]; ids=[]
for p,rep in sorted(groups.items()):
 ss=nltk.sent_tokenize(base[p]); out.append(' '.join(rep.get(i,s) for i,s in enumerate(ss)));ids.append(p)
d=root/'experiments/api_full_paragraphs';d.mkdir(parents=True,exist_ok=True);Dataset.from_dict({'text':out}).save_to_disk(str(d/'dataset'));json.dump({'source_indices':ids,'paragraphs':len(out)},open(d/'meta.json','w'),indent=2);print({'paragraphs':len(out),'replaced_sentences':sum(len(x) for x in groups.values())})
