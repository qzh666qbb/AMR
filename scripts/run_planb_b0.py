import json
from pathlib import Path
import amrlib
from penman import decode

root=Path(__file__).resolve().parents[1]
parsed=json.load(open(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5/parsed_amrs.json',encoding='utf-8'))
out=root/'experiments/planB_b0'; out.mkdir(parents=True,exist_ok=True)
amrs=[s for doc in parsed for s in doc if isinstance(s,str) and s.strip().startswith('(')][:10]
gtos=amrlib.load_gtos_model(device='cuda')
generated=gtos.generate(amrs)
texts = generated[0] if isinstance(generated, tuple) else generated
stog=amrlib.load_stog_model(device='cuda')
reparsed=stog.parse_sents(texts)
rows=[]
for i,(a,t,r) in enumerate(zip(amrs,texts,reparsed)):
 try: decode(r); valid=True
 except Exception: valid=False
 rows.append({'index':i,'source_amr':a,'generated_text':t,'reparsed_amr':r,'reparse_valid':valid,'nonempty_text':bool(t.strip())})
json.dump({'n':len(rows),'valid':sum(x['reparse_valid'] for x in rows),'rows':rows},open(out/'b0_roundtrip.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print({'n':len(rows),'valid':sum(x['reparse_valid'] for x in rows)})
