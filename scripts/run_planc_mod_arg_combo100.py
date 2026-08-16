from pathlib import Path
p=Path(__file__).with_name('run_planc_modifier_swap30.py')
s=p.read_text(encoding='utf-8').replace("r==':mod'","r in (':mod',':location',':time',':quant',':ARG1',':ARG2')").replace('[:30]','[:100]').replace('planC_modifier_swap30','planC_mod_arg_combo100').replace("'paragraphs':30","'paragraphs':100")
s=s.replace("trans.append(penman.encode(penman.Graph(ts,top=g.top))); rec.append(hit)","\n try:\n  encoded=penman.encode(penman.Graph(ts,top=g.top))\n except Exception:\n  encoded=a; hit=False\n trans.append(encoded); rec.append(hit)")
exec(compile(s,str(p),'exec'))
