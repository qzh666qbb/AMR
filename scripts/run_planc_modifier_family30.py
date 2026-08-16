from pathlib import Path
p=Path(__file__).with_name('run_planc_modifier_swap30.py')
s=p.read_text(encoding='utf-8').replace("r==':mod'","r in (':mod',':location',':time',':quant')").replace('planC_modifier_swap30','planC_modifier_family30')
s=s.replace("trans.append(penman.encode(penman.Graph(ts,top=g.top))); rec.append(hit)","\n try:\n  encoded=penman.encode(penman.Graph(ts,top=g.top))\n except Exception:\n  encoded=a; hit=False\n trans.append(encoded); rec.append(hit)")
exec(compile(s,str(p),'exec'))
