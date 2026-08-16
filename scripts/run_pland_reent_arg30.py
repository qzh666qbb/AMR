from pathlib import Path
p=Path(__file__).with_name('run_pland_reentrancy30.py')
s=p.read_text(encoding='utf-8').replace('planD_reentrancy30','planD_reent_arg30')
needle='try: encoded=penman.encode(penman.Graph(ts,top=g.top)) if hit else a'
inject="""\n if hit:\n  ai=[i for i,(ss,rr,tt) in enumerate(ts) if rr in (':ARG1',':ARG2') and isinstance(tt,str)]\n  if len(ai)>=2:\n   i,j=ai[:2]; ss,rr,tt=ts[i]; sj,rj,tj=ts[j]; ts[i]=(ss,rr,tj); ts[j]=(sj,rj,tt)\n try: encoded=penman.encode(penman.Graph(ts,top=g.top)) if hit else a"""
s=s.replace(needle,inject)
exec(compile(s,str(p),'exec'))
