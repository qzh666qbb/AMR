from pathlib import Path
p=Path(__file__).with_name('run_planb_b3_combo50.py')
s=p.read_text(encoding='utf-8').replace("[:50]","[:250]").replace("planB_b3_combo50","planB_b7_combo250").replace("'paragraphs':50","'paragraphs':250")
exec(compile(s,str(p),'exec'))
