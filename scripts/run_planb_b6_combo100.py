from pathlib import Path
p=Path(__file__).with_name('run_planb_b3_combo50.py')
s=p.read_text(encoding='utf-8').replace("[:50]","[:100]").replace("planB_b3_combo50","planB_b6_combo100").replace("'paragraphs':50","'paragraphs':100")
exec(compile(s,str(p),'exec'))
