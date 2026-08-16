from pathlib import Path
p=Path(__file__).with_name('run_api_syntax30.py')
s=p.read_text(encoding='utf-8').replace('[:30]','[:5]').replace("'temperature':0.2","'temperature':0.2,'max_tokens':1200").replace('api_syntax30','api_syntax5')
exec(compile(s,str(p),'exec'))
