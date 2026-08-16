from pathlib import Path
p=Path(__file__).with_name('run_api_natural5.py')
s=p.read_text(encoding='utf-8').replace('[:5]','[15:20]').replace('api_natural5','api_natural_batch4')
exec(compile(s,str(p),'exec'))
