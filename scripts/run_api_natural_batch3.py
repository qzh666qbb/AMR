from pathlib import Path
p=Path(__file__).with_name('run_api_natural5.py')
s=p.read_text(encoding='utf-8').replace('[:5]','[10:15]').replace('api_natural5','api_natural_batch3')
exec(compile(s,str(p),'exec'))
