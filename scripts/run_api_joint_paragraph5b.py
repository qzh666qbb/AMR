from pathlib import Path
p=Path(__file__).with_name('run_api_joint_paragraph5.py')
s=p.read_text(encoding='utf-8').replace('ids=[3,6,10,13,20]','ids=[26,27,28,39,40]').replace('api_joint_paragraph5','api_joint_paragraph5b')
exec(compile(s,str(p),'exec'))
