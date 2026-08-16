from pathlib import Path
p=Path(__file__).with_name('run_planc_modifier_family30.py')
s=p.read_text(encoding='utf-8').replace("r in (':mod',':location',':time',':quant')","r in (':mod',':location',':time',':quant',':ARG1',':ARG2')").replace('planC_modifier_family30','planC_mod_arg_combo30')
exec(compile(s,str(p),'exec'))
