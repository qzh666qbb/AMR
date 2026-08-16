import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]; p=root/'experiments/planE_e2_joint250'
base={r['index']:r for r in json.load(open(p/'quality_aware_selected.json',encoding='utf-8'))}; adap={r['index']:r for r in json.load(open(p/'adaptive_retry/quality_valid_best.json',encoding='utf-8'))}; cons={r['index']:r for r in json.load(open(p/'conservative_review250.json',encoding='utf-8'))}; source_map={r['index']:r['source'] for r in json.load(open(p/'scored_candidates.json',encoding='utf-8'))}
for i,r in adap.items():
    if i not in base or r['z_score']<base[i]['z_score']: base[i]=r
success=[i for i,r in base.items() if r['z_score']<=2.33]; adapt=[i for i in adap if adap[i]['z_score']<=2.33]; detect_fail=[i for i,r in base.items() if r['z_score']>2.33]; no_candidate=[i for i in range(250) if i not in base]
def take(xs,n): return sorted(xs)[:n]
chosen=[]
for label,xs,n in [('原始/质量感知成功',success,20),('自适应补救成功',adapt,20),('检测未逃逸',detect_fail,10),('无合格候选',no_candidate,10)]:
    for i in take(xs,n):
        r=base.get(i) or {'source':source_map.get(i,''),'attack':'','z_score':''}; c=cons.get(i,{})
        chosen.append({'序号':len(chosen)+1,'原段落索引':i,'类别':label,'原文':r.get('source',''),'攻击文本':r.get('attack',''),'z_score':r.get('z_score',''),'初步事实':c.get('fact','待复核'),'初步实体':c.get('entity','待复核'),'初步关系':c.get('relation','待复核'),'初步否定模态':c.get('polarity','待复核'),'初步可读性':c.get('readable','待复核'),'我的初步结论':c.get('reason','待复核'),'人工事实(0/1)':'','人工实体(0/1)':'','人工关系(0/1)':'','人工否定模态(0/1)':'','人工可读性(1-5)':'','人工最终有效(0/1)':''})
used={r['原段落索引'] for r in chosen}
for i in range(250):
    if len(chosen)>=60: break
    if i in used: continue
    r=base.get(i) or {'source':source_map.get(i,''),'attack':'','z_score':''}; c=cons.get(i,{})
    chosen.append({'序号':len(chosen)+1,'原段落索引':i,'类别':'补充抽样','原文':r.get('source',''),'攻击文本':r.get('attack',''),'z_score':r.get('z_score',''),'初步事实':c.get('fact','待复核'),'初步实体':c.get('entity','待复核'),'初步关系':c.get('relation','待复核'),'初步否定模态':c.get('polarity','待复核'),'初步可读性':c.get('readable','待复核'),'我的初步结论':c.get('reason','待复核'),'人工事实(0/1)':'','人工实体(0/1)':'','人工关系(0/1)':'','人工否定模态(0/1)':'','人工可读性(1-5)':'','人工最终有效(0/1)':''})
out=p/'human_review_table_60.csv'
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(chosen[0])); w.writeheader(); w.writerows(chosen)
md=p/'human_review_table_60.md'
with md.open('w',encoding='utf-8') as f:
    f.write('# E2-250 分层复核表（60段）\n\n请逐行比较“原段落”和“攻击文本”。只有事实、实体、关系、否定/模态都保持，且可读性≥3，才在“人工最终有效”填1。\n\nCSV文件可用Excel打开：`human_review_table_60.csv`。\n\n分层：原始/质量感知成功20段、自适应补救成功20段、检测未逃逸10段、无合格候选10段。\n')
print(out, len(chosen))
