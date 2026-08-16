import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]; p=root/'experiments/planE_e2_joint250'
reviews={r['index']:r for r in json.load(open(p/'conservative_review250.json',encoding='utf-8'))}
src=p/'human_review_table_60.csv'; dst=p/'reviewer1_filled_60.csv'; summary=p/'reviewer1_summary_60.md'
with src.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
valid_count=0
for row in rows:
 i=int(row['原段落索引']); r=reviews.get(i,{})
 fact=1 if r.get('fact',False) else 0; entity=1 if r.get('entity',False) else 0; relation=1 if r.get('relation',False) else 0; polarity=1 if r.get('polarity',False) else 0; readable=5 if r.get('readable',False) else max(1,min(2,int(r.get('score',1) or 1)))
 final=int(fact and entity and relation and polarity and readable>=3 and bool(row.get('攻击文本','').strip()))
 row['人工事实(0/1)']=fact; row['人工实体(0/1)']=entity; row['人工关系(0/1)']=relation; row['人工否定模态(0/1)']=polarity; row['人工可读性(1-5)']=readable; row['人工最终有效(0/1)']=final; row['我的初步结论']=r.get('reason',row.get('我的初步结论',''))
 valid_count+=final
with dst.open('w',encoding='utf-8-sig',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
low_valid=sum(int(r['人工最终有效(0/1)']) and r.get('z_score','') not in ('',None) and float(r['z_score'])<=2.33 for r in rows)
with summary.open('w',encoding='utf-8') as f:
 f.write(f'# 评审者1：60段首轮评价\n\n- 总样本：{len(rows)}\n- 严格有效：{valid_count}\n- 严格有效且规避检测：{low_valid}\n- 判定口径：事实、实体、关系、否定/模态均为1，且可读性≥3。\n\n说明：这是Codex基于保守规则完成的模型辅助首轮评价，不能替代论文要求的第二位独立真人标注者。\n')
print('rows',len(rows),'valid',valid_count,'valid_and_low',low_valid)
