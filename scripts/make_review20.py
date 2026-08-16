import csv,json
from pathlib import Path
import numpy as np
from datasets import load_from_disk
root=Path(__file__).resolve().parents[1]; z=np.load(root/'experiments/api_full_paragraphs/detection/machine_z_scores.npy'); ids=json.load(open(root/'experiments/api_full_paragraphs/meta.json'))['source_indices']; base=load_from_disk(str(root/'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))['text']; atk=load_from_disk(str(root/'experiments/api_full_paragraphs/dataset'))['text']; rows=[]
for j in np.argsort(z)[:20]: rows.append({'rank':int(j+1),'source_index':ids[j],'z_score':float(z[j]),'source_text':base[ids[j]],'attack_text':atk[j],'quality_1_5':'','semantic_preserved':'','major_error':'','notes':''})
with open(root/'experiments/api_full_paragraphs/human_review20.csv','w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
