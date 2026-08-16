import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'third_party/SWAN'))
from utils.amr_utils import normalize_amr_variables, generate_template_amr

source = json.load(open(root / 'experiments/planB_b1/b1_node_transform.json', encoding='utf-8'))
output = []
for row in source['rows']:
    raw = row['reparsed_amr']
    templated = generate_template_amr(normalize_amr_variables(raw))
    output.append([normalize_amr_variables(templated)])
path = root / 'experiments/planB_b1/parsed_amrs.json'
json.dump(output, open(path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(path)
