import json
from pathlib import Path
import nltk
import amrlib
import penman
import torch
from datasets import load_from_disk, Dataset

root = Path(__file__).resolve().parents[1]
dataset = load_from_disk(str(root / 'baselines/local/swan_acl2026_realnews/runs/formal_250x5'))
paragraphs = dataset['text'][:10]
mapping = {'person': 'individual', 'organization': 'institution', 'company': 'business', 'city': 'municipality', 'event': 'occurrence', 'report': 'publication'}
sentences = [nltk.sent_tokenize(p) for p in paragraphs]
flat = [s for group in sentences for s in group]

stog = amrlib.load_stog_model(device='cuda')
raw_amrs = [stog.parse_sents([s])[0] for s in flat]
del stog
torch.cuda.empty_cache()

transformed = []
changed = []
for amr in raw_amrs:
    graph = penman.decode(amr)
    triples = []
    hit = False
    for source, role, target in graph.triples:
        if role == ':instance' and target in mapping:
            triples.append((source, role, mapping[target]))
            hit = True
        else:
            triples.append((source, role, target))
    transformed.append(penman.encode(penman.Graph(triples, top=graph.top)))
    changed.append(hit)

gtos = amrlib.load_gtos_model(device='cuda')
generated = gtos.generate(transformed)
generated = generated[0] if isinstance(generated, tuple) else generated
del gtos
torch.cuda.empty_cache()

attack_paragraphs = []
cursor = 0
for group in sentences:
    count = len(group)
    attack_paragraphs.append(' '.join(generated[cursor:cursor + count]))
    cursor += count

out = root / 'experiments/planB_b1'
out.mkdir(parents=True, exist_ok=True)
Dataset.from_dict({'text': attack_paragraphs}).save_to_disk(str(out / 'dataset'))
json.dump({'paragraphs': len(paragraphs), 'source_sentences': len(flat), 'changed_sentences': sum(changed), 'texts': attack_paragraphs}, open(out / 'b1_sentence_level.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print({'paragraphs': len(paragraphs), 'source_sentences': len(flat), 'changed_sentences': sum(changed)})
