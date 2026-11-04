"""句子边界编辑的 L1/L2/L4 校验：切点枚举、受限执行、保真度量。

设计约束：只允许改动标点与功能词；实词的集合与相对顺序必须完全不变。
所有判定都是确定性的，不读取任何检测分数。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re

TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\sA-Za-z0-9]")

PUNCT = set(".,;:!?—-…\"'()")
CONNECTIVES = {
    "and", "but", "or", "nor", "so", "yet", "while", "because", "although",
    "though", "since", "however", "therefore", "thus", "meanwhile", "instead",
    "still", "then", "also",
}
PRONOUNS = {"it", "this", "they", "these", "those", "he", "she", "that",
            "which", "who", "whom", "whose", "them", "him", "her"}
ARTICLES = {"the", "a", "an"}

FUNCTION_WORDS = set("""
a an the and or but nor so yet for of to in on at by with from as than that this
these those it its they them their there here is are was were be been being am do
does did have has had having will would shall should can could may might must not
no never if then when while because although though since until unless after before
during between among into over under about which who whom whose what where why how
all any both each more most other some such only own same too very just also however
therefore thus meanwhile instead still even ever once again i you we us our your my
me he she him her his their theirs s t
""".split())

REMOVABLE = CONNECTIVES | PRONOUNS | ARTICLES | {"which", "who", "whom", "whose"}


@dataclass(frozen=True)
class Edit:
    op: str          # replace | insert | delete
    start: int
    end: int
    text: str

    def as_json(self):
        return asdict(self)


def tokens(text):
    return TOKEN.findall(text)


def content_sequence(text):
    """实词序列：排除功能词与标点后的有序小写词表。"""
    return [t.lower() for t in tokens(text)
            if re.fullmatch(r"[A-Za-z0-9']+", t) and t.lower() not in FUNCTION_WORDS]


def apply_edits(text, edits):
    """从右向左执行，避免偏移失效；重叠即报错。"""
    ordered = sorted(edits, key=lambda e: (e.start, e.end), reverse=True)
    last = len(text) + 1
    out = text
    for edit in ordered:
        if not (0 <= edit.start <= edit.end <= len(text)) or edit.end > last:
            raise ValueError("编辑区间非法或重叠")
        out = out[:edit.start] + edit.text + out[edit.end:]
        last = edit.start
    return out


def _key(token):
    """标点按原样比较；词大小写不敏感，避免首字母改动被误判为改词。"""
    return token if token in PUNCT else token.lower()


def validate(original, edited):
    """L2 契约：实词序列完全不变；新增只允许标点/功能词；删除只允许标点/功能词。"""
    added = Counter(_key(t) for t in tokens(edited)) - Counter(_key(t) for t in tokens(original))
    removed = Counter(_key(t) for t in tokens(original)) - Counter(_key(t) for t in tokens(edited))
    illegal_added = sorted({t for t in added if not (
        t in PUNCT or t in CONNECTIVES | PRONOUNS | ARTICLES)})
    illegal_removed = sorted({t for t in removed if not (
        t in PUNCT or t in REMOVABLE)})
    same_content = content_sequence(original) == content_sequence(edited)
    return {
        "content_sequence_equal": same_content,
        "added_tokens": sorted(added),
        "removed_tokens": sorted(removed),
        "illegal_added": illegal_added,
        "illegal_removed": illegal_removed,
        "passed": same_content and not illegal_added and not illegal_removed,
    }


def _capitalize_at(text, offset):
    """在原文坐标上，把 offset 之后第一个字母改成大写，作为显式编辑记录。"""
    index = offset
    while index < len(text) and not text[index].isalpha():
        index += 1
    if index < len(text) and text[index].islower():
        return Edit("replace", index, index + 1, text[index].upper())
    return None


def split_candidates(text, doc):
    """L1：只在合法句法边界给出切点，附带风险等级；同一结果只保留一次。"""
    raw = []
    for token in doc:
        if token.dep_ in ("cc", "conj") and token.pos_ in ("CCONJ", "VERB", "AUX"):
            after = token.idx + len(token.text) + 1
            edits = [Edit("replace", token.idx, after, ". ")]
            cap = _capitalize_at(text, after)
            if cap:
                edits.append(cap)
            rule = "coordinating_conjunction" if token.dep_ == "cc" else "verb_coordination"
            raw.append((rule, "low" if token.dep_ == "cc" else "medium", edits))
        elif token.dep_ == "mark" and token.head.dep_ == "advcl":
            before = text[:token.idx]
            cut = len(before) - len(before.rstrip(", "))
            if cut <= 0:
                continue
            edits = [Edit("replace", cut, token.idx, ". ")]
            cap = _capitalize_at(text, token.idx)
            if cap:
                edits.append(cap)
            raw.append(("subordinating_clause", "low", edits))
        elif token.dep_ == "advcl" and token.head.dep_ == "ROOT" and token.idx > 0:
            before = text[:token.idx]
            cut = len(before) - len(before.rstrip(", "))
            if cut <= 0:
                continue
            edits = [Edit("replace", cut, token.idx, ". ")]
            cap = _capitalize_at(text, token.idx)
            if cap:
                edits.append(cap)
            raw.append(("adverbial_clause", "medium", edits))
        elif token.dep_ == "parataxis" and token.idx > 0:
            before = text[:token.idx]
            cut = len(before) - len(before.rstrip(", "))
            if cut <= 0:
                continue
            edits = [Edit("replace", cut, token.idx, ". ")]
            cap = _capitalize_at(text, token.idx)
            if cap:
                edits.append(cap)
            raw.append(("parataxis", "medium", edits))
        elif token.dep_ == "relcl" and token.lower_ in ("which", "who"):
            before = text[:token.idx]
            if not before.rstrip().endswith(","):
                continue
            cut = len(before.rstrip()) - 1
            edits = [Edit("replace", cut, token.idx + len(token.text), ". It")]
            raw.append(("relative_clause", "high", edits))

    out, seen = [], set()
    for rule, risk, edits in raw:
        try:
            edited = apply_edits(text, edits)
        except ValueError:
            continue
        tail = edited.rsplit(". ", 1)[-1]
        if len(tail.split()) < 3 or edited in seen:
            continue
        seen.add(edited)
        out.append({"rule": rule, "risk": risk, "edits": edits})
    return out


def merge_candidates(text, doc):
    """L1：合句只做中性连接（分号或并列 and），不引入因果/转折。"""
    out = []
    for sent in doc.sents:
        end = sent.end_char
        if end < 2 or text[end - 1] not in ".!?":
            continue
        rest = text[end:]
        gap = len(rest) - len(rest.lstrip())
        if not gap:
            continue
        period_at = end - 1
        next_start = end + gap
        follower = next((t for t in doc if t.idx == next_start), None)
        lower = [Edit("replace", next_start, next_start + 1, text[next_start].lower())] \
            if follower is not None and follower.pos_ != "PROPN" and text[next_start].isupper() else []
        out.append({"rule": "semicolon_join", "risk": "low", "edits":
                    [Edit("replace", period_at, period_at + 1, ";")] + lower})
        out.append({"rule": "comma_and_join", "risk": "low", "edits":
                    [Edit("replace", period_at, period_at + 1, ", and")] + lower})
    return out


def clause_completeness(original, edited, nlp):
    """L1 门禁：编辑不得**新造**片段。

    原文本身可能是片段（C4 短文本就有 "A small farm of an idea."），不能算在编辑头上；
    只看编辑后新增的、以从属连词开头或缺主语的非祈使句。
    """
    def scan(text):
        rows = []
        for sent in nlp(text).sents:
            roots = [t for t in sent if t.dep_ == "ROOT"]
            root = roots[0] if roots else None
            finite = bool(root) and root.pos_ in ("VERB", "AUX")
            has_subject = any(t.dep_ in ("nsubj", "nsubjpass", "csubj", "expl") for t in sent)
            imperative = bool(root) and root.tag_ == "VB"
            fragment_start = bool(sent) and (sent[0].pos_ == "SCONJ" or sent[0].lower_ in {
                "and", "but", "or", "while", "because", "although", "though",
                "which", "who", "that", "whose"})
            rows.append({
                "sentence": sent.text.strip(),
                "root_pos": root.pos_ if root else None,
                "finite": finite,
                "has_subject": has_subject,
                "imperative": imperative,
                "fragment_start": fragment_start,
                "ok": finite and (has_subject or imperative) and not fragment_start,
            })
        return rows

    before = {r["sentence"] for r in scan(original) if not r["ok"]}
    rows = scan(edited)
    new_fragments = [r["sentence"] for r in rows if not r["ok"] and r["sentence"] not in before]
    return {"sentences": rows, "new_fragments": new_fragments,
            "passed": bool(rows) and not new_fragments}


def fidelity(original, edited, nlp):
    """L4：实体、否定、时态三条硬不变量；依存距离只作诊断，不作门禁。

    句界编辑必然改变句法树，所以不能要求树不变；要求的是内容与语义算子不变。
    """
    old, new = nlp(original), nlp(edited)
    negations = {"not", "no", "never", "n't", "none", "nobody", "nothing",
                 "neither", "nor", "without"}
    old_neg = sum(t.lower_ in negations for t in old)
    new_neg = sum(t.lower_ in negations for t in new)

    old_ent = sorted((e.label_, e.text.lower()) for e in old.ents)
    lowered = edited.lower()
    kept = [e for e in old_ent if e[1] in lowered]

    def tense(doc):
        return Counter((t.lemma_.lower(), t.morph.get("Tense")[0]) for t in doc
                       if t.pos_ in ("VERB", "AUX") and t.morph.get("Tense"))

    old_tense, new_tense = tense(old), tense(new)
    tense_ok = all(new_tense[k] >= v for k, v in old_tense.items())

    def triples(doc):
        return {(t.lemma_.lower(), t.dep_, t.head.lemma_.lower()) for t in doc
                if t.dep_ not in ("punct", "det") and t.lemma_.strip()}

    a, b = triples(old), triples(new)
    overlap = len(a & b)
    dep_f1 = 0.0 if not (a and b) else round(2 * overlap / (len(a) + len(b)), 4)
    return {
        "entities_preserved": len(kept) == len(old_ent),
        "entity_count": [len(old_ent), len(kept)],
        "negation_equal": old_neg == new_neg,
        "negation_count": [old_neg, new_neg],
        "tense_preserved": tense_ok,
        "structure_distance_f1": dep_f1,
        "passed": len(kept) == len(old_ent) and old_neg == new_neg and tense_ok,
    }
