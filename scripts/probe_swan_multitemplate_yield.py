"""探针：多模板注入产率。只用于工程决策，不作为论文结果。

问题：normal_length_coherent_pilot1 用模板47达到20/20，但20个槽位全同一模板。
真正可用的数据集必须用不同模板。本探针测不同模板下每槽位需要多少次尝试才能通过
S2Match>=0.75，据此估算规模化成本。不写入任何既有实验目录。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/swan_s2match_runtime_711a231d"))
sys.path.insert(0, str(ROOT / "third_party/SWAN_official_2ca0c207_clean/injection"))

BANK = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json"
EXAMPLES = ROOT / "third_party/SWAN_official_2ca0c207_clean/amr_bank/artifacts/normalized_amr_examples.json"
GLOVE = ROOT / "vectors/glove.6B.100d.txt"
MODEL_DIR = ROOT / "models/model_parse_xfm_bart_large-v0_1_0"
TOKENIZER_DIR = ROOT / "models/bart-large-tokenizer"
URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
THRESHOLD = 0.75
MIN_WORDS, MAX_WORDS = 5, 9
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
TERMINAL_RE = re.compile(r"[.!?]\s*$")


# 以下两段与 SWAN 官方 injection/injection_amr_utils.py 的 build_user_prompt /
# build_system_prompt 逐字一致。官方模块在 import 时就会加载 amrlib 默认模型，
# 无法在本环境直接导入，故内联；只用其文本，不改其措辞。
LENGTH_INSTRUCTION = (
    "\n\nAdditional fixed constraints:\n"
    f"- Write exactly {MIN_WORDS} to {MAX_WORDS} lexical words; "
    "count a hyphenated term as one word.\n"
    "- Write one complete, fluent sentence ending with '.', '!', or '?'.\n"
    "- Preserve the AMR meaning and context, and output no explanation."
)


def build_prompt(template: str, examples: list, context: str = "") -> str:
    example_text = ""
    for ex in examples:
        example_text += (
            f"\nTemplate AMR:\n{ex['normalized_template_amr']}\n"
            f"Possible sentence: {ex['sentence']}\n"
        )
    return f"""AMR (Abstract Meaning Representation) is a graph-based representation of a sentence's meaning.
        Each node is a concept and edges represent semantic roles or relationships.

        Below are some examples of template AMRs and corresponding sentences:
        {example_text}

        In the provided AMR, there are placeholders:
        - "NE" for named entities (e.g., "Alice", "France", "Google").
        - "N" for generic nouns (e.g., "a device", "an object").
        - "X" for unspecified concepts (e.g., "something", "an idea").

        Instructions:

        - Do not write "NE", "N", or "X" literally. Instead, replace them with appropriate English words to form a natural, meaningful sentence.
        - Ensure the generated sentence aligns with both the AMR structure and the given context.
        - Do not produce multiple sentences or lists.
        - Produce exactly one coherent sentence 

        AMR:
        {template}

        Context: {context}

        Please output only that one sentence.""" + LENGTH_INSTRUCTION


def system_prompt() -> str:
    return (
        "You are a helpful assistant. Follow the user's instructions to produce exactly one sentence.\n"
        "Do not provide any explanations of your reasoning."
    ).strip()


def strip_snt(amr_out: str) -> str:
    """amrlib 输出带 '# ::snt ...' 注释行；amr_py3 解析器无法处理它。"""
    if amr_out.startswith("# ::snt "):
        return "\n".join(amr_out.splitlines()[1:])
    return amr_out


def normalize_to_template(amr_str: str) -> str:
    """对齐官方 generate_template_amr：真实概念 → V/N/NE/X 占位符。

    S2Match 的目标 Bank 模板本身就是占位符形式，候选不归一化则两边不可比。
    """
    import penman
    from src.core.bank_scorer import _concept_class

    graph = penman.decode(amr_str)
    ne_nodes = {str(t[0]) for t in graph.triples if str(t[1]) == ":name"}
    triples = []
    for s, r, t in graph.triples:
        s, r, t = str(s), str(r), str(t)
        if r == ":instance":
            triples.append((s, r, "NE" if s in ne_nodes else _concept_class(t, s in ne_nodes)))
        elif t.startswith('"') and t.endswith('"'):
            triples.append((s, r, '"NE"'))
        else:
            triples.append((s, r, t))
    return penman.encode(penman.Graph(triples))


def call_api(prompt: str, key: str, max_tokens: int, system: str = "",
             retries: int = 4) -> dict:
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.9, "thinking": {"type": "disabled"}}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    last = None
    for attempt in range(retries):
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.load(resp)
            return {"text": payload["choices"][0]["message"]["content"].strip(),
                    "finish_reason": payload["choices"][0]["finish_reason"],
                    "usage": payload["usage"], "response_id": payload.get("id"),
                    "response_model": payload.get("model"),
                    "elapsed": time.monotonic() - t0}
        except Exception as exc:  # 网络抖动重试，不改变请求内容
            last = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"API 调用重试 {retries} 次仍失败: {type(last).__name__}: {last}")


def surface_gate(text: str) -> dict:
    words = WORD_RE.findall(text)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return {"words": len(words),
            "range_pass": MIN_WORDS <= len(words) <= MAX_WORDS,
            "single_sentence": len([s for s in sentences if s.strip()]) == 1,
            "terminal": bool(TERMINAL_RE.search(text)),
            "pass": (MIN_WORDS <= len(words) <= MAX_WORDS
                     and len([s for s in sentences if s.strip()]) == 1
                     and bool(TERMINAL_RE.search(text)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "experiments/swan_multiview_development1/generation_probe1")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY 未设置")
    if not GLOVE.is_file():
        raise SystemExit(f"缺少词向量：{GLOVE}")

    bank = json.loads(BANK.read_text())
    examples = json.loads(EXAMPLES.read_text())
    # 均匀抽样不同模板；跳过已知“有利”的47，避免重复旧结论
    step = len(bank) // args.templates
    picks = [min(i * step + 7, len(bank) - 1) for i in range(args.templates)]

    from src.core.amr_extractor import AmrlibBackend
    import amr_py3
    from s2match import compute_s2match_from_strings

    # 官方 py3 运行时只在 verbose=True 时定义 log_helper，错误路径却无条件引用它。
    if not hasattr(amr_py3, "log_helper"):
        class _SilentLog:
            def debug(self, *a):
                pass

            def error(self, *a):
                pass

        amr_py3.log_helper = _SilentLog()

    backend = AmrlibBackend(str(MODEL_DIR), str(TOKENIZER_DIR), device="cpu", batch_size=4)
    backend._load()
    sys_prompt = system_prompt()

    records, t_start = [], time.monotonic()
    in_tok = out_tok = calls = 0
    for idx in picks:
        template = bank[idx]["normalized_template_amr"]
        prompt = build_prompt(template, examples, context="")
        attempts, accepted = [], None
        for n in range(1, args.max_attempts + 1):
            r = call_api(prompt, key, args.max_tokens, system=sys_prompt)
            calls += 1
            in_tok += r["usage"]["prompt_tokens"]
            out_tok += r["usage"]["completion_tokens"]
            text = r["text"]
            gate = surface_gate(text)
            score, amr_norm = None, None
            if gate["pass"]:
                amr_out = strip_snt(backend.parse_batch([text])[0])
                try:
                    amr_norm = normalize_to_template(amr_out)
                    _, _, score = compute_s2match_from_strings(
                        amr_norm, template, vectors_file=str(GLOVE))
                except Exception as exc:  # 记录失败而非静默
                    score = None
                    gate["s2match_error"] = f"{type(exc).__name__}: {exc}"
                score = None if score is None else round(float(score), 4)
            attempts.append({"attempt": n, "text": text, "gate": gate, "s2match": score,
                             "amr_normalized": amr_norm,
                             "finish_reason": r["finish_reason"],
                             "usage": r["usage"], "response_id": r["response_id"]})
            print(f"  t{idx} try{n}: {gate['words']}w gate={gate['pass']} "
                  f"s2match={score} | {text[:70]}", flush=True)
            if score is not None and score >= THRESHOLD:
                accepted = n
                break
        records.append({"template_index": idx, "template": template,
                        "accepted_attempt": accepted, "attempts": attempts})
        print(f"[template {idx}] {'PASS' if accepted else 'FAIL'} "
              f"in {len(attempts)} attempts", flush=True)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "engineering_probe_not_a_paper_result",
        "threshold": THRESHOLD, "word_range": [MIN_WORDS, MAX_WORDS],
        "templates_tested": len(records),
        "max_attempts": args.max_attempts,
        "passed_templates": sum(r["accepted_attempt"] is not None for r in records),
        "mean_attempts_to_pass": round(
            sum(r["accepted_attempt"] for r in records if r["accepted_attempt"]) /
            max(1, sum(r["accepted_attempt"] is not None for r in records)), 2),
        "paid_calls": calls, "input_tokens": in_tok, "output_tokens": out_tok,
        "peak_price_cost_usd_upper": round((in_tok * 0.30 + out_tok * 1.20) / 1e6, 6),
        "wall_seconds": round(time.monotonic() - t_start, 1),
        "records": records,
    }
    path = out_dir / f"multitemplate_yield_t{args.templates}_a{args.max_attempts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "records"},
                     ensure_ascii=False, indent=2))
    print(f"\nwritten: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
