"""Record the assistant's strict audit of changed, detector-escaping repair candidates."""

import json
from pathlib import Path

import numpy as np


INVALID = {
    (20, "voice_control"): "The passive form 'Approachable was made' changes the incomplete source predicate into an unsupported participant structure and is not readable natural English.",
    (5, "nominalization"): "Reports that shutdowns are commanded instead of directly preserving the source commands, changing speech-act force.",
    (6, "nominalization"): "Adds that each proposition was stated, introducing unsupported metalinguistic claims.",
    (10, "nominalization"): "Contains four generic acquittal commands although the source contains three, adding a duplicate speech act.",
    (26, "nominalization"): "Turns the command to define the topic to Builder into a declarative description of an act directed to Builder.",
}

root = Path(__file__).resolve().parents[1]
out = root / "experiments" / "candidate_budget16_pilot30" / "feedback_driven_strict30"
rows = json.loads((out / "rows.json").read_text(encoding="utf-8"))
z = np.load(out / "detection" / "machine_z_scores.npy")
audited = []
for position, row in enumerate(rows):
    changed = row["source"] != row["attack"]
    escaped = bool(z[position] <= 2.33)
    screened = changed and escaped
    key = (int(row["index"]), row["strategy"])
    valid = screened and key not in INVALID
    reason = ""
    if screened:
        reason = INVALID.get(
            key,
            "Preserves all distinct source units, participant roles, vague placeholders, tense/modality and speech-act force; only exact duplicates or permitted structure are changed.",
        )
    audited.append(
        {
            **row,
            "z_score": float(z[position]),
            "text_changed": changed,
            "escaped": escaped,
            "strictly_audited": screened,
            "assistant_quality_valid": valid,
            "assistant_valid_success": valid,
            "assistant_reason": reason,
        }
    )

screened_rows = [row for row in audited if row["strictly_audited"]]
valid_rows = [row for row in screened_rows if row["assistant_valid_success"]]
successful_indices = sorted({int(row["index"]) for row in valid_rows})
summary = {}
for strategy in ("minimal_boundary", "clause_reorder", "voice_control", "nominalization"):
    group = [row for row in audited if row["strategy"] == strategy]
    summary[strategy] = {
        "n": len(group),
        "changed": sum(row["text_changed"] for row in group),
        "escaped": sum(row["escaped"] for row in group),
        "changed_and_escaped_reviewed": sum(row["strictly_audited"] for row in group),
        "strict_valid_successes": sum(row["assistant_valid_success"] for row in group),
    }

payload = {
    "protocol": {
        "reviewer": "OpenAI Codex assistant designated by the researcher",
        "screening": "Manually review every candidate with changed text and z<=2.33",
        "strict_rule": "No material proposition, role, placeholder, tense, modality, polarity, or speech-act change",
        "note": "AI-assistant audit; do not represent as human-subject annotation.",
    },
    "summary": summary,
    "reviewed_changed_escaped": len(screened_rows),
    "strict_valid_candidates": len(valid_rows),
    "successful_paragraph_indices": successful_indices,
    "successful_paragraphs": len(successful_indices),
    "strict_valid_asr": len(successful_indices) / 30,
    "rows": audited,
}
(out / "assistant_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# 反馈驱动严格修复实验",
    "",
    "仅对文本确实变化且z≤2.33的候选进行逐条严格复核；未变化文本不得计为攻击，检测失败候选无需质量复核。",
    "",
    "| 策略 | 候选 | 文本变化 | 检测逃逸 | 变化且逃逸并复核 | 严格Valid成功 |",
    "|---|---:|---:|---:|---:|---:|",
]
for strategy, row in summary.items():
    lines.append(
        f"| {strategy} | {row['n']} | {row['changed']} | {row['escaped']} | "
        f"{row['changed_and_escaped_reviewed']} | {row['strict_valid_successes']} |"
    )
lines.extend(
    [
        "",
        f"- 复核的变化且逃逸候选：{len(screened_rows)}。",
        f"- 严格质量合格且逃逸：{len(valid_rows)}。",
        f"- 成功段落：{successful_indices}。",
        f"- 严格Valid ASR：{len(successful_indices)}/30（{len(successful_indices)/30:.1%}）。",
        "",
        "该结果证明反馈驱动受控操作能在严格标准下恢复一部分攻击，但成功率远低于宽松单模型评审。"
        "复核来源应如实标为AI助手独立审计，正式论文仍宜追加真人复核。",
    ]
)
(out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, ensure_ascii=False, indent=2))
