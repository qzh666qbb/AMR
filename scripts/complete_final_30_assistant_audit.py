"""Apply the independent assistant's strict manual review to the final 30 selections."""

import json
from pathlib import Path


REASONS = {
    0: "Omits the standalone IBM references and the proposition that IBM is related to something.",
    1: "Conflates the organization with Prasar Bharti and collapses distinct earned-something and earned-only-something claims.",
    2: "Adds an interpretation about defining what a brand is and changes the marketer/Apple-brand relations.",
    3: "Adds that the appointment is current and that the Township was merely involved, while omitting the source's unusual library-as-appointee relation.",
    4: "Adds an obligation that both items must appear in the new report; this requirement is not stated by the source.",
    5: "Interprets an unspecified object as Google's feature and adds a responsibility claim.",
    6: "Retains only two of five distinct source propositions and omits the surprise-with-something and surprise-with-an-idea claims.",
    7: "Conflates Washington with Washington, D.C. and changes the participants of the distinct presentation commands/events.",
    8: "Introduces three distinct departing entities although the source does not establish that they are different.",
    9: "Turns multiple commands to define ICYMI into a past failure and omits their imperative force.",
    10: "Omits the generic nightclub variant and adds an instruction to repeat until the jury understands.",
    11: "Changes tell-QVC-something into tell QVC what you think and equates an unspecified aggressive something with plans.",
    12: "Collapses separate description events and adds simultaneity; it also weakens the standalone brilliant claim.",
    13: "Adds possibility, medical recognition, appropriateness and recurrence claims not present in the source.",
    14: "Merges a present holds-for-something relation with the separate 2015 held event, losing the temporal distinction.",
    15: "Omits the generic conference-held-something claim and conflates the conference's named event with the event held by SocialDevCamp Chicago.",
    16: "Changes the imperative 'Define BlackRock Inc.' into the declarative 'BlackRock Inc. is defined,' altering speech-act modality.",
    17: "Normalizes contradictory source roles into New York as recipient, omitting claims that New York was sold and was sold to the home.",
    18: "Changes NASCAR-defined-as-something into NASCAR-defined-as-a-concept and rearranges the source relations.",
    19: "Changes 'imagine a day arriving' into Jeff Deardorff's arrival being part of an imagined day.",
    20: "Keeps only the general made-himself-approachable claim and omits the other distinct forms, including approachable to one Aylesbury.",
    21: "Turns commands into a completed acquisition transaction and adds an unspecified seller interpretation.",
    22: "Turns commands to ask into claims about what a lawmaker can answer and changes the Washington/advice relation.",
    23: "Conflates separate plan and update propositions and adds that the plan is what the drivers' situation is about.",
    24: "Forces separate unspecified objects to be the same thing and assigns novelty to that shared item without support.",
    25: "Adds a temporal/causal relation between the lead and win and conflates the source's team and Council Bluffs Marshalltown roles.",
    26: "Changes 'define the topic ... to Builder' into Builder being tasked to define it, altering the semantic role.",
    27: "Changes an unspecified giver into DERA, interprets funding as money, and adds a program-needs-support claim.",
    28: "Turns imperative casting instructions into a completed event and collapses distinct cast-something and cast-role propositions.",
    29: "Conflates several hollering events and omits the separate unspecified person who hollered something.",
}

root = Path(__file__).resolve().parents[1]
path = root / "experiments" / "candidate_budget16_pilot30" / "final_30_assistant_audit.json"
payload = json.loads(path.read_text(encoding="utf-8"))
for row in payload["rows"]:
    index = int(row["index"])
    row.update(
        {
            "assistant_meaning_preserved": False,
            "assistant_facts_preserved": False,
            "assistant_readable": True,
            "assistant_major_error": True,
            "assistant_reason": REASONS[index],
            "assistant_quality_valid": False,
            "assistant_valid_success": False,
        }
    )
payload["summary"] = {
    "reviewed": len(payload["rows"]),
    "meaning_preserved": 0,
    "facts_preserved": 0,
    "readable": len(payload["rows"]),
    "major_error": len(payload["rows"]),
    "quality_valid": 0,
    "valid_success": 0,
    "selected_end_to_end_valid_asr": 0.0,
    "agreement_with_selection_judge_quality_labels": 0.0,
}
payload["interpretation"] = (
    "All 30 rows were selected because the earlier model judge marked them valid. The independent strict audit "
    "rejects all 30, so agreement on the binary quality label is 0/30. This audits only the final selected rows "
    "and does not prove that every candidate in the full pools is invalid."
)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

report = path.with_name("final_30_assistant_audit_report.md")
lines = [
    "# 最终30条独立助手复核",
    "",
    "复核者为OpenAI Codex助手，由研究者指定为独立复核；这不是人类受试者标注，论文中不得虚构为真人评审。",
    "复核采用严格零实质增删标准，要求命题、角色、实体、数字、极性、模态和言语行为保持一致。",
    "",
    "- 复核条数：30。",
    "- 语义与事实均通过：0/30。",
    "- 可读性通过：30/30。",
    "- 存在重大语义错误：30/30。",
    "- 最终选中结果的Valid ASR：0/30。",
    "- 与用于选优的原模型质量标签一致：0/30。",
    "",
    "该结果说明先前单模型评审严重偏宽松，不能用来声明30/30正式成功。它只否定当前最终选中的30条，"
    "不等价于对全部候选池逐条复核后的攻击上限。逐条理由见`final_30_assistant_audit.json`。",
]
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
