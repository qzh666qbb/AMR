"""Create the v2 evidence registry from preserved historical endpoint files."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/swan_reframing1/claims_v2.json"

# One endpoint per claim. Auxiliary observations never substitute for it.
DEFINITIONS = [
    ("CL01", "历史固定K=4攻击面", "historical_support",
     "experiments/planE_e2_joint250/feedback_audit.json", "/counts/valid_paragraphs"),
    ("CL02", "历史反馈补救攻击面", "historical_support",
     "experiments/planE_e2_joint250/feedback_audit.json", "/counts/combined_successes_using_saved_flags"),
    ("CL03", "短文本句界决策依赖", "historical_support",
     "experiments/confirmation_mechanism30/summary.json", "/punctuation_control/fixed_2_33"),
    ("CL04", "正常长度N守恒检测下降", "unconfirmed", None, "/H2"),
    ("CL05", "真实T(R(s))匹配参与下降", "unconfirmed", None, "/M1"),
    ("CL06", "真实R(s)变化与匹配相关", "unconfirmed", None, "/M2"),
    ("CL07", "whole-graph主要因果路径", "unconfirmed", None, "/causal_normalization"),
    ("CL08", "独立条件FPR下精确跨度恢复", "unconfirmed", None, "/H3"),
    ("CL09", "人工D的图作用域敏感性", "synthetic_scorer_diagnostic",
     "experiments/swan_amr_composition_decomposition1/summary.json",
     "/primary/composition_matcher_dilution"),
    ("CL10", "真实文本parser drift非必要", "withdrawn", None, "/valid_no_drift_counterfactual"),
    ("CL11", "严格匹配的跨水印特异性", "unconfirmed", None, "/matched_cross_watermark"),
    ("CL12", "有效分割无关防御", "unconfirmed", None, "/new_defense_holdout"),
    ("CL13", "所核查已有资产中联合合格数为零", "bounded_absence",
     "experiments/swan_positive_asset_audit1/summary.json", "/joint_qualifying_existing_documents"),
    ("CL14", "旧长度门直接过滤已观测达标候选", "candidate_filtering",
     "experiments/swan_length_gate_tradeoff1/summary.json", "/gate_effect/gate_rejected_threshold_passes"),
    ("CL15", "final1当前生成路线未达到pilot可行性门", "historical_support",
     "experiments/swan_count_neutral_final1/pilot_closure/audit.json",
     "/gate/passed"),
    ("CL16", "source-anchored生成路线未达到pilot结构门", "historical_support",
     "experiments/swan_source_anchored1/pilot_closure/audit.json",
     "/gate/passed"),
]


def build():
    claims = []
    for identifier, statement, ceiling, path, pointer in DEFINITIONS:
        historical = path is not None
        path = path or "experiments/swan_count_neutral_final1/final_endpoints.json"
        claims.append({
            "id": identifier, "statement": statement, "ceiling": ceiling,
            "primary_endpoint": {
                "path": path, "pointer": pointer,
                "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                if historical else None,
                "kind": "historical_saved" if historical else "prospective_confirmatory",
            },
        })
    return {"schema_version": 2, "no_automatic_upgrade": True, "claims": claims}


if __name__ == "__main__":
    with OUT.open("x") as stream:
        stream.write(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
    print(OUT)
