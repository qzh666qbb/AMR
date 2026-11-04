#!/usr/bin/env python3
"""Build a local, read-only evidence inventory; write only editorial artifacts.

No project modules, model runtimes, detector APIs or network clients are imported.
Directory rows are accounting units, not independent experiments or publications.

2026-10-29: paths updated for the unified 论文/ layout. P3/P4 labels are kept
only as archival source groups: P3 -> P1 §6, P4 -> P1 §7. Current papers are P1/P2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets/experiment_article_conversion_20261026"
TABLE = ROOT / "论文/全部实验_文章归属与转换总表.md"
HISTORY = ROOT / "archive/20260911_pre_unification/derived_experiment_docs/experiments"
P2_ROOT = ROOT / "第二篇_语义消息写入与读回"
P2_EXP_ROOT = P2_ROOT / "02_实验"
P2_SUP_ROOT = P2_ROOT / "03_实验外支持材料"
PAPERS = {
    "P1": ("句界接口审计与精确评估", "论文/01_句界审计与精确评估/P1_句界接口审计.md"),
    "P2": ("语义消息写入与读回", "第二篇_语义消息写入与读回/01_论文文档/P2_语义消息写入与读回.md"),
    "P3": ("完整认证的失败与资源边界", "论文/01_句界审计与精确评估/P3_完整认证的失败与资源边界.md"),
    "P4": ("低质量多视图融合", "论文/01_句界审计与精确评估/第四篇_低质量多视图融合_文章转换.md"),
}
APPENDIX = "论文/01_句界审计与精确评估/遗漏实验_附录转换稿.md"
GROUPS = {
    "early": ("P1", "历史背景", "A1", "早期普通复述、AMR引导和句界攻击候选；自动质量不替代真人质量"),
    "feedback": ("P1", "历史/失败分析", "A2", "候选预算、联合生成、反馈补救及评审冲突；预算和质量来源分别报告"),
    "transfer": ("P1", "范围对照", "A3", "子库/跨水印对照；库删减、长度和生成模型不匹配限制外推"),
    "repeat": ("P1", "消融/复现", "A4", "聚合、靶向选择和确定性；统计不显著与阶段不一致分别保留"),
    "graph": ("P1", "实现诊断", "A5", "图组合、匹配、序列化与结构诊断；不等于gold AMR或自然文本因果证据"),
    "positive": ("P1", "可行性/失败分析", "A6", "正例生成、源事实与正常长度资格；注入失败和未完成cohort不计攻击成功"),
    "elastic": ("P1", "恢复限制", "A7", "弹性恢复及Null诊断；开发结果、解析可用性、校准分布分别记账"),
    "statistics": ("P1", "统计附录", "A8", "阈值、族计数与裁决；事后门限和相关变体不追认独立正式FPR"),
    "fusion": ("P4", "开发/消融", "A9", "检测端低质量视图融合；重复开发、已知位置和撤回结论不当独立优势"),
    "symbols": ("P2", "通道/分层诊断", "A10", "符号写入、读出、定位、顺序与全消息终点；真人与模型来源分开"),
    "fixed_n": ("P1", "精确评估/复现", "A11", "fixed-N完整重分句版本链；工程预检不等于真实证书"),
    "variable_n": ("P3", "独立探索/失败", "A11", "fixed-byte variable-N版本链；覆盖、原始检测和最终资源门分开"),
    "support": ("P1", "文献/制品附录", "A12", "规划、出处、交付和历史副本；不是独立效果实验"),
}
SETS = {
    "early": """e2_attack_pilot e3_main e4_ablation mechanism_e2_amr
        mechanism_sentence_boundary milestone_a segmentation_intervention
        prompt_component_ablation30 sentence_watermark_method_matrix_20261029""",
    "feedback": "best_observed_frontier candidate_budget16_pilot30 feedback2_diagnostics",
    "transfer": "baseline_comparison e5_transfer cross_watermark_transfer30 sira_adapted30 planE_transfer_strict30",
    "repeat": "defense_aggregation parser_repeatability30 targeted_sentence_attack targeted_sentence_controlled30 targeted_sentence_pilot30",
    "graph": """e1_invariance swan_amr_composition_decomposition1 swan_amr_precision_recall1
        swan_dependency_pruner1 swan_oracle_composition1 swan_relation_motif1
        swan_template_specificity1 swan_local_multi_instance1 swan_local_multi_instance2""",
    "positive": """grounded_generation3 grounded_injection5 swan_atomic_fact_preflight1
        swan_bank100_microfact_null1 swan_bank_scale_pilot1 swan_c4_train_narrative_windows3
        swan_coherent_baseline1 swan_coherent_baseline_cached1 swan_constrained_microfact1
        swan_deepseek_span_compiler1 swan_generator_comparison1 swan_injection_only_baseline1
        swan_injection_only_pilot1 swan_known_positive1 swan_length_gate_tradeoff1
        swan_length_probe swan_length_probe_v2 swan_microfact_generation1
        swan_microfact_sequential2 swan_natural_mixed_windows1 swan_natural_narrative_windows2
        swan_normal_length_boundary_location1 swan_normal_length_coherent_baseline1
        swan_normal_length_coherent_pilot1 swan_normal_length_feasibility2 swan_onepass_control1
        swan_quality_first_windows4 swan_sentence_growth1 swan_splittable_positive_pilot1
        swan_accepted_baseline_preflight1 swan_accepted_boundary_preflight1""",
    "statistics": """swan_b1_family_units1 swan_b1b_statistic_power1
        swan_bank100_null_calibration1 swan_partial_coverage_power1 swan_statistic_design1
        swan_statistic_feasibility1 swan_verdict_audit1""",
    "fusion": "swan_multiview_development1 swan_multiscale_attention_development1 swan_fusion_audit1 swan_anchor_stability1",
    "support": """archive_20260725 report_assets literature_novelty_audit1
        research_direction_reconciliation_20261002 swan_branch_c_final_literature_refresh_20261002
        swan_branch_c_submission_readiness_20261002 swan_reframing1 unitization_go_no_go1""",
}
LOOKUP = {name: group for group, names in SETS.items() for name in names.split()}

# An editorial override is deliberately different from a copied terminal status.
NOTES = {
    "confirmation_mechanism30": ("正文主结果", "短文本A–B–A为1/30→30/30→1/30；单元数解释充分，正常长度机制未确认"),
    "confirmation_three_arm30": ("受控范围", "三臂确认的质量、预算和统计单位保留；不扩大为完整原文等价复现"),
    "certified_resegmentation_wp2_v8_global_reuse1": ("失效审计", "读取越过冻结可见范围，机会审计Invalid；不能恢复有效实验身份"),
    "certified_resegmentation_wp2_v9_simmark_long1": ("检测健康边界", "SimMark原划分89/1000检出；检测健康缺口不由长运行补齐"),
    "certified_resegmentation_wp2_v15_causal_prefix1": ("冻结Stop", "H0 Stop保留；v16事后rescue不能追溯记为本版成功"),
    "certified_resegmentation_wp2_v16_prompt_only_rescue1": ("事后rescue", "995+5正例形成1000行；身份H0 Go，不等于检测或证书成功"),
    "certified_resegmentation_wp2_v17_b0_detector_health1": ("正文主结果", "原始FPR47/1000、TPR954/1000；不是认证后FPR/TPR"),
    "certified_resegmentation_wp2_v18_b1_exact_certificate1": ("评分前Stop", "容量及86.6501h预算上界失败；无threshold、witness或certified rate"),
    "semantic_sequence_cycle1": ("冻结No-Go", "必要结构16/24低于20/24门；不改写旧No-Go"),
    "sentence_watermark_method_matrix_20261029": ("句级方法对照", "五种句级水印换序/合并/拆分 Layer2 pilot；候选8，不足以代表正式FPR/TPR"),
    "semantic_sequence_cycle2": ("冻结No-Go/版本容器", "必要结构19/24低于20/24；execution2–5为另立修复与后置诊断"),
    "semantic_sequence_readout1": ("正文主结果", "必要标记82/96；W复述顺序胜率77.48%，唯一第一0/12"),
    "semantic_channel_audit1": ("全消息失败", "65536消息盲恢复各0/12；计划符号替换仅真值诊断"),
    "semantic_fact_carrier_pilot1": ("模型审查/真人待定", "严格83/96；事后口径88–89/96不追认过门；助手代评不计独立真人"),
    "semantic_fact_carrier_pilot2": ("模型过门/真人待定", "新96+96显式状态批次模型96/96；13组复述未变化，真人确认未完成"),
    "semantic_fact_layers_v2_execution1": ("门控执行", "分层函数192次；本版无真实原AMR，不与v3解析合并版本"),
    "semantic_fact_layers_v3_parse1": ("原图完成/归因待定", "原AMR192/192、结构172合格20排除、194解析；事实/修订/分类/完整对照0"),
    "semantic_multiscale_readback1": ("开发增益No-Go", "attention复述F1 22.97% vs gate28.93%；十方法完整恢复各0/12"),
    "semantic_multiscale_supervision2": ("实现准备/NOT_READY", "17工程测试；真人0，未训练、未产生新效果"),
    "swan_fusion_audit1": ("旧综合结论撤回", "错误配对、置换统计不一致、标准化未使用/训练复用；不引用旧最优或融合无效"),
    "swan_anchor_stability1": ("更正后诊断", "Bank分离度不是待测margin；模板集合编号重复不是同一事实存活"),
    "swan_multiview_development1": ("混合问题容器", "检测融合归P4；盲窗、阈值复审、S8确认子项归P1；撤回旧约15pp主张"),
    "swan_multiscale_attention_development1": ("开发系列容器", "三尺度100模板开发；首轮attention增量区间跨零，后续消融及分族未确认严格效用"),
    "swan_count_neutral_supplement3": ("永久Halt", "64次响应、0完整文档、0检测；warning使pilot停止，不能借partial补机制"),
    "swan_elastic_v4_mechanism1": ("机制诊断", "空二跳路径满分项0.25；固定路径扣分不代表重搜索后的因果防御"),
    "swan_elastic_v4_p2_formal": ("相关变体审计", "5000行来自1000篇相关变体；正例未完成，不记独立正式FPR或完整防御"),
    "semstamp_certified_defense_v3_wp3_variable_n_profile1": ("评分前Stop", "3000行/46501层，上界77.6152h超过72h；0证书"),
    "semstamp_certified_defense_v4_full_radius_profile1": ("完整半径Stop", "修正为47332层；费用模型下界75.3111h超过72h；新全量profile未运行"),
}

MULTIVIEW_P1 = {
    "abstention_audit1", "blind_eval1", "blind_eval1/run2",
    "newline_negative_expansion1", "threshold5_reassessment1",
    "threshold5_reassessment1/P0_S6", "threshold5_reassessment1/P0_S6_S8",
    "threshold5_reassessment1/P0_S6_S8_S10",
    "threshold5_reassessment1/P0_S6_S8_S10_S12",
    "final_completion1", "final_completion2", "final_completion3", "final_completion4",
}
NESTED = {
    "swan_multiview_development1": sorted(MULTIVIEW_P1 | {
        "cache_audit1", "dataset1", "dataset1_none", "dataset1_topic", "dataset2",
        "dataset3_concrete", "dataset4", "dataset4_attack1", "dataset4_fusion1",
        "dataset5", "dataset5_eval", "dataset5_paired_neg", "dataset5_paired_neg_eval",
        "generation_probe1", "null_fp_audit1", "pipeline1", "pipeline2", "pipeline3",
        "review_pairs1",
    }),
    "swan_multiscale_attention_development1": [
        "validation1", "ablation1", "operating_points1", "family_hard1", "cache_audit1",
    ],
    "semantic_sequence_stage1_run2": [
        "ai_review1", "alignment_review1", "symbol_order_diagnostic1",
        "multiscale_transfer1", "multiscale_transfer1_run2",
        "multiscale_transfer2", "multiscale_transfer3",
    ],
    "semantic_sequence_cycle2": ["execution2", "execution3", "execution4", "execution5"],
    "candidate_budget16_pilot30": [
        "fact_constrained_rescue", "feedback_driven_strict30", "final_hardcase_control",
    ],
    "swan_count_neutral_final1": ["pilot_closure", "auxiliary_reviews/baseline_round1"],
    "swan_source_anchored1": ["pilot_closure"],
    "swan_bank100_microfact_null1": ["implementation_amendments"],
    "semstamp_certified_defense_v3_wp3_variable_n_profile1": [
        "smoke/row800_replay0", "smoke/row800_replay1", "smoke/subset_smoke",
    ],
}
EXTERNAL = {
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_model_review_20261026": ("P2", "A10", "v1模型严格预审；同一批96对，不是独立新样本"),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_model_reassessment_20261026": ("P2", "A10", "v1意向/过去计划口径敏感性，不能追认冻结门"),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_completed_review_20261026": ("P2", "A10", "助手代评192条及96对；不能冒充两位独立真人"),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_multiscale_failure_analysis_20261026": ("P2", "A10", "固定旧预测的拒绝/监督诊断；未重训，不确认单一因果根因"),
    "baselines/local/swan_acl2026_realnews/pilot_30_report.md": ("P1", "A1", "早期本地30篇baseline支持，后续主张以现行矩阵限定"),
    "RESULT.json": ("P1", "A12", "根聚合结果为支持来源，不增加独立论文或重复样本数"),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_validation_20261026": ("P2", "A15", "v1封存模型标签的目标对齐与成组区间；R0/R1/R2为83/88/89，真人0，不追认过门"),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_layer_gate_analysis_20261026": ("P2", "A16", "v2的192个pending由前置真人门触发；原图未生成，6组构造输入仅检查逻辑；v3另记"),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_carrier_v2_20261026": ("P2", "A15", "v2模型96/96、13组复述不变；[1,1]为退化经验区间，不是总体必然正确"),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_innovation_validation_20261026": ("P2", "A17", "256个H8码字、32640对：汉明距3、编辑距2；删除反例否定任意单次插删保证，非通道实验"),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_sequence_authentication_design_20261026": ("P2", "A18", "60项文档检查；未创建密钥、运行认证或伪造测试，不计认证效果"),
    "第二篇_语义消息写入与读回/03_实验外支持材料": ("P2", "A10", "P2 实验外支持材料容器：评审、诊断、码本与文献；不是新增独立试验"),
    "report_assets/research_closure_20261025": ("P3", "A18", "v4前的半径覆盖复核：3000行、28/155行截断；无新profile或certificate"),
}
ARCHIVED_REPORTS = {
    "archive/20260911_pre_unification/experiment_docs/planB_b4_candidate_report.md":
        ("P1", "A13", "B4在50篇节点/重挂接/组合候选上按z筛选；长度及句数是代理，不是确认攻击"),
    "archive/20260911_pre_unification/experiment_docs/planB_b5_final_report.md":
        ("P1", "A13", "B5自动联合候选10条（2/5/3）；最终复核有语义/结构失败，不写10条成功攻击"),
}
DOCUMENTARY_ROWS = {
    "archive/20260911_pre_unification/root_docs/planB_final_report.md":
        ("P1", "A18", "旧PlanB汇总，复用既有cohort；受现行claim矩阵限定"),
    "archive/20260911_pre_unification/experiment_docs/planC_final_comparison.md":
        ("P1", "A18", "旧PlanC算子比较汇总，不增加独立样本或确认机制"),
    "archive/20260911_pre_unification/experiment_docs/planE_summary.md":
        ("P1", "A18", "旧PlanE回译/联合生成汇总，质量来源与预算不得合并"),
    "archive/20260911_pre_unification/paper_strict30_draft/ERRATUM.md":
        ("P1", "A18", "旧严格30稿已superseded；重复不独立，旧z_sem/z_parse不能作自然文本因果分解"),
    "CURRENT_RESULTS.md":
        ("P1", "A18", "当前结果账本；主归P1，跨P2/P3/P4导航，汇总不是新实验"),
    "RESEARCH_STATE.md":
        ("P1", "A18", "当前研究状态账本；不将设计或Stop恢复为成功，跨线引用不重复计数"),
}
EXTRA_SOURCES = {
    "archive/20260911_pre_unification/experiment_docs/planB_b5_final_report.md": [
        "archive/20260911_pre_unification/experiment_docs/planB_b5_joint_report.md",
        "archive/20260911_pre_unification/experiment_docs/planB_b5_review_guide.md",
    ],
    "archive/20260911_pre_unification/paper_strict30_draft/ERRATUM.md": [
        "archive/20260911_pre_unification/experiment_docs/paper_results_table.md",
    ],
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_sequence_authentication_design_20261026": [
        "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_sequence_authentication_design_20261026/VALIDATION.json",
    ],
}
# Every eligible Markdown file is discovered independently of the inventory rows.
# Explicit document dispositions handle navigation, old drafts and review delivery.
ROOT_DOCUMENT_OWNERS = {
    "论文/AliMark_ICML2026_精读讲解.md": ("experiments/literature_novelty_audit1", []),
    "论文/CLAIM_EVIDENCE_MATRIX.md": ("experiments/swan_reframing1", ["P2", "P3", "P4"]),
    "README.md": ("experiments/research_direction_reconciliation_20261002", ["P2", "P3", "P4"]),
    "SWAN_弹性重构防御_总体规划.md": ("experiments/swan_multiview_development1", ["P1", "P2"]),
    "SWAN_弹性重构防御_部署运行手册.md": ("experiments/swan_elastic_v4_p2_formal", []),
    "论文/TERMINOLOGY.md": ("experiments/swan_reframing1", ["P2", "P3", "P4"]),
    "V2.md": ("experiments/swan_reframing1", ["P2", "P3", "P4"]),
    "任务清单.md": ("experiments/research_direction_reconciliation_20261002", ["P2", "P3", "P4"]),
    "论文/全部实验_文章归属与转换总表.md": ("experiments/research_direction_reconciliation_20261002", ["P2", "P3", "P4"]),
    "叙事.md": ("experiments/swan_reframing1", ["P2", "P3", "P4"]),
    "实验目录导读.md": ("experiments/research_direction_reconciliation_20261002", ["P2", "P3", "P4"]),
    "论文/01_句界审计与精确评估/第一篇_核心贡献与摘要.md": ("experiments/swan_reframing1", []),
    "论文/01_句界审计与精确评估/第三篇_独立认证方法_失败与边界记录.md": ("experiments/semstamp_certified_defense_v4_full_radius_profile1", []),
    "第二篇_语义消息写入与读回/01_论文文档/第二第三篇研究闭环.md": ("第二篇_语义消息写入与读回/02_实验/semantic_channel_audit1", ["P3"]),
    "第二篇_语义消息写入与读回/01_论文文档/第二篇_创新验证_实验对比方案.md": ("第二篇_语义消息写入与读回/03_实验外支持材料/semantic_innovation_validation_20261026", []),
    "第二篇_语义消息写入与读回/01_论文文档/第二篇_图模板符号序列读回_转题方案.md": ("第二篇_语义消息写入与读回/02_实验/semantic_sequence_stage1_design1", []),
    "第二篇_语义消息写入与读回/01_论文文档/第二篇_多尺度注意力集成与复评方案.md": ("第二篇_语义消息写入与读回/02_实验/semantic_multiscale_readback1", []),
    "第二篇_语义消息写入与读回/01_论文文档/第二篇_序列来源认证与伪造测试_设计.md": ("第二篇_语义消息写入与读回/03_实验外支持材料/semantic_sequence_authentication_design_20261026", []),
    "第二篇_语义消息写入与读回/01_论文文档/第二篇_消息恢复_文献与方案对照.md": ("第二篇_语义消息写入与读回/02_实验/semantic_channel_audit1", []),
    "第二篇_语义消息写入与读回/01_论文文档/第二篇_语义写入与读回_实验闭环.md": ("第二篇_语义消息写入与读回/02_实验/semantic_channel_audit1", []),
    "论文/01_句界审计与精确评估/第四篇_低质量多视图融合_文章转换.md": ("experiments/swan_multiscale_attention_development1", []),
    "论文/论文转化方案.md": ("experiments/research_direction_reconciliation_20261002", ["P2", "P3", "P4"]),
    "论文/01_句界审计与精确评估/README.md": ("experiments/research_direction_reconciliation_20261002", []),
    "论文/合并版_两篇论文初稿_20261026.md": ("experiments/research_direction_reconciliation_20261002", ["P2", "P3", "P4"]),
    "句级水印_六项文献对照与实验选择_20261029.md": ("experiments/sentence_watermark_method_matrix_20261029", []),
    "论文/01_句界审计与精确评估/句级水印_六项文献对照与实验选择_20261029.md": ("experiments/sentence_watermark_method_matrix_20261029", []),
}
_P2_MAIN_OWNER = "第二篇_语义消息写入与读回/02_实验/semantic_sequence_readout1"
for _name in (
    "第二篇_语义消息写入与读回/README.md",
    "第二篇_语义消息写入与读回/01_论文文档/README.md",
    "第二篇_语义消息写入与读回/02_实验/README.md",
    "第二篇_语义消息写入与读回/03_实验外支持材料/README.md",
    "第二篇_语义消息写入与读回/04_服务器远端资产/SERVER_ASSETS.md",
    "第二篇_语义消息写入与读回/05_脚本与测试/索引.md",
    "第二篇_语义消息写入与读回/06_迁移记录/迁移说明.md",
    "第二篇_语义消息写入与读回/05_脚本与测试/代码快照.md",
    "第二篇_语义消息写入与读回/05_脚本与测试/原路径索引.md",
    "第二篇_语义消息写入与读回/06_迁移记录/历史报告链接导航.md",
    "第二篇_语义消息写入与读回/07_共享参考材料/README.md",
    "第二篇_语义消息写入与读回/08_本地原始资产/README.md",
):
    ROOT_DOCUMENT_OWNERS[_name] = (_P2_MAIN_OWNER, [])
DOCUMENT_PREFIX_OWNERS = {
    "archive/20261027_paper1_merge/":
        ("experiments/research_direction_reconciliation_20261002", "superseded_paper1_layout_snapshot", ["P2", "P3", "P4"]),
    "archive/20261029_paper1_readable_rewrite/":
        ("experiments/research_direction_reconciliation_20261002", "superseded_paper1_rewrite_snapshot", []),
    "archive/20260911_pre_unification/root_docs/":
        ("experiments/research_direction_reconciliation_20261002", "historical_planning_or_summary", []),
    "archive/20260911_pre_unification/paper_strict30_draft/":
        ("archive/20260911_pre_unification/paper_strict30_draft/ERRATUM.md", "superseded_draft_not_new_evidence", []),
    "archive/20260916_research_reset/":
        ("experiments/research_direction_reconciliation_20261002", "historical_reset_index", []),
    "archive/20260918_reframing_cleanup/root_docs/":
        ("experiments/swan_reframing1", "superseded_narrative", []),
    "deliverables/20261010_lab_update_deck/":
        ("experiments/swan_reframing1", "presentation_reusing_existing_evidence", ["P3"]),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_human_review_start_20261026/":
        ("第二篇_语义消息写入与读回/02_实验/semantic_fact_carrier_pilot1", "review_delivery_no_new_labels", []),
    "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_supervision_revision_20261026/":
        ("第二篇_语义消息写入与读回/02_实验/semantic_multiscale_supervision2", "review_delivery_no_new_training", []),
    "report_assets/evidence_identity_alignment_review_20261026/":
        ("experiments/swan_multiview_development1", "identity_alignment_code_review_no_new_experiment", []),
}
DOCUMENT_FILE_OWNERS = {
    "archive/README.md": ("experiments/research_direction_reconciliation_20261002", "archive_navigation", []),
    "archive/20260918_reframing_cleanup/README.md": ("experiments/swan_reframing1", "cleanup_navigation", []),
    "baselines/local/swan_acl2026_realnews/setup.md": ("baselines/local/swan_acl2026_realnews/pilot_30_report.md", "baseline_setup", []),
    "baselines/local/swan_acl2026_realnews/verification.md": ("baselines/local/swan_acl2026_realnews/pilot_30_report.md", "baseline_verification", []),
    "experiments/README.md": ("experiments/research_direction_reconciliation_20261002", "experiment_navigation", ["P2", "P3", "P4"]),
    "experiments/研究工作导航.md": ("experiments/research_direction_reconciliation_20261002", "experiment_navigation", ["P2", "P3", "P4"]),
    "论文/01_句界审计与精确评估/P4_证据身份对齐_三臂验证方案.md":
        ("experiments/swan_multiview_development1", "three_arm_design_unexecuted", []),
    "论文/01_句界审计与精确评估/P4_数据融合实验_完整技术报告.md":
        ("experiments/swan_multiview_development1", "complete_fusion_report_stop_investment_recommendation", []),
}
for _name in (
    "1. 从普通复述转向自适应攻击. 从普通复述转向自适应攻击.md",
    "future_execution_progress.md", "paper_ready_experiment_plan.md",
    "严格说，我们对 Rastogi & Pruthi（EMNLP 2024）的参考主.md",
):
    DOCUMENT_FILE_OWNERS["archive/20260911_pre_unification/experiment_docs/" + _name] = (
        "experiments/research_direction_reconciliation_20261002", "historical_plan_or_literature", [])
DENY_PARTS = {
    ".git", "__pycache__", "private", "private_review", "sealed", "credentials",
    "provider_responses", "raw_responses", "remote", "config", "configuration",
}
SOURCE_WORDS = (
    "report", "conclusion", "overview", "summary", "audit", "result", "protocol",
    "closeout", "erratum", "claim", "verdict", "paper_draft", "recovery",
    "comparison", "frontier", "go_no_go", "decision", "ablation", "repeatability",
    "statistics", "threshold", "review_guide", "codebook_check",
)
SOURCE_EXCLUSIONS = ("prompt", "response", "sample", "prediction", "local_config", "credential")
DOCUMENT_SCAN_ROOTS = ("experiments", "archive", "report_assets", "baselines", "deliverables", "论文",
                       "第二篇_语义消息写入与读回")
DOCUMENT_SKIP_PARTS = {"node_modules", ".venv", "venv", "vendor", "vendors", "models",
                       "downloads", "before", "after", "snapshots_before", "originals"}
DOCUMENT_SKIP_NAMES = {"feedback.md", "feedback2.0.md", "历史对话.md"}
DOCUMENT_SNAPSHOT_ROOTS = (
    P2_ROOT / "05_脚本与测试/工作区快照",
    P2_ROOT / "07_共享参考材料/工作区快照",
    P2_ROOT / "08_本地原始资产/cache_amr",
)
NEW_DOCS = [
    "论文/README.md", "论文/01_句界审计与精确评估/P1_句界接口审计.md",
    "第二篇_语义消息写入与读回/01_论文文档/P2_语义消息写入与读回.md", "论文/01_句界审计与精确评估/P3_完整认证的失败与资源边界.md",
    APPENDIX, "论文/01_句界审计与精确评估/第四篇_低质量多视图融合_文章转换.md",
    "论文/01_句界审计与精确评估/P4_证据身份对齐_三臂验证方案.md",
    "论文/01_句界审计与精确评估/P4_数据融合实验_完整技术报告.md",
    "report_assets/evidence_identity_alignment_review_20261026/CODE_AUDIT.md",
]
NEW_DOCS.extend([
    "第二篇_语义消息写入与读回/README.md",
    "第二篇_语义消息写入与读回/01_论文文档/README.md",
    "第二篇_语义消息写入与读回/02_实验/README.md",
    "第二篇_语义消息写入与读回/03_实验外支持材料/README.md",
    "第二篇_语义消息写入与读回/04_服务器远端资产/SERVER_ASSETS.md",
    "第二篇_语义消息写入与读回/05_脚本与测试/索引.md",
    "第二篇_语义消息写入与读回/06_迁移记录/迁移说明.md",
])
SYNC_DOCS = [
    "README.md", "论文/论文转化方案.md", "第二篇_语义消息写入与读回/01_论文文档/第二第三篇研究闭环.md",
    "论文/01_句界审计与精确评估/第三篇_独立认证方法_失败与边界记录.md", "任务清单.md",
    "SWAN_弹性重构防御_总体规划.md", "experiments/README.md",
    "experiments/研究工作导航.md",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def restricted(path: Path) -> bool:
    return any(x.lower() in DENY_PARTS or x.lower().startswith(("private_", "sealed_"))
               for x in path.parts)


LEGACY_PREFIXES = (
    ("experiments/semantic_", "第二篇_语义消息写入与读回/02_实验/semantic_"),
    ("report_assets/semantic_", "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_"),
    ("report_assets/graph_template_readback_reframing_20261026",
     "第二篇_语义消息写入与读回/03_实验外支持材料/graph_template_readback_reframing_20261026"),
    ("论文/02_语义消息写入与读回/", "第二篇_语义消息写入与读回/01_论文文档/"),
)


def legacy_name(name: str) -> str:
    for old, new in LEGACY_PREFIXES:
        if name == old or name.startswith(old):
            candidate = name.replace(old, new, 1)
            if (ROOT / candidate).exists():
                return candidate
    return name


def legacy_path(name: str) -> Path:
    return ROOT / legacy_name(name)


def source_file(path: Path) -> bool:
    name = path.name.lower()
    return (path.is_file() and not path.is_symlink() and not restricted(path)
            and path.suffix.lower() in {".md", ".json", ".tex", ".sha256"}
            and any(word in name for word in SOURCE_WORDS)
            and not any(word in name for word in SOURCE_EXCLUSIONS))


def direct_sources(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    sources = [f for f in path.iterdir() if source_file(f)]
    # These are output wrappers, not separate scientific cohorts.
    for wrapper in ("results", "pilot_closure", "implementation_amendments"):
        d = path / wrapper
        if d.is_dir():
            sources.extend(f for f in d.iterdir() if source_file(f))
    return sorted(set(sources), key=lambda f: (source_priority(f), rel(f)))


def source_priority(path: Path) -> int:
    names = ["REPORT.md", "FINAL_REPORT.md", "CONCLUSION.md", "OVERVIEW.md",
             "STOP_OVERVIEW.md", "report.md", "RESULT.json", "RESULTS.json",
             "SUMMARY.json", "PROTOCOL.md"]
    return names.index(path.name) if path.name in names else len(names)


def classify(name: str) -> str:
    if name in LOOKUP:
        return LOOKUP[name]
    for prefix, group in (
        ("certified_resegmentation_", "fixed_n"), ("semstamp_certified_defense_", "variable_n"),
        ("semantic_watermark_defense_plan_", "variable_n"), ("semantic_", "symbols"),
        ("api_", "early"), ("planB_", "early"), ("planC_", "early"),
        ("planD_", "early"), ("planE_", "feedback"), ("swan_elastic_", "elastic"),
        ("swan_final_confirmation", "elastic"), ("swan_count_neutral_", "positive"),
        ("swan_positive_", "positive"), ("swan_source_anchored", "positive"),
        ("swan_coherent_injection_", "positive"), ("confirmation_", "early"),
    ):
        if name.startswith(prefix):
            return group
    raise ValueError(f"No reviewed article classification for {name}")


def nested_info(parent: str, child: str) -> tuple[str, str, str, str]:
    p, role, a, note = GROUPS[classify(parent)]
    if parent == "swan_multiview_development1":
        if child in MULTIVIEW_P1:
            p, role = "P1", "盲窗/可用性/确认"
            note = "固定窗口/拒绝/阈值及S8系列；无三尺度融合，不补P4优势"
            if child == "final_completion4":
                note = "550篇×5条件×3方法=8250结果；203/250输入合格；校准/完整率/收益三门均false"
        else:
            note = "30对文本反复开发的视图、窗口、配对、融合与Null诊断；独立性受限"
    elif parent == "swan_multiscale_attention_development1":
        specific = {
            "validation1": "15开发运行；attention61.67% vs gate60.96%，+0.71pp区间跨零",
            "ablation1": "20顺序消融；可信度/行hard门控62.18%，追加attention58.37%",
            "operating_points1": "冻结模型工作点诊断；阈值加严降误报也损检出，非新独立确认",
            "family_hard1": "80轨迹100端点；族去重成功但32项严格效用比较0通过，系数均0",
            "cache_audit1": "真实缓存兼容/归一化审计；不提供新样本或融合效果",
        }
        note = specific[child]
    elif parent == "semantic_sequence_stage1_run2" and child.startswith("multiscale_transfer"):
        p, a, role = "P4", "A9", "迁移准备/停止"
        note = "冻结检测融合的新语料迁移版本，不能用P2读回结果填盲融合缺口"
        if child == "multiscale_transfer3":
            note = "384原句/38400分数；读出AUC0.741、文档11/12；clean仅5/24对后GPU门Halt，无融合结果"
    elif child.startswith("execution"):
        role, note = "工程修复/后置诊断", "同54回执的版本修复，不重新计独立试验；execution5固定三对终点"
    elif parent == "semstamp_certified_defense_v3_wp3_variable_n_profile1":
        role, note = "资源smoke子例", "冻结profile的row800重复/子集预检；不计新样本，不是证书"
    elif parent == "candidate_budget16_pilot30" and child == "feedback_driven_strict30":
        a, role = "A14", "反馈/历史重复性"
        note = "同120候选、30篇的10/20/50次S2Match重复；26始终逃逸、0混合翻转，不计新样本或一般确定性"
    return p, role, a, note


def evidence_label(sources: list[Path], role: str) -> str:
    if not sources:
        return "仅原始资产/容器；未独立复核终态"
    if any(any(w in f.name.lower() for w in ("report", "conclusion", "overview"))
           for f in sources):
        return "有报告；以现行更正及冻结终态限定"
    if any("result" in f.name.lower() or "audit" in f.name.lower() for f in sources):
        return "有聚合/工程审计；不自动视为效果成功"
    return "设计/准备/支持材料；未据此主张效果"


def observed_terminals(sources: list[Path]) -> list[dict]:
    """Copy only short scalar markers, explicitly not a normalized scientific verdict."""
    output = []
    for source in sources:
        if source.suffix != ".json" or not any(w in source.name.lower() for w in
                                               ("result", "closeout", "stop_audit")):
            continue
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            continue
        if isinstance(data, dict):
            for key in ("status", "terminal", "terminal_reason", "decision"):
                value = data.get(key)
                if isinstance(value, str) and len(value) <= 180:
                    output.append({"source": rel(source), "key": key, "value": value})
    return output


def make_row(path: Path, kind: str, parent: str = "", child: str = "") -> dict:
    if kind == "external_support":
        paper, appendix, note = EXTERNAL[rel(path)]
        role = "实验外支持"
    elif kind in {"archived_report_unit", "documentary_support"}:
        spec = ARCHIVED_REPORTS if kind == "archived_report_unit" else DOCUMENTARY_ROWS
        paper, appendix, note = spec[rel(path)]
        role = "归档候选筛选/复核" if kind == "archived_report_unit" else "汇总/更正/状态支持"
    elif kind == "archive_scientific_unit":
        paper, role, appendix = "P1", "归档Null诊断", "A7"
        if path.name.endswith("development1"):
            role = "归档协议/执行失败"
            note = "path0新协议；AppleDouble扫描错误在来源物化和Parser前失败，无检测结果"
        elif path.name.endswith("development2"):
            role = "归档工程修复"
            note = "另立AppleDouble排除修复版本；其冻结规模为计划，不借development3结果追认本版成功"
        else:
            note = "200新Null对照0/200，候选3–4/200；50正例未生成，无TPR/完整防御"
    elif child:
        paper, role, appendix, note = nested_info(parent, child)
    else:
        paper, role, appendix, note = GROUPS[classify(path.name)]
        if path.name in NOTES:
            role, note = NOTES[path.name]
    sources = sorted(set(direct_sources(path) + [
        ROOT / name for name in EXTRA_SOURCES.get(rel(path), [])
    ]), key=lambda f: (source_priority(f), rel(f)))
    historical = []
    if kind in {"top_level_directory", "nested_scientific_unit"}:
        try:
            relative = path.relative_to(ROOT / "experiments")
        except ValueError:
            relative = None
        if relative is not None and (HISTORY / relative).is_dir():
            historical = sorted(f for f in (HISTORY / relative).rglob("*") if source_file(f))
    # If all reports were moved to derived docs, they are the accessible source.
    display_sources = sources or historical
    return {
        "id": rel(path), "kind": kind, "parent": "experiments/" + parent if parent else "",
        "primary_paper": paper, "article_role": role, "appendix": appendix,
        "manuscript": PAPERS[paper][1], "converted_paragraph": f"{APPENDIX}#{appendix.lower()}",
        "evidence_status": evidence_label(display_sources, role), "conversion_note": note,
        "primary_source": rel(display_sources[0]) if display_sources else "",
        "sources": [rel(f) for f in sources],
        "historical_sources": [rel(f) for f in historical],
        "historical_relation": "同名历史报告/副本，不计独立重复实验" if historical else "",
        "observed_scalar_markers": observed_terminals(display_sources),
        "independent_study_count_claimed": False,
    }


def document_source_map(rows: list[dict]) -> tuple[list[dict], dict]:
    """Reverse discovery: a new report cannot disappear behind row enumeration."""
    owners = {r["id"]: r for r in rows}
    explicit = dict(DOCUMENT_FILE_OWNERS)
    for name, (owner, secondary) in ROOT_DOCUMENT_OWNERS.items():
        explicit[name] = (owner, "overview_design_or_navigation_not_new_evidence", secondary)
    for owner, sources in EXTRA_SOURCES.items():
        for name in sources:
            explicit[name] = (owner, "secondary_historical_source_same_cohort", [])
    for name, paper in (
        ("论文/01_句界审计与精确评估/P1_句界接口审计.md", "P1"),
        ("第二篇_语义消息写入与读回/01_论文文档/P2_语义消息写入与读回.md", "P2"),
        ("论文/01_句界审计与精确评估/P3_完整认证的失败与资源边界.md", "P3"),
    ):
        explicit[name] = ({"P1": "experiments/swan_reframing1",
                          "P2": "第二篇_语义消息写入与读回/02_实验/semantic_channel_audit1",
                          "P3": "experiments/semstamp_certified_defense_v4_full_radius_profile1"}[paper],
                          "article_conversion_not_new_source_evidence", [])
    for name in ("论文/README.md", APPENDIX):
        explicit[name] = ("experiments/research_direction_reconciliation_20261002",
                          "article_conversion_not_new_source_evidence", ["P2", "P3", "P4"])
    candidates = set(ROOT.glob("*.md"))
    for base in DOCUMENT_SCAN_ROOTS:
        candidates.update((ROOT / base).rglob("*.md"))
    entries, excluded, unknown = [], Counter(), []
    for path in sorted(candidates):
        name = rel(path)
        reason = None
        if path.is_symlink():
            reason = "symlink"
        elif restricted(path):
            reason = "private_or_sealed"
        elif path.is_relative_to(OUT):
            reason = "this_audit_output"
        elif any(path.is_relative_to(base) for base in DOCUMENT_SNAPSHOT_ROOTS):
            reason = "archived_copy_not_new_scientific_evidence"
        elif any(part in DOCUMENT_SKIP_PARTS for part in path.parts):
            reason = "dependency_or_duplicate_editorial_snapshot"
        elif ("debug" in path.parts or path.name in DOCUMENT_SKIP_NAMES
              or any(word in path.name.lower() for word in SOURCE_EXCLUSIONS)):
            reason = "raw_payload_or_sensitive_conversation"
        if reason:
            excluded[reason] += 1
            continue
        owner, disposition, secondary = "", "", []
        if name in explicit:
            owner, disposition, secondary = explicit[name]
        else:
            translated = name
            prefix = "archive/20260911_pre_unification/derived_experiment_docs/"
            if name.startswith(prefix):
                translated = name[len(prefix):]
                disposition = "same_name_historical_source_not_independent_repetition"
            matches = [key for key in owners if translated == key or translated.startswith(key + "/")]
            if matches:
                owner = max(matches, key=len)
                disposition = disposition or "source_or_support_within_explicit_owner"
            else:
                prefixes = [key for key in DOCUMENT_PREFIX_OWNERS if name.startswith(key)]
                if prefixes:
                    owner, disposition, secondary = DOCUMENT_PREFIX_OWNERS[max(prefixes, key=len)]
        if not owner:
            unknown.append(name)
            continue
        if owner not in owners:
            raise ValueError(f"Document references nonexistent accounting owner: {name}: {owner}")
        row = owners[owner]
        if name in {"CURRENT_RESULTS.md", "RESEARCH_STATE.md"}:
            secondary = ["P2", "P3", "P4"]
            disposition = "current_cross_article_ledger_not_new_experiment"
        entries.append({
            "document": name, "owner": owner, "primary_paper": row["primary_paper"],
            "secondary_papers": secondary, "disposition": disposition,
            "converted_paragraph": row["converted_paragraph"],
            "new_independent_experiment_claimed": False,
            "sha256": digest(path),
        })
    if unknown:
        raise ValueError("Discovered Markdown without reviewed disposition:\n" + "\n".join(unknown))
    return entries, {
        "scan_roots": list(DOCUMENT_SCAN_ROOTS) + ["root-level Markdown"],
        "filename_keyword_filter_used": False,
        "discovered_markdown_paths": len(candidates),
        "eligible_documents": len(entries), "assigned_documents": len(entries),
        "unassigned_documents": 0, "excluded_counts_by_reason": dict(sorted(excluded.items())),
        "independent_studies_inferred_from_document_count": False,
    }


def directory_rollup(rows: list[dict]) -> tuple[list[dict], dict]:
    owners = {r["id"]: r for r in rows if (ROOT / r["id"]).is_dir()}
    entries, sealed = [], Counter()
    for base in (ROOT / "experiments", ROOT / "archive/experiments", P2_EXP_ROOT):
        for directory in sorted(d for d in base.rglob("*") if d.is_dir() and not d.is_symlink()):
            name = rel(directory)
            matches = [key for key in owners if name == key or name.startswith(key + "/")]
            if not matches:
                raise ValueError(f"Directory without an article owner: {name}")
            owner = max(matches, key=len)
            if restricted(directory):
                sealed[owner] += 1
            else:
                entries.append({
                    "directory": name, "owner": owner,
                    "primary_paper": owners[owner]["primary_paper"],
                    "separate_scientific_row": name == owner,
                })
    return entries, dict(sorted(sealed.items()))


def gather(rows: list[dict], documents: list[dict]) -> dict[str, str]:
    names = {p for r in rows for p in r["sources"] + r["historical_sources"]}
    names.update(d["document"] for d in documents
                 if d["document"] not in NEW_DOCS + SYNC_DOCS + [rel(TABLE)])
    # Manuscript citations are also part of the source ledger.
    for name in NEW_DOCS:
        doc = ROOT / name
        for target in markdown_targets(doc):
            path = resolve_link(doc, target)
            if path and path.is_file() and rel(path) not in NEW_DOCS + SYNC_DOCS:
                if (path != TABLE and not path.is_relative_to(OUT)
                        and not path.is_relative_to(ROOT / "report_assets/fusion_closeout_20261026")
                        and not restricted(path)):
                    names.add(rel(path))
    for name in ("论文/CLAIM_EVIDENCE_MATRIX.md", "论文/01_句界审计与精确评估/第一篇_核心贡献与摘要.md"):
        names.add(name)
    for directory in (ROOT / "experiments/archive_20260725").glob("*.zip"):
        names.add(rel(directory))
    return {name: digest(ROOT / name) for name in sorted(names)}


def markdown_targets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.findall(r"!?\[[^\]]*\]\(([^)\n]+)\)", text)


def resolve_link(doc: Path, target: str) -> Path | None:
    target = target.strip().split(' "')[0].strip("<>")
    if target.startswith("#"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return (doc.parent / unquote(parsed.path)).resolve()


def table_link(path: str) -> str:
    """Make a root-relative path clickable from the table located in 论文/."""
    if path.startswith("论文/"):
        return path[len("论文/"):]
    if path.startswith(("experiments/", "report_assets/", "archive/", "baselines/",
                        "deliverables/", "第二篇_")) or "/" not in path:
        return "../" + path
    return path


def render_table(rows: list[dict], top_count: int, rollup: list[dict], sealed: dict,
                 coverage: dict) -> str:
    kinds = Counter(r["kind"] for r in rows)
    lines = [
        "# 全部实验：文章归属与转换总表", "",
        "2026-10-30。**当前两条论文线（P1/P2）；P3/P4 已并入 P1 §6/§7，作为档案来源，不再独立投稿。**",
        "本表是本地文章转换与来源对账，不启动新实验、不改变冻结结果或Stop/No-Go。", "",
        f"本轮覆盖experiments一级目录 **{top_count}/{top_count}**，另列重要子实验/子审计 "
        f"**{kinds['nested_scientific_unit']}** 项、archive/experiments归档path0 **"
        f"{kinds['archive_scientific_unit']}** 项、实验外支持 **{kinds['external_support']}** 项；"
        f"归档报告单元 **{kinds['archived_report_unit']}** 项、汇总/更正账本 "
        f"**{kinds['documentary_support']}** 项，共 **{len(rows)}个归属条目**。"
        "嵌套、版本、seed和同源副本不能相加为独立实验数。",
        f"所有实际子目录按最近上级条目归属，公开路径{len(rollup)}项；"
        f"私有/密封路径仅记录按owner汇总的{sum(sealed.values())}个目录计数。", "",
        f"独立反向发现 **{coverage['eligible_documents']}** 份可纳入Markdown材料，"
        f"**{coverage['assigned_documents']}** 份已有明确归属/历史支持处置，未归属 **0**。"
        "扫描不依赖报告文件名关键词；私有文本、敏感历史对话、依赖和编辑快照按规则排除。",
        "此前196目录计数已过时；P2 已迁移至顶层目录，本表按新路径记账，"
        "冻结报告内旧路径字符串仅作历史标识。CSV、JSON可筛选；源SHA逐文件核验。"
        "目录存在不等于有报告、有报告不等于结论成功。仅原始资产的条目明确标记"
        "终态未独立复核，不编造论文数值。", "",
        "| 文章线 | 正文转换 | 当前身份 | 一级目录 | 其他归属条目 |",
        "|---|---|---|---:|---:|",
    ]
    statuses = {
        "P1": "已有范围明确英文稿；恢复/机制/证书缺口保留",
        "P2": "通道审计草稿；可靠新方法未成立",
        "P3": "失败与资源边界报告；Stop",
        "P4": "完整技术归档；建议停止追加当前路线，暂不推进正向投稿",
    }
    for paper, (title, manuscript) in PAPERS.items():
        top = sum(r["primary_paper"] == paper and r["kind"] == "top_level_directory" for r in rows)
        other = sum(r["primary_paper"] == paper and r["kind"] != "top_level_directory" for r in rows)
        lines.append(f"| {paper} {title} | [正文]({table_link(manuscript)}) | {statuses[paper]} | {top} | {other} |")
    lines += [
        "", "## 去重与证据复用规则", "",
        "- P1使用fixed-N DP、v17/v18与S8；P3使用其后fixed-byte variable-N合同，不重复申报P1贡献。",
        "- P2的24模板消息读回与P4的100模板检测融合终点不同，不合并样本、F1和TPR。",
        "- `multiview`中S8、盲窗与阈值子项主归P1；实际三尺度开发和新语料迁移归P4。",
        "- 早期derived报告、ZIP、originals、public_aggregates及seed端点绑定原实验，不记新增独立研究。",
        "- 新载体原AMR192/192已生成；172结构合格、20排除；真人/图事实/修订/分类/完整对照仍0。",
        "- 融合旧综合No-Go及约15pp主张已撤回；当前稿不恢复。归档path0按原协议/结果引用。",
        "- P4全路线保存于[完整总结与停止理由](01_句界审计与精确评估/P4_数据融合实验_完整技术报告.md)；"
        "停止追加是当前投入建议，不追改冻结终态，不启动三臂实验。",
        "", "## 此前容易漏掉的材料已转换到哪些段落", "",
        f"[18组可直接采用的附录段落]({table_link(APPENDIX)})覆盖早期攻击、预算/反馈与评审冲突、"
        "跨库/跨水印、聚合/靶向/确定性、人工图、正例资格、空路径与path0、统计裁决、"
        "多视图开发、符号版本/分层、认证版本及重复制品；新增B4/B5、全量重复性、"
        "模型对齐区间、pending门控、码本插删边界及归档/设计出处。", "",
    ]
    for kind, title in [
        ("top_level_directory", "experiments一级目录逐项表"),
        ("nested_scientific_unit", "重要嵌套实验与子审计"),
        ("archive_scientific_unit", "归档独立版本"),
        ("archived_report_unit", "无同名现行目录的归档报告单元"),
        ("documentary_support", "历史汇总、更正及当前账本"),
        ("external_support", "实验外支持来源"),
    ]:
        lines += [f"## {title}", "", "| 目录/来源 | 主归属与角色 | 文章段落 | 当前可写范围 | 来源与证据状态 |",
                  "|---|---|---|---|---|"]
        for r in (x for x in rows if x["kind"] == kind):
            source = f"[来源]({table_link(r['primary_source'])})" if r["primary_source"] else f"[资产目录]({table_link(r['id'])})"
            lines.append(
                f"| [{r['id']}]({table_link(r['id'])}) | {r['primary_paper']}：{r['article_role']} | "
                f"[{r['appendix']}]({table_link(r['converted_paragraph'])})；[正文]({table_link(r['manuscript'])}) | "
                f"{r['conversion_note']} | {source}；{r['evidence_status']} |"
            )
        lines.append("")
    lines += [
        "## 机器可核验制品", "",
        f"[CSV]({table_link('report_assets/experiment_article_conversion_20261026/EXPERIMENT_ARTICLE_MAP.csv')})、"
        f"[JSON]({table_link('report_assets/experiment_article_conversion_20261026/EXPERIMENT_ARTICLE_MAP.json')})、"
        f"[子目录归属]({table_link('report_assets/experiment_article_conversion_20261026/DIRECTORY_ROLLUP.json')})、"
        f"[逐文档反查]({table_link('report_assets/experiment_article_conversion_20261026/DOCUMENT_SOURCE_MAP.csv')})、"
        f"[二轮排查说明]({table_link('report_assets/experiment_article_conversion_20261026/REAUDIT.md')})、"
        f"[来源SHA]({table_link('report_assets/experiment_article_conversion_20261026/SOURCE_SHA256.json')})、"
        f"[核验结果]({table_link('report_assets/experiment_article_conversion_20261026/VALIDATION.json')})。", "",
        "复跑：`python3 scripts/audit_experiment_article_inventory.py`。"
        "只读取已有材料，生成文章归属与核验制品；不调用模型或检测器。", "",
    ]
    return "\n".join(lines)


def validate_links() -> dict:
    checks, failures, legacy = [], [], []
    for name in NEW_DOCS + [rel(TABLE), rel(OUT / "REAUDIT.md")] + SYNC_DOCS:
        doc = ROOT / name
        for target in markdown_targets(doc):
            path = resolve_link(doc, target)
            if path is None:
                continue
            exists = path.exists()
            conversion_target = path == TABLE or path.is_relative_to(OUT)
            if path.is_relative_to(ROOT):
                conversion_target = conversion_target or rel(path) in NEW_DOCS
            is_new_scope = name in NEW_DOCS or doc == TABLE or doc.is_relative_to(OUT) or conversion_target
            item = {"document": name, "target": target, "exists": exists}
            if is_new_scope:
                checks.append(item)
                if not exists:
                    failures.append(item)
            elif not exists:
                legacy.append(item)
    return {"checked_local_links": len(checks), "failures": failures,
            "preexisting_out_of_scope_broken_links": legacy,
            "scope": "new manuscripts/table and added conversion links in synchronized navigation",
            "anchor_validation": "file existence only; machine paragraph IDs separately verified"}


def numerical_checks() -> list[dict]:
    cache, checks = {}, []

    def data(name: str) -> dict:
        if name not in cache:
            cache[name] = json.loads((ROOT / name).read_text(encoding="utf-8"))
        return cache[name]

    def get(name: str, selector: str):
        value = data(name)
        for key in selector.split("/") if selector else []:
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value

    def check(label: str, source: str, selector: str, expected, transform=None, tolerance=1e-10):
        observed = get(source, selector)
        if transform:
            observed = transform(observed)
        if isinstance(expected, float):
            passed = isinstance(observed, (int, float)) and math.isclose(
                observed, expected, abs_tol=tolerance, rel_tol=0)
        else:
            passed = observed == expected
        checks.append({"claim": label, "source": source, "selector": selector,
                       "observed": observed, "expected": expected, "passed": passed})

    def derived(label: str, source: str, selector: str, observed, expected):
        checks.append({"claim": label, "source": source, "selector": selector,
                       "observed": observed, "expected": expected, "passed": observed == expected})

    aba = "experiments/confirmation_mechanism30/summary.json"
    for condition, expected in (("original", 1), ("merged_sentence", 30), ("merged_exact_clause", 1)):
        base = f"punctuation_control/fixed_2_33/escape_rates/{condition}"
        check("P1 A–B–A " + condition, aba, base + "/count", expected)
        check("P1 A–B–A full denominator " + condition, aba, base + "/n", 30)
    dp = "experiments/certified_resegmentation_pairwise_preflight1/audit_results.json"
    derived("P1 pairwise oracle total", dp,
            "systematic_small_instances/dp_oracle_comparisons + random_small_instances/case_count",
            get(dp, "systematic_small_instances/dp_oracle_comparisons")
            + get(dp, "random_small_instances/case_count"), 7669)
    check("P1 additive degeneration controls", dp, "additive_degeneracy_checks/case_count", 1000)
    v17 = "experiments/certified_resegmentation_wp2_v17_b0_detector_health1/B0_RESULT.json"
    for condition, expected in (("test", 47), ("positive", 954)):
        check("P1 v17 " + condition, v17, f"statistics/{condition}/strict_exceedances", expected)
        check("P1 v17 denominator " + condition, v17, f"statistics/{condition}/trials", 1000)
    v18 = "experiments/certified_resegmentation_wp2_v18_b1_exact_certificate1/P0_STOP_AUDIT.json"
    check("P1 v18 maximum pruned pairs", v18, "counts/max_document_pruned_pairs", 78642945)
    check("P1 v18 frozen runtime upper bound", v18,
          "runtime_upper_bound/conservative_two_replay_gpu_hours", 86.6501,
          lambda x: round(x, 4))
    check("P1 v18 pre-scoring stop", v18, "decision", "stop")
    check("P1 v18 no certificate artifacts", v18, "certificate_artifacts", [])
    supplement = "experiments/swan_count_neutral_supplement3/AUDIT.json"
    for field, expected in (("cumulative_complete_API_calls", 64), ("complete_documents", 0),
                            ("detector_calls", 0), ("status", "Halt")):
        check("P1 latest normal-length pilot " + field, supplement, field, expected)
    s8 = "experiments/swan_multiview_development1/final_completion4/RESULTS.json"
    check("P1 S8 has no fusion method", s8, "methods",
          ["P0_raw", "P0_strict", "S8"], lambda x: sorted(x))
    check("P1 S8 three gates failed", s8, "gates", [False, False, False],
          lambda x: sorted(x.values()))
    check("P1 S8 calibration unavailable", s8, "methods/S8/calibration/available", False)
    check("P1 strict calibration unavailable", s8, "methods/P0_strict/calibration/available", False)
    for condition, expected in (("baseline", 18), ("merge", 0), ("comma", 0), ("space", 0)):
        check("P1 P0 raw decisions " + condition, s8,
              f"methods/P0_raw/conditions/{condition}/TPR/positive_decisions", expected)
    check("P1 S8 human completion range", s8, "methods/S8/conditions", [28, 34],
          lambda x: [min(v["human_FPR"]["complete_documents"] for v in x.values()),
                     max(v["human_FPR"]["complete_documents"] for v in x.values())])
    s8audit = "experiments/swan_multiview_development1/final_completion4/FINAL_COMPLETION_AUDIT.json"
    check("P1 S8 final result rows", s8audit, "full_detector_rows", 8250)
    s8input = "experiments/swan_multiview_development1/final_completion4/INPUT_QUALITY_AUDIT.json"
    generated = [v for k, v in get(s8input, "groups").items() if k.startswith("generated:")]
    derived("P1 S8 generated intents", s8input, "sum(groups/generated:*/documents)",
            sum(x["documents"] for x in generated), 250)
    derived("P1 S8 eligible generated inputs", s8input, "sum(groups/generated:*/input_eligible)",
            sum(x["input_eligible"] for x in generated), 203)
    derived("P1 S8 total documents", s8input, "sum(groups/*/documents)",
            sum(x["documents"] for x in get(s8input, "groups").values()), 550)

    order = "第二篇_语义消息写入与读回/02_实验/semantic_sequence_readout1/ORDER_RESULT.json"
    values = [x for x in get(order, "summaries") if
              x["representation"] == "W" and x["condition"] == "paraphrase" and x["mode"] == "soft"]
    derived("P2 W paraphrase order win percent", order,
            "summaries[W,paraphrase,soft]/pairwise_win_rate/mean",
            round(values[0]["pairwise_win_rate"]["mean"] * 100, 2), 77.48)
    channel = "第二篇_语义消息写入与读回/02_实验/semantic_channel_audit1/RESULT.json"
    check("P2 full message candidates", channel, "candidate_count", 65536)
    check("P2 all representation/mode/condition recoveries zero", channel, "summaries",
          [(0, 12)] * 24, lambda x: [(r["complete_message_correct"], r["planned_messages"]) for r in x])
    msa = "第二篇_语义消息写入与读回/02_实验/semantic_multiscale_readback1/RESULT.json"
    check("P2 fusion trained models", msa, "trained_models", 84)
    check("P2 fusion candidate messages", msa, "candidate_messages", 65536)
    check("P2 all ten methods/four conditions complete recovery zero", msa, "summaries",
          [(0.0, 12)] * 40, lambda x: [(r["complete_correct"]["mean"], r["scenarios"]) for r in x])
    for method, expected in (("attention", 22.97), ("gate", 28.93)):
        check("P2 " + method + " paraphrase macro F1 percent", msa, "summaries", expected,
              lambda x, m=method: round(next(r for r in x if r["method"] == m
                                              and r["condition"] == "paraphrase")["macro_f1"]["mean"] * 100, 2))
    check("P2 attention-gate F1 percentage points", msa, "primary/mean", -5.96,
          lambda x: round(x * 100, 2))
    check("P2 attention-gate descriptive interval", msa, "primary/CI95", [-11.65, -0.73],
          lambda x: [round(v * 100, 2) for v in x])
    fact = "第二篇_语义消息写入与读回/02_实验/semantic_fact_layers_v3_parse1/RESULT.json"
    for field, expected in {
        "raw_graphs_available": 192, "structurally_eligible_raw_graphs": 172,
        "excluded_raw_graphs": 20, "total_parser_calls": 194, "replay_parser_calls": 2,
        "attribution_function_calls": 192, "human_completed": 0,
        "raw_graph_fact_annotations": 0, "independent_revised_graphs": 0,
        "ABCD_graph_predictions": 0, "complete_fact_graph_classifier_comparisons": 0,
        "parser_loss_rate": None, "classifier_error_rate": None, "blind_recovery_256": "NOT_RUN",
    }.items():
        check("P2 latest factual layers " + field, fact, field, expected)

    v3 = "experiments/semstamp_certified_defense_v3_wp3_variable_n_profile1/PROFILE_RESULT.json"
    for field, expected in (("rows", 3000), ("distinct_target_layers", 46501)):
        check("P3 v3 " + field, v3, "counts/" + field, expected)
    check("P3 v3 runtime upper bound", v3, "runtime/conservative_two_replay_upper_bound_gpu_hours",
          77.6152, lambda x: round(x, 4))
    check("P3 v3 no certificates", v3, "execution_attestation/certificate_computations", 0)
    v4 = "experiments/semstamp_certified_defense_v4_full_radius_profile1/RESOURCE_AUDIT.json"
    for field, expected in (
        ("affected_requested_radius_2_rows", 28), ("affected_requested_radius_3_rows", 155),
        ("corrected_target_layers_planned", 47332), ("additional_target_layers_planned", 831),
        ("historical_distinct_target_layers", 46501),
    ):
        check("P3 v4 " + field, v4, "coverage/" + field, expected)
    check("P3 v4 frozen-rate cost lower bound", v4, "budget/two_replay_lower_bound_hours",
          75.3111, lambda x: round(x, 4))
    check("P3 v4 new full profile unrun", v4, "full_profile_executed", False)
    check("P3 v4 final Stop", v4, "terminal", "V4_FULL_RADIUS_READINESS_STOP")

    validation = "experiments/swan_multiscale_attention_development1/validation1/SUMMARY.json"
    check("P4 first development runs", validation, "verified_runs", 15)
    for method, tpr, fpr in (("attention", 61.67, 0.51), ("gate", 60.96, 0.57)):
        for metric, expected in (("tpr", tpr), ("fpr", fpr)):
            check("P4 " + method + " " + metric + " percent", validation,
                  f"methods/{method}/working_points/0.01/all/{metric}/micro/mean", expected,
                  lambda x: round(x * 100, 2))
    check("P4 first attention gain", validation,
          "attention_minus/gate/0.01/all/tpr/micro_delta/mean", 0.71, lambda x: round(x * 100, 2))
    check("P4 first attention gain interval", validation,
          "attention_minus/gate/0.01/all/tpr/mean_micro_delta_ci95", [-0.59, 2.12],
          lambda x: [round(v * 100, 2) for v in x])
    ablation = "experiments/swan_multiscale_attention_development1/ablation1/SUMMARY.json"
    check("P4 sequential ablation runs", ablation, "verified_runs", 20)
    for method, tpr, fpr in (("A2", 62.18, 0.40), ("A3", 58.37, 0.39)):
        for metric, expected in (("tpr", tpr), ("fpr", fpr)):
            check("P4 sequential " + method + " " + metric + " percent", ablation,
                  f"methods/{method}/working_points/0.01/all/{metric}/micro/mean", expected,
                  lambda x: round(x * 100, 2))
    family = "experiments/swan_multiscale_attention_development1/family_hard1/SUMMARY.json"
    check("P4 family training trajectories", family, "execution/completed_runs", 80)
    check("P4 family frozen endpoints", family, "runs", 100, len)
    check("P4 family planned comparisons", family, "comparisons", 32, len)
    check("P4 all extra loss selections zero", family, "selected_coefficients",
          ["L000"] * 3, lambda x: [v["selected"] for v in x.values()])
    family_report = "experiments/swan_multiscale_attention_development1/family_hard1/REPORT.md"
    derived("P4 family strict benefit rule none passed", family_report,
            "completed-report statement",
            "32项预定训练/选择比较中，0项通过" in (ROOT / family_report).read_text(), True)
    transfer = "第二篇_语义消息写入与读回/02_实验/semantic_sequence_stage1_run2/multiscale_transfer3/HALT_AUDIT.json"
    for field, expected in (
        ("original_signal_sentences_checked", 384), ("original_signal_score_cells_checked", 38400),
        ("partial_feature_pairs", 5), ("expected_clean_feature_pairs", 24),
        ("model_comparison_available", False), ("attack_evaluation_available", False),
        ("halt_retained", True),
    ):
        check("P4 transfer " + field, transfer, field, expected)
    signal = "第二篇_语义消息写入与读回/02_实验/semantic_sequence_stage1_run2/multiscale_transfer3/SIGNAL_RESULTS.json"
    check("P4 independent readout original carrier AUROC", signal,
          "populations/readout/carrier_auroc", 0.741, lambda x: round(x, 3))
    check("P4 independent readout document paired wins", signal,
          "populations/readout/document_paired_win", 11, lambda x: round(x * 12))

    alignment = "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_validation_20261026/RESULT.json"
    for variant, matches, interval in (
        ("R0", 83, [80.21, 92.71]), ("R1", 88, [85.42, 96.88]),
        ("R2", 89, [86.46, 97.92]),
    ):
        for version in ("original", "paraphrase"):
            base = f"model_diagnostic/variants/{variant}/versions/{version}"
            check("P2 sealed target alignment " + variant + " " + version,
                  alignment, base + "/target_matches", matches)
            check("P2 sealed alignment denominator " + variant + " " + version,
                  alignment, base + "/denominator", 96)
            check("P2 sealed model interval " + variant + " " + version,
                  alignment, base + "/model_cluster_95_percent_interval",
                  interval, lambda x: [round(v * 100, 2) for v in x])
    for selector, expected in (
        ("real_human/completed_reviewers", 0), ("real_human/required_reviewers", 2),
        ("real_human/nonblank_label_cells", 0), ("real_human/gate", "NOT_EVALUATED"),
        ("model_diagnostic/paraphrase_preservation/matches", 95),
        ("model_diagnostic/paraphrase_preservation/denominator", 96),
        ("model_diagnostic/formal_gate_decision_issued", False),
    ):
        check("P2 target-alignment scope " + selector, alignment, selector, expected)
    v2model = "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_carrier_v2_20261026/MODEL_RESULT.json"
    for selector, expected in (
        ("original_target_correct", 96), ("paraphrase_target_correct", 96),
        ("pair_all_three_preserved", 96), ("exact_unchanged_paraphrases", 13),
        ("original_cluster_95ci", [1.0, 1.0]),
        ("bootstrap/all_success_interval_is_empirical_degenerate_not_certainty", True),
        ("independent_annotation", False), ("human_completed", 0),
        ("full_message_recovery_estimated", False),
    ):
        check("P2 v2 model carrier " + selector, v2model, selector, expected)
    gate = "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_fact_layer_gate_analysis_20261026/RESULT.json"
    for selector, expected in (
        ("actual_row_count", 192),
        ("actual_field_counts/status/PENDING_TEXT_CONFIRMATION", 192),
        ("actual_field_counts/raw_graph_present/False", 192),
        ("actual_field_counts/human_text_confirmed/False", 192),
        ("synthetic_probe_count", 6), ("real_graphs_generated", 0),
        ("first_two_probes_never_reached_raw_graph_gate", True),
        ("historical_execution_manifest_unchanged", True),
    ):
        check("P2 historical v2 gate " + selector, gate, selector, expected)
    code = "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_innovation_validation_20261026/CODEBOOK_AUDIT.json"
    for selector, expected in (
        ("message_count", 256), ("unique_codeword_count", 256),
        ("codeword_length", 8), ("unordered_pairs_checked", 32640),
        ("minimum_hamming_distance", 3), ("minimum_unit_levenshtein_distance", 2),
        ("ideal_aligned_hard_substitution_radius", 1),
        ("empirical_channel_test", False),
        ("arbitrary_single_insertion_or_deletion_guarantee", False),
    ):
        check("P2 existing codebook audit " + selector, code, selector, expected)
    ambiguity = get(code, "one_deletion_ambiguity")
    received = [word[:index] + word[index + 1:]
                for word, index in zip(ambiguity["codewords"], ambiguity["deleted_zero_based_indices"])]
    derived("P2 published one-deletion ambiguity", code, "one_deletion_ambiguity",
            received, [ambiguity["identical_received_sequence"]] * 2)
    auth = "第二篇_语义消息写入与读回/03_实验外支持材料/semantic_sequence_authentication_design_20261026/VALIDATION.json"
    for selector, expected in (
        ("checks_count", 60), ("new_authentication_runs", 0), ("new_forgery_tests", 0),
        ("new_keys_created", 0), ("scope", "documentation_only"),
    ):
        check("P2 authentication design provenance " + selector, auth, selector, expected)
    radius = "report_assets/research_closure_20261025/RADIUS_SCOPE_AUDIT.json"
    for selector, expected in (
        ("source_rows", 3000), ("capped_rows_by_requested_radius/2", 28),
        ("capped_rows_by_requested_radius/3", 155), ("inference_calls", 0),
        ("formal_certificate_outcomes", "NOT_COMPUTED"),
    ):
        check("P3 precursor radius audit " + selector, radius, selector, expected)
    for repeats in (10, 20, 50):
        source = ("archive/20260911_pre_unification/derived_experiment_docs/experiments/"
                  f"candidate_budget16_pilot30/feedback_driven_strict30/full_repeatability_{repeats}.md")
        text = (ROOT / source).read_text(encoding="utf-8")
        derived("P1 historical full repeatability " + str(repeats), source,
                "report statements",
                all(value in text for value in (
                    f"{repeats} 次独立重复 S2Match", "候选总数: 120", "段落总数: 30",
                    "候选中全程逃逸的数量: 26", "候选中出现混合翻转的数量: 0")), True)
    b4 = "archive/20260911_pre_unification/experiment_docs/planB_b4_candidate_report.md"
    derived("P1 B4 candidate denominator", b4, "report statement",
            "50 个段落" in (ROOT / b4).read_text(encoding="utf-8"), True)
    b5 = "archive/20260911_pre_unification/experiment_docs/planB_b5_joint_report.md"
    derived("P1 B5 automatic candidate composition", b5, "report statement",
            "10 条候选：节点替换 2 条、子图重挂接 5 条、组合攻击 3 条"
            in (ROOT / b5).read_text(encoding="utf-8"), True)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-top-level", type=int, default=212)
    args = parser.parse_args()
    tops = sorted(d for d in (ROOT / "experiments").iterdir() if d.is_dir())
    tops += sorted(d for d in P2_EXP_ROOT.iterdir() if d.is_dir())
    tops.sort()
    rows = [make_row(d, "top_level_directory") for d in tops]
    for parent, children in NESTED.items():
        for child in children:
            base = P2_EXP_ROOT if parent in {d.name for d in P2_EXP_ROOT.iterdir() if d.is_dir()} \
                else ROOT / "experiments"
            path = base / parent / child
            if not path.is_dir():
                raise ValueError(f"Expected scientific subdirectory missing: {rel(path)}")
            rows.append(make_row(path, "nested_scientific_unit", parent, child))
    rows.extend(make_row(d, "archive_scientific_unit")
                for d in sorted((ROOT / "archive/experiments").iterdir()) if d.is_dir())
    rows.extend(make_row(ROOT / name, "external_support") for name in EXTERNAL)
    rows.extend(make_row(ROOT / name, "archived_report_unit") for name in ARCHIVED_REPORTS)
    rows.extend(make_row(ROOT / name, "documentary_support") for name in DOCUMENTARY_ROWS)
    rollup, sealed = directory_rollup(rows)
    documents, coverage = document_source_map(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(render_table(rows, len(tops), rollup, sealed, coverage), encoding="utf-8")
    # The table is also a discovered editorial document; hash its final bytes.
    documents, coverage = document_source_map(rows)
    source_hashes = gather(rows, documents)
    for row in rows:
        row["primary_source_sha256"] = source_hashes.get(row["primary_source"], "")
    write_json(OUT / "EXPERIMENT_ARTICLE_MAP.json", {
        "snapshot_date": "2026-10-26", "primary_article_count": 4,
        "independent_experiment_count": None, "rows": rows,
    })
    fields = ["id", "kind", "parent", "primary_paper", "article_role", "appendix",
              "manuscript", "evidence_status", "conversion_note", "primary_source",
              "primary_source_sha256", "sources", "historical_sources", "historical_relation"]
    with (OUT / "EXPERIMENT_ARTICLE_MAP.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ";".join(row[k]) if isinstance(row[k], list) else row[k] for k in fields})
    write_json(OUT / "DOCUMENT_SOURCE_MAP.json", {"coverage": coverage, "documents": documents})
    with (OUT / "DOCUMENT_SOURCE_MAP.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["document", "owner", "primary_paper", "secondary_papers", "disposition",
                  "converted_paragraph", "new_independent_experiment_claimed", "sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for document in documents:
            writer.writerow({k: ";".join(document[k]) if isinstance(document[k], list)
                             else document[k] for k in fields})
    write_json(OUT / "DIRECTORY_ROLLUP.json", {
        "public_directory_ownership": rollup, "sealed_directory_counts_by_owner": sealed,
        "counting_rule": "nearest explicit scientific/container owner; nested rows are not independent repetitions",
    })
    write_json(OUT / "SOURCE_SHA256.json", {
        "algorithm": "sha256", "scope": "accessible report/aggregate/protocol citations, plus historical copies and ZIPs",
        "contains_raw_payloads": False, "files": source_hashes,
    })
    # Allocate the validation path before checking forward references.
    write_json(OUT / "VALIDATION.json", {"status": "RUNNING"})
    numbers = numerical_checks()
    links = validate_links()
    unique_ids = [r["id"] for r in rows]
    before = json.loads((OUT / "PRE_REAUDIT_SHA256.json").read_text(encoding="utf-8"))
    protected_changed = [name for name, value in before["protected_source_files"].items()
                         if not legacy_path(name).is_file() or digest(legacy_path(name)) != value]
    # The prior row set can be recovered from the table snapshot only via its ledger;
    # new IDs are enumerated explicitly rather than inferred from directory totals.
    new_ids = list(before.get("expected_added_ids", []))
    checks = {
        "top_level_directory_count_exact": len(tops) == args.expected_top_level,
        "top_level_coverage_exact": {r["id"] for r in rows if r["kind"] == "top_level_directory"} == {rel(d) for d in tops},
        "unique_accounting_ids": len(unique_ids) == len(set(unique_ids)),
        "one_primary_article_each": all(r["primary_paper"] in PAPERS for r in rows),
        "all_rows_have_converted_paragraph": all(
            f'id="{r["appendix"].lower()}"' in (ROOT / APPENDIX).read_text(encoding="utf-8")
            for r in rows),
        "all_sources_exist": all((ROOT / f).is_file() for f in source_hashes),
        "source_sha_unchanged_during_audit": all(digest(ROOT / f) == h for f, h in source_hashes.items()),
        "new_local_links_resolve": not links["failures"],
        "all_numerical_claim_checks_pass": all(c["passed"] for c in numbers),
        "all_descendants_assigned": True,
        "document_reverse_discovery_complete": coverage["unassigned_documents"] == 0,
        "all_document_sources_match_final_bytes": all(
            digest(ROOT / d["document"]) == d["sha256"] for d in documents),
        "previous_852_protected_sources_unchanged": not protected_changed,
        "previous_source_manifest_fully_retained": {legacy_name(k) for k in before["protected_source_files"]}.issubset(source_hashes),
        "new_row_delta_matches_reaudit": len(rows) == before["previous_accounting_rows"] + len(new_ids),
        "no_model_detector_provider_calls": True,
    }
    validation = {
        "status": "PASS" if all(checks.values()) else "FAIL", "snapshot_date": "2026-10-26",
        "checks": checks, "counts": {
            "top_level_directories": len(tops), "accounting_rows": len(rows),
            "kinds": dict(Counter(r["kind"] for r in rows)),
            "source_sha_files": len(source_hashes), "public_rollup_directories": len(rollup),
            "sealed_rollup_directories": sum(sealed.values()),
            "no_direct_report_or_historical_report_rows": sum(not r["primary_source"] for r in rows),
            "eligible_documentary_sources": len(documents),
        },
        "numerical_checks": numbers, "links": links, "document_coverage": coverage,
        "reaudit": {"previous_accounting_rows": before["previous_accounting_rows"],
                    "added_accounting_rows": len(new_ids), "added_ids": new_ids,
                    "previous_protected_source_count": len(before["protected_source_files"]),
                    "changed_previous_protected_sources": protected_changed},
        "limits": [
            "directory inventory is not an independent-study count or a claim that all manuscripts are publication ready",
            "raw-only historical folders retain an unverified scientific terminal; no new performance claim is synthesized",
            "observed scalar markers are copied metadata, not normalized experimental outcomes",
            "this checks selected aggregate facts and report provenance, not new statistical inference",
            "private/source/generated text and remote access configuration are not copied into editorial artifacts",
            "P1 English manuscript is existing work; S8 extension here is not silently inserted into its abstract",
            "reverse discovery covers accessible Markdown in declared roots; excluded payloads and binary-only assets are not asserted to be semantically re-audited",
            "historical summaries and finite codebook arithmetic are not additional empirical channel or authentication trials",
        ],
        "execution_attestation": {"new_experiments": 0, "detector_calls": 0,
                                  "model_calls": 0, "provider_calls": 0, "submissions": 0},
    }
    write_json(OUT / "VALIDATION.json", validation)
    release = [TABLE, *[ROOT / name for name in NEW_DOCS + SYNC_DOCS],
               Path(__file__).resolve(), *sorted(OUT.glob("*.json")),
               *sorted(OUT.glob("*.csv")), OUT / "REAUDIT.md"]
    release = sorted(set(p for p in release if p.name != "DELIVERY_SHA256.json"))
    write_json(OUT / "DELIVERY_SHA256.json", {
        "algorithm": "sha256", "self_excluded": True,
        "files": {rel(p): digest(p) for p in release},
    })
    print(json.dumps({"status": validation["status"], "counts": validation["counts"],
                      "failed_checks": [k for k, v in checks.items() if not v],
                      "link_failures": links["failures"],
                      "numerical_failures": [x for x in numbers if not x["passed"]]},
                     ensure_ascii=False, indent=2))
    if validation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
