# 论文前实验状态与补强计划

下面这份表是按“能不能直接写论文”来整理的，不是按“还有没有任何可做的事”来整理的。

## 结论先说

当前主线实验已经完成，可以开始写论文结果部分。

现在最缺的不是核心结论，而是投稿层面的收口工作：

- 把现有结果统一成一张主表和两三张主图；
- 把失败案例按类型写成一段分析；
- 如果要冲更强的投稿，再补一个同数据上的外部 baseline 或多评审人工复核。

## 已完成的实验

| 模块 | 状态 | 关键结果 | 产物 |
|---|---|---|---|
| 主结果 / 严格有效率 | 已完成 | Strict Valid ASR = 30.0%（9/30），95% CI = [14.7%, 49.4%] | `experiments/candidate_budget16_pilot30/feedback_driven_strict30/paper_final_bundle.md` |
| 阈值敏感性 | 已完成 | 0.65–0.80 形成平台，Strict Valid ASR 固定为 30.0% | `threshold_sensitivity.md/.png/.json` |
| 50 次重复稳定性（严格子集） | 已完成 | 9/9 成功段落在 50 次重复下全部稳定 | `repeatability_50.md/.json` |
| 50 次重复稳定性（全量 120 候选） | 已完成 | 120/120 候选二值判定稳定，0 个 mixed flip | `full_repeatability_50.md/.json` |
| 候选预算消融 | 已完成 | K=4/8/16 的 Valid ASR 都是 30.0%；K=1 为 33.3%，K=2 为 26.7% | `candidate_budget16_pilot30/judged_analysis.json` |
| 提示组件消融 | 已完成 | boundary-only / full E2 / weak / no_boundary 的质量-逃逸关系已量化 | `prompt_component_ablation30/analysis.json` |
| 失败案例分析 | 已完成 | 失败主要集中在模板回声、碎片化、信息空泛、关系/因果篡改、parser-shaped wording | `e3_main/quality_failure_analysis.md` |
| 复现性材料 | 已完成 | 脚本、随机种子、阈值、重复次数、输出文件都已落盘 | `scripts/*.py` + `experiments/*/*.json` |

## 论文里可以直接写的数值

### 主结果

- Strict Valid ASR: 30.0%（9/30）
- 95% CI: [14.7%, 49.4%]
- Strict candidate rate: 13.3%（16/120）
- 95% CI: [7.8%, 20.7%]
- 阈值平台：0.65–0.80

### 稳定性

- 全量 120 候选：0 个 mixed flip，120/120 stable
- 严格子集 16 候选：9/9 成功段落在 50 次重复中保持稳定

### 消融/对比

- 提示组件消融显示：不同 rewriting primitive 的逃逸率和质量存在明显权衡
- 候选预算消融显示：K 增大主要提升覆盖，但主结果在 K=4/8/16 上已经稳定

## 还建议补的工作

这些不是“核心结论缺失”，而是“投稿更稳”的增强项。

| 优先级 | 工作 | 为什么还值得做 | 推荐动作 |
|---|---|---|---|
| P0 | 统一主表和主图 | 让审稿人一眼看到核心结论 | 把现有 JSON 汇总成一张总表 |
| P0 | 失败案例表 | 论文里必须交代边界 | 选 6–10 个代表样本，按失败类型写 |
| P1 | 同数据外部 baseline | 防止被说“只是在自己设定里有效” | 若有可用 baseline，跑同样的检测管线 |
| P1 | 多评审人工复核 | 强化质量结论的可信度 | 对成功/失败样本各抽样复核 |
| P2 | 显著性/置信区间附录 | 提高统计表达完整性 | 把现有 CI、重复实验结果整理进附录 |
| P2 | 复现附录 | 提高可复查性 | 列出 seed、threshold、repeats、脚本入口 |

## 最小补强计划

如果目标是“尽快投稿”，我建议只补三件事：

1. 主表：把现有结果全部收成一张总表；
2. 失败案例：做一页 failure taxonomy；
3. 外部 baseline 或多评审复核：二选一，优先你们最容易拿到的数据那边。

如果目标是“尽量强”，再加：

- 一个同数据上的 external baseline；
- 一个 2 人以上的 blind re-check；
- 附录里补完整的 reproducibility checklist。

## 推荐执行顺序

1. 先冻结当前主线结果，不再改阈值和数据口径；
2. 生成论文主表与两张主图；
3. 整理失败案例与边界条件；
4. 补一个最便宜的增强项：外部 baseline 或多评审复核；
5. 进入写作。

## 需要保留的可复现信息

- 主阈值：2.33
- 反馈驱动阈值：0.65
- 阈值敏感性扫描范围：0.50–0.80
- 重复次数：50（稳定性），1000（候选预算抽样）
- 关键脚本：
  - `scripts/build_final_paper_bundle.py`
  - `scripts/analyze_threshold_sensitivity.py`
  - `scripts/analyze_feedback_driven_full_repeatability.py`
  - `scripts/analyze_feedback_driven_strict_repeatability.py`
  - `scripts/analyze_prompt_component_ablation.py`
  - `scripts/analyze_candidate_budget16_pilot.py`
  - `scripts/analyze_candidate_budget16_judgments.py`

