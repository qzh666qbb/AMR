# 未来实验执行进度

## 已完成：Milestone A 核心消融

复用冻结的 E2-250 候选池，对 `K=1/2/3/4` 和四种选择策略进行了1000次重复子采样。

核心结果：

- 随机选择的结果不随 K 改善，Valid ASR 约为43%；
- 最低 z 选择将 K=4 的纯检测逃逸率提高到99.6%，但质量通过率仅48.0%；
- 先质量过滤再选最低 z 时，Valid ASR 从 K=1 的43.1%依次提高到 K=2 的61.5%、K=3 的71.0%和 K=4 的76.4%；
- 多候选的主要收益是扩大“至少存在一个质量合格候选”的覆盖率，而不只是继续压低检测分；
- 全部1000候选的 AUROC 为0.728，而每段最低 z 选优后的250个结果 AUROC 为0.664；两种统计对象必须分开报告；
- 质量感知选优后的 AUROC 为0.686。

产物：

- `experiments/milestone_a/protocol.md`
- `experiments/milestone_a/results.json`
- `experiments/milestone_a/report.md`
- `experiments/milestone_a/candidate_budget_ablation.png`
- `scripts/analyze_milestone_a.py`

## 已完成：段落长度与复杂度分析

- 正式数据239/250段集中在4–6句，句数分层高度不均衡；
- token长度四分位的Valid ASR分别为70.4%、83.1%、75.0%和76.1%，没有单调长度效应；
- 探索性多变量回归没有显示句数、token数、实体代理数、数字数或原始z是强单变量决定因素；
- 由于样本构造基本固定为5句，跨长度结论需要新数据集或专门长度控制实验。

## 已完成：靶向句子攻击预算准备

根据SWAN的二值绿色句统计，计算每个段落至少需要翻转多少个超过0.65匹配阈值的句子才能使 `z≤2.33`：

- 27段无需进一步翻转；
- 27段最少翻转1句；
- 195段最少翻转2句；
- 1段最少翻转3句；
- 平均最少翻转1.68句。

每段已生成最高匹配边际、最接近阈值和随机绿色句三套等预算目标清单。

产物：

- `experiments/targeted_sentence_attack/targets.json`
- `experiments/targeted_sentence_attack/report.md`
- `experiments/targeted_sentence_attack/minimum_flip_distribution.png`
- `scripts/prepare_targeted_sentence_attack.py`

## 已完成：E3句界机制分析

- 普通改写改变句数的比例为21.5%，平均z降幅0.569；
- 句界攻击改变句数的比例为58.7%，平均z降幅1.144；
- AMR引导改变句数的比例为33.6%，平均z降幅1.064；
- 句界攻击减少句数时平均z降幅1.837，句数不变时仅0.258；
- 句界攻击增加句数的5个样本平均没有收益；
- AMR引导在句数不变时仍有0.799的平均z降幅，说明其包含句界之外的结构作用。

当前证据支持：句界攻击的主要收益来自合句，而不是任何句界变化都有效。

产物：

- `experiments/mechanism_sentence_boundary/results.json`
- `experiments/mechanism_sentence_boundary/report.md`
- `experiments/mechanism_sentence_boundary/family_mechanism_comparison.png`
- `scripts/analyze_sentence_boundary_mechanism.py`

## 已完成：E2 AMR机制分析

质量感知选优后的250段表现为：

- 平均句子/AMR数从5.004降到1.740；
- 平均最大模板匹配从0.807降到0.478；
- 绿色句比例从91.5%降到6.4%；
- 绿色句比例变化与z降幅相关系数为-0.984；
- 选中候选的AMR图解码率为100%，不支持“主要由解析失败造成”的解释；
- 质量合格但未逃逸的11段平均只减少0.091句，绿色句率只下降0.055；
- 质量合格且逃逸的191段平均减少3.398句，绿色句率下降0.877。

产物：

- `experiments/mechanism_e2_amr/results.json`
- `experiments/mechanism_e2_amr/report.md`
- `experiments/mechanism_e2_amr/amr_change_vs_z_drop.png`
- `scripts/analyze_e2_amr_mechanism.py`

## 已完成：轻量聚合防御原型

离线比较原始二值z、连续匹配均值、中位数、最大值、Top-2均值和soft-green聚合。

- 原始二值z：基线AUROC 0.985，攻击后0.686；
- 连续均值：基线0.996，攻击后0.516；
- 中位数：基线0.992，攻击后0.548；
- 最大值：基线0.976，攻击后0.258；
- Top-2均值：基线0.989，攻击后0.292；
- soft-green：基线0.993，攻击后0.372。

简单替换聚合函数没有恢复检测能力，部分方法甚至产生方向反转。后续防御应转向多粒度切分、跨句模板或多parser集成。

产物：

- `experiments/defense_aggregation/results.json`
- `experiments/defense_aggregation/report.md`
- `experiments/defense_aggregation/aggregation_auroc.png`
- `scripts/evaluate_aggregation_defenses.py`

## 靶向实验生成依赖（已解除）

下一实验是三组等预算靶向改写：

1. 攻击最高匹配边际句；
2. 攻击最接近0.65阈值的绿色句；
3. 攻击随机绿色句。

所有结果需要重新嵌入完整段落、重新解析AMR并运行SWAN。实验期间已配置DeepSeek API并完成生成；密钥仅保存在被Git忽略的本地 `.env` 中，不纳入任何实验产物或版本控制。

## 已完成：严格控制的靶向句子攻击先导实验

在29个完整配对段落上比较三种策略。每个策略每段一次独立DeepSeek调用，只重写预先指定的1–2个句子，非目标句逐字不变，句数完全不变。所有候选重新经过BART AMR解析和SWAN检测。

| 策略 | 平均z降幅 | 逃逸率 | 目标句翻转率 | AUROC |
|---|---:|---:|---:|---:|
| 最高匹配边际句 | 0.321 | 3.4% | 15.5% | 0.999 |
| 近0.65阈值句 | 0.249 | 0.0% | 12.1% | 1.000 |
| 随机绿色句 | 0.392 | 3.4% | 19.0% | 0.999 |

三组配对Wilcoxon检验均不显著。数字和否定表面保持率均为100%。结果表明：仅定位高匹配句并做局部结构改写不足以击穿SWAN；E2的强攻击效果主要依赖整段联合重组，尤其是合句造成的解析、模板匹配和聚合共同变化。

产物：

- `experiments/targeted_sentence_controlled30/candidates.json`
- `experiments/targeted_sentence_controlled30/dataset/parsed.json`
- `experiments/targeted_sentence_controlled30/detection/`
- `experiments/targeted_sentence_controlled30/analysis.json`
- `experiments/targeted_sentence_controlled30/report.md`
- `scripts/generate_targeted_sentence_controlled.py`
- `scripts/prepare_targeted_sentence_controlled.py`
- `scripts/analyze_targeted_sentence_controlled.py`

## 已完成：提示组件消融先导实验

在同一30段上，四种提示条件均使用DeepSeek、温度0.7、每段每条件一次独立调用，并统一重新进行BART AMR解析和SWAN检测。

| 条件 | 平均z降幅 | 逃逸率 | AUROC | 减少句数比例 |
|---|---:|---:|---:|---:|
| 弱同义改写 | 0.645 | 33.3% | 0.934 | 26.7% |
| 无句界谓词/论元改写 | 0.903 | 40.0% | 0.906 | 23.3% |
| 仅句界重组 | 3.005 | 93.3% | 0.828 | 100.0% |
| 完整E2联合改写 | 1.963 | 73.3% | 0.903 | 70.0% |

按实际输出分层，减少句数的66条候选逃逸率为90.9%，句数不变的54条只有22.2%。仅句界重组显著优于弱改写（配对Wilcoxon `p=1.39e-5`）、无句界改写（`p=3.79e-6`）和完整E2（`p=0.0165`）。

该结果说明：在当前RealNews五句段落和等调用预算下，合句/句界重组是最主要的攻击组件；增加谓词、论元和共指等完整E2指令没有进一步提高检测规避，反而可能增加生成约束与语义风险。

产物：

- `experiments/prompt_component_ablation30/candidates.json`
- `experiments/prompt_component_ablation30/dataset/parsed.json`
- `experiments/prompt_component_ablation30/detection/`
- `experiments/prompt_component_ablation30/analysis.json`
- `experiments/prompt_component_ablation30/report.md`
- `experiments/prompt_component_ablation30/human_review_blind.csv`
- `scripts/generate_prompt_component_ablation.py`
- `scripts/prepare_prompt_component_ablation.py`
- `scripts/analyze_prompt_component_ablation.py`
# 未来实验执行进展

## 候选预算 K=1/2/4/8/16 先导实验（30段，已完成自动评估）

- 每段通过一次 DeepSeek API 调用生成16个结构改写候选，共480个候选；全部完成AMR解析与SWAN检测。
- 对16候选随机无放回抽取K个，重复1000次：最低z选择的逃逸率从K=1的96.3%上升到K=4的99.8%，K=8后达到100%。
- 最低z均值随预算从0.043下降到-0.722，AUROC从0.774下降到0.675，说明候选搜索仍能持续压低检测分数，但逃逸率在K=4附近已基本饱和。
- 仅用数字与否定词做自动质量约束时，质量优先选择的自动Valid ASR由K=1的75.7%提高到K=4的87.6%、K=8的89.2%、K=16的90.0%，呈明显边际递减。
- 480个候选中79.4%同时通过数字与否定词约束，AMR非空率100%；最终Valid ASR仍需完成150条盲法人工语义评审。
- 多进程S2Match重算的人类基准存在1–2个样本的离散波动，AUROC固定采用此前`formal_250x5`正式人类基准，未根据本轮结果挑选。
