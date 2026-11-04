# SWAN 对象与证据术语（2026-09-18）

本规范服从 `任务清单.md`。适用于新协议、输出 schema、表格和论文。
历史字段保留原字节，由解释层映射；不得改写已冻结的原始结果。

| 符号 | 对象 | 新输出字段 | 可承担的证据 |
|---|---|---|---|
| `B` | 固定 AMR-Bank，有序秘密模板集合 | `bank_sha256`, `bank_template_order`, `bank_template_id` | 固定匹配目标；同次实验所有条件相同 |
| `R(s)` | 冻结 parser 对检测跨度 `s` 的原始 AMR | `raw_parsed_amr`, `raw_graph_sha256` | parser 输出描述 |
| `T(R(s))` | 规范化、概念抽象后的真实候选模板图 | `candidate_template_graph`, `candidate_graph_sha256` | 实际送入 S2Match 的候选表示 |
| `D(g1,g2)` | 用人工根及 `:snt1/:snt2` 拼接保存图的诊断对象 | `synthetic_composition_graph`, `evidence_class=synthetic_scorer_diagnostic` | 评分函数的实现诊断 |

`R(s)` 和 `T(R(s))` 均随检测跨度变化，均不是模板库。
`D(g1,g2)` 是人为构造，不是自然文本的 parser 输出，也不是 gold AMR。
若 D 的构造输入是 raw 图并再次模板化，必须记录
`composition_stage=raw_then_template` 及 `T(D(R(s1),R(s2)))`；
不得把它误标成 `T(R(合句文本))`。

新记录至少携带：

- 身份：`experiment_id`, `allocation_id`, `source_id`, `document_sha256`,
  `condition_id`, `unit_id`, `evidence_class`。
- 跨度：`unit_text`, `char_start`, `char_end`, `segmentation_spans`,
  `word_tokens_sha256`, `punctuation_edits`；字符 offset 按 Python Unicode
  code point 的左闭右开区间，另保存原始 UTF-8 文档哈希。
- 表示：上表字段及 `parser_revision`, `normalization_revision`,
  `segmenter_revision`, `s2match_revision`, `seed`, `cache_key`。
- 匹配：`bank_score_vector`, `max_bank_template_id`, `max_bank_score`,
  `target_template_id`, `target_score`, `s2match_precision`,
  `s2match_recall`, `s2match_f1`, `green`。
- 聚合：`num_detection_units`（N）, `green_count`（G）, `green_rate`,
  `lambda_null`, `z_score`, `threshold`, `detected`, `failure_reason`。
- 注入：`injection_accepted`, `injection_fallback`, `attempt_id`,
  `request_journal_id`, `usage`；接受与检测 green 使用不同阈值和字段。

`s2match_precision/recall` 是软图对齐量，调用方向固定为
`(candidate_template_graph, bank_template)`，不是语言学准确率或召回率。
图编辑距离也不是人类语义距离。每个图必须同时记录节点、边、三元组数
及解析状态；失败不能自动等同于 red 或低分。

“AMR 改变”必须注明上述对象或检测图分组改变。“语义保持”只能依据独立
盲质量评审的维度、分母和通过规则；词汇、大小写及词序完全不变只是
变换的不变量。无新词也可能改变逻辑、否定、指代或句法自然度。

检测链：文本 → 冻结分句跨度 → `R(s)` → `T(R(s))` → 与固定 `B`
逐模板匹配 → green → `(N,G)` → z → 条件阈值。

公式为 `z=(G-lambda*N)/sqrt(N*lambda*(1-lambda))`。
数学上 N≥1 时 `z_max=sqrt(N*(1-lambda)/lambda)`；
默认 lambda=.25、N=1 时 `z_max=sqrt(3)<2.33`。但官方
`detection_utils.detect_amr_sentences` 对 N<2 返回 NaN，
`detect_from_parsed_amrs.compute_zscore_for_doc` 对 N=1 仍计算该式。
因此必须记录检测入口；NaN 是无有效数值，不能暗中当作零分。

质量防火墙覆盖变换、选择、评审和纳入决策。官方逐句注入本身需要
parser/S2Match 接受采样，其内部日志隔离保存，不能给评审员或边界
选择器读取；这不等同于禁止注入算法使用自身的接受判据。
基线 TPR 门在基线盲质量冻结与独立 calibration 后判断，
且早于主条件的正例检测；任何条件结果均不得回流到构造或质量选择。

`C0`--`C6` 仅指新协议的条件；主张编号写作 `CLxx`，防止与旧矩阵
C1--C13 混淆。旧15单元 A/B 协议、16/16门和7--12词门均不活动。
