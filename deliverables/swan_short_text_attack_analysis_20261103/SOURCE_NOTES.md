# 来源核实与版本说明

核实日期：2026-11-03。本文记录支持本报告的最少必要摘录，不复制实验原文或完整第三方论文。

## 论文中的生成长度

来源：[SWAN, arXiv:2605.04305v1, §4.1](https://arxiv.org/html/2605.04305#S4.SS1)。

论文在 Evaluation metrics / Watermark detectability 中明确写道：

> “generate 5 new sentences given one sentence as initial context”

来源：[同文 §4.6](https://arxiv.org/html/2605.04305#S4.SS6)，Figure 3 图注：

> “1,250 generated sentences (250 samples × 5 sentences)”

因此，“每个来源生成五句”同时得到论文与代码支持。这不提供生成段落的具体词数分布，不能据此说原论文的段落也是 17–46 词。

本地确认集按空白分隔的计数为：30 篇、150 句、927 词；每篇 17–46 词、平均 30.9 词、中位数 30 词，平均每句 6.18 词。保存的本地注入参数同样为每句最多 50 tokens。句数与上限参数对齐不证明输出的实际词数分布对齐，这些指标已分别记录在 `metrics.json`。

论文 §4.1 的 Dataset 段落称取前 250 个句子；本地官方 README 下载说明称前 250 个样本。本报告不将两种抽样描述视为完全等价，也不依靠这个差异作攻击推断。

核实方法：读取 arXiv HTML 的 `S4.SS1` 与 `S4.SS6` 节。仅依靠网页读取工具的开头截断输出无法核实这两节，因此进一步从完整 HTML 提取目标章节。本报告使用原始论文，不以第三方论文摘要站作为核心证据。

## 三个来源的生成模型不一致

| 来源 | 注入/生成模型说明 | 使用方式 |
|---|---|---|
| 论文 HTML v1 §4.1 | `DeepSeek-R1-Distill-Qwen-14B`，temperature 0.6、top_p 0.9 | 原论文当前可核实声明 |
| 本地官方 README，快照目录 `SWAN_official_2ca0c207_clean` | 声称论文注入用 Claude 3.5 Sonnet v2，复述/质量用 Claude 3.7 Sonnet；仓库默认已更新为 Claude Sonnet 4.5 | 与论文存在冲突，不自行裁定哪项代表最终论文实际运行 |
| `confirmation_three_arm30` 保存的 runtime | 请求 `deepseek-v4-flash`；返回 `deepseek-flash`；注入 temperature 0.7、top_p 0.9 | 本报告 30 篇来源的实际记录 |

官方 README 来源：[README.md](../../third_party/SWAN_official_2ca0c207_clean/README.md)，模型说明约第 178–185 行；五句运行示例约第 215–227 行。文件字节指纹已纳入交付的 `INPUT_SHA256.json`。

本报告因此使用“基于 SWAN 官方组件的本地实验”这一表述，而不声称原论文同模型复现。具体输入 SHA 不能证明模型提供方实际返回了什么权重；模型标识按保存响应报告。

## 代码中的长度与计分规则

- [watermark_generation.py](../../third_party/SWAN_official_2ca0c207_clean/injection/watermark_generation.py)：`num_sentences` 默认 5，`max_new_tokens` 默认 50；后者作为逐句生成参数传递。约第 135 行只拼接 `current_sentences` 保存生成段落，上下文提示不并入保存文本。
- [detect_from_parsed_amrs.py](../../third_party/SWAN_official_2ca0c207_clean/detection/detect_from_parsed_amrs.py)：`compute_zscore_for_doc` 以全部保存的 AMR 单元计数；单元数为 1 且分母非零时，仍产生有限 $z$ 分数。
- [detection_utils.py](../../third_party/SWAN_official_2ca0c207_clean/detection/detection_utils.py)：`detect_amr_sentences` 在 `n_sents < 2` 时直接返回 NaN。
- [run_segmentation_intervention.py](../../scripts/run_segmentation_intervention.py)：机制实验实际使用的本地包装器调用 SWAN AMR/匹配组件，检查有限分数，计算 $z$ 并保存每单元 50 个模板分数。

所以，报告验证的是该本地流程的有效分数逃逸。官方预解析入口的公式一致是静态核实结果，不应表述为本次重新运行了官方 CLI。其他入口的 NaN 也没有被计作阴性结果。

## 本次复核覆盖范围

本报告核心三份历史审计清单共有 7,220 条 SHA 记录，去重后 3,709 个文件，全部匹配。该数字表示保存文件的校验记录，不表示 7,220 个独立样本或实验。

除 SHA 校验外，重算并检查了：

- 30 篇来源、五句注入记录与攻击来源的对应关系。
- 标点替换规则、词汇序列、各检测单元与原始 manifest 的对应关系。
- 模板最高分、0.65 绿色判定、文档单元数与 $z$ 公式。
- 三组攻击候选的冻结质量检查点，以及有反馈/无反馈选择规则。
- 完整队列、初始检出条件分母、全接受子集和文档级置信区间。

没有重新评审语义质量，没有进行新解析或 S2Match 回放，没有校准新阈值，没有补样。公开分享前应保留这种范围表述，勿把仅有的模型评审标成独立真人证据。
