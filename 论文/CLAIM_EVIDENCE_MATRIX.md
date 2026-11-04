# 统一投稿主张—证据矩阵

更新：2026-10-25，统一投稿版附弹性检测与语义序列工程评估。执行依据为 `任务清单.md`。
对象含义见 `TERMINOLOGY.md`。本矩阵的 CL 编号与条件 C0--C6 分开。
旧矩阵原件及原始哈希保存在 `experiments/swan_reframing1/originals/`。

## 2026-11-02 第一篇贡献定位与证据对应

本节同步[当前贡献说明](01_句界审计与精确评估/第一篇_核心贡献与摘要.md)和[中文完整稿](01_句界审计与精确评估/第一篇_中文完整稿.md)。这是写作组织更新，不新增确认端点，也不更改以下 CL/CR/EL 或机器可读裁决。

| 核心贡献 | 证据对应 | 允许范围 |
|---|---|---|
| 按适用条件组织完整流程评估 | 原论文／代码核验、分段与评分合同、当前正文 §2.4 | 审计方法与条件登记；不声称首次发现句界风险或四项机制已被因果识别 |
| 跨机制的条件性诊断 | CL03 短文本配对；五方法指定配置及人类参照干预结果，见正文 §5 | 五方法属于新增诊断证据流，不填补 SWAN CL04–CL08；校准集误报、条件分母与消融缺口保留 |
| 声明划分集合的精确最坏分数 | CR01–CR02、CR04 的局部评分范围；CR07 原始检测及 CR08 停止 | 两类评分的算法核验与独立原始检测；不称全部五方法认证，不报告真实认证成功率 |

五方法复用 Qwen/C4 接入配置，不等于各原论文的完全复现。SwordStamp 本轮为整句版；自然长度版重新校准，不证明原阈值迁移；PMark 缺同模型无水印对照及完整拆合条件。正常长度、有效计分次数守恒的句界效应仍是未确认端点。

AliMark/SwordStamp 已研究结构扰动，PMark 的分布不失真也不是任意编辑可检出的定理。本文贡献定位为条件明确的审计、诊断与有限范围精确求值；资源停止及修复负结果作为边界，不改变既有投稿 Go/No-Go。

## 2026-10-25 cycle2 固定三对诊断

用户授权的execution5协调器修复已完成，18份原预测保持封存，新增3份同模型
辅助位置参考，18/18后置评估与独立复算通过。来源：
[最终报告](../第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/execution5/FINAL_REPORT.md)、
[结果](../第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/execution5/RESULT.json)与
[核验](../第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/execution5/RESULT_AUDIT.json)。
原句/复述符号BA：W54.17%/50%、E145.83%/33.33%、C均33.33%；
复述身份F1：W39.85%、E123.37%、C36.03%。
3对均为开发集，R*=W仅为开发选择；复述参考接受20/24，缺失保留分母。
G1e=19/24、整体No-Go不变，无独立读出或人工G1/G2/G3，
G4未执行。诊断不支持事件子图相对完整图增益，不升级历史SS或CL/CR。

## 2026-10-24 cycle2 启动与中断历史范围

cycle2 为用户另行授权的开发性符号/盲定位诊断，
见 [启动记录](../第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/REPORT.md) 和
[实时状态](../第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/execution4/STATUS.json)。
协议、24模板字典、候选/参数与模型资产已冻结；7项工程检查、
CUDA FP32模型加载通过，首批54请求完成。初次同篇复述重复检查永久停止；
execution2保留停止记录，沿用全部回执、不重生成；G1e=19/24未过20/24，
重放误差0、峰值显存9.62%。execution4只修复正常EXITED状态读取与预算JSON落盘，
保留各次停止记录，完整CPU流程夹具通过；18/18盲预测已封存并通过只读SHA核验。
本地协调器在等待阶段因远程检查退出码255永久停止，位置参考0份、
SELECTION/RESULT不存在，符号BA、定位/身份F1和表示增益均未计算。
正例原文/复述W原始双窗口可用52/55、51/55，E1为49/55、47/55，
C均55/55；这些只说明证据可用性，不是正确率或效果提升。
构造参考与模型辅助参考不替代人工真值，正式 G1/G2/G3 未评估，
G4不执行，不报告FPR/TPR。本启动记录不升级SS01–SS04或既有CL/CR。

## 2026-10-24 语义序列 cycle1

以下 SS 为本轮开发性工程证据，独立于既有 CL/CR 的确认端点。
来源：[冻结协议](../第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle1/PROTOCOL.md)、
[最终报告](../第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle1/FINAL_REPORT.md)和
[补充审计](../第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle1/SUPPLEMENTARY_AUDIT.json)。

| 工程观察 | 实际结果/状态 | 证据边界 |
|---|---|---|
| SS01 当前字典写入读回必要约束是否过门 | 16/24，低于冻结 20/24；cycle1 No-Go | 生成→parser 整体通道；不是符号 BA、人工语义实现率或检测 TPR |
| SS02 归一化后完整目标结构覆盖 | 追加修正 1/24，严格图条件下也 1/24；两套图匹配一致 | 初始 0/24 来自归一化入口错误；原文件保留；G1e 不使用归一化且仍16/24 |
| SS03 共同候选与全排列顺序评分实现 | 七个工程测试方法通过；2,520 等频候选、560 小编辑路径、无序同分与精确重放 | 仅工程夹具；未得到真实盲事件、符号身份或顺序增益 |
| SS04 AMR 事件结构/顺序优于 Embedding | NOT_EVALUATED；后续神经识别与 Embedding 未运行 | 无共同候选实际比较、双人人工 G1/G2/G3 或配对 G4；长度/风格混杂仍未消除 |

保存产物核验通过不等于独立数据验证；人工标注零项、正式 G1 仍未评估。
本轮失败在冻结字典与批次范围内成立，不外推为所有 AMR 语义序列方案不可能。
没有改动既有机器可读 CL/CR 状态，不替代 CL04--CL08、CL12 或 CR08。
path0、S8 与旧多尺度实验的最终状态没有在本轮重新核验。

## 2026-10-14 历史弹性检测工程证据

2026-10-14补充：下列EL为新工程证据，不替代既有CL/CR的唯一确认端点。
旧正式Null面向100–220词人类文本，不能直接为17–48词正例提供实际1%FPR。

| 工程观察 | 当前结果/状态 | 证据边界 |
|---|---|---|
| EL01 冻结Bank Lattice detector在原人类Null上的FPR | `swan_elastic_formal_null1`，6/1000；单侧95%上界1.1808% | observed 0.6%，没有严格证明FPR≤1% |
| EL02 既有短文本正例/句界攻击检出 | `swan_elastic_positive_attack1`，28/30、27/30，条件存活27/28 | 既有开发资产、短长度global fallback；不称为正式TPR@1%FPR或有效分割无关防御 |
| EL03 短长度校准与固定标点压力 | `swan_elastic_length_boundary_audited2`全量与两套审计完成，联合工程No-Go；新FPR10/1000，baseline26/30、既有攻击17/30、五变体存活0/26 | 旧calibration短FPR168/1000；新FPR单侧95%上界1.6903%，严格1%未过；24个选中beam ERROR；逗号/换行/空格各0/30；开发资产、相关变体且无独立质量资格，详见`CLOSEOUT.md` |
| EL04 v2修复与开发门 | score-blind P0 Go：句界恢复600/600、完整路径5150/5150、Span预算低于4x；P1在calibration 144/1000因一个恢复图结构非法而工程停止 | P0只证明边界覆盖与复杂度；partial无RESULT/AUDIT，不是P1 No-Go且没有新FPR/TPR |
| EL05 v3非法边策略与P1b | P1b全量与双端audit完成，No-Go；baseline20/30未达24/30，五变体全部条件存活18/20，1210/1210全分母，选中非法边0 | v3解决结构非法边导致整轮中断：7条无效候选边均在DP前排除；但baseline开发检出门失败。已观察开发资产，不报告独立FPR或正式TPR；停止当前资产调参与P2准备，详见`CLOSEOUT.md` |

当前允许定位：**句界作为水印检测接口的统一研究：SWAN 行为审计 +
SemStamp/SimMark exact resegmentation 框架与真实系统可行性边界。**
正常长度、盲质量合格、句数守恒和条件特异 FPR 的确认实验尚未完成；
final1 与 source-anchored 路线均已在 pilot 进入分支 C。既有短文本结果
不能替代这个主端点。

2026-10-06 投稿口径纠正：`certified_resegmentation_wp2_*` 与 SWAN 是
同一篇 TMLR 稿件中的两个证据流，不是两篇独立投稿；“投稿收口”是交付阶段，
不是第三条研究支线。v17 B0 的 detector-health Go 和 v18 score-blind
readiness Stop 仍不是 SWAN count-neutral 主端点；v18 没有产生 certificate
score 或鲁棒性统计。因此统一投稿不改变证据隔离，CR 系列主张不能升级、
降级或替代 CL04--CL08。

| 主张 | 唯一主端点 | 当前裁决 | 证据边界 |
|---|---|---|---|
| CL01 固定 K=4 暴露历史攻击面 | 历史250篇固定预算 Valid ASR，191/250 | 历史范围支持 | 使用当时保存的质量标签；不重新标为双人盲人工评审 |
| CL02 反馈补救扩大历史成功集合 | 同一250篇累计 Valid ASR，218/250 | 反馈威胁模型支持 | 不与固定预算合并为一个无反馈端点 |
| CL03 短文本检测决策依赖句界 | 严格标点 A-B-A 配对翻转，逃逸1/30→30/30→1/30 | 短文本支持 | 17--46词、五单元；未证明独立质量或条件特异FPR |
| CL04 效应超出检测单元数减少 | final1 H2：C0−C2配对TPR差，N逐篇相等 | **未确认** | 缺正常长度最终主端点；代数反事实及oracle批次不得替代 |
| CL05 真实候选匹配参与下降 | final1 M1：同一最终集的 T(R(s)) 最大bank匹配配对差 | **未确认** | 必须H2通过，真实表示、C3控制、质量与FPR门齐全 |
| CL06 parser输出结构变化与匹配变化相关 | final1 M2：预注册raw结构变化与真实匹配变化关联 | **未确认** | R(s)与T(R(s))分别报告；关联不识别单独因果贡献 |
| CL07 whole-graph归一化是主要因果路径 | 真实候选图上的预注册中介/对照识别端点 | **未确认；当前不计划作强主张** | 人工D不足以提供该端点 |
| CL08 精确跨度恢复能在目标FPR下恢复证据 | final1 H3：C4−C2配对TPR差 | **未确认** | 历史共用阈值下的可逆性归CL03；C4不是部署防御 |
| CL09 S2Match对人工图作用域敏感 | D相对组成候选图的文档级目标F1差，中位−0.3889 | **仅synthetic_scorer_diagnostic** | 108/108→0/108是目标green描述，不是真实攻击ASR |
| CL10 真实合句攻击不需要parser drift | 自然文本中合法、可识别的无drift对照端点 | **撤回真实文本主张** | D不是无drift的自然文本反事实；也不是gold AMR |
| CL11 SWAN特异性 | 匹配source、模型、长度、预算、质量与FPR的跨水印效应差 | **未确认；条件性支线** | KGW 0/30、SynthID 1/30仅为未匹配范围对照 |
| CL12 有效分割无关防御 | 新开发集过门后的锁定human test/positive holdout联合门 | **未确认；当前三条路线为负结果** | 不放宽同一开发集阈值或top-k |
| CL13 正常长度合格正例已存在 | 已核查资产中的联合资格数，0 | **在核查范围内不支持** | 不表示所有可能的公开资源都不存在 |
| CL14 旧长度门过滤了部分已观测达标候选 | 246个被拒候选中的4个S2Match≥.75 | 候选级直接过滤支持 | 两个槽前缀受影响；不能外推完整文档产率 |
| CL15 final1 当前生成路线未达到 pilot 可行性门 | 8个冻结source的结构、质量和gate全分母 | **支持；分支C停止** | 8/8终态、6/8结构通过、当前标签0/6；A/B标签为模型辅助来源，不是两名独立真人质量估计；未运行变换或检测 |
| CL16 source-anchored 当前生成路线未达到 pilot 结构门 | 8个冻结source的结构与gate全分母 | **支持；分支C停止** | 8/8终态、5/8结构通过、60个完成槽均为fallback；质量未执行而非失败；parser warning停止项未解决；未运行变换或检测 |

## Certified-resegmentation 主张

| 主张 | 唯一主端点 | 当前裁决 | 证据边界 |
|---|---|---|---|
| CR01 独立跨度 fixed-`N` 最坏分数可精确求解 | additive DP 对 brute-force oracle | **支持** | 5,691系统实例+2,000随机实例；仅覆盖声明的有限lattice |
| CR02 一阶相邻跨度 detector 可由 edge-state DP 精确求解 | pairwise DP 对 brute-force oracle | **支持** | 5,669系统实例+2,000随机实例，另有1,000退化对照 |
| CR03 post-certificate split-conformal 与 strict ties 实现正确 | 99,980合成null的80个独立test门 | **合成范围支持** | 不代表真实human-null exchangeability或真实FPR |
| CR04 SemStamp/SimMark 固定接口属于一阶pairwise类 | 官方源码审计与真实provider等价 | **支持** | SimMark早期等价用sentinel gamma；不等于detector健康 |
| CR05 编译factorized exact core可处理最大已验证shape | v4/v6双replay与oracle | **支持** | 最大已验证38,664,070 pairs；不自动覆盖v18的83,540,900 pairs |
| CR06 SimMark已获得可解释的真实certificate鲁棒率 | v9 detector-health前置门 | **不支持** | clean-positive仅89/1000，不能将后续rate归因于重分句 |
| CR07 最终SemStamp detector健康 | v17 B0 | **支持** | test FPR 47/1000，positive TPR 954/1000；不等于certificate |
| CR08 最终SemStamp exact certificate与certified rate | v18 B1 | **未执行/不支持** | P0因78,642,945 pruned pairs与86.6501h预算Stop，正式score调用为0 |

每条主张只有一个主端点，辅助数值只解释范围。CL03 的 A-B-A 是一个
预先定义的配对行为设计，不把多组探索结果拼接成替代端点。
机器可读记录为 `experiments/swan_reframing1/claims_v2.json`；
`scripts/lint_research_claims.py` 检查唯一端点、证据哈希、JSON路径和
新确认端点的状态。主端点缺失、哈希变化或确认门未满足时自动降级，
任何历史探索结果都不能触发升级。

保留的辅助事实：

- 249/250仅表示至少存在一个低分候选；不是有效攻击率。
- 独立三臂短文本无反馈29/30仅支持当时的短文本设置。
- 默认lambda=.25下，N=1的公式上限为sqrt(3)=1.732<2.33。
  官方直接AMR接口对N<2返回NaN，预解析入口对N=1仍算z；
  必须区分接口、公式与历史runner，不将NaN记为攻击成功。
- green-rate代数替换允许非整数虚拟G，只是数值充分性分析。
- 人工D中precision下降多于recall，是评分实现诊断；605个历史生成
  候选中的图规模/precision关系是另一组描述性关联，二者不合为因果证据。
- AliMark与SwordStamp已有merge/split、重分句和检测单元效应讨论。
  不声称首次提出攻击原语；SIRA始终写作SIRA-adapted。

旧 `swan_final_confirmation1` 为终止的不完整可行性实验，未完成
其Gate A，不能追认成功或正式失败。旧DeepSeek试点和15单元A/B
下游协议均不活动。`swan_count_neutral_final1` 已完成8-source pilot，
在当前冻结标签下因生成质量0/6进入分支C。`swan_source_anchored1`也已
完成8-source pilot，因结构仅5/8通过而在质量前进入分支C。两条路线的
main、变换和检测均未运行。这些结果不确认CL04--CL08，也不能外推为所有
DeepSeek或SWAN正例生成路线不可行。项目不再开启新的付费生成路线。
冻结证据稿和预投稿审计已完成；SWAN机器可读 `claims_v2.json` 无状态升级，
统一稿机器可读主张见
`experiments/swan_branch_c_final_literature_refresh_20261002/UNIFIED_CLAIMS.json`。
统一投稿范围见
`experiments/swan_branch_c_final_literature_refresh_20261002/UNIFIED_SUBMISSION_SCOPE.md`。
唯一一次投稿前文献刷新已使用，TMLR已选为同行评审目标，统一英文稿已
纳入CL与CR两组证据；CL03--CL08和CR06--CR08裁决均未因合稿而升级。当前
没有活动研究任务，外部提交只涉及作者确认与公开制品清理，不自动授权
任何新实验或第二次文献刷新。
