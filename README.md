# AMR / SWAN 研究工作区

更新日期：2026-11-04

**当前执行：** [句向量辅助 AMR 检测](deliverables/embedding_amr_20261104/实现与运行说明.md)
已冻结 6,090 行短文本基准复评输入，执行三臂、三条件的新流程校准与测试；
语义和质量评审使用 API 模型或助手，并明确标注来源，不再设置人员评审前置条件。
旧冻结协议及 Stop/No-Go 记录保持历史含义，不追溯改写。
实际进度与终态报告入口：[当前状态](deliverables/embedding_amr_20261104/当前状态.md)。

**2026-11-02 的并行工程记录：** 用户已授权补全缺口审计中的全部W/X/S项目。
多尺度迁移已完成24对×6条件并通过最终双审计；简单均值融合有排序信号，
复杂融合未稳定胜过均值。正常长度8篇pilot已完整结束，0/8合格文档；
path0两臂2500行已完成，但生成自动合格0/50，正例权衡仍未验证。
原字典复述320句读回已完成，符号BA32.29%（模型辅助映射）、完整序列唯一恢复0/12，
未支持可靠读回。当前监督原图补跑运行中；见[最新结果](experiments/swan_research_completion_20261102/result_update2/UPDATE.md)。
迁移及pilot结果见[结果更新](experiments/swan_research_completion_20261102/result_update1/UPDATE.md)，
最新path0阶段见[执行账本](experiments/swan_research_completion_20261102/PLAN.md)。
提示贡献对照432/432次生成与账单已验证，读回仍待GPU。
正常长度主实验、身份对齐、path0正例、独立确认、秘密Bank、二元反馈、
多Bank及多解析器后继已实现并排队。完整机器结果、质量与事实身份核验
不能用“部署成功”替代。参见[全项执行账本](experiments/swan_research_completion_20261102/PLAN.md)。
SemStamp公共缺件已恢复6项同SHA资产，仍需三份密封原输入才能继续。

论文与实验写作现按两条线管理：P1 句界审计与精确评估、P2 语义消息读回。
P1 写作文档集中在 [论文/](论文/README.md)；P2 已独立为顶层目录
[第二篇_语义消息写入与读回](第二篇_语义消息写入与读回/README.md)，文档、实验、支持材料、代码快照、评审原件、训练检查点与服务器远端证据都在其中。
本轮整理结果与SHA见 [独立整理校验](第二篇_语义消息写入与读回/06_迁移记录/整理校验.json)，服务器保留项见 [远端清单](第二篇_语义消息写入与读回/04_服务器远端资产/SERVER_ASSETS.md)。
原 P3 认证失败工作并入 P1 §6，原 P4 融合/恢复实验并入 P1 §7，不再作为独立论文线。
全部实验归属见[总表](论文/全部实验_文章归属与转换总表.md)。
P1已有英文稿；P2证据受限；两条线不表示两篇投稿就绪。
旧版（四条线）已归档至 [20261027_paper1_merge](archive/20261027_paper1_merge/ARCHIVE_MANIFEST.md)。
原统一稿中的SWAN与fixed-N认证合稿不变，后续variable-N失败报告作为P1 §6.2记账。

事实载体分层已实际生成192/192原AMR，172结构合格、20排除；
随后v2读图器完成192原图和171助手修订子集的开发读出，不能再记图分类0。
语义确认按新协议采用模型或助手，修订图归因需单独核验。原图来源见
[真实原图执行报告](第二篇_语义消息写入与读回/02_实验/semantic_fact_layers_v3_parse1/REPORT.md)。
下方按日期保存历史进度，不恢复原停止版本。

找实验目录时，先看 [研究工作导航](experiments/研究工作导航.md)：按研究问题区分窗口检测、多尺度融合、新语料迁移、语义序列和精确重分句。执行计划仍以 [任务清单](任务清单.md) 为准。

> **cycle2当前结果（2026-10-25）**：
> 用户明确授权启动 [`semantic_sequence_cycle2`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/REPORT.md)，
> 重点验证符号识别与盲事件定位。协议、提示、代码及资产 SHA 已冻结；
> 7 项工程检查和 CUDA FP32 模型加载通过，首批54次生成全部完成。
> 初始执行因同篇负例复述等于原文触发永久停止；execution2保留原停止记录，
> 沿用全部回执、不重生成，CUDA pilot结果19/24，低于20/24，整体No-Go。
> execution4修复Supervisor状态读取和预算JSON落盘错误，保留各次停止记录；
> execution5修复远程查询断连容错，沿用18份原盲预测，新增3份模型辅助参考，
> 已完成18/18后置评估并独立复算；累计57请求，无重复生成。
> 原句/复述BA：W54.17%/50.00%、E145.83%/33.33%、C均33.33%；
> W身份F1为42.30%/39.85%。仅3对开发诊断，整体No-Go，不扩展21对。
> W/E1/C 使用共同文本候选，预测先封存再加入参考标签。
> 正式人工 G1/G2/G3 未评估，G4 未运行；本轮不产生 detector FPR/TPR。
> 最终结果见 [`FINAL_REPORT.md`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/execution5/FINAL_REPORT.md)，
> 规则见 [`PROTOCOL.md`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/execution5/PROTOCOL.md)。

> **cycle1 历史收口（2026-10-24）**：
> [`semantic_sequence_cycle1`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle1/FINAL_REPORT.md)
> 已完成写入读回 pilot 的审计与封存。24 个固定载体必要约束成功 16，
> 未达到冻结 20/24 门，当前版本 **No-Go**。完整目标图覆盖经归一化入口修正
> 为 1/24，原结果与修正来源分别保留；不影响关口判定。
> 全部 2,520 等频排列、精确 DP 和共同候选机制通过工程检查，
> 但原句/复述符号识别、盲定位及 AMR/Embedding 顺序增益均未实测。
> 独立双人人工 G1 未评估；四阶段整体未完成。cycle1 相关进程已退出，
> 收口时未补样、未启动后继；其后用户另行授权 cycle2，
> 不据 cycle1 裁决改写 path0、S8 或 CL/CR 历史结论。
> 证据与复现见
> [`报告`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle1/FINAL_REPORT.md)、
> [`收口记录`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle1/CLOSEOUT.json)、
> [`未解决事项`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle1/FOLLOWUP.md)。

> **历史任务记录（2026-10-14）**：用户曾要求规划并执行SWAN弹性检测后续验证。
> [`swan_elastic_length_boundary_audited2`](experiments/swan_elastic_length_boundary_audited2/CLOSEOUT.md)
> 已完成全部7214行与两套审计，GPU已释放，最终联合工程No-Go。
> 补短文本桶后独立FPR为10/1000（单侧95%上界1.6903%），开发正例26/30、
> 既有句界改写17/30、五变体最坏条件存活0/26；24个选中beam有ERROR。
> [`系统总体规划`](SWAN_弹性重构防御_总体规划.md)与
> [`部署运行手册`](SWAN_弹性重构防御_部署运行手册.md)是本轮专项入口。
> 旧投稿与失败实验状态以下保留为历史；新工程评估不自动升级正式鲁棒性、
> 正常长度质量cohort或certificate主张。
> 初版在192条calibration后因上游图恢复审计缺口停止；当前版本补逐Span
> 日志绑定，真模型108个Span输出及固定16篇评分等价核对通过。
> 旧calibration在本轮短Null上的FPR为168/1000=16.8%，此前短文本高检出率
> 不能解释为实际1%FPR下的鲁棒性。完整结论与后续顺序见
> [`CLOSEOUT.md`](experiments/swan_elastic_length_boundary_audited2/CLOSEOUT.md)。
> v2 P1在calibration 144/1000处因一个3词候选Span的恢复图结构非法而
> fail-closed停止；该partial无RESULT/AUDIT，不是P1 No-Go。现已另立v3：
> 结构或round-trip非法边不进入DP，断路样本abstain并保留全分母。v3
> score-blind P0已Go；P1b随后从calibration第0行完成1000+210全量运行。
> baseline仅20/30，低于冻结24/30门，最终No-Go；五变体全部条件存活18/20、
> 1210/1210全分母和选中非法边0通过。7条无效候选边均在DP前排除，远端与
> 本地零模型audit均pass。按止损规则不继续在当前30篇上调参，也不启动P2。
> 完整结论见
> [`CLOSEOUT.md`](experiments/swan_elastic_v3_development1/CLOSEOUT.md)。

## 执行与证据入口

2026-11-04 新增：[句向量辅助 AMR 检测实现与运行说明](deliverables/embedding_amr_20261104/实现与运行说明.md)。
入口为 `scripts/run_embedding_amr_detection.py`，支持对应融合、AMR 单路及关闭对应核验的消融，
以及独立校准/测试。现默认接入官方归一化/S2Match，保留显式结构特征评分选项；
已验证300个官方逐模板分数精确一致；本轮34项融合检查、8项解析审计与7项Bank回归通过。
当前复评使用完整历史30篇写入尝试、各1,000篇独立来源校准和测试控制组，
以实际写入Bank50重新校准；另报告21篇全部写入成功的子组。
90条API模型评审不读取Bank或分数、不筛选检测分母。采用新文档统计量，
不沿用旧阈值；最终效果只以完整终态与审计为准，公开Bank和短文本复用限制结论范围。

按以下顺序阅读和执行：

1. [`任务清单.md`](任务清单.md)：当前唯一执行计划、停止规则和完成标准；
2. [`UNIFIED_SUBMISSION_SCOPE.md`](experiments/swan_branch_c_final_literature_refresh_20261002/UNIFIED_SUBMISSION_SCOPE.md)：
   统一投稿问题、两条证据流和主张边界；
3. [`SUBMISSION_DECISION.md`](experiments/swan_branch_c_submission_readiness_20261002/SUBMISSION_DECISION.md)：
   统一投稿价值、允许渠道和后续门禁；
4. [`VENUE_DECISION.md`](experiments/swan_branch_c_final_literature_refresh_20261002/VENUE_DECISION.md)：
   最终文献刷新、TMLR选择与外部提交边界；
5. [`TMLR main.pdf`](experiments/swan_branch_c_final_literature_refresh_20261002/tmlr_submission/main.pdf)：
   纳入全部研究内容的匿名英文统一稿；
6. [`叙事.md`](叙事.md)：从 Plan A 到当前纠偏的研究演进；
7. [`RESEARCH_STATE.md`](RESEARCH_STATE.md)：现有运行状态与开放问题；历史协议均受顶部纠偏说明约束；
8. [`CLAIM_EVIDENCE_MATRIX.md`](论文/CLAIM_EVIDENCE_MATRIX.md)：v2主张裁决；CL编号与实验条件C0--C6分开；
9. [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md)：历史事实与实验产物索引，不是当前任务授权文件；
10. [`BRANCH_C_SYSTEM_AUDIT.md`](experiments/swan_reframing1/BRANCH_C_SYSTEM_AUDIT.md)：
   SWAN 证据流的冻结中文审计稿；
11. [`BRANCH_C_CLOSURE.md`](experiments/swan_reframing1/BRANCH_C_CLOSURE.md)：
   停止判定、未执行端点及未来重启条件；
12. [`CHECKLIST_STATUS_20260920.md`](experiments/swan_reframing1/CHECKLIST_STATUS_20260920.md)：
   236项checkbox的证据对账与未完成项分类。

当前 cycle2 的阶段状态以
[`STATUS.json`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/execution5/STATUS.json) 为准。
已封存 cycle1 的结果以
[`FINAL_REPORT.md`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle1/FINAL_REPORT.md) 和
[`STATUS.json`](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle1/STATUS.json) 为准。
以下投稿与旧工程段落保留各自日期的证据范围，不作为恢复旧实验的执行授权。

## 当前研究定位

> 句界是句级语义水印检测接口的一部分：以 SWAN 行为审计识别问题，
> 以 SemStamp/SimMark exact resegmentation 框架界定可证明保证及其
> 真实系统可行性边界。

当前不允许把项目描述为 AMR 语义层面的攻击。AMR-Bank 始终固定；
必须区分固定模板库、真实 parser 产生的候选模板图和人为组合诊断图。
SWAN 与 certified-resegmentation 是同一篇稿件中的两个证据流，不是两篇
独立投稿；两者的系统特异主张仍不得互相补证。

## 当前运行状态

- 2026-10-25：cycle2 execution5协调器修复与固定三对诊断完成，
  18/18后置评估、54组逐篇指标、12汇总和144已知位置预测核验通过。
  原句/复述BA：W54.17%/50%、E145.83%/33.33%、C33.33%/33.33%。
  总57请求，未重新生成；无独立读出或人工真值，整体19/24 No-Go不变。
  见[最终报告](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/execution5/FINAL_REPORT.md)。
- 2026-10-24历史：cycle2 独立冻结的识别/定位诊断流水线已核验；
  首批54请求已完成，execution2在保留初次工程停止记录后沿用全部文本，
  不重生成；写入门19/24、No-Go，重放误差0、峰值显存9.62%。
  execution4仅修复执行工程问题，完整CPU流程夹具通过，
  18/18盲预测已封存，协调器在等待期间因远程检查失败永久停止，
  参考与最终指标未计算，当前没有活动的后置评估。
  [启动核验](第二篇_语义消息写入与读回/02_实验/semantic_sequence_cycle2/execution4/START_VERIFICATION.json)
  只证明已启动和加载，不表示符号/定位效果已通过；
- 2026-10-24：语义序列 cycle1 在写入读回关口 16/24 后 No-Go 收口。
  54 个生成请求、24 个原句解析与三次额外重放完成；后续神经符号读出、
  盲定位和 Embedding 比较未运行。两个保存产物核验实现与七项工程回归通过，
  只证明完整性和实现逻辑，不证明实际顺序增益或低 FPR；
- SWAN弹性检测v1短文本独立校准与标点压力评估已完成并审计，联合工程
  No-Go；v2 P1在144/1000处工程停止。v3非法边策略工程门通过，但P1b因
  baseline 20/30未达24/30而No-Go；弹性重构当前路线已按预注册规则停止，
  无新增付费API或文本生成；
- `certified_resegmentation_wp2_*` 在因果证据上独立于 SWAN CL04--CL08，
  但纳入同一篇统一投稿；v18 已在 score-blind P0 Stop，未产生最终
  certificate 或鲁棒性统计，当前不授权 v19、exact 或 cheap-bound 后继；
- 统一投稿准备已完成口径重构：主会立即投稿No-Go；TMLR为唯一同行评审
  目标；
- 一次性文献刷新已使用；英文TMLR稿已纳入SWAN、exact DP、官方接口、
  远端工程、v17 B0 Go和v18 Stop；
- 2026-10-06的投稿收口状态保留；外部提交待作者、机构、冲突、伦理和公开
  制品许可确认。2026-10-14新增任务仅在冻结长度/压力评估范围内执行；
- 正常长度、质量合格、检测单元数 `N` 不变的确认实验尚未完成；
- P0已保存39,457文件哈希与82份原始文档；P1/P2已建立术语、纠偏和单主端点检查；
- `swan_count_neutral_final1` 已完成并封存八 source pilot；当前冻结标签下
  baseline质量为0/6，pilot gate失败并进入分支C；
- final1 未运行变换、神经parser、S2Match、检测或main，不得在相同source
  上调参重启；该实验的main已永久阻断；
- `swan_source_anchored1`已完成独立八source Pilot：5/8结构通过，低于
  6/8门，质量前进入分支C；其main、变换和检测均未运行；
- source-anchored只读closure核验3,925个文件并通过第二次重放，但
  parser warning停止项仍未解决；
- 独立的定向工程链已得到一个53词、10句、10/10严格注入且10/10 green的
  探索性连贯基线；两次同后端事件日志复核通过、第三次未通过，故只记为
  2/3模型辅助质量门，不记为真人盲评。该结果使用单一有利模板47，不满足
  100--220词、来源锚定或独立FPR要求，也不解锁句界攻击；
- 后续正常长度扩展得到126词、20句、20/20严格注入且20/20 green的探索性
  基线，z=7.7460；两次检测盲模型辅助质量复核均通过。它仍全部使用模板47，
  且缺少source anchoring、两名独立真人评审和经验FPR，因此正式Gate与句界
  攻击继续阻断；
- 收口后一次只读对账命令误读sealed main正文，该分配已按D007永久标记
  为已观察/污染；事件不改变pilot结果，但禁止任何未来复用；
- 旧脚本、旧协议或单个实验目录不得自行恢复为活动任务。

final1 结论见
[`PILOT_CONCLUSION.md`](experiments/swan_count_neutral_final1/PILOT_CONCLUSION.md)。
当前A/B标签来自语言模型辅助评审，不得称为两名独立真人评审，也不得用于
解锁最终检测。
source-anchored结论见
[`PILOT_CONCLUSION.md`](experiments/swan_source_anchored1/PILOT_CONCLUSION.md)。

## 目录规则

- `experiments/`：原始实验、协议、检查点和审计产物；状态见 [`experiments/README.md`](experiments/README.md)；
- `archive/`：已归档的旧叙事、旧规划、旧论文和历史脚本；归档内容不是当前权威入口；
- `scripts/`：历史与当前脚本混合目录；只有被当前预注册 manifest 固定的脚本才允许执行；
- `third_party/`：第三方实现和冻结 checkout；
- `artifacts/`、`models/`、`vectors/`：大体积本地资产，不作为研究结论入口；
- `.secrets/`：本地凭据隔离区，已被 Git 忽略，旧脚本不得默认读取。

## 防误用规则

- 不从 `archive/` 恢复旧主线，除非 `任务清单.md` 明确要求；
- 不把实验目录中的 `report.md` 自动视为当前结论；
- 不运行默认读取根目录 `api.txt` 或 `1.txt` 的旧脚本；这些默认路径已被有意移除；
- 不移动、删除或覆盖原始实验证据；归档只整理入口文件；
- 任何新结论必须进入 `CLAIM_EVIDENCE_MATRIX.md` 并通过独立审计。

术语见[`TERMINOLOGY.md`](论文/TERMINOLOGY.md)，纠偏记录与公式/功效计算见
[`swan_reframing1`](experiments/swan_reframing1/decision_log.md)。
这些离线准备不构成新正例、TPR、FPR或ASR结果。
