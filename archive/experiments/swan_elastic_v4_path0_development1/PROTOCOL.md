# SWAN v4 空路径候选：新版开发比较协议

版本：path0-development-v1，2026-10-19。用户已授权在服务器继续。旧实验、51/5000 原计数及历史裁决保持封存。

本协议承接[机制诊断](../swan_elastic_v4_mechanism1/REPORT.md)，替代旧合同作为本轮执行依据。它是看过历史结果后制定的新开发协议，不是事前正式性能证明。

两臂分别为原 v4 和 path0。唯一分数变化是路径分量两侧为空时从 1 改为 0；概念/关系分量、Bank100、Parser、格图、完整 count-state DP、非法边剔除不变。权重 0.25/0.50/0.25，score_center=0.65，green_threshold=0.75，p2=p3=0.18；双方在同一新 calibration 上独立拟合，主 alpha=0.01。预声明 alpha 曲线为 0.001/0.002/0.005/0.01/0.02/0.05，仅用于开发权衡，不能据此追改主端点。

## 冻结规模与来源

新 calibration 1000 篇，独立 Null validation 200 篇，正例 50 个原始生成槽位。17–24、25–32、33–40、41–48 词四层分别为 calibration 250/250/250/250、Null 50/50/50/50、正例目标 13/13/12/12。

固定 C4 realnewslike revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`，依次扫描 train shard 20–23，每 shard 至多 50000 行；网络每 shard 至多 3 次尝试，不因检测分数补样。每 source 只分配一个窗口。按原始行序选取第一个由完整句子组成、落在目标词层的窗口；窗口必须包含 ASCII 字母，不能包含 URL。原文、URL、locator、allocation ID、规范化窗口及已见生成文本均与本地和远端历史快照查重。快照存哈希及文件清单，不公开正文；所有新集合之间的身份交集必须为零。

Null 与正例都报告 baseline、semicolon_all、semicolon_pairs、comma_all、newline_all、space_all。变体按原文成组；变体数量不充作独立原文数。词序保持校验实际执行。正例输入缺失或资格失败不能当作零篇已完成任务；完整 50 槽位仍是预声明分母。

## 正例与生成预算

先冻结 50 个 source/prompt 槽位，再冻结可执行生成配置与预算文件，最后才能生成。生成使用已验证的 SWAN AMR 目标生成流程；不可用普通续写冒充水印正例，也不可用历史正例冒充新来源。每个槽位必须保存所有尝试、资格和失败原因，禁止读取两臂检测结果后重试、替换或补样。模型、目标模板分配、句子数、长度指令、每句最大尝试数、调用与费用上限、盲审规则必须在生成前物化；若 provider 或依赖缺失，明确标注 pending_generation，先完成独立 Null 开发比较。不能把缺少正例的 Null 结果裁为整体 Go。

至少 40/50 资格合格是完整开发比较的样本充分性条件。未达条件记 insufficient_positives，保留原分母及结果；生成方案另立版本。

## 端点与判断

开发 FPR 目标放宽为 observed ≤1%：干净及每个压力条件各 ≤2/200。原文级任一变体误报单列，作为更强的联合指标，不要求它也 ≤1%。

baseline≥80% 是目标，低于目标不机械终止候选。主鲁棒端点为 baseline 已检出原文中，五个条件各自的存活率及其中最小值；80% 为目标。五条件全部存活为次指标，同时报告全 50 槽位的各条件 TPR。不得将“最差条件比例”写成“每篇所有条件均存活”。允许用明确报告的 TPR/鲁棒性损失换取 FPR 改善，取消所有指标不得下降的门。最终分类为目标达到、存在权衡、无改善或样本不足；完整正例比较前只能报告 Null 阶段完成。

报告全部分子/分母、双侧 95% Clopper–Pearson 区间及 FPR 单侧上界。正式后继阶段需要新的独立 calibration/test 与正例，报告 observed FPR 和区间，不强制单侧 95% 上界≤1%；若上界超过 1%，只能说“本次观测为…”，不能主张统计上确认 FPR≤1%。正常长度泛化仍未确认。

## 工程与运行

P0 实际执行：对照全 Bank 分数与旧 scorer 一致；候选仅路径空空改变；两套 scorer/detector 指纹不同；DP 与小图穷举一致；非法边及断路 abstain；非有限值拒绝；calibrator 交叉身份拒绝；checkpoint 绑定 runtime/input SHA。

每篇保留全部候选边、Span SHA、原 AMR、Parser 指纹、原图 SHA、beam 恢复记录、PENMAN/round-trip 证据、每个模板的三分量/总分、最佳/次优模板、两臂重新执行后的路径与换模板/换路径统计。Parser 只运行一次，双方使用同一 AMR。旧 cache 不具备恢复证据时不能复用。

Supervisor 和 flock 持久托管，逐篇原子 checkpoint，heartbeat 与日志区分 source_preparation、calibration、null_validation、pending_generation、complete、failed。SHA、依赖版本、模型文件 SHA、配置、输入清单均绑定 runtime；报告异常也写 failed，不能遗留 running。结果不影响冻结规模和 alpha。
