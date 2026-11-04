# SWAN v4 path0 开发比较运行状态

2026-10-19 03:22 UTC 快照：远端 Supervisor 状态 **RUNNING**，1000 篇新 calibration 已完成 **74/1000**，真实 RTX 3090 Parser 已开始处理。后台任务独立于本地终端持续运行，当前没有 stderr 异常。

执行程序为 `swan-elastic-v4-path0-development3`。校准完成后自动运行 200 篇独立 Null 的 baseline 和五种固定压力变体，共 1200 行；每行保存两臂全 Bank 重评分和完整 DP 结果。进度以远端 HEARTBEAT.json 为准，本文件不是实时监控。

新来源已全部冻结：calibration 1000，Null 200，正例 source 槽位 50。四长度层分别为 250/250/250/250、50/50/50/50、13/13/12/12；来源身份交集 0。C4 train shard 20，固定 revision；扫描至原始行 1257 即达到预声明配额。没有根据检测结果补样。

本地和远端 P0：10400 次全 Bank 分量比较、240 次 DP/穷举一致性案例通过；对照身份与原 scorer 相同，path0 身份不同；非有限分数、非法边、跨臂 calibration 和错误 checkpoint 身份均被拒绝。临时集成测试还验证完整候选档案、共享 AMR 冷/热复放一致及 1250 来源配额恢复。

development1 在历史扫描遇到 AppleDouble 元数据失败，development2 在运行身份建立前因 SciPy 缺失失败；失败日志和状态保留。补齐 SciPy1.14.1 后，development3 沿用已冻结输入、另立 runtime/cache/checkpoint。两次失败都发生在检测之前。

**待完成**：1000 校准及独立 Null 比较；50 正例生成与质量资格、TPR 和各条件存活。当前环境没有可用的既有生成 API 凭证，正例 source 槽位只代表已预分配来源。不能将 Null 完成或低误报写成整体开发 Go。

执行依据：[新版开发协议](../swan_elastic_v4_path0_development1/PROTOCOL.md)、[执行修订](RUN_AMENDMENT.md)。新版 observed FPR 目标 1%，主鲁棒端点为最差单条件存活，联合存活为次指标；允许明确报告性能权衡。正式 P2 尚未启动，旧 51/5000 和 No-Go 不变。
