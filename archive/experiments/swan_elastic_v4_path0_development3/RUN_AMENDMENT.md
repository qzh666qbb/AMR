# SWAN v4 path0：第三次工程执行

2026-10-19。本次执行沿用[新版开发协议](../swan_elastic_v4_path0_development1/PROTOCOL.md)的全部规模、来源、资格、alpha 和主次端点，保留 development1/development2 的失败记录。

development1 在 AppleDouble 元数据扫描失败。development2 完成全部 1250 个来源冻结与身份校验后，在 runtime 构建时发现 SciPy 未安装，记 failed；没有加载 Parser、没有产生检测行或冻结 runtime。

已安装 SciPy 1.14.1。本次在 P0 中先实际检查全部依赖版本，并记录执行修订 SHA。沿用 development2 冻结的输入清单、来源 allocation ID 和历史快照；这些输入与已见历史的关系以原冻结时刻为准，不能因其自身已分配而重新取样。development2 的 PROTOCOL.md 作为输入 SHA 所绑定的原件保留在本目录，双方共同约束本次执行。

代码、P0、runtime、cache 和逐篇 checkpoint 属于 development3。统计量、源数据和样本数不变；依赖修复发生在观察检测结果之前。
