# SWAN v4 path0：第二次工程执行

2026-10-19。本次执行沿用[新版开发协议](../swan_elastic_v4_path0_development1/PROTOCOL.md)的全部规模、来源、生成资格、alpha 和主次端点。协议 version 为 `path0-development-v1`，执行目录版本为 development2。

development1 在历史扫描时读到 AppleDouble `._*.json` 二进制元数据而抛出 UnicodeDecodeError，已记录 failed；当时没有物化新来源、没有加载 Parser，也没有检测结果。本次保留原失败任务，不覆盖其 P0 或日志。

唯一工程修正：历史扫描将文件名以 `._` 开头的 AppleDouble 文件作为元数据排除，同时记录其路径与 SHA。其对应的实际 JSON/JSONL 仍必须完整扫描；其他解析错误仍显式失败。allocation ID 使用执行目录名，避免跨执行身份混用。

冻结本地和远端历史快照后按新版协议执行实际 P0、独立来源分配、1000 篇两臂校准和 200 篇 × 6 条件 Null 比较。50 个正例 source 槽位先冻结；生成凭证或依赖未就绪时标注 pending_generation，不发布完整 TPR 或开发 Go。
