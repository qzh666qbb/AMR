# SWAN v4 path0 开发比较：Null 阶段完成

状态：50 个新正例槽位已冻结，待生成。当前没有 TPR 或鲁棒存活结果，不能裁为整体 Go。

| 条件 | 对照误报 | path0 误报 |
|---|---:|---:|
| baseline | 0/200 | 3/200 |
| semicolon_all | 0/200 | 3/200 |
| semicolon_pairs | 0/200 | 3/200 |
| comma_all | 0/200 | 3/200 |
| newline_all | 0/200 | 3/200 |
| space_all | 0/200 | 4/200 |

全部置信区间、原文级联合误报和预声明 alpha 曲线见 NULL_RESULT.json。

这批数据仅用于开发，正常长度与正式泛化尚未确认。
