# B5 人工复核表使用说明

文件 `planB_b5_human_review.csv` 包含三种方法各 10 个最低 z-score 候选，共 30 条。

请逐条填写：

- `human_quality`：1–5 分（1=不可读，5=自然流畅）
- `semantic_preserved`：yes / partial / no
- `major_error`：yes / no
- `notes`：简要说明错译、事实改变、句子缺失或明显不自然之处

判定“可接受攻击候选”的最低条件：质量 ≥4、语义为 yes 或可解释的 partial、无 major error。只有满足条件后，才可报告其检测分数下降；否则应归入失败案例。
