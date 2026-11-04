# 英文稿原文出处与优化目标修订

2026-10-25。修改现有英文稿并重新编译PDF；使用既有冻结证据，没有新实验、
阈值调整或外部投稿。旧稿及构建记录保存在工作区
`report_assets/manuscript_revision_20261025/before/`。

## 本稿定位

目前可以继续完成范围明确的系统审计与精确评估稿。重点是可复查的SWAN
A–B–A行为、明确声明的完整威胁集，以及正确性与真实系统可行性的区分。
句界风险、z-test、单元数平方根效应和标准DP原理均有先例，不能列为原创
理论贡献。正常长度count-neutral机制与真实最终certificate仍缺失，因此
本次编辑不证明新颖性充分或录用可期。

## 写入英文稿的修改

| 位置 | 修改 |
|---|---|
| Abstract、Introduction、贡献列表 | 明确句界风险已有先例；将贡献收敛为具体行为证据、精确评估合同和实现审计三项 |
| Related Work | 补充原论文的具体章节，并区分已知风险、基础推论和本框架的任务差异 |
| Part I / Detection chain and statistical boundary | 将统计上限明确写成SWAN已有公式结合$G\leq N$的直接推论；补充严格阈值的必要单元数条件 |
| Part II / Detection, alignment, and worst-case certification | 新增SWAN、AliMark、本框架的方程推导与对照表，标记为方法比较 |
| Exact dynamic programs / Soundness | 说明正确性使用标准最优子结构论证；复杂度中使用$k+1$包含零修改状态 |
| Discussion、Limitations、Conclusion | 全文对齐原创性边界；不声称新水印、全新攻击类别或一般不可能性定理 |

## 原文出处

| 原文 | 已有说明 |
|---|---|
| [SWAN §3.3](https://arxiv.org/html/2605.04305v1#S3.SS3) | 逐句AMR bank最大匹配与one-proportion z-test |
| [SWAN §6 Future Directions](https://arxiv.org/html/2605.04305v1#S6.SS0.SSS0.Px1) | 拆句/合句改变AMR结构的挑战；多句AMR及AMR子图方向 |
| [SemStamp附录B](https://arxiv.org/html/2310.03991v2#A2)、[局限性](https://arxiv.org/html/2310.03991v2#Sx1.SS0.SSS0.Px1) | 句级z-test；句间编辑及重排风险 |
| [SimMark §4.3](https://arxiv.org/html/2502.02787v2#S4.SS3) | Paraphrase+Merge攻击实验 |
| [AliMark §2](https://arxiv.org/html/2605.29434v1#S2)、[§3.2](https://arxiv.org/html/2605.29434v1#S3.SS2)，式(4)–(6) | 结构扰动的信号损失；重构候选、block-edit DP及最终最大标准化分数 |
| [SwordStamp §4](https://arxiv.org/html/2608.27666v1#S4)、[§5.3](https://arxiv.org/html/2608.27666v1#S5.SS3) | 重分句改变检测单元；固定命中率下分数随$\sqrt n$变化 |

## 优化目标的具体区别

**原SWAN没有重分句DP。** 它在分句器给出的每个单元上，取与模板bank的
最大匹配值，转成命中计数，再计算文档z分数。最大值的搜索对象是模板，
没有搜索全部句界，也不对秘密比特序列做编辑对齐。

**AliMark有两层优化。** 对每个重构候选和秘密序列长度，block-edit DP
求最低插入、删除和替换代价；将代价归一化后，外层再取全部尝试中最高的
标准化水印分数。原RS使用输入、单次相邻合并和单次句中点拆分，名义候选数
为$2N$。长度相关归一化不同，外层最高z分数不必对应一次全局最低原始代价。

**本框架对检测分数作最坏情况优化。** 固定token、词序和单元数，在声明的
合法边界重定位集合中取最低检测分数。DP路径构造文本划分；一阶pairwise
评分的状态保留前一跨度、单元数及修改预算。它不对比特序列做插删替换对齐。

| 维度 | SWAN | AliMark | 本框架 |
|---|---|---|---|
| DP是否存在 | 原文档检测无DP | block-edit alignment DP | segmentation / edge-state DP |
| DP求什么 | 不适用 | 两条比特序列的最小编辑代价 | 全部声明合法划分的最小检测分数 |
| 文档最终选择 | 当前分句的命中z分数 | 所试重构/对齐的最高标准化z分数 | 威胁集内最低检测分数 |
| 检出条件含义 | 当前分句可检出 | 存在一个所试重构/对齐提供足够证据 | 所有声明合法划分均可检出 |
| 保证范围 | 无划分一致性保证 | 对齐最优不等于全部划分认证 | 限定威胁集和local评分器；不覆盖AliMark聚合检测器 |

这项区别明确了不同任务，但不自动证明本框架是全局首创算法。稿件中没有
加入SWAN与AliMark的性能比较，没有评价AliMark实测FPR是否有效，也没有
把我们的local-score DP当作AliMark认证。

## 证据状态

30篇短文本A–B–A、7,669组pairwise oracle、1,000组退化对照、
SemStamp v17 FPR47/1000与TPR954/1000，以及v18在评分前停止的记录均不变。
不存在新增certificate、阈值、witness或鲁棒率。

当前[英文源稿](main.tex)、[PDF](main.pdf)及[构建记录](BUILD.json)
由`EDITORIAL_MANIFEST.sha256`索引。父目录的历史MANIFEST及旧投稿裁决
保留原样；本次编辑不追溯改变其日期、原始哈希或实验结论。
