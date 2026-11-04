# 论文主结果表草案

这张表把当前已经完成的核心实验压成一个可以直接改进到论文里的版本。



## 主结论

| 维度 | 结果 |
|---|---:|
| Strict Valid ASR | 30.0%（9/30） |
| 95% CI | [14.7%, 49.4%] |
| Strict candidate rate | 13.3%（16/120） |
| 95% CI | [7.8%, 20.7%] |
| Threshold plateau | 0.65–0.80 |
| 50-repeat stable candidates | 120/120 |
| 50-repeat mixed flips | 0 |

## 候选预算消融

| K | Escape rate | Surface-valid rate | Semantic-valid rate | Valid ASR | Mean z |
|---:|---:|---:|---:|---:|---:|
| 1 | 100.0% | 96.7% | 33.3% | 33.3% | -0.311 |
| 2 | 100.0% | 100.0% | 26.7% | 26.7% | -0.418 |
| 4 | 100.0% | 100.0% | 30.0% | 30.0% | -0.680 |
| 8 | 100.0% | 100.0% | 30.0% | 30.0% | -0.710 |
| 16 | 100.0% | 100.0% | 30.0% | 30.0% | -0.722 |

## 提示组件消融

| 方法 | Mean attack z | Mean z drop | Escape rate | AUROC | 说明 |
|---|---:|---:|---:|---:|---|
| Weak paraphrase | 2.526 | 0.645 | 33.3% | 0.934 | 变化最弱 |
| No boundary change | 2.268 | 0.903 | 40.0% | 0.906 | 只做词面/论元调整 |
| Boundary only | 0.166 | 3.005 | 93.3% | 0.828 | 最强逃逸 primitive |
| Full E2 | 1.208 | 1.963 | 73.3% | 0.903 | 完整方法 |

## 稳定性

| 检查 | 结果 |
|---|---:|
| 严格子集 9/9 成功段落 50-repeat 稳定性 | 100% |
| 全量 120 候选 binary flip | 0 |
| 全量 120 候选 stable candidates | 120/120 |

## 失败分析摘要

高频失败模式：

1. repetition / template echo
2. fragmentation
3. vagueness / information loss
4. unsupported causal or relational change
5. parser-shaped unnatural wording

## 论文里建议的写法

- 主文只保留 1 张主结果表 + 1 张稳定性图；
- 消融表放正文或附录；
- 失败分析放正文末尾或附录；
- 复现性检查单放附录。

