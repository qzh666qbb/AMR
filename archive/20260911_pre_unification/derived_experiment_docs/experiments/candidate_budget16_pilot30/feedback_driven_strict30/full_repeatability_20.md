# 全量候选重复性审计

对全部 120 个候选做了 20 次独立重复 S2Match 计算。

- 候选总数: 120
- 段落总数: 30
- 平均候选逃逸率: 0.217
- 全部段落平均ASR: 0.367
- 候选中有任意逃逸的数量: 26
- 候选中全程逃逸的数量: 26
- 候选中出现混合翻转的数量: 0

段落成功频率:
4=20/20, 5=20/20, 6=20/20, 10=20/20, 11=20/20, 13=20/20, 14=20/20, 16=20/20, 20=20/20, 23=20/20, 26=20/20

最不稳定的候选（按是否翻转排序）:
- paragraph 0 / candidate 0 / minimal_boundary : escape_rate=0.000, base_z=3.873, unique={'3.8729833462074166': 20}
- paragraph 0 / candidate 1 / clause_reorder : escape_rate=0.000, base_z=3.873, unique={'3.8729833462074166': 20}
- paragraph 0 / candidate 2 / voice_control : escape_rate=0.000, base_z=3.873, unique={'3.8729833462074166': 20}
- paragraph 0 / candidate 3 / nominalization : escape_rate=0.000, base_z=3.873, unique={'3.8729833462074166': 20}
- paragraph 1 / candidate 0 / minimal_boundary : escape_rate=0.000, base_z=3.873, unique={'3.8729833462074166': 20}
- paragraph 1 / candidate 1 / clause_reorder : escape_rate=0.000, base_z=3.873, unique={'3.8729833462074166': 20}
- paragraph 1 / candidate 2 / voice_control : escape_rate=0.000, base_z=2.840, unique={'2.840187787218772': 20}
- paragraph 1 / candidate 3 / nominalization : escape_rate=0.000, base_z=2.840, unique={'2.840187787218772': 20}
- paragraph 2 / candidate 0 / minimal_boundary : escape_rate=0.000, base_z=3.873, unique={'3.8729833462074166': 20}
- paragraph 2 / candidate 1 / clause_reorder : escape_rate=0.000, base_z=3.873, unique={'3.8729833462074166': 20}