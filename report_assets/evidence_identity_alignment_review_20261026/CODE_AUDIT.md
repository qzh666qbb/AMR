# 第四篇证据身份对齐：代码核查

2026-10-26。结论：**部分实现，尚不能执行完整三臂身份验证。**
已有局部对应选择和规则保护；没有严格无对齐模式、候选集内身份Oracle、
完整金标身份评分及独立文档校准闭环。本次未改检测运行代码。

审查范围为当前multiview候选、视图、对应、检测、运行器、评审导出及
相关测试；其他论文的DP oracle、原句界恢复和已知槽位oracle不能替代
本问题。身份指两个观测支持同一事实发生实例，须保留核心角色、否定、
对象、发生状态与时间/条件；高语义相似和相同模板不作身份真值。

## 1. 已实现的逻辑

| 功能 | 实现证据 | 当前含义 |
|---|---|---|
| 事件观测与结构签名 | [extract_events](file:///Users/bytedance/AMR/src/core/multiview_views.py#L46-L122) | 谓词、角色、否定、实体和数值进入签名，支持反向角色归一化；不是独立语义真值 |
| 图事件冲突 | [event_conflicts](file:///Users/bytedance/AMR/src/core/multiview_alignment.py#L42-L54) | 同谓词下检查否定、数值、ARG0/1/2差异 |
| 表层与图事件冲突 | [surface_conflicts](file:///Users/bytedance/AMR/src/core/multiview_alignment.py#L57-L81) | 有明确同谓词线索时检查否定、数值及显式名字的ARG0/1；无线索保持未知 |
| 身份分组 | [event_groups](file:///Users/bytedance/AMR/src/core/multiview_alignment.py#L84-L118) | 相同签名合组；修饰差异的合组检查核心身份、跨度重叠和部分保护关系 |
| 容量约束对应 | [align](file:///Users/bytedance/AMR/src/core/multiview_alignment.py#L121-L217) | predicted/position/semantic，MILP求最优；事件组容量1、候选默认容量3 |
| Bank与对应优化分离 | [align接口](file:///Users/bytedance/AMR/src/core/multiview_alignment.py#L121-L162) | 优化器不接收Bank分数；相似度、位置和稳定并列排序选择对应 |
| 图正文去重与审计输出 | [detect](file:///Users/bytedance/AMR/src/core/multiview_detector.py#L69-L154) | 绑定偏移、SHA、parser日志；一张图计分一次，保留缺失原因及对应轨迹 |
| 缓存与未校准状态 | [运行器](file:///Users/bytedance/AMR/scripts/run_swan_multiview.py#L169-L257)、[检测输出](file:///Users/bytedance/AMR/src/core/multiview_detector.py#L143-L154) | cache-only缺失报错；is_watermarked/p_value为空，不伪造检测判定 |

这些是工程实现证据。已保存的单来源记录显示有跨片段对应，但人工
身份标签和新FPR/TPR未测，不能据匹配数宣称对应正确。
[实际实现报告](../../experiments/swan_multiview_development1/IMPLEMENTATION_REPORT.md)

## 2. 尚未实现或需限制的逻辑

| 编号 | 发现 | 证据与影响 |
|---|---|---|
| G01 | 无严格无对齐臂和身份Oracle | `align()`及DetectorConfig只接受predicted/position/semantic；`position`仍先分组、枚举边并求MILP。运行器五种配置没有A0/A2。[模式定义](file:///Users/bytedance/AMR/src/core/multiview_detector.py#L19-L30)、[五配置](file:///Users/bytedance/AMR/scripts/run_swan_multiview.py#L207-L217) |
| G02 | 没有完整金标身份评分 | 评审导出只池化各方法selected对应，提供空fact_id/same_fact等字段，没有读取返回标签、裁决、补漏事实或计算身份precision/recall。[评审导出](file:///Users/bytedance/AMR/scripts/export_swan_multiview_review.py#L19-L69) |
| G03 | 候选内可表达域受限 | 事件需要核心角色；对应先要求支持跨度重叠或文本SHA相同。完整非连续支持及一对多/多对一组搜索不存在。Oracle不能仅过滤这些预测edges，否则继承预测漏边。[分组排除](file:///Users/bytedance/AMR/src/core/multiview_alignment.py#L91-L118)、[边筛选](file:///Users/bytedance/AMR/src/core/multiview_alignment.py#L132-L161) |
| G04 | 状态与发生实例保护不完整 | 相同signature直接合组，不检查位置；不同signature在核心相同且重叠时只保护time/condition/concession/location/polarity，未保护mode。存在将重复发生合组、或忽略部分模态差异的静态路径，尚未测真实错误率。[合组分支](file:///Users/bytedance/AMR/src/core/multiview_alignment.py#L99-L110) |
| G05 | 冲突规则不等于完整语义合同 | event_conflicts在谓词不同即返回空；表层仅同词元、显式ARG0/1名字检查，没有接收方ARG2的表层绑定、完整别名/共指、计划—已发生分类或否定作用域验证。[图冲突](file:///Users/bytedance/AMR/src/core/multiview_alignment.py#L42-L54)、[表层提取](file:///Users/bytedance/AMR/src/core/multiview_views.py#L159-L194) |
| G06 | 完整检测校准与身份效用未接通 | 返回uncalibrated_development_diagnostic；运行器audit只检查缓存/恢复绑定。没有各三臂独立Null阈值及文档级FPR/TPR输出。[检测终态](file:///Users/bytedance/AMR/src/core/multiview_detector.py#L143-L154)、[audit分支](file:///Users/bytedance/AMR/scripts/run_swan_multiview.py#L289-L294) |

G04/G05是代码路径与覆盖限制，不是从新样本估计出的语义错误率。
缺角色被排除也不表示对应事实不存在。三臂主比较应保留原子观测，
避免把A1预测的身份组作为Oracle不可更改的真值。

## 3. 现有“验证”和Oracle的实际含义

运行器的`audit`验证缓存及parser恢复事件绑定，不能确认两侧是否支持
同一真实事实。`swan_multiview_preflight.py`生成的事实字段为空，标签
身份为generation_trace_only_not_gold，预检查不是完成的金标评审。
[预检字段](file:///Users/bytedance/AMR/scripts/swan_multiview_preflight.py#L150-L194)

`multiscale_blind.assemble_features(carriers=...)`使用已知载体位置，
默认版本按窗口重叠聚合；没有角色/状态身份标签。
[几何聚合](file:///Users/bytedance/AMR/src/core/multiscale_blind.py#L78-L94)
P2的目标AMR/符号诊断和其他分支的组合DP oracle分别检验不同对象，
不能在本稿记为候选集内证据身份Oracle。

## 4. 本次实际测试

执行现有命令：

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_swan_multiview.py -v
```

结果：14项通过，exit code 0，unittest报告耗时5.339秒。
覆盖候选偏移/预算、事件细节、反向角色、否定/角色/数值冲突、真实
spaCy被动句线索、重复容量、部分时间保护、短图并列选择、缺失及缓存
miss报错。[现有测试](file:///Users/bytedance/AMR/tests/test_swan_multiview.py)

该套测试没有覆盖完整真人身份真值、状态差异保护、重复发生实例、
独立金标Oracle、未知位置恢复recall或独立文档FPR。手写图及规则输入
只验证工程逻辑；本次没有新增语义性能实验、AMR推理、向量编码、训练、
API调用或watermark detector运行，不能写成对齐效用过门。

## 5. 三臂接入前的工程顺序

1. 固定原子候选/事件观测ID与对应全集，保存全集、图正文和代码SHA。
2. 接入严格A0局部关联；A1使用冻结预测规则；A2使用独立标注确认关系，
   不共享A1的相似度/拒配过滤，不读取目标模板或正负标签。
3. 明确状态、别名/共指、重复发生实例和无事件载体处理，保留unknown；
   只增加未经数据验证的规则不能直接称为修复成功。
4. 补完整事实清单与候选支持标注，实施预测封存后评分，输出身份、
   覆盖、错误拼接和全分母恢复指标。
5. 复用相同Bank分数、固定权重及图SHA去重，各臂完整Null校准后才报告
   盲检测；Oracle假阳只作诊断。模式改变另立运行版本，不覆盖旧结果。

详细比较合同见[三臂方案](../../论文/01_句界审计与精确评估/P4_证据身份对齐_三臂验证方案.md)。
审查源码SHA、受保护原来源与文档/链接验证见
[来源清单](SOURCE_SHA256.json)、[本轮核验](VALIDATION.json)。
