# SWAN 弹性重构防御部署运行手册

## 1. 已部署环境

- 远端工程：`/workspace/swan-elastic-reconstruct`
- 虚拟环境：`/workspace/swan-elastic-reconstruct/.venv`
- GPU：RTX 3090 24 GB
- Parser：`models/model_parse_xfm_bart_large-v0_1_0`
- Tokenizer：`models/bart-large-tokenizer`
- 模型归档 SHA256：`2fcfe2113527accb752b6c574994022b3375b66d59da5042356c0f853e879849`

## 2. 上线前检查

```bash
cd /workspace/swan-elastic-reconstruct
.venv/bin/python -m unittest discover -s tests -p 'test_swan_*' -v
.venv/bin/python -m src.core.preflight \
  --device cuda \
  --batch-size 8 \
  --parse-smoke
```

验收条件：

- 12 项自动化测试全部通过；
- `cuda_available` 为 `true`；
- GPU 名称为 `NVIDIA GeForce RTX 3090`；
- Parser 指纹为 `amrlib:a6bb135d6884fca45d0c7ea9f22fc6aa7f9349d3c4672bba115b402a5d707c08`；
- 烟测输出 `watch -> :ARG0 cat` 与 `watch -> :ARG1 bird`。

## 3. 小规模预计算

输入必须是 JSONL，每行至少包含一个字符串字段 `text`。

```bash
cd /workspace/swan-elastic-reconstruct
.venv/bin/python -m src.core.precompute_amr_cache \
  --input /workspace/input/frozen_manifest.jsonl \
  --text-field text \
  --cache-db build/amr_cache.db \
  --device cuda \
  --model-batch-size 8 \
  --request-batch-size 128 \
  --document-batch-size 32 \
  --max-boundary-hops 3 \
  --max-span-words 80 \
  --limit 100
```

先运行 `--limit 100` 和 `--limit 1000` 两档吞吐基准，再决定全量预算。重复执行同一命令会按 Parser 指纹命中缓存；验收时要求 `parser_calls=0`。

## 4. Bank100 盲检测

```bash
cd /workspace/swan-elastic-reconstruct
.venv/bin/python -m src.core.bank_detector \
  --text-file /workspace/input/document.txt \
  --bank third_party/SWAN_official_2ca0c207_clean/amr_bank/banks/amr_bank_100.json \
  --cache-db build/bank_detector.db \
  --device cuda \
  --model-batch-size 16 \
  --request-batch-size 128 \
  --max-boundary-hops 3 \
  --max-span-words 80 \
  --score-center 0.65 \
  --green-threshold 0.75
```

验收时记录：

- Bank SHA256：`7e51393dacc5473ac9b3d96a12ba2a2591503432277ede174a125864e755a777`；
- Bank scorer 指纹：`swan-bank-scorer:2962c5bd...938ecd12`；
- Detector 指纹：`swan-bank-detector:74233cbd...b4818e`；
- Parser 指纹：`amrlib:a6bb135d...d707c08`；
- 输出中的完整路径、每个 Span 的最佳模板下标和三项子分数。

检测器不读取目标句序列。`viterbi_aligner.py` 仅用于旧回归测试，不得用于盲检测结果。

## 5. 经验 Null 校准

JSONL 输入默认从 `text` 字段读取；JSON manifest 可用
`--collection-field` 选择列表。以下命令仅展示已完成的 `n=2` 烟测：

```bash
cd /workspace/swan-elastic-reconstruct
.venv/bin/python -m src.core.null_calibration \
  --input experiments/swan_bank100_null_calibration1/manifest.json \
  --collection-field machine_sources \
  --text-field source \
  --limit 2 \
  --output build/null_c4_real_smoke_n2.json \
  --cache-db build/null_c4_real_smoke.db \
  --device cuda \
  --model-batch-size 16 \
  --request-batch-size 128
```

烟测结果：

- 2 篇 C4 源文本，413 个唯一 Span；
- 统计量为 `-0.9897721590`、`-1.3499509931`；
- 复放为 413 cache hit、0 miss、0 parser call，除 `created_at` 外制品完全一致；
- Null 制品 SHA256：`bf2abb07...f12939b`；
- 最小可分辨 p 值为 `1/3`，**不能报告 FPR 或显著检出结论**。

使用冻结制品校准单篇检测：

```bash
.venv/bin/python -m src.core.bank_detector \
  --text-file /workspace/input/document.txt \
  --cache-db build/bank_detector.db \
  --calibration build/null_c4_real_smoke_n2.json \
  --alpha 0.05 \
  --device cuda
```

检测器、Bank 或 Parser 任一指纹不一致时命令必须失败。桶内 Null
少于 `min_bucket_size` 时输出 `scope=global_fallback`。

## 6. 正式 1% Null 校准与独立 FPR

正式运行目录为：

```bash
/workspace/swan-elastic-reconstruct/experiments/swan_elastic_formal_null1
```

冻结输入：

- `INPUT_MANIFEST.json` SHA256：
  `59e430ed2b71235f78cbffde242d8e855710a404b84243e4268183d73fb986ce`
- `inputs/calibration_windows.jsonl`：1000 rows，SHA256
  `9987263442459fa2803afc70b34c8e8e5b204433579445f4786ac6c901befa49`
- `inputs/independent_test_windows.jsonl`：1000 rows，SHA256
  `ce607a1f52cad9d880c43f2a4f251991e3ad5ad931ac039465ed027799134959`
- 两组在 allocation ID、locator、规范化 source SHA、原文字节 SHA、
  URL SHA 和窗口 SHA 上交集均为 0。

正式运行参数：

```bash
cd /workspace/swan-elastic-reconstruct
flock -n /var/lock/swan-elastic-formal-null1.lock \
  .venv/bin/python -u -m src.core.formal_null_evaluation run \
  --output-root experiments/swan_elastic_formal_null1 \
  --cache-db build/formal_null_v1.db \
  --device cuda \
  --model-batch-size 16 \
  --request-batch-size 128 \
  --document-batch-size 16 \
  --max-boundary-hops 3 \
  --max-span-words 80 \
  --score-center 0.65 \
  --green-threshold 0.75 \
  --alpha 0.01
```

正式结果已完成：

- `results/CALIBRATION.json` SHA256：
  `a739d390819e65b22fd153ad363490076f08a4e1181fe40b348e9ed8665f7c7d`
- `results/calibration_rows.jsonl`：1000 rows，SHA256
  `1e952a3d497540322729a7291b2e8dbd571c2b6bbd1e733c13f5f236dd27da02`
- `results/independent_test_rows.jsonl`：1000 rows，SHA256
  `aa8c773d84f203a48126f03496d08799dbc600645203e0570ecab9d8e6ef1d19`
- `results/RESULT.json` SHA256：
  `341344a27cf49ea493c9b7087f2710655db601d4a742316ae767603db507b60b`
- `results/REPORT.md` SHA256：
  `3c9d69be154643566c59a82c56bd1a54831b5e90c0dce321e2db16a7beb3df00`

独立 FPR：

- 目标 alpha：0.01；
- false positives：6/1000；
- observed FPR：0.006；
- 双侧 95% 精确 Clopper-Pearson CI：[0.002205, 0.013013]；
- 单侧 95% 精确上界：0.011808。

PENMAN 审计：

- calibration：105/1000 篇文档有 warning，162/37232 个 Span 有
  warning，graph-valid 与 round-trip 均为 37232/37232；
- independent test：114/1000 篇文档有 warning，201/37276 个 Span 有
  warning，graph-valid 与 round-trip 均为 37276/37276；
- warning 按 Span 保留。任何结构非法或不可 round-trip 的 AMR 都会在统计
  finalize 前 fail-closed。

## 7. 冻结正例与句界攻击评估

正式运行目录为：

```bash
/workspace/swan-elastic-reconstruct/experiments/swan_elastic_positive_attack1
```

本实验只使用 `swan_elastic_formal_null1/results/CALIBRATION.json` 做推理校准，
不重拟合经验 Null，也不按新检测器结果筛选攻击样本。输入由
`confirmation_three_arm30` 的 30 篇 baseline 正例、30 篇 frozen
`boundary_only` no-feedback quality-valid 攻击样本，以及
`swan_normal_length_boundary_location1` 的 C0/R0/R1/R4 组成。

冻结输入：

- `INPUT_MANIFEST.json` SHA256：
  `753f282116dd0821de6b1a477afcb83b4506d2a7c8d4e86f4c76ad3fbec8aa2e`
- `inputs/evaluation_rows.jsonl`：64 rows，SHA256：
  `544c2251c0beeb8096ec01595e07d58509c500abd720b76bae0eec90f196e134`
- 正式 `swan_elastic_formal_null1/results/CALIBRATION.json` SHA256：
  `a739d390819e65b22fd153ad363490076f08a4e1181fe40b348e9ed8665f7c7d`

正式运行参数：

```bash
cd /workspace/swan-elastic-reconstruct
.venv/bin/python -u -m src.core.positive_attack_evaluation run \
  --output-root experiments/swan_elastic_positive_attack1 \
  --cache-db build/formal_positive_attack_v1.db \
  --device cuda \
  --model-batch-size 16 \
  --request-batch-size 128 \
  --document-batch-size 8 \
  --max-boundary-hops 3 \
  --max-span-words 80 \
  --score-center 0.65 \
  --green-threshold 0.75 \
  --alpha 0.01
```

正式结果已完成：

- `RUNTIME_MANIFEST.json` SHA256：
  `7b9ab3067bb91d9eb49aff55a4d4afaf08069935b517c37041992ebec0759825`
- `results/evaluation_rows.jsonl`：64 rows，SHA256：
  `5bb685f26ad02d898c1c2e31afa282bbdcc1fedc5bb2cdc133d827e7eb880ab7`
- `results/RESULT.json` SHA256：
  `0be534acc92eaaf206c7feba374a0db78f53417b5634c0c81e9318681c9f54d8`
- `results/REPORT.md` SHA256：
  `b8b4928a9a1e0e974a99e9370d4e01c2a3efdd8a5ebaa28fa4b05f40e9cbcebd`

核心指标：

- baseline positive detection：28/30 = 0.933333；
- historical-positive denominator 下 baseline detection：27/29 = 0.931034；
- boundary attack detection：27/30 = 0.900000；
- robust survival given baseline detected：27/28 = 0.964286；
- evasion given baseline detected：1/28 = 0.035714，唯一逃逸组为
  `c4-validation-252`；
- normal-length boundary case：C0/R0/R1/R4 全部检出，p-value 均为
  `0.0017421602787456446`。

PENMAN 审计：

- baseline：327/327 graph-valid，327/327 round-trip-valid；2 个 Span 有
  recoverable warning；
- boundary attack：284/284 graph-valid，284/284 round-trip-valid；0 warning；
- normal-length：228/228 graph-valid，228/228 round-trip-valid；0 warning。

短 confirmation 文本落在 0-80 词范围，而正式 Null 无该词数桶样本，因此这
60 行使用 `scope=global_fallback`；不能据此称为已确认的TPR@1%实际FPR。
4个normal-length条件使用桶内校准，但来自同一篇文档，仅为case study。

## 8. 短文本独立校准与固定标点压力验证

计划与冻结门限：
[`experiments/swan_elastic_length_boundary_audited2/PLAN.md`](experiments/swan_elastic_length_boundary_audited2/PLAN.md)。
本轮1000短calibration、1000独立test、5214行压力/既有输入，独立缓存为
`build/length_boundary_audited_v2.db`。原检测器与原calibration不变，新calibration
只补短文本桶；已有长文本桶统计逐项相等才允许继续。测试不参与拟合。

```bash
# 本地准备已完成；同一输入重跑须字节相等
.venv-swan-positive/bin/python -m src.core.length_boundary_validation prepare

# 远端启动前核验
cd /workspace/swan-elastic-reconstruct
.venv/bin/python -m unittest tests.test_swan_length_boundary_validation -v

# run_remote.sh含flock、断点恢复、固定参数及完成后的零模型audit
bash experiments/swan_elastic_length_boundary_audited2/run_remote.sh
```

初版在192条calibration后因上游图恢复审计缺口停止，partial及日志已本地封存；
`STOP.json`绑定其SHA，旧Supervisor关闭autostart，旧启动脚本拒绝恢复。
任务由Supervisor托管，任务名`swan-elastic-length-boundary-audited2`，全量现已完成。
审计器逐Span绑定所有尝试beam的日志，选中beam有ERROR时联合工程门失败，
但不按错误/分数删除样本。108个Span和16篇固定回放验证插桩不改变输出。
该任务出错后停止，不自动反复重启；修复可恢复的问题且SHA/参数相同后再启动。

全量7214行与两套审计已完成，22163个唯一Span的cache/图SHA绑定通过。
独立收尾任务`swan-elastic-length-boundary-audited2-finalize`已正常退出，
产出`COMPLETION_AUDIT.json`及`DECISION.md`；GPU已释放。新增3项异常测试
在本地/远端通过；收尾未改冻结评分实现或重启GPU任务。
新短FPR10/1000，单侧95%上界1.6903%；开发正例26/30、既有攻击17/30、
五变体最坏条件存活0/26。24个选中beam有ERROR，全分母保留，联合工程No-Go。
完整结果与后续顺序见[`CLOSEOUT.md`](experiments/swan_elastic_length_boundary_audited2/CLOSEOUT.md)。

```bash
supervisorctl status swan-elastic-length-boundary-audited2
cat experiments/swan_elastic_length_boundary_audited2/HEARTBEAT.json
tail -n 10 experiments/swan_elastic_length_boundary_audited2/RUN.log
```

结果输出为`results/RESULT.json`、`results/REPORT.md`、`results/CALIBRATION.json`；
完成后的审计文件为`AUDIT.json`。审计逐行重算原/新calibration的p-value，
验证SHA、输入/输出身份与汇总分母，新增模型调用为0。

```bash
.venv/bin/python -m src.core.length_boundary_validation audit \
  --output-root experiments/swan_elastic_length_boundary_audited2

.venv/bin/python experiments/swan_elastic_length_boundary_audited2/validate_completion.py \
  --output-root experiments/swan_elastic_length_boundary_audited2 \
  --cache-db build/length_boundary_audited_v2.db
```

工程Go要求干净与每个固定Null变体的FPR单侧95%上界≤2%，以及原文可检出
条件下五变体全部存活率≥80%；该探索门不等同于严格1%FPR证据。严格1%支撑
另要求干净短文本FPR单侧95%上界≤1%。失败保留结论，不修改本轮阈值和数据。

## 8.1 v2停止与v3修复开发门

v2不修改或重开v1结果。P0使用score-blind边界恢复与Span预算检查，已通过：
五变体恢复600/600、5150行全部有完整路径，正例/历史Null预算为
1.4793x/2.9582x。P1必须使用v2完整流程从头拟合1000条短calibration，再
评估30篇baseline、30篇既有攻击和150篇固定五变体；development不参与拟合。

2026-10-14用户提供纠正SSH入口后，远端Python 3.10.12、RTX 3090和磁盘门
通过；39项SWAN测试通过（1项按远端条件跳过），v2代码、P1 runner、
manifest及两份输入的五个SHA全部一致。Supervisor任务启动后完成144条
calibration；随后3词候选Span `FORT SILL, Okla.,`的选中beam产生PENMAN
反序列化warning，恢复图结构审计失败，任务按fail-closed门停止。
development为0/210，GPU已释放且无`RESULT.json`或`AUDIT.json`。
恢复白名单、SHA和精确命令见
[`START.md`](experiments/swan_elastic_v2_development1/START.md)，静态状态见
`START_STATUS.json`，停止证据见`PARTIAL_STOP.json`。该partial不构成P1
No-Go，且不再恢复。

v3已按方案版本化：保持v2候选拓扑和count-state DP，但将selected-beam
`ERROR/CRITICAL`、`graph_valid=false`或`roundtrip_valid=false`的边在DP前
排除；证据缺失仍停止，无完整合法路径则abstain并保留全分母。v3 P0重新
执行后为Go：五变体恢复600/600、5150/5150完整拓扑路径、Span预算
1.4793x/2.9582x，合成非法高分边和全非法路径门均通过，禁止调用为0。

P1b使用`experiments/swan_elastic_v3_development1`、独立
`build/swan_elastic_v3_development1.db`、flock和
`swan-elastic-v3-development1` Supervisor。远端SWAN测试Ran 50、0 fail、
1项按环境条件skip，白名单逐文件SHA和prepare重放通过；启动前v3 results/cache
均不存在。任务从calibration 0连续完成1000 calibration和210 development，
v2 partial未复制或读取。原故障row 152的非法边在DP前排除，文档经合法路径
完成。全流程共7条无效候选边、选中无效边0，calibration/development
abstain均为0，1210/1210全分母保留。

```bash
.venv/bin/python -m src.core.elastic_v3_development audit \
  --output-root experiments/swan_elastic_v3_development1
sha256sum experiments/swan_elastic_v3_development1/RESULT.json
```

远端自动audit与本地上述零模型audit均pass，`RESULT.json` SHA为
`7e623f7b937ba4baf0d2b2bca82121974602f53053350573d0077f4ecc50d53c`。
P1b最终baseline20/30，未达24/30；五变体全部条件存活18/20，通过80%门。
因此整体No-Go：停止在当前30篇上补规则，不准备新来源P2，也不升级正式
FPR/TPR或鲁棒性主张。完整收口见
[`CLOSEOUT.md`](experiments/swan_elastic_v3_development1/CLOSEOUT.md)。

## 8.2 v4 P2 原件审计与离线阈值探索

2026-10-15已同步`experiments/swan_elastic_v4_development1`与
`experiments/swan_elastic_v4_p2_formal`的远端原件。32个文件SHA全部一致；
源代码快照单独保存在`experiments/swan_elastic_v4_p2_audit1/evidence/runtime_source`。
本地v4 development文件与该快照不同，存在导入缺失；不要直接恢复该runner
或跳过P0。原P2没有执行时code SHA冻结记录，不能用事后快照补成预冻结。

本地可复现的审计入口为：

```bash
.venv/bin/python scripts/audit_swan_elastic_v4_p2.py
```

审计从绑定分布重算p值，不调用Parser或完整DP。原51/5000逐行复现；
无计数错误、非有限分数或阳性判定不一致，原始制品保持不变。其5000变体与
1000校准原文完全同源，正式FPR无效；P2正例为0，报告生成异常导致原
heartbeat停留于`development 5000/5000, running`，该状态不能证明进程存活。

P1b原件开发统计为25/30、20/25。固定检测器仅调整alpha，在两套分布全部
alpha≤0.01决策断点上均无联合可行解：保住P1b正例门时压力行误报至少
74/5000；P2校准在alpha=0.01时开发baseline仅22/30。
后继须先做新来源开发与执行契约修复，再冻结独立正式确认。
完整证据与止损条件见
[`REPORT.md`](experiments/swan_elastic_v4_p2_audit1/REPORT.md)和
[`FOLLOWUP.md`](experiments/swan_elastic_v4_p2_audit1/FOLLOWUP.md)。

## 8.3 v4 结构机制诊断与后继开发

2026-10-16完成6210行已选路径特征分析。复现入口：

```bash
.venv/bin/python scripts/analyze_swan_elastic_v4_mechanism.py
```

原审计包及P1b/P2制品SHA保持不变。新制品位于
`experiments/swan_elastic_v4_mechanism1/`，包括报告、没有正文的CSV、JSON、
特征来源SHA和开发比较合同。

当前参考代码对AMR二跳路径空-空计满分，匹配贡献0.25。39/51误报、
17/30 baseline与29/30空格正例变体包含该项；正例也会受取消奖励影响。
固定已选路径的贡献分解不能替代改后检测结果。P1b记录的scorer/decoder
SHA与当前文件一致，但P2缺实际代码SHA，不能据此确定P2运行时根因。

唯一候选只取消路径空-空奖励，已在`src/core/elastic_v4_path0.py`实施，双臂runner为
`src/core/elastic_v4_path0_development.py`。原机制合同封存；2026-10-19执行依据为
[新版协议](experiments/swan_elastic_v4_path0_development1/PROTOCOL.md)及
[执行修订](experiments/swan_elastic_v4_path0_development3/RUN_AMENDMENT.md)。
开发observed FPR目标1%，最差单个条件存活为主，五变体全部存活为次，
允许明确报告TPR/FPR权衡。正式后继报告独立观测值和区间，不要求上界机械过门。

本地和远端P0已实际通过。1250来源及两端历史快照已冻结，来源交集0。
development1在AppleDouble元数据扫描失败，development2因缺SciPy在runtime前失败；
日志/P0/failed保留。安装SciPy1.14.1后，development3沿用原输入启动Supervisor
后台运行：1000篇校准后自动处理200篇×6条件Null；50正例source槽位待生成。
当前无新TPR和整体Go。全部候选AMR、恢复beam、全Bank分量及两臂完整DP证据逐篇保存。

在远端工程根目录查看：

```bash
supervisorctl status swan-elastic-v4-path0-development3
cat experiments/swan_elastic_v4_path0_development3/HEARTBEAT.json
tail -20 experiments/swan_elastic_v4_path0_development3/RUN.log
```

本地进度快照见[STATUS.md](experiments/swan_elastic_v4_path0_development3/STATUS.md)。
运行代码、输入或配置身份改变时checkpoint拒绝恢复，修复须另立执行版本。
若仅任务被中断且所有绑定不变，使用Supervisor重启同一程序即可逐篇续跑。

## 9. 全量运行门禁

启动全量任务前必须冻结：

1. 输入 JSONL 与 SHA256；
2. `text` 字段定义和去重规则；
3. `max_boundary_hops`、`max_span_words`；
4. Parser 与 tokenizer 指纹；
5. 输出数据库路径、备份位置和停止条件。
6. Bank 文件、结构权重、`score_center`、`green_threshold` 和三类指纹；
7. Null 校准集与独立保留集的分离规则、词数分桶、显著性水平；
8. Parser 失败、PENMAN 恢复警告和缺失 Span 的计数门限。
9. 执行代码SHA、P0通过证据、hop惩罚和checkpoint重判策略；
10. calibration/test原文身份交集为0；五变体按原文分组，不能扩大独立分母。

1% 显著性至少需要 99 个 Null 样本才能得到可分辨 p 值，但
`n=99` 只是数学下限，不等于功效和分布覆盖充分。正式 FPR 必须在未参与
拟合的保留集上报告；正式 TPR 必须使用冻结的未攻击与攻击正例集。

当前代码已完成Bank盲检测、原正式Null校准/FPR和既有正例/攻击探索性检出；
短长度与各标点条件FPR补全已完成，联合工程No-Go。有效防御主张必须同时引用冻结攻击、
同适用范围独立Null与数据资格证据，不能仅据28/30或27/30升级。
