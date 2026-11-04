# Scripts 使用说明

本目录混合保存早期探索脚本、已终止协议脚本、审计脚本和未来可能复用的工具。脚本存在不代表它仍被授权运行。

## 当前 WP2 状态

WP2-v13 因官方 sampler 的 `MAX_TRIALS` fallback 不可达而在 attempt 372
封存 Stop。v14 修复了控制流，但其“372 条严格逐条复现”合同在 attempt 2
被真实 fallback 导致的 output identity 变化否定，因此也已按合同封存 Stop。

SemStamp v15 causal-prefix G1 已完成 1,111 attempts 并自报 1,000
eligible；冻结 H0 随后发现 5 条 prompt-only 空 continuation 被错误计为
positive，因此 v15 按合同 Stop。v16 以明确披露的 post-hoc `995+5`
validation rescue 补齐并通过独立 H0；v17 B0 detector health 随后 Go。

v18 B1 exact certificate 已完成 3,000-row score-blind readiness 并按
冻结合同 Stop：最大 pruned/all-allowed pairs 分别为 78,642,945 /
83,540,900，双 replay 保守预算为 86.6501 GPU-hours，超过 50M /
72h 门。自动 gate 未启动正式 GPU certificate，模型、region、score 和
threshold 调用均为 0；v17 B0 Go 保持不变。
该序列在证据上独立于SWAN CL04--CL08，不能补齐这些端点；在发布结构上
则与SWAN审计纳入同一篇TMLR稿，不单独投稿。2026-10-02方向归位及
2026-10-06投稿纠正后，仍不授权v19、exact或cheap-bound后继。

根因是继承的 `validate_output` 记录了 `output_equals_prompt`，但当 prompt
本身包含至少两个合法单位时没有产生 exclusion code。不得删除 5 条后补样或
从 attempt 1111 继续；任何重试必须另立后继协议并从 attempt 0 重跑。

相关入口：

| 脚本 | 用途 |
|---|---|
| `run_certified_resegmentation_wp2_v15_generation.py` | 已封存的 v15 generation runner；不得恢复或修改本次终态 |
| `semstamp_bounded_generation.py` | `MAX_TRIALS=100` 终止语义修复 |
| `freeze_certified_resegmentation_wp2_v14_prefix.py` | 从冻结 v13 公开日志重建前缀门 |
| `audit_certified_resegmentation_wp2_v15_h0.py` | 已执行的 v15 identity-only 审计；因 prompt-only positive 拒绝终态 |
| `test_audit_certified_resegmentation_wp2_v15_h0.py` | H0 合成终态与失败门测试 |
| `audit_certified_resegmentation_local_bound.py` | CPU 组合审计、5k 局部下界和重新定中心反例 |
| `certified_resegmentation_wp2_v13_b0_support.py` | 无模型身份/统计/replay/checkpoint 库，无正式 detector 入口 |
| `audit_certified_resegmentation_wp2_v13_offline.py` | 合成双 replay 与已公开历史 score 回归 |
| `test_certified_resegmentation_wp2_v13_offline.py` | 离线准备测试 |
| `run_certified_resegmentation_wp2_v18_b1.py` | 已封存 v18 B1 readiness 与未启动的双 replay runner |
| `audit_certified_resegmentation_wp2_v18_b1.py` | 已封存 v18 readiness/终态独立 auditor |
| `test_certified_resegmentation_wp2_v18_b1.py` | v18 renderer、证书不变量、篡改与统计回归测试 |
| `audit_certified_resegmentation_wp2_v18_p0_stop.py` | v18 P0 Stop 的 workload、预算、零 certificate 独立复核 |

v15 的 G1/H0 Stop 历史证据仍保留在
[`v15 实验目录`](../experiments/certified_resegmentation_wp2_v15_causal_prefix1/)。
当前 v18 协议、合同和运行证据位于
[`v18 实验目录`](../experiments/certified_resegmentation_wp2_v18_b1_exact_certificate1/)。
v18 已封存为 `b1_readiness_stop`；任何 exact 后继都必须另立协议并先升级
已验证 core/预算，但该条件不构成当前授权。SimMark 继续保持 Stop，
SemStamp B1 也没有产生鲁棒性统计结果。当前执行边界见
[`方向归位裁决`](../experiments/research_direction_reconciliation_20261002/DECISION.md)。

## SWAN 历史阶段规则

1. 当前没有获授权的付费实验阶段；
2. `run_swan_count_neutral_final1.py` 是 SWAN count-neutral 合同的入口，只有最终
   `manifest.json` 验证通过且该付费阶段独立授权后才能发送请求；
3. 文件名包含 `planB`、`planC`、`planD`、`planE`、`api_`、`preflight`、`final_confirmation` 的脚本默认视为历史或已终止；
4. 默认读取根目录 `1.txt`、`api.txt` 或旧 `任务规划.md` 的脚本已被有意失活，不得通过把旧文件移回根目录来恢复；
5. 需要凭据的新运行必须使用显式参数、独立预算授权和新协议，不得恢复隐式 key-file 默认值；
6. `audit_*` 脚本也只能在对应冻结产物上运行，不能把审计重算当作新实验；
7. 修改历史脚本前先确认不会破坏现有 artifact 的恢复能力。

当前执行计划见根目录 [`任务清单.md`](../任务清单.md)。

纠偏阶段允许的无模型基础设施：`freeze_swan_reframing.py`、
`freeze_swan_claims_v2.py`、`lint_research_claims.py`、
`swan_reframing_statistics.py`及相应纯函数测试。
它们只处理保存文件、术语、公式和设计假设，不授权旧实验runner。
freeze为只创建操作，已有文件使用check；prepare/check/status在同一终端串行。

final1 的完整手动命令、质量评审顺序和检测门见
[`experiments/swan_count_neutral_final1/README.md`](../experiments/swan_count_neutral_final1/README.md)。
pilot 上限为 USD 27.13 / 9601 calls，main 上限为
USD 1694.88 / 600000 calls；两阶段授权互不继承。代码不会代填两名真人
评审，也不会把 parser/S2Match 缺失变成低分。
