# AMR-SWAN Research Map

## Goal

在可信复现 SWAN 的基础上，研究 AMR 引导的受约束自适应重写能否在保持语义与事实的同时逃逸水印检测，并分析模板匹配、句子边界和解析器依赖造成的鲁棒性缺口。

## Confirmed decisions

- paper line: attack + robustness audit (D1-A)
- threat model: white-box upper bound + gray-box main result (D2-A)
- data: RealNews first, domain transfer later (D3-A)
- compute: RTX 4060 8GB + DeepSeek API hybrid (D4-C)
- defense: one lightweight defense after the attack is stable (D5-A)

## Research nodes

1. **SWAN baseline reproduction — in progress**
   - objective: establish a trustworthy detector and official-paraphrase comparison line
   - on success: move to AMR invariance audit and attack pilot
   - on failure: classify environment, implementation, model-version, or data mismatch before repair
2. **AMR invariance audit and 50-paragraph pilot — pending**
3. **White-box/gray-box main experiment on RealNews — pending**
4. **Ablation, transfer, human evaluation, and lightweight defense — pending**
5. **Paper writing and final evidence consolidation — pending**

## Baseline gate

No comparison-heavy attack claim is allowed until node 1 is verified or explicitly downgraded with a documented comparability caveat.
