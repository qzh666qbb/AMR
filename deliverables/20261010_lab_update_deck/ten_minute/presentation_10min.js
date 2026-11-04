const pptxgen = require("pptxgenjs");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");
const { autoFontSize } = require("./pptxgenjs_helpers/text");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "quzehao";
pptx.title = "语义水印的句界鲁棒性研究进展";
pptx.subject = "10-minute weekly research update";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Noto Sans CJK SC",
  bodyFontFace: "Noto Sans CJK SC",
  lang: "zh-CN",
};
pptx.defineSlideMaster({
  title: "BASE",
  background: { color: "F7F9FC" },
  objects: [],
  slideNumber: {
    x: 12.5, y: 7.05, w: 0.35, h: 0.2,
    fontFace: "Noto Sans CJK SC", fontSize: 9,
    color: "94A3B8", align: "right", margin: 0,
  },
});

const FONT = "Noto Sans CJK SC";
const C = {
  ink: "102A43", ink2: "334E68", slate: "627D98", muted: "9FB3C8",
  line: "D9E2EC", white: "FFFFFF", paper: "F7F9FC",
  blue: "2457D6", paleBlue: "EAF0FF",
  cyan: "1F9FB5", paleCyan: "E5F7FA",
  coral: "E4574F", paleCoral: "FDECE9",
  gold: "D69E2E", paleGold: "FFF7DF",
  green: "2E8B72", paleGreen: "E7F4EF",
  gray: "718096", paleGray: "EDF2F7",
};

function box(slide, x, y, w, h, fill = C.white, line = C.line, width = 1) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color: fill },
    line: { color: line, width },
  });
}

function arrow(slide, x, y, w, h, color = C.slate, width = 1.5) {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h,
    line: { color, width, beginArrowType: "none", endArrowType: "triangle" },
  });
}

function header(slide, title, section, subtitle = "") {
  slide.addText(section, {
    x: 0.65, y: 0.2, w: 3.5, h: 0.22,
    fontFace: FONT, fontSize: 10, bold: true, color: C.blue,
    charSpacing: 1.1, margin: 0,
  });
  slide.addText(title, autoFontSize(title, FONT, {
    x: 0.65, y: 0.55, w: 12.0, h: 0.72,
    mode: "shrink", fontSize: 33, minFontSize: 26, maxFontSize: 33,
    bold: true, color: C.ink, margin: 0,
  }));
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.67, y: 1.29, w: 11.8, h: 0.3,
      fontFace: FONT, fontSize: 13, color: C.slate, margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.65, y: 1.7, w: 12.0, h: 0,
    line: { color: C.line, width: 1 },
  });
}

function footer(slide, text = "实验室周会 · 语义水印的句界鲁棒性研究进展") {
  slide.addText(text, {
    x: 0.65, y: 7.07, w: 11.55, h: 0.17,
    fontFace: FONT, fontSize: 8.2, color: C.muted, margin: 0,
  });
}

function bullets(slide, items, x, y, w, h, size = 14, color = C.ink2) {
  const runs = items.map((text, i) => ({
    text,
    options: {
      bullet: { indent: size * 1.1 },
      hanging: size * 0.3,
      breakLine: i < items.length - 1,
    },
  }));
  slide.addText(runs, {
    x, y, w, h, fontFace: FONT, fontSize: size, color,
    margin: 0.04, breakLine: false, paraSpaceAfterPt: 10,
  });
}

function metric(slide, x, y, w, value, label, color) {
  box(slide, x, y, w, 0.92, C.white, color, 1.2);
  slide.addText(value, autoFontSize(value, FONT, {
    x: x + 0.1, y: y + 0.13, w: w - 0.2, h: 0.34,
    mode: "shrink", fontSize: 23, minFontSize: 15, maxFontSize: 23,
    bold: true, color, align: "center", margin: 0,
  }));
  slide.addText(label, {
    x: x + 0.1, y: y + 0.58, w: w - 0.2, h: 0.18,
    fontFace: FONT, fontSize: 9.5, color: C.slate, align: "center", margin: 0,
  });
}

function band(slide, text, color = C.blue, fill = C.paleBlue) {
  box(slide, 0.8, 6.28, 11.72, 0.52, fill, color, 1.1);
  slide.addText(text, {
    x: 1.05, y: 6.43, w: 11.2, h: 0.22,
    fontFace: FONT, fontSize: 12.2, bold: true, color, align: "center", margin: 0,
  });
}

function addNotesAndCheck(slide, script, selfNote) {
  slide.addNotes(`【汇报讲稿】\n${script}\n\n【自我理解备注】\n${selfNote}`);
  warnIfSlideHasOverlaps(slide, pptx, {
    muteContainment: true,
    ignoreLines: true,
    ignoreDecorativeShapes: true,
  });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function resultBar(slide, x, y, label, value, color) {
  slide.addText(label, {
    x, y: y - 0.01, w: 1.25, h: 0.24,
    fontFace: FONT, fontSize: 10.5, color: C.ink2, margin: 0,
  });
  if (value < 100) box(slide, x + 1.35, y, 2.85, 0.22, C.paleGray, C.paleGray, 0);
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 1.35, y, w: Math.max(0.02, 2.85 * value / 100), h: 0.22,
    fill: { color }, line: { color },
  });
  slide.addText(`${value}%`, {
    x: x + 3.5, y: y - 0.27, w: 0.68, h: 0.2,
    fontFace: FONT, fontSize: 9, bold: true, color, align: "right", margin: 0,
  });
}

// 1
{
  const slide = pptx.addSlide("BASE");
  slide.background = { color: C.ink };
  slide.addText("实验室周会 · 研究进展", {
    x: 0.78, y: 0.55, w: 3.2, h: 0.25,
    fontFace: FONT, fontSize: 11, bold: true, color: "75D5DE", charSpacing: 1.4, margin: 0,
  });
  slide.addText("语义水印的\n句界鲁棒性研究进展", {
    x: 0.78, y: 1.25, w: 6.4, h: 1.65,
    fontFace: FONT, fontSize: 34, bold: true, color: C.white, margin: 0,
  });
  slide.addText("从 SWAN 句界依赖审计到 SemStamp 可认证重分句防御", {
    x: 0.82, y: 3.2, w: 6.0, h: 0.7,
    fontFace: FONT, fontSize: 16, color: "CBD5E1", margin: 0,
  });
  const roles = [
    ["研究者", "假设 · 方案 · 指标 · 裁决", C.blue],
    ["Agent", "推理 · 日志 · 统计 · 重放", C.cyan],
    ["证据", "Go / Stop · 主张边界", C.green],
  ];
  roles.forEach((r, i) => {
    const y = 1.35 + i * 1.48;
    box(slide, 8.15, y, 3.85, 0.86, "163B51", r[2], 1.4);
    slide.addText(r[0], { x: 8.45, y: y + 0.18, w: 1.0, h: 0.25, fontFace: FONT, fontSize: 15, bold: true, color: r[2], margin: 0 });
    slide.addText(r[1], { x: 9.55, y: y + 0.2, w: 2.05, h: 0.23, fontFace: FONT, fontSize: 11, color: C.white, margin: 0 });
    if (i < 2) arrow(slide, 10.08, y + 0.92, 0, 0.42, C.muted, 1.5);
  });
  slide.addText("汇报人：quzehao  ·  2026.10.10", {
    x: 0.8, y: 6.72, w: 4.1, h: 0.22,
    fontFace: FONT, fontSize: 10, color: C.muted, margin: 0,
  });
  addNotesAndCheck(
    slide,
    "这次汇报主要回答两个问题。第一，为什么前期复杂 AMR 攻击效果有限，而句界重组非常有效；第二，在确认句界是关键诱因后，我们如何把它形式化为可认证重分句问题。实验由 Agent 流水线按我冻结的方案自动执行，研究问题、实验分组、指标和 Go/Stop 标准由我确定。",
    "课题最终论文题名尚未完全固定，本标题是组会用的中性研究进展标题。两条证据流不能混淆：SWAN 是系统审计；SemStamp 是独立认证防御探索。"
  );
}

// 2
{
  const slide = pptx.addSlide("BASE");
  header(slide, "目录：从问题发现到当前终态", "01 · 汇报结构");
  const items = [
    ["01", "研究背景与核心问题", C.gray],
    ["02", "研究目标、创新思路与技术架构", C.blue],
    ["03", "实验方案、环境与人机职责", C.cyan],
    ["04", "关键结果、异常与证据边界", C.coral],
    ["05", "下一阶段计划与总结", C.green],
  ];
  items.forEach((d, i) => {
    const y = 2.02 + i * 0.82;
    slide.addShape(pptx.ShapeType.ellipse, { x: 1.1, y, w: 0.48, h: 0.48, fill: { color: d[2] }, line: { color: d[2] } });
    slide.addText(d[0], { x: 1.1, y: y + 0.11, w: 0.48, h: 0.18, fontFace: FONT, fontSize: 9, bold: true, color: C.white, align: "center", margin: 0 });
    slide.addText(d[1], { x: 1.88, y: y + 0.07, w: 5.3, h: 0.3, fontFace: FONT, fontSize: 16, bold: true, color: C.ink, margin: 0 });
    if (i < 4) slide.addShape(pptx.ShapeType.line, { x: 1.34, y: y + 0.49, w: 0, h: 0.33, line: { color: C.line, width: 2 } });
  });
  box(slide, 8.15, 2.15, 3.6, 3.15, C.paleBlue, C.blue, 1.2);
  slide.addText("10 分钟讲法", { x: 8.55, y: 2.55, w: 2.8, h: 0.3, fontFace: FONT, fontSize: 18, bold: true, color: C.blue, align: "center", margin: 0 });
  slide.addText("问题 2 min\n设计与架构 2 min\n结果与异常 4 min\n计划与总结 2 min", {
    x: 8.7, y: 3.2, w: 2.5, h: 1.35,
    fontFace: FONT, fontSize: 14, color: C.ink2, align: "center", margin: 0,
  });
  band(slide, "详细 Plan A--AE 时间线保留在 37 页版本中，本次主讲证据主线。", C.gray, C.paleGray);
  footer(slide);
  addNotesAndCheck(
    slide,
    "汇报先讲为什么从 AMR 算子转向句界，再说明我如何设计控制实验。之后介绍 Agent 流水线如何执行、主要结果是什么，以及哪些路线因为证据或资源门停止。最后给出下一阶段的选择，而不是把所有 Stop 包装成成功。",
    "这一页控制在 20 秒。若老师要求逐版本追问，打开 37 页详细版，不要在 10 分钟稿中展开所有 Plan。"
  );
}

// 3
{
  const slide = pptx.addSlide("BASE");
  header(slide, "研究背景：早期找到了攻击面，但没有找到诱因", "02 · 研究背景");
  const nodes = [
    ["分句", "units"], ["BART AMR", "R(s)"], ["模板化", "T(R(s))"],
    ["S2Match", "固定 B"], ["聚合", "z(G,N)"],
  ];
  nodes.forEach((n, i) => {
    const x = 0.82 + i * 2.42;
    box(slide, x, 2.05, 1.8, 0.9, i === 0 ? C.paleCoral : C.white, i === 0 ? C.coral : C.line, 1.2);
    slide.addText(n[0], { x: x + 0.12, y: 2.25, w: 1.56, h: 0.25, fontFace: FONT, fontSize: 13.5, bold: true, color: i === 0 ? C.coral : C.ink, align: "center", margin: 0 });
    slide.addText(n[1], { x: x + 0.12, y: 2.62, w: 1.56, h: 0.18, fontFace: FONT, fontSize: 9, color: C.slate, align: "center", margin: 0 });
    if (i < 4) arrow(slide, x + 1.83, 2.5, 0.46, 0, C.muted, 1.2);
  });
  const history = [
    ["Plan A", "基线", "AUROC 0.985", C.blue],
    ["Plan B--D", "图算子", "低分 20%--26.7%", C.gray],
    ["Plan C", "语义质量", "仅 46% 通过", C.gold],
    ["Plan E", "反馈搜索", "低分显著但混入质量失败", C.coral],
  ];
  history.forEach((d, i) => {
    const x = 0.88 + i * 3.02;
    box(slide, x, 3.65, 2.62, 1.45, C.white, d[3], 1.1);
    slide.addText(d[0], { x: x + 0.18, y: 3.9, w: 0.8, h: 0.22, fontFace: FONT, fontSize: 11, bold: true, color: d[3], margin: 0 });
    slide.addText(d[1], { x: x + 1.02, y: 3.88, w: 1.35, h: 0.24, fontFace: FONT, fontSize: 13, bold: true, color: C.ink, align: "right", margin: 0 });
    slide.addText(d[2], { x: x + 0.18, y: 4.45, w: 2.25, h: 0.35, fontFace: FONT, fontSize: 11, color: C.slate, align: "center", margin: 0 });
  });
  band(slide, "老师上次反馈对应的核心缺口：哪个变量真正触发检测下降？", C.coral, C.paleCoral);
  footer(slide);
  addNotesAndCheck(
    slide,
    "前期我们从节点替换、边调整、重入拓扑一直做到反馈式联合改写。简单图操作会被重解析规范化，强图操作又容易破坏事实。Plan E 虽然找到了大量低分候选，但不能区分句界、parser、模板匹配和统计聚合各自的作用。老师提出的寻找诱因，正好击中了这个缺口。",
    "不要把 Plan E 的 249/250 说成有效攻击率，它只是至少一个低分候选的覆盖率。继续设计更强 AMR 算子会进一步放大事实错误。"
  );
}

// 4
{
  const slide = pptx.addSlide("BASE");
  header(slide, "研究目标与创新思路：从“找攻击”转向“验证句界”", "03 · 研究目标");
  slide.addText("我提出并冻结的诱因假设", {
    x: 0.85, y: 1.98, w: 3.1, h: 0.32,
    fontFace: FONT, fontSize: 17, bold: true, color: C.blue, margin: 0,
  });
  box(slide, 0.85, 2.48, 4.15, 2.95, C.paleBlue, C.blue, 1.4);
  slide.addText("句界同时改变", { x: 1.2, y: 2.85, w: 3.45, h: 0.32, fontFace: FONT, fontSize: 18, bold: true, color: C.ink, align: "center", margin: 0 });
  metric(slide, 1.18, 3.48, 1.48, "N", "检测单元数", C.coral);
  metric(slide, 3.18, 3.48, 1.48, "Scope", "候选图作用域", C.cyan);
  slide.addText("→ 改变统计上限与模板匹配", { x: 1.25, y: 4.75, w: 3.35, h: 0.28, fontFace: FONT, fontSize: 13.5, bold: true, color: C.blue, align: "center", margin: 0 });
  const preds = [
    ["P1", "只改标点仍改变判定", C.blue],
    ["P2", "恢复 units 后判定恢复", C.cyan],
    ["P3", "N=1 的统计上限低于阈值", C.coral],
    ["P4", "扩大图作用域降低局部匹配", C.gold],
  ];
  preds.forEach((p, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 5.75 + col * 3.2, y = 2.2 + row * 1.52;
    box(slide, x, y, 2.82, 1.08, C.white, p[2], 1.2);
    slide.addText(p[0], { x: x + 0.18, y: y + 0.15, w: 0.5, h: 0.25, fontFace: FONT, fontSize: 13, bold: true, color: p[2], margin: 0 });
    slide.addText(p[1], { x: x + 0.78, y: y + 0.18, w: 1.78, h: 0.55, fontFace: FONT, fontSize: 11.5, bold: true, color: C.ink2, margin: 0 });
  });
  band(slide, "扩展思路：对声明的全部合法 variable-N 重分句求 exact worst case。", C.green, C.paleGreen);
  footer(slide);
  addNotesAndCheck(
    slide,
    "我的核心直觉是，句界不是普通文本特征，而是检测器接口的一部分。它一方面决定统计样本数 N，另一方面决定每张 AMR 图覆盖多大文本范围。基于这个直觉，我把问题拆成四个可证伪预测，并进一步提出对全部合法重分句求最坏值。",
    "不要声称首次提出句界攻击。可守住的是 SWAN 的系统特异反例，以及完整 variable-N exact 求解的工程验证。"
  );
}

// 5
{
  const slide = pptx.addSlide("BASE");
  header(slide, "整体技术架构：我负责研究闭环，Agent 负责自动化执行", "04 · 技术架构");
  const lanes = [
    ["研究者设计", C.blue, C.paleBlue, ["提出假设", "定义 threat model", "选择指标/对照", "冻结 Go/Stop"]],
    ["Agent 执行", C.cyan, C.paleCyan, ["加载数据/模型", "推理与 detector", "DP / C++ core", "日志、SHA、指标"]],
    ["研究者裁决", C.green, C.paleGreen, ["复核异常", "解释证据", "决定 Stop/转向", "限定论文主张"]],
  ];
  lanes.forEach((l, i) => {
    const x = 0.75 + i * 4.15;
    box(slide, x, 2.0, 3.65, 3.85, l[2], l[1], 1.5);
    slide.addText(l[0], { x: x + 0.3, y: 2.32, w: 3.05, h: 0.34, fontFace: FONT, fontSize: 18, bold: true, color: l[1], align: "center", margin: 0 });
    bullets(slide, l[3], x + 0.48, 3.05, 2.7, 1.9, 13);
    if (i < 2) arrow(slide, x + 3.75, 3.95, 0.3, 0, C.slate, 2);
  });
  band(slide, "Agent 是可重复、可审计的实验执行工具，不是自主完成课题研究。", C.coral, C.paleCoral);
  footer(slide);
  addNotesAndCheck(
    slide,
    "这里需要明确角色边界。Agent 不是自主完成课题，它相当于自动化实验工程工具。我负责研究假设、实验合同、指标和停止规则；Agent 负责按合同跑数据、推理、日志和统计。出现异常时，由流水线暴露证据，再由我根据冻结规则裁决。",
    "本项目大部分是推理、检测和算法计算，不是模型训练。导师问你做了什么时，强调 threat model、A--B--A、指标口径、Go/Stop 和研究方向选择。"
  );
}

// 6
{
  const slide = pptx.addSlide("BASE");
  header(slide, "实验方案设计：四组证据逐层验证同一假设", "05 · 实验设计");
  const cols = [0.75, 2.45, 6.35, 9.55, 11.45];
  const widths = [1.55, 3.75, 3.05, 1.75, 1.1];
  const headers = ["实验组", "数据与对照", "核心指标", "人负责", "Agent"];
  headers.forEach((h, i) => {
    box(slide, cols[i], 2.0, widths[i], 0.48, C.ink, C.ink, 0);
    slide.addText(h, { x: cols[i], y: 2.13, w: widths[i], h: 0.2, fontFace: FONT, fontSize: 10.5, bold: true, color: C.white, align: "center", margin: 0 });
  });
  const rows = [
    ["诱因筛选", "30 个新 C4；三臂；K=4", "Valid ASR", "分组/质量门", "推理统计"],
    ["严格因果", "同一 30 篇；A--B--A", "配对翻转、p", "标点控制", "检测重放"],
    ["机制拆分", "N 代数 + 固定图组合", "z、F1、green", "反事实口径", "计算指标"],
    ["范围对照", "SIRA / KGW / SynthID", "无反馈 ASR", "基线选择", "生成检测"],
    ["认证预检", "3×1000；radius 0--3", "FPR/TPR、GPU-h", "资源门", "profile"],
  ];
  rows.forEach((r, ri) => {
    const y = 2.48 + ri * 0.67;
    r.forEach((v, ci) => {
      box(slide, cols[ci], y, widths[ci], 0.67, ri % 2 ? C.paleGray : C.white, C.line, 0.5);
      slide.addText(v, { x: cols[ci] + 0.08, y: y + 0.18, w: widths[ci] - 0.16, h: 0.28, fontFace: FONT, fontSize: ci === 1 ? 9.8 : 10.2, bold: ci === 0, color: ci === 3 ? C.blue : ci === 4 ? C.cyan : C.ink2, align: "center", margin: 0 });
    });
  });
  band(slide, "实验逻辑和指标由我冻结；数据加载、模型推理、日志采集和指标计算由 Agent 执行。", C.blue, C.paleBlue);
  footer(slide);
  addNotesAndCheck(
    slide,
    "实验按证据强度逐层设计。先用公平三臂筛选诱因，再用标点-only A--B--A 做单变量控制，然后分别分析统计路径与匹配路径。跨水印实验用于限制主张范围。认证分支使用独立 cohort，先过 detector health，再看资源可行性。",
    "Valid ASR 同时要求质量通过和逃逸。A--B--A 是最强行为证据；代数和人工图只是机制支持。"
  );
}

// 7
{
  const slide = pptx.addSlide("BASE");
  header(slide, "实验环境与算力约束：先冻结资源门，再允许 Agent 执行", "06 · 实验环境");
  const systems = [
    ["本地", "协议 / CPU oracle / 审计", C.blue],
    ["远端 RTX 3090", "24GB · 隔离环境", C.cyan],
    ["Agent Supervisor", "checkpoint / SHA / 重放", C.green],
  ];
  systems.forEach((s, i) => {
    const x = 0.85 + i * 4.05;
    box(slide, x, 2.05, 3.45, 1.25, C.white, s[2], 1.3);
    slide.addText(s[0], { x: x + 0.25, y: 2.32, w: 2.95, h: 0.3, fontFace: FONT, fontSize: 17, bold: true, color: s[2], align: "center", margin: 0 });
    slide.addText(s[1], { x: x + 0.3, y: 2.76, w: 2.85, h: 0.25, fontFace: FONT, fontSize: 11.5, color: C.slate, align: "center", margin: 0 });
    if (i < 2) arrow(slide, x + 3.53, 2.68, 0.42, 0, C.muted, 1.5);
  });
  const limits = [
    ["200M", "单层 pairs"], ["2M", "union spans"], ["16GiB", "RSS"],
    ["72h", "双 replay"], ["2×", "确定重放"], ["0", "本轮模型训练"],
  ];
  limits.forEach((m, i) => {
    const x = 0.85 + (i % 3) * 4.05;
    const y = 3.92 + Math.floor(i / 3) * 1.0;
    metric(slide, x, y, 3.45, m[0], m[1], i === 3 ? C.coral : i === 5 ? C.gray : C.cyan);
  });
  band(slide, "冻结对象：代码、模型、数据、tokenizer、cohort identity；任一硬门失败即阻断后继任务。", C.gold, C.paleGold);
  footer(slide);
  addNotesAndCheck(
    slide,
    "本地主要做协议、oracle 和审计，远端 3090 做模型推理与特征计算。我们提前冻结代码、模型、身份和资源门。Agent 自动管理 checkpoint、日志和 SHA，并在任一硬门失败时停止后继任务。本轮没有重新训练模型。",
    "72 GPU-hours 是预注册上限，不是机器不能继续跑。双 replay 用于验证确定性和产物 identity。"
  );
}

// 8
{
  const slide = pptx.addSlide("BASE");
  header(slide, "实验结果：句界诱因得到确认，但认证证书未进入执行", "07 · 结果与分析", "以下结果由 Agent 流水线运行产生，并经人工复核");
  box(slide, 0.75, 2.02, 5.65, 3.95, C.white, C.blue, 1.4);
  slide.addText("SWAN：诱因与机制", { x: 1.05, y: 2.3, w: 5.05, h: 0.3, fontFace: FONT, fontSize: 17, bold: true, color: C.blue, align: "center", margin: 0 });
  resultBar(slide, 1.05, 3.02, "普通复述", 6.7, C.gray);
  resultBar(slide, 1.05, 3.65, "Plan E", 53.3, C.cyan);
  resultBar(slide, 1.05, 4.28, "句界重组", 100.0, C.blue);
  slide.addText("A--B--A", { x: 1.1, y: 5.08, w: 1.0, h: 0.23, fontFace: FONT, fontSize: 11, bold: true, color: C.coral, margin: 0 });
  slide.addText("1/30  →  30/30  →  1/30", { x: 2.15, y: 5.02, w: 3.4, h: 0.32, fontFace: FONT, fontSize: 17, bold: true, color: C.coral, align: "center", margin: 0 });
  slide.addText("p=3.73e-9  ·  zmax=1.732<2.33  ·  F1 0.8197→0.4502", { x: 1.1, y: 5.53, w: 4.95, h: 0.24, fontFace: FONT, fontSize: 10.5, color: C.slate, align: "center", margin: 0 });
  box(slide, 6.85, 2.02, 5.7, 3.95, C.white, C.green, 1.4);
  slide.addText("SemStamp：健康与资源", { x: 7.15, y: 2.3, w: 5.1, h: 0.3, fontFace: FONT, fontSize: 17, bold: true, color: C.green, align: "center", margin: 0 });
  metric(slide, 7.25, 2.95, 2.1, "4.7%", "test FPR", C.blue);
  metric(slide, 9.95, 2.95, 2.1, "95.4%", "positive TPR", C.green);
  metric(slide, 7.25, 4.2, 2.1, "46,501", "target solves", C.cyan);
  metric(slide, 9.95, 4.2, 2.1, "77.6152h", "保守双 replay", C.coral);
  slide.addText("77.6152 > 72：certificate 保持未启动", { x: 7.3, y: 5.48, w: 4.65, h: 0.27, fontFace: FONT, fontSize: 12, bold: true, color: C.coral, align: "center", margin: 0 });
  band(slide, "结果由流水线计算；诱因解释、证据等级和 Stop 裁决由研究者负责。", C.blue, C.paleBlue);
  footer(slide);
  addNotesAndCheck(
    slide,
    "SWAN 方面，公平比较先把句界筛出来，严格 A--B--A 又证明判定能随句界可逆翻转。统计上单单元最高 z 低于阈值，图组合诊断也显示作用域会稀释匹配。SemStamp 方面 detector health 健康，variable-N core 通过，但全量 profile 超过 72 小时，因此没有正式 certificate。",
    "必须说结果由 Agent 运行、经人工复核，不说 Agent 发现机制。图组合只是 scorer diagnostic。77.6152>72 是 readiness Stop，不代表 SemStamp 不鲁棒。"
  );
}

// 9
{
  const slide = pptx.addSlide("BASE");
  header(slide, "阶段结果图①：句界优势不依赖检测反馈", "07 · 结果与分析", "Plan G：同一 30 篇、同模型、同 K=4、同质量门");
  slide.addChart(
    pptx.ChartType.bar,
    [
      { name: "有反馈", labels: ["普通复述", "完整 Plan E", "句界重组"], values: [6.7, 53.3, 100.0] },
      { name: "无反馈", labels: ["普通复述", "完整 Plan E", "句界重组"], values: [6.7, 33.3, 96.7] },
    ],
    {
      x: 0.75, y: 1.95, w: 7.45, h: 4.15,
      catAxisLabelFontFace: FONT, catAxisLabelFontSize: 11,
      valAxisLabelFontFace: FONT, valAxisLabelFontSize: 9,
      valAxisMinVal: 0, valAxisMaxVal: 110, valAxisMajorUnit: 20,
      valGridLine: { color: C.line, width: 1 },
      showLegend: true, legendPos: "b",
      legendFontFace: FONT, legendFontSize: 10,
      showValue: true, showTitle: false,
      dataLabelPosition: "outEnd", dataLabelColor: C.ink2,
      dataLabelFormatCode: "0.0",
      chartColors: [C.coral, C.blue],
      showBorder: false,
      showCatName: false,
      showValAxisTitle: false,
      showCatAxisTitle: false,
    }
  );
  box(slide, 8.7, 2.08, 3.55, 3.62, C.paleBlue, C.blue, 1.3);
  slide.addText("读图结论", { x: 9.02, y: 2.42, w: 2.9, h: 0.3, fontFace: FONT, fontSize: 17, bold: true, color: C.blue, align: "center", margin: 0 });
  metric(slide, 9.28, 3.02, 2.4, "29/30", "句界无反馈 Valid ASR", C.blue);
  bullets(slide, [
    "复杂 Plan E 在无反馈时降至 33.3%",
    "句界无反馈仍为 96.7%",
    "诱因来自操作本身，不依赖最低分筛选",
  ], 9.02, 4.25, 2.9, 1.15, 10.5);
  band(slide, "该图完成“诱因筛选”，严格单变量因果仍由下一步 A--B--A 提供。", C.blue, C.paleBlue);
  footer(slide);
  addNotesAndCheck(
    slide,
    "这张图只比较同一批 30 篇、相同预算的实验。普通复述无论是否反馈都是 6.7%；完整 Plan E 从 53.3% 降到 33.3%；句界重组即使不读检测分数仍为 96.7%。所以主要效果不是来自最低分筛选，而是句界操作本身。",
    "这是公平方法比较，不是严格因果。生成式句界候选仍可能伴随词汇变化，因此下一张核心行为证据仍是标点-only A--B--A。"
  );
}

{
  const slide = pptx.addSlide("BASE");
  header(slide, "阶段结果图②：统计路径与匹配路径同时得到支持", "07 · 结果与分析", "两条证据强度不同：公式是确定解释，人工图是 scorer diagnostic");
  box(slide, 0.78, 2.0, 5.2, 3.95, C.white, C.coral, 1.3);
  slide.addText("统计路径", { x: 1.08, y: 2.32, w: 4.6, h: 0.3, fontFace: FONT, fontSize: 17, bold: true, color: C.coral, align: "center", margin: 0 });
  slide.addText("N=1 时的 z", { x: 1.15, y: 3.0, w: 1.35, h: 0.25, fontFace: FONT, fontSize: 11.5, color: C.ink2, margin: 0 });
  box(slide, 2.65, 3.02, 2.45, 0.32, C.paleGray, C.paleGray, 0);
  slide.addShape(pptx.ShapeType.rect, {
    x: 2.65, y: 3.02, w: 2.45 * 1.732 / 2.5, h: 0.32,
    fill: { color: C.cyan }, line: { color: C.cyan },
  });
  const thresholdX = 2.65 + 2.45 * 2.33 / 2.5;
  slide.addShape(pptx.ShapeType.line, {
    x: thresholdX, y: 2.82, w: 0, h: 0.72,
    line: { color: C.coral, width: 2, dash: "dash" },
  });
  slide.addText("阈值 2.33", { x: thresholdX - 0.42, y: 2.72, w: 0.84, h: 0.16, fontFace: FONT, fontSize: 8.5, bold: true, color: C.coral, align: "center", margin: 0 });
  slide.addText("zmax=1.732", { x: 2.72, y: 3.48, w: 1.6, h: 0.25, fontFace: FONT, fontSize: 12, bold: true, color: C.cyan, margin: 0 });
  metric(slide, 1.15, 4.25, 1.8, "21/30", "只替换 N", C.coral);
  metric(slide, 3.55, 4.25, 1.8, "30/30", "只替换 green-rate", C.cyan);
  box(slide, 6.45, 2.0, 5.85, 3.95, C.white, C.cyan, 1.3);
  slide.addText("匹配路径：S2Match 中位 F1", { x: 6.78, y: 2.32, w: 5.2, h: 0.3, fontFace: FONT, fontSize: 17, bold: true, color: C.cyan, align: "center", margin: 0 });
  const f1 = [["原句级图", 0.8197, C.blue], ["人工组合图", 0.4502, C.coral], ["真实合句图", 0.5200, C.cyan]];
  f1.forEach((d, i) => {
    const x = 6.95 + i * 1.7;
    slide.addShape(pptx.ShapeType.rect, {
      x, y: 5.35 - d[1] * 2.65, w: 1.05, h: d[1] * 2.65,
      fill: { color: d[2] }, line: { color: d[2] },
    });
    slide.addText(d[1].toFixed(4), { x: x - 0.12, y: 5.04 - d[1] * 2.65, w: 1.29, h: 0.23, fontFace: FONT, fontSize: 10.5, bold: true, color: d[2], align: "center", margin: 0 });
    slide.addText(d[0], { x: x - 0.23, y: 5.48, w: 1.5, h: 0.28, fontFace: FONT, fontSize: 9.5, color: C.ink2, align: "center", margin: 0 });
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 6.75, y: 5.35 - 0.65 * 2.65, w: 5.15, h: 0,
    line: { color: C.gold, width: 1.3, dash: "dash" },
  });
  slide.addText("green threshold 0.65", { x: 10.15, y: 3.37, w: 1.7, h: 0.2, fontFace: FONT, fontSize: 8.5, color: C.gold, align: "right", margin: 0 });
  band(slide, "统计路径解释 N 效应；图结果只证明作用域敏感，不能排除真实 parser drift。", C.gold, C.paleGold);
  footer(slide);
  addNotesAndCheck(
    slide,
    "左侧是确定的统计解释：N 等于 1 时，即使唯一单元命中，z 最高也只有 1.732，低于 2.33。右侧是匹配诊断：句级图中位 F1 是 0.8197，人工组合后降到 0.4502，真实合句重解析是 0.5200。",
    "21/30 和 30/30 只是代数充分性，不是文本因果比例。人工组合图不是 gold AMR，所以只能说明 scorer 对图作用域敏感。"
  );
}

{
  const slide = pptx.addSlide("BASE");
  header(slide, "阶段结果图③：认证链最终停在资源门，而不是算法错误", "07 · 结果与分析", "三次 score-blind 预算均未进入正式 certificate");
  const x0 = 2.45, scaleW = 8.7, max = 90;
  const rows = [
    ["旧链 v18", 86.65, C.coral],
    ["v3 公式 A", 76.5346, C.cyan],
    ["v3 公式 B", 77.6152, C.blue],
  ];
  const gateX = x0 + scaleW * 72 / max;
  slide.addShape(pptx.ShapeType.line, {
    x: gateX, y: 2.05, w: 0, h: 3.72,
    line: { color: C.coral, width: 2, dash: "dash" },
  });
  slide.addText("72 h 硬门", { x: gateX - 0.46, y: 1.79, w: 0.92, h: 0.2, fontFace: FONT, fontSize: 9, bold: true, color: C.coral, align: "center", margin: 0 });
  rows.forEach((r, i) => {
    const y = 2.48 + i * 1.05;
    slide.addText(r[0], { x: 0.85, y: y + 0.06, w: 1.3, h: 0.25, fontFace: FONT, fontSize: 12, bold: true, color: C.ink2, margin: 0 });
    box(slide, x0, y, scaleW, 0.42, C.paleGray, C.paleGray, 0);
    slide.addShape(pptx.ShapeType.rect, {
      x: x0, y, w: scaleW * r[1] / max, h: 0.42,
      fill: { color: r[2] }, line: { color: r[2] },
    });
    slide.addText(`${r[1]} h`, { x: 11.35, y: y + 0.05, w: 0.85, h: 0.24, fontFace: FONT, fontSize: 10.5, bold: true, color: r[2], align: "right", margin: 0 });
  });
  const chain = [
    ["detector health", "GO", C.green],
    ["exact core", "GO", C.green],
    ["full profile", "STOP", C.coral],
    ["certificate", "未启动", C.gray],
  ];
  chain.forEach((d, i) => {
    const x = 1.0 + i * 3.0;
    box(slide, x, 5.35, 2.5, 0.6, C.white, d[2], 1.1);
    slide.addText(d[0], { x: x + 0.15, y: 5.53, w: 1.3, h: 0.2, fontFace: FONT, fontSize: 10.2, bold: true, color: C.ink2, margin: 0 });
    slide.addText(d[1], { x: x + 1.65, y: 5.51, w: 0.62, h: 0.2, fontFace: FONT, fontSize: 9.2, bold: true, color: d[2], align: "right", margin: 0 });
  });
  band(slide, "资源 Stop 只说明冻结实现与 RTX 3090 预算不足，不提供 SemStamp 鲁棒性结论。", C.coral, C.paleCoral);
  footer(slide);
  addNotesAndCheck(
    slide,
    "资源图显示旧链 v18 外推 86.65 小时，新 v3 两条冻结公式分别是 76.5346 和 77.6152 小时，全部超过 72 小时门。前置 detector health 和 exact core 已经通过，但 full profile Stop 后 certificate 保持未启动。",
    "不要说只差一点所以基本成功。资源门是预注册合同；没有 certificate 就不能报告 certified TPR。"
  );
}

// 12
{
  const slide = pptx.addSlide("BASE");
  header(slide, "当前问题与异常：数据、方法、实现和资源需要分开解释", "08 · 问题与异常");
  const issues = [
    ["外部有效性", "核心 A--B--A 仅 30 篇，平均 30.9 词", C.gold],
    ["正常长度", "23/30 可合并，但 0/30 有自然拆分点", C.coral],
    ["防御证据", "局部证据、经验尾、motif 均未过门", C.coral],
    ["renderer", "v1 有 42/3000 句界落入 atom", C.cyan],
    ["runner", "v2 WP4 误用 fixed-N solver", C.cyan],
    ["吞吐语义", "batch128 改变冻结 region array", C.cyan],
    ["资源", "77.6152h，超过硬门 5.6152h", C.coral],
  ];
  issues.forEach((d, i) => {
    const y = 1.95 + i * 0.61;
    box(slide, 0.82, y, 11.65, 0.48, i % 2 ? C.paleGray : C.white, C.line, 0.5);
    slide.addText(d[0], { x: 1.05, y: y + 0.13, w: 1.55, h: 0.2, fontFace: FONT, fontSize: 11.5, bold: true, color: d[2], margin: 0 });
    slide.addText(d[1], { x: 2.85, y: y + 0.13, w: 8.7, h: 0.2, fontFace: FONT, fontSize: 11, color: C.ink2, margin: 0 });
    slide.addShape(pptx.ShapeType.ellipse, { x: 11.78, y: y + 0.14, w: 0.17, h: 0.17, fill: { color: d[2] }, line: { color: d[2] } });
  });
  band(slide, "自动化不等于不会出错；价值在于异常可追踪、可封存、可按合同停止。", C.coral, C.paleCoral);
  footer(slide);
  addNotesAndCheck(
    slide,
    "当前问题不能只归结为算力。核心因果结果仍是短文本，正常长度对照在构造阶段失败，SWAN 防御也没通过。流水线还抓到了 renderer、solver 调用和 batch 语义漂移三个实现问题。最后才是资源超限。",
    "42 条冲突后来修复，但 v1 Stop 保留；fixed-N 的 83 行前缀已封存；batch128 虽快 1.0333 倍，但不能用于正式证书。"
  );
}

// 10
{
  const slide = pptx.addSlide("BASE");
  header(slide, "下一阶段计划：先做研究决策，再决定是否重启实验", "09 · 下一阶段");
  const tracks = [
    ["近期必做", C.blue, C.paleBlue, ["固定 SWAN 系统审计主线", "完善 claim–evidence matrix", "整理图表、局限与复现包"]],
    ["条件性重启", C.gold, C.paleGold, ["正常长度需新独立资产 + 真人质量门", "Certificate 需另立 v4", "要求 batch64 等价且预算至少降低 7.8%"]],
    ["禁止操作", C.coral, C.paleCoral, ["不删超限样本", "不缩半径或 target layer", "不事后提高资源门"]],
  ];
  tracks.forEach((t, i) => {
    const x = 0.75 + i * 4.15;
    box(slide, x, 2.0, 3.65, 3.85, t[2], t[1], 1.4);
    slide.addText(t[0], { x: x + 0.3, y: 2.34, w: 3.05, h: 0.32, fontFace: FONT, fontSize: 18, bold: true, color: t[1], align: "center", margin: 0 });
    bullets(slide, t[3], x + 0.46, 3.05, 2.75, 2.0, 12.5);
  });
  band(slide, "我决定研究是否值得重启并冻结新协议；Agent 只在授权后执行。", C.green, C.paleGreen);
  footer(slide);
  addNotesAndCheck(
    slide,
    "下一步首先不是继续跑，而是确定论文是否收敛为 SWAN 系统审计。正常长度端点只有在能构造全新、独立、真人质量合格资产时才值得重启。Certificate 若继续，必须另立版本，在不改变语义和 threat set 的前提下降低至少约 7.8% 预算。",
    "不要主动承诺马上重启 GPU 长跑。7.8% 来自资源缺口；先让导师判断论文主线与继续投入价值。"
  );
}

// 11
{
  const slide = pptx.addSlide("BASE");
  header(slide, "总结：把“低分现象”收敛为可审计证据", "10 · 总结");
  const points = [
    ["诱因", "句界改变 detector unitization", C.blue],
    ["行为证据", "固定词汇/词序，1/30→30/30→1/30", C.coral],
    ["机制支持", "统计上限下降 + 图作用域匹配稀释", C.cyan],
    ["主张边界", "短文本 SWAN 审计，不是通用攻击", C.gold],
    ["防御终态", "detector/core 已验证，certificate 未启动", C.green],
    ["人机边界", "人负责研究设计；Agent 负责自动执行与留痕", C.ink2],
  ];
  points.forEach((p, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.82 + col * 6.18, y = 2.02 + row * 1.28;
    box(slide, x, y, 5.65, 0.98, C.white, p[2], 1.2);
    slide.addText(p[0], { x: x + 0.28, y: y + 0.27, w: 1.2, h: 0.25, fontFace: FONT, fontSize: 13, bold: true, color: p[2], margin: 0 });
    slide.addText(p[1], { x: x + 1.55, y: y + 0.25, w: 3.72, h: 0.36, fontFace: FONT, fontSize: 12.2, bold: true, color: C.ink2, margin: 0 });
  });
  band(slide, "最稳妥的论文定位：SWAN 句界接口审计；认证防御是独立且尚未产出 certificate 的证据流。", C.blue, C.paleBlue);
  footer(slide);
  addNotesAndCheck(
    slide,
    "我们把一个反馈搜索中的低分现象收敛成了句界诱因，并通过 A--B--A、统计和匹配诊断建立分层证据。当前能守住的是范围受限的 SWAN 系统审计，不是通用攻击，也不是成熟防御。Agent 提高了执行和复现效率，但研究设计、异常解释和最终裁决仍由研究者负责。",
    "最大贡献的标准回答：固定词汇与词序下，可逆句界干预明确定位了 SWAN detector 对 unitization 的依赖。"
  );
}

// 12
{
  const slide = pptx.addSlide("BASE");
  slide.background = { color: C.ink };
  slide.addText("Q&A", {
    x: 0.8, y: 0.8, w: 3.0, h: 0.7,
    fontFace: FONT, fontSize: 37, bold: true, color: C.white, margin: 0,
  });
  slide.addText("建议讨论的三个研究决策", {
    x: 0.84, y: 1.72, w: 4.5, h: 0.34,
    fontFace: FONT, fontSize: 15, color: "CBD5E1", margin: 0,
  });
  const qs = [
    ["01", "论文是否固定为 SWAN 句界接口审计？", C.blue],
    ["02", "正常长度端点是否值得重建独立人工质量资产？", C.gold],
    ["03", "SemStamp v4 是否值得为 7.8% 资源缺口继续投入？", C.green],
  ];
  qs.forEach((q, i) => {
    const y = 2.48 + i * 1.12;
    box(slide, 0.85, y, 8.1, 0.78, "163B51", q[2], 1.2);
    slide.addText(q[0], { x: 1.1, y: y + 0.2, w: 0.55, h: 0.25, fontFace: FONT, fontSize: 12, bold: true, color: q[2], margin: 0 });
    slide.addText(q[1], { x: 1.8, y: y + 0.18, w: 6.65, h: 0.3, fontFace: FONT, fontSize: 14, bold: true, color: C.white, margin: 0 });
  });
  box(slide, 9.7, 2.48, 2.45, 3.02, "163B51", C.cyan, 1.2);
  slide.addText("备查材料", { x: 10.05, y: 2.85, w: 1.75, h: 0.3, fontFace: FONT, fontSize: 16, bold: true, color: C.cyan, align: "center", margin: 0 });
  slide.addText("37 页完整时间线\n\n逐页讲稿\n\n所有 Go / Stop 与 SHA 链", {
    x: 10.02, y: 3.48, w: 1.82, h: 1.35,
    fontFace: FONT, fontSize: 12, color: "D8E2EA", align: "center", margin: 0,
  });
  slide.addText("谢谢", { x: 0.85, y: 6.48, w: 1.2, h: 0.3, fontFace: FONT, fontSize: 16, bold: true, color: C.white, margin: 0 });
  addNotesAndCheck(
    slide,
    "我的建议是先把论文主线固定下来，再决定是否继续投入正常长度或证书工程。请老师和同门重点帮我判断：现有系统审计证据是否足以形成稳定论文主线，以及两个未闭环端点是否还有继续投入的价值。",
    "这是主动请导师做研究取舍，不是把未完成任务甩给导师。若问为什么不直接继续：两个缺口分别受资产可构造性和严格资源门限制，继续前应先验证预期收益。"
  );
}

async function main() {
  await pptx.writeFile({ fileName: "lab_update_10min_human_agent_20261010.pptx" });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
