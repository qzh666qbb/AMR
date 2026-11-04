const fs = require("fs");
const pptxgen = require("pptxgenjs");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");
const { autoFontSize } = require("./pptxgenjs_helpers/text");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "quzehao";
pptx.title = "完整实验演进：从 AMR 攻击搜索到句界诱因与认证防御";
pptx.subject = "Plan A--AE detailed research progress";
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
  line: "D9E2EC", paper: "F7F9FC", white: "FFFFFF",
  blue: "2457D6", paleBlue: "EAF0FF",
  cyan: "1F9FB5", paleCyan: "E5F7FA",
  coral: "E4574F", paleCoral: "FDECE9",
  gold: "D69E2E", paleGold: "FFF7DF",
  green: "2E8B72", paleGreen: "E7F4EF",
  gray: "718096", paleGray: "EDF2F7",
};
const CHAPTER = {
  overview: ["总览", C.ink],
  search: ["攻击搜索 · Plan A--F", C.gray],
  causal: ["诱因与因果 · Plan G--M", C.blue],
  normal: ["正常长度端点 · Plan N--P", C.gold],
  cert: ["认证证据流 · Plan Q--X", C.cyan],
  defense: ["认证防御 · Plan Y--AE", C.green],
  close: ["终态", C.coral],
};
const notes = [];

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

function footer(slide, text = "来源：叙事.md；当前主张以 CLAIM_EVIDENCE_MATRIX.md 为准。") {
  slide.addText(text, {
    x: 0.65, y: 7.07, w: 11.55, h: 0.17,
    fontFace: FONT, fontSize: 8.2, color: C.muted, margin: 0,
  });
}

function header(slide, title, chapter = "overview", subtitle = "") {
  const [label, color] = CHAPTER[chapter];
  slide.addText(label, {
    x: 0.65, y: 0.2, w: 4.2, h: 0.22,
    fontFace: FONT, fontSize: 10, bold: true, color, charSpacing: 1.1, margin: 0,
  });
  slide.addText(title, autoFontSize(title, FONT, {
    x: 0.65, y: 0.53, w: 12.0, h: 0.72,
    mode: "shrink", fontSize: 33, minFontSize: 25, maxFontSize: 33,
    bold: true, color: C.ink, margin: 0, breakLine: false,
  }));
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.67, y: 1.28, w: 11.8, h: 0.3,
      fontFace: FONT, fontSize: 13, color: C.slate, margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.65, y: 1.68, w: 12.0, h: 0,
    line: { color: C.line, width: 1 },
  });
}

function status(slide, x, y, text, kind = "stop", w = 1.02) {
  const map = {
    go: [C.green, C.paleGreen],
    stop: [C.coral, C.paleCoral],
    audit: [C.gold, C.paleGold],
    historical: [C.gray, C.paleGray],
  };
  const [color, fill] = map[kind];
  box(slide, x, y, w, 0.3, fill, color, 1);
  slide.addText(text, {
    x, y: y + 0.04, w, h: 0.17,
    fontFace: FONT, fontSize: 9, bold: true, color, align: "center", margin: 0,
  });
}

function bullets(slide, items, x, y, w, h, size = 14, color = C.ink2) {
  const runs = [];
  items.forEach((item, index) => {
    runs.push({
      text: item,
      options: {
        bullet: { indent: size * 1.1 },
        hanging: size * 0.3,
        breakLine: index < items.length - 1,
      },
    });
  });
  slide.addText(runs, {
    x, y, w, h,
    fontFace: FONT, fontSize: size, color,
    margin: 0.04, breakLine: false, paraSpaceAfterPt: 10,
    valign: "top",
  });
}

function metric(slide, x, y, w, value, label, color) {
  box(slide, x, y, w, 0.82, C.white, color, 1.2);
  slide.addText(value, autoFontSize(value, FONT, {
    x: x + 0.12, y: y + 0.1, w: w - 0.24, h: 0.31,
    mode: "shrink", fontSize: 22, minFontSize: 15, maxFontSize: 22,
    bold: true, color, align: "center", margin: 0,
  }));
  slide.addText(label, {
    x: x + 0.12, y: y + 0.53, w: w - 0.24, h: 0.18,
    fontFace: FONT, fontSize: 9.5, color: C.slate, align: "center", margin: 0,
  });
}

function decisionBand(slide, text, kind = "stop") {
  const map = {
    go: [C.green, C.paleGreen],
    stop: [C.coral, C.paleCoral],
    audit: [C.gold, C.paleGold],
    historical: [C.gray, C.paleGray],
  };
  const [color, fill] = map[kind];
  box(slide, 0.8, 6.28, 11.72, 0.52, fill, color, 1.1);
  slide.addText(text, {
    x: 1.05, y: 6.43, w: 11.2, h: 0.22,
    fontFace: FONT, fontSize: 12.2, bold: true, color, align: "center", margin: 0,
  });
}

function finalize(slide, title, note) {
  slide.addNotes(note);
  notes.push({ title, note });
  warnIfSlideHasOverlaps(slide, pptx, {
    muteContainment: true,
    ignoreLines: true,
    ignoreDecorativeShapes: true,
  });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function stageSlide(d) {
  const slide = pptx.addSlide("BASE");
  header(slide, `${d.plan}｜${d.title}`, d.chapter, d.subtitle || d.question);
  const color = CHAPTER[d.chapter][1];
  slide.addText("我们做了什么", {
    x: 0.82, y: 1.95, w: 2.0, h: 0.3,
    fontFace: FONT, fontSize: 16, bold: true, color, margin: 0,
  });
  bullets(slide, d.actions, 0.9, 2.42, 6.05, 2.82, d.actions.length > 3 ? 12.3 : 13.5);
  slide.addShape(pptx.ShapeType.line, {
    x: 7.18, y: 2.0, w: 0, h: 3.8,
    line: { color: C.line, width: 1 },
  });
  slide.addText("关键证据", {
    x: 7.55, y: 1.95, w: 1.8, h: 0.3,
    fontFace: FONT, fontSize: 16, bold: true, color, margin: 0,
  });
  const metrics = d.metrics.slice(0, 6);
  const cols = metrics.length > 3 ? 2 : 1;
  const mw = cols === 2 ? 2.15 : 4.55;
  metrics.forEach((m, i) => {
    const col = cols === 2 ? i % 2 : 0;
    const row = cols === 2 ? Math.floor(i / 2) : i;
    metric(slide, 7.55 + col * 2.4, 2.42 + row * 1.02, mw, m[0], m[1], m[2] || color);
  });
  if (d.limit) {
    box(slide, 7.55, 5.42, 4.78, 0.52, C.paleGray, C.line, 0.8);
    slide.addText(d.limit, {
      x: 7.82, y: 5.54, w: 4.24, h: 0.27,
      fontFace: FONT, fontSize: 9.8, color: C.slate,
      align: "center", valign: "mid", margin: 0,
    });
  }
  decisionBand(slide, d.decision, d.kind || "stop");
  footer(slide);
  finalize(slide, `${d.plan}｜${d.title}`, d.note);
}

function bar(slide, x, y, w, label, value, max, color, suffix = "%") {
  slide.addText(label, {
    x, y: y - 0.02, w: 1.8, h: 0.25,
    fontFace: FONT, fontSize: 11.5, color: C.ink2, margin: 0,
  });
  if (value < max) box(slide, x + 1.9, y, w, 0.24, C.paleGray, C.paleGray, 0);
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 1.9, y, w: Math.max(0.02, w * value / max), h: 0.24,
    fill: { color }, line: { color },
  });
  slide.addText(`${value}${suffix}`, {
    x: x + 1.9 + w - 0.75, y: y - 0.3, w: 0.7, h: 0.2,
    fontFace: FONT, fontSize: 9.3, bold: true, color, align: "right", margin: 0,
  });
}

// 1. Title
{
  const slide = pptx.addSlide("BASE");
  slide.background = { color: C.ink };
  slide.addText("完整实验演进详细版", {
    x: 0.78, y: 0.56, w: 3.5, h: 0.28,
    fontFace: FONT, fontSize: 11, bold: true, color: "75D5DE", charSpacing: 1.5, margin: 0,
  });
  slide.addText("从 AMR 攻击搜索\n到句界诱因与认证防御", {
    x: 0.78, y: 1.28, w: 7.2, h: 1.7,
    fontFace: FONT, fontSize: 34, bold: true, color: C.white, margin: 0,
  });
  slide.addText("Plan A--AE：逐阶段解释问题、实验、关键数据、Go/Stop 与主张边界", {
    x: 0.82, y: 3.32, w: 6.6, h: 0.55,
    fontFace: FONT, fontSize: 16, color: "CBD5E1", margin: 0,
  });
  const phases = [
    ["A--F", "攻击搜索", C.gray], ["G--M", "诱因与因果", C.blue],
    ["N--P", "正常长度", C.gold], ["Q--X", "认证工程", C.cyan],
    ["Y--AE", "认证防御", C.green],
  ];
  phases.forEach((p, i) => {
    const y = 1.2 + i * 1.02;
    box(slide, 8.5, y, 3.65, 0.68, "163B51", p[2], 1.4);
    slide.addText(p[0], {
      x: 8.78, y: y + 0.17, w: 0.78, h: 0.23,
      fontFace: FONT, fontSize: 12, bold: true, color: p[2], margin: 0,
    });
    slide.addText(p[1], {
      x: 9.78, y: y + 0.15, w: 1.95, h: 0.25,
      fontFace: FONT, fontSize: 13, bold: true, color: C.white, margin: 0,
    });
  });
  slide.addText("quzehao  ·  2026.10.10", {
    x: 0.8, y: 6.76, w: 3.0, h: 0.2,
    fontFace: FONT, fontSize: 10, color: C.muted, margin: 0,
  });
  finalize(slide, "完整实验演进", "这是一份详细版。开场先说明：我不会把每个 Plan 当作独立贡献，而是按五个阶段解释研究问题如何收敛。最重要的行为结果仍是 SWAN 的短文本 A--B--A；后面的 certificate 工作是独立证据流。");
}

// 2. Executive summary
{
  const slide = pptx.addSlide("BASE");
  header(slide, "先给结论：我们得到的是三个互不替代的终态", "overview");
  const cols = [
    ["SWAN 审计", "短文本句界依赖成立", "正常长度句数守恒未确认", C.blue, C.paleBlue],
    ["旧认证链", "detector health 与 exact DP 成立", "v18 certificate 未启动", C.cyan, C.paleCyan],
    ["新防御链", "renderer 与完整 variable-N core 成立", "77.6152 GPU-h 资源 Stop", C.green, C.paleGreen],
  ];
  cols.forEach((d, i) => {
    const x = 0.78 + i * 4.12;
    box(slide, x, 2.02, 3.7, 3.95, d[4], d[3], 1.5);
    slide.addText(d[0], {
      x: x + 0.25, y: 2.35, w: 3.2, h: 0.32,
      fontFace: FONT, fontSize: 18, bold: true, color: d[3], align: "center", margin: 0,
    });
    slide.addText("已建立", {
      x: x + 0.3, y: 3.05, w: 0.8, h: 0.22,
      fontFace: FONT, fontSize: 10, bold: true, color: C.green, margin: 0,
    });
    slide.addText(d[1], {
      x: x + 0.3, y: 3.4, w: 3.1, h: 0.65,
      fontFace: FONT, fontSize: 13.5, bold: true, color: C.ink, align: "center", margin: 0,
    });
    slide.addText("未建立", {
      x: x + 0.3, y: 4.42, w: 0.8, h: 0.22,
      fontFace: FONT, fontSize: 10, bold: true, color: C.coral, margin: 0,
    });
    slide.addText(d[2], {
      x: x + 0.3, y: 4.77, w: 3.1, h: 0.65,
      fontFace: FONT, fontSize: 13, color: C.ink2, align: "center", margin: 0,
    });
  });
  decisionBand(slide, "整套工作不是一个连续的“成功故事”，而是三条保留各自 Stop 的证据链。", "audit");
  footer(slide);
  finalize(slide, "三个终态", "先把终态说清，后面细节才不会被误解。SWAN 的行为主张、旧 certified-resegmentation 的软件证据、新 SemStamp defense 的资源审计不能互相补证，也不能把资源 Stop 写成鲁棒性结果。");
}

// 3. Roadmap
{
  const slide = pptx.addSlide("BASE");
  header(slide, "研究问题经历了五次收敛", "overview", "从“找最强攻击”逐步变成“审计检测接口，再形式化认证问题”");
  const phases = [
    ["A--F", "搜索攻击面", "AMR 算子、反馈搜索、指标纠偏", C.gray],
    ["G--M", "定位诱因", "三臂、A--B--A、机制拆分、创新审计", C.blue],
    ["N--P", "补正常长度", "两条生成路线与定向正例工程", C.gold],
    ["Q--X", "建立认证链", "exact DP、官方接口、健康门、资源 Stop", C.cyan],
    ["Y--AE", "尝试认证防御", "refined atoms、variable-N core、全量 profile", C.green],
  ];
  phases.forEach((p, i) => {
    const x = 0.73 + i * 2.5;
    slide.addShape(pptx.ShapeType.ellipse, {
      x: x + 0.84, y: 2.05, w: 0.34, h: 0.34,
      fill: { color: p[3] }, line: { color: p[3] },
    });
    if (i < phases.length - 1) {
      slide.addShape(pptx.ShapeType.line, {
        x: x + 1.18, y: 2.22, w: 2.16, h: 0,
        line: { color: C.line, width: 2 },
      });
    }
    box(slide, x, 2.72, 2.18, 2.55, C.white, p[3], 1.4);
    slide.addText(p[0], {
      x: x + 0.2, y: 3.0, w: 1.78, h: 0.3,
      fontFace: FONT, fontSize: 16, bold: true, color: p[3], align: "center", margin: 0,
    });
    slide.addText(p[1], {
      x: x + 0.18, y: 3.48, w: 1.82, h: 0.32,
      fontFace: FONT, fontSize: 15, bold: true, color: C.ink, align: "center", margin: 0,
    });
    slide.addText(p[2], {
      x: x + 0.22, y: 4.05, w: 1.74, h: 0.72,
      fontFace: FONT, fontSize: 10.5, color: C.slate, align: "center", valign: "mid", margin: 0,
    });
  });
  decisionBand(slide, "每次转向都由前一阶段的失败模式触发，而不是为了维持一个预设结论。", "historical");
  footer(slide);
  finalize(slide, "五次收敛", "这页是全稿导航。老师如果时间有限，可以只讲到 Plan M，再用两页概括认证工程；如果要完整汇报，就按 A--AE 顺序展开。");
}

// 4. Contract
{
  const slide = pptx.addSlide("BASE");
  header(slide, "先统一对象：句界进入 SWAN 检测合同的两个位置", "overview");
  const nodes = [
    ["文本", "词汇 / 词序"], ["分句", "决定 N"], ["BART AMR", "raw R(s)"],
    ["模板化", "T(R(s))"], ["S2Match", "固定 Bank B"], ["聚合", "G, N → z"],
  ];
  nodes.forEach((n, i) => {
    const x = 0.72 + i * 2.08;
    box(slide, x, 2.38, 1.52, 1.1, i === 1 ? C.paleCoral : C.white, i === 1 ? C.coral : C.line, 1.2);
    slide.addText(n[0], { x: x + 0.1, y: 2.62, w: 1.32, h: 0.25, fontFace: FONT, fontSize: 14, bold: true, color: i === 1 ? C.coral : C.ink, align: "center", margin: 0 });
    slide.addText(n[1], { x: x + 0.08, y: 3.03, w: 1.36, h: 0.2, fontFace: FONT, fontSize: 9.5, color: C.slate, align: "center", margin: 0 });
    if (i < nodes.length - 1) arrow(slide, x + 1.58, 2.93, 0.4, 0, C.muted, 1.2);
  });
  slide.addShape(pptx.ShapeType.line, { x: 2.28, y: 3.75, w: 0, h: 1.0, line: { color: C.coral, width: 2 } });
  slide.addShape(pptx.ShapeType.line, { x: 2.28, y: 4.75, w: 8.78, h: 0, line: { color: C.coral, width: 2 } });
  arrow(slide, 11.06, 4.75, 0, -0.92, C.coral, 2);
  slide.addText("统计路径：N 变化直接改变 z 的上限", { x: 2.6, y: 4.08, w: 3.8, h: 0.3, fontFace: FONT, fontSize: 15, bold: true, color: C.coral, margin: 0 });
  slide.addShape(pptx.ShapeType.line, { x: 2.28, y: 5.35, w: 5.48, h: 0, line: { color: C.cyan, width: 2 } });
  arrow(slide, 7.76, 5.35, 0, -1.42, C.cyan, 2);
  slide.addText("表示路径：候选图作用域变化影响匹配与 G", { x: 2.6, y: 5.66, w: 5.3, h: 0.3, fontFace: FONT, fontSize: 15, bold: true, color: C.cyan, margin: 0 });
  decisionBand(slide, "模板库 B 始终固定；后续不再使用“改变了水印 AMR”这一混淆表述。", "audit");
  footer(slide);
  finalize(slide, "SWAN 检测合同", "整个叙事的对象必须先统一。我们改变的不是秘密模板库 B，而是固定模板库之外的输入分割、候选图表示和统计聚合。句界有统计路径和表示路径，后面的对照实验分别审计。");
}

const stages = [
  {
    plan: "Plan A", title: "建立本地基线并做第一轮表层探索", chapter: "search",
    question: "SWAN 在本地资产上是否可复现，普通改写是否已经足够？",
    actions: ["在 250 段 RealNews 上运行本地 SWAN 管线。", "比较普通复述、句界变换和 AMR 引导改写。", "记录 AUROC、TPR 与历史有效成功率，作为后续探索起点。"],
    metrics: [["0.985", "AUROC", C.blue], ["89.2%", "TPR@1% FPR", C.cyan], ["94.4%", "TPR@5% FPR", C.green], ["6.7%", "普通复述历史成功率", C.gray], ["30.9%", "句界变换历史成功率", C.gold], ["21.1%", "AMR 引导历史成功率", C.coral]],
    limit: "这些是当时配置的复现实证；三种攻击的探索分母和质量口径不同，不能直接横比。",
    decision: "基线健康，普通改写较弱；进入图层操作探索。", kind: "historical",
    note: "Plan A 先确认本地管线有较高检测性能，再看三类改写。句界变换已经显得更强，但当时分母和质量口径不统一，所以它只是诱因线索，不是确认结果。",
  },
  {
    plan: "Plan B", title: "直接修改 AMR 图，但收益被规范化与语义漂移抵消", chapter: "search",
    actions: ["B1：节点同义替换、边角色反向、序列化顺序重排。", "B2：把 modifier 子图重挂到根节点。", "B3：串联节点替换与 modifier 重挂，再经 AMR-to-Text 与 BART 重解析。", "用序列化顺序重排作为负对照，验证只改书写顺序不会生效。"],
    metrics: [["3.148", "组合攻击平均 z", C.coral], ["20.0%", "z<2.33", C.coral], ["0.982", "攻击后 AUROC", C.blue]],
    limit: "浅层修改常被 BART 规范化抹平；强修改会造成实体混淆和关系漂移。",
    decision: "简单节点/边/子图操作不足，转向更强但受约束的算子。", kind: "historical",
    note: "Plan B 的核心直觉是直接改图应比改文本更有效。结果相反：弱修改在重解析时被抹平，强修改虽然能降分，却破坏事实。这里第一次暴露攻击效果与语义质量的冲突。",
  },
  {
    plan: "Plan C", title: "增强 AMR 算子后，攻击与语义保真出现尖锐权衡", chapter: "search",
    actions: ["C1：只交换 :mod 子图，语义风险低但覆盖率很低。", "C2：扩展到 location/time/quant，覆盖增加但类型错误出现。", "C3：交换 ARG1/ARG2，覆盖高但容易颠倒施受事。", "C4：合并修饰与核心论元操作，并用 Pareto 排序平衡分数与质量。"],
    metrics: [["5/30", "C1 可执行句", C.gray], ["23%", "C4 低分率", C.coral], ["46%", "语义质量通过率", C.gold], ["0.974", "攻击后 AUROC", C.blue]],
    limit: "算子越强，越容易改动事件核心角色；低分不等于有效攻击。",
    decision: "手工强算子难同时满足降分与保真，转向重入拓扑。", kind: "historical",
    note: "Plan C 不是单纯加大扰动，而是系统比较不同语义风险的算子。最终只有 46% 通过质量门，说明继续追求更强图操作会把研究变成事实篡改。",
  },
  {
    plan: "Plan D", title: "重入拓扑也没有成为稳定攻击入口", chapter: "search",
    actions: ["修改 raw AMR 的共享节点：拆分、合并、重绑定和论元重排。", "每段至少生成 16 个候选，在 30 段探索样本上评估。", "同时记录低分率与语义质量，避免只看 detector score。"],
    metrics: [["≥16", "每段候选", C.gray], ["26.7%", "低分率", C.coral], ["50%", "质量通过率", C.gold]],
    limit: "结果只说明已尝试的局部 reentrancy 算子收益有限，不证明所有此类方法无效。",
    decision: "停止继续堆局部算子，转向检测反馈驱动的候选搜索。", kind: "historical",
    note: "Plan D 尝试 AMR 的重入结构，但低分率与质量仍不理想。这里的正确口径是开发选择，而不是一般不可能性定理。",
  },
  {
    plan: "Plan E", title: "反馈搜索发现大面积低分，但原始覆盖率混入质量失败", chapter: "search",
    actions: ["E1：两条往返回译支线。", "E2：每段 4 个候选，联合句界、语态、名词化、句序和指代改写。", "E4：单独探索句界臂；E5：增加质量过滤与失败补救。", "记录候选级和文档级最低分覆盖，但暂未统一 Valid ASR。"],
    metrics: [["93.4%", "1000 候选低分率", C.coral], ["249/250", "至少一个低分", C.coral], ["0.728", "候选 AUROC", C.blue], ["6.6%", "TPR@1% FPR", C.cyan], ["93.3%", "E4 原始逃逸率", C.gold]],
    limit: "249/250 是最低分覆盖率，不经过严格质量门，不能写成有效攻击率。",
    decision: "确认存在攻击面，但机制与真实 Valid ASR 尚未识别。", kind: "historical",
    note: "Plan E 是早期效果最强的一段，但也是最容易误报的数字。99.6% 只能说明四个候选中常有低分，不能说明质量合格。E4 的高逃逸率把注意力引向句界。",
  },
  {
    plan: "Plan F", title: "重算分母：从最低分覆盖率纠正为质量约束 Valid ASR", chapter: "search",
    actions: ["冻结每段 4 候选预算。", "只有同时通过质量门且检测分低于阈值才计为 Valid Attack。", "对固定预算失败样本执行预先限定的自适应补救。", "保留 249/250，但降级为低分覆盖诊断。"],
    metrics: [["191/250", "固定预算成功", C.blue], ["76.4%", "固定预算 Valid ASR", C.blue], ["218/250", "补救后累计成功", C.cyan], ["87.2%", "累计 Valid ASR", C.cyan], ["99.6%", "仅低分覆盖", C.gray]],
    limit: "指标纠偏确认攻击面存在，但不证明完整 Plan E 是最优方法。",
    decision: "口径闭合后，进入独立样本的公平方法比较。", kind: "go",
    note: "Plan F 的价值是把早期最亮眼的 99.6% 降级为正确口径。固定预算 Valid ASR 是 76.4%，补救后 87.2%。这是后续所有比较的分母基础。",
  },
];
stages.forEach(stageSlide);

// 11. Plan G
{
  const slide = pptx.addSlide("BASE");
  header(slide, "Plan G｜独立三臂确认：句界重组胜过复杂联合改写", "causal", "30 个新 C4 提示；同模型、K=4、同质量门");
  bar(slide, 0.85, 2.25, 6.45, "普通复述", 6.7, 100, C.gray);
  bar(slide, 0.85, 3.05, 6.45, "完整 Plan E", 53.3, 100, C.cyan);
  bar(slide, 0.85, 3.85, 6.45, "仅句界重组", 100.0, 100, C.blue);
  box(slide, 9.5, 2.1, 2.8, 3.6, C.ink, C.ink, 1);
  slide.addText("无反馈选择", { x: 9.82, y: 2.5, w: 2.15, h: 0.3, fontFace: FONT, fontSize: 14, bold: true, color: "CBD5E1", align: "center", margin: 0 });
  slide.addText("29/30", { x: 9.75, y: 3.1, w: 2.3, h: 0.65, fontFace: FONT, fontSize: 31, bold: true, color: "75D5DE", align: "center", margin: 0 });
  slide.addText("句界 Valid ASR", { x: 9.85, y: 3.9, w: 2.1, h: 0.25, fontFace: FONT, fontSize: 11, color: "CBD5E1", align: "center", margin: 0 });
  slide.addText("复杂指令不是优势来源\n检测反馈也不是主要来源", { x: 9.78, y: 4.62, w: 2.24, h: 0.55, fontFace: FONT, fontSize: 11.5, bold: true, color: C.white, align: "center", margin: 0 });
  decisionBand(slide, "方法排序完成：句界成为主要诱因；但生成式句界臂仍可能伴随词汇变化。", "go");
  footer(slide);
  finalize(slide, "Plan G｜独立三臂", "三臂实验推翻了越复杂越强的早期判断。有反馈结果是 2/30、16/30、30/30；无反馈是 2/30、10/30、29/30。它完成诱因收敛，但还不是单变量因果实验。");
}

// 12. Plan H design
{
  const slide = pptx.addSlide("BASE");
  header(slide, "Plan H｜把句界做成单变量：严格 A--B--A", "causal");
  const items = [
    ["A 原始", "5 个句号边界", "默认检测为 5 units", C.blue],
    ["B 分号", "只替换 4 个标点", "默认检测为 1 unit", C.coral],
    ["A′ 恢复", "文本保持 B", "检测端恢复 5 spans", C.cyan],
  ];
  items.forEach((d, i) => {
    const x = 0.82 + i * 4.15;
    box(slide, x, 2.05, 3.65, 3.85, C.white, d[3], 1.5);
    slide.addText(d[0], { x: x + 0.3, y: 2.38, w: 3.05, h: 0.32, fontFace: FONT, fontSize: 18, bold: true, color: d[3], align: "center", margin: 0 });
    for (let j = 0; j < 5; j += 1) {
      const ux = x + 0.45 + j * 0.57;
      box(slide, ux, 3.15, 0.46, 0.48, i === 1 ? C.paleCoral : C.paleBlue, d[3], 1);
      slide.addText(`S${j + 1}${i === 1 && j < 4 ? ";" : ""}`, { x: ux, y: 3.28, w: 0.46, h: 0.18, fontFace: FONT, fontSize: 8.5, bold: true, color: C.ink, align: "center", margin: 0 });
    }
    if (i === 1) box(slide, x + 0.36, 3.06, 3.0, 0.66, "FFFFFF", d[3], 1.4);
    slide.addText(d[1], { x: x + 0.4, y: 4.13, w: 2.85, h: 0.28, fontFace: FONT, fontSize: 13, bold: true, color: C.ink, align: "center", margin: 0 });
    slide.addText(d[2], { x: x + 0.4, y: 4.68, w: 2.85, h: 0.45, fontFace: FONT, fontSize: 12, color: C.slate, align: "center", margin: 0 });
  });
  arrow(slide, 4.5, 3.55, 0.38, 0, C.coral, 2);
  arrow(slide, 8.65, 3.55, 0.38, 0, C.cyan, 2);
  decisionBand(slide, "固定模板库、词汇、词序；只让外部句界与 detector units 变化。", "go");
  footer(slide);
  finalize(slide, "Plan H｜A--B--A 设计", "B 条件只替换四个句末标点。A' 不恢复原始文本，只给 detector 保存的五个边界。这个设计把语言生成、词汇变化和候选选择全部拿掉。");
}

// 13. Plan H result
{
  const slide = pptx.addSlide("BASE");
  slide.background = { color: C.ink };
  slide.addText("Plan H · 严格标点干预", { x: 0.72, y: 0.4, w: 3.4, h: 0.24, fontFace: FONT, fontSize: 10, bold: true, color: "75D5DE", margin: 0 });
  slide.addText("判定随句界可逆翻转", { x: 0.72, y: 0.9, w: 6.3, h: 0.58, fontFace: FONT, fontSize: 31, bold: true, color: C.white, margin: 0 });
  const vals = [["1/30", "原始 5 units", C.cyan], ["30/30", "分号后 1 unit", C.coral], ["1/30", "恢复 5 units", C.cyan]];
  vals.forEach((d, i) => {
    const x = 1.0 + i * 4.1;
    slide.addShape(pptx.ShapeType.ellipse, { x, y: 2.05, w: 3.0, h: 3.0, fill: { color: i === 1 ? "3E2731" : "163B51" }, line: { color: d[2], width: 3 } });
    slide.addText(d[0], { x: x + 0.2, y: 2.86, w: 2.6, h: 0.65, fontFace: FONT, fontSize: 32, bold: true, color: d[2], align: "center", margin: 0 });
    slide.addText(d[1], { x: x + 0.25, y: 3.9, w: 2.5, h: 0.28, fontFace: FONT, fontSize: 12.5, color: "D8E2EA", align: "center", margin: 0 });
  });
  arrow(slide, 4.12, 3.55, 0.78, 0, C.coral, 3);
  arrow(slide, 8.22, 3.55, 0.78, 0, C.cyan, 3);
  slide.addText("p = 3.73e-9", { x: 4.18, y: 4.4, w: 0.68, h: 0.2, fontFace: FONT, fontSize: 8.5, color: "F8B6AF", align: "center", margin: 0 });
  slide.addText("p = 3.73e-9", { x: 8.28, y: 4.4, w: 0.68, h: 0.2, fontFace: FONT, fontSize: 8.5, color: "A8E3E8", align: "center", margin: 0 });
  slide.addText("支持：17--46 词短文本上，SWAN 最终判定依赖外部句界。", { x: 1.25, y: 6.02, w: 10.85, h: 0.34, fontFace: FONT, fontSize: 16, bold: true, color: C.white, align: "center", margin: 0 });
  slide.addText("不支持：所有标点版本严格语义等价；“AMR 语义被攻击”；正常长度同样成立。", { x: 1.25, y: 6.55, w: 10.85, h: 0.25, fontFace: FONT, fontSize: 10.5, color: C.muted, align: "center", margin: 0 });
  finalize(slide, "Plan H｜A--B--A 结果", "这是全项目最强的行为结果：1/30 变成 30/30，再回到 1/30。两个方向的配对检验都为 3.73e-9。结论必须限定在这组短文本和 SWAN 判定上。");
}

// 14. Plan I
{
  const slide = pptx.addSlide("BASE");
  header(slide, "Plan I｜统计拆分：N=1 时公式上限已经低于阈值", "causal");
  box(slide, 0.82, 2.0, 5.15, 3.95, C.ink, C.ink, 1);
  slide.addText("z = (G − λN) / √[Nλ(1−λ)]", { x: 1.15, y: 2.45, w: 4.5, h: 0.55, fontFace: FONT, fontSize: 24, bold: true, color: C.white, align: "center", margin: 0 });
  slide.addText("λ=.25, N=1, G=1", { x: 1.4, y: 3.28, w: 4.0, h: 0.28, fontFace: FONT, fontSize: 13, color: "CBD5E1", align: "center", margin: 0 });
  slide.addText("zmax = √3 = 1.732", { x: 1.3, y: 3.95, w: 4.2, h: 0.55, fontFace: FONT, fontSize: 27, bold: true, color: "75D5DE", align: "center", margin: 0 });
  slide.addText("< 2.33", { x: 1.9, y: 4.83, w: 3.0, h: 0.4, fontFace: FONT, fontSize: 20, bold: true, color: "F8B6AF", align: "center", margin: 0 });
  metric(slide, 6.75, 2.2, 2.15, "21/30", "只替换 N 后低于阈值", C.coral);
  metric(slide, 9.35, 2.2, 2.15, "30/30", "只替换 green-rate", C.cyan);
  bullets(slide, [
    "直接 AMR 工具对 N<2 返回 NaN；预解析入口对 N=1 仍计算公式。",
    "代数替换允许非整数虚拟 G，只描述数值充分性。",
    "真实文本中统计路径与表示路径仍同时变化。",
  ], 6.72, 3.75, 5.25, 1.7, 12.2);
  decisionBand(slide, "统计路径有确定解释，但代数替换不是文本因果贡献估计。", "go");
  footer(slide);
  finalize(slide, "Plan I｜统计拆分", "N=1 的上限 1.732 直接解释为什么单单元无法过 2.33。21/30 和 30/30 是代数充分性；它们不能被写成真实文本的因果比例。");
}

// 15. Graph diagnostic
{
  const slide = pptx.addSlide("BASE");
  header(slide, "匹配函数诊断｜候选图扩大后局部模板分数被稀释", "causal", "固定组成图与模板库，比较句级图、人工组合图和真实合句重解析图");
  const data = [
    ["原句级候选图", 0.8197, "108/108 green", C.blue],
    ["人工组合图", 0.4502, "0/108 green", C.coral],
    ["真实合句重解析", 0.5200, "8/108 green", C.cyan],
  ];
  data.forEach((d, i) => {
    const x = 1.0 + i * 3.95;
    slide.addShape(pptx.ShapeType.rect, { x, y: 5.5 - d[1] * 3.55, w: 2.25, h: d[1] * 3.55, fill: { color: d[3] }, line: { color: d[3] } });
    slide.addText(d[1].toFixed(4), { x, y: 5.13 - d[1] * 3.55, w: 2.25, h: 0.3, fontFace: FONT, fontSize: 15, bold: true, color: d[3], align: "center", margin: 0 });
    slide.addText(d[0], { x: x - 0.2, y: 5.77, w: 2.65, h: 0.35, fontFace: FONT, fontSize: 11.5, bold: true, color: C.ink2, align: "center", margin: 0 });
    slide.addText(d[2], { x: x - 0.2, y: 6.25, w: 2.65, h: 0.24, fontFace: FONT, fontSize: 10.5, color: d[3], align: "center", margin: 0 });
  });
  slide.addShape(pptx.ShapeType.line, { x: 0.85, y: 3.19, w: 11.35, h: 0, line: { color: C.gold, width: 1.5, dash: "dash" } });
  slide.addText("green threshold 0.65", { x: 10.0, y: 2.9, w: 2.0, h: 0.2, fontFace: FONT, fontSize: 9.5, color: C.gold, align: "right", margin: 0 });
  box(slide, 8.55, 1.95, 3.65, 0.72, C.paleGray, C.line, 0.8);
  slide.addText("precision 中位 −0.5048\nrecall 中位 −0.1123", { x: 8.85, y: 2.1, w: 3.05, h: 0.42, fontFace: FONT, fontSize: 11.2, bold: true, color: C.ink2, align: "center", margin: 0 });
  footer(slide, "限制：人工组合图不是自然文本的 gold AMR，不能据此排除真实攻击中的 parser drift。");
  finalize(slide, "匹配函数诊断", "人工组合把中位 F1 从 0.8197 降到 0.4502，green 从 108/108 变为 0/108；真实合句重解析中位为 0.5200。它只证明 whole-graph S2Match 对作用域敏感。");
}

// 16--19
[
  {
    plan: "Plan J", title: "强基线与跨水印对照限定适用范围", chapter: "causal",
    actions: ["在同一 30 段、K=4 条件下加入 SIRA-adapted。", "为 KGW 与 SynthID-Text 分别重新生成各自水印文本。", "在目标检测分数写入前冻结无反馈候选选择。"],
    metrics: [["13/30", "SIRA Valid ASR", C.gold], ["16/30", "Plan E", C.cyan], ["30/30", "句界有反馈", C.blue], ["29/30", "SWAN 无反馈", C.blue], ["0/30", "KGW", C.gray], ["1/30", "SynthID-Text", C.cyan]],
    limit: "SIRA 不是官方等价复现；三系统长度与生成模型未完全匹配。",
    decision: "句界方法不是天然通用攻击；结果与 SWAN 句级管线特异性相容。", kind: "go",
    note: "SIRA-adapted 为 43.3%，Plan E 53.3%，句界 100%。跨系统无反馈结果是 SWAN 96.7%、KGW 0、SynthID 3.3%。这些结果用于限定主张，而不是证明其他系统普遍安全。",
  },
  {
    plan: "Plan K", title: "三条分割不变防御路线均未过预注册门", chapter: "causal",
    actions: ["局部 coverage / precision / 节点互斥实例计数。", "用人类校准集估计模板特异经验尾概率。", "关系锚定 motif 定位候选子图。", "另测直接重新分句的端到端效果与误报。"],
    metrics: [["0", "局部证据恢复文档", C.coral], ["4/15", "经验尾原图", C.coral], ["2/15", "经验尾重解析", C.coral], ["41/56", "motif 原图 recall@5", C.gold], ["33/56", "motif 重解析", C.gold], ["23→19", "重分句有效逃逸", C.cyan]],
    limit: "重新分句 p=.125，独立人类误报由 3/250 升到 5/250。",
    decision: "只能报告防御负结果，不能声称已有 segmentation-invariant defense。", kind: "stop",
    note: "Plan K 按三条路线探索局部恢复。Motif 两边都只差一个，但仍按预注册门停止，没有事后扩大 top-k。直接重分句改善不显著且误报上升。",
  },
  {
    plan: "Plan L", title: "正常长度、句数守恒对照在构造阶段失败", chapter: "causal",
    actions: ["识别短文本可能放大 N 效应：确认集平均 30.9 词、固定五句。", "设计“一处合并 + 一处自然拆分”，保持总 unit 数不变。", "先做自然性与可构造性预检，再决定是否调用 parser 和 detector。", "用跨文档 oracle 组合仅做机制压力测试。"],
    metrics: [["30.9", "确认集平均词数", C.gold], ["23/30", "存在可合并位置", C.cyan], ["0/30", "存在自然拆分位置", C.coral], ["144–181", "oracle 组合词数", C.gray]],
    limit: "跨主题 oracle 组合质量不合格且端点已知，不能冒充正常长度自然文档。",
    decision: "构造门失败，未运行 parser/detector，没有新的 ASR。", kind: "stop",
    note: "Plan L 想把 N 保持不变，从而隔离超出统计上限的影响。但 30 个短文本没有一个满足自然拆分规则，所以实验在昂贵模型前停止。",
  },
  {
    plan: "Plan M", title: "创新性审计收窄了可以成立的论文主张", chapter: "causal",
    actions: ["系统核验 AliMark、SwordStamp/EDA 与相关 resegmentation 工作。", "放弃“首次句界攻击”“首次 sqrt(N) 效应”“首次重分句防御”。", "把贡献收窄到 SWAN 系统特异的可逆句界反例与证据分层。", "人工图组合降级为 scorer diagnostic。"],
    metrics: [["3", "放弃的 first claims", C.coral], ["1", "保留的系统审计主线", C.blue], ["A–B–A", "核心行为证据", C.blue]],
    limit: "创新性来自系统审计闭环与主张边界，不来自一般重分句概念。",
    decision: "论文主线收敛为范围受限的 SWAN 检测接口审计。", kind: "go",
    note: "Plan M 是研究主张纠偏。邻近工作已覆盖 split/merge、N 效应和防御，不能再讲 first。可守住的是 SWAN 上固定模板库、词汇与词序不变、判定可逆翻转。",
  },
].forEach(stageSlide);

// 20--22
[
  {
    plan: "Plan N", title: "正常长度 final1 pilot：生成完成，但 baseline 资格为 0/6", chapter: "normal",
    actions: ["在 8 个全新 source 上执行官方逐句注入。", "完成 3,467 次 DeepSeek 调用，8 篇均达到 100--220 词。", "先过结构门与 baseline 标签，再决定是否进入句数守恒变换。", "对终态做离线收尾与第二次恢复核验。"],
    metrics: [["8/8", "达到长度门", C.green], ["6/8", "通过结构门", C.gold], ["0/6", "冻结 baseline 正例", C.coral], ["3467", "模型调用", C.gray], ["0", "新增 detector 调用", C.cyan]],
    limit: "辅助 A/B 标签来自语言模型，不满足两名独立真人盲评。",
    decision: "pilot gate=false；main、变换、parser、S2Match 和 detector 均保持封存。", kind: "stop",
    note: "Plan N 生成了正常长度文本，但 6 篇结构合格文档没有一篇满足冻结 baseline 标签，所以没有进入攻击阶段。这不是攻击失败，而是正例资产路线失败。",
  },
  {
    plan: "Plan O", title: "source-anchored pilot：表面门与 S2Match 资格仍未闭合", chapter: "normal",
    actions: ["每篇 10 个生成槽绑定全新 C4 source 的 10 个句子。", "在 parser/S2Match 前增加检测盲表面门。", "所有 slot 按固定候选上限终止，失败时不后验扩展。", "无计算 closure 重建并复验 3,925 个文件。"],
    metrics: [["1891", "模型调用", C.gray], ["5/8", "完整通过结构门", C.gold], ["60", "完成槽且全为 fallback", C.coral], ["0", "S2Match≥0.75", C.coral], ["3925", "closure 文件", C.cyan]],
    limit: "其余 3 篇在 6/2/2 units 后终止；运行期 parser warning 未被持久化。",
    decision: "质量评审、变换、检测与 main 均未运行；分支 C 保持。", kind: "stop",
    note: "Plan O 试图用 source anchoring 提高可追溯性，但 60 个完成槽全靠 fallback，且没有一个通过 0.75 的模板门。它关闭的是资产构造路线，不是句界机制。",
  },
  {
    plan: "Plan P", title: "定向正例工程证明局部可行，但没有解锁正式句界实验", chapter: "normal",
    actions: ["用有利模板 47 构造 53 词、10 句事件日志。", "扩展到 126 词、20 句，并做检测盲辅助复核。", "对 126 词文本做 count-neutral 边界位置预检。", "另立全新 source-anchored 21 槽方案并修复 PROPN 门错误。"],
    metrics: [["10/10", "53词 strict injection", C.green], ["5.4772", "53词 z", C.blue], ["20/20", "126词 green", C.green], ["7.7460", "126词 z", C.blue], ["0", "自然句内拆分点", C.coral], ["0/1", "21槽单槽 accepted", C.coral]],
    limit: "定向文档缺少 source anchoring、双真人评审、正式 cohort 与经验 FPR。",
    decision: "只关闭工程可行性疑问，不改变 Branch C，也不补 CL04--CL08。", kind: "audit",
    note: "Plan P 证明可以人工构造一篇正常长度、严格注入且可检出的文本，但它不是正式 cohort。边界预检仍找不到自然拆分点，新的 21 槽路线最高 S2Match 只有 0.5625。",
  },
].forEach(stageSlide);

// 23--30
[
  {
    plan: "Plan Q", title: "广义 unitization 主线 No-Go，保留 exact certificate 切口", chapter: "cert",
    actions: ["检索 AliMark、SwordStamp 与跨系统 resegmentation benchmark。", "审计滑窗、多切分、子句检测、alignment 与 best-of-N 邻近工作。", "放弃泛化的“句级水印普遍依赖 unitization”主线。", "把问题改写为固定词序与预算 k 下的最坏值精确求解。"],
    metrics: [["NO-GO", "广义 unitization 主线", C.coral], ["exact", "保留的算法切口", C.cyan], ["k", "冻结边界移动预算", C.gold]],
    limit: "这一方向从一开始不是 SWAN 主线，不能补正常长度未完成端点。",
    decision: "转向可验证鲁棒性证书，而不是继续普通 merge/split benchmark。", kind: "go",
    note: "Plan Q 再次做创新性止损。普通重分句 benchmark 已有强邻近工作，唯一值得继续的是在声明 threat set 上求严格最坏值。",
  },
  {
    plan: "Plan R", title: "零模型预检证明 exact DP 与统计流程可正确实现", chapter: "cert",
    actions: ["先实现独立跨度加性 detector 的 exact DP，并对 brute force。", "接口审计后改为一阶相邻 span pair 的 edge-state DP。", "实现 split-conformal、strict ties 与 Clopper--Pearson fail-closed 统计。", "验证 canonical adapter、稳定去重与 cohort split。"],
    metrics: [["5691+2000", "span DP oracle", C.cyan], ["5669+2000", "edge DP oracle", C.cyan], ["1000", "退化一致性", C.green], ["39977", "300-token 合并状态", C.blue], ["0.8054s", "300-token 最大例", C.green], ["80/80", "统计门", C.gold]],
    limit: "软件正确不等于真实 positive 的证书非空洞。",
    decision: "Algorithm-ready；获准进入官方接口与真实资产预检。", kind: "go",
    note: "Plan R 在零模型条件下把数学和统计先做对。关键转折是确认 SemStamp 和 SimMark 不是独立 span score，而是相邻 pair score，因此重新实现 edge-state DP。",
  },
  {
    plan: "Plan S", title: "官方接口等价通过，但旧公开资产容量不足", chapter: "cert",
    actions: ["在 RTX 3090 建立 SemStamp/SimMark 隔离环境。", "冻结代码、模型、数据、Arrow blob、环境与 SHA manifest。", "完成两系统官方/直接实现的有限 P0 等价预检。", "对 4,000 行公开资产做两次 score-blind 资格重放。"],
    metrics: [["43/60", "SemStamp P0 eligible", C.gold], ["258/258", "SemStamp 等价门", C.green], ["60/60", "SimMark P0", C.green], ["240/240", "SimMark 等价门", C.green], ["100/119/102", "SemStamp 三角色 eligible", C.coral], ["63/113/96/129", "SimMark 四角色", C.coral]],
    limit: "所有角色均低于预注册容量门；SimMark γ=.5 只是 sentinel。",
    decision: "真实 WP2 Stop；未进入 gamma、certificate、FPR/TPR 或 non-vacuity。", kind: "stop",
    note: "Plan S 证明接口等价，但公开资产在 5--35 atom 门下流失严重。此后远端任务统一使用 Supervisor、flock、checkpoint、0600 权限和 SHA 回传。",
  },
  {
    plan: "Plan T", title: "v2--v8 逐层排除 adapter、core 与吞吐假设", chapter: "cert",
    actions: ["v2：native-512 adapter 因 999/1000 list 边界无 whitespace Stop。", "v3：virtual separator 恢复资格，但 Python DP 超 60 秒。", "v4：C++17 grouped core 通过 10,000 例与最大真实结构。", "v5：external pair table 在 5,000,001 pairs 触发上限。", "v6--v8：factorized core、batch 与 token reuse 依次审计。"],
    metrics: [["18.4s", "v4 最大结构", C.green], ["38,664,070", "v4 pairs", C.blue], ["22.44h", "SemStamp v6", C.cyan], ["71.81h", "SimMark v6", C.coral], ["1.0027×", "batch 最佳加速", C.gold], ["0.6748%", "全局复用减少", C.gray]],
    limit: "v8 还违反冻结读取范围，因此记为 Invalid/Stop。",
    decision: "价值在于排除工程假设，而不是得到 certificate。", kind: "audit",
    note: "Plan T 是工程排障链。每个版本只改一个假设：adapter、Python 状态、pair 物化、batch underfill、跨文档复用。最终没有一个简单优化足以通过预算门。",
  },
  {
    plan: "Plan U", title: "长跑证明 detector health 必须先于 certificate", chapter: "cert",
    actions: ["v9 单独接受 RTX 3090 约三天运行，不回写旧预算门。", "SimMark 先做 fit 与结构计算，再审计官方 histogram 与 clean baseline。", "SemStamp v10 改为 P0 health 通过后才自动启动 certificate。", "Supervisor 在健康门失败时自动早停。"],
    metrics: [["29h11m", "SimMark v9", C.gray], ["γ=0.0", "SimMark official fit", C.coral], ["89/1000", "SimMark clean positive", C.coral], ["9.25%", "SimMark k=1 FPR", C.coral], ["974/999", "SemStamp clean positive", C.green], ["10.88%", "SemStamp FPR CP95 上界", C.coral]],
    limit: "SemStamp certificate 保持 0/1799，节省约 18.6 GPU-hours。",
    decision: "攻击前 detector 不健康时，任何 certificate 下降都不可归因。", kind: "stop",
    note: "Plan U 的教训是先做健康门。SimMark 和 SemStamp 当时都没有满足独立 FPR/TPR 条件，所以继续算 certificate 只会得到不可解释结果。",
  },
  {
    plan: "Plan V", title: "重建全新资产，并把每个生成控制流错误显式封存", chapter: "cert",
    actions: ["v11 确认 official/direct 等价并定位 SimMark prompt/pooling 问题。", "v12 冻结 12,612 个合法身份及 null/generation/reserve 队列。", "v13 因 fallback 不可达长时间卡住；v14 又破坏严格前缀 identity。", "v15 生成 1,000 positive 后发现 5 条空 continuation。", "v16 披露式 post-hoc rescue：995 条保留 + 固定队列补 5 条。"],
    metrics: [["2000/2000", "SimMark 等价", C.green], ["1999/1999", "SemStamp 等价", C.green], ["12612", "合法身份", C.cyan], ["84981s", "v13 卡住", C.coral], ["5", "v15 validator 漏检", C.coral], ["995+5", "v16 combined positive", C.green]],
    limit: "v16 Go 只说明资产身份与 validator 闭合，不提供 detector/certificate 结果。",
    decision: "获得干净的 1,000 positive，允许进入正式 detector health。", kind: "go",
    note: "Plan V 不是反复调参，而是逐次修复并保留历史 Stop。v16 的 post-hoc rescue 被明确披露，不能把 v15 改写为成功。",
  },
  {
    plan: "Plan W", title: "v17 detector health 通过，v18 在 certificate 前资源 Stop", chapter: "cert",
    actions: ["v17 用冻结的 1000 calibration null、1000 test null、1000 positive。", "阈值取 calibration 第 951 顺序统计量。", "做 2×3000 replay 与 6000 项 official/direct 等价。", "v18 用三个 Supervisor 完成 3000 行 score-blind readiness。"],
    metrics: [["47/1000", "test FPR", C.blue], ["5.95%", "FPR CP95 上界", C.cyan], ["954/1000", "positive TPR", C.green], ["94.16%", "TPR CP95 下界", C.green], ["83,540,900", "最大 all-allowed pairs", C.coral], ["86.65h", "双 replay 外推", C.coral]],
    limit: "B1 正式状态为 STOPPED / Not started；score 与 certificate 调用均为 0。",
    decision: "证明当前 exact 实现与预算不足，不证明 SemStamp 不鲁棒。", kind: "stop",
    note: "v17 是旧认证链最重要的 Go：健康 detector 成立。v18 随后在 50M pair cap 和 72h 预算两处失败，所以 certificate 没有开始。",
  },
  {
    plan: "Plan X", title: "证据归位：同一投稿，但两条证据流不能互相补证", chapter: "cert",
    actions: ["SWAN 保留短文本 A--B--A，正常长度端点继续未确认。", "certificate 流保留 exact DP、接口复现、健康失败与 v17 Go/v18 Stop。", "不创建 v19，不删超限样本、不缩 threat set、不调高预算。", "统一稿转向 TMLR，主会立即投稿判为 No-Go。"],
    metrics: [["TMLR", "同行评审目标", C.cyan], ["0", "活动远端计算", C.green], ["2", "独立证据流", C.blue], ["0", "certificate 结果", C.coral]],
    limit: "共享“句界如何进入检测合同”的总问题，不共享系统特异证据。",
    decision: "旧 SWAN 投稿链与 certified_resegmentation_wp2_v18 工程链关闭。", kind: "audit",
    note: "Plan X 的关键是区分同一投稿和互相补证。两条线可以共同讨论 unitization，但 SemStamp 结果不能补 SWAN 的 CL04--CL08。",
  },
].forEach(stageSlide);

// 31--37
[
  {
    plan: "Plan Y", title: "另立 SemStamp 认证防御计划并重新做创新边界", chapter: "defense",
    actions: ["不创建旧链 v19，另立 semstamp_certified_defense。", "把目标改成 exact variable-N certificate + certificate-guided encoding。", "冻结 boundary insertion/deletion/relocation 与 5% FPR/clean TPR 目标。", "建立 Carrier--Decoder Contract：renderer、算法、统计、来源任一不健康即 Stop。"],
    metrics: [["SemStamp", "唯一主防御对象", C.green], ["47/1000", "冻结健康 FPR", C.blue], ["954/1000", "冻结健康 TPR", C.green], ["b≤3", "边界预算", C.gold]],
    limit: "SWAN 只提供历史动机；SimMark 健康口径未闭合。",
    decision: "保留的切口是完整 variable-N 证书与 worst-case-aware encoding。", kind: "go",
    note: "Plan Y 重新做立项审计。已有工作覆盖普通 split/merge defense，因此新切口必须是完整 variable-N exact certificate，以及以最坏情况反馈指导编码。",
  },
  {
    plan: "Plan Z", title: "v1 算法通过，但 fixed-byte renderer 无法表达 42 条真实句界", chapter: "defense",
    actions: ["把每个 Unicode 非空白 maximal span 定义为不可再分 atom。", "按最终 edge count 分层求解 insertion/deletion/relocation。", "纯 CPU 完成 systematic、random 与 fixed-N reduction。", "将冻结 3,000 条 v17 cohort 接入 renderer health。"],
    metrics: [["5000/5000", "systematic", C.green], ["2000/2000", "random oracle", C.green], ["1000/1000", "fixed-N reduction", C.green], ["2958/3000", "exact render", C.gold], ["42", "NLTK 边界落入 atom", C.coral]],
    limit: "42 条按 calibration/test/positive 为 17/20/5；协议禁止删样或 WP3 改 threat model。",
    decision: "WP3_HEALTH_STOP：载体合同失败，不是 DP 错误。", kind: "stop",
    note: "v1 最重要的教训是算法 Go 不能掩盖 renderer Stop。42 条真实句界无法由固定 byte atom 表达，因此整个 v1 在 smoke 前停止。",
  },
  {
    plan: "Plan AA--AB", title: "v2 refined atoms 修复 renderer，并完整复现 detector health", chapter: "defense",
    actions: ["只在冻结 NLTK unit endpoint 细化 lexical spans，保留 raw UTF-8。", "两次 3,000-row model-free replay 修复全部 42 条历史失败。", "保留并审计 59 个 zero-byte gap，再重跑 5000/2000/1000 算法测试。", "依次执行 P0、2×96 smoke 和 2×3000 full replay。"],
    metrics: [["3000/3000", "逐字节恢复", C.green], ["42/42", "risk rows 修复", C.green], ["59/59", "zero-byte gaps", C.green], ["192/192", "smoke exact", C.green], ["6000/6000", "full score exact", C.green], ["593.52s", "full replay", C.cyan]],
    limit: "首次 smoke 因缺少 CUBLAS_WORKSPACE_CONFIG 在首个 embedding 前失败，空前缀封存后重启。",
    decision: "v2 detector health Go：FPR 4.7%、TPR 95.4%，certificate 仍为 0。", kind: "go",
    note: "v2 用 refined atoms 正确修复 v1，而不是覆盖 v1 Stop。两次 full replay 与 v17 完全一致，说明 renderer 修复没有改变 detector。",
  },
  {
    plan: "Plan AC", title: "v2 WP4 发现 runner 漏算 variable-N，并用严格子集证明超 cap", chapter: "defense",
    actions: ["冻结 3,000 行、radius 0--3、2M spans、50M pairs 与 72h 双 replay 门。", "首次 runner 完成 83 行后，静态复核发现只调用 fixed-N solver。", "封存无效前缀，禁止 resume；另立纠正协议。", "利用 fixed-N 是 variable-N 严格子集，在 unchanged row800 重放。"],
    metrics: [["83", "无效前缀行", C.gray], ["769,113", "all-allowed spans", C.cyan], ["83,540,900", "fixed-N pairs", C.coral], ["+33,540,900", "超过 50M cap", C.coral], ["21/21", "纠正审计", C.green]],
    limit: "即使未计算其余 variable-N 分支，严格子集已经足以判 Stop。",
    decision: "WP4_RESOURCE_PREFLIGHT_READINESS_STOP_CORE_CAP；WP5/WP6 未启动。", kind: "stop",
    note: "Plan AC 先发现实现错误，再用数学集合关系做保守裁决。83 行旧结果不能当作 variable-N profile；row800 的 fixed-N 子集已超过 50M，所以无需读取 score 即可 Stop。",
  },
  {
    plan: "Plan AD", title: "v3 完整 C++ core 通过大规模验证，但 batch 优化改变 region", chapter: "defense",
    actions: ["按最终 edge count 实现完整 variable-N C++ core，cap 提至预冻结 200M。", "macOS arm64 与 Linux x86-64 各做两次正式 correctness replay。", "对 row800 七层 target edge 扫描，并做 133M-pair 放大验证。", "实测 batch 64/128/256/512 的特征吞吐与 region identity。"],
    metrics: [["4×", "跨架构确定 replay", C.green], ["133,713,262", "放大 pairs", C.blue], ["4.46GB", "最大 child RSS", C.cyan], ["109.72s", "最慢 core", C.cyan], ["1.0333×", "batch128 probe", C.gold], ["≠", "maximum region SHA", C.coral]],
    limit: "batch128 两次彼此确定，但与 canonical batch64 region array 不同。",
    decision: "core Go；THROUGHPUT_STOP_BATCH_DEPENDENT_REGION，后续只能 batch64。", kind: "audit",
    note: "Plan AD 证明完整 core 在 200M cap 内可运行。Batch 128 看起来稍快，但在最大 workload 上改变 certificate scorer 输入，因此不是可接受优化。",
  },
  {
    plan: "Plan AE", title: "全量 variable-N profile 只失败于 72 GPU-hours 资源门", chapter: "defense",
    actions: ["扩展 core 复用带 native allowed flags 的 radius union sidecar。", "先对 row800 做双 smoke 和三 cohort 小样本。", "用 6 个 spawn worker 跑满 3,000 行、radius 0--3、全部 target layers。", "独立复算 legal text 去重、token、span、pair、RSS 与两条 replay 公式。"],
    metrics: [["3000/3000", "完整分母", C.green], ["46,501", "target solves", C.blue], ["67,001,110", "legal texts", C.cyan], ["3.774B", "MPNet tokens", C.cyan], ["6.316B", "pruned pairs", C.blue], ["77.6152h", "保守双 replay", C.coral]],
    limit: "结构、内存、去重、200M pair 与 2M span 门全部通过；仅 72h 门失败。",
    decision: "V3_VARIABLE_N_PROFILE_READINESS_STOP；新 WP4/WP5 与 certificate 未启动。", kind: "stop",
    note: "Plan AE 是最新终态。两条公式得到 76.5346 和 77.6152 小时，保守取后者，超过 72 小时 5.6152 小时。它只说明冻结实现和硬件下资源不足。",
  },
].forEach(stageSlide);

// 38. Final
{
  const slide = pptx.addSlide("BASE");
  header(slide, "最终证据矩阵：哪些能讲，哪些必须明确说没有", "close");
  const rows = [
    ["SWAN 句界诱因", "短文本 A--B--A 1/30→30/30→1/30", "已建立", "go"],
    ["统计机制", "N=1 时 zmax=1.732<2.33", "已建立", "go"],
    ["匹配机制", "S2Match 对图作用域敏感", "诊断证据", "audit"],
    ["正常长度机制", "count-neutral 自然文本确认", "未完成", "stop"],
    ["SWAN 防御", "segmentation-invariant detector", "未得到", "stop"],
    ["SemStamp 证书", "detector health 与 exact core", "软件证据成立", "go"],
    ["Certified TPR", "完整 variable-N certificate", "未启动", "stop"],
  ];
  rows.forEach((r, i) => {
    const y = 1.95 + i * 0.62;
    box(slide, 0.78, y, 11.75, 0.5, i % 2 ? C.paleGray : C.white, C.line, 0.5);
    slide.addText(r[0], { x: 1.03, y: y + 0.14, w: 2.25, h: 0.2, fontFace: FONT, fontSize: 11.5, bold: true, color: C.ink, margin: 0 });
    slide.addText(r[1], { x: 3.35, y: y + 0.14, w: 5.75, h: 0.2, fontFace: FONT, fontSize: 10.8, color: C.ink2, margin: 0 });
    slide.addText(r[2], { x: 9.2, y: y + 0.14, w: 1.7, h: 0.2, fontFace: FONT, fontSize: 10.8, bold: true, color: r[3] === "go" ? C.green : r[3] === "audit" ? C.gold : C.coral, margin: 0 });
    status(slide, 11.15, y + 0.1, r[3] === "go" ? "GO" : r[3] === "audit" ? "DIAG" : "STOP", r[3], 0.88);
  });
  decisionBand(slide, "当前最稳妥的投稿定位：SWAN 句界接口审计；认证防御作为独立、未产出 certificate 的工程证据流。", "audit");
  footer(slide);
  finalize(slide, "最终证据矩阵", "收尾时不要再按投入规模讲贡献。真正能讲的是 A--B--A、统计解释和作用域诊断；正常长度、防御和 certified TPR 都必须明确未完成。");
}

async function main() {
  await pptx.writeFile({ fileName: "lab_update_full_narrative_20261010.pptx" });
  const body = notes.map((n, i) => `## ${i + 1}. ${n.title}\n\n${n.note}\n`).join("\n");
  fs.writeFileSync(
    "详细版逐页讲稿.md",
    `# 完整实验演进详细版逐页讲稿\n\n共 ${notes.length} 页；建议 30--40 分钟，时间不足时只讲 1--19、29、36--37 页。\n\n${body}`,
    "utf8"
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
