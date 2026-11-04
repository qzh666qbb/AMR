const pptxgen = require("pptxgenjs");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");
const { autoFontSize } = require("./pptxgenjs_helpers/text");
const { imageSizingContain } = require("./pptxgenjs_helpers/image");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "quzehao";
pptx.subject = "SWAN sentence-boundary mechanism and certified-defense progress";
pptx.title = "从复杂 AMR 攻击到句界诱因";
pptx.company = "Xidian University";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Noto Sans CJK SC",
  bodyFontFace: "Noto Sans CJK SC",
  lang: "zh-CN",
};
pptx.defineSlideMaster({
  title: "BASE",
  background: { color: "F6F8FB" },
  objects: [],
  slideNumber: {
    x: 12.55,
    y: 7.05,
    w: 0.35,
    h: 0.2,
    fontFace: "Noto Sans CJK SC",
    fontSize: 9,
    color: "94A3B8",
    align: "right",
    margin: 0,
  },
});

const W = 13.333;
const H = 7.5;
const FONT = "Noto Sans CJK SC";
const C = {
  ink: "102A43",
  ink2: "243B53",
  cobalt: "2457D6",
  cyan: "22A7B8",
  coral: "E4574F",
  paper: "F6F8FB",
  white: "FFFFFF",
  slate: "64748B",
  muted: "94A3B8",
  line: "D9E2EC",
  paleBlue: "EAF0FF",
  paleCyan: "E7F7F8",
  paleCoral: "FDECE9",
  paleGray: "EDF2F7",
  green: "2E8B72",
  paleGreen: "E8F5F0",
  gold: "D69E2E",
};

const SOURCE_NOTE =
  "数据来源：CURRENT_RESULTS.md、CLAIM_EVIDENCE_MATRIX.md、叙事.md、semantic_watermark_defense PLAN v0.17。";

function addSectionLabel(slide, text) {
  slide.addText(text, {
    x: 0.65,
    y: 0.22,
    w: 3.0,
    h: 0.22,
    fontFace: FONT,
    fontSize: 10,
    bold: true,
    color: C.cobalt,
    charSpacing: 1.2,
    margin: 0,
  });
}

function addTitle(slide, title, subtitle = null, section = "研究进展") {
  addSectionLabel(slide, section);
  slide.addText(title, autoFontSize(title, FONT, {
    x: 0.65,
    y: 0.56,
    w: 12.0,
    h: 0.72,
    mode: "shrink",
    fontSize: 34,
    minFontSize: 27,
    maxFontSize: 34,
    bold: true,
    color: C.ink,
    margin: 0,
    breakLine: false,
  }));
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.67,
      y: 1.28,
      w: 11.7,
      h: 0.36,
      fontFace: FONT,
      fontSize: 14,
      color: C.slate,
      margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.65,
    y: subtitle ? 1.73 : 1.47,
    w: 12.0,
    h: 0,
    line: { color: C.line, width: 1 },
  });
}

function addFooter(slide, text = SOURCE_NOTE) {
  slide.addText(text, {
    x: 0.65,
    y: 7.05,
    w: 11.55,
    h: 0.18,
    fontFace: FONT,
    fontSize: 8.5,
    color: C.muted,
    margin: 0,
  });
}

function addBox(slide, x, y, w, h, opts = {}) {
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w,
    h,
    rectRadius: 0,
    fill: { color: opts.fill || C.white, transparency: opts.transparency || 0 },
    line: {
      color: opts.line || opts.fill || C.line,
      width: opts.lineWidth === undefined ? 1 : opts.lineWidth,
      transparency: opts.lineTransparency || 0,
    },
    shadow: opts.shadow
      ? { type: "outer", color: "AAB7C4", opacity: 0.14, blur: 1.5, angle: 45, distance: 1 }
      : undefined,
  });
}

function addMetric(slide, x, y, w, value, label, color = C.cobalt) {
  slide.addText(value, {
    x,
    y,
    w,
    h: 0.58,
    fontFace: FONT,
    fontSize: 29,
    bold: true,
    color,
    margin: 0,
    align: "center",
  });
  slide.addText(label, {
    x,
    y: y + 0.61,
    w,
    h: 0.28,
    fontFace: FONT,
    fontSize: 11,
    color: C.slate,
    margin: 0,
    align: "center",
  });
}

function addStatus(slide, x, y, text, kind = "go") {
  const color = kind === "go" ? C.green : kind === "warn" ? C.gold : C.coral;
  const fill = kind === "go" ? C.paleGreen : kind === "warn" ? "FFF8E6" : C.paleCoral;
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w: 0.84,
    h: 0.28,
    fill: { color: fill },
    line: { color, width: 1 },
  });
  slide.addText(text, {
    x,
    y: y + 0.01,
    w: 0.84,
    h: 0.2,
    fontFace: FONT,
    fontSize: 9,
    bold: true,
    color,
    align: "center",
    margin: 0,
  });
}

function addArrow(slide, x, y, w, h, color = C.cobalt, width = 2) {
  slide.addShape(pptx.ShapeType.line, {
    x,
    y,
    w,
    h,
    line: { color, width, beginArrowType: "none", endArrowType: "triangle" },
  });
}

function addBoundaryStrip(slide, x, y, unitW = 0.66, gap = 0.08, mode = "five") {
  if (mode === "five") {
    for (let i = 0; i < 5; i += 1) {
      addBox(slide, x + i * (unitW + gap), y, unitW, 0.55, {
        fill: i % 2 === 0 ? C.paleBlue : C.paleCyan,
        line: i % 2 === 0 ? C.cobalt : C.cyan,
      });
      slide.addText(`S${i + 1}`, {
        x: x + i * (unitW + gap),
        y: y + 0.13,
        w: unitW,
        h: 0.22,
        fontFace: FONT,
        fontSize: 11,
        bold: true,
        color: C.ink,
        align: "center",
        margin: 0,
      });
    }
    return;
  }
  addBox(slide, x, y, 5 * unitW + 4 * gap, 0.55, {
    fill: C.paleCoral,
    line: C.coral,
    lineWidth: 1.5,
  });
  for (let i = 0; i < 5; i += 1) {
    slide.addText(`S${i + 1}${i < 4 ? " ;" : ""}`, {
      x: x + i * (unitW + gap),
      y: y + 0.13,
      w: unitW,
      h: 0.22,
      fontFace: FONT,
      fontSize: 11,
      bold: true,
      color: C.ink,
      align: "center",
      margin: 0,
    });
  }
}

function addBar(slide, x, y, w, label, value, color, max = 100, suffix = "%") {
  slide.addText(label, {
    x,
    y: y - 0.01,
    w: 1.85,
    h: 0.29,
    fontFace: FONT,
    fontSize: 12,
    color: C.ink2,
    margin: 0,
  });
  if (value < max) {
    slide.addShape(pptx.ShapeType.rect, {
      x: x + 1.95,
      y,
      w,
      h: 0.24,
      fill: { color: C.paleGray },
      line: { color: C.paleGray },
    });
  }
  slide.addShape(pptx.ShapeType.rect, {
    x: x + 1.95,
    y,
    w: Math.max(0.02, (w * value) / max),
    h: 0.24,
    fill: { color },
    line: { color },
  });
  slide.addText(`${value.toFixed(1)}${suffix}`, {
    x: x + 1.95 + w - 0.76,
    y: y - 0.31,
    w: 0.68,
    h: 0.22,
    fontFace: FONT,
    fontSize: 9.5,
    bold: true,
    color,
    margin: 0,
    align: "right",
  });
}

function finalizeSlide(slide, notes) {
  if (notes) slide.addNotes(notes);
  warnIfSlideHasOverlaps(slide, pptx, {
    muteContainment: true,
    ignoreLines: true,
    ignoreDecorativeShapes: true,
  });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

// Slide 1
{
  const slide = pptx.addSlide("BASE");
  slide.background = { color: C.ink };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: H,
    fill: { color: C.ink },
    line: { color: C.ink },
  });
  slide.addText("研究进展汇报", {
    x: 0.78,
    y: 0.55,
    w: 2.4,
    h: 0.25,
    fontFace: FONT,
    fontSize: 11,
    bold: true,
    color: "75D5DE",
    charSpacing: 1.6,
    margin: 0,
  });
  slide.addText("从复杂 AMR 攻击\n到句界诱因", {
    x: 0.78,
    y: 1.25,
    w: 6.1,
    h: 1.65,
    fontFace: FONT,
    fontSize: 35,
    bold: true,
    color: C.white,
    margin: 0,
    breakLine: false,
    lineSpacingMultiple: 0.92,
  });
  slide.addText("按“直觉 → 可证伪预测 → 对照实验”重构 SWAN 机制证据，并汇报后续认证防御的真实终态", {
    x: 0.82,
    y: 3.15,
    w: 5.8,
    h: 0.85,
    fontFace: FONT,
    fontSize: 16,
    color: "CBD5E1",
    margin: 0,
    breakLine: false,
  });
  addBoundaryStrip(slide, 7.65, 1.35, 0.72, 0.11, "five");
  slide.addText("5 个检测单元", {
    x: 7.65,
    y: 2.02,
    w: 4.04,
    h: 0.28,
    fontFace: FONT,
    fontSize: 11,
    color: "CBD5E1",
    margin: 0,
    align: "center",
  });
  addArrow(slide, 9.38, 2.45, 0, 0.65, C.coral, 3);
  addBoundaryStrip(slide, 7.65, 3.3, 0.72, 0.11, "one");
  slide.addText("1 个检测单元", {
    x: 7.65,
    y: 3.96,
    w: 4.04,
    h: 0.28,
    fontFace: FONT,
    fontSize: 11,
    color: "F8B6AF",
    margin: 0,
    align: "center",
  });
  addArrow(slide, 9.38, 4.42, 0, 0.65, C.cyan, 3);
  addBoundaryStrip(slide, 7.65, 5.3, 0.72, 0.11, "five");
  slide.addText("恢复原边界", {
    x: 7.65,
    y: 5.96,
    w: 4.04,
    h: 0.28,
    fontFace: FONT,
    fontSize: 11,
    color: "A8E3E8",
    margin: 0,
    align: "center",
  });
  slide.addText("quzehao  ·  2026.10.10", {
    x: 0.8,
    y: 6.73,
    w: 3.8,
    h: 0.22,
    fontFace: FONT,
    fontSize: 10,
    color: "94A3B8",
    margin: 0,
  });
  finalizeSlide(
    slide,
    "开场只讲一句：上次我展示的是从 Plan A 到 E 的攻击搜索；老师提醒我不要继续堆算子，而要先找诱因。今天汇报的主线就是：句界为什么成为诱因，我如何按这个直觉倒推实验，以及哪些结论仍然没有做出来。"
  );
}

// Slide 2
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "上次汇报只回答了“能否攻击”，没有回答“为什么”",
    "Plan A--E 从简单改写一路走到多候选反馈搜索，但仍是方法堆叠",
    "问题重写"
  );
  slide.addImage({
    path: "previous_montage.png",
    ...imageSizingContain("previous_montage.png", 0.72, 2.05, 5.85, 3.65),
  });
  addBox(slide, 0.72, 5.83, 5.85, 0.65, { fill: C.paleGray, line: C.line });
  slide.addText("旧稿的结构：逐页解释 Plan A → B → C → D → E", {
    x: 1.05,
    y: 6.03,
    w: 5.2,
    h: 0.24,
    fontFace: FONT,
    fontSize: 13,
    bold: true,
    color: C.ink2,
    align: "center",
    margin: 0,
  });
  const steps = [
    ["A", "普通改写", "弱"],
    ["B", "AMR 图操作", "20.0%"],
    ["C", "强算子", "质量下降"],
    ["D", "重入拓扑", "26.7%"],
    ["E", "反馈搜索", "低分显著"],
  ];
  steps.forEach((item, i) => {
    const y = 2.05 + i * 0.79;
    slide.addShape(pptx.ShapeType.ellipse, {
      x: 7.05,
      y,
      w: 0.44,
      h: 0.44,
      fill: { color: i === 4 ? C.coral : C.cobalt },
      line: { color: i === 4 ? C.coral : C.cobalt },
    });
    slide.addText(item[0], {
      x: 7.05,
      y: y + 0.08,
      w: 0.44,
      h: 0.18,
      fontFace: FONT,
      fontSize: 10,
      bold: true,
      color: C.white,
      align: "center",
      margin: 0,
    });
    slide.addText(item[1], {
      x: 7.72,
      y: y + 0.02,
      w: 2.05,
      h: 0.25,
      fontFace: FONT,
      fontSize: 14,
      bold: true,
      color: C.ink,
      margin: 0,
    });
    slide.addText(item[2], {
      x: 9.95,
      y: y + 0.04,
      w: 1.4,
      h: 0.22,
      fontFace: FONT,
      fontSize: 11,
      color: i === 4 ? C.coral : C.slate,
      margin: 0,
    });
    if (i < steps.length - 1) {
      slide.addShape(pptx.ShapeType.line, {
        x: 7.27,
        y: y + 0.45,
        w: 0,
        h: 0.34,
        line: { color: C.line, width: 2 },
      });
    }
  });
  addBox(slide, 7.02, 6.0, 5.25, 0.62, { fill: C.paleCoral, line: C.coral });
  slide.addText("缺口：没有把“哪个变量触发了检测下降”单独拿出来验证", {
    x: 7.3,
    y: 6.17,
    w: 4.7,
    h: 0.24,
    fontFace: FONT,
    fontSize: 13,
    bold: true,
    color: C.coral,
    margin: 0,
    align: "center",
  });
  addFooter(slide, "旧稿：ppt演示(1).pptx；历史结果保留，但不再按算子清单组织汇报。");
  finalizeSlide(
    slide,
    "这一页承认上次汇报的叙事问题：我把大量工作按时间顺序摆出来了，却没有一个明确的因果问题。Plan E 的低分率只是提示存在攻击面，不能告诉我们句界、AMR parser、模板匹配和统计聚合谁在起作用。"
  );
}

// Slide 3
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "同预算比较把句界重组筛成了主要诱因",
    "30 个新 C4 提示、同模型、每段 K=4、同质量门；句界臂在不读取检测分数时仍为 29/30",
    "诱因筛选"
  );
  addBar(slide, 0.85, 2.15, 6.55, "普通复述", 6.7, C.muted);
  addBar(slide, 0.85, 2.95, 6.55, "SIRA-adapted", 43.3, C.gold);
  addBar(slide, 0.85, 3.75, 6.55, "完整 Plan E", 53.3, C.cyan);
  addBar(slide, 0.85, 4.55, 6.55, "仅句界重组", 100.0, C.cobalt);
  slide.addText("有反馈 Valid ASR", {
    x: 2.8,
    y: 5.2,
    w: 4.2,
    h: 0.35,
    fontFace: FONT,
    fontSize: 13,
    bold: true,
    color: C.slate,
    align: "center",
    margin: 0,
  });
  addBox(slide, 9.55, 2.05, 2.85, 3.6, { fill: C.ink, line: C.ink });
  addMetric(slide, 9.95, 2.55, 2.05, "29/30", "无反馈 Valid ASR", C.white);
  slide.addShape(pptx.ShapeType.line, {
    x: 10.0,
    y: 3.67,
    w: 1.95,
    h: 0,
    line: { color: "52708C", width: 1 },
  });
  slide.addText("反馈选择只增加", {
    x: 9.95,
    y: 4.02,
    w: 2.05,
    h: 0.3,
    fontFace: FONT,
    fontSize: 11,
    color: "CBD5E1",
    align: "center",
    margin: 0,
  });
  slide.addText("3.3 个百分点", {
    x: 9.95,
    y: 4.35,
    w: 2.05,
    h: 0.4,
    fontFace: FONT,
    fontSize: 20,
    bold: true,
    color: "75D5DE",
    align: "center",
    margin: 0,
  });
  slide.addText("直觉转向：效果主要来自操作本身，而不是“多试几个再挑最低分”", {
    x: 9.83,
    y: 5.0,
    w: 2.3,
    h: 0.45,
    fontFace: FONT,
    fontSize: 11,
    color: "CBD5E1",
    align: "center",
    margin: 0,
  });
  addFooter(
    slide,
    "三臂主结果：普通复述 2/30、完整 Plan E 16/30、仅句界 30/30；SIRA-adapted 13/30，不是官方模型等价复现。"
  );
  finalizeSlide(
    slide,
    "这里是诱因筛选，不是最终因果实验。公平三臂推翻了我原来的直觉：更复杂的联合结构改写并不更强。仅句界重组在无反馈条件下仍是 29/30，说明检测器反馈不是主要来源。下一步必须把句界本身做成单变量。"
  );
}

// Slide 4
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "直觉：句界同时改变检测单元数和候选图作用域",
    "SWAN 的模板库 B 固定不变；变化发生在输入如何被切分、解析、匹配和聚合",
    "机制直觉"
  );
  const nodes = [
    ["文本", "固定词汇/词序"],
    ["分句", "决定 N"],
    ["BART AMR", "每单元一张图"],
    ["候选模板图", "T(R(s))"],
    ["S2Match", "与固定 B 匹配"],
    ["z 统计", "由 G 与 N 聚合"],
  ];
  nodes.forEach((n, i) => {
    const x = 0.72 + i * 2.08;
    addBox(slide, x, 2.55, 1.52, 1.12, {
      fill: i === 1 ? C.paleCoral : i === 5 ? C.paleBlue : C.white,
      line: i === 1 ? C.coral : i === 5 ? C.cobalt : C.line,
      lineWidth: i === 1 || i === 5 ? 1.5 : 1,
    });
    slide.addText(n[0], {
      x: x + 0.1,
      y: 2.78,
      w: 1.32,
      h: 0.28,
      fontFace: FONT,
      fontSize: 14,
      bold: true,
      color: i === 1 ? C.coral : C.ink,
      align: "center",
      margin: 0,
    });
    slide.addText(n[1], {
      x: x + 0.08,
      y: 3.15,
      w: 1.36,
      h: 0.22,
      fontFace: FONT,
      fontSize: 9.5,
      color: C.slate,
      align: "center",
      margin: 0,
    });
    if (i < nodes.length - 1) addArrow(slide, x + 1.56, 3.11, 0.45, 0, C.muted, 1.5);
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 2.28,
    y: 3.9,
    w: 0,
    h: 1.0,
    line: { color: C.coral, width: 2 },
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 2.28,
    y: 4.9,
    w: 8.73,
    h: 0,
    line: { color: C.coral, width: 2 },
  });
  addArrow(slide, 11.01, 4.9, 0, -0.82, C.coral, 2);
  slide.addText("路径 A：N 变小，统计上限下降", {
    x: 2.55,
    y: 4.22,
    w: 3.6,
    h: 0.35,
    fontFace: FONT,
    fontSize: 15,
    bold: true,
    color: C.coral,
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 2.28,
    y: 5.45,
    w: 5.45,
    h: 0,
    line: { color: C.cyan, width: 2 },
  });
  addArrow(slide, 7.73, 5.45, 0, -1.35, C.cyan, 2);
  slide.addText("路径 B：单图覆盖范围变大，局部模板匹配可能被稀释", {
    x: 2.55,
    y: 5.72,
    w: 6.2,
    h: 0.35,
    fontFace: FONT,
    fontSize: 15,
    bold: true,
    color: C.cyan,
    margin: 0,
  });
  addBox(slide, 9.15, 5.48, 3.2, 0.78, { fill: C.paleGray, line: C.line });
  slide.addText("不是“攻击了模板库”\n模板库 B 始终固定", {
    x: 9.42,
    y: 5.65,
    w: 2.66,
    h: 0.45,
    fontFace: FONT,
    fontSize: 12,
    bold: true,
    color: C.ink2,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "这是我后来形成的核心直觉。句界不是一个普通的语言改写特征，它位于 detector 的接口上：一方面直接决定统计样本量 N；另一方面决定每张 AMR 图覆盖多大的文本范围，进而影响 S2Match 和 green 数 G。后续实验都从这两条路径倒推。"
  );
}

// Slide 5
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "这一直觉导出四个可证伪预测",
    "每个预测都对应一个独立实验；失败时应推翻或收窄直觉",
    "实验倒推"
  );
  const preds = [
    ["P1", "只改句末标点", "固定词汇和词序时，判定仍应改变", C.cobalt],
    ["P2", "恢复 detector units", "若是 unitization，恢复五个单元应恢复判定", C.cyan],
    ["P3", "只看统计量", "N=1 时，即使全部命中也可能低于阈值", C.coral],
    ["P4", "固定 parser 输出", "扩大候选图作用域也应降低模板匹配", C.gold],
  ];
  preds.forEach((p, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.85 + col * 6.05;
    const y = 2.03 + row * 2.13;
    addBox(slide, x, y, 5.55, 1.63, { fill: C.white, line: p[3], lineWidth: 1.5 });
    slide.addShape(pptx.ShapeType.rect, {
      x,
      y,
      w: 0.88,
      h: 1.63,
      fill: { color: p[3] },
      line: { color: p[3] },
    });
    slide.addText(p[0], {
      x: x + 0.12,
      y: y + 0.54,
      w: 0.64,
      h: 0.35,
      fontFace: FONT,
      fontSize: 17,
      bold: true,
      color: C.white,
      align: "center",
      margin: 0,
    });
    slide.addText(p[1], {
      x: x + 1.18,
      y: y + 0.32,
      w: 3.95,
      h: 0.3,
      fontFace: FONT,
      fontSize: 16,
      bold: true,
      color: C.ink,
      margin: 0,
    });
    slide.addText(p[2], {
      x: x + 1.18,
      y: y + 0.85,
      w: 3.95,
      h: 0.38,
      fontFace: FONT,
      fontSize: 12.5,
      color: C.slate,
      margin: 0,
    });
  });
  slide.addText("下面四页按 P1 → P4 顺序报告证据", {
    x: 3.75,
    y: 6.25,
    w: 5.8,
    h: 0.38,
    fontFace: FONT,
    fontSize: 14,
    bold: true,
    color: C.cobalt,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "这一页把老师说的“先有直觉，再倒推实验”变成具体流程。P1 和 P2是行为因果反例，P3拆统计路径，P4只做 scorer diagnostic。四项证据强度不同，我不会把它们混成一个强因果结论。"
  );
}

// Slide 6
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "A--B--A 只改标点，并在检测端恢复原边界",
    "同一批 30 个五句短文本；固定模板库、词汇和词序",
    "P1 + P2"
  );
  const xs = [0.83, 4.78, 8.73];
  const labels = [
    ["A  原始文本", "句号边界", "默认检测为 5 个单元", C.cobalt],
    ["B  分号合句", "只替换 4 个标点", "默认检测为 1 个单元", C.coral],
    ["A′ 恢复边界", "文本不改", "向检测器提供保存的 5 个跨度", C.cyan],
  ];
  labels.forEach((d, i) => {
    addBox(slide, xs[i], 2.0, 3.55, 3.8, { fill: C.white, line: d[3], lineWidth: 1.5 });
    slide.addText(d[0], {
      x: xs[i] + 0.25,
      y: 2.28,
      w: 3.05,
      h: 0.3,
      fontFace: FONT,
      fontSize: 16,
      bold: true,
      color: d[3],
      align: "center",
      margin: 0,
    });
    addBoundaryStrip(slide, xs[i] + 0.37, 3.0, 0.47, 0.06, i === 1 ? "one" : "five");
    slide.addText(d[1], {
      x: xs[i] + 0.4,
      y: 4.02,
      w: 2.75,
      h: 0.28,
      fontFace: FONT,
      fontSize: 13,
      bold: true,
      color: C.ink2,
      align: "center",
      margin: 0,
    });
    slide.addText(d[2], {
      x: xs[i] + 0.35,
      y: 4.55,
      w: 2.85,
      h: 0.65,
      fontFace: FONT,
      fontSize: 12,
      color: C.slate,
      align: "center",
      valign: "mid",
      margin: 0,
    });
  });
  addArrow(slide, 4.42, 3.78, 0.28, 0, C.coral, 2.5);
  addArrow(slide, 8.37, 3.78, 0.28, 0, C.cyan, 2.5);
  slide.addText("词汇、词序、模板库 B 均保持不变", {
    x: 3.72,
    y: 6.18,
    w: 5.9,
    h: 0.36,
    fontFace: FONT,
    fontSize: 15,
    bold: true,
    color: C.ink,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "A-B-A 是最关键的补实验。B 条件只把四个句末标点换成分号，不替换任何词，也不改变词序。A' 不恢复原始文本，只在检测端恢复保存的五个 unit spans。如果 B 改变判定、A' 又恢复判定，触发变量就落在句界接口，而不是泛泛的改写质量。"
  );
}

// Slide 7
{
  const slide = pptx.addSlide("BASE");
  slide.background = { color: C.ink };
  slide.addText("P1 + P2  严格标点控制", {
    x: 0.75,
    y: 0.42,
    w: 3.8,
    h: 0.28,
    fontFace: FONT,
    fontSize: 11,
    bold: true,
    color: "75D5DE",
    charSpacing: 1.2,
    margin: 0,
  });
  slide.addText("判定随句界可逆翻转", {
    x: 0.75,
    y: 0.93,
    w: 7.0,
    h: 0.65,
    fontFace: FONT,
    fontSize: 31,
    bold: true,
    color: C.white,
    margin: 0,
  });
  const items = [
    [1.05, "1/30", "原始 5 单元", C.cyan],
    [5.12, "30/30", "分号后 1 单元", C.coral],
    [9.18, "1/30", "恢复 5 单元", C.cyan],
  ];
  items.forEach((d, i) => {
    slide.addShape(pptx.ShapeType.ellipse, {
      x: d[0],
      y: 2.2,
      w: 3.0,
      h: 3.0,
      fill: { color: i === 1 ? "3E2731" : "163B51" },
      line: { color: d[3], width: 3 },
    });
    slide.addText(d[1], {
      x: d[0] + 0.2,
      y: 3.0,
      w: 2.6,
      h: 0.75,
      fontFace: FONT,
      fontSize: i === 1 ? 34 : 31,
      bold: true,
      color: d[3],
      align: "center",
      margin: 0,
    });
    slide.addText(d[2], {
      x: d[0] + 0.25,
      y: 4.05,
      w: 2.5,
      h: 0.34,
      fontFace: FONT,
      fontSize: 13,
      color: "D8E2EA",
      align: "center",
      margin: 0,
    });
  });
  addArrow(slide, 4.2, 3.7, 0.75, 0, C.coral, 3);
  addArrow(slide, 8.28, 3.7, 0.75, 0, C.cyan, 3);
  slide.addText("p = 3.73e-9", {
    x: 4.13,
    y: 4.45,
    w: 0.86,
    h: 0.24,
    fontFace: FONT,
    fontSize: 10,
    color: "F8B6AF",
    align: "center",
    margin: 0,
  });
  slide.addText("p = 3.73e-9", {
    x: 8.2,
    y: 4.45,
    w: 0.84,
    h: 0.24,
    fontFace: FONT,
    fontSize: 10,
    color: "A8E3E8",
    align: "center",
    margin: 0,
  });
  slide.addText("支持的有限结论：在这组 17--46 词短文本上，SWAN 的最终判定依赖外部句界。", {
    x: 1.2,
    y: 6.1,
    w: 10.95,
    h: 0.42,
    fontFace: FONT,
    fontSize: 16,
    bold: true,
    color: C.white,
    align: "center",
    margin: 0,
  });
  slide.addText("不自动证明：所有标点版本语义严格等价；AMR 语义被攻击；正常长度同样成立。", {
    x: 1.2,
    y: 6.65,
    w: 10.95,
    h: 0.3,
    fontFace: FONT,
    fontSize: 11,
    color: "94A3B8",
    align: "center",
    margin: 0,
  });
  finalizeSlide(
    slide,
    "这是今天最重要的一页。逃逸从 1/30 变为 30/30，恢复保存边界后又回到 1/30，两个方向的配对检验都是 3.73e-9。我要把口径说窄：它证明短文本判定依赖句界，不证明所有语义等价，也不证明 AMR 本身被篡改。"
  );
}

// Slide 8
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "统计路径：单检测单元的理论上限低于阈值",
    "默认 λ=.25；预解析入口在 N=1 时仍计算 z",
    "P3"
  );
  addBox(slide, 0.82, 2.0, 5.25, 3.95, { fill: C.ink, line: C.ink });
  slide.addText("z = (G − λN) / √[Nλ(1−λ)]", {
    x: 1.2,
    y: 2.45,
    w: 4.5,
    h: 0.65,
    fontFace: FONT,
    fontSize: 25,
    bold: true,
    color: C.white,
    align: "center",
    margin: 0,
  });
  slide.addText("当 N = 1 且唯一单元为 green", {
    x: 1.4,
    y: 3.35,
    w: 4.1,
    h: 0.32,
    fontFace: FONT,
    fontSize: 13,
    color: "CBD5E1",
    align: "center",
    margin: 0,
  });
  slide.addText("zmax = √3 = 1.732", {
    x: 1.3,
    y: 3.95,
    w: 4.3,
    h: 0.65,
    fontFace: FONT,
    fontSize: 29,
    bold: true,
    color: "75D5DE",
    align: "center",
    margin: 0,
  });
  slide.addText("<  2.33 检测阈值", {
    x: 1.55,
    y: 4.83,
    w: 3.8,
    h: 0.38,
    fontFace: FONT,
    fontSize: 17,
    bold: true,
    color: "F8B6AF",
    align: "center",
    margin: 0,
  });
  slide.addText("代数充分性拆分（30 个句界选择）", {
    x: 6.75,
    y: 2.02,
    w: 4.9,
    h: 0.4,
    fontFace: FONT,
    fontSize: 17,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  addBox(slide, 6.73, 2.75, 5.25, 1.18, { fill: C.paleCoral, line: C.coral });
  addMetric(slide, 7.05, 2.98, 1.5, "21/30", "只替换 N", C.coral);
  slide.addText("保持原命中率", {
    x: 9.05,
    y: 3.1,
    w: 2.2,
    h: 0.28,
    fontFace: FONT,
    fontSize: 13,
    color: C.ink2,
    margin: 0,
  });
  addBox(slide, 6.73, 4.2, 5.25, 1.18, { fill: C.paleCyan, line: C.cyan });
  addMetric(slide, 7.05, 4.43, 1.5, "30/30", "只替换 green-rate", C.cyan);
  slide.addText("保持原句数", {
    x: 9.05,
    y: 4.55,
    w: 2.2,
    h: 0.28,
    fontFace: FONT,
    fontSize: 13,
    color: C.ink2,
    margin: 0,
  });
  addBox(slide, 6.73, 5.72, 5.25, 0.63, { fill: C.paleGray, line: C.line });
  slide.addText("这两行允许非整数虚拟 G，只说明数值充分性，不是文本因果估计。", {
    x: 6.98,
    y: 5.92,
    w: 4.75,
    h: 0.22,
    fontFace: FONT,
    fontSize: 10.5,
    color: C.slate,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "P3 得到确定的统计解释：N=1 时，即使唯一单元命中，z 最高也只有 1.732，低于 2.33。对真实句界选择做代数替换，句数变化单独对 21/30 数值上足够，green-rate 变化对 30/30 足够。但这是充分性分析，不是两个文本反事实。"
  );
}

// Slide 9
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "匹配路径：候选图扩大后局部模板分数被稀释",
    "确定性图组合诊断固定了组成图与模板库，但人工组合图不是自然文本的 gold AMR",
    "P4"
  );
  const bars = [
    ["原句级候选图", 0.8197, "108/108 green", C.cobalt],
    ["人工组合图 D(g₁,g₂)", 0.4502, "0/108 green", C.coral],
    ["真实合句重解析图", 0.5200, "8/108 green", C.cyan],
  ];
  bars.forEach((d, i) => {
    const x = 1.08 + i * 3.9;
    slide.addShape(pptx.ShapeType.rect, {
      x,
      y: 5.58 - d[1] * 3.55,
      w: 2.2,
      h: d[1] * 3.55,
      fill: { color: d[3] },
      line: { color: d[3] },
    });
    slide.addText(d[1].toFixed(4), {
      x,
      y: 5.18 - d[1] * 3.55,
      w: 2.2,
      h: 0.32,
      fontFace: FONT,
      fontSize: 16,
      bold: true,
      color: d[3],
      align: "center",
      margin: 0,
    });
    slide.addText(d[0], {
      x: x - 0.18,
      y: 5.85,
      w: 2.56,
      h: 0.48,
      fontFace: FONT,
      fontSize: 11.5,
      bold: true,
      color: C.ink2,
      align: "center",
      margin: 0,
    });
    slide.addText(d[2], {
      x: x - 0.18,
      y: 6.42,
      w: 2.56,
      h: 0.28,
      fontFace: FONT,
      fontSize: 10.5,
      color: d[3],
      align: "center",
      margin: 0,
    });
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.85,
    y: 3.275,
    w: 11.4,
    h: 0,
    line: { color: C.gold, width: 1.5, dash: "dash" },
  });
  slide.addText("S2Match green 阈值 0.65", {
    x: 9.8,
    y: 3.0,
    w: 2.3,
    h: 0.25,
    fontFace: FONT,
    fontSize: 10,
    color: C.gold,
    align: "right",
    margin: 0,
  });
  addBox(slide, 8.63, 1.95, 3.6, 0.75, { fill: C.paleGray, line: C.line });
  slide.addText("precision 中位下降 0.5048\nrecall 中位下降 0.1123", {
    x: 8.92,
    y: 2.12,
    w: 3.0,
    h: 0.42,
    fontFace: FONT,
    fontSize: 11.5,
    bold: true,
    color: C.ink2,
    align: "center",
    margin: 0,
  });
  addFooter(
    slide,
    "口径：该实验只说明 whole-graph S2Match 对候选图作用域敏感；不能据此排除真实攻击中的 parser drift。"
  );
  finalizeSlide(
    slide,
    "P4 的证据等级比 A-B-A 低。把两张保存图确定性组合后，中位 F1 从 0.8197 降到 0.4502，108 个已接受目标全部变成 non-green。这验证了 scorer 的作用域敏感性。但人工组合图不是自然文本 parser 输出，所以不能说真实攻击完全不需要 parser drift。"
  );
}

// Slide 10
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "范围对照表明它不是通用水印移除方法",
    "三种水印分别生成自己的水印文本；候选选择在读取目标检测分数前冻结",
    "结论边界"
  );
  addBar(slide, 0.95, 2.35, 6.55, "SWAN", 96.7, C.cobalt);
  addBar(slide, 0.95, 3.35, 6.55, "KGW", 0.0, C.muted);
  addBar(slide, 0.95, 4.35, 6.55, "SynthID-Text", 3.3, C.cyan);
  slide.addText("无反馈 Valid ASR", {
    x: 3.15,
    y: 5.08,
    w: 4.25,
    h: 0.35,
    fontFace: FONT,
    fontSize: 13,
    bold: true,
    color: C.slate,
    align: "center",
    margin: 0,
  });
  addBox(slide, 9.6, 2.15, 2.65, 3.35, { fill: C.white, line: C.line });
  slide.addText("可以说", {
    x: 9.9,
    y: 2.5,
    w: 2.05,
    h: 0.28,
    fontFace: FONT,
    fontSize: 15,
    bold: true,
    color: C.green,
    align: "center",
    margin: 0,
  });
  slide.addText("结果与 SWAN 的句级检测管线特异性相容", {
    x: 9.9,
    y: 2.95,
    w: 2.05,
    h: 0.78,
    fontFace: FONT,
    fontSize: 13,
    color: C.ink2,
    align: "center",
    valign: "mid",
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 9.95,
    y: 3.95,
    w: 1.95,
    h: 0,
    line: { color: C.line, width: 1 },
  });
  slide.addText("不能说", {
    x: 9.9,
    y: 4.22,
    w: 2.05,
    h: 0.28,
    fontFace: FONT,
    fontSize: 15,
    bold: true,
    color: C.coral,
    align: "center",
    margin: 0,
  });
  slide.addText("已经证明强系统特异性或其他水印普遍鲁棒", {
    x: 9.9,
    y: 4.62,
    w: 2.05,
    h: 0.58,
    fontFace: FONT,
    fontSize: 11.5,
    color: C.slate,
    align: "center",
    valign: "mid",
    margin: 0,
  });
  addBox(slide, 1.25, 5.82, 10.55, 0.62, { fill: C.paleCoral, line: C.coral });
  slide.addText("限制：SWAN 平均 30.9 词；KGW / SynthID 约 162.8 / 160.5 词，生成模型和长度未完全匹配。", {
    x: 1.55,
    y: 6.02,
    w: 9.95,
    h: 0.24,
    fontFace: FONT,
    fontSize: 12,
    color: C.coral,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "范围对照不是为了证明 KGW 和 SynthID 安全，而是防止把句界方法包装成通用水印攻击。当前设置下 SWAN 是 96.7%，KGW 是 0，SynthID 是 3.3%。因为长度和生成模型不匹配，这只能作为边界线索，不能做强系统特异性因果主张。"
  );
}

// Slide 11
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "防御探索没有闭环，正常长度因果端点也未补齐",
    "所有路线按预注册门停止；没有把“接近通过”改写成正结果",
    "负结果"
  );
  const rows = [
    ["局部多实例证据", "开发集三种表示均未恢复", "0/15", "STOP"],
    ["模板特异经验尾", "原图 4/15；重解析 2/15", "< 8/15 双门", "STOP"],
    ["关系锚定 motif", "原图 41/56；重解析 33/56", "各差 1 个", "STOP"],
    ["直接重新分句", "有效逃逸 23/30 → 19/30", "p = .125", "线索"],
    ["句数守恒构造", "23/30 可合并；0/30 可自然拆分", "未运行 detector", "STOP"],
  ];
  const y0 = 2.0;
  rows.forEach((r, i) => {
    const y = y0 + i * 0.83;
    addBox(slide, 0.78, y, 11.78, 0.64, {
      fill: i % 2 === 0 ? C.white : C.paleGray,
      line: C.line,
      lineWidth: 0.6,
    });
    slide.addText(r[0], {
      x: 1.02,
      y: y + 0.17,
      w: 2.15,
      h: 0.22,
      fontFace: FONT,
      fontSize: 12.5,
      bold: true,
      color: C.ink,
      margin: 0,
    });
    slide.addText(r[1], {
      x: 3.35,
      y: y + 0.17,
      w: 4.15,
      h: 0.22,
      fontFace: FONT,
      fontSize: 11.5,
      color: C.ink2,
      margin: 0,
    });
    slide.addText(r[2], {
      x: 7.75,
      y: y + 0.17,
      w: 2.15,
      h: 0.22,
      fontFace: FONT,
      fontSize: 11.5,
      bold: true,
      color: i === 3 ? C.gold : C.coral,
      margin: 0,
    });
    addStatus(slide, 10.85, y + 0.18, r[3], i === 3 ? "warn" : "stop");
  });
  addBox(slide, 0.78, 6.32, 11.78, 0.5, { fill: C.paleBlue, line: C.cobalt });
  slide.addText("因此当前只能主张“句界诱因已定位”，不能主张“已有有效分割不变防御”或“正常长度机制已确认”。", {
    x: 1.02,
    y: 6.47,
    w: 11.3,
    h: 0.22,
    fontFace: FONT,
    fontSize: 12.5,
    bold: true,
    color: C.cobalt,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "这页解释为什么后面没有一个漂亮的 SWAN 防御结果。三条局部证据路线都没过开发门；重新分句把 23/30 降到 19/30，但 p=.125 且误报也上升；句数守恒对照在当前短文本里连自然拆分点都构造不出来。它们是负结果和设计约束。"
  );
}

// Slide 12
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "我把直觉进一步形式化为可认证重分句问题",
    "这是独立的 SemStamp 防御证据流，不补 SWAN 的正常长度端点",
    "后续工作"
  );
  addBox(slide, 0.75, 2.02, 3.25, 3.95, { fill: C.white, line: C.coral, lineWidth: 1.5 });
  slide.addText("SWAN 系统审计", {
    x: 1.05,
    y: 2.33,
    w: 2.65,
    h: 0.34,
    fontFace: FONT,
    fontSize: 18,
    bold: true,
    color: C.coral,
    align: "center",
    margin: 0,
  });
  slide.addText("whole-graph S2Match\n难以分解 exact certificate\n\n短文本 A--B--A 已成立\n正常长度机制未完成", {
    x: 1.15,
    y: 3.02,
    w: 2.45,
    h: 1.75,
    fontFace: FONT,
    fontSize: 13,
    color: C.ink2,
    align: "center",
    valign: "mid",
    margin: 0,
    breakLine: false,
  });
  addStatus(slide, 1.95, 5.17, "审计终态", "warn");
  addArrow(slide, 4.2, 3.95, 1.15, 0, C.cobalt, 2.5);
  slide.addText("Carrier--Decoder\nContract", {
    x: 4.15,
    y: 3.14,
    w: 1.25,
    h: 0.55,
    fontFace: FONT,
    fontSize: 11,
    bold: true,
    color: C.cobalt,
    align: "center",
    margin: 0,
  });
  addBox(slide, 5.55, 2.02, 6.98, 3.95, { fill: C.white, line: C.cobalt, lineWidth: 1.5 });
  slide.addText("SemStamp certified defense", {
    x: 5.9,
    y: 2.33,
    w: 6.25,
    h: 0.34,
    fontFace: FONT,
    fontSize: 18,
    bold: true,
    color: C.cobalt,
    align: "center",
    margin: 0,
  });
  const stages = [
    ["v1", "fixed-byte renderer", "42/3000 不可表达句界", "STOP"],
    ["v2", "refined atoms", "3000/3000 逐字节恢复", "GO"],
    ["v2", "detector health", "FPR 4.7% / TPR 95.4%", "GO"],
    ["v3", "variable-N core", "133,713,262-pair 验证", "GO"],
    ["v3", "full profile", "77.6152 > 72 GPU-h", "STOP"],
  ];
  stages.forEach((s, i) => {
    const y = 2.95 + i * 0.55;
    slide.addText(s[0], {
      x: 5.92,
      y,
      w: 0.45,
      h: 0.22,
      fontFace: FONT,
      fontSize: 10.5,
      bold: true,
      color: C.slate,
      margin: 0,
    });
    slide.addText(s[1], {
      x: 6.48,
      y,
      w: 1.75,
      h: 0.22,
      fontFace: FONT,
      fontSize: 10.5,
      bold: true,
      color: C.ink,
      margin: 0,
    });
    slide.addText(s[2], {
      x: 8.35,
      y,
      w: 2.55,
      h: 0.22,
      fontFace: FONT,
      fontSize: 10.2,
      color: C.ink2,
      margin: 0,
    });
    addStatus(slide, 11.15, y - 0.02, s[3], s[3] === "GO" ? "go" : "stop");
  });
  addBox(slide, 1.3, 6.33, 10.75, 0.5, { fill: C.paleCoral, line: C.coral });
  slide.addText("最终没有生成 certificate：失败的是资源 readiness，不是授权不足，也不是已证明 SemStamp 不鲁棒。", {
    x: 1.55,
    y: 6.48,
    w: 10.25,
    h: 0.22,
    fontFace: FONT,
    fontSize: 12.5,
    bold: true,
    color: C.coral,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "后面的工作是在做什么：我没有继续堆 SWAN 算子，而是把句界直觉抽象成可认证 threat set。因为 SemStamp 的相邻 pair score 可分解，而且 v17 detector 健康，所以转到 SemStamp。renderer、detector health 和 variable-N core 都通过，最后全量资源 profile 是 77.6152 小时，超过 72 小时门，certificate 没启动。"
  );
}

// Slide 13
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "当前结论：诱因已定位，完整防御证书仍未产生",
    "把已支持、未支持和需要导师判断的内容分开",
    "总结"
  );
  const cols = [
    [0.77, "已支持", C.green, C.paleGreen, [
      "短文本 SWAN 判定对句界可逆敏感",
      "N 统计路径有确定代数解释",
      "whole-graph S2Match 对图作用域敏感",
    ]],
    [4.47, "未支持", C.coral, C.paleCoral, [
      "正常长度、句数守恒的因果端点",
      "通用跨水印攻击或强系统特异性",
      "有效 segmentation-invariant 防御",
    ]],
    [8.17, "后续选择", C.cobalt, C.paleBlue, [
      "论文主线收敛为 SWAN 系统审计",
      "认证防御作为独立工程证据流",
      "只有新版本资源方案过门才重启 certificate",
    ]],
  ];
  cols.forEach((c) => {
    addBox(slide, c[0], 2.0, 3.45, 3.95, { fill: c[3], line: c[2], lineWidth: 1.5 });
    slide.addText(c[1], {
      x: c[0] + 0.25,
      y: 2.33,
      w: 2.95,
      h: 0.35,
      fontFace: FONT,
      fontSize: 19,
      bold: true,
      color: c[2],
      align: "center",
      margin: 0,
    });
    c[4].forEach((t, i) => {
      slide.addShape(pptx.ShapeType.ellipse, {
        x: c[0] + 0.38,
        y: 3.08 + i * 0.78,
        w: 0.16,
        h: 0.16,
        fill: { color: c[2] },
        line: { color: c[2] },
      });
      slide.addText(t, {
        x: c[0] + 0.7,
        y: 2.98 + i * 0.78,
        w: 2.4,
        h: 0.5,
        fontFace: FONT,
        fontSize: 12.3,
        color: C.ink2,
        margin: 0,
        breakLine: false,
      });
    });
  });
  slide.addText("希望老师确认：明天之后是否把论文叙事固定为“SWAN 句界接口审计”，停止继续追逐一个尚未闭环的防御正结果？", {
    x: 1.05,
    y: 6.3,
    w: 11.2,
    h: 0.52,
    fontFace: FONT,
    fontSize: 15,
    bold: true,
    color: C.ink,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "最后不要把所有工作汇成一个模糊的“大工程”。我建议向老师明确三层：已支持的是短文本句界依赖；未支持的是正常长度、通用性和有效防御；后续选择是把论文收敛为系统审计，认证防御保留为独立证据流。这里主动请老师判断主线是否应就此冻结。"
  );
}

// Slide 14
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "附录：从 Plan A 到 Plan AE 的证据演进",
    "不是 31 个并列实验，而是三次研究问题收敛",
    "附录"
  );
  const phases = [
    [0.85, 3.08, 3.4, C.muted, "攻击搜索", "A--F", "从自然改写到反馈搜索\n发现攻击面，纠正有效率口径"],
    [4.58, 3.08, 3.4, C.cobalt, "诱因与因果", "G--P", "三臂筛选句界\nA--B--A、统计拆分、范围审计"],
    [8.31, 3.08, 3.95, C.cyan, "形式化与认证", "Q--AE", "exact DP、detector health\nvariable-N core 与资源 readiness"],
  ];
  phases.forEach((p, i) => {
    addBox(slide, p[0], p[1], p[2], 2.05, {
      fill: i === 0 ? C.paleGray : i === 1 ? C.paleBlue : C.paleCyan,
      line: p[3],
      lineWidth: 1.5,
    });
    slide.addText(p[4], {
      x: p[0] + 0.25,
      y: p[1] + 0.3,
      w: p[2] - 0.5,
      h: 0.34,
      fontFace: FONT,
      fontSize: 18,
      bold: true,
      color: p[3],
      align: "center",
      margin: 0,
    });
    slide.addText(p[5], {
      x: p[0] + 0.25,
      y: p[1] + 0.78,
      w: p[2] - 0.5,
      h: 0.28,
      fontFace: FONT,
      fontSize: 13,
      bold: true,
      color: C.ink,
      align: "center",
      margin: 0,
    });
    slide.addText(p[6], {
      x: p[0] + 0.35,
      y: p[1] + 1.22,
      w: p[2] - 0.7,
      h: 0.55,
      fontFace: FONT,
      fontSize: 11.5,
      color: C.slate,
      align: "center",
      margin: 0,
      breakLine: false,
    });
    if (i < phases.length - 1) addArrow(slide, p[0] + p[2] + 0.12, 4.1, 0.2, 0, C.muted, 2);
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 1.05,
    y: 2.52,
    w: 10.95,
    h: 0,
    line: { color: C.line, width: 2 },
  });
  [1.35, 5.08, 8.81, 11.75].forEach((x, i) => {
    slide.addShape(pptx.ShapeType.ellipse, {
      x,
      y: 2.39,
      w: 0.26,
      h: 0.26,
      fill: { color: i === 0 ? C.muted : i === 1 ? C.cobalt : i === 2 ? C.cyan : C.coral },
      line: { color: i === 0 ? C.muted : i === 1 ? C.cobalt : i === 2 ? C.cyan : C.coral },
    });
  });
  slide.addText("搜索", { x: 0.95, y: 2.0, w: 0.8, h: 0.25, fontFace: FONT, fontSize: 10, color: C.slate, align: "center", margin: 0 });
  slide.addText("诱因", { x: 4.68, y: 2.0, w: 0.8, h: 0.25, fontFace: FONT, fontSize: 10, color: C.cobalt, align: "center", margin: 0 });
  slide.addText("认证", { x: 8.41, y: 2.0, w: 0.8, h: 0.25, fontFace: FONT, fontSize: 10, color: C.cyan, align: "center", margin: 0 });
  slide.addText("资源 Stop", { x: 11.22, y: 2.0, w: 1.35, h: 0.25, fontFace: FONT, fontSize: 10, color: C.coral, align: "center", margin: 0 });
  addFooter(slide, "完整阶段叙事见：/Users/bytedance/AMR/叙事.md。");
  finalizeSlide(
    slide,
    "如果老师问后面为什么做了这么多版本，用这一页回答：不是 31 个平行实验，而是三次收敛。A-F 找攻击面，G-P 找诱因与因果，Q-AE 把句界问题形式化成认证问题。每一段都按 Go/Stop 门结束。"
  );
}

// Slide 15
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "附录：SemStamp 健康门与 variable-N core 已通过",
    "资源 Stop 之前，detector identity、算法正确性和最大规模均已有独立证据",
    "附录"
  );
  const metrics = [
    ["test FPR", "47/1000", "4.7%", C.cobalt],
    ["positive TPR", "954/1000", "95.4%", C.green],
    ["systematic oracle", "5000/5000", "PASS", C.cyan],
    ["random oracle", "2000/2000", "PASS", C.cyan],
    ["fixed-N reduction", "1000/1000", "PASS", C.cyan],
    ["maximum scale", "133,713,262", "pairs", C.coral],
  ];
  metrics.forEach((m, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.78 + col * 4.18;
    const y = 2.05 + row * 2.0;
    addBox(slide, x, y, 3.65, 1.55, { fill: C.white, line: m[3], lineWidth: 1.2 });
    slide.addText(m[0], {
      x: x + 0.25,
      y: y + 0.22,
      w: 3.15,
      h: 0.25,
      fontFace: FONT,
      fontSize: 11,
      bold: true,
      color: C.slate,
      align: "center",
      margin: 0,
    });
    slide.addText(m[1], {
      x: x + 0.25,
      y: y + 0.58,
      w: 3.15,
      h: 0.46,
      fontFace: FONT,
      fontSize: m[1].length > 10 ? 22 : 26,
      bold: true,
      color: m[3],
      align: "center",
      margin: 0,
    });
    slide.addText(m[2], {
      x: x + 0.25,
      y: y + 1.13,
      w: 3.15,
      h: 0.22,
      fontFace: FONT,
      fontSize: 10,
      color: C.slate,
      align: "center",
      margin: 0,
    });
  });
  addBox(slide, 1.5, 6.0, 10.3, 0.6, { fill: C.paleBlue, line: C.cobalt });
  slide.addText("完整 profile 还保留了 3000/3000 行、42/42 risk rows、59/59 zero-byte gaps 和全部 46,501 个 target layers。", {
    x: 1.8,
    y: 6.18,
    w: 9.7,
    h: 0.24,
    fontFace: FONT,
    fontSize: 12,
    bold: true,
    color: C.cobalt,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "这一页只在老师追问“资源 Stop 前到底做成了什么”时用。Detector health 的 FPR/TPR 健康，variable-N core 对 oracle 和 fixed-N reduction 全通过，最大规模超过 1.33 亿 pairs。Stop 不是因为算法结果错，而是全量特征投影预算超门。"
  );
}

// Slide 16
{
  const slide = pptx.addSlide("BASE");
  addTitle(
    slide,
    "附录：最终停止项是 77.6152 GPU-hours 的资源门",
    "两条冻结公式分别为 76.5346 和 77.6152 GPU-hours；取保守上界",
    "附录"
  );
  slide.addText("72 h", {
    x: 9.46,
    y: 1.96,
    w: 0.75,
    h: 0.25,
    fontFace: FONT,
    fontSize: 11,
    bold: true,
    color: C.coral,
    align: "center",
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 9.84,
    y: 2.28,
    w: 0,
    h: 3.35,
    line: { color: C.coral, width: 2, dash: "dash" },
  });
  const scaleW = 10.25;
  const startX = 1.0;
  const maxH = 85;
  const values = [
    ["公式 A", 76.5346, C.cyan],
    ["公式 B", 77.6152, C.cobalt],
  ];
  values.forEach((v, i) => {
    const y = 2.65 + i * 1.25;
    slide.addText(v[0], {
      x: startX,
      y: y + 0.03,
      w: 1.0,
      h: 0.3,
      fontFace: FONT,
      fontSize: 13,
      bold: true,
      color: C.ink,
      margin: 0,
    });
    slide.addShape(pptx.ShapeType.rect, {
      x: startX + 1.15,
      y,
      w: scaleW,
      h: 0.4,
      fill: { color: C.paleGray },
      line: { color: C.paleGray },
    });
    slide.addShape(pptx.ShapeType.rect, {
      x: startX + 1.15,
      y,
      w: (scaleW * v[1]) / maxH,
      h: 0.4,
      fill: { color: v[2] },
      line: { color: v[2] },
    });
    slide.addText(`${v[1].toFixed(4)} h`, {
      x: 9.72,
      y: y + 0.02,
      w: 1.32,
      h: 0.28,
      fontFace: FONT,
      fontSize: 12,
      bold: true,
      color: C.white,
      align: "right",
      margin: 0,
    });
  });
  addBox(slide, 1.0, 5.45, 11.2, 0.84, { fill: C.white, line: C.line });
  slide.addText("主要预算来源", {
    x: 1.3,
    y: 5.72,
    w: 1.65,
    h: 0.24,
    fontFace: FONT,
    fontSize: 12,
    bold: true,
    color: C.slate,
    margin: 0,
  });
  slide.addText("canonical batch-64 特征投影 128,650.4 秒 / replay", {
    x: 3.05,
    y: 5.72,
    w: 4.55,
    h: 0.24,
    fontFace: FONT,
    fontSize: 12,
    bold: true,
    color: C.cobalt,
    margin: 0,
  });
  slide.addText("超过硬门 5.6152 h（7.80%）", {
    x: 8.35,
    y: 5.72,
    w: 3.25,
    h: 0.24,
    fontFace: FONT,
    fontSize: 12,
    bold: true,
    color: C.coral,
    align: "right",
    margin: 0,
  });
  addBox(slide, 1.0, 6.48, 11.2, 0.42, { fill: C.paleCoral, line: C.coral });
  slide.addText("batch 128 虽快 1.0333×，但会改变冻结 region array，因此不能用来跨过资源门。", {
    x: 1.25,
    y: 6.59,
    w: 10.7,
    h: 0.22,
    fontFace: FONT,
    fontSize: 11.5,
    color: C.coral,
    align: "center",
    margin: 0,
  });
  addFooter(slide);
  finalizeSlide(
    slide,
    "公式 A 和 B 都超过 72 小时，保守上界是 77.6152，超 5.6152 小时。主要成本不是 C++ core，而是 canonical batch-64 的 MPNet 特征投影。Batch 128 虽略快，但 maximum workload 上 region array 发生漂移，所以协议不允许拿它救预算。"
  );
}

async function main() {
  await pptx.writeFile({ fileName: "lab_update_sentence_boundary_20261010.pptx" });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
