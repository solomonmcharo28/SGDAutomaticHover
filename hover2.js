// ---------- Helpers ----------
//* SPDX-License-Identifier: MIT
function clamp01(x){ return Math.max(0, Math.min(1, x)); }
function clamp255(x){ return Math.round(Math.max(0, Math.min(255, x))); }

function parseRGB(text){
  const parts = text.split(",").map(s => s.trim()).filter(Boolean);
  if (parts.length !== 3) return null;
  const nums = parts.map(n => Number(n));
  if (nums.some(n => Number.isNaN(n))) return null;
  return nums.map(clamp255);
}

function rgbToCss([r,g,b]){ return `rgb(${r}, ${g}, ${b})`; }
function rgb01([r,g,b]){ return [r/255, g/255, b/255]; }
function rgb255([r,g,b]){ return [clamp255(r), clamp255(g), clamp255(b)]; }

// Relative luminance (sRGB -> linear)
function srgbToLinear(c){
  return (c <= 0.04045) ? (c/12.92) : Math.pow((c+0.055)/1.055, 2.4);
}
function relLuminance(rgb01v){
  const [r,g,b] = rgb01v.map(srgbToLinear);
  return 0.2126*r + 0.7152*g + 0.0722*b;
}

// WCAG-ish contrast ratio
function contrastRatio(L1, L2){
  const lighter = Math.max(L1, L2);
  const darker  = Math.min(L1, L2);
  return (lighter + 0.05) / (darker + 0.05);
}

// ---------- Differentiable model pieces ----------
/**
 * Hover color as a differentiable function H(B).
 * We'll do: H = mix(B, target, t) where target is a slightly darker + slightly shifted color.
 * (This makes H depend on B smoothly, so “chain rule” applies.)
 */
function hoverFromBase(base01, theta){
  // theta = [t, fr, fg, fb]
  const t  = clamp01(theta[0]);
  const fr = clamp01(theta[1]);
  const fg = clamp01(theta[2]);
  const fb = clamp01(theta[3]);

  const [r,g,b] = base01;

  // "previous algorithm" target, now driven by theta
  const target = [
    clamp01(r * fr),
    clamp01(g * fg),
    clamp01(b * fb),
  ];

  // mix(base, target, t)
  return [
    (1 - t) * r + t * target[0],
    (1 - t) * g + t * target[1],
    (1 - t) * b + t * target[2],
  ];
}


/**
 * Text color parameterization (differentiable):
 * represent text color in HSV-ish (we'll just use unconstrained RGB params through sigmoid)
 * so SGD can move smoothly.
 */
function sigmoid(x){ return 1 / (1 + Math.exp(-x)); }
function textColorFromParams(p){ // p: [pr,pg,pb] in R
  return [sigmoid(p[0]), sigmoid(p[1]), sigmoid(p[2])]; // in [0,1]
}

/**
 * Loss to maximize contrast: we minimize negative contrast + mild regularization.
 * We want high contrast between background and text.
 */
function lossForText(bg01, text01){
  const Lbg = relLuminance(bg01);
  const Ltx = relLuminance(text01);
  const cr = contrastRatio(Lbg, Ltx);

  const targetContrast = 6.0;
  const contrastLoss = Math.pow(Math.max(0, targetContrast - cr), 2);

  // NEW: color target = complementary hue of background, reasonably saturated
  const [h,s,v] = rgbToHsv(bg01);
  const targetH = (h + 0.5) % 1;              // +180°
  const targetS = clamp01(Math.max(0.55, s)); // keep some saturation
  // pick a V that tends to be readable (not perfect, contrastLoss enforces readability)
  const targetV = (Lbg > 0.5) ? 0.15 : 0.95;  // dark text on light bg, light on dark bg
  const targetRgb = hsvToRgb([targetH, targetS, targetV]);

  // “style” term: pull toward target color so it actually changes with bg
  const styleWeight = 0.18; // increase if you want more variation
  const styleLoss =
    (text01[0]-targetRgb[0])**2 +
    (text01[1]-targetRgb[1])**2 +
    (text01[2]-targetRgb[2])**2;

  // small regularization
  const reg = 0.0015 * (text01[0]**2 + text01[1]**2 + text01[2]**2);

  return contrastLoss + styleWeight * styleLoss + reg;
}


/**
 * Numerical gradient (finite differences) for SGD.
 * Good enough for UI. Keeps everything simple.
 */
function gradNumerical(bg01, params){
  const eps = 1e-3;
  const baseLoss = lossForText(bg01, textColorFromParams(params));
  const g = [0,0,0];
  for (let i=0;i<3;i++){
    const p2 = params.slice();
    p2[i] += eps;
    const l2 = lossForText(bg01, textColorFromParams(p2));
    g[i] = (l2 - baseLoss) / eps;
  }
  return { baseLoss, g };
}

/**
 * SGD optimize text params given background.
 */
function optimizeText(bg01, {
  steps = 160,
  lr = 0.35,
  initParams = [0,0,0]
} = {}){
  let p = initParams.slice();

  for (let s=0; s<steps; s++){
    const { g } = gradNumerical(bg01, p);
    // SGD update
    p = p.map((v, i) => v - lr * g[i]);
    // mild lr decay
    if (s === 60) lr *= 0.6;
    if (s === 110) lr *= 0.7;
  }
  return { params: p, color01: textColorFromParams(p), loss: lossForText(bg01, textColorFromParams(p)) };
}
// ---------- Hover-parameter SGD (decide hover via SGD, then call hoverFromBase) ----------
function lossForHoverTheta(base01, theta){
  const hover01 = hoverFromBase(base01, theta);
  

  // A) Make hover noticeably different from base (push toward target diff)
const diff =
  (hover01[0]-base01[0])**2 +
  (hover01[1]-base01[1])**2 +
  (hover01[2]-base01[2])**2;

// Target squared-distance in RGB01 space (tune this)
const targetDiff = 0.06; // try 0.06 → 0.12 for stronger contrast
const diffLoss = Math.pow(diff - targetDiff, 2);

  // B) Avoid hover being too dark or too bright
  const Lb = relLuminance(base01);
  const Lh = relLuminance(hover01);
const lumDiff = Math.abs(Lh - Lb);

// push luminance difference too
const targetLum = 0.25; // strong brightness shift
const lumLoss = Math.pow(lumDiff - targetLum, 2);
  const extremeLoss =
    Math.pow(Math.max(0, 0.08 - Lh), 2) +
    Math.pow(Math.max(0, Lh - 0.92), 2);

  // C) Keep theta in a sane neighborhood (regularization)
  const reg = 0.002 * (
    (theta[0]-0.18)**2 +  // t
    (theta[1]-0.82)**2 +  // fr
    (theta[2]-0.86)**2 +  // fg
    (theta[3]-0.92)**2    // fb
  );

  return diffLoss + 0.6 * lumLoss + 0.2 * extremeLoss + reg;
}

function gradThetaNumerical(base01, theta){
  const eps = 1e-3;
  const baseLoss = lossForHoverTheta(base01, theta);
  const g = [0,0,0,0];
  for (let i=0;i<4;i++){
    const t2 = theta.slice();
    t2[i] += eps;
    const l2 = lossForHoverTheta(base01, t2);
    g[i] = (l2 - baseLoss) / eps;
  }
  return { baseLoss, g };
}

function optimizeHoverTheta(base01, {
  steps = 900,
  lr = 1.4,
  initTheta = [0.18, 0.82, 0.86, 0.92]
} = {}){
  let theta = initTheta.slice();

  for (let s=0; s<steps; s++){
    const { g } = gradThetaNumerical(base01, theta);
    theta = theta.map((v, i) => v - lr * g[i]);

    // constrain
    theta[0] = Math.max(0.35, clamp01(theta[0]));              // t
    theta[1] = clamp01(Math.max(0.35, theta[1]));;// fr
    theta[2] = clamp01(Math.max(0.35, theta[2])); // fg
    theta[3] = clamp01(Math.max(0.35, theta[3]));// fb

    if (s === 60) lr *= 0.6;
    if (s === 110) lr *= 0.7;
  }

  return { theta, loss: lossForHoverTheta(base01, theta) };
}


// ---------- Chain-rule pipeline ----------
function applyChain(baseRgb255){
  const base01 = rgb01(baseRgb255);

  // 1) base button bg = B (from form)

  // 2) base text chosen by SGD: T_base
  const baseText = optimizeText(base01, { initParams: [0.5, 0.5, 0.5] });

  // 3) hover parameters decided by SGD
  const hoverOpt = optimizeHoverTheta(base01);

  // 4) hover bg computed by calling the previous hover algorithm with theta
  const hover01 = hoverFromBase(base01, hoverOpt.theta);

  // 5) hover text chosen by SGD based on hover bg
  const hoverText = optimizeText(hover01, { initParams: baseText.params });

  // 6) page background (optional)
  const pageBg01 = [
    clamp01(base01[0] * 0.10),
    clamp01(base01[1] * 0.10),
    clamp01(base01[2] * 0.10),
  ];

  return { base01, baseText, hover01, hoverText, pageBg01, hoverOpt };
}


// ---------- DOM wiring ----------
const form = document.getElementById("colorForm");
const rgbInput = document.getElementById("rgbInput");
const cta = document.getElementById("cta");
const readout = document.getElementById("readout");

function setCssVars({ base01, baseText, hover01, hoverText, pageBg01, hoverOpt }){

  const base255 = rgb255(base01.map(v => v*255));
  const baseText255 = rgb255(baseText.color01.map(v => v*255));
  const hover255 = rgb255(hover01.map(v => v*255));
  console.log(hover255.toString());
  const hoverText255 = rgb255(hoverText.color01.map(v => v*255));
  const pageBg255 = rgb255(pageBg01.map(v => v*255));

  document.documentElement.style.setProperty("--btn-bg", rgbToCss(base255));
  document.documentElement.style.setProperty("--btn-fg", rgbToCss(baseText255));
  document.documentElement.style.setProperty("--btn-hover-bg", rgbToCss(hover255));
  document.documentElement.style.setProperty("--btn-hover-fg", rgbToCss(hoverText255));
  document.documentElement.style.setProperty("--page-bg", rgbToCss(pageBg255));
const th = hoverOpt.theta.map(x => Number(x).toFixed(4)).join(", ");

readout.textContent =
`=== BASE STATE =================================

Base text (T₀):                 ${rgbToCss(baseText255)}
Base background (B₀, derived):  ${rgbToCss(baseBg255)}

Base hover text (SGD):
  T₀ʰ = ${rgbToCss(baseHoverText255)}
  loss = ${baseHoverText.loss.toFixed(4)}

=== HOVER STATE ================================

Hover parameters (θ):
  [${th}]
  loss = ${hoverOpt.loss.toFixed(4)}

Hover background (B₁):          ${rgbToCss(hoverBg255)}

Hover text (SGD):
  T₁ = ${rgbToCss(hoverText255)}
  loss = ${hoverText.loss.toFixed(4)}

=== TRANSFORMATION CHAIN =======================

T₀
 → derive B₀
 → SGD(T₀, B₀) = T₀ʰ
 → derive B₁
 → SGD(T₀ʰ, B₁) = T₁
`;

}

function applyFromInput(){
  const rgb = parseRGB(rgbInput.value);
  if (!rgb){
    readout.textContent = "Invalid RGB. Use format: r, g, b  (e.g. 34, 139, 230)";
    return;
  }
  const result = applyChain(rgb);
  setCssVars(result);
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  applyFromInput();
});

// initial paint
applyFromInput();

function rgbToHsv([r,g,b]) {
  const max = Math.max(r,g,b), min = Math.min(r,g,b);
  const d = max - min;
  let h = 0;
  const s = max === 0 ? 0 : d / max;
  const v = max;

  if (d !== 0) {
    switch (max) {
      case r: h = ((g - b) / d) % 6; break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h /= 6;
    if (h < 0) h += 1;
  }
  return [h, s, v];
}

function hsvToRgb([h,s,v]) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);

  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    case 5: return [v, p, q];
  }
}
