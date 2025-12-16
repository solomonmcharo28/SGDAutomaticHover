// ---------- Helpers ----------
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
function hoverFromBase(base01){
  const t = 0.18;

  // Create a "darker & slightly cooler" target derived from base
  const [r,g,b] = base01;
  const target = [
    clamp01(r * 0.82),
    clamp01(g * 0.86),
    clamp01(b * 0.92),
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

// ---------- Chain-rule pipeline ----------
function applyChain(baseRgb255){
  const base01 = rgb01(baseRgb255);

  // 1) base button bg = B (from form)
  // 2) base text chosen by SGD: T_base = argmin loss(B, T)
  const baseText = optimizeText(base01, { initParams: [0.5, 0.5, 0.5] });

  // 3) hover bg derived from base: H = H(B)
  const hover01 = hoverFromBase(base01);

  // 4) hover text chosen by SGD: T_hover = argmin loss(H(B), T)
  //    (This is the second stage that depends on base via H(B), i.e. chain rule path.)
  const hoverText = optimizeText(hover01, { initParams: baseText.params });

  // 5) also set page background based on base (optional)
  //    (a subtle darkened complementary feel)
  const pageBg01 = [
    clamp01(base01[0] * 0.10),
    clamp01(base01[1] * 0.10),
    clamp01(base01[2] * 0.10),
  ];

  return { base01, baseText, hover01, hoverText, pageBg01 };
}

// ---------- DOM wiring ----------
const form = document.getElementById("colorForm");
const rgbInput = document.getElementById("rgbInput");
const cta = document.getElementById("cta");
const readout = document.getElementById("readout");

function setCssVars({ base01, baseText, hover01, hoverText, pageBg01 }){
  const base255 = rgb255(base01.map(v => v*255));
  const baseText255 = rgb255(baseText.color01.map(v => v*255));
  const hover255 = rgb255(hover01.map(v => v*255));
  const hoverText255 = rgb255(hoverText.color01.map(v => v*255));
  const pageBg255 = rgb255(pageBg01.map(v => v*255));

  document.documentElement.style.setProperty("--btn-bg", rgbToCss(base255));
  document.documentElement.style.setProperty("--btn-fg", rgbToCss(baseText255));
  document.documentElement.style.setProperty("--btn-hover-bg", rgbToCss(hover255));
  document.documentElement.style.setProperty("--btn-hover-fg", rgbToCss(hoverText255));
  document.documentElement.style.setProperty("--page-bg", rgbToCss(pageBg255));

  readout.textContent =
`Base button background (B):     ${rgbToCss(base255)}
Base text via SGD (T_base):      ${rgbToCss(baseText255)}   loss=${baseText.loss.toFixed(4)}

Hover background H(B):           ${rgbToCss(hover255)}
Hover text via SGD (T_hover):    ${rgbToCss(hoverText255)}  loss=${hoverText.loss.toFixed(4)}

Chain: B → H(B) → optimal hover text`;
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
