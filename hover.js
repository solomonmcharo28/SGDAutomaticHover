// === helpers & SGD (unchanged from previous version) ===
// clamp01, wrap01, rgb↔hsv, luminance, computeHoverColor, etc.

/***********************
 *  Form integration
 ***********************/
function parseRgbInput(value){
  const parts = value.split(',').map(v => parseInt(v.trim(), 10));
  if (parts.length !== 3 || parts.some(v => isNaN(v) || v < 0 || v > 255)) {
    return null;
  }
  return parts.map(v => v / 255);
}

function cssRgb01(rgb){
  return `rgb(${Math.round(rgb[0]*255)} ${Math.round(rgb[1]*255)} ${Math.round(rgb[2]*255)})`;
}

function recompute(el){
  const style = getComputedStyle(el);
  const parse = c =>
    c.match(/rgb[a]?\(([^)]+)\)/)[1]
     .split(',')
     .slice(0,3)
     .map(v => parseFloat(v)/255);

  const fg = parse(style.color);
  const bg = parse(style.backgroundColor);

  const hover = computeHoverColor(fg, bg);
  el.style.setProperty('--hover-fg', cssRgb01(hover));
}

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('colorForm');
  const input = document.getElementById('rgbInput');
  const btn = document.getElementById('demoBtn');

  // initial hover computation
  recompute(btn);

  form.addEventListener('submit', e => {
    e.preventDefault(); // prevent page reload

    const rgb = parseRgbInput(input.value);
    if (!rgb) return;

    btn.style.setProperty('--fg', cssRgb01(rgb));
    recompute(btn);
  });
});
/***********************
 *  Color utilities
 ***********************/
const clamp01 = x => Math.min(1, Math.max(0, x));
const wrap01  = x => (x % 1 + 1) % 1;

function hexToRgb01(hex){
  const s = hex.replace('#','');
  const full = s.length === 3 ? s.split('').map(c=>c+c).join('') : s;
  const n = parseInt(full, 16);
  return [(n>>16&255)/255, (n>>8&255)/255, (n&255)/255];
}

function rgb01ToCss([r,g,b]){
  return `rgb(${Math.round(r*255)} ${Math.round(g*255)} ${Math.round(b*255)})`;
}

function rgbToHsv([r,g,b]){
  const max = Math.max(r,g,b), min = Math.min(r,g,b);
  const d = max - min;
  let h = 0;
  if (d){
    h = max===r ? ((g-b)/d)%6 : max===g ? (b-r)/d+2 : (r-g)/d+4;
    h /= 6;
  }
  return [wrap01(h), max===0?0:d/max, max];
}

function hsvToRgb([h,s,v]){
  const i = Math.floor(h*6);
  const f = h*6 - i;
  const p = v*(1-s);
  const q = v*(1-f*s);
  const t = v*(1-(1-f)*s);
  return [
    [v,t,p],[q,v,p],[p,v,t],
    [p,q,v],[t,p,v],[v,p,q]
  ][i%6];
}

function srgbToLinear(c){
  return c <= 0.04045 ? c/12.92 : ((c+0.055)/1.055)**2.4;
}

function luminance(rgb){
  const [r,g,b] = rgb.map(srgbToLinear);
  return 0.2126*r + 0.7152*g + 0.0722*b;
}

const softplus = x => Math.log1p(Math.exp(x));

/***********************
 *  SGD optimizer
 ***********************/
function computeHoverColor(rgb0, rgbBg){
  const hsv0 = rgbToHsv(rgb0);
  let d = [0,0,0];

  const Ybg = luminance(rgbBg);

  const loss = (dh,ds,dv)=>{
    const h = wrap01(hsv0[0]+dh);
    const s = clamp01(hsv0[1]+ds);
    const v = clamp01(hsv0[2]+dv);
    const rgb = hsvToRgb([h,s,v]);

    const Y = luminance(rgb);
    const legible = softplus(0.35 - Math.abs(Y - Ybg));

    const dhWrap = Math.min(Math.abs(h-hsv0[0]), 1-Math.abs(h-hsv0[0]));
    const stay =
      140*dhWrap**2 +
        7*(s-hsv0[1])**2 +
        2*(v-hsv0[2])**2;

    return legible + 0.35*stay;
  };

  let lr = 0.14;
  const eps = 1e-3;

  for(let i=0;i<90;i++){
    const grad = [0,0,0].map((_,k)=>{
      const dp = [...d]; dp[k]+=eps;
      const dm = [...d]; dm[k]-=eps;
      return (loss(...dp)-loss(...dm))/(2*eps);
    });

    d = d.map((v,i)=>v - lr*grad[i]);
    lr *= 0.985;
  }

  return hsvToRgb([
    wrap01(hsv0[0]+d[0]),
    clamp01(hsv0[1]+d[1]),
    clamp01(hsv0[2]+d[2])
  ]);
}

/***********************
 *  Init on DOM ready
 ***********************/
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.sgd-hover').forEach(el=>{
    const style = getComputedStyle(el);

    const fg = style.color;
    const bg = style.backgroundColor;

    const parse = c=>{
      if(c.startsWith('#')) return hexToRgb01(c);
      const m = c.match(/rgb[a]?\(([^)]+)\)/);
      return m[1].split(',').slice(0,3).map(v=>parseFloat(v)/255);
    };

    const hover = computeHoverColor(parse(fg), parse(bg));
    el.style.setProperty('--hover-fg', rgb01ToCss(hover));
  });
});
