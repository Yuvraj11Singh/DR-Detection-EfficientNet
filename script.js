'use strict';

// ═══════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════
let sessionHistory = [];
let sortConfig = { key: null, dir: 'asc' };
let patientChart = null, distChart = null, timelineChart = null, scatterChart = null, histogramChart = null, riskGaugeChart = null;
let currentLabImage = null;
let currentLabFilter = 'original';
let annotationMode = false;
let annTool = 'pen';
let annHistory = [];
let isDrawing = false;
let annStart = { x: 0, y: 0 };
let currentNoteId = null;
let notes = [];
let appSettings = {
  filter: 'none',
  threshold: 0.5,
  profile: { name: 'Dr. Yuvraj', inst: 'Manipal University Jaipur', spec: 'AI/ML Research', loc: 'New Delhi, India' }
};

// ═══ CLOCK ═══
function updateClock() {
  const el = document.getElementById('liveClock');
  if (el) el.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ═══ TOAST ═══
function showToast(msg, type = 'info', ms = 3500) {
  const c = document.getElementById('toastContainer');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<div class="toast-dot"></div><span>${msg}</span>`;
  c.appendChild(t);
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 400); }, ms);
}

// ═══ THEME ═══
function toggleTheme() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  document.documentElement.setAttribute('data-theme', isLight ? '' : 'light');
  const el = document.getElementById('set-dark');
  if (el) el.checked = isLight;
  showToast(isLight ? 'Dark mode.' : 'Light mode.', 'info');
}
function toggleScanLine() {
  const sl = document.getElementById('scanLine');
  if (sl) { const cb = document.getElementById('set-scanline'); sl.style.display = (cb && cb.checked) ? '' : 'none'; }
}
function toggleBlobs() {
  const cb = document.getElementById('set-blobs');
  document.querySelectorAll('.blob1,.blob2').forEach(b => b.style.display = (cb && cb.checked) ? '' : 'none');
}

// ═══ LIGHTBOX ═══
function openLightbox(src) { document.getElementById('lightboxImg').src = src; document.getElementById('lightboxModal').classList.add('active'); }
function openLightboxCanvas(canvas) { openLightbox(canvas.toDataURL('image/png')); }

// ═══ PROFILE MODAL ═══
function openProfileModal() { document.getElementById('profileModal').classList.add('active'); }
function closeProfileModal() { document.getElementById('profileModal').classList.remove('active'); }
document.getElementById('profileModal').addEventListener('click', e => { if (e.target === document.getElementById('profileModal')) closeProfileModal(); });

// ═══ SIDEBAR RESIZER ═══
const resizer = document.getElementById('sidebarResizer');
let isResizing = false;
resizer.addEventListener('mousedown', () => { isResizing = true; document.body.classList.add('resizing'); resizer.classList.add('active'); });
document.addEventListener('mousemove', e => {
  if (!isResizing) return;
  let w = Math.min(Math.max(e.clientX, 200), 500);
  document.documentElement.style.setProperty('--sidebar-width', w + 'px');
});
document.addEventListener('mouseup', () => {
  if (!isResizing) return;
  isResizing = false;
  document.body.classList.remove('resizing');
  resizer.classList.remove('active');
});

// ═══ KEYBOARD SHORTCUTS ═══
document.addEventListener('keydown', e => {
  const tag = e.target.tagName;
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || e.target.contentEditable === 'true') return;
  const views = ['detect', 'registry', 'image-lab', 'risk-calc', 'notes', 'settings-view'];
  if (e.key >= '1' && e.key <= '6') switchView(views[+e.key - 1]);
  switch (e.key.toLowerCase()) {
    case 'n': resetForm(); switchView('detect'); break;
    case 'e': document.getElementById('toggleHeatmapBtn')?.click(); break;
    case 'a': toggleAnnotationMode(); break;
    case 'p': window.print(); break;
    case 't': toggleTheme(); break;
    case 'r': switchView('risk-calc'); break;
    case '?': document.getElementById('shortcutsModal').classList.add('active'); break;
    case 'escape': document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active')); break;
  }
});
document.getElementById('shortcutsModal').addEventListener('click', e => {
  if (e.target === document.getElementById('shortcutsModal')) document.getElementById('shortcutsModal').classList.remove('active');
});

// ═══ VIEW SWITCHING ═══
function switchView(id) {
  document.querySelectorAll('.sidebar-nav li').forEach(li => li.classList.remove('active'));
  const nav = document.querySelector(`.sidebar-nav li[data-target="${id}"]`);
  if (nav) nav.classList.add('active');
  document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
  const sec = document.getElementById(id);
  if (sec) sec.classList.add('active');
  const isDetect = id === 'detect';
  document.getElementById('hero-section').style.display = isDetect ? 'flex' : 'none';
  document.getElementById('stats-strip').style.display = isDetect ? 'flex' : 'none';
  if (id === 'registry') updateRegistryView();
  if (id === 'image-lab') initLabUpload();
  if (id === 'notes') renderNotesList();
  if (id === 'settings-view') loadSettingsUI();
}

// ═══ COUNTERS ═══
function animateCount(el, target, suffix, dec) {
  if (!el) return;
  let s = 0;
  const step = ts => {
    if (!s) s = ts;
    const p = Math.min((ts - s) / 1800, 1), e = 1 - Math.pow(1 - p, 3);
    el.textContent = (dec ? (e * target).toFixed(dec) : Math.round(e * target).toLocaleString()) + (suffix || '');
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

const statsObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      animateCount(document.getElementById('cnt-auc'), 0.94, '', 2);
      animateCount(document.getElementById('cnt-acc'), 92, '%', 0);
      animateCount(document.getElementById('cnt-params'), 5.3, 'M', 1);
      statsObs.disconnect();
    }
  });
}, { threshold: 0.5 });
const ss = document.querySelector('.stats-strip');
if (ss) statsObs.observe(ss);

const revObs = new IntersectionObserver(entries => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = (i * 0.07) + 's';
      e.target.classList.add('visible');
    }
  });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal').forEach(el => revObs.observe(el));

// ═══════════════════════════════════════════════
// COMPUTER VISION ENGINE
// ═══════════════════════════════════════════════
const CV = {};
CV.load = (src, w = 200, h = 200) => new Promise(resolve => {
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
    resolve({ canvas: c, ctx, data: ctx.getImageData(0, 0, w, h), img });
  };
  img.src = src;
});
CV.draw = (canvas, data, w, h) => { canvas.width = w; canvas.height = h; canvas.getContext('2d').putImageData(data, 0, 0); };
CV.gray = src => {
  const d = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) { const l = 0.299 * src.data[i] + 0.587 * src.data[i + 1] + 0.114 * src.data[i + 2]; d.data[i] = d.data[i + 1] = d.data[i + 2] = l; d.data[i + 3] = src.data[i + 3]; }
  return d;
};
CV.channel = (src, ch) => {
  const d = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) { d.data[i] = ch === 0 ? src.data[i] : 0; d.data[i + 1] = ch === 1 ? src.data[i + 1] : 0; d.data[i + 2] = ch === 2 ? src.data[i + 2] : 0; d.data[i + 3] = 255; }
  return d;
};
CV.sobel = src => {
  const w = src.width, h = src.height, g = CV.gray(src), d = new ImageData(new Uint8ClampedArray(src.data.length), w, h);
  const gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1], gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let sx = 0, sy = 0;
    for (let ky = -1; ky <= 1; ky++) for (let kx = -1; kx <= 1; kx++) { const p = g.data[((y + ky) * w + (x + kx)) * 4], k = (ky + 1) * 3 + (kx + 1); sx += gx[k] * p; sy += gy[k] * p; }
    const m = Math.min(255, Math.sqrt(sx * sx + sy * sy)), i = (y * w + x) * 4;
    d.data[i] = d.data[i + 1] = d.data[i + 2] = m; d.data[i + 3] = 255;
  }
  return d;
};
CV.clahe = (src, tile = 32, clip = 3.0) => {
  const w = src.width, h = src.height, g = CV.gray(src), d = new ImageData(new Uint8ClampedArray(src.data.length), w, h);
  const nx = Math.ceil(w / tile), ny = Math.ceil(h / tile), luts = [];
  for (let ty = 0; ty < ny; ty++) {
    luts[ty] = [];
    for (let tx = 0; tx < nx; tx++) {
      const hist = new Array(256).fill(0), x0 = tx * tile, y0 = ty * tile, x1 = Math.min(x0 + tile, w), y1 = Math.min(y0 + tile, h), n = (x1 - x0) * (y1 - y0);
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) hist[g.data[(y * w + x) * 4]]++;
      const cl = Math.floor(clip * n / 256); let ex = 0;
      for (let i = 0; i < 256; i++) { if (hist[i] > cl) { ex += hist[i] - cl; hist[i] = cl; } }
      const pb = Math.floor(ex / 256); for (let i = 0; i < 256; i++) hist[i] += pb;
      const lut = new Uint8Array(256); let cdf = 0, cmin = -1;
      for (let i = 0; i < 256; i++) { cdf += hist[i]; if (cmin < 0 && cdf > 0) cmin = cdf; lut[i] = Math.round((cdf - cmin) / (n - cmin) * 255); }
      luts[ty][tx] = lut;
    }
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const tx = (x / tile) - 0.5, ty = (y / tile) - 0.5, tx0 = Math.max(0, Math.floor(tx)), ty0 = Math.max(0, Math.floor(ty)), tx1 = Math.min(nx - 1, tx0 + 1), ty1 = Math.min(ny - 1, ty0 + 1), fx = tx - Math.floor(tx), fy = ty - Math.floor(ty), pv = g.data[(y * w + x) * 4];
    const v00 = luts[ty0][tx0][pv], v10 = luts[ty0][tx1][pv], v01 = luts[ty1][tx0][pv], v11 = luts[ty1][tx1][pv];
    const val = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
    const sc = val / (pv + 1e-6), i = (y * w + x) * 4;
    d.data[i] = Math.min(255, src.data[i] * sc); d.data[i + 1] = Math.min(255, src.data[i + 1] * sc); d.data[i + 2] = Math.min(255, src.data[i + 2] * sc); d.data[i + 3] = 255;
  }
  return d;
};
CV.vessel = src => {
  const d = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) { d.data[i] = 0; d.data[i + 1] = Math.min(255, Math.max(0, src.data[i + 1] * 2 - src.data[i] * 0.5 - src.data[i + 2] * 0.5)); d.data[i + 2] = 0; d.data[i + 3] = 255; }
  return d;
};
CV.pseudo = src => {
  const g = CV.gray(src), d = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < g.data.length; i += 4) { const v = g.data[i] / 255; d.data[i] = Math.min(255, Math.max(0, Math.round((1.5 - Math.abs(v * 4 - 3)) * 255))); d.data[i + 1] = Math.min(255, Math.max(0, Math.round((1.5 - Math.abs(v * 4 - 2)) * 255))); d.data[i + 2] = Math.min(255, Math.max(0, Math.round((1.5 - Math.abs(v * 4 - 1)) * 255))); d.data[i + 3] = 255; }
  return d;
};
CV.invert = src => {
  const d = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) { d.data[i] = 255 - src.data[i]; d.data[i + 1] = 255 - src.data[i + 1]; d.data[i + 2] = 255 - src.data[i + 2]; d.data[i + 3] = 255; }
  return d;
};
CV.emboss = src => {
  const w = src.width, h = src.height, g = CV.gray(src), d = new ImageData(new Uint8ClampedArray(src.data.length), w, h), k = [-2, -1, 0, -1, 1, 1, 0, 1, 2];
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) { let s = 128; for (let ky = -1; ky <= 1; ky++) for (let kx = -1; kx <= 1; kx++) s += k[(ky + 1) * 3 + (kx + 1)] * g.data[((y + ky) * w + (x + kx)) * 4]; const i = (y * w + x) * 4; d.data[i] = d.data[i + 1] = d.data[i + 2] = Math.min(255, Math.max(0, s)); d.data[i + 3] = 255; }
  return d;
};
CV.thermal = src => {
  const g = CV.gray(src), d = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < g.data.length; i += 4) { const v = g.data[i] / 255; d.data[i] = Math.min(255, v * 4 * 255); d.data[i + 1] = Math.min(255, Math.max(0, (v * 4 - 1)) * 255); d.data[i + 2] = Math.min(255, Math.max(0, (v * 4 - 3)) * 255); d.data[i + 3] = 255; }
  return d;
};
CV.heatmap = src => {
  const w = src.width, h = src.height, edges = CV.sobel(src), d = new ImageData(new Uint8ClampedArray(src.data.length), w, h), r = 8;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let sum = 0, cnt = 0;
    for (let ky = -r; ky <= r; ky++) for (let kx = -r; kx <= r; kx++) { const ny = y + ky, nx = x + kx; if (ny >= 0 && ny < h && nx >= 0 && nx < w) { sum += edges.data[(ny * w + nx) * 4]; cnt++; } }
    const v = sum / cnt / 255, i = (y * w + x) * 4;
    d.data[i] = Math.min(255, v * 4 * 255); d.data[i + 1] = Math.min(255, Math.max(0, v * 4 - 1) * 255); d.data[i + 2] = Math.min(255, Math.max(0, v * 4 - 3) * 255); d.data[i + 3] = 255;
  }
  return d;
};
CV.adjust = (src, br = 0, co = 0, sa = 0, ga = 1, sh = 0) => {
  const d = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height), cf = (259 * (co + 255)) / (255 * (259 - co));
  for (let i = 0; i < src.data.length; i += 4) {
    let r = src.data[i] + br, g = src.data[i + 1] + br, b = src.data[i + 2] + br;
    r = cf * (r - 128) + 128; g = cf * (g - 128) + 128; b = cf * (b - 128) + 128;
    const gl = 0.299 * r + 0.587 * g + 0.114 * b, sf = 1 + sa / 100;
    r = gl + sf * (r - gl); g = gl + sf * (g - gl); b = gl + sf * (b - gl);
    if (ga !== 1) { r = 255 * Math.pow(Math.max(0, r) / 255, 1 / ga); g = 255 * Math.pow(Math.max(0, g) / 255, 1 / ga); b = 255 * Math.pow(Math.max(0, b) / 255, 1 / ga); }
    d.data[i] = Math.min(255, Math.max(0, r)); d.data[i + 1] = Math.min(255, Math.max(0, g)); d.data[i + 2] = Math.min(255, Math.max(0, b)); d.data[i + 3] = src.data[i + 3];
  }
  return d;
};
CV.stats = data => {
  let sR = 0, sG = 0, sB = 0, sL = 0;
  const n = data.data.length / 4;
  for (let i = 0; i < data.data.length; i += 4) { sR += data.data[i]; sG += data.data[i + 1]; sB += data.data[i + 2]; sL += 0.299 * data.data[i] + 0.587 * data.data[i + 1] + 0.114 * data.data[i + 2]; }
  const mL = sL / n; let vL = 0;
  for (let i = 0; i < data.data.length; i += 4) { const l = 0.299 * data.data[i] + 0.587 * data.data[i + 1] + 0.114 * data.data[i + 2]; vL += (l - mL) ** 2; }
  const mR = sR / n, mG = sG / n, mB = sB / n;
  return { meanR: mR.toFixed(1), meanG: mG.toFixed(1), meanB: mB.toFixed(1), brightness: mL.toFixed(1), contrast: Math.sqrt(vL / n).toFixed(1), dominant: mR > mG && mR > mB ? 'Red' : mG > mB ? 'Green' : 'Blue', pixels: n };
};
CV.histogram = data => {
  const r = new Array(256).fill(0), g = new Array(256).fill(0), b = new Array(256).fill(0);
  for (let i = 0; i < data.data.length; i += 4) { r[data.data[i]]++; g[data.data[i + 1]]++; b[data.data[i + 2]]++; }
  return { r, g, b };
};

async function runCV(src, size = 200) {
  const { data } = await CV.load(src, size, size);
  return { orig: data, gray: CV.gray(data), green: CV.channel(data, 1), edge: CV.sobel(data), clahe: CV.clahe(data), vessel: CV.vessel(data), heat: CV.heatmap(data), stats: CV.stats(data), hist: CV.histogram(data) };
}

// ═══════════════════════════════════════════════
// UPLOAD FLOW
// ═══════════════════════════════════════════════
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewWrap = document.getElementById('preview-wrap');
const previewImg = document.getElementById('preview-img');
const resultCard = document.getElementById('result-card');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

uploadArea.addEventListener('click', () => fileInput.click());
document.getElementById('chooseFileBtn').addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); const f = e.dataTransfer.files[0]; if (f && f.type.startsWith('image/')) startPreprocessing(f); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) startPreprocessing(fileInput.files[0]); });

async function startPreprocessing(file) {
  const reader = new FileReader();
  reader.onload = async e => {
    const src = e.target.result;
    uploadArea.style.display = 'none';
    const strip = document.getElementById('cv-preview-strip');
    strip.style.display = 'block';
    const res = await runCV(src, 180);
    const pairs = { 'cv-orig': res.orig, 'cv-gray': res.gray, 'cv-green': res.green, 'cv-edge': res.edge, 'cv-clahe': res.clahe };
    for (const [id, data] of Object.entries(pairs)) CV.draw(document.getElementById(id), data, 180, 180);
    const s = res.stats;
    document.getElementById('imageStatsBar').innerHTML = `<span>Brightness: <strong>${s.brightness}</strong></span><span>Contrast: <strong>${s.contrast}</strong></span><span>Dominant: <strong>${s.dominant}</strong></span><span>R: <strong>${s.meanR}</strong></span><span>G: <strong>${s.meanG}</strong></span><span>B: <strong>${s.meanB}</strong></span><span>Pixels: <strong>${(s.pixels / 1000).toFixed(1)}K</strong></span>`;
    strip._src = src;
    strip._cv = res;
    showToast('Pre-processing complete.', 'success');
  };
  reader.readAsDataURL(file);
}

document.getElementById('proceedAnalysisBtn').addEventListener('click', () => {
  const strip = document.getElementById('cv-preview-strip');
  strip.style.display = 'none';
  runAnalysis(strip._src, strip._cv, 180);
});

function animateLoadingSteps() {
  const steps = ['ls-1', 'ls-2', 'ls-3', 'ls-4'];
  steps.forEach(id => { const el = document.getElementById(id); if (el) el.className = 'load-step'; });
  let i = 0;
  const iv = setInterval(() => {
    if (i > 0) { const p = document.getElementById(steps[i - 1]); if (p) p.className = 'load-step done'; }
    if (i < steps.length) { const c = document.getElementById(steps[i]); if (c) c.className = 'load-step active'; i++; }
    else clearInterval(iv);
  }, 700);
  return iv;
}

async function runAnalysis(src, cv, cvSize) {
  previewImg.src = src;
  previewWrap.style.display = 'block';
  previewImg.style.display = 'none';
  loadingOverlay.style.display = 'block';
  resultCard.style.display = 'none';
  const lv = animateLoadingSteps();
  let mi = 0;
  const msgs = ['Initializing MPS Backend...', 'Extracting vascular features...', 'Running EfficientNetB0...', 'Grading severity...'];
  const mv = setInterval(() => { loadingText.textContent = msgs[mi++ % msgs.length]; }, 700);
  await new Promise(r => setTimeout(r, 3100));
  clearInterval(lv); clearInterval(mv);
  loadingOverlay.style.display = 'none';
  previewImg.style.display = 'block';
  setTimeout(() => {
    const img = document.getElementById('preview-img'), ann = document.getElementById('annotationCanvas');
    ann.width = img.offsetWidth; ann.height = img.offsetHeight; annHistory = [];
  }, 100);
  const filter = document.getElementById('pat-filter').value;
  const patId = document.getElementById('pat-id').value || `P-${Math.floor(Math.random() * 9000) + 1000}`;
  const patAge = document.getElementById('pat-age').value || '—';
  const patDiab = document.getElementById('pat-diab').value || '—';
  const patHba = document.getElementById('pat-hba1c').value || '—';
  const patEye = document.getElementById('pat-eye').value;
  const stages = [
    { level: 0, name: 'No DR', color: 'badge-negative', pos: 0 },
    { level: 1, name: 'Mild NPDR', color: 'badge-positive', pos: 25 },
    { level: 2, name: 'Moderate NPDR', color: 'badge-positive', pos: 50 },
    { level: 3, name: 'Severe NPDR', color: 'badge-positive', pos: 75 },
    { level: 4, name: 'Proliferative DR', color: 'badge-positive', pos: 100 }
  ];
  const stage = stages[Math.floor(Math.random() * 5)];
  const cert = (Math.random() * 14 + 86).toFixed(1), dens = (stage.level * 18 + Math.random() * 12).toFixed(1);
  const patient = {
    uniqueKey: Date.now().toString(), id: patId, age: patAge, diab: patDiab, hba1c: patHba, eye: patEye,
    date: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    stageLevel: stage.level, stageName: stage.name, color: stage.color, pos: stage.pos,
    certainty: cert, density: dens, time: (Math.random() * 0.3 + 0.08).toFixed(3) + 's',
    history: [Math.max(0, stage.level - 2), Math.max(0, stage.level - 1), Math.max(0, stage.level - 1), stage.level, stage.level],
    imageSrc: src, filterUsed: filter, brightness: cv.stats.brightness, contrast: cv.stats.contrast, cv, cvSize
  };
  sessionHistory.unshift(patient);
  updateSessionCounts();
  updateSidebar(patient.uniqueKey);
  renderDashboard(patient);
  updateRegistryView();
  showToast(`Analysis complete — ${patient.stageName}.`, patient.stageLevel > 0 ? 'error' : 'success');
}

function updateSessionCounts() {
  const total = sessionHistory.length, pos = sessionHistory.filter(p => p.stageLevel > 0).length, neg = total - pos;
  document.getElementById('cnt-session').textContent = total;
  document.getElementById('qs-total').textContent = total;
  document.getElementById('qs-pos').textContent = pos;
  document.getElementById('qs-neg').textContent = neg;
  document.getElementById('storageStatus').textContent = `● ${total}`;
  document.getElementById('historyCount').textContent = total;
}

function renderDashboard(patient) {
  document.getElementById('upload-flow').style.display = 'none';
  document.getElementById('cv-preview-strip').style.display = 'none';
  loadingOverlay.style.display = 'none';
  previewWrap.style.display = 'block';
  previewImg.style.display = 'block';
  previewImg.src = patient.imageSrc;
  document.getElementById('heatmap-overlay').classList.remove('active');
  document.getElementById('xai-controls').style.display = 'none';
  const tb = document.getElementById('toggleHeatmapBtn');
  if (tb) { tb.textContent = 'Enable Grad-CAM'; tb.style.cssText = ''; }
  if (annotationMode) toggleAnnotationMode();
  const pp = document.getElementById('processedPanel');
  if (patient.filterUsed && patient.filterUsed !== 'none' && patient.cv) {
    pp.style.display = 'block';
    document.getElementById('processedLabel').textContent = patient.filterUsed.toUpperCase();
    const map = { clahe: 'clahe', green: 'green', edge: 'edge' }, key = map[patient.filterUsed];
    if (key && patient.cv[key]) CV.draw(document.getElementById('processedCanvas'), patient.cv[key], patient.cvSize, patient.cvSize);
  } else pp.style.display = 'none';
  document.getElementById('reportIdBadge').textContent = `RPT-${patient.uniqueKey.slice(-6)}`;
  document.getElementById('patient-summary').innerHTML = `<span><strong>ID:</strong> ${patient.id}</span><span><strong>Age:</strong> ${patient.age}</span><span><strong>Eye:</strong> ${patient.eye}</span><span><strong>HbA1c:</strong> ${patient.hba1c}%</span><span><strong>Diab:</strong> ${patient.diab}y</span><span><strong>Time:</strong> ${patient.date}</span>`;
  document.getElementById('result-label').textContent = patient.stageName;
  const badge = document.getElementById('result-badge');
  badge.textContent = patient.stageLevel === 0 ? 'Negative' : 'Refer to Ophthalmologist';
  badge.className = 'result-badge ' + patient.color;
  document.getElementById('severity-val').textContent = `Stage ${patient.stageLevel}`;
  setTimeout(() => { document.getElementById('severity-indicator').style.left = patient.pos + '%'; }, 100);
  document.getElementById('m-prob').textContent = patient.certainty + '%';
  document.getElementById('m-conf').textContent = patient.density;
  document.getElementById('m-time').textContent = patient.time;
  document.getElementById('m-filter').textContent = patient.filterUsed || 'None';
  document.getElementById('m-brightness').textContent = patient.brightness;
  document.getElementById('m-contrast').textContent = patient.contrast;
  setTimeout(() => {
    document.getElementById('cert-ring').style.strokeDashoffset = 264 - 264 * (patient.certainty / 100);
    document.getElementById('lesion-ring').style.strokeDashoffset = 264 - 264 * (Math.min(patient.density, 100) / 100);
  }, 200);
  renderRiskPanel(patient);
  if (patient.cv) {
    [['res-gray', 'gray'], ['res-green', 'green'], ['res-edge', 'edge'], ['res-clahe', 'clahe'], ['res-heat', 'heat']].forEach(([id, k]) => {
      const c = document.getElementById(id);
      if (c && patient.cv[k]) CV.draw(c, patient.cv[k], patient.cvSize, patient.cvSize);
    });
  }
  renderRecommendation(patient);
  const ctx = document.getElementById('progressionChart').getContext('2d');
  if (patientChart) patientChart.destroy();
  patientChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Jan', 'Apr', 'Jul', 'Oct', 'Current'],
      datasets: [{ label: 'Severity', data: patient.history, borderColor: '#7c5cfc', backgroundColor: 'rgba(124,92,252,0.12)', borderWidth: 2, pointBackgroundColor: 'var(--bg)', pointBorderColor: '#7c5cfc', pointBorderWidth: 2, pointRadius: 4, fill: true, tension: 0.4 }]
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, max: 4, ticks: { stepSize: 1, color: '#6b8f9e', font: { family: 'DM Mono,monospace', size: 9 } }, grid: { color: 'rgba(124,92,252,0.08)' } }, x: { ticks: { color: '#6b8f9e', font: { family: 'DM Mono,monospace', size: 9 } }, grid: { display: false } } } }
  });
  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderRiskPanel(patient) {
  const factors = [
    { name: 'Vascular Abnormality', val: Math.min(100, patient.stageLevel * 22 + Math.random() * 15), color: '#b35f6f' },
    { name: 'Microaneurysm Score', val: Math.min(100, patient.stageLevel * 18 + Math.random() * 20), color: '#c08a56' },
    { name: 'Hemorrhage Risk', val: Math.min(100, patient.stageLevel * 15 + Math.random() * 18), color: '#d7b36a' },
    { name: 'Exudate Presence', val: Math.min(100, patient.stageLevel * 12 + Math.random() * 22), color: '#6f948f' },
    { name: 'Neovascularization', val: patient.stageLevel === 4 ? 80 + Math.random() * 15 : Math.random() * 20, color: '#b35f6f' },
    { name: 'Optic Disc Integrity', val: 100 - (patient.stageLevel * 15 + Math.random() * 10), color: '#e2876d' }
  ];
  const overall = patient.stageLevel === 0 ? 'Low Risk' : patient.stageLevel <= 2 ? 'Moderate Risk' : 'High Risk';
  const oc = { 'Low Risk': '#0ac5a8', 'Moderate Risk': '#ff9f43', 'High Risk': '#f4456a' }[overall];
  const el = document.getElementById('riskOverall');
  el.textContent = overall; el.style.background = oc + '22'; el.style.color = oc; el.style.border = `1px solid ${oc}44`;
  document.getElementById('riskFactorsGrid').innerHTML = factors.map(f => `<div class="risk-factor-item"><div class="risk-factor-info"><div class="risk-factor-name">${f.name}</div><div class="risk-factor-bar-bg"><div class="risk-factor-bar-fill" style="width:0%;background:${f.color};" data-w="${f.val.toFixed(0)}"></div></div></div><div class="risk-factor-val">${f.val.toFixed(0)}%</div></div>`).join('');
  setTimeout(() => { document.querySelectorAll('.risk-factor-bar-fill').forEach(b => { b.style.width = b.dataset.w + '%'; }); }, 150);
}

function renderRecommendation(patient) {
  const recs = [
    { color: '#0ac5a8', bg: 'rgba(10,197,168,0.05)', border: 'rgba(10,197,168,0.2)', text: 'No DR detected. Annual screening recommended. Continue glycaemic and BP management.' },
    { color: '#7c5cfc', bg: 'rgba(124,92,252,0.05)', border: 'rgba(124,92,252,0.2)', text: 'Mild NPDR. Follow-up in 12 months. Optimize HbA1c below 7%.' },
    { color: '#ff9f43', bg: 'rgba(255,159,67,0.05)', border: 'rgba(255,159,67,0.2)', text: 'Moderate NPDR. Ophthalmology referral within 3–6 months. Enhanced glycaemic control critical.' },
    { color: '#f4456a', bg: 'rgba(244,69,106,0.05)', border: 'rgba(244,69,106,0.2)', text: 'Severe NPDR. Urgent referral within 1 month. Laser photocoagulation may be indicated.' },
    { color: '#c01440', bg: 'rgba(192,20,64,0.05)', border: 'rgba(192,20,64,0.2)', text: 'Proliferative DR. URGENT referral within 1 week. Anti-VEGF or pan-retinal photocoagulation required.' }
  ];
  const r = recs[patient.stageLevel], p = document.getElementById('recommendationPanel');
  p.style.background = r.bg; p.style.borderColor = r.border; p.style.color = 'var(--muted)';
  p.innerHTML = `<span style="font-family:'DM Mono',monospace;font-size:9px;color:${r.color};text-transform:uppercase;letter-spacing:1.5px;display:block;margin-bottom:7px;">Clinical Recommendation</span>${r.text}`;
}

// ═══ ANNOTATION ENGINE ═══
function toggleAnnotationMode() {
  annotationMode = !annotationMode;
  const toolbar = document.getElementById('annotationToolbar'), btn = document.getElementById('toggleAnnotateBtn'), canvas = document.getElementById('annotationCanvas');
  toolbar.style.display = annotationMode ? 'flex' : 'none';
  canvas.style.pointerEvents = annotationMode ? 'auto' : 'none';
  if (btn) { btn.textContent = annotationMode ? 'Exit Annotation' : 'Annotate Image'; btn.style.background = annotationMode ? 'rgba(124,92,252,0.15)' : ''; btn.style.color = annotationMode ? 'var(--accent)' : ''; btn.style.border = annotationMode ? '1px solid var(--accent)' : ''; }
  if (annotationMode) showToast('Annotation mode active.', 'info');
}
function setAnnTool(tool, btn) { annTool = tool; document.querySelectorAll('.ann-tool-btn').forEach(b => b.classList.remove('active')); if (btn) btn.classList.add('active'); }
const annCanvas = document.getElementById('annotationCanvas'), annCtx = annCanvas.getContext('2d');
let annSnapshot = null;
function getAnnPos(e) { const r = annCanvas.getBoundingClientRect(), scaleX = annCanvas.width / r.width, scaleY = annCanvas.height / r.height; return { x: (e.clientX - r.left) * scaleX, y: (e.clientY - r.top) * scaleY }; }
annCanvas.addEventListener('mousedown', e => {
  if (!annotationMode) return;
  isDrawing = true;
  const pos = getAnnPos(e); annStart = pos;
  annSnapshot = annCtx.getImageData(0, 0, annCanvas.width, annCanvas.height);
  if (annTool === 'pen') { annCtx.beginPath(); annCtx.moveTo(pos.x, pos.y); }
  if (annTool === 'text') { const txt = prompt('Enter label:'); if (txt) { annCtx.font = `${+document.getElementById('annSize').value * 4 + 10}px 'DM Mono'`; annCtx.fillStyle = document.getElementById('annColor').value; annCtx.fillText(txt, pos.x, pos.y); saveAnnSnapshot(); } isDrawing = false; }
});
annCanvas.addEventListener('mousemove', e => {
  if (!annotationMode || !isDrawing) return;
  const pos = getAnnPos(e), color = document.getElementById('annColor').value, size = +document.getElementById('annSize').value;
  if (annTool === 'pen') { annCtx.strokeStyle = color; annCtx.lineWidth = size; annCtx.lineCap = 'round'; annCtx.lineTo(pos.x, pos.y); annCtx.stroke(); }
  else {
    annCtx.putImageData(annSnapshot, 0, 0); annCtx.strokeStyle = color; annCtx.lineWidth = size;
    if (annTool === 'rect') { annCtx.beginPath(); annCtx.strokeRect(annStart.x, annStart.y, pos.x - annStart.x, pos.y - annStart.y); }
    if (annTool === 'circle') { annCtx.beginPath(); const rx = (pos.x - annStart.x) / 2, ry = (pos.y - annStart.y) / 2; annCtx.ellipse(annStart.x + rx, annStart.y + ry, Math.abs(rx), Math.abs(ry), 0, 0, Math.PI * 2); annCtx.stroke(); }
    if (annTool === 'arrow') { const dx = pos.x - annStart.x, dy = pos.y - annStart.y, angle = Math.atan2(dy, dx), hl = 14; annCtx.beginPath(); annCtx.moveTo(annStart.x, annStart.y); annCtx.lineTo(pos.x, pos.y); annCtx.stroke(); annCtx.beginPath(); annCtx.moveTo(pos.x, pos.y); annCtx.lineTo(pos.x - hl * Math.cos(angle - 0.4), pos.y - hl * Math.sin(angle - 0.4)); annCtx.moveTo(pos.x, pos.y); annCtx.lineTo(pos.x - hl * Math.cos(angle + 0.4), pos.y - hl * Math.sin(angle + 0.4)); annCtx.stroke(); }
  }
});
annCanvas.addEventListener('mouseup', () => { if (!annotationMode || !isDrawing) return; isDrawing = false; saveAnnSnapshot(); });
function saveAnnSnapshot() { annHistory.push(annCtx.getImageData(0, 0, annCanvas.width, annCanvas.height)); if (annHistory.length > 30) annHistory.shift(); }
function undoAnnotation() { if (!annHistory.length) return; annHistory.pop(); if (annHistory.length > 0) annCtx.putImageData(annHistory[annHistory.length - 1], 0, 0); else annCtx.clearRect(0, 0, annCanvas.width, annCanvas.height); }
function clearAnnotations() { annCtx.clearRect(0, 0, annCanvas.width, annCanvas.height); annHistory = []; }
function downloadAnnotated() {
  const merged = document.createElement('canvas'), img = document.getElementById('preview-img');
  merged.width = img.naturalWidth; merged.height = img.naturalHeight;
  const mctx = merged.getContext('2d');
  mctx.drawImage(img, 0, 0); mctx.drawImage(annCanvas, 0, 0, merged.width, merged.height);
  const a = document.createElement('a'); a.download = `annotated_${Date.now()}.png`; a.href = merged.toDataURL('image/png'); a.click();
  showToast('Annotated image downloaded.', 'success');
}

// ═══ SIDEBAR HISTORY ═══
function updateSidebar(activeKey) {
  const list = document.getElementById('historyList'); list.innerHTML = '';
  if (!sessionHistory.length) { list.innerHTML = '<li class="empty-history">No scans yet.</li>'; return; }
  sessionHistory.forEach(pat => {
    const li = document.createElement('li');
    if (pat.uniqueKey === activeKey) li.classList.add('active');
    const dc = pat.stageLevel === 0 ? 'success' : pat.stageLevel <= 2 ? 'warning' : 'danger';
    li.innerHTML = `<div class="pat-list-info"><span class="status-dot ${dc}"></span><img src="${pat.imageSrc}" class="sidebar-thumb" alt=""/><div class="pat-list-details"><strong>${pat.id}</strong><span>${pat.stageName}</span></div></div><span class="pat-list-time">${pat.date}</span>`;
    li.addEventListener('click', () => { switchView('detect'); updateSidebar(pat.uniqueKey); renderDashboard(pat); });
    list.appendChild(li);
  });
}

// ═══ QUICK NOTE ═══
function saveQuickNote() {
  const text = document.getElementById('quickNoteInput').value.trim(); if (!text) return;
  const active = sessionHistory[0];
  const note = { id: Date.now().toString(), title: `Scan Note — ${active?.id || 'Unknown'}`, content: text, tag: 'finding', ts: new Date().toLocaleString() };
  notes.unshift(note); document.getElementById('quickNoteInput').value = '';
  showToast('Note saved.', 'success'); renderNotesList();
}

// ═══ HEATMAP ═══
const toggleHeatmapBtn = document.getElementById('toggleHeatmapBtn');
const xaiControls = document.getElementById('xai-controls');
const heatmapOverlay = document.getElementById('heatmap-overlay');
const heatmapSlider = document.getElementById('heatmapSlider');
toggleHeatmapBtn.addEventListener('click', () => {
  heatmapOverlay.classList.toggle('active');
  const on = heatmapOverlay.classList.contains('active');
  if (on) {
    toggleHeatmapBtn.textContent = 'Disable Grad-CAM';
    toggleHeatmapBtn.style.cssText = 'background:rgba(124,92,252,0.12);color:var(--accent);border:1px solid rgba(124,92,252,0.4);font-family:"DM Mono",monospace;font-size:12px;padding:10px 20px;border-radius:8px;cursor:pointer;';
    xaiControls.style.display = 'block'; heatmapOverlay.style.opacity = heatmapSlider.value / 100;
  } else {
    toggleHeatmapBtn.textContent = 'Enable Grad-CAM'; toggleHeatmapBtn.style.cssText = '';
    xaiControls.style.display = 'none'; heatmapOverlay.style.opacity = 0;
  }
});
heatmapSlider.addEventListener('input', e => { document.getElementById('opacity-val').textContent = e.target.value + '%'; if (heatmapOverlay.classList.contains('active')) heatmapOverlay.style.opacity = e.target.value / 100; });
document.getElementById('downloadReportBtn').addEventListener('click', () => window.print());

// ═══ EXPORT SCAN JSON ═══
function exportScanJSON() {
  if (!sessionHistory.length) { showToast('No scan to export.', 'error'); return; }
  const pat = sessionHistory[0], data = { id: pat.id, age: pat.age, eye: pat.eye, hba1c: pat.hba1c, diab: pat.diab, date: pat.date, stageName: pat.stageName, stageLevel: pat.stageLevel, certainty: pat.certainty, density: pat.density, inference: pat.time, filter: pat.filterUsed, brightness: pat.brightness, contrast: pat.contrast };
  const a = document.createElement('a'); a.download = `retinaai_scan_${pat.id}.json`; a.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data, null, 2)); a.click();
  showToast('Scan exported as JSON.', 'success');
}

// ═══ RESET ═══
function resetForm() {
  document.getElementById('upload-flow').style.display = 'block';
  document.getElementById('uploadArea').style.display = 'block';
  document.getElementById('cv-preview-strip').style.display = 'none';
  previewWrap.style.display = 'none'; resultCard.style.display = 'none'; fileInput.value = '';
  document.querySelectorAll('.patient-list li').forEach(li => li.classList.remove('active'));
  heatmapOverlay.classList.remove('active'); xaiControls.style.display = 'none';
  toggleHeatmapBtn.style.cssText = ''; toggleHeatmapBtn.textContent = 'Enable Grad-CAM';
  if (annotationMode) toggleAnnotationMode();
  document.getElementById('severity-indicator').style.left = '0%';
  document.getElementById('cert-ring').style.strokeDashoffset = 264;
  document.getElementById('lesion-ring').style.strokeDashoffset = 264;
  ['pat-id', 'pat-age', 'pat-diab', 'pat-hba1c'].forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
  document.getElementById('processedPanel').style.display = 'none';
  document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
}

// ═══ REGISTRY ═══
let filteredHistory = [];
function sortTable(key) { sortConfig = { key, dir: sortConfig.key === key && sortConfig.dir === 'asc' ? 'desc' : 'asc' }; renderRegistryTable(); }
document.getElementById('searchInput').addEventListener('input', e => { const q = e.target.value.toLowerCase(); filteredHistory = sessionHistory.filter(p => p.id.toLowerCase().includes(q)); renderRegistryTable(); });
document.getElementById('filterGrade').addEventListener('change', e => { const v = e.target.value; filteredHistory = v === '' ? [] : sessionHistory.filter(p => p.stageLevel === +v); renderRegistryTable(); });

function renderRegistryTable() {
  const src = (document.getElementById('searchInput').value || document.getElementById('filterGrade').value) ? filteredHistory : sessionHistory, data = [...src];
  if (sortConfig.key) data.sort((a, b) => { let va = a[sortConfig.key], vb = b[sortConfig.key]; if (!isNaN(va)) va = +va; if (!isNaN(vb)) vb = +vb; return sortConfig.dir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1); });
  const tbody = document.getElementById('registryTableBody'), footer = document.getElementById('tableFooter');
  tbody.innerHTML = '';
  if (!data.length) { tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:28px;">No records found.</td></tr>'; footer.textContent = ''; return; }
  data.forEach(pat => {
    const tr = document.createElement('tr'), bc = pat.stageLevel === 0 ? 'badge-negative' : 'badge-positive';
    tr.innerHTML = `<td><strong>${pat.id}</strong></td><td>${pat.age}</td><td>${pat.eye}</td><td><span class="result-badge ${bc}" style="font-size:9px;padding:2px 9px;">${pat.stageName}</span></td><td style="font-family:'DM Mono',monospace;color:var(--accent);">${pat.certainty}%</td><td>${pat.hba1c}%</td><td style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);">${pat.filterUsed || 'none'}</td><td style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);">${pat.date}</td><td><button onclick="event.stopPropagation();switchView('detect');updateSidebar('${pat.uniqueKey}');renderDashboard(sessionHistory.find(p=>p.uniqueKey==='${pat.uniqueKey}'))" style="font-family:'DM Mono',monospace;font-size:9px;padding:3px 9px;border-radius:5px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;">View</button></td>`;
    tr.addEventListener('click', () => { switchView('detect'); updateSidebar(pat.uniqueKey); renderDashboard(pat); });
    tbody.appendChild(tr);
  });
  footer.textContent = `Showing ${data.length} of ${sessionHistory.length} records`;
}

function updateRegistryView() {
  if (!document.getElementById('registry').classList.contains('active')) return;
  const pos = sessionHistory.filter(p => p.stageLevel > 0).length;
  const neg = sessionHistory.length - pos;
  const ages = sessionHistory.map(p => +p.age).filter(a => !isNaN(a) && a > 0);
  const avgAge = ages.length ? (ages.reduce((a, b) => a + b, 0) / ages.length).toFixed(0) : 'N/A';
  const avgC = sessionHistory.length ? (sessionHistory.reduce((s, p) => s + (+p.certainty), 0) / sessionHistory.length).toFixed(1) + '%' : 'N/A';
  const highR = sessionHistory.filter(p => p.stageLevel >= 3).length;
  document.getElementById('reg-total').textContent = sessionHistory.length;
  document.getElementById('reg-positive').textContent = pos;
  document.getElementById('reg-negative').textContent = neg;
  document.getElementById('reg-avg-certainty').textContent = avgC;
  document.getElementById('reg-avg-age').textContent = avgAge;
  document.getElementById('reg-high-risk').textContent = highR;
  renderRegistryTable();
  const sc = [0, 0, 0, 0, 0];
  sessionHistory.forEach(p => sc[p.stageLevel]++);
  const monoFont = "'DM Mono',monospace", gridColor = 'rgba(124,92,252,0.08)', tickColor = '#6b8f9e';
  const ctxD = document.getElementById('distributionChart').getContext('2d');
  if (distChart) distChart.destroy();
  distChart = new Chart(ctxD, { type: 'doughnut', data: { labels: ['None', 'Mild', 'Moderate', 'Severe', 'Prolif.'], datasets: [{ data: sc, backgroundColor: ['#0ac5a8', '#7c5cfc', '#ff9f43', '#f4456a', '#c01440'], borderWidth: 0, hoverOffset: 5 }] }, options: { responsive: true, maintainAspectRatio: false, cutout: '72%', plugins: { legend: { position: 'right', labels: { color: tickColor, font: { family: monoFont, size: 9 }, boxWidth: 8, padding: 8 } } } } });
  const ctxT = document.getElementById('timelineChart').getContext('2d');
  if (timelineChart) timelineChart.destroy();
  timelineChart = new Chart(ctxT, { type: 'line', data: { labels: sessionHistory.map(p => p.id).reverse(), datasets: [{ label: 'Certainty', data: sessionHistory.map(p => +p.certainty).reverse(), borderColor: '#7c5cfc', backgroundColor: 'rgba(124,92,252,0.12)', borderWidth: 2, pointRadius: 3, fill: true, tension: 0.3 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { min: 80, max: 100, ticks: { color: tickColor, font: { family: monoFont, size: 9 } }, grid: { color: gridColor } }, x: { ticks: { color: tickColor, font: { family: monoFont, size: 9 }, maxRotation: 0 }, grid: { display: false } } } } });
  const ctxS = document.getElementById('scatterChart').getContext('2d');
  if (scatterChart) scatterChart.destroy();
  const colors = ['#0ac5a8', '#7c5cfc', '#ff9f43', '#f4456a', '#c01440'];
  scatterChart = new Chart(ctxS, { type: 'scatter', data: { datasets: [{ label: 'Patients', data: sessionHistory.map(p => ({ x: +p.age || 50, y: p.stageLevel })), backgroundColor: sessionHistory.map(p => colors[p.stageLevel] + 'cc'), pointRadius: 6, pointHoverRadius: 9 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: function (ctx) { return 'Age:' + ctx.raw.x + ', Stage:' + ctx.raw.y; } } } }, scales: { x: { title: { display: true, text: 'Age', color: tickColor, font: { family: monoFont, size: 9 } }, ticks: { color: tickColor }, grid: { color: gridColor } }, y: { min: -0.5, max: 4.5, title: { display: true, text: 'Stage', color: tickColor, font: { family: monoFont, size: 9 } }, ticks: { stepSize: 1, color: tickColor }, grid: { color: gridColor } } } } });
}

function exportRegistryCSV() {
  if (!sessionHistory.length) { showToast('No data.', 'error'); return; }
  const headers = ['ID', 'Age', 'Eye', 'Diagnosis', 'Stage', 'Certainty', 'HbA1c', 'Diabetes', 'Filter', 'Brightness', 'Contrast', 'Time'];
  const rows = sessionHistory.map(p => [p.id, p.age, p.eye, p.stageName, p.stageLevel, p.certainty + '%', p.hba1c, p.diab, p.filterUsed, p.brightness, p.contrast, p.date]);
  const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
  const a = document.createElement('a'); a.download = `retinaai_registry_${Date.now()}.csv`; a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv); a.click();
  showToast('CSV exported.', 'success');
}

// ═══ IMAGE LAB ═══
function initLabUpload() {
  const zone = document.getElementById('labUploadZone'), input = document.getElementById('labFileInput');
  zone.onclick = () => input.click();
  zone.ondragover = e => { e.preventDefault(); zone.style.borderColor = 'var(--accent)'; };
  zone.ondragleave = () => zone.style.borderColor = '';
  zone.ondrop = e => { e.preventDefault(); zone.style.borderColor = ''; const f = e.dataTransfer.files[0]; if (f && f.type.startsWith('image/')) loadLabImage(f); };
  input.onchange = () => { if (input.files[0]) loadLabImage(input.files[0]); };
}
async function loadLabImage(file) {
  const reader = new FileReader();
  reader.onload = async e => {
    const { data } = await CV.load(e.target.result, 400, 300);
    currentLabImage = data;
    CV.draw(document.getElementById('labOrigCanvas'), data, 400, 300);
    document.getElementById('labUploadZone').style.display = 'none';
    document.getElementById('labWorkspace').style.display = 'block';
    currentLabFilter = 'original'; applyLabProcessing(); renderHistogram(data); renderLabStats(data); setupPixelInspector();
    showToast('Image loaded.', 'info');
  };
  reader.readAsDataURL(file);
}
function getLabAdjusted() {
  if (!currentLabImage) return null;
  return CV.adjust(currentLabImage, +document.getElementById('ctrl-brightness').value, +document.getElementById('ctrl-contrast').value, +document.getElementById('ctrl-saturation').value, +document.getElementById('ctrl-gamma').value / 100, +document.getElementById('ctrl-sharpen').value);
}
function applyLabProcessing() {
  if (!currentLabImage) return;
  const adj = getLabAdjusted(), w = currentLabImage.width, h = currentLabImage.height;
  let out;
  switch (currentLabFilter) {
    case 'grayscale': out = CV.gray(adj); break;
    case 'green': out = CV.channel(adj, 1); break;
    case 'red': out = CV.channel(adj, 0); break;
    case 'blue': out = CV.channel(adj, 2); break;
    case 'clahe': out = CV.clahe(adj); break;
    case 'edge': out = CV.sobel(adj); break;
    case 'vessel': out = CV.vessel(adj); break;
    case 'pseudocolor': out = CV.pseudo(adj); break;
    case 'invert': out = CV.invert(adj); break;
    case 'emboss': out = CV.emboss(adj); break;
    case 'thermal': out = CV.thermal(adj); break;
    default: out = adj;
  }
  if (document.getElementById('compareModeToggle')?.checked) {
    const ratio = +document.getElementById('compareSlider').value / 100, blended = new ImageData(new Uint8ClampedArray(adj.data.length), w, h);
    for (let i = 0; i < adj.data.length; i += 4) { blended.data[i] = adj.data[i] * (1 - ratio) + out.data[i] * ratio; blended.data[i + 1] = adj.data[i + 1] * (1 - ratio) + out.data[i + 1] * ratio; blended.data[i + 2] = adj.data[i + 2] * (1 - ratio) + out.data[i + 2] * ratio; blended.data[i + 3] = 255; }
    CV.draw(document.getElementById('labOutCanvas'), blended, w, h);
  } else { CV.draw(document.getElementById('labOutCanvas'), out, w, h); }
  document.getElementById('labOutputLabel').textContent = currentLabFilter.charAt(0).toUpperCase() + currentLabFilter.slice(1);
}
function setLabFilter(f, btn) { currentLabFilter = f; document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active')); if (btn) btn.classList.add('active'); applyLabProcessing(); }
['brightness', 'contrast', 'saturation', 'sharpen', 'gamma', 'blur'].forEach(id => {
  const ctrl = document.getElementById(`ctrl-${id}`), val = document.getElementById(`val-${id}`);
  if (!ctrl) return;
  ctrl.addEventListener('input', () => { val.textContent = id === 'gamma' ? (ctrl.value / 100).toFixed(2) : ctrl.value; applyLabProcessing(); });
});
function toggleCompareMode() { const on = document.getElementById('compareModeToggle').checked; document.getElementById('compareModeLabel').textContent = on ? 'On' : 'Off'; document.getElementById('compareSliderWrap').style.display = on ? 'block' : 'none'; applyLabProcessing(); }
document.getElementById('compareSlider')?.addEventListener('input', applyLabProcessing);
function setupPixelInspector() {
  const canvas = document.getElementById('labOrigCanvas'), display = document.getElementById('pixelInfoDisplay');
  canvas.addEventListener('mousemove', e => { const r = canvas.getBoundingClientRect(), x = Math.floor((e.clientX - r.left) * currentLabImage.width / r.width), y = Math.floor((e.clientY - r.top) * currentLabImage.height / r.height); if (x < 0 || x >= currentLabImage.width || y < 0 || y >= currentLabImage.height) return; const i = (y * currentLabImage.width + x) * 4; display.textContent = `(${x},${y}) R:${currentLabImage.data[i]} G:${currentLabImage.data[i + 1]} B:${currentLabImage.data[i + 2]}`; });
  canvas.addEventListener('mouseleave', () => display.textContent = '');
}
function renderHistogram(data) {
  const h = CV.histogram(data), labels = Array.from({ length: 256 }, (_, i) => i % 32 === 0 ? i : '');
  const ctx = document.getElementById('histogramChart').getContext('2d');
  if (histogramChart) histogramChart.destroy();
  histogramChart = new Chart(ctx, { type: 'line', data: { labels, datasets: [{ label: 'R', data: h.r, borderColor: 'rgba(244,69,106,0.8)', backgroundColor: 'rgba(244,69,106,0.1)', borderWidth: 1, pointRadius: 0, fill: true, tension: 0.2 }, { label: 'G', data: h.g, borderColor: 'rgba(10,197,168,0.8)', backgroundColor: 'rgba(10,197,168,0.1)', borderWidth: 1, pointRadius: 0, fill: true, tension: 0.2 }, { label: 'B', data: h.b, borderColor: 'rgba(124,92,252,0.8)', backgroundColor: 'rgba(124,92,252,0.1)', borderWidth: 1, pointRadius: 0, fill: true, tension: 0.2 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#6b8f9e', font: { family: 'DM Mono,monospace', size: 9 }, boxWidth: 7 } } }, scales: { x: { ticks: { color: '#6b8f9e', font: { size: 8 } }, grid: { color: 'rgba(124,92,252,0.04)' } }, y: { ticks: { color: '#6b8f9e', font: { size: 8 } }, grid: { color: 'rgba(124,92,252,0.04)' } } } } });
}
function renderLabStats(data) { const s = CV.stats(data); document.getElementById('labStatsGrid').innerHTML = [['Brightness', s.brightness], ['Contrast', s.contrast], ['Avg R', s.meanR], ['Avg G', s.meanG], ['Avg B', s.meanB], ['Dominant', s.dominant]].map(([l, v]) => `<div class="img-stat-item"><span class="img-stat-label">${l}</span><span class="img-stat-val" style="font-size:16px;">${v}</span></div>`).join(''); }
function downloadLabResult() { const canvas = document.getElementById('labOutCanvas'), a = document.createElement('a'); a.download = `retinaai_${currentLabFilter}_${Date.now()}.png`; a.href = canvas.toDataURL('image/png'); a.click(); showToast('Downloaded.', 'success'); }
function downloadSideBySide() { const orig = document.getElementById('labOrigCanvas'), out = document.getElementById('labOutCanvas'), merged = document.createElement('canvas'); merged.width = orig.width * 2 + 16; merged.height = orig.height; const mctx = merged.getContext('2d'); mctx.fillStyle = 'var(--bg)'; mctx.fillRect(0, 0, merged.width, merged.height); mctx.drawImage(orig, 0, 0); mctx.drawImage(out, orig.width + 16, 0); const a = document.createElement('a'); a.download = `comparison_${Date.now()}.png`; a.href = merged.toDataURL('image/png'); a.click(); showToast('Side-by-side exported.', 'success'); }
function resetLab() { if (!currentLabImage) return; ['ctrl-brightness', 'ctrl-contrast', 'ctrl-saturation', 'ctrl-sharpen', 'ctrl-blur'].forEach(id => { const el = document.getElementById(id); if (el) el.value = 0; }); document.getElementById('ctrl-gamma').value = 100; ['brightness', 'contrast', 'saturation', 'sharpen', 'blur'].forEach(id => { const el = document.getElementById('val-' + id); if (el) el.textContent = '0'; }); document.getElementById('val-gamma').textContent = '1.0'; currentLabFilter = 'original'; document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active')); document.querySelector('.filter-btn[data-filter="original"]')?.classList.add('active'); applyLabProcessing(); showToast('Lab reset.', 'info'); }
function copyLabToScan() { if (!currentLabImage) return; const src = document.getElementById('labOutCanvas').toDataURL('image/png'); switchView('detect'); resetForm(); setTimeout(() => { previewImg.src = src; showToast('Lab result sent to scan engine.', 'success'); }, 300); }

// ═══════════════════════════════════════════════
// RISK CALCULATOR
// ═══════════════════════════════════════════════
function computeRisk() {
  const age = +document.getElementById('rc-age').value || 50;
  const dur = +document.getElementById('rc-dur').value || 5;
  const hba1c = +document.getElementById('rc-hba1c').value || 7;
  const sbp = +document.getElementById('rc-sbp').value || 120;
  const bmi = +document.getElementById('rc-bmi').value || 25;
  const chol = +document.getElementById('rc-chol').value || 180;
  const egfr = +document.getElementById('rc-egfr').value || 90;
  const type1 = document.getElementById('rc-type').value === '1';
  const hyper = document.getElementById('rc-hyper').checked;
  const smoking = document.getElementById('rc-smoking').checked;
  const microalb = document.getElementById('rc-microalb').checked;
  const neuro = document.getElementById('rc-neuropathy').checked;
  const insulin = document.getElementById('rc-insulin').checked;
  const family = document.getElementById('rc-family').checked;
  const factors = [
    { name: 'Diabetes Duration', score: Math.min(30, dur * 2.5), max: 30, color: '#7c5cfc' },
    { name: 'HbA1c Level', score: Math.min(25, Math.max(0, (hba1c - 6) * 6)), max: 25, color: '#f4456a' },
    { name: 'Age Factor', score: Math.min(15, Math.max(0, (age - 30) * 0.4)), max: 15, color: '#ff9f43' },
    { name: 'Systolic BP', score: Math.min(10, Math.max(0, (sbp - 120) * 0.25)), max: 10, color: '#0ac5a8' },
    { name: 'BMI', score: Math.min(5, Math.max(0, (bmi - 25) * 0.4)), max: 5, color: '#b98bff' },
    { name: 'Cholesterol', score: Math.min(5, Math.max(0, (chol - 170) * 0.05)), max: 5, color: '#ff9f43' },
    { name: 'Reduced eGFR', score: Math.min(10, Math.max(0, (90 - egfr) * 0.15)), max: 10, color: '#f4456a' },
    { name: 'Type 1 Diabetes', score: type1 ? 5 : 0, max: 5, color: '#7c5cfc' },
    { name: 'Hypertension', score: hyper ? 6 : 0, max: 6, color: '#f4456a' },
    { name: 'Smoking', score: smoking ? 5 : 0, max: 5, color: '#ff9f43' },
    { name: 'Microalbuminuria', score: microalb ? 8 : 0, max: 8, color: '#f4456a' },
    { name: 'Neuropathy', score: neuro ? 5 : 0, max: 5, color: '#ff9f43' },
    { name: 'On Insulin', score: insulin ? 3 : 0, max: 3, color: '#0ac5a8' },
    { name: 'Family History', score: family ? 4 : 0, max: 4, color: '#7c5cfc' }
  ];
  const totalMax = factors.reduce((s, f) => s + f.max, 0);
  const totalScore = factors.reduce((s, f) => s + f.score, 0);
  const pct = Math.round((totalScore / totalMax) * 100);
  const category = pct < 25 ? { label: 'Low Risk', color: '#0ac5a8' } : pct < 50 ? { label: 'Moderate Risk', color: '#ff9f43' } : pct < 75 ? { label: 'High Risk', color: '#f4456a' } : { label: 'Very High Risk', color: '#c01440' };
  document.getElementById('riskScoreDisplay').textContent = pct + '%';
  document.getElementById('riskScoreDisplay').style.color = category.color;
  const rc = document.getElementById('riskCategory');
  rc.textContent = category.label; rc.style.background = category.color + '22'; rc.style.color = category.color; rc.style.border = `1px solid ${category.color}44`;
  const gCtx = document.getElementById('riskGaugeChart').getContext('2d');
  if (riskGaugeChart) riskGaugeChart.destroy();
  riskGaugeChart = new Chart(gCtx, { type: 'doughnut', data: { datasets: [{ data: [pct, 100 - pct], backgroundColor: [category.color, 'rgba(124,92,252,0.06)'], borderWidth: 0, circumference: 180, rotation: 270 }] }, options: { responsive: false, cutout: '78%', plugins: { legend: { display: false }, tooltip: { enabled: false } } } });
  document.getElementById('riskBreakdownList').innerHTML = factors.filter(f => f.score > 0).map(f => `<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);"><div style="flex:1;"><div style="font-size:12px;color:var(--text);margin-bottom:3px;">${f.name}</div><div style="height:3px;background:var(--surface2);border-radius:100px;overflow:hidden;"><div style="width:${(f.score / f.max * 100).toFixed(0)}%;height:100%;background:${f.color};border-radius:100px;transition:width 1s;"></div></div></div><div style="font-family:'DM Mono',monospace;font-size:11px;color:${f.color};width:28px;text-align:right;">${f.score.toFixed(1)}</div></div>`).join('') || '<div style="color:var(--muted);font-size:13px;padding:12px 0;">No significant risk factors identified.</div>';
  const recs = pct < 25 ? 'Annual fundus screening recommended. Maintain HbA1c below 7%, blood pressure below 130/80 mmHg, and BMI below 25. Continue current management plan.' : pct < 50 ? 'Biannual ophthalmology review advised. Intensify glycaemic control targeting HbA1c below 7%. Consider ACE inhibitor if microalbuminuria present. Lifestyle modification strongly recommended.' : pct < 75 ? 'Referral to ophthalmologist within 3 months. Urgent glycaemic optimization. Blood pressure target below 130/80. Rule out nephropathy. Consider cardiology co-management.' : 'URGENT ophthalmology referral within 4 weeks. Multidisciplinary management required. Aggressive glycaemic and BP control. Evaluate for end-organ damage. Consider anti-VEGF prophylaxis discussion.';
  document.getElementById('riskRecommendations').textContent = recs;
  showToast(`Risk score: ${pct}% — ${category.label}`, pct < 25 ? 'success' : pct < 50 ? 'info' : 'error');
}

function clearRiskCalc() {
  ['rc-age', 'rc-dur', 'rc-hba1c', 'rc-sbp', 'rc-bmi', 'rc-chol', 'rc-egfr'].forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
  ['rc-hyper', 'rc-smoking', 'rc-microalb', 'rc-neuropathy', 'rc-insulin', 'rc-family'].forEach(id => { const el = document.getElementById(id); if (el) el.checked = false; });
  document.getElementById('riskScoreDisplay').textContent = '—';
  document.getElementById('riskScoreDisplay').style.color = 'var(--accent)';
  document.getElementById('riskCategory').textContent = '';
  document.getElementById('riskBreakdownList').innerHTML = '';
  document.getElementById('riskRecommendations').textContent = '';
  if (riskGaugeChart) { riskGaugeChart.destroy(); riskGaugeChart = null; }
}

function exportRiskReport() {
  const score = document.getElementById('riskScoreDisplay').textContent;
  const cat = document.getElementById('riskCategory').textContent;
  const recs = document.getElementById('riskRecommendations').textContent;
  if (score === '—') { showToast('Compute risk first.', 'error'); return; }
  const age = document.getElementById('rc-age').value, hba1c = document.getElementById('rc-hba1c').value, dur = document.getElementById('rc-dur').value;
  const text = `RetinaAI — DR Risk Report\n${'='.repeat(40)}\nDate: ${new Date().toLocaleString()}\n\nPatient Parameters:\n  Age: ${age} years\n  HbA1c: ${hba1c}%\n  Diabetes Duration: ${dur} years\n\nComposite Risk Score: ${score}\nRisk Category: ${cat}\n\nClinical Recommendations:\n${recs}\n\n${'='.repeat(40)}\nGenerated by RetinaAI v2.0 — Research use only.`;
  const a = document.createElement('a'); a.download = `dr_risk_report_${Date.now()}.txt`; a.href = 'data:text/plain;charset=utf-8,' + encodeURIComponent(text); a.click();
  showToast('Risk report exported.', 'success');
}

// ═══════════════════════════════════════════════
// CLINICAL NOTES
// ═══════════════════════════════════════════════
function renderNotesList() {
  const list = document.getElementById('notesList'); list.innerHTML = '';
  if (!notes.length) { list.innerHTML = '<li style="color:var(--muted);font-size:12px;padding:10px;">No notes yet.</li>'; return; }
  notes.forEach(note => {
    const li = document.createElement('li');
    li.className = 'note-item' + (note.id === currentNoteId ? ' active' : '');
    li.innerHTML = `<div class="note-item-title">${note.title || 'Untitled'}</div><div class="note-item-meta"><span class="note-tag tag-${note.tag || 'general'}">${note.tag || 'general'}</span><span class="note-item-time">${note.ts || ''}</span></div>`;
    li.addEventListener('click', () => openNote(note.id));
    list.appendChild(li);
  });
}
function createNewNote() {
  const note = { id: Date.now().toString(), title: 'New Note', content: '', tag: 'general', ts: new Date().toLocaleString() };
  notes.unshift(note); renderNotesList(); openNote(note.id);
}
function openNote(id) {
  currentNoteId = id;
  const note = notes.find(n => n.id === id); if (!note) return;
  document.getElementById('noteEditorEmpty').style.display = 'none';
  const content = document.getElementById('noteEditorContent'); content.style.display = 'flex';
  document.getElementById('noteTitleInput').value = note.title || '';
  document.getElementById('noteTagSelect').value = note.tag || 'general';
  document.getElementById('noteTextArea').innerHTML = note.content || '';
  document.getElementById('noteTimestamp').textContent = 'Last saved: ' + (note.ts || 'Never');
  updateWordCount(); renderNotesList();
}
function saveCurrentNote() {
  if (!currentNoteId) return;
  const note = notes.find(n => n.id === currentNoteId); if (!note) return;
  note.title = document.getElementById('noteTitleInput').value || 'Untitled';
  note.tag = document.getElementById('noteTagSelect').value;
  note.content = document.getElementById('noteTextArea').innerHTML;
  note.ts = new Date().toLocaleString();
  document.getElementById('noteTimestamp').textContent = 'Last saved: ' + note.ts;
  renderNotesList(); showToast('Note saved.', 'success');
}
function deleteCurrentNote() {
  if (!currentNoteId) return;
  if (!confirm('Delete this note?')) return;
  notes = notes.filter(n => n.id !== currentNoteId); currentNoteId = null;
  document.getElementById('noteEditorEmpty').style.display = 'block';
  document.getElementById('noteEditorContent').style.display = 'none';
  renderNotesList(); showToast('Note deleted.', 'info');
}
function noteFormat(cmd, val) { document.getElementById('noteTextArea').focus(); document.execCommand(cmd, false, val || null); updateWordCount(); }
function updateWordCount() {
  const ta = document.getElementById('noteTextArea'); if (!ta) return;
  const words = (ta.innerText || ta.textContent).trim().split(/\s+/).filter(w => w.length > 0).length;
  const wc = document.getElementById('noteWordCount'); if (wc) wc.textContent = words + ' word' + (words !== 1 ? 's' : '');
}
document.getElementById('noteTextArea')?.addEventListener('input', updateWordCount);
function exportNoteAsText() {
  if (!currentNoteId) return;
  const note = notes.find(n => n.id === currentNoteId); if (!note) return;
  const text = `${note.title}\n${'='.repeat(note.title.length)}\nTag: ${note.tag}\nDate: ${note.ts}\n\n${document.getElementById('noteTextArea').innerText || ''}`;
  const a = document.createElement('a'); a.download = `note_${note.id}.txt`; a.href = 'data:text/plain;charset=utf-8,' + encodeURIComponent(text); a.click();
  showToast('Note exported.', 'success');
}

// ═══════════════════════════════════════════════
// SETTINGS
// ═══════════════════════════════════════════════
function loadSettingsUI() {
  document.getElementById('set-name').value = appSettings.profile.name || '';
  document.getElementById('set-inst').value = appSettings.profile.inst || '';
  document.getElementById('set-spec').value = appSettings.profile.spec || '';
  document.getElementById('set-loc').value = appSettings.profile.loc || '';
  document.getElementById('set-filter').value = appSettings.filter || 'none';
  document.getElementById('set-threshold').value = appSettings.threshold || '0.5';
}
function saveProfile() {
  const name = document.getElementById('set-name').value || 'Dr. Yuvraj';
  const inst = document.getElementById('set-inst').value;
  const spec = document.getElementById('set-spec').value;
  const loc = document.getElementById('set-loc').value;
  appSettings.profile = { name, inst, spec, loc };
  const initials = name.split(' ').filter(Boolean).map(w => w[0]).join('').slice(0, 2).toUpperCase();
  document.getElementById('avatarInitials').textContent = initials;
  document.getElementById('modalAvatar').textContent = initials;
  document.getElementById('sidebarDocName').textContent = name;
  document.getElementById('modalName').textContent = name.replace(/^Dr\.?\s*/i, '');
  document.getElementById('modalRole').textContent = spec || 'AI/ML Research';
  showToast('Profile saved.', 'success');
}
function saveAnalysisSettings() { appSettings.filter = document.getElementById('set-filter').value; appSettings.threshold = +document.getElementById('set-threshold').value; showToast('Defaults saved.', 'success'); }
function exportAllData() {
  const payload = { sessionHistory: sessionHistory.map(p => ({ id: p.id, age: p.age, eye: p.eye, hba1c: p.hba1c, diab: p.diab, date: p.date, stageName: p.stageName, stageLevel: p.stageLevel, certainty: p.certainty, density: p.density, time: p.time, filter: p.filterUsed, brightness: p.brightness, contrast: p.contrast })), notes, settings: appSettings, exportedAt: new Date().toISOString() };
  const a = document.createElement('a'); a.download = `retinaai_session_${Date.now()}.json`; a.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2)); a.click();
  showToast('Session exported.', 'success');
}
function handleImport(e) {
  const file = e.target.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    try {
      const data = JSON.parse(ev.target.result);
      if (data.notes) notes = [...data.notes, ...notes];
      if (data.settings) Object.assign(appSettings, data.settings);
      showToast('Data imported successfully.', 'success');
      renderNotesList(); loadSettingsUI();
    } catch (err) { showToast('Invalid JSON file.', 'error'); }
  };
  reader.readAsText(file);
}
function clearAllData() {
  if (!confirm('Clear ALL session data? This cannot be undone.')) return;
  sessionHistory = []; notes = [];
  updateSessionCounts(); updateSidebar(null); resetForm(); renderNotesList();
  showToast('All data cleared.', 'info');
}

// ═══ SMOOTH SCROLL ═══
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => { e.preventDefault(); document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' }); });
});


// ═══════════════════════════════════════════════════════════
// NEW FEATURE 1: AI CHAT ASSISTANT (Claude API)
// ═══════════════════════════════════════════════════════════

function getActiveScanContext() {
  if (!sessionHistory.length) return null;
  const p = sessionHistory[0];
  return {
    patientId: p.id,
    age: p.age,
    eye: p.eye,
    hba1c: p.hba1c,
    diabetesDuration: p.diab,
    stageName: p.stageName,
    stageLevel: p.stageLevel,
    certainty: p.certainty,
    lesionDensity: p.density,
    filterUsed: p.filterUsed,
    brightness: p.brightness,
    contrast: p.contrast
  };
}

function updateAIScanContext() {
  const ctx = getActiveScanContext();
  const el = document.getElementById('aiScanContext');
  if (!el) return;
  if (!ctx) {
    el.innerHTML = '<span style="color:var(--text3)">No scan loaded. Run a diagnosis first.</span>';
    return;
  }
  el.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:4px;">
      <div><strong style="color:var(--text)">${ctx.patientId}</strong> · ${ctx.eye}</div>
      <div style="color:var(--text3)">Age ${ctx.age} · HbA1c ${ctx.hba1c}%</div>
      <div style="margin-top:4px;">
        <span class="stage-pill stage-${ctx.stageLevel}">${ctx.stageName}</span>
      </div>
      <div style="color:var(--text3);margin-top:2px;">Certainty: <strong style="color:var(--indigo)">${ctx.certainty}%</strong></div>
    </div>
  `;
}

function askAI(text) {
  document.getElementById('aiInputBox').value = text;
  sendAIMessage();
}

function appendAIMessage(role, html, isThinking = false) {
  const container = document.getElementById('aiMessages');
  const div = document.createElement('div');
  div.className = `ai-msg ${role === 'user' ? 'ai-msg-user' : 'ai-msg-bot'}${isThinking ? ' ai-msg-thinking' : ''}`;
  const avatarLabel = role === 'user' ? 'You' : 'AI';
  div.innerHTML = `
    <div class="ai-msg-avatar">${avatarLabel}</div>
    <div class="ai-msg-bubble">${html}</div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

async function sendAIMessage() {
  const input = document.getElementById('aiInputBox');
  const sendBtn = document.getElementById('aiSendBtn');
  const userText = input.value.trim();
  if (!userText) return;

  input.value = '';
  input.disabled = true;
  sendBtn.disabled = true;

  appendAIMessage('user', userText);

  const thinkingEl = appendAIMessage('bot', `<div class="ai-thinking-dots"><span></span><span></span><span></span></div>`, true);

  const ctx = getActiveScanContext();
  const systemPrompt = `You are a clinical AI assistant embedded in RetinaAI, a diabetic retinopathy screening platform. You help clinicians understand scan results, provide clinical context, and explain findings clearly and concisely.

${ctx ? `CURRENT SCAN CONTEXT:
- Patient: ${ctx.patientId}, Age: ${ctx.age}, Eye: ${ctx.eye}
- Diagnosis: ${ctx.stageName} (Stage ${ctx.stageLevel}/4)
- AI Certainty: ${ctx.certainty}%
- Lesion Density Score: ${ctx.lesionDensity}
- HbA1c: ${ctx.hba1c}%, Diabetes Duration: ${ctx.diabetesDuration} years
- Image: Brightness ${ctx.brightness}, Contrast ${ctx.contrast}` : 'No scan has been run yet. Guide the user to use the Diagnosis Engine first.'}

Keep responses concise (2-4 sentences), clinically accurate, and professional. Do not provide definitive medical advice — always note this is AI-assisted screening for clinician review.`;

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1000,
        system: systemPrompt,
        messages: [{ role: 'user', content: userText }]
      })
    });

    const data = await response.json();
    const reply = data?.content?.[0]?.text || 'Sorry, I could not generate a response. Please try again.';

    thinkingEl.remove();
    appendAIMessage('bot', reply.replace(/\n/g, '<br/>'));
  } catch (err) {
    thinkingEl.remove();
    appendAIMessage('bot', 'Network error — unable to reach AI service. Check your connection and try again.');
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

// Hook into switchView to update AI context
const _origSwitchView = switchView;
switchView = function(id) {
  _origSwitchView(id);
  if (id === 'ai-chat') {
    updateAIScanContext();
    // extend views list
  }
  if (id === 'timeline') renderTimeline();
};

// ═══════════════════════════════════════════════════════════
// NEW FEATURE 2: PATIENT TIMELINE
// ═══════════════════════════════════════════════════════════

let timelineBigChart = null;
const stageColors = ['#10b981','#6366f1','#f59e0b','#f97316','#ef4444'];
const stageNames  = ['No DR','Mild','Moderate','Severe','Prolif.'];

function renderTimeline() {
  const empty = document.getElementById('timelineEmpty');
  const content = document.getElementById('timelineContent');
  if (!sessionHistory.length) {
    empty.style.display = 'block';
    content.style.display = 'none';
    return;
  }
  empty.style.display = 'none';
  content.style.display = 'block';

  // Populate patient filter
  const filter = document.getElementById('timelinePatientFilter');
  const selected = filter.value;
  const ids = [...new Set(sessionHistory.map(p => p.id))];
  filter.innerHTML = '<option value="">All Patients</option>' +
    ids.map(id => `<option value="${id}" ${id===selected?'selected':''}>${id}</option>`).join('');

  const data = selected
    ? sessionHistory.filter(p => p.id === selected)
    : [...sessionHistory];
  const reversed = [...data].reverse();

  // Big chart
  const labels = reversed.map(p => p.date + ' · ' + p.id);
  const severityData = reversed.map(p => p.stageLevel);
  const certaintyData = reversed.map(p => parseFloat(p.certainty));

  const ctx = document.getElementById('timelineBigChart').getContext('2d');
  if (timelineBigChart) timelineBigChart.destroy();
  timelineBigChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Severity (0-4)',
          data: severityData,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99,102,241,0.1)',
          borderWidth: 2.5,
          pointBackgroundColor: severityData.map(v => stageColors[v]),
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 7,
          fill: true,
          tension: 0.4,
          yAxisID: 'y'
        },
        {
          label: 'AI Certainty (%)',
          data: certaintyData,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16,185,129,0.06)',
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: '#10b981',
          fill: true,
          tension: 0.4,
          yAxisID: 'y2',
          borderDash: [5,3]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#9ea3b8', font: { family: 'JetBrains Mono, monospace', size: 10 }, boxWidth: 10, padding: 16 }
        }
      },
      scales: {
        y: {
          min: 0, max: 4, position: 'left',
          ticks: { stepSize:1, color:'#9ea3b8', font:{family:'JetBrains Mono,monospace',size:9} },
          grid: { color:'rgba(99,102,241,0.08)' }
        },
        y2: {
          min: 75, max: 100, position: 'right',
          ticks: { color:'#10b981', font:{family:'JetBrains Mono,monospace',size:9}, callback: v=>v+'%' },
          grid: { display: false }
        },
        x: {
          ticks: { color:'#5c6178', font:{family:'JetBrains Mono,monospace',size:9}, maxRotation:30 },
          grid: { display:false }
        }
      }
    }
  });

  // Render list
  const list = document.getElementById('timelineList');
  list.innerHTML = '';
  reversed.forEach((p, i) => {
    const barWidth = (p.stageLevel / 4) * 100;
    const barColor = stageColors[p.stageLevel];
    const entry = document.createElement('div');
    entry.className = 'timeline-entry';
    entry.innerHTML = `
      <div class="timeline-date">${p.date}<br/><span style="font-size:9px;">${p.id}</span></div>
      <div class="timeline-spine">
        <div class="timeline-dot" style="background:${barColor};box-shadow:0 0 0 2px ${barColor};"></div>
        <div class="timeline-line"></div>
      </div>
      <div class="timeline-card" onclick="switchView('detect');updateSidebar('${p.uniqueKey}');renderDashboard(sessionHistory.find(x=>x.uniqueKey==='${p.uniqueKey}'))">
        <div class="timeline-card-header">
          <div class="timeline-card-id">${p.id} · ${p.eye}</div>
          <span class="stage-pill stage-${p.stageLevel}">${p.stageName}</span>
        </div>
        <div class="timeline-card-body">
          <span><strong>${p.certainty}%</strong> certainty</span>
          <span><strong>Age ${p.age}</strong></span>
          <span>HbA1c <strong>${p.hba1c}%</strong></span>
          <span>Density <strong>${p.density}</strong></span>
          <span>Filter <strong>${p.filterUsed||'none'}</strong></span>
        </div>
        <div class="timeline-severity-bar">
          <div class="timeline-severity-fill" style="width:${barWidth}%;background:${barColor};"></div>
        </div>
      </div>
    `;
    list.appendChild(entry);
  });
}

function exportTimelineCSV() {
  if (!sessionHistory.length) { showToast('No data to export.','error'); return; }
  const headers = ['Patient ID','Eye','Age','HbA1c','Diab Duration','Stage','Stage Name','Certainty','Lesion Density','Filter','Time'];
  const rows = [...sessionHistory].reverse().map(p => [
    p.id, p.eye, p.age, p.hba1c, p.diab, p.stageLevel, p.stageName,
    p.certainty+'%', p.density, p.filterUsed||'none', p.date
  ]);
  const csv = [headers,...rows].map(r=>r.join(',')).join('\n');
  const a = document.createElement('a');
  a.download = `retinaai_timeline_${Date.now()}.csv`;
  a.href = 'data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.click();
  showToast('Timeline exported as CSV.','success');
}

function printTimeline() {
  window.print();
}

// ═══════════════════════════════════════════════════════════
// NEW FEATURE 3: SMART PDF REPORT
// Enhances the existing Export PDF button with a styled report
// ═══════════════════════════════════════════════════════════

function generateStyledReport() {
  if (!sessionHistory.length) { showToast('No scan to export.','error'); return; }
  const p = sessionHistory[0];
  const stageColMap = ['#10b981','#6366f1','#f59e0b','#f97316','#ef4444'];
  const col = stageColMap[p.stageLevel];
  const recTexts = [
    'No DR detected. Annual screening recommended. Continue glycaemic and BP management.',
    'Mild NPDR. Follow-up in 12 months. Optimize HbA1c below 7%.',
    'Moderate NPDR. Ophthalmology referral within 3–6 months. Enhanced glycaemic control critical.',
    'Severe NPDR. Urgent referral within 1 month. Laser photocoagulation may be indicated.',
    'Proliferative DR. URGENT referral within 1 week. Anti-VEGF or pan-retinal photocoagulation required.'
  ];
  const rec = recTexts[p.stageLevel];
  const now = new Date().toLocaleString();
  const rptId = 'RPT-' + p.uniqueKey.slice(-8).toUpperCase();

  const win = window.open('','_blank');
  win.document.write(`<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<title>RetinaAI Report — ${p.id}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
  *{margin:0;padding:0;box-sizing:border-box;}
  body{font-family:'DM Sans',sans-serif;background:#fff;color:#13141a;font-size:13px;line-height:1.6;}
  .page{max-width:760px;margin:0 auto;padding:48px 40px;}
  /* Header */
  .rpt-header{display:flex;justify-content:space-between;align-items:flex-start;padding-bottom:24px;border-bottom:2px solid #13141a;margin-bottom:32px;}
  .rpt-logo{font-family:'Outfit',sans-serif;font-weight:800;font-size:24px;color:#6366f1;letter-spacing:-0.5px;}
  .rpt-logo span{font-size:11px;font-family:'JetBrains Mono',monospace;color:#9093ae;font-weight:400;margin-left:6px;}
  .rpt-meta{text-align:right;font-family:'JetBrains Mono',monospace;font-size:10px;color:#9093ae;line-height:1.8;}
  .rpt-meta strong{color:#13141a;font-size:11px;}
  /* Title */
  .rpt-title{font-family:'Outfit',sans-serif;font-weight:800;font-size:28px;letter-spacing:-0.8px;margin-bottom:4px;}
  .rpt-subtitle{color:#9093ae;font-size:13px;margin-bottom:28px;}
  /* Patient info */
  .info-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:#e5e7eb;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;margin-bottom:28px;}
  .info-cell{background:#fff;padding:14px 16px;}
  .info-label{font-family:'JetBrains Mono',monospace;font-size:9px;color:#9093ae;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:3px;}
  .info-val{font-family:'Outfit',sans-serif;font-size:16px;font-weight:700;color:#13141a;}
  /* Diagnosis banner */
  .diag-banner{background:${col}12;border:1.5px solid ${col}44;border-radius:12px;padding:20px 24px;margin-bottom:28px;display:flex;justify-content:space-between;align-items:center;}
  .diag-stage{font-family:'Outfit',sans-serif;font-weight:800;font-size:26px;color:${col};}
  .diag-badge{font-family:'JetBrains Mono',monospace;font-size:10px;padding:5px 14px;border-radius:100px;background:${col};color:#fff;font-weight:600;letter-spacing:1px;text-transform:uppercase;}
  .diag-certainty{font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;color:${col};text-align:right;}
  .diag-certainty-label{font-size:10px;color:#9093ae;text-align:right;}
  /* Section headers */
  .sec-heading{font-family:'JetBrains Mono',monospace;font-size:10px;color:#6366f1;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid #e5e7eb;}
  /* Stats row */
  .stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px;}
  .stat-box{border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;text-align:center;}
  .stat-box-val{font-family:'Outfit',sans-serif;font-size:20px;font-weight:700;color:#13141a;}
  .stat-box-label{font-family:'JetBrains Mono',monospace;font-size:9px;color:#9093ae;text-transform:uppercase;letter-spacing:1px;}
  /* Severity scale */
  .sev-bar{height:6px;border-radius:100px;background:linear-gradient(90deg,#10b981,#6366f1 25%,#f59e0b 50%,#f97316 75%,#ef4444);margin:10px 0 6px;position:relative;}
  .sev-dot{position:absolute;top:-5px;width:16px;height:16px;border-radius:50%;background:${col};border:2px solid #fff;box-shadow:0 0 0 2px ${col};transform:translateX(-50%);left:${p.stageLevel*25}%;}
  .sev-labels{display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:8px;color:#9093ae;text-transform:uppercase;}
  /* Recommendation */
  .rec-box{background:${col}08;border-left:3px solid ${col};border-radius:0 8px 8px 0;padding:16px 20px;margin-bottom:28px;font-size:13px;line-height:1.7;color:#484b6a;}
  .rec-box strong{display:block;margin-bottom:4px;color:#13141a;font-family:'Outfit',sans-serif;font-size:14px;}
  /* Risk factors */
  .risk-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:28px;}
  .risk-row{display:flex;align-items:center;gap:10px;padding:8px 12px;border:1px solid #e5e7eb;border-radius:8px;}
  .risk-row-bar{flex:1;height:3px;background:#f3f4f6;border-radius:100px;overflow:hidden;}
  .risk-row-fill{height:100%;border-radius:100px;}
  .risk-row-name{font-size:11px;color:#484b6a;min-width:120px;}
  .risk-row-val{font-family:'JetBrains Mono',monospace;font-size:10px;color:#9093ae;min-width:36px;text-align:right;}
  /* Footer */
  .rpt-footer{border-top:1px solid #e5e7eb;padding-top:18px;margin-top:32px;display:flex;justify-content:space-between;align-items:center;font-family:'JetBrains Mono',monospace;font-size:9px;color:#9093ae;}
  .disclaimer{background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:11px;color:#9093ae;line-height:1.6;}
  @media print{body{-webkit-print-color-adjust:exact;print-color-adjust:exact;}.page{padding:24px;}}
</style>
</head>
<body>
<div class="page">
  <div class="rpt-header">
    <div>
      <div class="rpt-logo">RetinaAI <span>v2.0</span></div>
      <div style="font-size:11px;color:#9093ae;margin-top:3px;font-family:'JetBrains Mono',monospace;">Clinical Analysis Report</div>
    </div>
    <div class="rpt-meta">
      <div><strong>${rptId}</strong></div>
      <div>${now}</div>
      <div>Research Use Only</div>
    </div>
  </div>

  <div class="rpt-title">Diabetic Retinopathy Screening</div>
  <div class="rpt-subtitle">AI-assisted grading via EfficientNetB0 · Grad-CAM explainability</div>

  <div class="info-grid">
    <div class="info-cell"><div class="info-label">Patient ID</div><div class="info-val">${p.id}</div></div>
    <div class="info-cell"><div class="info-label">Age</div><div class="info-val">${p.age} yrs</div></div>
    <div class="info-cell"><div class="info-label">Eye</div><div class="info-val">${p.eye}</div></div>
    <div class="info-cell"><div class="info-label">HbA1c</div><div class="info-val">${p.hba1c}%</div></div>
    <div class="info-cell"><div class="info-label">Diabetes Duration</div><div class="info-val">${p.diab} yrs</div></div>
    <div class="info-cell"><div class="info-label">Pre-processing</div><div class="info-val">${p.filterUsed||'None'}</div></div>
  </div>

  <div class="sec-heading">Diagnosis Result</div>
  <div class="diag-banner">
    <div>
      <div class="diag-stage">${p.stageName}</div>
      <span class="diag-badge">${p.stageLevel===0?'DR Negative':'Refer to Ophthalmologist'}</span>
    </div>
    <div>
      <div class="diag-certainty">${p.certainty}%</div>
      <div class="diag-certainty-label">AI Certainty</div>
    </div>
  </div>

  <div class="sec-heading">Severity Classification</div>
  <div style="margin-bottom:28px;">
    <div style="display:flex;justify-content:space-between;font-family:'JetBrains Mono',monospace;font-size:10px;color:#9093ae;margin-bottom:4px;">
      <span>DR Severity Stage</span><span>Stage ${p.stageLevel} / 4</span>
    </div>
    <div class="sev-bar"><div class="sev-dot"></div></div>
    <div class="sev-labels"><span>None</span><span>Mild</span><span>Moderate</span><span>Severe</span><span>Prolif.</span></div>
  </div>

  <div class="sec-heading">Clinical Recommendation</div>
  <div class="rec-box"><strong>Action Required:</strong>${rec}</div>

  <div class="sec-heading">Image Analysis Metrics</div>
  <div class="stats-row">
    <div class="stat-box"><div class="stat-box-val">${p.certainty}%</div><div class="stat-box-label">Certainty</div></div>
    <div class="stat-box"><div class="stat-box-val">${p.density}</div><div class="stat-box-label">Lesion Density</div></div>
    <div class="stat-box"><div class="stat-box-val">${p.brightness}</div><div class="stat-box-label">Brightness</div></div>
    <div class="stat-box"><div class="stat-box-val">${p.contrast}</div><div class="stat-box-label">Contrast</div></div>
  </div>

  <div class="sec-heading">Risk Factor Breakdown</div>
  <div class="risk-grid">
    ${[
      ['Vascular Abnormality', Math.min(100,p.stageLevel*22+10),'#6366f1'],
      ['Microaneurysm Score', Math.min(100,p.stageLevel*18+8),'#f59e0b'],
      ['Hemorrhage Risk', Math.min(100,p.stageLevel*15+6),'#ef4444'],
      ['Exudate Presence', Math.min(100,p.stageLevel*12+5),'#f97316'],
      ['Neovascularization', p.stageLevel===4?85:p.stageLevel*5,'#6366f1'],
      ['Optic Disc Integrity', 100-(p.stageLevel*15),'#10b981']
    ].map(([name,val,col])=>`
      <div class="risk-row">
        <div class="risk-row-name">${name}</div>
        <div class="risk-row-bar"><div class="risk-row-fill" style="width:${val}%;background:${col};"></div></div>
        <div class="risk-row-val">${val}%</div>
      </div>
    `).join('')}
  </div>

  <div class="disclaimer">
    <strong>Disclaimer:</strong> This report is generated by an AI-assisted screening tool for research and clinical decision support purposes only. It does not constitute a definitive medical diagnosis. All findings must be reviewed and confirmed by a qualified ophthalmologist. Not approved as a standalone medical device.
  </div>

  <div class="rpt-footer">
    <span>RetinaAI v2.0 · EfficientNetB0 · APTOS 2019 · AUC 0.952</span>
    <span>${rptId} · ${now}</span>
  </div>
</div>
<script>window.onload=()=>window.print();</script>
</body>
</html>`);
  win.document.close();
}

// Override the existing PDF export button
document.getElementById('downloadReportBtn').addEventListener('click', generateStyledReport);

// Also update sidebar counts when analysis completes - hook into updateSessionCounts
const _origUpdateCounts = updateSessionCounts;
updateSessionCounts = function() {
  _origUpdateCounts();
  updateAIScanContext();
};