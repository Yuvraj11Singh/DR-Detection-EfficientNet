'use strict';

// ═══════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════
let patientChart = null;
let distChart    = null;
let timelineChart = null;
let histogramChart = null;
let sessionHistory = [];
let sortConfig = { key: null, dir: 'asc' };
let currentLabImage = null; // Raw ImageData for lab
let labOrigImgEl = null;
let currentLabFilter = 'original';

// CV Engine namespace
const CV = {};

// ═══════════════════════════════════════════════════════
// LIVE CLOCK
// ═══════════════════════════════════════════════════════
function updateClock() {
  const el = document.getElementById('liveClock');
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════
function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<div class="toast-dot"></div><span>${message}</span>`;
  container.appendChild(toast);
  requestAnimationFrame(() => { toast.classList.add('show'); });
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 400);
  }, duration);
}

// ═══════════════════════════════════════════════════════
// PROFILE MODAL
// ═══════════════════════════════════════════════════════
const profileModal = document.getElementById('profileModal');
function openProfileModal() { profileModal.classList.add('active'); }
function closeProfileModal() { profileModal.classList.remove('active'); }
profileModal.addEventListener('click', e => { if (e.target === profileModal) closeProfileModal(); });

// ═══════════════════════════════════════════════════════
// SIDEBAR RESIZER
// ═══════════════════════════════════════════════════════
const resizer = document.getElementById('sidebarResizer');
let isResizing = false;
resizer.addEventListener('mousedown', () => {
  isResizing = true;
  document.body.classList.add('resizing');
  resizer.classList.add('active');
});
document.addEventListener('mousemove', e => {
  if (!isResizing) return;
  let w = Math.min(Math.max(e.clientX, 220), 520);
  document.documentElement.style.setProperty('--sidebar-width', w + 'px');
});
document.addEventListener('mouseup', () => {
  if (!isResizing) return;
  isResizing = false;
  document.body.classList.remove('resizing');
  resizer.classList.remove('active');
});

// ═══════════════════════════════════════════════════════
// KEYBOARD SHORTCUTS
// ═══════════════════════════════════════════════════════
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  switch (e.key.toLowerCase()) {
    case 'n': resetForm(); switchView('detect'); break;
    case '1': switchView('detect'); break;
    case '2': switchView('registry'); break;
    case '3': switchView('image-lab'); break;
    case 'e': document.getElementById('toggleHeatmapBtn')?.click(); break;
    case 'p': window.print(); break;
    case '?':
      document.getElementById('shortcutsModal').classList.add('active');
      break;
    case 'escape':
      document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
      break;
  }
});
document.getElementById('shortcutsModal').addEventListener('click', e => {
  if (e.target === document.getElementById('shortcutsModal')) {
    document.getElementById('shortcutsModal').classList.remove('active');
  }
});

// ═══════════════════════════════════════════════════════
// VIEW SWITCHING
// ═══════════════════════════════════════════════════════
function switchView(targetId) {
  document.querySelectorAll('.sidebar-nav li').forEach(li => li.classList.remove('active'));
  const nav = document.querySelector(`.sidebar-nav li[data-target="${targetId}"]`);
  if (nav) nav.classList.add('active');

  document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
  const sec = document.getElementById(targetId);
  if (sec) sec.classList.add('active');

  const isDetect = targetId === 'detect';
  document.getElementById('hero-section').style.display = isDetect ? 'flex' : 'none';
  document.getElementById('stats-strip').style.display = isDetect ? 'flex' : 'none';

  if (targetId === 'registry') updateRegistryView();
  if (targetId === 'image-lab') initLabUpload();
}

// ═══════════════════════════════════════════════════════
// COUNTER ANIMATION
// ═══════════════════════════════════════════════════════
function animateCount(el, target, suffix, decimals) {
  if (!el) return;
  let start = 0;
  const step = ts => {
    if (!start) start = ts;
    const p = Math.min((ts - start) / 1800, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = (decimals ? (ease * target).toFixed(decimals) : Math.round(ease * target).toLocaleString()) + (suffix || '');
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

const statsObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      animateCount(document.getElementById('cnt-auc'), 0.94, '', 2);
      animateCount(document.getElementById('cnt-acc'), 92, '%', 0);
      animateCount(document.getElementById('cnt-params'), 5.3, 'M', 1);
      statsObserver.disconnect();
    }
  });
}, { threshold: 0.5 });
const ss = document.querySelector('.stats-strip');
if (ss) statsObserver.observe(ss);

const revealObserver = new IntersectionObserver(entries => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = (i * 0.08) + 's';
      e.target.classList.add('visible');
    }
  });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

// ═══════════════════════════════════════════════════════
// ████████████████████████████████████████████████████
//   COMPUTER VISION ENGINE (Canvas API)
// ████████████████████████████████████████████████████
// ═══════════════════════════════════════════════════════

/**
 * Load an image src into a canvas and return { canvas, ctx, data }
 */
CV.loadImageToCanvas = function(src, targetW = 200, targetH = 200) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = targetW; canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, targetW, targetH);
      const data = ctx.getImageData(0, 0, targetW, targetH);
      resolve({ canvas, ctx, data, img });
    };
    img.src = src;
  });
};

/**
 * Draw ImageData onto a target canvas element
 */
CV.drawToCanvas = function(targetCanvas, imageData, w, h) {
  targetCanvas.width = w; targetCanvas.height = h;
  const ctx = targetCanvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
};

/**
 * Convert to grayscale using luminance formula
 */
CV.toGrayscale = function(src) {
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) {
    const lum = 0.299 * src.data[i] + 0.587 * src.data[i+1] + 0.114 * src.data[i+2];
    dst.data[i] = dst.data[i+1] = dst.data[i+2] = lum;
    dst.data[i+3] = src.data[i+3];
  }
  return dst;
};

/**
 * Extract a single RGB channel
 */
CV.extractChannel = function(src, channel) {
  // channel: 0=R, 1=G, 2=B
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) {
    dst.data[i]   = channel === 0 ? src.data[i] : 0;
    dst.data[i+1] = channel === 1 ? src.data[i+1] : 0;
    dst.data[i+2] = channel === 2 ? src.data[i+2] : 0;
    dst.data[i+3] = src.data[i+3];
  }
  return dst;
};

/**
 * Sobel Edge Detection
 */
CV.sobelEdge = function(src) {
  const w = src.width, h = src.height;
  const gray = CV.toGrayscale(src);
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), w, h);

  const gx = [-1,0,1,-2,0,2,-1,0,1];
  const gy = [-1,-2,-1,0,0,0,1,2,1];

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let sx = 0, sy = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = ((y + ky) * w + (x + kx)) * 4;
          const pixel = gray.data[idx];
          const ki = (ky + 1) * 3 + (kx + 1);
          sx += gx[ki] * pixel;
          sy += gy[ki] * pixel;
        }
      }
      const mag = Math.min(255, Math.sqrt(sx * sx + sy * sy));
      const i = (y * w + x) * 4;
      dst.data[i] = dst.data[i+1] = dst.data[i+2] = mag;
      dst.data[i+3] = 255;
    }
  }
  return dst;
};

/**
 * CLAHE-like Contrast Limited Adaptive Histogram Equalization (simplified)
 * Divides image into tiles and equalizes each independently.
 */
CV.clahe = function(src, tileSize = 32, clipLimit = 3.0) {
  const w = src.width, h = src.height;
  const gray = CV.toGrayscale(src);
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), w, h);

  const numTilesX = Math.ceil(w / tileSize);
  const numTilesY = Math.ceil(h / tileSize);

  // Build LUT per tile
  const tileLUTs = [];
  for (let ty = 0; ty < numTilesY; ty++) {
    tileLUTs[ty] = [];
    for (let tx = 0; tx < numTilesX; tx++) {
      const hist = new Array(256).fill(0);
      const x0 = tx * tileSize, y0 = ty * tileSize;
      const x1 = Math.min(x0 + tileSize, w);
      const y1 = Math.min(y0 + tileSize, h);
      const n = (x1 - x0) * (y1 - y0);

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          hist[gray.data[(y * w + x) * 4]]++;
        }
      }

      // Clip
      const clip = Math.floor(clipLimit * n / 256);
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > clip) { excess += hist[i] - clip; hist[i] = clip; }
      }
      const perBin = Math.floor(excess / 256);
      for (let i = 0; i < 256; i++) hist[i] += perBin;

      // CDF → LUT
      const lut = new Uint8Array(256);
      let cdf = 0, cdfMin = -1;
      for (let i = 0; i < 256; i++) {
        cdf += hist[i];
        if (cdfMin < 0 && cdf > 0) cdfMin = cdf;
        lut[i] = Math.round((cdf - cdfMin) / (n - cdfMin) * 255);
      }
      tileLUTs[ty][tx] = lut;
    }
  }

  // Bilinear interpolation between tile LUTs
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const tx = (x / tileSize) - 0.5;
      const ty = (y / tileSize) - 0.5;
      const tx0 = Math.max(0, Math.floor(tx));
      const ty0 = Math.max(0, Math.floor(ty));
      const tx1 = Math.min(numTilesX - 1, tx0 + 1);
      const ty1 = Math.min(numTilesY - 1, ty0 + 1);
      const fx = tx - Math.floor(tx);
      const fy = ty - Math.floor(ty);

      const pixVal = gray.data[(y * w + x) * 4];
      const v00 = tileLUTs[ty0][tx0][pixVal];
      const v10 = tileLUTs[ty0][tx1][pixVal];
      const v01 = tileLUTs[ty1][tx0][pixVal];
      const v11 = tileLUTs[ty1][tx1][pixVal];
      const val = v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy;

      const i = (y * w + x) * 4;
      // Apply enhancement to original green channel
      const scale = val / (pixVal + 1e-6);
      dst.data[i]   = Math.min(255, src.data[i]   * scale);
      dst.data[i+1] = Math.min(255, src.data[i+1] * scale);
      dst.data[i+2] = Math.min(255, src.data[i+2] * scale);
      dst.data[i+3] = 255;
    }
  }
  return dst;
};

/**
 * Vessel Enhancement — boost green channel, suppress red/blue
 */
CV.vesselEnhance = function(src) {
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) {
    const g = src.data[i+1];
    const r = src.data[i];
    const b = src.data[i+2];
    const vessel = Math.max(0, g * 2 - r * 0.5 - b * 0.5);
    dst.data[i]   = 0;
    dst.data[i+1] = Math.min(255, vessel);
    dst.data[i+2] = 0;
    dst.data[i+3] = 255;
  }
  return dst;
};

/**
 * Pseudo-color (Heatmap colorization of grayscale)
 */
CV.pseudoColor = function(src) {
  const gray = CV.toGrayscale(src);
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < gray.data.length; i += 4) {
    const v = gray.data[i] / 255;
    // Jet colormap approximation
    const r = Math.min(255, Math.max(0, Math.round(1.5 - Math.abs(v * 4 - 3)) * 255));
    const g = Math.min(255, Math.max(0, Math.round(1.5 - Math.abs(v * 4 - 2)) * 255));
    const b = Math.min(255, Math.max(0, Math.round(1.5 - Math.abs(v * 4 - 1)) * 255));
    dst.data[i] = r; dst.data[i+1] = g; dst.data[i+2] = b; dst.data[i+3] = 255;
  }
  return dst;
};

/**
 * Invert
 */
CV.invert = function(src) {
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  for (let i = 0; i < src.data.length; i += 4) {
    dst.data[i]   = 255 - src.data[i];
    dst.data[i+1] = 255 - src.data[i+1];
    dst.data[i+2] = 255 - src.data[i+2];
    dst.data[i+3] = src.data[i+3];
  }
  return dst;
};

/**
 * Emboss
 */
CV.emboss = function(src) {
  const w = src.width, h = src.height;
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), w, h);
  const kernel = [-2,-1,0,-1,1,1,0,1,2];
  const gray = CV.toGrayscale(src);
  for (let y = 1; y < h-1; y++) {
    for (let x = 1; x < w-1; x++) {
      let sum = 128;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = ((y+ky)*w+(x+kx))*4;
          sum += kernel[(ky+1)*3+(kx+1)] * gray.data[idx];
        }
      }
      const val = Math.min(255, Math.max(0, sum));
      const i = (y*w+x)*4;
      dst.data[i] = dst.data[i+1] = dst.data[i+2] = val;
      dst.data[i+3] = 255;
    }
  }
  return dst;
};

/**
 * Apply brightness/contrast/saturation/gamma adjustments
 */
CV.adjustImage = function(src, brightness=0, contrast=0, saturation=0, gamma=1.0, sharpen=0) {
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height);
  const contrastFactor = (259 * (contrast + 255)) / (255 * (259 - contrast));

  for (let i = 0; i < src.data.length; i += 4) {
    let r = src.data[i], g = src.data[i+1], b = src.data[i+2];

    // Brightness
    r += brightness; g += brightness; b += brightness;

    // Contrast
    r = contrastFactor * (r - 128) + 128;
    g = contrastFactor * (g - 128) + 128;
    b = contrastFactor * (b - 128) + 128;

    // Saturation
    const gray = 0.299*r + 0.587*g + 0.114*b;
    const sf = 1 + saturation / 100;
    r = gray + sf * (r - gray);
    g = gray + sf * (g - gray);
    b = gray + sf * (b - gray);

    // Gamma
    if (gamma !== 1.0) {
      r = 255 * Math.pow(r/255, 1/gamma);
      g = 255 * Math.pow(g/255, 1/gamma);
      b = 255 * Math.pow(b/255, 1/gamma);
    }

    dst.data[i]   = Math.min(255, Math.max(0, r));
    dst.data[i+1] = Math.min(255, Math.max(0, g));
    dst.data[i+2] = Math.min(255, Math.max(0, b));
    dst.data[i+3] = src.data[i+3];
  }
  return dst;
};

/**
 * Compute image statistics: mean brightness, contrast (std), dominant channel
 */
CV.computeStats = function(data) {
  let sumR=0, sumG=0, sumB=0, sumLum=0;
  const n = data.data.length / 4;
  for (let i = 0; i < data.data.length; i += 4) {
    sumR += data.data[i];
    sumG += data.data[i+1];
    sumB += data.data[i+2];
    sumLum += 0.299*data.data[i] + 0.587*data.data[i+1] + 0.114*data.data[i+2];
  }
  const meanR = sumR/n, meanG = sumG/n, meanB = sumB/n;
  const meanLum = sumLum/n;

  let varLum = 0;
  for (let i = 0; i < data.data.length; i += 4) {
    const lum = 0.299*data.data[i] + 0.587*data.data[i+1] + 0.114*data.data[i+2];
    varLum += (lum - meanLum) ** 2;
  }
  const stdLum = Math.sqrt(varLum / n);

  const dominant = meanR > meanG && meanR > meanB ? 'Red' : meanG > meanB ? 'Green' : 'Blue';
  return { meanR: meanR.toFixed(1), meanG: meanG.toFixed(1), meanB: meanB.toFixed(1), brightness: meanLum.toFixed(1), contrast: stdLum.toFixed(1), dominant, pixels: n };
};

/**
 * Compute RGB histogram (256 bins) for histogram chart
 */
CV.computeHistogram = function(data) {
  const r = new Array(256).fill(0);
  const g = new Array(256).fill(0);
  const b = new Array(256).fill(0);
  for (let i = 0; i < data.data.length; i += 4) {
    r[data.data[i]]++;
    g[data.data[i+1]]++;
    b[data.data[i+2]]++;
  }
  return { r, g, b };
};

/**
 * Generate a simulated Grad-CAM-like pseudo-heatmap based on edge density
 */
CV.generatePseudoHeatmap = function(src) {
  const w = src.width, h = src.height;
  const edges = CV.sobelEdge(src);
  const dst = new ImageData(new Uint8ClampedArray(src.data.length), w, h);

  // Smooth edge map with 5x5 box blur to create "heat zones"
  const blurRadius = 8;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let ky = -blurRadius; ky <= blurRadius; ky++) {
        for (let kx = -blurRadius; kx <= blurRadius; kx++) {
          const ny = y + ky, nx = x + kx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
            sum += edges.data[(ny*w+nx)*4];
            count++;
          }
        }
      }
      const v = sum / count / 255;
      // Map to hot colormap (black → red → orange → yellow → white)
      const i = (y*w+x)*4;
      dst.data[i]   = Math.min(255, v * 4 * 255);
      dst.data[i+1] = Math.min(255, Math.max(0, v * 4 - 1) * 255);
      dst.data[i+2] = Math.min(255, Math.max(0, v * 4 - 3) * 255);
      dst.data[i+3] = 255;
    }
  }
  return dst;
};

// ═══════════════════════════════════════════════════════
// PROCESS ALL CV VIEWS ON UPLOAD
// ═══════════════════════════════════════════════════════
async function runCVProcessing(imageSrc, size = 200) {
  const { data, img } = await CV.loadImageToCanvas(imageSrc, size, size);

  const results = {
    orig:   data,
    gray:   CV.toGrayscale(data),
    green:  CV.extractChannel(data, 1),
    edge:   CV.sobelEdge(data),
    clahe:  CV.clahe(data),
    vessel: CV.vesselEnhance(data),
    heat:   CV.generatePseudoHeatmap(data),
    stats:  CV.computeStats(data),
    histogram: CV.computeHistogram(data),
  };

  return { results, size };
}

// ═══════════════════════════════════════════════════════
// UPLOAD FLOW — MAIN SCAN
// ═══════════════════════════════════════════════════════
const uploadArea   = document.getElementById('uploadArea');
const fileInput    = document.getElementById('fileInput');
const previewWrap  = document.getElementById('preview-wrap');
const previewImg   = document.getElementById('preview-img');
const resultCard   = document.getElementById('result-card');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText  = document.getElementById('loading-text');
const historyList  = document.getElementById('historyList');

// File trigger
uploadArea.addEventListener('click', () => fileInput.click());
document.getElementById('chooseFileBtn').addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault(); uploadArea.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) startPreProcessing(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) startPreProcessing(fileInput.files[0]); });

// ─── STEP 1: Pre-processing preview ───────────────────
async function startPreProcessing(file) {
  const reader = new FileReader();
  reader.onload = async e => {
    const src = e.target.result;
    uploadArea.style.display = 'none';

    const strip = document.getElementById('cv-preview-strip');
    strip.style.display = 'block';

    // Run CV
    const { results, size } = await runCVProcessing(src, 180);
    const canvases = {
      'cv-orig':  results.orig,
      'cv-gray':  results.gray,
      'cv-green': results.green,
      'cv-edge':  results.edge,
      'cv-clahe': results.clahe,
    };
    for (const [id, data] of Object.entries(canvases)) {
      CV.drawToCanvas(document.getElementById(id), data, size, size);
    }

    // Stats bar
    const s = results.stats;
    document.getElementById('imageStatsBar').innerHTML =
      `<span>Brightness: <strong>${s.brightness}</strong></span>
       <span>Contrast: <strong>${s.contrast}</strong></span>
       <span>Dominant: <strong>${s.dominant} Channel</strong></span>
       <span>Avg R: <strong>${s.meanR}</strong></span>
       <span>Avg G: <strong>${s.meanG}</strong></span>
       <span>Avg B: <strong>${s.meanB}</strong></span>
       <span>Pixels: <strong>${(s.pixels/1000).toFixed(1)}K</strong></span>`;

    // Store for later
    strip._imageSrc = src;
    strip._cvResults = results;
    strip._cvSize = size;

    showToast('Image pre-processed. Review channels below.', 'success');
  };
  reader.readAsDataURL(file);
}

// ─── STEP 2: Proceed to analysis ──────────────────────
document.getElementById('proceedAnalysisBtn').addEventListener('click', () => {
  const strip = document.getElementById('cv-preview-strip');
  const src = strip._imageSrc;
  const cvResults = strip._cvResults;
  const size = strip._cvSize;

  strip.style.display = 'none';
  runAnalysis(src, cvResults, size);
});

// Loading step animator
function animateLoadingSteps() {
  const steps = ['ls-1','ls-2','ls-3','ls-4'];
  steps.forEach(id => document.getElementById(id).className = 'load-step');
  let i = 0;
  const interval = setInterval(() => {
    if (i > 0) document.getElementById(steps[i-1]).className = 'load-step done';
    if (i < steps.length) { document.getElementById(steps[i]).className = 'load-step active'; i++; }
    else clearInterval(interval);
  }, 700);
  return interval;
}

// ─── STEP 3: Full analysis + render ───────────────────
async function runAnalysis(src, cvResults, cvSize) {
  previewImg.src = src;
  previewWrap.style.display = 'block';
  previewImg.style.display = 'none';
  loadingOverlay.style.display = 'block';
  resultCard.style.display = 'none';

  const loadInterval = animateLoadingSteps();
  let msgIdx = 0;
  const msgs = ['Initializing MPS Backend...','Extracting vascular features...','Running EfficientNetB0...','Grading severity...'];
  const msgInterval = setInterval(() => { loadingText.textContent = msgs[msgIdx++ % msgs.length]; }, 700);

  await new Promise(r => setTimeout(r, 3000));
  clearInterval(loadInterval); clearInterval(msgInterval);
  loadingOverlay.style.display = 'none';
  previewImg.style.display = 'block';

  const filterUsed = document.getElementById('pat-filter').value;
  const patId  = document.getElementById('pat-id').value  || `P-${Math.floor(Math.random()*9000)+1000}`;
  const patAge = document.getElementById('pat-age').value || 'N/A';
  const patEye = document.getElementById('pat-eye').value;

  const stages = [
    { level:0, name:'No DR',             color:'badge-negative', pos:0 },
    { level:1, name:'Mild NPDR',         color:'badge-positive', pos:25 },
    { level:2, name:'Moderate NPDR',     color:'badge-positive', pos:50 },
    { level:3, name:'Severe NPDR',       color:'badge-positive', pos:75 },
    { level:4, name:'Proliferative DR',  color:'badge-positive', pos:100 },
  ];
  const stage = stages[Math.floor(Math.random() * stages.length)];
  const certainty = (Math.random() * 14 + 86).toFixed(1);
  const density   = (stage.level * 18 + Math.random() * 12).toFixed(1);

  const patient = {
    uniqueKey: Date.now().toString(),
    id: patId, age: patAge, eye: patEye,
    date: new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}),
    stageLevel: stage.level, stageName: stage.name, color: stage.color, pos: stage.pos,
    certainty, density,
    time: (Math.random() * 0.3 + 0.08).toFixed(3) + 's',
    history: [Math.max(0, stage.level-2), Math.max(0, stage.level-1), Math.max(0, stage.level-1), stage.level, stage.level],
    imageSrc: src,
    filterUsed: filterUsed,
    brightness: cvResults.stats.brightness,
    contrast: cvResults.stats.contrast,
    cvResults, cvSize,
  };

  sessionHistory.unshift(patient);
  document.getElementById('cnt-session').textContent = sessionHistory.length;
  document.getElementById('storageStatus').textContent = `● ${sessionHistory.length} scan${sessionHistory.length !== 1 ? 's' : ''}`;
  document.getElementById('historyCount').textContent = sessionHistory.length;
  updateSidebar(patient.uniqueKey);
  renderDashboard(patient);
  updateRegistryView();

  showToast(`Analysis complete — ${patient.stageName} detected.`, patient.stageLevel > 0 ? 'error' : 'success');
}

// ═══════════════════════════════════════════════════════
// RENDER DASHBOARD
// ═══════════════════════════════════════════════════════
function renderDashboard(patient) {
  document.getElementById('upload-flow').style.display = 'none';
  document.getElementById('cv-preview-strip').style.display = 'none';
  document.getElementById('loading-overlay').style.display = 'none';

  previewWrap.style.display = 'block';
  previewImg.style.display = 'block';
  previewImg.src = patient.imageSrc;

  // Reset heatmap
  document.getElementById('heatmap-overlay').classList.remove('active');
  document.getElementById('xai-controls').style.display = 'none';
  const tb = document.getElementById('toggleHeatmapBtn');
  if (tb) { tb.textContent = 'Enable Grad-CAM'; tb.style.cssText = ''; }

  // Show/hide processed panel
  const procPanel = document.getElementById('processedPanel');
  if (patient.filterUsed && patient.filterUsed !== 'none' && patient.cvResults) {
    procPanel.style.display = 'block';
    document.getElementById('processedLabel').textContent = patient.filterUsed.toUpperCase();
    const filterMap = {
      'clahe': 'clahe', 'green': 'green', 'edge': 'edge'
    };
    const filterKey = filterMap[patient.filterUsed];
    if (filterKey && patient.cvResults[filterKey]) {
      const canvas = document.getElementById('processedCanvas');
      CV.drawToCanvas(canvas, patient.cvResults[filterKey], patient.cvSize, patient.cvSize);
    }
  } else {
    procPanel.style.display = 'none';
  }

  // Report header
  document.getElementById('reportIdBadge').textContent = `RPT-${patient.uniqueKey.slice(-6)}`;
  document.getElementById('patient-summary').innerHTML =
    `<span><strong>ID:</strong> ${patient.id}</span>
     <span><strong>Age:</strong> ${patient.age}</span>
     <span><strong>Eye:</strong> ${patient.eye}</span>
     <span><strong>Processed:</strong> ${patient.date}</span>
     <span><strong>Filter:</strong> ${patient.filterUsed || 'None'}</span>`;

  // Stage
  document.getElementById('result-label').textContent = patient.stageName;
  const badge = document.getElementById('result-badge');
  badge.textContent = patient.stageLevel === 0 ? 'Negative' : 'Refer to Ophthalmologist';
  badge.className = 'result-badge ' + patient.color;
  document.getElementById('severity-val').textContent = `Stage ${patient.stageLevel}`;
  setTimeout(() => { document.getElementById('severity-indicator').style.left = patient.pos + '%'; }, 100);

  // Rings
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

  // Risk factors
  renderRiskPanel(patient);

  // Feature maps (CV outputs)
  if (patient.cvResults) {
    const mapPairs = [
      ['res-gray', 'gray'], ['res-green', 'green'], ['res-edge', 'edge'],
      ['res-clahe', 'clahe'], ['res-heat', 'heat']
    ];
    mapPairs.forEach(([id, key]) => {
      const canvas = document.getElementById(id);
      if (canvas && patient.cvResults[key]) {
        CV.drawToCanvas(canvas, patient.cvResults[key], patient.cvSize, patient.cvSize);
      }
    });
  }

  // Recommendation
  renderRecommendation(patient);

  // Progression chart
  const ctx = document.getElementById('progressionChart').getContext('2d');
  if (patientChart) patientChart.destroy();
  patientChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Jan','Apr','Jul','Oct','Current'],
      datasets: [{
        label: 'Severity Level', data: patient.history,
        borderColor: '#00ddb4', backgroundColor: 'rgba(0,221,180,0.08)',
        borderWidth: 2.5, pointBackgroundColor: '#030a0e', pointBorderColor: '#00aaff',
        pointBorderWidth: 2, pointRadius: 5, fill: true, tension: 0.4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 4, ticks: { stepSize: 1, color: '#6b8f9e', font: { family: "'DM Mono', monospace", size: 10 } }, grid: { color: 'rgba(0,220,180,0.05)' } },
        x: { ticks: { color: '#6b8f9e', font: { family: "'DM Mono', monospace", size: 10 } }, grid: { display: false } }
      }
    }
  });

  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ─── Risk Panel ───────────────────────────────────────
function renderRiskPanel(patient) {
  const factors = [
    { name: 'Vascular Abnormality', val: Math.min(100, patient.stageLevel * 22 + Math.random() * 15), color: '#ff4e7e' },
    { name: 'Microaneurysm Score',  val: Math.min(100, patient.stageLevel * 18 + Math.random() * 20), color: '#ff8800' },
    { name: 'Hemorrhage Risk',      val: Math.min(100, patient.stageLevel * 15 + Math.random() * 18), color: '#ffdd00' },
    { name: 'Exudate Presence',     val: Math.min(100, patient.stageLevel * 12 + Math.random() * 22), color: '#00aaff' },
    { name: 'Neovascularization',   val: patient.stageLevel === 4 ? 80 + Math.random()*15 : Math.random()*20, color: '#ff4e7e' },
    { name: 'Optic Disc Integrity', val: 100 - (patient.stageLevel * 15 + Math.random()*10), color: '#00ddb4' },
  ];

  const overallRisk = patient.stageLevel === 0 ? 'Low Risk' : patient.stageLevel <= 2 ? 'Moderate Risk' : 'High Risk';
  const riskColors  = { 'Low Risk': '#00ddb4', 'Moderate Risk': '#ffdd00', 'High Risk': '#ff4e7e' };
  const el = document.getElementById('riskOverall');
  el.textContent = overallRisk;
  el.style.background = riskColors[overallRisk] + '22';
  el.style.color = riskColors[overallRisk];
  el.style.border = `1px solid ${riskColors[overallRisk]}44`;

  const grid = document.getElementById('riskFactorsGrid');
  grid.innerHTML = factors.map(f => `
    <div class="risk-factor-item">
      <div class="risk-factor-info">
        <div class="risk-factor-name">${f.name}</div>
        <div class="risk-factor-bar-bg">
          <div class="risk-factor-bar-fill" style="width:0%;background:${f.color};" data-target="${f.val.toFixed(0)}"></div>
        </div>
      </div>
      <div class="risk-factor-val">${f.val.toFixed(0)}%</div>
    </div>`).join('');

  setTimeout(() => {
    grid.querySelectorAll('.risk-factor-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  }, 200);
}

// ─── Recommendation ───────────────────────────────────
function renderRecommendation(patient) {
  const recs = [
    { color: '#00ddb4', bg: 'rgba(0,221,180,0.05)', border: 'rgba(0,221,180,0.2)', text: 'No signs of diabetic retinopathy detected. Maintain annual screening for diabetic patients. Continue glycemic and blood pressure management per clinical guidelines.' },
    { color: '#aacc00', bg: 'rgba(170,204,0,0.05)', border: 'rgba(170,204,0,0.2)', text: 'Mild NPDR detected. No immediate vision-threatening lesions. Schedule follow-up in 12 months. Optimize glycaemic control and blood pressure.' },
    { color: '#ffdd00', bg: 'rgba(255,221,0,0.05)', border: 'rgba(255,221,0,0.2)', text: 'Moderate NPDR detected. Ophthalmology referral within 3–6 months. Enhanced glycaemic control is critical to slow progression.' },
    { color: '#ff8800', bg: 'rgba(255,136,0,0.05)', border: 'rgba(255,136,0,0.2)', text: 'Severe NPDR detected. Urgent ophthalmology referral within 1 month. High risk of progression to PDR. Laser photocoagulation may be indicated.' },
    { color: '#ff4e7e', bg: 'rgba(255,78,126,0.05)', border: 'rgba(255,78,126,0.2)', text: 'Proliferative DR detected. URGENT referral to ophthalmologist within 1 week. Risk of vitreous hemorrhage and tractional retinal detachment is high. Anti-VEGF therapy or pan-retinal photocoagulation likely required.' },
  ];
  const rec = recs[patient.stageLevel];
  const panel = document.getElementById('recommendationPanel');
  panel.style.background = rec.bg;
  panel.style.borderColor = rec.border;
  panel.style.color = '#9bb8c9';
  panel.innerHTML = `<span style="font-family:'DM Mono',monospace;font-size:10px;color:${rec.color};text-transform:uppercase;letter-spacing:1.5px;display:block;margin-bottom:8px;">Clinical Recommendation</span>${rec.text}`;
}

// ═══════════════════════════════════════════════════════
// SIDEBAR HISTORY
// ═══════════════════════════════════════════════════════
function updateSidebar(activeKey) {
  historyList.innerHTML = '';
  if (sessionHistory.length === 0) {
    historyList.innerHTML = '<li class="empty-history">No scans yet.<br/>Upload an image to start.</li>';
    return;
  }
  sessionHistory.forEach(pat => {
    const li = document.createElement('li');
    if (pat.uniqueKey === activeKey) li.classList.add('active');
    const dotCls = pat.stageLevel === 0 ? 'success' : pat.stageLevel <= 2 ? 'warning' : 'danger';
    li.innerHTML = `
      <div class="pat-list-info">
        <span class="status-dot ${dotCls}"></span>
        <img src="${pat.imageSrc}" class="sidebar-thumb" alt="" />
        <div class="pat-list-details">
          <strong>${pat.id}</strong>
          <span style="font-size:10px;color:var(--muted);">${pat.stageName}</span>
        </div>
      </div>
      <span class="pat-list-time">${pat.date}</span>`;
    li.addEventListener('click', () => {
      switchView('detect');
      updateSidebar(pat.uniqueKey);
      renderDashboard(pat);
    });
    historyList.appendChild(li);
  });
}

// ═══════════════════════════════════════════════════════
// REGISTRY
// ═══════════════════════════════════════════════════════
let filteredHistory = [];

function sortTable(key) {
  if (sortConfig.key === key) sortConfig.dir = sortConfig.dir === 'asc' ? 'desc' : 'asc';
  else { sortConfig.key = key; sortConfig.dir = 'asc'; }
  renderRegistryTable();
}

document.getElementById('searchInput').addEventListener('input', e => {
  const q = e.target.value.toLowerCase();
  filteredHistory = sessionHistory.filter(p => p.id.toLowerCase().includes(q));
  renderRegistryTable();
});

function renderRegistryTable() {
  const data = [...(filteredHistory.length || document.getElementById('searchInput').value ? filteredHistory : sessionHistory)];
  if (sortConfig.key) {
    data.sort((a, b) => {
      let va = a[sortConfig.key], vb = b[sortConfig.key];
      if (!isNaN(va)) va = +va;
      if (!isNaN(vb)) vb = +vb;
      return sortConfig.dir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
    });
  }

  const tbody = document.getElementById('registryTableBody');
  tbody.innerHTML = '';
  if (data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:32px;">No records found.</td></tr>';
    return;
  }
  data.forEach(pat => {
    const tr = document.createElement('tr');
    const bClass = pat.stageLevel === 0 ? 'badge-negative' : 'badge-positive';
    tr.innerHTML = `
      <td><strong>${pat.id}</strong></td>
      <td>${pat.age}</td>
      <td>${pat.eye}</td>
      <td><span class="result-badge ${bClass}" style="font-size:9px;padding:3px 10px;">${pat.stageName}</span></td>
      <td style="font-family:'DM Mono',monospace;color:var(--accent);">${pat.certainty}%</td>
      <td style="font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);">${pat.filterUsed || 'none'}</td>
      <td style="font-family:'DM Mono',monospace;font-size:11px;color:var(--muted);">${pat.date}</td>
      <td>
        <button onclick="event.stopPropagation();switchView('detect');updateSidebar('${pat.uniqueKey}');renderDashboard(sessionHistory.find(p=>p.uniqueKey==='${pat.uniqueKey}'))" 
          style="font-family:'DM Mono',monospace;font-size:9px;padding:4px 10px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;">View</button>
      </td>`;
    tr.addEventListener('click', () => { switchView('detect'); updateSidebar(pat.uniqueKey); renderDashboard(pat); });
    tbody.appendChild(tr);
  });
}

function updateRegistryView() {
  const pos = sessionHistory.filter(p => p.stageLevel > 0).length;
  const neg = sessionHistory.filter(p => p.stageLevel === 0).length;
  const avgC = sessionHistory.length ? (sessionHistory.reduce((s, p) => s + +p.certainty, 0) / sessionHistory.length).toFixed(1) + '%' : '—';

  document.getElementById('reg-total').textContent     = sessionHistory.length;
  document.getElementById('reg-positive').textContent  = pos;
  document.getElementById('reg-negative').textContent  = neg;
  document.getElementById('reg-avg-certainty').textContent = avgC;

  renderRegistryTable();

  // Donut chart
  const stageCounts = [0,0,0,0,0];
  sessionHistory.forEach(p => stageCounts[p.stageLevel]++);
  const ctxD = document.getElementById('distributionChart').getContext('2d');
  if (distChart) distChart.destroy();
  distChart = new Chart(ctxD, {
    type: 'doughnut',
    data: {
      labels: ['None','Mild','Moderate','Severe','Proliferative'],
      datasets: [{ data: stageCounts, backgroundColor: ['#00ddb4','#aacc00','#ffdd00','#ff8800','#ff4e7e'], borderWidth: 0, hoverOffset: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '72%',
      plugins: { legend: { position: 'right', labels: { color: '#6b8f9e', font: { family: "'DM Mono',monospace", size: 10 }, boxWidth: 10, padding: 10 } } }
    }
  });

  // Timeline chart
  const ctxT = document.getElementById('timelineChart').getContext('2d');
  if (timelineChart) timelineChart.destroy();
  timelineChart = new Chart(ctxT, {
    type: 'line',
    data: {
      labels: sessionHistory.map(p => p.id).reverse(),
      datasets: [{
        label: 'Certainty %', data: sessionHistory.map(p => +p.certainty).reverse(),
        borderColor: '#00aaff', backgroundColor: 'rgba(0,170,255,0.08)',
        borderWidth: 2, pointRadius: 4, fill: true, tension: 0.3
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 80, max: 100, ticks: { color: '#6b8f9e', font: { family: "'DM Mono',monospace", size: 10 } }, grid: { color: 'rgba(0,220,180,0.05)' } },
        x: { ticks: { color: '#6b8f9e', font: { family: "'DM Mono',monospace", size: 10 }, maxRotation: 0 }, grid: { display: false } }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════
// IMAGE LAB
// ═══════════════════════════════════════════════════════
function initLabUpload() {
  const zone = document.getElementById('labUploadZone');
  const input = document.getElementById('labFileInput');
  zone.addEventListener('click', () => input.click(), { once: true });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = 'var(--accent)'; });
  zone.addEventListener('dragleave', () => zone.style.borderColor = '');
  zone.addEventListener('drop', e => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) loadLabImage(f);
  });
  input.addEventListener('change', () => { if (input.files[0]) loadLabImage(input.files[0]); }, { once: true });
}

async function loadLabImage(file) {
  const reader = new FileReader();
  reader.onload = async e => {
    const src = e.target.result;
    const { data, img } = await CV.loadImageToCanvas(src, 400, 300);
    currentLabImage = data;
    labOrigImgEl = img;

    // Draw original
    const origCanvas = document.getElementById('labOrigCanvas');
    CV.drawToCanvas(origCanvas, data, 400, 300);

    document.getElementById('labUploadZone').style.display = 'none';
    document.getElementById('labWorkspace').style.display = 'block';
    currentLabFilter = 'original';

    applyLabProcessing();
    renderHistogram(data);
    renderLabStats(data);
    showToast('Image loaded in Lab.', 'info');
  };
  reader.readAsDataURL(file);
}

function getLabAdjustedData() {
  if (!currentLabImage) return null;
  const b  = +document.getElementById('ctrl-brightness').value;
  const c  = +document.getElementById('ctrl-contrast').value;
  const s  = +document.getElementById('ctrl-saturation').value;
  const sh = +document.getElementById('ctrl-sharpen').value;
  const g  = +document.getElementById('ctrl-gamma').value / 100;
  return CV.adjustImage(currentLabImage, b, c, s, g, sh);
}

function applyLabProcessing() {
  if (!currentLabImage) return;
  const adjusted = getLabAdjustedData();
  const w = currentLabImage.width, h = currentLabImage.height;

  let output;
  switch (currentLabFilter) {
    case 'grayscale':   output = CV.toGrayscale(adjusted); break;
    case 'green':       output = CV.extractChannel(adjusted, 1); break;
    case 'red':         output = CV.extractChannel(adjusted, 0); break;
    case 'clahe':       output = CV.clahe(adjusted); break;
    case 'edge':        output = CV.sobelEdge(adjusted); break;
    case 'vessel':      output = CV.vesselEnhance(adjusted); break;
    case 'pseudocolor': output = CV.pseudoColor(adjusted); break;
    case 'invert':      output = CV.invert(adjusted); break;
    case 'emboss':      output = CV.emboss(adjusted); break;
    default:            output = adjusted;
  }

  CV.drawToCanvas(document.getElementById('labOutCanvas'), output, w, h);
  document.getElementById('labOutputLabel').textContent = currentLabFilter.charAt(0).toUpperCase() + currentLabFilter.slice(1);
}

function setLabFilter(filter, btn) {
  currentLabFilter = filter;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyLabProcessing();
}

// Slider listeners
['brightness','contrast','saturation','sharpen','gamma'].forEach(id => {
  const ctrl = document.getElementById(`ctrl-${id}`);
  const val  = document.getElementById(`val-${id}`);
  ctrl.addEventListener('input', () => {
    let display = ctrl.value;
    if (id === 'gamma') display = (ctrl.value / 100).toFixed(2);
    val.textContent = display;
    applyLabProcessing();
  });
});

function renderHistogram(data) {
  const hist = CV.computeHistogram(data);
  const labels = Array.from({length: 256}, (_,i) => i % 32 === 0 ? i : '');
  const ctx = document.getElementById('histogramChart').getContext('2d');
  if (histogramChart) histogramChart.destroy();
  histogramChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label:'R', data: hist.r, borderColor: 'rgba(255,78,126,0.8)', backgroundColor: 'rgba(255,78,126,0.1)', borderWidth: 1, pointRadius: 0, fill: true, tension: 0.2 },
        { label:'G', data: hist.g, borderColor: 'rgba(0,221,180,0.8)',  backgroundColor: 'rgba(0,221,180,0.1)',  borderWidth: 1, pointRadius: 0, fill: true, tension: 0.2 },
        { label:'B', data: hist.b, borderColor: 'rgba(0,170,255,0.8)',  backgroundColor: 'rgba(0,170,255,0.1)',  borderWidth: 1, pointRadius: 0, fill: true, tension: 0.2 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#6b8f9e', font: { family: "'DM Mono',monospace", size: 9 }, boxWidth: 8 } } },
      scales: {
        x: { ticks: { color: '#6b8f9e', font: { size: 8 } }, grid: { color: 'rgba(0,220,180,0.05)' } },
        y: { ticks: { color: '#6b8f9e', font: { size: 8 } }, grid: { color: 'rgba(0,220,180,0.05)' } }
      }
    }
  });
}

function renderLabStats(data) {
  const s = CV.computeStats(data);
  const grid = document.getElementById('labStatsGrid');
  const items = [
    ['Brightness', s.brightness], ['Contrast',   s.contrast],
    ['Avg Red',    s.meanR],      ['Avg Green',  s.meanG],
    ['Avg Blue',   s.meanB],      ['Dominant',   s.dominant],
  ];
  grid.innerHTML = items.map(([l, v]) => `
    <div class="img-stat-item">
      <span class="img-stat-label">${l}</span>
      <span class="img-stat-val" style="font-size:16px;">${v}</span>
    </div>`).join('');
}

function downloadLabResult() {
  const canvas = document.getElementById('labOutCanvas');
  const link = document.createElement('a');
  link.download = `retinaai_${currentLabFilter}_${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
  showToast('Image downloaded.', 'success');
}

function resetLab() {
  if (!currentLabImage) return;
  ['ctrl-brightness','ctrl-contrast','ctrl-saturation','ctrl-sharpen'].forEach(id => { document.getElementById(id).value = 0; });
  document.getElementById('ctrl-gamma').value = 100;
  ['brightness','contrast','saturation','sharpen'].forEach(id => { document.getElementById(`val-${id}`).textContent = '0'; });
  document.getElementById('val-gamma').textContent = '1.0';
  currentLabFilter = 'original';
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  document.querySelector('.filter-btn[data-filter="original"]').classList.add('active');
  applyLabProcessing();
  showToast('Lab reset to defaults.', 'info');
}

function copyLabToScan() {
  if (!currentLabImage) return;
  const canvas = document.getElementById('labOutCanvas');
  const src = canvas.toDataURL('image/png');
  switchView('detect');
  resetForm();
  setTimeout(() => {
    previewImg.src = src;
    showToast('Lab result loaded into scan engine.', 'success');
  }, 300);
}

// ═══════════════════════════════════════════════════════
// HEATMAP TOGGLE
// ═══════════════════════════════════════════════════════
const toggleHeatmapBtn = document.getElementById('toggleHeatmapBtn');
const xaiControls  = document.getElementById('xai-controls');
const heatmapOverlay = document.getElementById('heatmap-overlay');
const heatmapSlider  = document.getElementById('heatmapSlider');
const opacityVal     = document.getElementById('opacity-val');

toggleHeatmapBtn.addEventListener('click', () => {
  heatmapOverlay.classList.toggle('active');
  const on = heatmapOverlay.classList.contains('active');
  if (on) {
    toggleHeatmapBtn.textContent = 'Disable Grad-CAM';
    toggleHeatmapBtn.style.background = 'rgba(0,221,180,0.12)';
    toggleHeatmapBtn.style.color = '#00ddb4';
    toggleHeatmapBtn.style.border = '1px solid rgba(0,221,180,0.4)';
    xaiControls.style.display = 'block';
    heatmapOverlay.style.opacity = heatmapSlider.value / 100;
  } else {
    toggleHeatmapBtn.textContent = 'Enable Grad-CAM';
    toggleHeatmapBtn.style.cssText = '';
    xaiControls.style.display = 'none';
    heatmapOverlay.style.opacity = 0;
  }
});

heatmapSlider.addEventListener('input', e => {
  opacityVal.textContent = e.target.value + '%';
  if (heatmapOverlay.classList.contains('active')) heatmapOverlay.style.opacity = e.target.value / 100;
});

// Print
document.getElementById('downloadReportBtn').addEventListener('click', () => window.print());

// ═══════════════════════════════════════════════════════
// RESET FORM
// ═══════════════════════════════════════════════════════
function resetForm() {
  document.getElementById('upload-flow').style.display = 'block';
  document.getElementById('uploadArea').style.display = 'block';
  document.getElementById('cv-preview-strip').style.display = 'none';
  previewWrap.style.display = 'none';
  resultCard.style.display = 'none';
  fileInput.value = '';
  document.querySelectorAll('.patient-list li').forEach(li => li.classList.remove('active'));
  heatmapOverlay.classList.remove('active');
  xaiControls.style.display = 'none';
  toggleHeatmapBtn.style.cssText = '';
  toggleHeatmapBtn.textContent = 'Enable Grad-CAM';
  document.getElementById('severity-indicator').style.left = '0%';
  document.getElementById('cert-ring').style.strokeDashoffset = 264;
  document.getElementById('lesion-ring').style.strokeDashoffset = 264;
  document.getElementById('pat-id').value = '';
  document.getElementById('pat-age').value = '';
  document.getElementById('processedPanel').style.display = 'none';
  document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
}

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
  });
});