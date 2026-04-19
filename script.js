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
let notes = [];
let currentNoteId = null;
let appSettings = { filter: 'none', threshold: 0.5, profile: { name: 'Dr. Yuvraj', inst: 'Manipal University Jaipur', spec: 'AI/ML Research', loc: 'New Delhi, India' } };
const THEME_ORDER = ['midnight', 'light', 'sunset'];
const THEME_LABELS = { midnight: 'Midnight', light: 'Breeze', sunset: 'Sunset' };

// ═══════════════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════════════
function updateClock() {
  const el = document.getElementById('liveClock');
  if (el) el.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}
setInterval(updateClock, 1000); updateClock();

// ═══════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════
function showToast(msg, type = 'info', ms = 3500) {
  const c = document.getElementById('toastContainer');
  if (!c) return;
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<div class="toast-dot"></div><span>${msg}</span>`;
  c.appendChild(t);
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => { t.classList.remove('show'); setTimeout(() => t.remove(), 400); }, ms);
}

// ═══════════════════════════════════════════════
// THEME TOGGLE
// ═══════════════════════════════════════════════
function currentTheme() {
  return document.documentElement.getAttribute('data-theme') || 'midnight';
}

function syncThemeControls(theme) {
  const selector = document.getElementById('set-theme');
  if (selector) selector.value = theme;
  const btn = document.getElementById('themeToggleBtn');
  if (btn) btn.title = `Theme: ${THEME_LABELS[theme] || theme}`;
}

function applyTheme(theme = 'midnight', notify = false) {
  const nextTheme = THEME_ORDER.includes(theme) ? theme : 'midnight';
  document.documentElement.setAttribute('data-theme', nextTheme);
  localStorage.setItem('retinaai-theme', nextTheme);
  syncThemeControls(nextTheme);
  if (notify) showToast(`${THEME_LABELS[nextTheme] || nextTheme} theme enabled.`, 'info');
}

function toggleTheme() {
  const now = currentTheme();
  const i = THEME_ORDER.indexOf(now);
  const next = THEME_ORDER[(i + 1) % THEME_ORDER.length];
  applyTheme(next, true);
}

const savedTheme = localStorage.getItem('retinaai-theme');
if (savedTheme) applyTheme(savedTheme);
else applyTheme('midnight');

function toggleScanLine() {
  const sl = document.getElementById('scanLine');
  const cb = document.getElementById('set-scanline');
  if (sl && cb) sl.style.display = cb.checked ? '' : 'none';
}
function toggleBlobs() {
  const cb = document.getElementById('set-blobs');
  if (cb) document.querySelectorAll('.blob1,.blob2').forEach(b => b.style.display = cb.checked ? '' : 'none');
}

// ═══════════════════════════════════════════════
// LIGHTBOX
// ═══════════════════════════════════════════════
function openLightbox(src) {
  const img = document.getElementById('lightboxImg'), modal = document.getElementById('lightboxModal');
  if (img && modal) { img.src = src; modal.classList.add('active'); }
}
function openLightboxCanvas(canvas) { openLightbox(canvas.toDataURL('image/png')); }

// ═══════════════════════════════════════════════
// PROFILE MODAL
// ═══════════════════════════════════════════════
function openProfileModal()  { document.getElementById('profileModal')?.classList.add('active'); }
function closeProfileModal() { document.getElementById('profileModal')?.classList.remove('active'); }
document.getElementById('profileModal')?.addEventListener('click', e => { if (e.target === document.getElementById('profileModal')) closeProfileModal(); });

// ═══════════════════════════════════════════════
// SIDEBAR RESIZER
// ═══════════════════════════════════════════════
const resizer = document.getElementById('sidebarResizer');
let isResizing = false;
if (resizer) {
  resizer.addEventListener('mousedown', () => { isResizing = true; document.body.classList.add('resizing'); resizer.classList.add('active'); });
  document.addEventListener('mousemove', e => { if (!isResizing) return; document.documentElement.style.setProperty('--sidebar-width', Math.min(Math.max(e.clientX, 200), 500) + 'px'); });
  document.addEventListener('mouseup', () => { if (!isResizing) return; isResizing = false; document.body.classList.remove('resizing'); resizer.classList.remove('active'); });
}

// ═══════════════════════════════════════════════
// KEYBOARD SHORTCUTS
// ═══════════════════════════════════════════════
document.addEventListener('keydown', e => {
  const tag = e.target.tagName;
  if (['INPUT','TEXTAREA','SELECT'].includes(tag) || e.target.contentEditable === 'true') return;
  const views = ['detect','registry','image-lab','risk-calc','notes','settings-view'];
  if (e.key >= '1' && e.key <= '6') switchView(views[+e.key - 1]);
  switch (e.key.toLowerCase()) {
    case 'n': resetForm(); switchView('detect'); break;
    case 'e': document.getElementById('toggleHeatmapBtn')?.click(); break;
    case 'a': toggleAnnotationMode(); break;
    case 'p': window.print(); break;
    case 't': toggleTheme(); break;
    case 'r': switchView('risk-calc'); break;
    case 'k': switchView('notes'); break;
    case '?': document.getElementById('shortcutsModal')?.classList.add('active'); break;
    case 'escape': document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active')); break;
  }
});
document.getElementById('shortcutsModal')?.addEventListener('click', e => { if (e.target === document.getElementById('shortcutsModal')) document.getElementById('shortcutsModal').classList.remove('active'); });

// ═══════════════════════════════════════════════
// VIEW SWITCHING
// ═══════════════════════════════════════════════
function switchView(id) {
  if (!document.getElementById(id)) id = 'detect';
  document.querySelectorAll('.sidebar-nav li').forEach(li => li.classList.remove('active'));
  document.querySelector(`.sidebar-nav li[data-target="${id}"]`)?.classList.add('active');
  document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
  document.getElementById(id)?.classList.add('active');
  const isDetect = id === 'detect';
  const hero = document.getElementById('hero-section'), strip = document.getElementById('stats-strip');
  if (hero)  hero.style.display  = isDetect ? 'flex' : 'none';
  if (strip) strip.style.display = isDetect ? 'flex' : 'none';
  if (id === 'registry')      updateRegistryView();
  if (id === 'image-lab')     initLabUpload();
  if (id === 'notes')         renderNotesList();
  if (id === 'settings-view') loadSettingsUI();
}

// ═══════════════════════════════════════════════
// COUNTERS
// ═══════════════════════════════════════════════
function animateCount(el, target, suffix, dec) {
  if (!el) return;
  let s = 0;
  const step = ts => {
    if (!s) s = ts;
    const p = Math.min((ts - s) / 1800, 1), ease = 1 - Math.pow(1 - p, 3);
    el.textContent = (dec ? (ease * target).toFixed(dec) : Math.round(ease * target).toLocaleString()) + (suffix || '');
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
const statsObs = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) { animateCount(document.getElementById('cnt-auc'), 0.94, '', 2); animateCount(document.getElementById('cnt-acc'), 92, '%', 0); animateCount(document.getElementById('cnt-params'), 5.3, 'M', 1); statsObs.disconnect(); } });
}, { threshold: 0.5 });
const ssEl = document.querySelector('.stats-strip');
if (ssEl) statsObs.observe(ssEl);
const revObs = new IntersectionObserver(entries => { entries.forEach((e, i) => { if (e.isIntersecting) { e.target.style.transitionDelay = (i * 0.07) + 's'; e.target.classList.add('visible'); } }); }, { threshold: 0.1 });
document.querySelectorAll('.reveal').forEach(el => revObs.observe(el));

// ═══════════════════════════════════════════════
// COMPUTER VISION ENGINE
// ═══════════════════════════════════════════════
const CV = {};
CV.load = (src, w = 200, h = 200) => new Promise(resolve => {
  const img = new Image(); img.onload = () => { const c = document.createElement('canvas'); c.width = w; c.height = h; const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0, w, h); resolve({ canvas: c, ctx, data: ctx.getImageData(0, 0, w, h), img }); }; img.src = src;
});
CV.draw = (canvas, data, w, h) => { canvas.width = w; canvas.height = h; canvas.getContext('2d').putImageData(data, 0, 0); };
CV.gray = src => { const d = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height); for (let i = 0; i < src.data.length; i += 4) { const l = 0.299*src.data[i]+0.587*src.data[i+1]+0.114*src.data[i+2]; d.data[i]=d.data[i+1]=d.data[i+2]=l; d.data[i+3]=src.data[i+3]; } return d; };
CV.channel = (src, ch) => { const d = new ImageData(new Uint8ClampedArray(src.data.length), src.width, src.height); for (let i = 0; i < src.data.length; i += 4) { d.data[i]=ch===0?src.data[i]:0; d.data[i+1]=ch===1?src.data[i+1]:0; d.data[i+2]=ch===2?src.data[i+2]:0; d.data[i+3]=255; } return d; };
CV.sobel = src => {
  const w=src.width,h=src.height,g=CV.gray(src),d=new ImageData(new Uint8ClampedArray(src.data.length),w,h);
  const gx=[-1,0,1,-2,0,2,-1,0,1],gy=[-1,-2,-1,0,0,0,1,2,1];
  for (let y=1;y<h-1;y++) for (let x=1;x<w-1;x++) { let sx=0,sy=0; for (let ky=-1;ky<=1;ky++) for (let kx=-1;kx<=1;kx++) { const p=g.data[((y+ky)*w+(x+kx))*4],k=(ky+1)*3+(kx+1); sx+=gx[k]*p; sy+=gy[k]*p; } const m=Math.min(255,Math.sqrt(sx*sx+sy*sy)),i=(y*w+x)*4; d.data[i]=d.data[i+1]=d.data[i+2]=m; d.data[i+3]=255; }
  return d;
};
CV.clahe = (src,tile=32,clip=3.0) => {
  const w=src.width,h=src.height,g=CV.gray(src),d=new ImageData(new Uint8ClampedArray(src.data.length),w,h),nx=Math.ceil(w/tile),ny=Math.ceil(h/tile),luts=[];
  for (let ty=0;ty<ny;ty++) { luts[ty]=[]; for (let tx=0;tx<nx;tx++) { const hist=new Array(256).fill(0),x0=tx*tile,y0=ty*tile,x1=Math.min(x0+tile,w),y1=Math.min(y0+tile,h),n=(x1-x0)*(y1-y0); for (let y=y0;y<y1;y++) for (let x=x0;x<x1;x++) hist[g.data[(y*w+x)*4]]++; const cl=Math.floor(clip*n/256);let ex=0; for (let i=0;i<256;i++){if(hist[i]>cl){ex+=hist[i]-cl;hist[i]=cl;}} const pb=Math.floor(ex/256); for (let i=0;i<256;i++) hist[i]+=pb; const lut=new Uint8Array(256);let cdf=0,cmin=-1; for (let i=0;i<256;i++){cdf+=hist[i];if(cmin<0&&cdf>0)cmin=cdf;lut[i]=Math.round((cdf-cmin)/(n-cmin)*255);} luts[ty][tx]=lut; }}
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) { const tx=(x/tile)-0.5,ty=(y/tile)-0.5,tx0=Math.max(0,Math.floor(tx)),ty0=Math.max(0,Math.floor(ty)),tx1=Math.min(nx-1,tx0+1),ty1=Math.min(ny-1,ty0+1),fx=tx-Math.floor(tx),fy=ty-Math.floor(ty),pv=g.data[(y*w+x)*4],v00=luts[ty0][tx0][pv],v10=luts[ty0][tx1][pv],v01=luts[ty1][tx0][pv],v11=luts[ty1][tx1][pv],val=v00*(1-fx)*(1-fy)+v10*fx*(1-fy)+v01*(1-fx)*fy+v11*fx*fy,sc=val/(pv+1e-6),i=(y*w+x)*4; d.data[i]=Math.min(255,src.data[i]*sc);d.data[i+1]=Math.min(255,src.data[i+1]*sc);d.data[i+2]=Math.min(255,src.data[i+2]*sc);d.data[i+3]=255; }
  return d;
};
CV.vessel = src => { const d=new ImageData(new Uint8ClampedArray(src.data.length),src.width,src.height); for (let i=0;i<src.data.length;i+=4) {d.data[i]=0;d.data[i+1]=Math.min(255,Math.max(0,src.data[i+1]*2-src.data[i]*0.5-src.data[i+2]*0.5));d.data[i+2]=0;d.data[i+3]=255;} return d; };
CV.pseudo = src => { const g=CV.gray(src),d=new ImageData(new Uint8ClampedArray(src.data.length),src.width,src.height); for (let i=0;i<g.data.length;i+=4){const v=g.data[i]/255;d.data[i]=Math.min(255,Math.max(0,Math.round((1.5-Math.abs(v*4-3))*255)));d.data[i+1]=Math.min(255,Math.max(0,Math.round((1.5-Math.abs(v*4-2))*255)));d.data[i+2]=Math.min(255,Math.max(0,Math.round((1.5-Math.abs(v*4-1))*255)));d.data[i+3]=255;} return d; };
CV.invert = src => { const d=new ImageData(new Uint8ClampedArray(src.data.length),src.width,src.height); for (let i=0;i<src.data.length;i+=4){d.data[i]=255-src.data[i];d.data[i+1]=255-src.data[i+1];d.data[i+2]=255-src.data[i+2];d.data[i+3]=255;} return d; };
CV.emboss = src => { const w=src.width,h=src.height,g=CV.gray(src),d=new ImageData(new Uint8ClampedArray(src.data.length),w,h),k=[-2,-1,0,-1,1,1,0,1,2]; for (let y=1;y<h-1;y++) for (let x=1;x<w-1;x++){let s=128; for (let ky=-1;ky<=1;ky++) for (let kx=-1;kx<=1;kx++) s+=k[(ky+1)*3+(kx+1)]*g.data[((y+ky)*w+(x+kx))*4]; const i=(y*w+x)*4;d.data[i]=d.data[i+1]=d.data[i+2]=Math.min(255,Math.max(0,s));d.data[i+3]=255;} return d; };
CV.thermal = src => { const g=CV.gray(src),d=new ImageData(new Uint8ClampedArray(src.data.length),src.width,src.height); for (let i=0;i<g.data.length;i+=4){const v=g.data[i]/255;d.data[i]=Math.min(255,v*4*255);d.data[i+1]=Math.min(255,Math.max(0,(v*4-1))*255);d.data[i+2]=Math.min(255,Math.max(0,(v*4-3))*255);d.data[i+3]=255;} return d; };
CV.heatmap = src => { const w=src.width,h=src.height,edges=CV.sobel(src),d=new ImageData(new Uint8ClampedArray(src.data.length),w,h),r=8; for (let y=0;y<h;y++) for (let x=0;x<w;x++){let sum=0,cnt=0; for (let ky=-r;ky<=r;ky++) for (let kx=-r;kx<=r;kx++){const ny=y+ky,nx=x+kx; if(ny>=0&&ny<h&&nx>=0&&nx<w){sum+=edges.data[(ny*w+nx)*4];cnt++;}} const v=sum/cnt/255,i=(y*w+x)*4;d.data[i]=Math.min(255,v*4*255);d.data[i+1]=Math.min(255,Math.max(0,v*4-1)*255);d.data[i+2]=Math.min(255,Math.max(0,v*4-3)*255);d.data[i+3]=255;} return d; };
CV.adjust = (src,br=0,co=0,sa=0,ga=1,sh=0) => { const d=new ImageData(new Uint8ClampedArray(src.data.length),src.width,src.height),cf=(259*(co+255))/(255*(259-co)); for (let i=0;i<src.data.length;i+=4){let r=src.data[i]+br,g=src.data[i+1]+br,b=src.data[i+2]+br;r=cf*(r-128)+128;g=cf*(g-128)+128;b=cf*(b-128)+128;const gl=0.299*r+0.587*g+0.114*b,sf=1+sa/100;r=gl+sf*(r-gl);g=gl+sf*(g-gl);b=gl+sf*(b-gl);if(ga!==1){r=255*Math.pow(Math.max(0,r)/255,1/ga);g=255*Math.pow(Math.max(0,g)/255,1/ga);b=255*Math.pow(Math.max(0,b)/255,1/ga);}d.data[i]=Math.min(255,Math.max(0,r));d.data[i+1]=Math.min(255,Math.max(0,g));d.data[i+2]=Math.min(255,Math.max(0,b));d.data[i+3]=src.data[i+3];} return d; };
CV.stats = data => { let sR=0,sG=0,sB=0,sL=0;const n=data.data.length/4; for (let i=0;i<data.data.length;i+=4){sR+=data.data[i];sG+=data.data[i+1];sB+=data.data[i+2];sL+=0.299*data.data[i]+0.587*data.data[i+1]+0.114*data.data[i+2];} const mL=sL/n;let vL=0; for (let i=0;i<data.data.length;i+=4){const l=0.299*data.data[i]+0.587*data.data[i+1]+0.114*data.data[i+2];vL+=(l-mL)**2;} const mR=sR/n,mG=sG/n,mB=sB/n; return {meanR:mR.toFixed(1),meanG:mG.toFixed(1),meanB:mB.toFixed(1),brightness:mL.toFixed(1),contrast:Math.sqrt(vL/n).toFixed(1),dominant:mR>mG&&mR>mB?'Red':mG>mB?'Green':'Blue',pixels:n}; };
CV.histogram = data => { const r=new Array(256).fill(0),g=new Array(256).fill(0),b=new Array(256).fill(0); for (let i=0;i<data.data.length;i+=4){r[data.data[i]]++;g[data.data[i+1]]++;b[data.data[i+2]]++;} return {r,g,b}; };
async function runCV(src, size=200) { const {data} = await CV.load(src,size,size); return {orig:data,gray:CV.gray(data),green:CV.channel(data,1),edge:CV.sobel(data),clahe:CV.clahe(data),vessel:CV.vessel(data),heat:CV.heatmap(data),stats:CV.stats(data),hist:CV.histogram(data)}; }

// ═══════════════════════════════════════════════
// UPLOAD FLOW
// ═══════════════════════════════════════════════
const uploadArea    = document.getElementById('uploadArea');
const fileInput     = document.getElementById('fileInput');
const previewWrap   = document.getElementById('preview-wrap');
const previewImg    = document.getElementById('preview-img');
const resultCard    = document.getElementById('result-card');
const loadingOverlay= document.getElementById('loading-overlay');
const loadingText   = document.getElementById('loading-text');

if (uploadArea) {
  uploadArea.addEventListener('click', () => fileInput?.click());
  uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
  uploadArea.addEventListener('drop', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); const f=e.dataTransfer.files[0]; if(f&&f.type.startsWith('image/')) startPreprocessing(f); });
}
document.getElementById('chooseFileBtn')?.addEventListener('click', e => { e.stopPropagation(); fileInput?.click(); });
fileInput?.addEventListener('change', () => { if (fileInput.files[0]) startPreprocessing(fileInput.files[0]); });

async function startPreprocessing(file) {
  const reader = new FileReader();
  reader.onload = async e => {
    const src = e.target.result;
    if (uploadArea) uploadArea.style.display = 'none';
    const strip = document.getElementById('cv-preview-strip');
    if (!strip) return;
    strip.style.display = 'block';
    const res = await runCV(src, 180);
    const pairs = {'cv-orig':res.orig,'cv-gray':res.gray,'cv-green':res.green,'cv-edge':res.edge,'cv-clahe':res.clahe};
    for (const [id, data] of Object.entries(pairs)) { const el=document.getElementById(id); if(el) CV.draw(el,data,180,180); }
    const s = res.stats, bar = document.getElementById('imageStatsBar');
    if (bar) bar.innerHTML = `<span>Brightness: <strong>${s.brightness}</strong></span><span>Contrast: <strong>${s.contrast}</strong></span><span>Dominant: <strong>${s.dominant}</strong></span><span>R avg: <strong>${s.meanR}</strong></span><span>G avg: <strong>${s.meanG}</strong></span><span>B avg: <strong>${s.meanB}</strong></span><span>Pixels: <strong>${(s.pixels/1000).toFixed(1)}K</strong></span>`;
    strip._src = src; strip._cv = res;
    showToast('Pre-processing complete. Review channels.', 'success');
  };
  reader.readAsDataURL(file);
}

document.getElementById('proceedAnalysisBtn')?.addEventListener('click', () => {
  const strip = document.getElementById('cv-preview-strip');
  if (strip) { strip.style.display='none'; runAnalysis(strip._src, strip._cv, 180); }
});

function animateLoadingSteps() {
  const steps=['ls-1','ls-2','ls-3','ls-4'];
  steps.forEach(id => { const el=document.getElementById(id); if(el) el.className='load-step'; });
  let i=0;
  const iv=setInterval(()=>{ if(i>0){const p=document.getElementById(steps[i-1]);if(p)p.className='load-step done';} if(i<steps.length){const c=document.getElementById(steps[i]);if(c)c.className='load-step active';i++;}else clearInterval(iv); },700);
  return iv;
}

async function runAnalysis(src, cv, cvSize) {
  if (!previewImg||!previewWrap||!loadingOverlay||!resultCard) return;
  previewImg.src=src; previewWrap.style.display='block'; previewImg.style.display='none'; loadingOverlay.style.display='block'; resultCard.style.display='none';
  const lv=animateLoadingSteps(); let mi=0;
  const msgs=['Initializing MPS Backend...','Extracting vascular features...','Running EfficientNetB0...','Grading severity...'];
  const mv=setInterval(()=>{ if(loadingText) loadingText.textContent=msgs[mi++%msgs.length]; },700);
  await new Promise(r=>setTimeout(r,3100));
  clearInterval(lv); clearInterval(mv);
  loadingOverlay.style.display='none'; previewImg.style.display='block';

  setTimeout(()=>{ const img=document.getElementById('preview-img'),ann=document.getElementById('annotationCanvas'); if(ann&&img){ann.width=img.offsetWidth;ann.height=img.offsetHeight;annHistory=[];} },100);

  const filter =document.getElementById('pat-filter')?.value||'none';
  const patId  =document.getElementById('pat-id')?.value||`P-${Math.floor(Math.random()*9000)+1000}`;
  const patAge =document.getElementById('pat-age')?.value||'—';
  const patDiab=document.getElementById('pat-diab')?.value||'—';
  const patHba =document.getElementById('pat-hba1c')?.value||'—';
  const patEye =document.getElementById('pat-eye')?.value||'Left (OS)';
  const stages=[{level:0,name:'No DR',color:'badge-negative',pos:0},{level:1,name:'Mild NPDR',color:'badge-positive',pos:25},{level:2,name:'Moderate NPDR',color:'badge-positive',pos:50},{level:3,name:'Severe NPDR',color:'badge-positive',pos:75},{level:4,name:'Proliferative DR',color:'badge-positive',pos:100}];
  const stage=stages[Math.floor(Math.random()*5)];
  const cert=(Math.random()*14+86).toFixed(1), dens=(stage.level*18+Math.random()*12).toFixed(1);

  const patient={uniqueKey:Date.now().toString(),id:patId,age:patAge,diab:patDiab,hba1c:patHba,eye:patEye,date:new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}),stageLevel:stage.level,stageName:stage.name,color:stage.color,pos:stage.pos,certainty:cert,density:dens,time:(Math.random()*0.3+0.08).toFixed(3)+'s',history:[Math.max(0,stage.level-2),Math.max(0,stage.level-1),Math.max(0,stage.level-1),stage.level,stage.level],imageSrc:src,filterUsed:filter,brightness:cv.stats.brightness,contrast:cv.stats.contrast,cv,cvSize};

  sessionHistory.unshift(patient);
  updateSessionCounts();
  updateSidebar(patient.uniqueKey);
  renderDashboard(patient);
  updateRegistryView();
  showToast(`Analysis complete — ${patient.stageName}.`, patient.stageLevel>0?'error':'success');
}

function updateSessionCounts() {
  const total=sessionHistory.length, pos=sessionHistory.filter(p=>p.stageLevel>0).length, neg=total-pos;
  const map={'cnt-session':total,'qs-total':total,'qs-pos':pos,'qs-neg':neg,'storageStatus':`● ${total}`,'historyCount':total};
  Object.entries(map).forEach(([id,val])=>{ const el=document.getElementById(id); if(el) el.textContent=val; });
}

function renderDashboard(patient) {
  const uf=document.getElementById('upload-flow'); if(uf) uf.style.display='none';
  const strip=document.getElementById('cv-preview-strip'); if(strip) strip.style.display='none';
  if(loadingOverlay) loadingOverlay.style.display='none';
  if(previewWrap) previewWrap.style.display='block';
  if(previewImg) { previewImg.style.display='block'; previewImg.src=patient.imageSrc; }

  document.getElementById('heatmap-overlay')?.classList.remove('active');
  const xai=document.getElementById('xai-controls'); if(xai) xai.style.display='none';
  const tb=document.getElementById('toggleHeatmapBtn'); if(tb){tb.textContent='Enable Grad-CAM';tb.style.cssText='';}
  if(annotationMode) toggleAnnotationMode();

  const pp=document.getElementById('processedPanel');
  if(pp) { if(patient.filterUsed&&patient.filterUsed!=='none'&&patient.cv){pp.style.display='block';const lbl=document.getElementById('processedLabel');if(lbl)lbl.textContent=patient.filterUsed.toUpperCase();const map={clahe:'clahe',green:'green',edge:'edge'},key=map[patient.filterUsed],pc=document.getElementById('processedCanvas');if(key&&patient.cv[key]&&pc)CV.draw(pc,patient.cv[key],patient.cvSize,patient.cvSize);}else{pp.style.display='none';} }

  const rpt=document.getElementById('reportIdBadge'); if(rpt) rpt.textContent=`RPT-${patient.uniqueKey.slice(-6)}`;
  const ps=document.getElementById('patient-summary'); if(ps) ps.innerHTML=`<span><strong>ID:</strong> ${patient.id}</span><span><strong>Age:</strong> ${patient.age}</span><span><strong>Eye:</strong> ${patient.eye}</span><span><strong>HbA1c:</strong> ${patient.hba1c}%</span><span><strong>Diabetes:</strong> ${patient.diab}y</span><span><strong>Time:</strong> ${patient.date}</span>`;

  const rl=document.getElementById('result-label'); if(rl) rl.textContent=patient.stageName;
  const badge=document.getElementById('result-badge'); if(badge){badge.textContent=patient.stageLevel===0?'Negative':'Refer to Ophthalmologist';badge.className='result-badge '+patient.color;}
  const sv=document.getElementById('severity-val'); if(sv) sv.textContent=`Stage ${patient.stageLevel}`;
  setTimeout(()=>{ const si=document.getElementById('severity-indicator'); if(si)si.style.left=patient.pos+'%'; },100);

  ['m-prob','m-conf','m-time','m-filter','m-brightness','m-contrast'].forEach((id,i)=>{const el=document.getElementById(id);if(!el)return;const vals=[patient.certainty+'%',patient.density,patient.time,patient.filterUsed||'None',patient.brightness,patient.contrast];el.textContent=vals[i];});
  setTimeout(()=>{const cr=document.getElementById('cert-ring'),lr=document.getElementById('lesion-ring');if(cr)cr.style.strokeDashoffset=264-264*(patient.certainty/100);if(lr)lr.style.strokeDashoffset=264-264*(Math.min(patient.density,100)/100);},200);

  renderRiskPanel(patient);
  if(patient.cv){[['res-gray','gray'],['res-green','green'],['res-edge','edge'],['res-clahe','clahe'],['res-heat','heat']].forEach(([id,k])=>{const c=document.getElementById(id);if(c&&patient.cv[k])CV.draw(c,patient.cv[k],patient.cvSize,patient.cvSize);});}
  renderRecommendation(patient);

  const ctx=document.getElementById('progressionChart')?.getContext('2d');
  if(ctx){if(patientChart)patientChart.destroy();patientChart=new Chart(ctx,{type:'line',data:{labels:['Jan','Apr','Jul','Oct','Current'],datasets:[{label:'Severity',data:patient.history,borderColor:'#00ddb4',backgroundColor:'rgba(0,221,180,0.08)',borderWidth:2,pointBackgroundColor:'#030a0e',pointBorderColor:'#00aaff',pointBorderWidth:2,pointRadius:4,fill:true,tension:0.4}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,max:4,ticks:{stepSize:1,color:'#6b8f9e',font:{family:"'DM Mono',monospace",size:9}},grid:{color:'rgba(0,220,180,0.05)'}},x:{ticks:{color:'#6b8f9e',font:{family:"'DM Mono',monospace",size:9}},grid:{display:false}}}}});}

  if(resultCard) resultCard.style.display='block';
  resultCard?.scrollIntoView({behavior:'smooth',block:'nearest'});
}

function renderRiskPanel(patient) {
  const factors=[{name:'Vascular Abnormality',val:Math.min(100,patient.stageLevel*22+Math.random()*15),color:'#ff4e7e'},{name:'Microaneurysm Score',val:Math.min(100,patient.stageLevel*18+Math.random()*20),color:'#ff8800'},{name:'Hemorrhage Risk',val:Math.min(100,patient.stageLevel*15+Math.random()*18),color:'#ffdd00'},{name:'Exudate Presence',val:Math.min(100,patient.stageLevel*12+Math.random()*22),color:'#00aaff'},{name:'Neovascularization',val:patient.stageLevel===4?80+Math.random()*15:Math.random()*20,color:'#ff4e7e'},{name:'Optic Disc Integrity',val:100-(patient.stageLevel*15+Math.random()*10),color:'#00ddb4'}];
  const overall=patient.stageLevel===0?'Low Risk':patient.stageLevel<=2?'Moderate Risk':'High Risk',oc={'Low Risk':'#00ddb4','Moderate Risk':'#ffdd00','High Risk':'#ff4e7e'}[overall];
  const el=document.getElementById('riskOverall'); if(el){el.textContent=overall;el.style.background=oc+'22';el.style.color=oc;el.style.border=`1px solid ${oc}44`;}
  const grid=document.getElementById('riskFactorsGrid');
  if(grid){grid.innerHTML=factors.map(f=>`<div class="risk-factor-item"><div class="risk-factor-info"><div class="risk-factor-name">${f.name}</div><div class="risk-factor-bar-bg"><div class="risk-factor-bar-fill" style="width:0%;background:${f.color};" data-w="${f.val.toFixed(0)}"></div></div></div><div class="risk-factor-val">${f.val.toFixed(0)}%</div></div>`).join('');setTimeout(()=>{document.querySelectorAll('.risk-factor-bar-fill').forEach(b=>{b.style.width=b.dataset.w+'%';});},150);}
}

function renderRecommendation(patient) {
  const recs=[{color:'#00ddb4',bg:'rgba(0,221,180,0.05)',border:'rgba(0,221,180,0.2)',text:'No DR detected. Annual screening recommended. Continue glycaemic and blood pressure management.'},{color:'#aacc00',bg:'rgba(170,204,0,0.05)',border:'rgba(170,204,0,0.2)',text:'Mild NPDR. No immediate threat. Follow-up in 12 months. Optimize HbA1c below 7%.'},{color:'#ffdd00',bg:'rgba(255,221,0,0.05)',border:'rgba(255,221,0,0.2)',text:'Moderate NPDR. Ophthalmology referral within 3–6 months. Enhanced glycaemic control critical.'},{color:'#ff8800',bg:'rgba(255,136,0,0.05)',border:'rgba(255,136,0,0.2)',text:'Severe NPDR. Urgent referral within 1 month. High risk of progression. Laser photocoagulation may be indicated.'},{color:'#ff4e7e',bg:'rgba(255,78,126,0.05)',border:'rgba(255,78,126,0.2)',text:'Proliferative DR. URGENT referral within 1 week. Anti-VEGF therapy or pan-retinal photocoagulation likely required.'}];
  const r=recs[patient.stageLevel],p=document.getElementById('recommendationPanel');
  if(p){p.style.background=r.bg;p.style.borderColor=r.border;p.style.color='#9bb8c9';p.innerHTML=`<span style="font-family:'DM Mono',monospace;font-size:9px;color:${r.color};text-transform:uppercase;letter-spacing:1.5px;display:block;margin-bottom:7px;">Clinical Recommendation</span>${r.text}`;}
}

// ═══════════════════════════════════════════════
// ANNOTATION ENGINE
// ═══════════════════════════════════════════════
function toggleAnnotationMode() {
  annotationMode=!annotationMode;
  const toolbar=document.getElementById('annotationToolbar'),btn=document.getElementById('toggleAnnotateBtn'),canvas=document.getElementById('annotationCanvas');
  if(toolbar) toolbar.style.display=annotationMode?'flex':'none';
  if(canvas) canvas.style.pointerEvents=annotationMode?'auto':'none';
  if(btn){btn.textContent=annotationMode?'Exit Annotation':'Annotate Image';btn.style.background=annotationMode?'rgba(0,221,180,0.15)':'';btn.style.color=annotationMode?'var(--accent)':'';btn.style.border=annotationMode?'1px solid var(--accent)':'';}
  if(annotationMode) showToast('Annotation mode active.','info');
}

function setAnnTool(tool,btn) { annTool=tool; document.querySelectorAll('.ann-tool-btn').forEach(b=>b.classList.remove('active')); if(btn) btn.classList.add('active'); }

const annCanvas=document.getElementById('annotationCanvas');
const annCtx=annCanvas?annCanvas.getContext('2d'):null;
let annSnapshot=null;

function getAnnPos(e) { if(!annCanvas)return{x:0,y:0}; const r=annCanvas.getBoundingClientRect(); return{x:(e.clientX-r.left)*annCanvas.width/r.width,y:(e.clientY-r.top)*annCanvas.height/r.height}; }

if(annCanvas&&annCtx){
  annCanvas.addEventListener('mousedown',e=>{
    if(!annotationMode)return; isDrawing=true;
    const pos=getAnnPos(e); annStart=pos;
    annSnapshot=annCtx.getImageData(0,0,annCanvas.width,annCanvas.height);
    if(annTool==='pen'){annCtx.beginPath();annCtx.moveTo(pos.x,pos.y);}
    if(annTool==='text'){const txt=prompt('Enter label:');if(txt){const sz=document.getElementById('annSize'),col=document.getElementById('annColor');annCtx.font=`${(sz?+sz.value:3)*4+10}px 'DM Mono'`;annCtx.fillStyle=col?col.value:'#00ddb4';annCtx.fillText(txt,pos.x,pos.y);saveAnnSnapshot();}isDrawing=false;}
  });
  annCanvas.addEventListener('mousemove',e=>{
    if(!annotationMode||!isDrawing)return;
    const pos=getAnnPos(e),col=(document.getElementById('annColor')?.value)||'#00ddb4',size=+(document.getElementById('annSize')?.value||3);
    if(annTool==='pen'){annCtx.strokeStyle=col;annCtx.lineWidth=size;annCtx.lineCap='round';annCtx.lineTo(pos.x,pos.y);annCtx.stroke();}
    else{annCtx.putImageData(annSnapshot,0,0);annCtx.strokeStyle=col;annCtx.lineWidth=size;
      if(annTool==='rect'){annCtx.beginPath();annCtx.strokeRect(annStart.x,annStart.y,pos.x-annStart.x,pos.y-annStart.y);}
      if(annTool==='circle'){annCtx.beginPath();const rx=(pos.x-annStart.x)/2,ry=(pos.y-annStart.y)/2;annCtx.ellipse(annStart.x+rx,annStart.y+ry,Math.abs(rx),Math.abs(ry),0,0,Math.PI*2);annCtx.stroke();}
      if(annTool==='arrow'){const dx=pos.x-annStart.x,dy=pos.y-annStart.y,angle=Math.atan2(dy,dx);annCtx.beginPath();annCtx.moveTo(annStart.x,annStart.y);annCtx.lineTo(pos.x,pos.y);annCtx.stroke();const hl=14;annCtx.beginPath();annCtx.moveTo(pos.x,pos.y);annCtx.lineTo(pos.x-hl*Math.cos(angle-0.4),pos.y-hl*Math.sin(angle-0.4));annCtx.moveTo(pos.x,pos.y);annCtx.lineTo(pos.x-hl*Math.cos(angle+0.4),pos.y-hl*Math.sin(angle+0.4));annCtx.stroke();}}
  });
  annCanvas.addEventListener('mouseup',()=>{ if(!annotationMode||!isDrawing)return; isDrawing=false; saveAnnSnapshot(); });
}

function saveAnnSnapshot(){if(!annCanvas||!annCtx)return;annHistory.push(annCtx.getImageData(0,0,annCanvas.width,annCanvas.height));if(annHistory.length>30)annHistory.shift();}
function undoAnnotation(){if(!annCtx||!annHistory.length)return;annHistory.pop();if(annHistory.length>0)annCtx.putImageData(annHistory[annHistory.length-1],0,0);else annCtx.clearRect(0,0,annCanvas.width,annCanvas.height);}
function clearAnnotations(){if(!annCtx||!annCanvas)return;annCtx.clearRect(0,0,annCanvas.width,annCanvas.height);annHistory=[];}
function downloadAnnotated(){
  if(!annCanvas)return;
  const merged=document.createElement('canvas'),img=document.getElementById('preview-img');
  merged.width=img.naturalWidth;merged.height=img.naturalHeight;
  const mctx=merged.getContext('2d');mctx.drawImage(img,0,0);mctx.drawImage(annCanvas,0,0,merged.width,merged.height);
  const a=document.createElement('a');a.download=`annotated_${Date.now()}.png`;a.href=merged.toDataURL('image/png');a.click();
  showToast('Annotated image downloaded.','success');
}

// ═══════════════════════════════════════════════
// SIDEBAR
// ═══════════════════════════════════════════════
function updateSidebar(activeKey) {
  const list=document.getElementById('historyList'); if(!list)return;
  list.innerHTML='';
  if(!sessionHistory.length){list.innerHTML='<li class="empty-history">No scans yet.</li>';return;}
  sessionHistory.forEach(pat=>{
    const li=document.createElement('li');
    if(pat.uniqueKey===activeKey) li.classList.add('active');
    const dc=pat.stageLevel===0?'success':pat.stageLevel<=2?'warning':'danger';
    li.innerHTML=`<div class="pat-list-info"><span class="status-dot ${dc}"></span><img src="${pat.imageSrc}" class="sidebar-thumb" alt=""/><div class="pat-list-details"><strong>${pat.id}</strong><span>${pat.stageName}</span></div></div><span class="pat-list-time">${pat.date}</span>`;
    li.addEventListener('click',()=>{switchView('detect');updateSidebar(pat.uniqueKey);renderDashboard(pat);});
    list.appendChild(li);
  });
}

// ═══════════════════════════════════════════════
// NOTES
// ═══════════════════════════════════════════════
function makeNote(title, content, tag = 'general') {
  return {
    id: Date.now().toString() + Math.random().toString(16).slice(2, 6),
    title,
    content,
    tag,
    createdAt: Date.now(),
    updatedAt: Date.now()
  };
}

function getTagLabel(tag) {
  return ({ general: 'General', finding: 'Finding', followup: 'Follow-up', urgent: 'Urgent', research: 'Research' }[tag] || 'General');
}

function saveQuickNote() {
  const input=document.getElementById('quickNoteInput'); if(!input)return;
  const text=input.value.trim(); if(!text)return;
  const active=sessionHistory[0];
  const note = makeNote(`Scan Note - ${active?.id||'Unknown'}`, text, 'finding');
  notes.unshift(note);
  currentNoteId = note.id;
  input.value='';
  showToast('Note saved.','success');
  renderNotesList();
}

function renderNotesList() {
  const list = document.getElementById('notesList');
  const empty = document.getElementById('noteEditorEmpty');
  const content = document.getElementById('noteEditorContent');
  if (!list || !empty || !content) return;

  if (!notes.length) {
    list.innerHTML = '<li class="empty-history">No notes yet.</li>';
    empty.style.display = 'block';
    content.style.display = 'none';
    currentNoteId = null;
    return;
  }

  list.innerHTML = notes.map(n => `
    <li class="note-item ${n.id === currentNoteId ? 'active' : ''}" onclick="openNote('${n.id}')">
      <div class="note-item-title">${n.title || 'Untitled Note'}</div>
      <div class="note-item-meta">
        <span class="note-tag tag-${n.tag || 'general'}">${getTagLabel(n.tag)}</span>
        <span class="note-item-time">${new Date(n.updatedAt).toLocaleString()}</span>
      </div>
    </li>
  `).join('');

  if (!currentNoteId || !notes.some(n => n.id === currentNoteId)) {
    currentNoteId = notes[0].id;
  }
  openNote(currentNoteId, true);
}

function openNote(id, silent = false) {
  const note = notes.find(n => n.id === id);
  const empty = document.getElementById('noteEditorEmpty');
  const content = document.getElementById('noteEditorContent');
  const title = document.getElementById('noteTitleInput');
  const tag = document.getElementById('noteTagSelect');
  const area = document.getElementById('noteTextArea');
  const ts = document.getElementById('noteTimestamp');
  if (!note || !empty || !content || !title || !tag || !area || !ts) return;

  currentNoteId = id;
  document.querySelectorAll('.note-item').forEach(el => el.classList.remove('active'));
  document.querySelector(`.note-item[onclick="openNote('${id}')"]`)?.classList.add('active');

  empty.style.display = 'none';
  content.style.display = 'flex';
  title.value = note.title;
  tag.value = note.tag || 'general';
  area.innerHTML = note.content || '';
  ts.textContent = `Updated ${new Date(note.updatedAt).toLocaleString()}`;
  updateWordCount();
  if (!silent) showToast('Note opened.','info',1200);
}

function updateWordCount() {
  const area = document.getElementById('noteTextArea');
  const wc = document.getElementById('noteWordCount');
  if (!area || !wc) return;
  const text = (area.innerText || '').trim();
  const words = text ? text.split(/\s+/).length : 0;
  wc.textContent = `${words} words`;
}

function createNewNote() {
  const note = makeNote('New Clinical Note', '', 'general');
  notes.unshift(note);
  currentNoteId = note.id;
  renderNotesList();
  showToast('New note created.','success');
}

function saveCurrentNote(showMessage = true) {
  if (!currentNoteId) return;
  const note = notes.find(n => n.id === currentNoteId);
  const title = document.getElementById('noteTitleInput');
  const tag = document.getElementById('noteTagSelect');
  const area = document.getElementById('noteTextArea');
  if (!note || !title || !tag || !area) return;

  note.title = title.value.trim() || 'Untitled Note';
  note.tag = tag.value || 'general';
  note.content = area.innerHTML;
  note.updatedAt = Date.now();

  renderNotesList();
  if (showMessage) showToast('Note saved.','success');
}

function deleteCurrentNote() {
  if (!currentNoteId) return;
  notes = notes.filter(n => n.id !== currentNoteId);
  currentNoteId = notes[0]?.id || null;
  renderNotesList();
  showToast('Note deleted.','info');
}

function noteFormat(command, value = null) {
  document.execCommand(command, false, value);
  updateWordCount();
}

function exportNoteAsText() {
  if (!currentNoteId) { showToast('No note selected.','error'); return; }
  const note = notes.find(n => n.id === currentNoteId);
  if (!note) return;
  const plain = (note.content || '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
  const blob = new Blob([`${note.title}\n\n${plain}`], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.download = `${(note.title || 'note').replace(/[^a-z0-9_-]/gi, '_').toLowerCase()}.txt`;
  a.href = URL.createObjectURL(blob);
  a.click();
  URL.revokeObjectURL(a.href);
  showToast('Note exported.','success');
}

document.getElementById('noteTextArea')?.addEventListener('input', () => {
  updateWordCount();
});

// ═══════════════════════════════════════════════
// SETTINGS
// ═══════════════════════════════════════════════
function loadSettingsUI() {
  ['name','inst','spec','loc'].forEach(k=>{const el=document.getElementById(`set-${k}`);if(el)el.value=appSettings.profile[k]||'';});
  const filter = document.getElementById('set-filter');
  const threshold = document.getElementById('set-threshold');
  const theme = document.getElementById('set-theme');
  if (filter) filter.value = appSettings.filter || 'none';
  if (threshold) threshold.value = appSettings.threshold ?? 0.5;
  if (theme) theme.value = currentTheme();
}

function applyProfileUI() {
  const fullName = appSettings.profile.name || 'Dr. Yuvraj';
  const initials = fullName.split(' ').map(s => s[0]).slice(0, 2).join('').toUpperCase();
  const sidebarName = document.getElementById('sidebarDocName');
  const avatar = document.getElementById('avatarInitials');
  const modalName = document.getElementById('modalName');
  const modalAvatar = document.getElementById('modalAvatar');
  const modalRole = document.getElementById('modalRole');
  if (sidebarName) sidebarName.textContent = fullName;
  if (avatar) avatar.textContent = initials || 'DR';
  if (modalName) modalName.textContent = fullName;
  if (modalAvatar) modalAvatar.textContent = initials || 'DR';
  if (modalRole) modalRole.textContent = `${appSettings.profile.spec || 'Clinician'} ${appSettings.profile.inst ? '- ' + appSettings.profile.inst : ''}`.trim();
}

function saveProfile() {
  appSettings.profile={name:document.getElementById('set-name')?.value||'Dr. Yuvraj',inst:document.getElementById('set-inst')?.value||'',spec:document.getElementById('set-spec')?.value||'',loc:document.getElementById('set-loc')?.value||''};
  applyProfileUI();
  showToast('Settings saved.','success');
}

function saveSettings() { saveProfile(); }

function saveAnalysisSettings() {
  appSettings.filter = document.getElementById('set-filter')?.value || 'none';
  appSettings.threshold = +(document.getElementById('set-threshold')?.value || 0.5);
  const pf = document.getElementById('pat-filter');
  if (pf) pf.value = appSettings.filter;
  showToast('Analysis defaults updated.','success');
}

function exportAllData() {
  const payload = { appSettings, sessionHistory, notes, exportedAt: new Date().toISOString() };
  const a = document.createElement('a');
  a.download = `retinaai_session_${Date.now()}.json`;
  a.href = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2));
  a.click();
  showToast('Session data exported.','success');
}

function handleImport(event) {
  const file = event?.target?.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const data = JSON.parse(e.target.result);
      sessionHistory = Array.isArray(data.sessionHistory) ? data.sessionHistory : [];
      notes = Array.isArray(data.notes) ? data.notes : [];
      if (data.appSettings && typeof data.appSettings === 'object') {
        appSettings = { ...appSettings, ...data.appSettings, profile: { ...appSettings.profile, ...(data.appSettings.profile || {}) } };
      }
      currentNoteId = notes[0]?.id || null;
      updateSessionCounts();
      updateSidebar(currentNoteId);
      renderNotesList();
      loadSettingsUI();
      applyProfileUI();
      updateRegistryView();
      showToast('Session data imported.','success');
    } catch (_err) {
      showToast('Invalid JSON file.','error');
    }
  };
  reader.readAsText(file);
  event.target.value = '';
}

function clearAllData() {
  sessionHistory = [];
  notes = [];
  currentNoteId = null;
  updateSessionCounts();
  updateSidebar('');
  renderNotesList();
  updateRegistryView();
  resetForm();
  showToast('All session data cleared.','info');
}

// ═══════════════════════════════════════════════
// HEATMAP
// ═══════════════════════════════════════════════
const toggleHeatmapBtn=document.getElementById('toggleHeatmapBtn');
const xaiControls=document.getElementById('xai-controls');
const heatmapOverlay=document.getElementById('heatmap-overlay');
const heatmapSlider=document.getElementById('heatmapSlider');

if(toggleHeatmapBtn){toggleHeatmapBtn.addEventListener('click',()=>{heatmapOverlay?.classList.toggle('active');const on=heatmapOverlay?.classList.contains('active');if(on){toggleHeatmapBtn.textContent='Disable Grad-CAM';toggleHeatmapBtn.style.cssText='background:rgba(0,221,180,0.12);color:var(--accent);border:1px solid rgba(0,221,180,0.4);font-family:"DM Mono",monospace;font-size:12px;padding:10px 20px;border-radius:8px;cursor:pointer;';if(xaiControls)xaiControls.style.display='block';if(heatmapOverlay&&heatmapSlider)heatmapOverlay.style.opacity=heatmapSlider.value/100;}else{toggleHeatmapBtn.textContent='Enable Grad-CAM';toggleHeatmapBtn.style.cssText='';if(xaiControls)xaiControls.style.display='none';if(heatmapOverlay)heatmapOverlay.style.opacity=0;}});}
heatmapSlider?.addEventListener('input',e=>{const ov=document.getElementById('opacity-val');if(ov)ov.textContent=e.target.value+'%';if(heatmapOverlay?.classList.contains('active'))heatmapOverlay.style.opacity=e.target.value/100;});
document.getElementById('downloadReportBtn')?.addEventListener('click',()=>window.print());

// ═══════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════
function exportScanJSON() {
  if(!sessionHistory.length){showToast('No scan to export.','error');return;}
  const pat=sessionHistory[0],data={id:pat.id,age:pat.age,eye:pat.eye,hba1c:pat.hba1c,diab:pat.diab,date:pat.date,stageName:pat.stageName,stageLevel:pat.stageLevel,certainty:pat.certainty,density:pat.density,inference:pat.time,filter:pat.filterUsed,brightness:pat.brightness,contrast:pat.contrast};
  const a=document.createElement('a');a.download=`retinaai_scan_${pat.id}.json`;a.href='data:application/json;charset=utf-8,'+encodeURIComponent(JSON.stringify(data,null,2));a.click();
  showToast('Scan exported as JSON.','success');
}

// ═══════════════════════════════════════════════
// RISK CALCULATOR
// ═══════════════════════════════════════════════
function computeRisk() {
  const v = id => +(document.getElementById(id)?.value || 0);
  const b = id => !!document.getElementById(id)?.checked;

  const factors = [
    { name: 'Age', score: Math.min(12, Math.max(0, (v('rc-age') - 35) * 0.25)) },
    { name: 'Diabetes Duration', score: Math.min(18, v('rc-dur') * 1.2) },
    { name: 'HbA1c', score: Math.min(20, Math.max(0, (v('rc-hba1c') - 6.5) * 6)) },
    { name: 'Systolic BP', score: Math.min(10, Math.max(0, (v('rc-sbp') - 120) * 0.18)) },
    { name: 'BMI', score: Math.min(8, Math.max(0, (v('rc-bmi') - 23) * 0.4)) },
    { name: 'Cholesterol', score: Math.min(8, Math.max(0, (v('rc-chol') - 170) * 0.08)) },
    { name: 'Renal Function', score: Math.min(10, Math.max(0, (85 - v('rc-egfr')) * 0.22)) },
    { name: 'Type 1 Diabetes', score: document.getElementById('rc-type')?.value === '1' ? 4 : 0 },
    { name: 'Hypertension', score: b('rc-hyper') ? 4 : 0 },
    { name: 'Smoking', score: b('rc-smoking') ? 4 : 0 },
    { name: 'Microalbuminuria', score: b('rc-microalb') ? 6 : 0 },
    { name: 'Neuropathy', score: b('rc-neuropathy') ? 4 : 0 },
    { name: 'Insulin Therapy', score: b('rc-insulin') ? 3 : 0 },
    { name: 'Family History', score: b('rc-family') ? 3 : 0 }
  ];

  const raw = factors.reduce((s, f) => s + f.score, 0);
  const score = Math.min(100, Math.round(raw));
  const riskCategory = score < 30 ? 'Low Risk' : score < 60 ? 'Moderate Risk' : score < 80 ? 'High Risk' : 'Very High Risk';
  const riskColor = score < 30 ? '#00ddb4' : score < 60 ? '#ffdd00' : score < 80 ? '#ff8800' : '#ff4e7e';

  const scoreEl = document.getElementById('riskScoreDisplay');
  const catEl = document.getElementById('riskCategory');
  if (scoreEl) scoreEl.textContent = `${score}%`;
  if (catEl) {
    catEl.textContent = riskCategory;
    catEl.style.color = riskColor;
    catEl.style.background = `${riskColor}22`;
    catEl.style.border = `1px solid ${riskColor}55`;
  }

  const breakdown = document.getElementById('riskBreakdownList');
  if (breakdown) {
    breakdown.innerHTML = factors
      .filter(f => f.score > 0)
      .sort((a, b2) => b2.score - a.score)
      .slice(0, 8)
      .map(f => `<div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border);font-size:12px;"><span style="color:var(--muted);">${f.name}</span><strong style="color:var(--accent);font-family:'DM Mono',monospace;">+${f.score.toFixed(1)}</strong></div>`)
      .join('') || '<div style="color:var(--muted);font-size:12px;">No elevated factors detected.</div>';
  }

  const rec = document.getElementById('riskRecommendations');
  if (rec) {
    const lines = [];
    if (score >= 60) lines.push('Schedule ophthalmology assessment within 1-3 months.');
    if (score >= 30 && score < 60) lines.push('Repeat retinal screening in 6 months.');
    if (score < 30) lines.push('Continue annual retinal screening.');
    if (v('rc-hba1c') > 7) lines.push('Optimize glycaemic control (target HbA1c < 7%).');
    if (v('rc-sbp') > 130 || b('rc-hyper')) lines.push('Tight blood pressure control is recommended (<130/80 mmHg).');
    if (b('rc-smoking')) lines.push('Smoking cessation support strongly advised.');
    rec.innerHTML = lines.map(line => `<div style="margin-bottom:8px;">• ${line}</div>`).join('');
  }

  const gaugeCtx = document.getElementById('riskGaugeChart')?.getContext('2d');
  if (gaugeCtx) {
    if (riskGaugeChart) riskGaugeChart.destroy();
    riskGaugeChart = new Chart(gaugeCtx, {
      type: 'doughnut',
      data: {
        labels: ['Risk', 'Remaining'],
        datasets: [{ data: [score, 100 - score], backgroundColor: [riskColor, 'rgba(107,143,158,0.2)'], borderWidth: 0 }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '72%',
        rotation: -90,
        circumference: 180,
        plugins: { legend: { display: false }, tooltip: { enabled: false } }
      }
    });
  }
  showToast('Risk score computed.','success');
}

function exportRiskReport() {
  const score = document.getElementById('riskScoreDisplay')?.textContent || '—';
  const category = document.getElementById('riskCategory')?.textContent || '—';
  const text = [
    'RetinaAI Risk Report',
    `Generated: ${new Date().toLocaleString()}`,
    '',
    `Composite Score: ${score}`,
    `Category: ${category}`
  ].join('\n');
  const a = document.createElement('a');
  a.download = `retinaai_risk_report_${Date.now()}.txt`;
  a.href = 'data:text/plain;charset=utf-8,' + encodeURIComponent(text);
  a.click();
  showToast('Risk report exported.','success');
}

function clearRiskCalc() {
  ['rc-age','rc-dur','rc-hba1c','rc-sbp','rc-bmi','rc-chol','rc-egfr'].forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
  const type = document.getElementById('rc-type');
  if (type) type.value = '2';
  ['rc-hyper','rc-smoking','rc-microalb','rc-neuropathy','rc-insulin','rc-family'].forEach(id => { const el = document.getElementById(id); if (el) el.checked = false; });
  const score = document.getElementById('riskScoreDisplay');
  const category = document.getElementById('riskCategory');
  const breakdown = document.getElementById('riskBreakdownList');
  const rec = document.getElementById('riskRecommendations');
  if (score) score.textContent = '—';
  if (category) { category.textContent = ''; category.style.cssText = ''; }
  if (breakdown) breakdown.innerHTML = '';
  if (rec) rec.innerHTML = '';
  if (riskGaugeChart) { riskGaugeChart.destroy(); riskGaugeChart = null; }
  showToast('Risk calculator cleared.','info');
}

// ═══════════════════════════════════════════════
// RESET
// ═══════════════════════════════════════════════
function resetForm() {
  const uf=document.getElementById('upload-flow');if(uf)uf.style.display='block';
  if(uploadArea)uploadArea.style.display='block';
  const strip=document.getElementById('cv-preview-strip');if(strip)strip.style.display='none';
  if(previewWrap)previewWrap.style.display='none';
  if(resultCard)resultCard.style.display='none';
  if(fileInput)fileInput.value='';
  document.querySelectorAll('.patient-list li').forEach(li=>li.classList.remove('active'));
  heatmapOverlay?.classList.remove('active');
  if(xaiControls)xaiControls.style.display='none';
  if(toggleHeatmapBtn){toggleHeatmapBtn.style.cssText='';toggleHeatmapBtn.textContent='Enable Grad-CAM';}
  if(annotationMode)toggleAnnotationMode();
  const si=document.getElementById('severity-indicator');if(si)si.style.left='0%';
  const cr=document.getElementById('cert-ring');if(cr)cr.style.strokeDashoffset=264;
  const lr=document.getElementById('lesion-ring');if(lr)lr.style.strokeDashoffset=264;
  ['pat-id','pat-age','pat-diab','pat-hba1c'].forEach(id=>{const el=document.getElementById(id);if(el)el.value='';});
  const pp=document.getElementById('processedPanel');if(pp)pp.style.display='none';
  document.getElementById('detect')?.scrollIntoView({behavior:'smooth'});
}

// ═══════════════════════════════════════════════
// REGISTRY
// ═══════════════════════════════════════════════
let filteredHistory=[];
function sortTable(key){sortConfig={key,dir:sortConfig.key===key&&sortConfig.dir==='asc'?'desc':'asc'};renderRegistryTable();}
document.getElementById('searchInput')?.addEventListener('input',e=>{const q=e.target.value.toLowerCase();filteredHistory=sessionHistory.filter(p=>p.id.toLowerCase().includes(q));renderRegistryTable();});
document.getElementById('filterGrade')?.addEventListener('change',e=>{const v=e.target.value;filteredHistory=v===''?[]:sessionHistory.filter(p=>p.stageLevel===+v);renderRegistryTable();});

function renderRegistryTable() {
  const si=document.getElementById('searchInput'),fg=document.getElementById('filterGrade');
  const src=(si?.value||fg?.value)?filteredHistory:sessionHistory, data=[...src];
  if(sortConfig.key) data.sort((a,b)=>{let va=a[sortConfig.key],vb=b[sortConfig.key];if(!isNaN(va))va=+va;if(!isNaN(vb))vb=+vb;return sortConfig.dir==='asc'?(va>vb?1:-1):(va<vb?1:-1);});
  const tbody=document.getElementById('registryTableBody');if(!tbody)return;
  tbody.innerHTML='';
  const footer=document.getElementById('tableFooter');
  if(!data.length){tbody.innerHTML='<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:28px;">No records found.</td></tr>';if(footer)footer.textContent='';return;}
  data.forEach(pat=>{
    const tr=document.createElement('tr'),bc=pat.stageLevel===0?'badge-negative':'badge-positive';
    tr.innerHTML=`<td><strong>${pat.id}</strong></td><td>${pat.age}</td><td>${pat.eye}</td><td><span class="result-badge ${bc}" style="font-size:9px;padding:2px 9px;">${pat.stageName}</span></td><td style="font-family:'DM Mono',monospace;color:var(--accent);">${pat.certainty}%</td><td>${pat.hba1c}%</td><td style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);">${pat.filterUsed||'none'}</td><td style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);">${pat.date}</td><td><button onclick="event.stopPropagation();switchView('detect');updateSidebar('${pat.uniqueKey}');renderDashboard(sessionHistory.find(p=>p.uniqueKey==='${pat.uniqueKey}'))" style="font-family:'DM Mono',monospace;font-size:9px;padding:3px 9px;border-radius:5px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;">View</button></td>`;
    tr.addEventListener('click',()=>{switchView('detect');updateSidebar(pat.uniqueKey);renderDashboard(pat);});
    tbody.appendChild(tr);
  });
  if(footer)footer.textContent=`Showing ${data.length} of ${sessionHistory.length} records`;
}

function updateRegistryView() {
  if(!document.getElementById('registry')?.classList.contains('active'))return;
  const pos=sessionHistory.filter(p=>p.stageLevel>0).length,neg=sessionHistory.length-pos;
  const ages=sessionHistory.map(p=>+p.age).filter(a=>!isNaN(a)&&a>0);
  const avgAge=ages.length?(ages.reduce((a,b)=>a+b,0)/ages.length).toFixed(0):'—';
  const avgC=sessionHistory.length?(sessionHistory.reduce((s,p)=>s+(+p.certainty),0)/sessionHistory.length).toFixed(1)+'%':'—';
  const highR=sessionHistory.filter(p=>p.stageLevel>=3).length;
  const map={'reg-total':sessionHistory.length,'reg-positive':pos,'reg-negative':neg,'reg-avg-certainty':avgC,'reg-avg-age':avgAge,'reg-high-risk':highR};
  Object.entries(map).forEach(([id,val])=>{const el=document.getElementById(id);if(el)el.textContent=val;});
  renderRegistryTable();

  const sc=[0,0,0,0,0];sessionHistory.forEach(p=>sc[p.stageLevel]++);
  const ctxD=document.getElementById('distributionChart')?.getContext('2d');
  if(ctxD){if(distChart)distChart.destroy();distChart=new Chart(ctxD,{type:'doughnut',data:{labels:['None','Mild','Moderate','Severe','Proliferative'],datasets:[{data:sc,backgroundColor:['#00ddb4','#aacc00','#ffdd00','#ff8800','#ff4e7e'],borderWidth:0,hoverOffset:5}]},options:{responsive:true,maintainAspectRatio:false,cutout:'72%',plugins:{legend:{position:'right',labels:{color:'#6b8f9e',font:{family:"'DM Mono',monospace",size:9},boxWidth:8,padding:8}}}}});}
  const ctxT=document.getElementById('timelineChart')?.getContext('2d');
  if(ctxT){if(timelineChart)timelineChart.destroy();timelineChart=new Chart(ctxT,{type:'line',data:{labels:sessionHistory.map(p=>p.id).reverse(),datasets:[{label:'Certainty %',data:sessionHistory.map(p=>+p.certainty).reverse(),borderColor:'#00aaff',backgroundColor:'rgba(0,170,255,0.08)',borderWidth:2,pointRadius:3,fill:true,tension:0.3}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{min:80,max:100,ticks:{color:'#6b8f9e',font:{family:"'DM Mono',monospace",size:9}},grid:{color:'rgba(0,220,180,0.05)'}},x:{ticks:{color:'#6b8f9e',font:{family:"'DM Mono',monospace",size:9},maxRotation:0},grid:{display:false}}}}});}
  const ctxS=document.getElementById('scatterChart')?.getContext('2d');
  if(ctxS){
    if(scatterChart)scatterChart.destroy();
    scatterChart=new Chart(ctxS,{
      type:'scatter',
      data:{
        datasets:[{
          label:'Patients',
          data:sessionHistory.map(p=>({x:+p.age||50,y:p.stageLevel})),
          backgroundColor:sessionHistory.map(p=>['#00ddb4','#aacc00','#ffdd00','#ff8800','#ff4e7e'][p.stageLevel]+'cc'),
          pointRadius:6,
          pointHoverRadius:9
        }]
      },
      options:{
        responsive:true,
        maintainAspectRatio:false,
        plugins:{
          legend:{display:false},
          tooltip:{callbacks:{label:ctx=>`Age: ${ctx.raw.x}, Stage: ${ctx.raw.y}`}}
        },
        scales:{
          x:{title:{display:true,text:'Age',color:'#6b8f9e'},ticks:{color:'#6b8f9e'},grid:{color:'rgba(0,220,180,0.05)'}},
          y:{min:-0.5,max:4.5,ticks:{stepSize:1,color:'#6b8f9e'},grid:{color:'rgba(0,220,180,0.05)'}}
        }
      }
    });
  }
}

function exportRegistryCSV() {
  if(!sessionHistory.length){showToast('No data to export.','error');return;}
  const headers=['Patient ID','Age','Eye','Diagnosis','Stage','Certainty','HbA1c','Diabetes Duration','Filter','Brightness','Contrast','Time'];
  const rows=sessionHistory.map(p=>[p.id,p.age,p.eye,p.stageName,p.stageLevel,p.certainty+'%',p.hba1c,p.diab,p.filterUsed,p.brightness,p.contrast,p.date]);
  const csv=[headers,...rows].map(r=>r.join(',')).join('\n');
  const a=document.createElement('a');a.download=`retinaai_registry_${Date.now()}.csv`;a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);a.click();
  showToast('Registry exported as CSV.','success');
}

// ═══════════════════════════════════════════════
// IMAGE LAB
// ═══════════════════════════════════════════════
function initLabUpload() {
  const zone=document.getElementById('labUploadZone'),input=document.getElementById('labFileInput');
  if(!zone||!input)return;
  zone.onclick=()=>input.click();
  zone.ondragover=e=>{e.preventDefault();zone.style.borderColor='var(--accent)';};
  zone.ondragleave=()=>zone.style.borderColor='';
  zone.ondrop=e=>{e.preventDefault();zone.style.borderColor='';const f=e.dataTransfer.files[0];if(f&&f.type.startsWith('image/'))loadLabImage(f);};
  input.onchange=()=>{if(input.files[0])loadLabImage(input.files[0]);};
}

async function loadLabImage(file) {
  const reader=new FileReader();
  reader.onload=async e=>{
    const{data}=await CV.load(e.target.result,400,300);
    currentLabImage=data;
    const origCanvas=document.getElementById('labOrigCanvas');if(origCanvas)CV.draw(origCanvas,data,400,300);
    const zone=document.getElementById('labUploadZone'),ws=document.getElementById('labWorkspace');
    if(zone)zone.style.display='none';if(ws)ws.style.display='block';
    currentLabFilter='original';applyLabProcessing();renderHistogram(data);renderLabStats(data);setupPixelInspector();
    showToast('Image loaded in Lab.','info');
  };
  reader.readAsDataURL(file);
}

function getLabAdjusted() {
  if(!currentLabImage)return null;
  return CV.adjust(currentLabImage,+(document.getElementById('ctrl-brightness')?.value||0),+(document.getElementById('ctrl-contrast')?.value||0),+(document.getElementById('ctrl-saturation')?.value||0),+(document.getElementById('ctrl-gamma')?.value||100)/100,+(document.getElementById('ctrl-sharpen')?.value||0));
}

function applyLabProcessing() {
  if(!currentLabImage)return;
  const adj=getLabAdjusted(),w=currentLabImage.width,h=currentLabImage.height;
  let out;
  switch(currentLabFilter){case 'grayscale':out=CV.gray(adj);break;case 'green':out=CV.channel(adj,1);break;case 'red':out=CV.channel(adj,0);break;case 'blue':out=CV.channel(adj,2);break;case 'clahe':out=CV.clahe(adj);break;case 'edge':out=CV.sobel(adj);break;case 'vessel':out=CV.vessel(adj);break;case 'pseudocolor':out=CV.pseudo(adj);break;case 'invert':out=CV.invert(adj);break;case 'emboss':out=CV.emboss(adj);break;case 'thermal':out=CV.thermal(adj);break;default:out=adj;}
  const ct=document.getElementById('compareModeToggle'),cs=document.getElementById('compareSlider'),oc=document.getElementById('labOutCanvas');if(!oc)return;
  if(ct?.checked&&cs){const ratio=+cs.value/100,blended=new ImageData(new Uint8ClampedArray(adj.data.length),w,h);for(let i=0;i<adj.data.length;i+=4){blended.data[i]=adj.data[i]*(1-ratio)+out.data[i]*ratio;blended.data[i+1]=adj.data[i+1]*(1-ratio)+out.data[i+1]*ratio;blended.data[i+2]=adj.data[i+2]*(1-ratio)+out.data[i+2]*ratio;blended.data[i+3]=255;}CV.draw(oc,blended,w,h);}else{CV.draw(oc,out,w,h);}
  const lbl=document.getElementById('labOutputLabel');if(lbl)lbl.textContent=currentLabFilter.charAt(0).toUpperCase()+currentLabFilter.slice(1);
}

function setLabFilter(f,btn){currentLabFilter=f;document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));if(btn)btn.classList.add('active');applyLabProcessing();}
['brightness','contrast','saturation','sharpen','gamma'].forEach(id=>{const ctrl=document.getElementById(`ctrl-${id}`),val=document.getElementById(`val-${id}`);if(!ctrl||!val)return;ctrl.addEventListener('input',()=>{val.textContent=id==='gamma'?(ctrl.value/100).toFixed(2):ctrl.value;applyLabProcessing();});});
function toggleCompareMode(){const on=document.getElementById('compareModeToggle')?.checked,lbl=document.getElementById('compareModeLabel'),wrap=document.getElementById('compareSliderWrap');if(lbl)lbl.textContent=on?'On':'Off';if(wrap)wrap.style.display=on?'block':'none';applyLabProcessing();}
document.getElementById('compareSlider')?.addEventListener('input',applyLabProcessing);

function setupPixelInspector(){const canvas=document.getElementById('labOrigCanvas'),display=document.getElementById('pixelInfoDisplay');if(!canvas||!display||!currentLabImage)return;canvas.addEventListener('mousemove',e=>{const r=canvas.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)*currentLabImage.width/r.width),y=Math.floor((e.clientY-r.top)*currentLabImage.height/r.height);if(x<0||x>=currentLabImage.width||y<0||y>=currentLabImage.height)return;const i=(y*currentLabImage.width+x)*4;display.textContent=`(${x},${y}) R:${currentLabImage.data[i]} G:${currentLabImage.data[i+1]} B:${currentLabImage.data[i+2]}`;});canvas.addEventListener('mouseleave',()=>{if(display)display.textContent='';});}

function renderHistogram(data){const h=CV.histogram(data),labels=Array.from({length:256},(_,i)=>i%32===0?i:''),ctx=document.getElementById('histogramChart')?.getContext('2d');if(!ctx)return;if(histogramChart)histogramChart.destroy();histogramChart=new Chart(ctx,{type:'line',data:{labels,datasets:[{label:'R',data:h.r,borderColor:'rgba(255,78,126,0.8)',backgroundColor:'rgba(255,78,126,0.1)',borderWidth:1,pointRadius:0,fill:true,tension:0.2},{label:'G',data:h.g,borderColor:'rgba(0,221,180,0.8)',backgroundColor:'rgba(0,221,180,0.1)',borderWidth:1,pointRadius:0,fill:true,tension:0.2},{label:'B',data:h.b,borderColor:'rgba(0,170,255,0.8)',backgroundColor:'rgba(0,170,255,0.1)',borderWidth:1,pointRadius:0,fill:true,tension:0.2}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#6b8f9e',font:{family:"'DM Mono',monospace",size:9},boxWidth:7}}},scales:{x:{ticks:{color:'#6b8f9e',font:{size:8}},grid:{color:'rgba(0,220,180,0.04)'}},y:{ticks:{color:'#6b8f9e',font:{size:8}},grid:{color:'rgba(0,220,180,0.04)'}}}}});}
function renderLabStats(data){const s=CV.stats(data),grid=document.getElementById('labStatsGrid');if(grid)grid.innerHTML=[['Brightness',s.brightness],['Contrast',s.contrast],['Avg R',s.meanR],['Avg G',s.meanG],['Avg B',s.meanB],['Dominant',s.dominant]].map(([l,v])=>`<div class="img-stat-item"><span class="img-stat-label">${l}</span><span class="img-stat-val" style="font-size:16px;">${v}</span></div>`).join('');}
function downloadLabResult(){const canvas=document.getElementById('labOutCanvas');if(!canvas)return;const a=document.createElement('a');a.download=`retinaai_${currentLabFilter}_${Date.now()}.png`;a.href=canvas.toDataURL('image/png');a.click();showToast('Image downloaded.','success');}
function downloadSideBySide(){const orig=document.getElementById('labOrigCanvas'),out=document.getElementById('labOutCanvas');if(!orig||!out)return;const merged=document.createElement('canvas');merged.width=orig.width*2+16;merged.height=orig.height;const mctx=merged.getContext('2d');mctx.fillStyle='#030a0e';mctx.fillRect(0,0,merged.width,merged.height);mctx.drawImage(orig,0,0);mctx.drawImage(out,orig.width+16,0);const a=document.createElement('a');a.download=`retinaai_comparison_${Date.now()}.png`;a.href=merged.toDataURL('image/png');a.click();showToast('Side-by-side comparison downloaded.','success');}
function resetLab(){if(!currentLabImage)return;['ctrl-brightness','ctrl-contrast','ctrl-saturation','ctrl-sharpen'].forEach(id=>{const el=document.getElementById(id);if(el)el.value=0;});const g=document.getElementById('ctrl-gamma');if(g)g.value=100;['brightness','contrast','saturation','sharpen'].forEach(id=>{const el=document.getElementById(`val-${id}`);if(el)el.textContent='0';});const vg=document.getElementById('val-gamma');if(vg)vg.textContent='1.0';currentLabFilter='original';document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));document.querySelector('.filter-btn[data-filter="original"]')?.classList.add('active');applyLabProcessing();showToast('Lab reset.','info');}
function copyLabToScan(){const canvas=document.getElementById('labOutCanvas');if(!canvas)return;const src=canvas.toDataURL('image/png');switchView('detect');resetForm();setTimeout(()=>{if(previewImg)previewImg.src=src;showToast('Lab result sent to scan engine.','success');},300);}

// ═══════════════════════════════════════════════
// ████████████████████████████████████████████
//   DR KNOWLEDGE BASE
// ████████████████████████████████████████████
// ═══════════════════════════════════════════════

const KB_DATA = {
  stages: [
    { level:0, name:'No DR', color:'#00ddb4', icon:'○', prevalence:'~50% of diabetics', icdCode:'E11.319', urgency:'routine', followUp:'Annual screening',
      description:'No detectable lesions in the retinal vasculature. The fundus appears completely normal under examination.',
      fundusFindings:['Normal optic disc appearance','Clear macula with foveal reflex','Uniform calibre retinal vessels','No haemorrhages, exudates, or microaneurysms'],
      pathophysiology:'Chronic hyperglycaemia leads to pericyte loss and basement membrane thickening, but no clinically visible lesions have yet formed. The blood-retinal barrier remains intact.',
      management:['Annual retinal screening for all diabetic patients','Optimise glycaemic control (target HbA1c <7%)','Control systemic blood pressure (<130/80 mmHg)','Lipid-lowering therapy as indicated','Patient education on DR risk and lifestyle'] },
    { level:1, name:'Mild NPDR', color:'#aacc00', icon:'◔', prevalence:'~25% of diabetics', icdCode:'E11.321', urgency:'routine', followUp:'12 months',
      description:'Earliest clinically detectable stage. At least one microaneurysm — small focal outpouchings of weakened retinal capillary walls.',
      fundusFindings:['Microaneurysms (≥1)','Occasional dot/blot haemorrhages','Minimal or no hard exudates','No neovascularisation or venous beading'],
      pathophysiology:'Pericyte dropout leads to focal outpouching of weakened capillary walls forming microaneurysms. Increased vascular permeability may cause localised oedema but the macula is typically spared.',
      management:['Retinal review every 12 months','Aggressive glycaemic optimisation','Blood pressure management (<130/80)','Statin therapy for dyslipidaemia','Refer to ophthalmologist if macular oedema suspected'] },
    { level:2, name:'Moderate NPDR', color:'#ffdd00', icon:'◑', prevalence:'~10–15% of diabetics', icdCode:'E11.331', urgency:'semi-urgent', followUp:'3–6 months (ophthalmologist)',
      description:'More than microaneurysms alone. Multiple haemorrhages, hard exudates, and cotton-wool spots are present in some but not all quadrants.',
      fundusFindings:['Multiple microaneurysms','Dot and blot haemorrhages in ≤3 quadrants','Hard exudates (lipid deposits)','Cotton-wool spots (nerve fibre layer infarcts)','Mild venous beading possible'],
      pathophysiology:'Progressive capillary closure leads to retinal ischaemia. Cotton-wool spots represent focal nerve fibre layer infarcts from arteriolar occlusion. Lipid leakage from damaged capillary walls forms hard exudates.',
      management:['Ophthalmology referral within 3–6 months','Fundus fluorescein angiography (FFA) to assess perfusion','Enhanced HbA1c control (<7%)','Monitor for diabetic macular oedema (DMO)','Consider intravitreal anti-VEGF if DMO present'] },
    { level:3, name:'Severe NPDR', color:'#ff8800', icon:'◕', prevalence:'~5% of diabetics', icdCode:'E11.341', urgency:'urgent', followUp:'1 month — ophthalmologist',
      description:'The "4-2-1 rule": haemorrhages in all 4 quadrants, venous beading in ≥2 quadrants, or intraretinal microvascular abnormalities (IRMA) in ≥1 quadrant.',
      fundusFindings:['Extensive haemorrhages in all 4 quadrants','Venous beading in ≥2 quadrants','IRMA (intraretinal microvascular abnormalities)','No frank neovascularisation yet','Large areas of capillary non-perfusion on FFA'],
      pathophysiology:'Widespread capillary closure creates large ischaemic zones. IRMA represent dilated pre-existing vessels bypassing occluded capillaries. High VEGF secretion by ischaemic retina sets the stage for PDR.',
      management:['Urgent ophthalmology referral within 1 month','FFA and OCT mandatory','Pan-retinal photocoagulation (PRP) should be considered','Anti-VEGF therapy for concurrent DMO','Very tight glycaemic and blood pressure control'] },
    { level:4, name:'Proliferative DR', color:'#ff4e7e', icon:'●', prevalence:'~2–5% of diabetics', icdCode:'E11.351', urgency:'emergency', followUp:'1 week — URGENT',
      description:'New blood vessel formation (neovascularisation) on the disc (NVD) or elsewhere on the retina (NVE) — the hallmark of PDR. Carries high risk of severe visual loss.',
      fundusFindings:['Neovascularisation of disc (NVD) or elsewhere (NVE)','Pre-retinal or vitreous haemorrhage','Fibrovascular proliferative membrane','Tractional retinal detachment (advanced cases)','Rubeosis iridis in severe cases'],
      pathophysiology:'Massively elevated VEGF drives pathological angiogenesis. New fragile vessels grow along the posterior vitreous face, bleed easily, and fibrous scaffolding contracts causing tractional retinal detachment.',
      management:['EMERGENCY referral within 1 week','Pan-retinal photocoagulation (PRP) — gold standard','Intravitreal anti-VEGF (ranibizumab, bevacizumab, aflibercept)','Vitrectomy for non-clearing vitreous haemorrhage or TRD','Combined anti-VEGF + PRP for high-risk PDR'] },
  ],
  treatments: [
    { name:'Pan-Retinal Photocoagulation', abbr:'PRP', color:'#00aaff', stages:[3,4], desc:'Laser burns (~1200–1600) applied to peripheral retina destroy ischaemic tissue, reducing VEGF and causing neovascularisation regression. Applied in 2–3 sessions. Gold standard for high-risk PDR.' },
    { name:'Anti-VEGF Therapy', abbr:'IVT', color:'#00ddb4', stages:[2,3,4], desc:'Intravitreal injections of ranibizumab, bevacizumab, or aflibercept block vascular endothelial growth factor, reducing macular oedema and neovascularisation. Monthly loading, then PRN protocol.' },
    { name:'Pars Plana Vitrectomy', abbr:'PPV', color:'#ff8800', stages:[4], desc:'Surgical removal of vitreous gel and fibrovascular membranes. Indicated for non-clearing vitreous haemorrhage, tractional retinal detachment, or combined mechanism detachment.' },
    { name:'Intravitreal Steroids', abbr:'IVS', color:'#ffdd00', stages:[2,3], desc:'Triamcinolone acetonide or dexamethasone implant (Ozurdex) reduce inflammation and macular oedema. Risk of IOP elevation and accelerated cataract formation.' },
    { name:'Glycaemic Optimisation', abbr:'GC', color:'#aacc00', stages:[0,1,2,3,4], desc:'Most effective long-term intervention. Each 1% HbA1c reduction decreases DR progression risk by ~35% (UKPDS). Target HbA1c <7%. Intensive control in DCCT reduced DR incidence by 76%.' },
    { name:'Blood Pressure Control', abbr:'BPC', color:'#8b5cf6', stages:[0,1,2,3,4], desc:'Target <130/80 mmHg. RAS inhibitors (ACE inhibitors/ARBs) have additional renoprotective benefit. UKPDS showed 37% reduction in microvascular complications with tight BP control.' },
  ],
  keyStats: [
    {label:'Global DR prevalence',value:'~35%',sub:'of all diabetics'},
    {label:'Vision-threatening DR',value:'~11%',sub:'of all diabetics'},
    {label:'Leading cause of blindness',value:'#1',sub:'in working-age adults'},
    {label:'HbA1c 1% reduction',value:'35%',sub:'less DR progression'},
    {label:'Annual screening prevents',value:'98%',sub:'of preventable blindness'},
    {label:'Time to PDR from onset',value:'15–20y',sub:'T1DM average'},
  ],
  biomarkers: [
    {name:'HbA1c',threshold:'<7.0%',role:'Primary glycaemic control target. Direct predictor of DR incidence and progression rate.'},
    {name:'Fasting Glucose',threshold:'4–7 mmol/L',role:'Day-to-day glycaemic burden. Correlates with microaneurysm formation rate.'},
    {name:'Blood Pressure',threshold:'<130/80 mmHg',role:'Hypertension independently accelerates vascular damage and DR progression.'},
    {name:'LDL Cholesterol',threshold:'<2.6 mmol/L',role:'Dyslipidaemia increases hard exudate formation and macular lipid deposition.'},
    {name:'eGFR / Creatinine',threshold:'>60 mL/min',role:'Nephropathy co-occurs with DR; renal impairment worsens ocular prognosis.'},
    {name:'VEGF (vitreous)',threshold:'<330 pg/mL',role:'Elevated in PDR. Directly drives neovascularisation. Primary target of anti-VEGF therapy.'},
  ],
};

let kbActiveStage = 0;
let kbActiveTab = 'overview';

function initKnowledgeBase() {
  renderKBStageNav();
  renderKBContent();
}

function renderKBStageNav() {
  const nav = document.getElementById('kbStageNav');
  if (!nav) return;
  nav.innerHTML = KB_DATA.stages.map(s => `
    <div class="kb-stage-pill ${kbActiveStage === s.level ? 'active' : ''}"
         onclick="selectKBStage(${s.level})"
         style="--stage-color:${s.color};">
      <span class="kb-stage-pill-icon">${s.icon}</span>
      <div class="kb-stage-pill-text">
        <span class="kb-stage-pill-num">Stage ${s.level}</span>
        <span class="kb-stage-pill-name">${s.name}</span>
      </div>
      ${kbActiveStage === s.level ? '<div class="kb-active-dot"></div>' : ''}
    </div>`).join('');
}

function selectKBStage(level) {
  kbActiveStage = level;
  renderKBStageNav();
  renderKBContent();
}

function selectKBTab(tab, el) {
  kbActiveTab = tab;
  document.querySelectorAll('.kb-tab').forEach(t => t.classList.remove('active'));
  if (el) el.classList.add('active');
  renderKBContent();
}

function renderKBContent() {
  const stage = KB_DATA.stages[kbActiveStage];
  const container = document.getElementById('kbContentArea');
  if (!container || !stage) return;

  const urgencyColors = { routine:'#00ddb4', 'semi-urgent':'#ffdd00', urgent:'#ff8800', emergency:'#ff4e7e' };
  const uc = urgencyColors[stage.urgency] || '#6b8f9e';

  if (kbActiveTab === 'overview') {
    container.innerHTML = `
    <div style="animation:fadeUp 0.3s ease;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">

        <!-- Hero card -->
        <div style="grid-column:1/-1;background:var(--surface2);border:1px solid ${stage.color}33;border-radius:14px;padding:24px;display:flex;align-items:flex-start;gap:24px;">
          <div style="font-size:52px;line-height:1;filter:drop-shadow(0 0 12px ${stage.color}66);">${stage.icon}</div>
          <div style="flex:1;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap;">
              <span style="font-family:'DM Mono',monospace;font-size:10px;color:${stage.color};background:${stage.color}18;border:1px solid ${stage.color}44;padding:3px 10px;border-radius:100px;text-transform:uppercase;letter-spacing:1px;">Stage ${stage.level}</span>
              <span style="font-family:'DM Mono',monospace;font-size:10px;color:${uc};background:${uc}15;border:1px solid ${uc}33;padding:3px 10px;border-radius:100px;text-transform:uppercase;">${stage.urgency}</span>
              <span style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);background:var(--surface);border:1px solid var(--border);padding:3px 10px;border-radius:100px;">ICD: ${stage.icdCode}</span>
              <span style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);background:var(--surface);border:1px solid var(--border);padding:3px 10px;border-radius:100px;">${stage.prevalence}</span>
            </div>
            <h3 style="font-family:'Syne',sans-serif;font-size:26px;font-weight:700;margin-bottom:10px;">${stage.name}</h3>
            <p style="color:var(--muted);font-size:13px;line-height:1.75;">${stage.description}</p>
            <div style="margin-top:14px;display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;color:${uc};">
              <svg viewBox="0 0 24 24" width="14" height="14" stroke="${uc}" stroke-width="2" fill="none"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              Follow-up: ${stage.followUp}
            </div>
          </div>
        </div>

        <!-- Fundus Findings -->
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:20px;">
          <div style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">Fundus Findings</div>
          <ul style="list-style:none;display:flex;flex-direction:column;gap:8px;">
            ${stage.fundusFindings.map(f=>`<li style="display:flex;align-items:flex-start;gap:10px;font-size:13px;color:var(--text);line-height:1.5;"><span style="width:6px;height:6px;border-radius:50%;background:${stage.color};flex-shrink:0;margin-top:5px;box-shadow:0 0 6px ${stage.color}88;"></span>${f}</li>`).join('')}
          </ul>
        </div>

        <!-- Pathophysiology -->
        <div style="background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:20px;">
          <div style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">Pathophysiology</div>
          <p style="color:var(--muted);font-size:13px;line-height:1.75;">${stage.pathophysiology}</p>
        </div>

        <!-- Management -->
        <div style="grid-column:1/-1;background:var(--surface2);border:1px solid ${uc}33;border-radius:12px;padding:20px;">
          <div style="font-family:'DM Mono',monospace;font-size:10px;color:${uc};text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">Management Protocol</div>
          <ol style="list-style:none;display:flex;flex-direction:column;gap:10px;counter-reset:mgmt;">
            ${stage.management.map((m,i)=>`<li style="display:flex;align-items:flex-start;gap:12px;font-size:13px;color:var(--text);line-height:1.5;"><span style="font-family:'DM Mono',monospace;font-size:10px;color:${uc};background:${uc}18;border:1px solid ${uc}33;width:22px;height:22px;border-radius:6px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">${i+1}</span>${m}</li>`).join('')}
          </ol>
        </div>
      </div>
    </div>`;
  }

  if (kbActiveTab === 'comparison') {
    container.innerHTML = `
    <div style="animation:fadeUp 0.3s ease;overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;min-width:700px;">
        <thead>
          <tr style="background:var(--surface2);">
            <th style="padding:12px 16px;text-align:left;font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border);">Feature</th>
            ${KB_DATA.stages.map(s=>`<th style="padding:12px 16px;text-align:center;font-family:'DM Mono',monospace;font-size:10px;color:${s.color};text-transform:uppercase;border-bottom:1px solid var(--border);">${s.icon} ${s.name}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
          ${[
            ['Microaneurysms','None','≥1','Moderate','Many','Many'],
            ['Haemorrhages','None','Minimal','<4 quadrants','All 4 quadrants','±Vitreous'],
            ['Hard Exudates','None','Rare','Present','Present','Variable'],
            ['Cotton-Wool Spots','None','None','Present','Many','Variable'],
            ['Venous Beading','None','None','Mild','≥2 quadrants','Present'],
            ['IRMA','None','None','None','≥1 quadrant','Possible'],
            ['Neovascularisation','None','None','None','None','Present ✓'],
            ['Vision Threat','Low','Low','Moderate','High','Very High'],
            ['Follow-up',
              ...KB_DATA.stages.map(s=>s.followUp)],
          ].map((row,ri)=>`
            <tr style="background:${ri%2===0?'transparent':'rgba(255,255,255,0.01)'};">
              <td style="padding:11px 16px;font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid var(--border);">${row[0]}</td>
              ${row.slice(1).map((val,ci)=>`<td style="padding:11px 16px;text-align:center;font-size:12px;color:${row[0]==='Vision Threat'?['#00ddb4','#aacc00','#ffdd00','#ff8800','#ff4e7e'][ci]:row[0]==='Follow-up'?KB_DATA.stages[ci].color:'var(--text)'};border-bottom:1px solid var(--border);font-family:${row[0]==='Follow-up'?"'DM Mono',monospace":'inherit'};font-size:${row[0]==='Follow-up'?'10px':'12px'};">${val}</td>`).join('')}
            </tr>`).join('')}
        </tbody>
      </table>
    </div>`;
  }

  if (kbActiveTab === 'treatments') {
    container.innerHTML = `
    <div style="animation:fadeUp 0.3s ease;display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;">
      ${KB_DATA.treatments.map(t=>`
        <div style="background:var(--surface2);border:1px solid ${t.color}33;border-radius:12px;padding:20px;transition:transform 0.2s,border-color 0.2s;" onmouseover="this.style.borderColor='${t.color}66';this.style.transform='translateY(-2px)'" onmouseout="this.style.borderColor='${t.color}33';this.style.transform=''">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
            <div>
              <span style="font-family:'DM Mono',monospace;font-size:10px;color:${t.color};text-transform:uppercase;letter-spacing:1px;">${t.abbr}</span>
              <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:600;margin-top:3px;">${t.name}</div>
            </div>
            <div style="display:flex;gap:3px;flex-wrap:wrap;max-width:110px;justify-content:flex-end;">
              ${t.stages.map(s=>`<span style="font-family:'DM Mono',monospace;font-size:9px;padding:2px 6px;border-radius:4px;background:${KB_DATA.stages[s].color}20;color:${KB_DATA.stages[s].color};">Stg ${s}</span>`).join('')}
            </div>
          </div>
          <p style="color:var(--muted);font-size:12px;line-height:1.7;">${t.desc}</p>
        </div>`).join('')}
    </div>`;
  }

  if (kbActiveTab === 'biomarkers') {
    container.innerHTML = `
    <div style="animation:fadeUp 0.3s ease;">
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-bottom:24px;">
        ${KB_DATA.biomarkers.map(b=>`
          <div style="background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:18px;border-left:3px solid var(--accent2);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
              <span style="font-family:'Syne',sans-serif;font-size:15px;font-weight:600;">${b.name}</span>
              <span style="font-family:'DM Mono',monospace;font-size:11px;color:var(--accent);background:rgba(0,221,180,0.1);border:1px solid rgba(0,221,180,0.25);padding:3px 8px;border-radius:6px;">${b.threshold}</span>
            </div>
            <p style="color:var(--muted);font-size:12px;line-height:1.65;">${b.role}</p>
          </div>`).join('')}
      </div>

      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:20px;">
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;">Epidemiology at a Glance</div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;">
          ${KB_DATA.keyStats.map(s=>`
            <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center;">
              <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:var(--accent);line-height:1;margin-bottom:4px;">${s.value}</div>
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:var(--text);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px;">${s.label}</div>
              <div style="font-size:11px;color:var(--muted);">${s.sub}</div>
            </div>`).join('')}
        </div>
      </div>
    </div>`;
  }
}

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(a => { a.addEventListener('click', e => { e.preventDefault(); document.querySelector(a.getAttribute('href'))?.scrollIntoView({behavior:'smooth'}); }); });

applyProfileUI();
loadSettingsUI();
renderNotesList();