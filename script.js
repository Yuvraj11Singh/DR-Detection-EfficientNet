// Counter animation
function animateCount(el, target, suffix, decimals) {
  let start = 0;
  const duration = 1800;
  const step = timestamp => {
    if (!start) start = timestamp;
    const progress = Math.min((timestamp - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const val = ease * target;
    el.textContent = (decimals ? val.toFixed(decimals) : Math.round(val).toLocaleString()) + (suffix || '');
    if (progress < 1) requestAnimationFrame(step);
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
statsObserver.observe(document.querySelector('.stats-strip'));

// Scroll reveal
const revealObserver = new IntersectionObserver(entries => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = (i * 0.08) + 's';
      e.target.classList.add('visible');
    }
  });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

// Upload logic
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewWrap = document.getElementById('preview-wrap');
const previewImg = document.getElementById('preview-img');
const resultCard = document.getElementById('result-card');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

// Ensure button and div click events are separated to avoid inline onclick attributes
uploadArea.addEventListener('click', () => fileInput.click());

const chooseFileBtn = document.getElementById('chooseFileBtn');
if (chooseFileBtn) {
  chooseFileBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    fileInput.click();
  });
}

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault(); uploadArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) processFile(file);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) processFile(fileInput.files[0]); });

const loadingMessages = [
  'Preprocessing image...', 'Extracting features...', 'Running EfficientNetB0...', 'Generating confidence scores...'
];

function processFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    uploadArea.style.display = 'none';
    previewWrap.style.display = 'block';
    loadingOverlay.style.display = 'block';
    resultCard.style.display = 'none';

    let msgIdx = 0;
    const msgInterval = setInterval(() => {
      msgIdx = (msgIdx + 1) % loadingMessages.length;
      loadingText.textContent = loadingMessages[msgIdx];
    }, 700);

    // Simulate model inference (replace with real API call)
    setTimeout(() => {
      clearInterval(msgInterval);
      loadingOverlay.style.display = 'none';
      showResult();
    }, 3000);
  };
  reader.readAsDataURL(file);
}

function showResult() {
  // This is demo output — replace with actual backend response
  const prob = Math.random();
  const isPositive = prob > 0.5;
  const confidence = isPositive ? prob : (1 - prob);
  const confPct = (confidence * 100).toFixed(1);

  document.getElementById('result-label').textContent = isPositive ? 'DR Detected' : 'No DR Detected';
  const badge = document.getElementById('result-badge');
  badge.textContent = isPositive ? 'Positive' : 'Negative';
  badge.className = 'result-badge ' + (isPositive ? 'badge-positive' : 'badge-negative');
  document.getElementById('conf-val').textContent = confPct + '%';
  document.getElementById('conf-bar').style.width = confPct + '%';
  document.getElementById('m-prob').textContent = (prob * 100).toFixed(1) + '%';
  document.getElementById('m-conf').textContent = confPct + '%';
  document.getElementById('m-time').textContent = (Math.random() * 0.4 + 0.1).toFixed(2) + 's';

  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function resetForm() {
  uploadArea.style.display = 'block';
  previewWrap.style.display = 'none';
  resultCard.style.display = 'none';
  fileInput.value = '';
}

const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
if (analyzeAnotherBtn) {
    analyzeAnotherBtn.addEventListener('click', resetForm);
}

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
  });
});