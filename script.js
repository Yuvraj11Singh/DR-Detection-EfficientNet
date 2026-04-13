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

// Ensure button and div click events are separated
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
    
    // Hide form and upload area
    document.getElementById('patientForm').style.display = 'none';
    uploadArea.style.display = 'none';
    
    // Show preview and loading
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

// Result Generation with Patient Data Integration
function showResult() {
  // Capture Patient Data
  const patId = document.getElementById('pat-id').value || 'Unknown';
  const patAge = document.getElementById('pat-age').value || 'N/A';
  const patEye = document.getElementById('pat-eye').value;

  // Populate Summary
  document.getElementById('patient-summary').innerHTML = `
    <span><strong>ID:</strong> ${patId}</span>
    <span><strong>Age:</strong> ${patAge}</span>
    <span><strong>Eye:</strong> ${patEye}</span>
    <span><strong>Date:</strong> ${new Date().toLocaleDateString()}</span>
  `;

  // Demo output
  const prob = Math.random();
  const isPositive = prob > 0.5;
  const confidence = isPositive ? prob : (1 - prob);
  const confPct = (confidence * 100).toFixed(1);

  document.getElementById('result-label').textContent = isPositive ? 'DR Detected' : 'No DR Detected';
  const badge = document.getElementById('result-badge');
  badge.textContent = isPositive ? 'Positive' : 'Negative';
  badge.className = 'result-badge ' + (isPositive ? 'badge-positive' : 'badge-negative');
  
  document.getElementById('conf-val').textContent = confPct + '%';
  
  // Small timeout to allow CSS transition to happen
  setTimeout(() => {
    document.getElementById('conf-bar').style.width = confPct + '%';
  }, 100);
  
  document.getElementById('m-prob').textContent = (prob * 100).toFixed(1) + '%';
  document.getElementById('m-conf').textContent = confPct + '%';
  document.getElementById('m-time').textContent = (Math.random() * 0.4 + 0.1).toFixed(2) + 's';

  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Reset Form Logic
function resetForm() {
  document.getElementById('patientForm').style.display = 'block';
  uploadArea.style.display = 'block';
  previewWrap.style.display = 'none';
  resultCard.style.display = 'none';
  fileInput.value = '';
  
  // Reset Heatmap
  const overlay = document.getElementById('heatmap-overlay');
  overlay.classList.remove('active');
  const toggleBtn = document.getElementById('toggleHeatmapBtn');
  if(toggleBtn) {
      toggleBtn.textContent = 'Toggle Explainability (Grad-CAM)';
      toggleBtn.style.background = 'var(--accent)';
      toggleBtn.style.color = '#030a0e';
      toggleBtn.style.border = 'none';
  }

  document.getElementById('conf-bar').style.width = '0%';
  
  // Clear patient form
  document.getElementById('pat-id').value = '';
  document.getElementById('pat-age').value = '';
  
  // Scroll back up
  document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
}

// Button Listeners
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
if (analyzeAnotherBtn) {
    analyzeAnotherBtn.addEventListener('click', resetForm);
}

// Grad-CAM Toggle Logic
const toggleHeatmapBtn = document.getElementById('toggleHeatmapBtn');
if(toggleHeatmapBtn) {
  toggleHeatmapBtn.addEventListener('click', () => {
    const overlay = document.getElementById('heatmap-overlay');
    overlay.classList.toggle('active');
    
    if(overlay.classList.contains('active')) {
        toggleHeatmapBtn.textContent = 'Hide Explainability';
        toggleHeatmapBtn.style.background = 'rgba(0, 221, 180, 0.2)';
        toggleHeatmapBtn.style.color = '#00ddb4';
        toggleHeatmapBtn.style.border = '1px solid #00ddb4';
    } else {
        toggleHeatmapBtn.textContent = 'Toggle Explainability (Grad-CAM)';
        toggleHeatmapBtn.style.background = 'var(--accent)';
        toggleHeatmapBtn.style.color = '#030a0e';
        toggleHeatmapBtn.style.border = 'none';
    }
  });
}

// Download PDF Mock Logic
const downloadReportBtn = document.getElementById('downloadReportBtn');
if(downloadReportBtn) {
  downloadReportBtn.addEventListener('click', () => {
    window.print(); 
  });
}

// Smooth scroll for nav links
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
  });
});