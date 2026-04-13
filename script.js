let patientChart = null; // Global reference for Chart.js

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

// Upload logic elements
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
  'Preprocessing image...', 'Extracting features...', 'Running EfficientNetB0...', 'Grading severity...'
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

    // Simulate model inference (3 seconds)
    setTimeout(() => {
      clearInterval(msgInterval);
      loadingOverlay.style.display = 'none';
      showResult();
    }, 3000);
  };
  reader.readAsDataURL(file);
}

// Result Generation with Advanced Clinical Features
function showResult() {
  const patId = document.getElementById('pat-id').value || 'Unknown';
  const patAge = document.getElementById('pat-age').value || 'N/A';
  const patEye = document.getElementById('pat-eye').value;

  document.getElementById('patient-summary').innerHTML = `
    <span><strong>ID:</strong> ${patId}</span>
    <span><strong>Age:</strong> ${patAge}</span>
    <span><strong>Eye:</strong> ${patEye}</span>
    <span><strong>Date:</strong> ${new Date().toLocaleDateString()}</span>
  `;

  const severityStages = [
    { level: 0, name: "No DR", color: "badge-negative", pos: 0 },
    { level: 1, name: "Mild NPDR", color: "badge-positive", pos: 25 },
    { level: 2, name: "Moderate NPDR", color: "badge-positive", pos: 50 },
    { level: 3, name: "Severe NPDR", color: "badge-positive", pos: 75 },
    { level: 4, name: "Proliferative DR", color: "badge-positive", pos: 100 }
  ];
  
  const stage = severityStages[Math.floor(Math.random() * severityStages.length)];
  const networkCertainty = (Math.random() * (99.9 - 85.0) + 85.0).toFixed(1);
  const lesionDensity = (stage.level * 22 + Math.random() * 10).toFixed(1);

  document.getElementById('result-label').textContent = stage.name;
  const badge = document.getElementById('result-badge');
  badge.textContent = stage.level === 0 ? 'Negative' : 'Refer to Ophthalmologist';
  badge.className = 'result-badge ' + stage.color;
  
  document.getElementById('severity-val').textContent = `Stage ${stage.level}`;
  setTimeout(() => { document.getElementById('severity-indicator').style.left = stage.pos + '%'; }, 100);
  
  // --- RADIAL METRICS ANIMATION ---
  document.getElementById('m-prob').textContent = networkCertainty + '%';
  document.getElementById('m-conf').textContent = lesionDensity;
  document.getElementById('m-time').textContent = (Math.random() * 0.3 + 0.1).toFixed(2) + 's';

  // Calculate SVG stroke offsets (Circumference is ~264)
  setTimeout(() => {
    const certOffset = 264 - (264 * (networkCertainty / 100));
    const lesionOffset = 264 - (264 * (Math.min(lesionDensity, 100) / 100));
    document.getElementById('cert-ring').style.strokeDashoffset = certOffset;
    document.getElementById('lesion-ring').style.strokeDashoffset = lesionOffset;
  }, 200);

  // --- CHART.JS INITIALIZATION ---
  const ctx = document.getElementById('progressionChart').getContext('2d');
  
  if (patientChart) patientChart.destroy(); // Destroy old chart if doing a new scan
  
  // Mock historical data trending towards current severity
  const historicalData = [Math.max(0, stage.level - 2), Math.max(0, stage.level - 1), Math.max(0, stage.level - 1), stage.level, stage.level];

  patientChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Jan', 'Apr', 'Jul', 'Oct', 'Current'],
      datasets: [{
        label: 'Severity Level',
        data: historicalData,
        borderColor: '#00ddb4',
        backgroundColor: 'rgba(0, 221, 180, 0.1)',
        borderWidth: 3,
        pointBackgroundColor: '#030a0e',
        pointBorderColor: '#00aaff',
        pointBorderWidth: 2,
        pointRadius: 5,
        fill: true,
        tension: 0.4 // Smooth curves
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { 
          beginAtZero: true, 
          max: 4, 
          ticks: { stepSize: 1, color: '#6b8f9e' },
          grid: { color: 'rgba(0,220,180,0.05)' }
        },
        x: { 
          ticks: { color: '#6b8f9e' },
          grid: { display: false }
        }
      }
    }
  });

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
  
  // Reset Heatmap & XAI Controls
  const overlay = document.getElementById('heatmap-overlay');
  const xaiControls = document.getElementById('xai-controls');
  const toggleBtn = document.getElementById('toggleHeatmapBtn');
  
  overlay.classList.remove('active');
  xaiControls.style.display = 'none';
  
  if(toggleBtn) {
      toggleBtn.textContent = 'Enable Explainability';
      toggleBtn.style.background = 'var(--accent)';
      toggleBtn.style.color = '#030a0e';
      toggleBtn.style.border = 'none';
  }

  document.getElementById('severity-indicator').style.left = '0%';
  document.getElementById('cert-ring').style.strokeDashoffset = 264;
  document.getElementById('lesion-ring').style.strokeDashoffset = 264;
  
  // Clear patient form
  document.getElementById('pat-id').value = '';
  document.getElementById('pat-age').value = '';
  
  // Scroll back up
  document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
}

const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
if (analyzeAnotherBtn) {
    analyzeAnotherBtn.addEventListener('click', resetForm);
}

// Advanced Grad-CAM Controls
const toggleHeatmapBtn = document.getElementById('toggleHeatmapBtn');
const xaiControls = document.getElementById('xai-controls');
const heatmapOverlay = document.getElementById('heatmap-overlay');
const heatmapSlider = document.getElementById('heatmapSlider');
const opacityVal = document.getElementById('opacity-val');

if(toggleHeatmapBtn) {
  toggleHeatmapBtn.addEventListener('click', () => {
    heatmapOverlay.classList.toggle('active');
    
    if(heatmapOverlay.classList.contains('active')) {
        toggleHeatmapBtn.textContent = 'Disable Explainability';
        toggleHeatmapBtn.style.background = 'rgba(0, 221, 180, 0.2)';
        toggleHeatmapBtn.style.color = '#00ddb4';
        toggleHeatmapBtn.style.border = '1px solid #00ddb4';
        xaiControls.style.display = 'block';
        
        // Apply slider value immediately
        heatmapOverlay.style.opacity = heatmapSlider.value / 100;
    } else {
        toggleHeatmapBtn.textContent = 'Enable Explainability';
        toggleHeatmapBtn.style.background = 'var(--accent)';
        toggleHeatmapBtn.style.color = '#030a0e';
        toggleHeatmapBtn.style.border = 'none';
        xaiControls.style.display = 'none';
        heatmapOverlay.style.opacity = 0;
    }
  });
}

// Live Opacity Updates for Heatmap
if(heatmapSlider) {
  heatmapSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    opacityVal.textContent = val + '%';
    if(heatmapOverlay.classList.contains('active')) {
      heatmapOverlay.style.opacity = val / 100;
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