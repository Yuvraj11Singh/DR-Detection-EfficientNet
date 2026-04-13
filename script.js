let patientChart = null;

// --- DYNAMIC SESSION DATABASE ---
// This array holds all patients scanned during the current browser session.
let sessionHistory = []; 

// UI Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewWrap = document.getElementById('preview-wrap');
const previewImg = document.getElementById('preview-img');
const resultCard = document.getElementById('result-card');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const historyList = document.getElementById('historyList');

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

const revealObserver = new IntersectionObserver(entries => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      e.target.style.transitionDelay = (i * 0.08) + 's';
      e.target.classList.add('visible');
    }
  });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

// UPLOAD LOGIC
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
  'Initializing MPS Backend...', 'Extracting vascular features...', 'Running EfficientNetB0...', 'Grading severity...'
];

function processFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    
    // Hide form and upload area
    document.getElementById('upload-flow').style.display = 'none';
    
    previewWrap.style.display = 'block';
    loadingOverlay.style.display = 'block';
    resultCard.style.display = 'none';
    previewImg.style.display = 'none'; // Hide image until loaded

    let msgIdx = 0;
    const msgInterval = setInterval(() => {
      msgIdx = (msgIdx + 1) % loadingMessages.length;
      loadingText.textContent = loadingMessages[msgIdx];
    }, 700);

    // Simulate model inference
    setTimeout(() => {
      clearInterval(msgInterval);
      loadingOverlay.style.display = 'none';
      previewImg.style.display = 'block';
      
      // Generate patient data for the "new" upload
      const patId = document.getElementById('pat-id').value || `P-${Math.floor(Math.random()*10000)}`;
      const patAge = document.getElementById('pat-age').value || 'N/A';
      const patEye = document.getElementById('pat-eye').value;
      const severityStages = [
        { level: 0, name: "No DR", color: "badge-negative", pos: 0 },
        { level: 1, name: "Mild NPDR", color: "badge-positive", pos: 25 },
        { level: 2, name: "Moderate NPDR", color: "badge-positive", pos: 50 },
        { level: 3, name: "Severe NPDR", color: "badge-positive", pos: 75 },
        { level: 4, name: "Proliferative DR", color: "badge-positive", pos: 100 }
      ];
      const stage = severityStages[Math.floor(Math.random() * severityStages.length)];
      
      const newPatientData = {
        uniqueKey: Date.now().toString(), // Generate a unique ID for the session array
        id: patId, 
        age: patAge, 
        eye: patEye, 
        date: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
        stageLevel: stage.level, 
        stageName: stage.name, 
        color: stage.color, 
        pos: stage.pos,
        certainty: (Math.random() * (99.9 - 85.0) + 85.0).toFixed(1),
        density: (stage.level * 22 + Math.random() * 10).toFixed(1),
        time: (Math.random() * 0.3 + 0.1).toFixed(2) + "s",
        history: [Math.max(0, stage.level - 2), Math.max(0, stage.level - 1), Math.max(0, stage.level - 1), stage.level, stage.level],
        imageSrc: e.target.result // Use the Base64 string of the uploaded file
      };

      // Push to the top of our session history array
      sessionHistory.unshift(newPatientData);
      
      // Update the Sidebar UI
      updateSidebar(newPatientData.uniqueKey);

      // Render the Dashboard with the new data
      renderDashboard(newPatientData);
    }, 3000);
  };
  reader.readAsDataURL(file);
}

// --- UPDATE SIDEBAR FUNCTION ---
function updateSidebar(activeKey) {
  // Clear the list
  historyList.innerHTML = '';
  
  if (sessionHistory.length === 0) {
      historyList.innerHTML = `<li style="text-align:center; padding: 20px; color: var(--muted); font-size: 12px; border: 1px dashed var(--border); cursor: default;">No scans processed yet.<br>Upload an image to start.</li>`;
      return;
  }

  // Build the list from the array
  sessionHistory.forEach(patient => {
    const li = document.createElement('li');
    if (patient.uniqueKey === activeKey) li.classList.add('active');
    
    // Determine status dot color
    let dotClass = 'success';
    if(patient.stageLevel > 1) dotClass = 'warning';
    if(patient.stageLevel > 3) dotClass = 'danger';

    li.innerHTML = `
      <div class="pat-list-info">
        <span class="status-dot ${dotClass}"></span> 
        <img src="${patient.imageSrc}" class="sidebar-thumb" alt="Scan Thumb" />
        <div class="pat-list-details">
          <strong>${patient.id}</strong>
          <span style="font-size:10px; color:var(--muted);">${patient.stageName}</span>
        </div>
      </div>
      <span class="pat-list-time">${patient.date}</span>
    `;

    // Add click listener to re-render this patient
    li.addEventListener('click', () => {
      updateSidebar(patient.uniqueKey); // Update active state
      renderDashboard(patient);
    });

    historyList.appendChild(li);
  });
}

// --- MASTER DASHBOARD RENDERER ---
function renderDashboard(patient) {
  // 1. Adjust View Visibility
  document.getElementById('upload-flow').style.display = 'none';
  document.getElementById('loading-overlay').style.display = 'none';
  
  previewWrap.style.display = 'block';
  previewImg.style.display = 'block';
  previewImg.src = patient.imageSrc;

  // Reset XAI if it was open
  document.getElementById('heatmap-overlay').classList.remove('active');
  document.getElementById('xai-controls').style.display = 'none';
  const toggleBtn = document.getElementById('toggleHeatmapBtn');
  if(toggleBtn) {
      toggleBtn.textContent = 'Enable Explainability';
      toggleBtn.style.background = 'var(--accent)';
      toggleBtn.style.color = '#030a0e';
      toggleBtn.style.border = 'none';
  }

  // 2. Populate Patient Summary
  document.getElementById('patient-summary').innerHTML = `
    <span><strong>ID:</strong> ${patient.id}</span>
    <span><strong>Age:</strong> ${patient.age}</span>
    <span><strong>Eye:</strong> ${patient.eye}</span>
    <span><strong>Time Processed:</strong> ${patient.date}</span>
  `;

  // 3. Populate Severity
  document.getElementById('result-label').textContent = patient.stageName;
  const badge = document.getElementById('result-badge');
  badge.textContent = patient.stageLevel === 0 ? 'Negative' : 'Refer to Ophthalmologist';
  badge.className = 'result-badge ' + patient.color;
  
  document.getElementById('severity-val').textContent = `Stage ${patient.stageLevel}`;
  setTimeout(() => { document.getElementById('severity-indicator').style.left = patient.pos + '%'; }, 100);
  
  // 4. Populate Radial Rings
  document.getElementById('m-prob').textContent = patient.certainty + '%';
  document.getElementById('m-conf').textContent = patient.density;
  document.getElementById('m-time').textContent = patient.time;

  setTimeout(() => {
    const certOffset = 264 - (264 * (patient.certainty / 100));
    const lesionOffset = 264 - (264 * (Math.min(patient.density, 100) / 100));
    document.getElementById('cert-ring').style.strokeDashoffset = certOffset;
    document.getElementById('lesion-ring').style.strokeDashoffset = lesionOffset;
  }, 200);

  // 5. Draw Chart
  const ctx = document.getElementById('progressionChart').getContext('2d');
  if (patientChart) patientChart.destroy(); 
  
  patientChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Jan', 'Apr', 'Jul', 'Oct', 'Current'],
      datasets: [{
        label: 'Severity Level',
        data: patient.history,
        borderColor: '#00ddb4',
        backgroundColor: 'rgba(0, 221, 180, 0.1)',
        borderWidth: 3,
        pointBackgroundColor: '#030a0e',
        pointBorderColor: '#00aaff',
        pointBorderWidth: 2,
        pointRadius: 5,
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: 4, ticks: { stepSize: 1, color: '#6b8f9e' }, grid: { color: 'rgba(0,220,180,0.05)' } },
        x: { ticks: { color: '#6b8f9e' }, grid: { display: false } }
      }
    }
  });

  // 6. Reveal Dashboard
  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// --- RESET FORM FOR NEW SCAN ---
function resetForm() {
  document.getElementById('upload-flow').style.display = 'block';
  previewWrap.style.display = 'none';
  resultCard.style.display = 'none';
  fileInput.value = '';
  
  // Remove active state from sidebar items
  document.querySelectorAll('.patient-list li').forEach(li => li.classList.remove('active'));

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
  
  document.getElementById('pat-id').value = '';
  document.getElementById('pat-age').value = '';
  
  document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
}

// Hook up the new sidebar button
document.getElementById('sidebarNewScanBtn').addEventListener('click', resetForm);

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

if(heatmapSlider) {
  heatmapSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    opacityVal.textContent = val + '%';
    if(heatmapOverlay.classList.contains('active')) {
      heatmapOverlay.style.opacity = val / 100;
    }
  });
}

const downloadReportBtn = document.getElementById('downloadReportBtn');
if(downloadReportBtn) {
  downloadReportBtn.addEventListener('click', () => {
    window.print(); 
  });
}

document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    e.preventDefault();
    document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth' });
  });
});