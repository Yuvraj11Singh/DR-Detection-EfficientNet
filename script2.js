// ── State ──
let joiner = null;
let pipelineRunning = false;
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── View switching ──
function switchView(v) {
  ['ta','agent','hr'].forEach(id => {
    document.getElementById('view-' + id).classList.toggle('active', id === v);
    document.getElementById('tab-' + id).classList.toggle('active', id === v);
  });
}

function scrollTo(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Prefill ──
function prefill() {
  const tom = new Date(); tom.setDate(tom.getDate() + 1);
  document.getElementById('f-name').value = 'Priya Sharma';
  document.getElementById('f-role').value = 'Senior Product Manager';
  document.getElementById('f-dept').value = 'Delivery';
  document.getElementById('f-mode').value = 'Remote (WFH)';
  document.getElementById('f-doj').value = tom.toISOString().split('T')[0];
  document.getElementById('f-email').value = 'priya.sharma.personal@gmail.com';
  document.getElementById('f-address').value = 'Flat 4B, Prestige Towers, Whitefield, Bengaluru 560066';
  document.getElementById('f-mgr').value = 'Rahul Gupta';
  document.getElementById('f-mgr-email').value = 'rahul.gupta@acmecorp.com';
  document.getElementById('f-buddy').value = 'Ananya Iyer';
  document.getElementById('f-buddyid').value = '@ananya.iyer';
  document.getElementById('f-company').value = 'Acme Corp';
  document.getElementById('f-channel').value = '#general';
  document.getElementById('f-assets').value = 'Laptop + Mouse + Keyboard';
  document.getElementById('f-os').value = 'macOS';
  document.getElementById('f-culture').value = 'We value ownership, async-first collaboration, and learning out loud. Every voice matters.';
}

function fmtDate(d) {
  if (!d) return 'soon';
  const dt = new Date(d + 'T00:00:00');
  return dt.toLocaleDateString('en-IN', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
}

function nowTime() {
  return new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function initials(name) {
  return name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
}

// ── TA Submit ──
async function submitTA() {
  const name = document.getElementById('f-name').value.trim();
  const role = document.getElementById('f-role').value.trim();
  if (!name || !role) { alert('Please fill in at least name and role.'); return; }

  joiner = {
    name, role,
    dept: document.getElementById('f-dept').value.trim(),
    mode: document.getElementById('f-mode').value,
    doj: document.getElementById('f-doj').value,
    dojFmt: fmtDate(document.getElementById('f-doj').value),
    email: document.getElementById('f-email').value.trim(),
    address: document.getElementById('f-address').value.trim(),
    mgr: document.getElementById('f-mgr').value.trim(),
    mgrEmail: document.getElementById('f-mgr-email').value.trim(),
    buddy: document.getElementById('f-buddy').value.trim(),
    buddyId: document.getElementById('f-buddyid').value.trim(),
    company: document.getElementById('f-company').value.trim() || 'the company',
    channel: document.getElementById('f-channel').value.trim() || '#general',
    assets: document.getElementById('f-assets').value,
    os: document.getElementById('f-os').value,
    culture: document.getElementById('f-culture').value.trim(),
    firstName: name.split(' ')[0]
  };

  const btn = document.getElementById('ta-submit');
  btn.disabled = true;
  document.getElementById('ta-spin').style.display = 'block';
  document.getElementById('ta-label').textContent = 'Triggering agent…';

  await sleep(1200);

  document.getElementById('ta-spin').style.display = 'none';
  document.getElementById('ta-label').textContent = 'Submitted!';
  document.getElementById('success-name').textContent = name;
  document.getElementById('ta-success').style.display = 'block';

  // Prep pipeline in background
  prepPipeline();
}

// ── Pipeline task helpers ──
let taskCount = 0;
let doneCount = 0;

function makeTaskCard(id, sectionId, iconEmoji, iconClass, name, channel, meta, isAI = true) {
  taskCount++;
  const section = document.getElementById('tasks-' + sectionId);
  const card = document.createElement('div');
  card.className = 'task-card';
  card.id = 'tc-' + id;
  let metaHTML = meta.map(m => `<div class="meta-row"><span class="meta-k">${m.k}</span><span class="meta-v">${m.v}</span></div>`).join('');
  card.innerHTML = `
    <div class="tc-header">
      <div class="tc-header-left">
        <div class="tc-icon ${iconClass}">${iconEmoji}</div>
        <div>
          <div class="tc-name">${name}</div>
          <div class="tc-channel">${channel}</div>
        </div>
      </div>
      <div class="tc-status">
        <div class="tcs-dot" id="dot-${id}"></div>
        <span class="tcs-label" id="stlbl-${id}">Queued</span>
      </div>
    </div>
    <div class="tc-body" id="tcb-${id}">
      ${isAI ? `<div class="tc-skeleton"><div class="skel" style="width:55%"></div><div class="skel" style="width:88%"></div><div class="skel" style="width:70%"></div></div>` : ''}
      ${!isAI ? `<div class="tc-meta">${metaHTML}</div>` : ''}
    </div>`;
  section.appendChild(card);
  return card;
}

function setRunning(id) {
  const c = document.getElementById('tc-' + id);
  if (c) c.className = 'task-card tc-running';
  const d = document.getElementById('dot-' + id);
  if (d) { d.className = 'tcs-dot running'; }
  const l = document.getElementById('stlbl-' + id);
  if (l) { l.textContent = 'Running…'; l.className = 'tcs-label running'; }
}

function setDone(id, subject, body, meta, channel, dest) {
  doneCount++;
  updateProgress();
  const c = document.getElementById('tc-' + id);
  if (c) c.className = 'task-card tc-done';
  const d = document.getElementById('dot-' + id);
  if (d) d.className = 'tcs-dot done';
  const l = document.getElementById('stlbl-' + id);
  if (l) { l.textContent = 'Delivered · ' + nowTime(); l.className = 'tcs-label done'; }
  const b = document.getElementById('tcb-' + id);
  if (b) {
    let metaHTML = (meta || []).map(m => `<div class="meta-row"><span class="meta-k">${m.k}</span><span class="meta-v">${m.v}</span></div>`).join('');
    let html = '';
    if (metaHTML) html += `<div class="tc-meta">${metaHTML}</div>`;
    if (subject) html += `<div class="msg-subject">${subject}</div>`;
    if (body) html += `<div class="msg-body" id="msgb-${id}">${body}</div>`;
    b.innerHTML = html;
    c.innerHTML += `<div class="tc-footer"><div class="tf-sent">✓ Delivered via ${channel} → ${dest}</div>${body ? `<button class="copy-btn" onclick="copyMsg('${id}')">Copy</button>` : ''}</div>`;
  }
  // update nav badge
  const nb = document.getElementById('nbadge-' + id);
  if (nb) { nb.textContent = '✓'; nb.className = 'day-item-badge badge-done'; }
}

function setActionDone(id, lines, channel, dest) {
  doneCount++;
  updateProgress();
  const c = document.getElementById('tc-' + id);
  if (c) c.className = 'task-card tc-done';
  const d = document.getElementById('dot-' + id);
  if (d) d.className = 'tcs-dot done';
  const l = document.getElementById('stlbl-' + id);
  if (l) { l.textContent = 'Dispatched · ' + nowTime(); l.className = 'tcs-label done'; }
  const b = document.getElementById('tcb-' + id);
  if (b) {
    let html = `<div class="tc-action-body"><div class="tc-action-icon">✅</div><div class="tc-action-text"><strong>Completed by agent</strong> — ${channel} → ${dest}<div class="tc-checklist">`;
    lines.forEach(l => { html += `<div class="check-item"><span class="ci-icon">✓</span>${l}</div>`; });
    html += '</div></div></div>';
    b.innerHTML = html;
    c.innerHTML += `<div class="tc-footer"><div class="tf-sent">✓ ${channel}</div><div class="tf-info">${nowTime()}</div></div>`;
  }
  const nb = document.getElementById('nbadge-' + id);
  if (nb) { nb.textContent = '✓'; nb.className = 'day-item-badge badge-done'; }
}

function updateProgress() {
  const pct = Math.round((doneCount / taskCount) * 100);
  document.getElementById('pb-fill').style.width = pct + '%';
  document.getElementById('pb-pct').textContent = pct + '%';
}

function copyMsg(id) {
  const el = document.getElementById('msgb-' + id);
  if (el) navigator.clipboard.writeText(el.innerText);
}

// ── Claude API ──
async function callClaude(system, user) {
  try {
    const resp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1000,
        system,
        messages: [{ role: 'user', content: user }]
      })
    });
    const data = await resp.json();
    return data.content?.[0]?.text?.trim() || '';
  } catch(e) { return '[Could not generate — API error]'; }
}

// ── Prepare pipeline UI ──
function prepPipeline() {
  if (!joiner) return;
  document.getElementById('pipeline-idle').style.display = 'none';
  document.getElementById('pipeline-content').style.display = 'block';

  // Profile card
  document.getElementById('profile-card').innerHTML = `
    <div class="profile-left">
      <div class="profile-avatar">${initials(joiner.name)}</div>
      <div>
        <div class="profile-name">${joiner.name}</div>
        <div class="profile-role">${joiner.role} · ${joiner.dept} · ${joiner.company}</div>
      </div>
    </div>
    <div class="profile-tags">
      <span class="ptag ptag-blue">${joiner.mode}</span>
      <span class="ptag ptag-amber">Joins ${joiner.dojFmt}</span>
      <span class="ptag ptag-green">Buddy: ${joiner.buddy}</span>
    </div>`;

  // Progress steps
  document.getElementById('pb-steps').innerHTML = `
    <span class="pb-step" id="pbs-dayminus1">◦ Day −1</span>
    <span class="pb-step" id="pbs-day0">◦ Day 0</span>
    <span class="pb-step" id="pbs-day1">◦ Day 1</span>
    <span class="pb-step" id="pbs-day23">◦ Day 2–3</span>`;

  // Create all task cards (skeletons)
  // Day -1
  makeTaskCard('it', 'dayminus1', '💻', 'tc-icon icon-it', 'IT asset requisition', 'IT Ticketing system', [], false);
  makeTaskCard('admin', 'dayminus1', '📦', 'tc-icon icon-admin', 'Welcome kit dispatch', 'Admin team notification', [], false);
  makeTaskCard('welcome', 'dayminus1', '📧', 'tc-icon icon-email', 'Welcome email to joiner', `Outlook → ${joiner.email || joiner.name}`, []);
  // Day 0
  makeTaskCard('accounts', 'day0', '🔑', 'tc-icon icon-it', 'System access provisioning', 'M365 · Slack · Google Drive', [], false);
  makeTaskCard('buddy', 'day0', '💬', 'tc-icon icon-slack', 'Buddy introduction', `Slack DM → ${joiner.buddyId || joiner.buddy}`, []);
  makeTaskCard('org', 'day0', '📣', 'tc-icon icon-slack', 'Org-wide announcement', `Slack → ${joiner.channel}`, []);
  // Day 1
  makeTaskCard('doccollect', 'day1', '📋', 'tc-icon icon-hr', 'Document collection email', `Outlook → ${joiner.email || joiner.name}`, []);
  makeTaskCard('apptletter', 'day1', '📜', 'tc-icon icon-doc', 'Appointment letter', `Outlook → ${joiner.email || joiner.name}`, []);
  makeTaskCard('ld', 'day1', '🗓', 'tc-icon icon-hr', 'L&D onboarding schedule', `Outlook calendar → ${joiner.email || joiner.name}`, []);
  // Day 2-3
  makeTaskCard('hrms', 'day23', '🗄', 'tc-icon icon-it', 'HRMS employee record update', 'HRMS / HCM system', [], false);
  makeTaskCard('handoff', 'day23', '🤝', 'tc-icon icon-email', 'Manager handoff & project assignment', `Outlook → ${joiner.mgrEmail || joiner.mgr}`, []);
}

// ── Run full pipeline ──
async function runPipeline() {
  if (!joiner || pipelineRunning) return;
  pipelineRunning = true;
  doneCount = 0;
  taskCount = 10; // total tasks
  updateProgress();

  const J = joiner;
  const sys = `You are an HR communications writer for ${J.company}. Write warm, genuine, professional messages. Be specific, no generic filler. No markdown, no asterisks. Return only the message text.`;

  // ── DAY -1 ──
  document.getElementById('pbs-dayminus1').className = 'pb-step active';

  // IT requisition (non-AI)
  setRunning('it');
  await sleep(900);
  setActionDone('it', [
    `Asset: ${J.assets} (${J.os})`,
    `Delivery address: ${J.address || 'Home address on file'}`,
    `Required by: ${J.dojFmt}`,
    `Ticket ID: IT-${Math.floor(Math.random()*9000+1000)} raised in system`
  ], 'IT Ticketing system', 'IT Support team');

  await sleep(400);

  // Admin welcome kit (non-AI)
  setRunning('admin');
  await sleep(800);
  setActionDone('admin', [
    `Welcome kit dispatched to: ${J.address || 'Home address on file'}`,
    `Contents: Offer letter, Company handbook, Branded merchandise`,
    `Estimated delivery: ${J.dojFmt}`,
    `Tracking ID: ADM-${Math.floor(Math.random()*9000+1000)}`
  ], 'Admin team', 'Courier dispatched');

  await sleep(400);

  // Welcome email (AI)
  setRunning('welcome');
  const welcomeText = await callClaude(sys,
    `Write a warm welcome email to ${J.name} joining as ${J.role} in the ${J.dept} team at ${J.company} on ${J.dojFmt}. Work mode: ${J.mode}. Buddy: ${J.buddy}. Manager: ${J.mgr}. ${J.culture ? 'Culture: ' + J.culture : ''} Mention what they can expect on Day 1, name their buddy, and close warmly. Under 180 words. From the HR team. Just the email body, no subject.`
  );
  setDone('welcome',
    `Welcome to ${J.company}, ${J.firstName}! 🎉`,
    welcomeText,
    [{ k: 'To:', v: J.email || J.name }, { k: 'From:', v: `hr@${J.company.toLowerCase().replace(/\s/g,'')}.com` }, { k: 'Via:', v: 'Outlook' }],
    'Outlook', J.email || J.name
  );

  document.getElementById('pbs-dayminus1').className = 'pb-step done';
  document.getElementById('pbs-day0').className = 'pb-step active';
  await sleep(600);

  // ── DAY 0 ──

  // Accounts (non-AI)
  setRunning('accounts');
  await sleep(1000);
  setActionDone('accounts', [
    `Microsoft 365 account created: ${J.firstName.toLowerCase()}.${J.name.split(' ').slice(-1)[0].toLowerCase()}@${J.company.toLowerCase().replace(/\s/g,'')}.com`,
    `Slack workspace invite sent to ${J.email || J.name}`,
    `Google Drive shared folder access granted`,
    `Temporary password sent via secure link`
  ], 'M365 + Slack + Drive', J.email || J.name);

  await sleep(400);

  // Buddy intro (AI)
  setRunning('buddy');
  const buddyText = await callClaude(sys,
    `Write a friendly Slack DM to ${J.buddy} (${J.buddyId}), telling them they are the buddy for ${J.name} who joins as ${J.role} on ${J.dojFmt}. Ask them to reach out with a quick intro call in week 1 and be the go-to for any culture or process questions. Warm, under 90 words. From HR.`
  );
  setDone('buddy', null, buddyText,
    [{ k: 'To:', v: J.buddyId || J.buddy }, { k: 'Via:', v: 'Slack DM' }],
    'Slack', J.buddyId || J.buddy
  );

  await sleep(400);

  // Org announcement (AI)
  setRunning('org');
  const orgText = await callClaude(sys,
    `Write a short Slack announcement for ${J.channel} welcoming ${J.name} who joins as ${J.role} in the ${J.dept} team on ${J.dojFmt}. Work mode: ${J.mode}. Manager: ${J.mgr}. Invite team to say hello. Under 65 words. Upbeat, human, not corporate. From HR.`
  );
  setDone('org', null, orgText,
    [{ k: 'To:', v: J.channel }, { k: 'Via:', v: 'Slack' }],
    'Slack', J.channel
  );

  document.getElementById('pbs-day0').className = 'pb-step done';
  document.getElementById('pbs-day1').className = 'pb-step active';
  await sleep(600);

  // ── DAY 1 ──

  // Doc collection (AI)
  setRunning('doccollect');
  const docText = await callClaude(sys,
    `Write an email to ${J.name} (${J.role}) asking them to submit the following documents by end of Day 1: Aadhaar card, PAN card, last 3 months payslips, educational certificates, previous employer relieving letter, passport-size photo. Mention they can reply to this email with attachments. Professional but friendly. Under 130 words. From the HR team.`
  );
  setDone('doccollect',
    `Action needed: Please submit your joining documents`,
    docText,
    [{ k: 'To:', v: J.email || J.name }, { k: 'From:', v: `hr@${J.company.toLowerCase().replace(/\s/g,'')}.com` }, { k: 'Via:', v: 'Outlook' }],
    'Outlook', J.email || J.name
  );

  await sleep(400);

  // Appointment letter (AI)
  setRunning('apptletter');
  const apptText = await callClaude(sys,
    `Write a formal appointment letter for ${J.name} being appointed as ${J.role} in the ${J.dept} department at ${J.company}, joining on ${J.dojFmt}. Work arrangement: ${J.mode}. Reporting to: ${J.mgr}. Include: warm welcome, role confirmation, start date, reporting structure, a brief note on probation period (3 months standard), and a line about looking forward to their contribution. Under 200 words. Formal but welcoming.`
  );
  setDone('apptletter',
    `Appointment Letter — ${J.name} · ${J.role}`,
    apptText,
    [{ k: 'To:', v: J.email || J.name }, { k: 'Type:', v: 'Formal appointment letter' }, { k: 'Via:', v: 'Outlook' }],
    'Outlook', J.email || J.name
  );

  await sleep(400);

  // L&D schedule (non-AI)
  setRunning('ld');
  await sleep(700);
  setActionDone('ld', [
    `Day 1: Company overview & culture session — 10:00 AM (Google Meet)`,
    `Day 1: HR policies & benefits walkthrough — 2:00 PM (Google Meet)`,
    `Day 2: Role-specific onboarding with ${J.mgr} — 11:00 AM`,
    `Day 3: Tools & systems deep-dive — 10:00 AM`,
    `Week 2: First 1:1 with ${J.buddy} (buddy check-in)`,
    `All calendar invites sent via Outlook to ${J.email || J.name}`
  ], 'Outlook Calendar', J.email || J.name);

  document.getElementById('pbs-day1').className = 'pb-step done';
  document.getElementById('pbs-day23').className = 'pb-step active';
  await sleep(600);

  // ── DAY 2-3 ──

  // HRMS update (non-AI)
  setRunning('hrms');
  await sleep(900);
  setActionDone('hrms', [
    `Employee ID assigned: EMP-${Math.floor(Math.random()*90000+10000)}`,
    `Record created: ${J.name} · ${J.role} · ${J.dept}`,
    `Status: Active from ${J.dojFmt}`,
    `Reporting manager linked: ${J.mgr}`,
    `Work mode tagged: ${J.mode}`
  ], 'HRMS / HCM', 'Employee database');

  await sleep(400);

  // Handoff email (AI)
  setRunning('handoff');
  const handoffText = await callClaude(sys,
    `Write a brief email to ${J.mgr} (${J.mgrEmail || 'hiring manager'}) informing them that ${J.name}'s onboarding is complete. Mention: all documentation collected, accounts provisioned, L&D schedule done, employee is ready for project assignment. Ask the manager to schedule the first project briefing. Professional and concise. Under 100 words. From HR.`
  );
  setDone('handoff',
    `${J.firstName} is onboarding-complete — project assignment ready`,
    handoffText,
    [{ k: 'To:', v: J.mgrEmail || J.mgr }, { k: 'From:', v: `hr@${J.company.toLowerCase().replace(/\s/g,'')}.com` }, { k: 'Via:', v: 'Outlook' }],
    'Outlook', J.mgrEmail || J.mgr
  );

  document.getElementById('pbs-day23').className = 'pb-step done';
  await sleep(400);

  // Final
  document.getElementById('final-success').style.display = 'block';
  document.getElementById('fs-sub').textContent = `${J.name} is fully onboarded. 10 tasks completed across Day −1 through Day 2–3. ${J.mgr} has been notified for project assignment.`;

  // Update HR view
  buildHRView();
  pipelineRunning = false;
}

// ── HR view ──
function buildHRView() {
  if (!joiner) return;
  const J = joiner;
  document.getElementById('hr-notice').style.display = 'none';
  document.getElementById('hr-content').style.display = 'block';
  document.getElementById('hr-notice-title').textContent = `Agent has completed pre-work for ${J.name}`;

  const tasks = [
    { icon: '📞', title: 'Day 1 connect call with employee', desc: `Call ${J.name} to check in, answer any questions, and ensure they have system access. 15–20 min.`, who: 'hr', action: 'Mark as done' },
    { icon: '📄', title: 'Review submitted documents', desc: `${J.name} will email documents. Review completeness and flag any missing items. Store to SharePoint.`, who: 'hr', action: 'Mark as done' },
    { icon: '🎤', title: 'In-person onboarding presentation', desc: `Run the Day 1 onboarding presentation (if hybrid) or share slides over Meet. Agent has already sent the invite.`, who: 'hr', action: 'Mark as done' },
    { icon: '✅', title: 'HRMS documentation sign-off', desc: `Confirm all documents are in order and mark employee record as documentation-complete in HRMS.`, who: 'hr', action: 'Mark as done' },
    { icon: '🤖', title: 'Welcome email sent', desc: `Sent automatically by agent on Day −1 to ${J.email}.`, who: 'agent' },
    { icon: '🤖', title: 'Appointment letter issued', desc: `Generated and emailed by agent on Day 1.`, who: 'agent' },
    { icon: '🤖', title: 'Buddy introduced', desc: `${J.buddy} notified via Slack DM by agent.`, who: 'agent' },
    { icon: '🤖', title: 'L&D schedule dispatched', desc: `4 calendar invites sent by agent to ${J.email}.`, who: 'agent' }
  ];

  const grid = document.getElementById('hr-tasks-grid');
  grid.innerHTML = '';
  tasks.forEach((t, i) => {
    const card = document.createElement('div');
    card.className = 'hr-task-card';
    card.innerHTML = `
      <div class="hr-tc-top">
        <div class="hr-tc-icon">${t.icon}</div>
        <span class="hr-tc-status ${t.who === 'agent' ? 'hts-agent' : 'hts-hr'}">${t.who === 'agent' ? '🤖 Agent done' : '👤 HR action'}</span>
      </div>
      <div class="hr-tc-title">${t.title}</div>
      <div class="hr-tc-desc">${t.desc}</div>
      ${t.action ? `<button class="hr-action-btn" id="hr-act-${i}" onclick="markHRDone(${i})">✓ ${t.action}</button>` : ''}`;
    grid.appendChild(card);
  });
}

function markHRDone(i) {
  const btn = document.getElementById('hr-act-' + i);
  if (btn) { btn.textContent = '✓ Done'; btn.className = 'hr-action-btn done-btn'; btn.disabled = true; }
}

// ── Auto-run pipeline when view switches ──
const _origSwitch = switchView;
window.switchView = function(v) {
  _origSwitch(v);
  if (v === 'agent' && joiner && !pipelineRunning && doneCount === 0) {
    setTimeout(runPipeline, 400);
  }
};

// Default date
const tom = new Date(); tom.setDate(tom.getDate() + 1);
document.getElementById('f-doj').value = tom.toISOString().split('T')[0];