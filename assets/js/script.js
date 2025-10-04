/* =========================
   Cosmic background
   ========================= */
var PARTICLE_NUM = 200;
var PARTICLE_BASE_RADIUS = 0.5;
var FL = 500;
var DEFAULT_SPEED = 2;
var BOOST_SPEED = 100;

var canvas, context;
var canvasWidth, canvasHeight;
var centerX, centerY;
var mouseX, mouseY;
var speed = DEFAULT_SPEED;
var targetSpeed = DEFAULT_SPEED;
var particles = [];

function update_position(e) {
  const _t = e.target.closest('.btn');
  if (_t) {
    const r = _t.getBoundingClientRect();
    ['x','y'].forEach(c => {
      _t.style.setProperty(`--${c}`, `${e[`client${c.toUpperCase()}`] - r[c]}px`);
    });
  }
}

function loop() {
  context.save();
  context.fillStyle = 'rgba(0,0,0,0.2)';
  context.fillRect(0, 0, canvasWidth, canvasHeight);
  context.restore();

  speed += (targetSpeed - speed) * 0.01;

  context.beginPath();
  for (let i = 0; i < PARTICLE_NUM; i++) {
    const p = particles[i];
    p.pastZ = p.z;
    p.z -= speed;
    if (p.z <= 0) { randomizeParticle(p); continue; }

    const cx = centerX - (mouseX - centerX) * 1.25;
    const cy = centerY - (mouseY - centerY) * 1.25;
    const rx = p.x - cx;
    const ry = p.y - cy;
    const f = FL / p.z;
    const x = cx + rx * f;
    const y = cy + ry * f;
    const r = PARTICLE_BASE_RADIUS * f;

    context.moveTo(x, y);
    context.arc(x, y, r, 0, Math.PI * 2);
  }
  context.fill();
}
function randomizeParticle(p) {
  p.x = Math.random() * canvasWidth;
  p.y = Math.random() * canvasHeight;
  p.z = Math.random() * 1500 + 500;
  return p;
}
function Particle(x, y, z) {
  this.x = x || 0;
  this.y = y || 0;
  this.z = z || 0;
  this.pastZ = 0;
}

/* =========================
   Main logic
   ========================= */
window.addEventListener('load', async function () {
  // canvas setup
  canvas = document.getElementById('c');
  function resize() {
    canvasWidth = canvas.width = window.innerWidth;
    canvasHeight = canvas.height = window.innerHeight;
    centerX = canvasWidth * 0.5;
    centerY = canvasHeight * 0.5;
    context = canvas.getContext('2d');
    context.fillStyle = 'rgb(255,255,255)';
  }
  window.addEventListener('resize', resize);
  resize();

  mouseX = centerX; mouseY = centerY;
  for (let i = 0; i < PARTICLE_NUM; i++) {
    particles[i] = randomizeParticle(new Particle());
    particles[i].z -= 500 * Math.random();
  }
  document.addEventListener('mousemove', (e) => { mouseX = e.clientX; mouseY = e.clientY; update_position(e); });
  document.addEventListener('mousedown', () => targetSpeed = BOOST_SPEED);
  document.addEventListener('mouseup', () => targetSpeed = DEFAULT_SPEED);
  setInterval(loop, 1000 / 120);

  // sections
  const sections = {
    getStarted: document.getElementById('get-started-section'),
    levelSelection: document.getElementById('level-selection-section'),
    beginner: document.getElementById('beginner-section'),
    scientist: document.getElementById('scientist-section')
  };

  function resetAnimations(sectionId) {
    const section = sections[sectionId];
    if (!section) return;
    const animatable = section.querySelectorAll('h1, .btn, .level-message, .level-buttons');
    animatable.forEach(el => {
      el.classList.remove('absorb', 'slide-out');
      el.style.opacity = '0';
      el.style.transform = 'translateY(-50px)';
      el.style.animation = 'none';
      el.offsetHeight; // reflow
      el.style.animation = '';
      el.style.animation = 'floatInFromTop 1.5s ease-out forwards';
    });
  }

  function hideAll() { Object.values(sections).forEach(sec => { sec.classList.remove('active'); sec.style.display = 'none'; }); }
  function showSection(sectionId, pushState = true) {
    hideAll();
    sections[sectionId].style.display = 'block';
    setTimeout(() => { sections[sectionId].classList.add('active'); resetAnimations(sectionId); }, 10);
    if (pushState) history.pushState({ section: sectionId }, '', `#${sectionId}`);
  }

  // history: back/forward arrows
  window.addEventListener('popstate', (e) => {
    const sectionId = e.state?.section || 'getStarted';
    hideAll();
    sections[sectionId].style.display = 'block';
    setTimeout(() => { sections[sectionId].classList.add('active'); resetAnimations(sectionId); }, 10);
  });

  // init
  hideAll();
  sections.getStarted.style.display = 'block';
  sections.getStarted.classList.add('active');
  history.replaceState({ section: 'getStarted' }, '', '#getStarted');

  // Get Started → Level Selection
  const getStartedBtn = document.querySelector('#get-started-section .btn');
  getStartedBtn.addEventListener('click', (e) => {
    e.preventDefault();
    const h1 = document.querySelector('#get-started-section h1');
    getStartedBtn.classList.add('slide-out');
    h1.classList.add('absorb');
    setTimeout(() => showSection('levelSelection'), 1200);
  });

  // Beginner / Scientist buttons
  const beginnerBtn  = document.querySelector('#level-selection-section .level-buttons .btn:nth-child(1)');
  const scientistBtn = document.querySelector('#level-selection-section .level-buttons .btn:nth-child(2)');

  /* ========= BEGINNER ========= */
  const beginnerColumns = [
    'koi_period','koi_duration','koi_impact','koi_depth','koi_model_snr',
    'prad_srad_ratio','teq_derived','insol','koi_steff','koi_srad'
  ];
  const simpleDescriptions = {
    'koi_period': 'Orbital period of the planet (days).',
    'koi_duration': 'Duration of the planet’s transit (hours).',
    'koi_impact': 'Impact parameter, related to transit trajectory.',
    'koi_depth': 'Transit depth, how much the star dims (ppm).',
    'koi_model_snr': 'Signal-to-noise ratio of the transit model.',
    'prad_srad_ratio': 'Planet-to-star radius ratio.',
    'teq_derived': 'Equilibrium temperature of the planet (Kelvin).',
    'insol': 'Insolation, stellar flux received by the planet.',
    'koi_steff': 'Stellar effective temperature (Kelvin).',
    'koi_srad': 'Stellar radius (solar radii).'
  };
  const sliderRanges = {
    'koi_period':      {min: 0.1, max: 1000,   step: 0.001, value: 0.1},
    'koi_duration':    {min: 0.1, max: 50,     step: 0.01,  value: 0.1},
    'koi_impact':      {min: 0,   max: 2,      step: 0.01,  value: 0},
    'koi_depth':       {min: 0,   max: 100000, step: 1,     value: 0},
    'koi_model_snr':   {min: 0,   max: 1000,   step: 1,     value: 0},
    'prad_srad_ratio': {min: 0,   max: 100,    step: 0.01,  value: 0},
    'teq_derived':     {min: 0,   max: 3000,   step: 1,     value: 0},
    'insol':           {min: 0,   max: 1000000,step: 0.01,  value: 0},
    'koi_steff':       {min: 2000,max: 10000,  step: 1,     value: 2000},
    'koi_srad':        {min: 0.1, max: 10,     step: 0.01,  value: 0.1}
  };

  // DOM
  const presetSelect = document.querySelector('.preset-select');
  const nameInput = document.querySelector('.beginner-section input[type="text"]');
  const predictBtn = document.querySelector('.predict-btn');
  const predictionResult = document.querySelector('.prediction-result');
  const tbody = document.querySelector('.parameters-table tbody');
  const sliderContainers = {
    'koi_period':      document.getElementById('orbital-sliders'),
    'koi_duration':    document.getElementById('orbital-sliders'),
    'koi_impact':      document.getElementById('orbital-sliders'),
    'koi_depth':       document.getElementById('transit-sliders'),
    'koi_model_snr':   document.getElementById('transit-sliders'),
    'prad_srad_ratio': document.getElementById('transit-sliders'),
    'teq_derived':     document.getElementById('planet-sliders'),
    'insol':           document.getElementById('planet-sliders'),
    'koi_steff':       document.getElementById('stellar-sliders'),
    'koi_srad':        document.getElementById('stellar-sliders')
  };

  function clearSliders() { Object.values(sliderContainers).forEach(c => c.innerHTML = ''); }
  function buildSliders(initial = null) {
    clearSliders();
    beginnerColumns.forEach(col => {
      const wrap = document.createElement('div');
      wrap.className = 'slider-div';
      const label = document.createElement('label'); label.textContent = col;
      const sliderWrapper = document.createElement('div'); sliderWrapper.className = 'slider-wrapper';
      const slider = document.createElement('input');
      slider.type = 'range'; slider.id = `slider_${col}`;
      const r = sliderRanges[col];
      slider.min = r.min; slider.max = r.max; slider.step = r.step; slider.value = initial?.[col] ?? r.value;
      const valueSpan = document.createElement('span'); valueSpan.textContent = slider.value;
      slider.addEventListener('input', () => valueSpan.textContent = slider.value);
      sliderWrapper.appendChild(slider); sliderWrapper.appendChild(valueSpan);
      wrap.appendChild(label); wrap.appendChild(sliderWrapper);
      sliderContainers[col].appendChild(wrap);
    });
  }
  function fillTable() {
    tbody.innerHTML = '';
    beginnerColumns.forEach(col => {
      const tr = document.createElement('tr');
      const td1 = document.createElement('td'); td1.textContent = col;
      const td2 = document.createElement('td'); td2.textContent = simpleDescriptions[col] || '—';
      tr.appendChild(td1); tr.appendChild(td2);
      tbody.appendChild(tr);
    });
  }

  // fetch presets
  let presets = [];
  try {
    const resp = await fetch('http://127.0.0.1:5000/presets');
    if (resp.ok) presets = await resp.json();
  } catch (e) { console.error("Presets fetch fail", e); }

  beginnerBtn.addEventListener('click', (e) => {
    e.preventDefault();
    presetSelect.innerHTML = '<option value="">Manual Input</option>';
    presets.forEach((p,i) => {
      const opt = document.createElement('option');
      opt.value = i; opt.textContent = p.kepler_name || `Preset ${i+1}`;
      presetSelect.appendChild(opt);
    });
    buildSliders(); fillTable();

    presetSelect.onchange = () => {
      const idx = presetSelect.value;
      if (idx === '') { nameInput.value = ''; buildSliders(); }
      else {
        const preset = presets[idx];
        nameInput.value = preset.kepler_name || '';
        buildSliders(preset);
      }
      fillTable();
    };
    showSection('beginner');
  });

  predictBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    predictionResult.style.display = 'block'; predictionResult.textContent = 'Predicting...';
    const row = {}; beginnerColumns.forEach(col => row[col] = parseFloat(document.getElementById(`slider_${col}`).value));
    try {
      const resp = await fetch('http://127.0.0.1:5000/predict', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(row)
      });
      if (!resp.ok) throw new Error();
      const data = await resp.json();
      predictionResult.textContent = `Your planet has a ${(data.prob*100).toFixed(2)}% chance of being an exoplanet candidate.`;
    } catch { predictionResult.textContent = 'Prediction failed.'; }
  });

  /* ========= SCIENTIST ========= */
  scientistBtn.addEventListener('click', (e) => { e.preventDefault(); showSection('scientist'); });
  const scientistForm = document.getElementById('scientist-form');
  const scientistResult = document.getElementById('scientist-result');
  if (scientistForm) {
    scientistForm.addEventListener('submit', async (e) => {
      e.preventDefault(); scientistResult.textContent = 'Predicting...';
      const formData = new FormData(scientistForm); const payload={}; formData.forEach((v,k)=>payload[k]=parseFloat(v));
      try {
        const resp = await fetch('http://127.0.0.1:5000/predict_scientist', {
          method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)
        });
        if (!resp.ok) throw new Error();
        const data = await resp.json();
        scientistResult.textContent = `Prediction: ${data.prediction}`;
      } catch { scientistResult.textContent = 'Prediction failed.'; }
    });
  }

  // back buttons
  document.querySelectorAll('.back-btn').forEach(btn => {
    btn.addEventListener('click', (e) => { e.preventDefault(); showSection('levelSelection'); });
  });
});
