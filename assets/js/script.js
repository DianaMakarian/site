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
  // Scroll to top on page load/refresh
  window.scrollTo(0, 0);
  
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
    scientist: document.getElementById('scientist-section'),
    learnMore: document.getElementById('learnMore')
  };

  function resetAnimations(sectionId) {
    const section = sections[sectionId];
    if (!section) return;
    const animatable = section.querySelectorAll('h1, h2, .btn, .level-message, .level-buttons, .exoplanet-info, .intro-text, .section-block');
    animatable.forEach(el => {
      el.classList.remove('absorb', 'slide-out');
      el.style.opacity = '0';
      el.style.transform = 'translateY(-50px)';
      el.style.animation = 'none';
      el.offsetHeight; // reflow
      el.style.animation = '';
      el.style.animation = 'floatInFromTop 0.8s ease-out forwards';
    });
    // Special handling for exoplanet-info in learnMore section
    if (sectionId === 'learnMore') {
      const exoplanetInfo = section.querySelector('.exoplanet-info');
      if (exoplanetInfo) {
        exoplanetInfo.style.display = 'block';
        exoplanetInfo.classList.add('visible');
      }
    }
  }

  function hideAll() { 
    Object.values(sections).forEach(sec => { 
      sec.classList.remove('active'); 
      sec.style.display = 'none'; 
      if (sec.id === 'learnMore') {
        const exoplanetInfo = sec.querySelector('.exoplanet-info');
        if (exoplanetInfo) {
          exoplanetInfo.style.display = 'none';
          exoplanetInfo.classList.remove('visible');
        }
      }
    }); 
  }

  function showSection(sectionId, pushState = true) {
    hideAll();
    sections[sectionId].style.display = 'block';
    sections[sectionId].classList.add('active');
    resetAnimations(sectionId);
    if (pushState) history.pushState({ section: sectionId }, '', `#${sectionId}`);
    // Scroll to top only for non-beginner sections
    if (sectionId !== 'beginner') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  // Navigation menu
  document.querySelectorAll('nav .nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const sectionId = link.getAttribute('href').substring(1);
      showSection(sectionId);
    });
  });

  // history: back/forward arrows
  window.addEventListener('popstate', (e) => {
    const sectionId = e.state?.section || 'getStarted';
    hideAll();
    sections[sectionId].style.display = 'block';
    sections[sectionId].classList.add('active');
    resetAnimations(sectionId);
    // Scroll to top only for non-beginner sections
    if (sectionId !== 'beginner') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
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
    showSection('levelSelection');
  });

  // Beginner / Scientist buttons
  const beginnerBtn = document.querySelector('#level-selection-section .level-buttons .btn:nth-child(1)');
  const scientistBtn = document.querySelector('#level-selection-section .level-buttons .btn:nth-child(2)');

  /* ========= BEGINNER ========= */
  const beginnerColumns = [
    'koi_period','koi_duration','koi_impact','koi_depth','koi_model_snr',
    'prad_srad_ratio','teq_derived','insol','koi_steff','koi_srad'
  ];
  const simpleDescriptions = {
    'koi_period': 'Orbital period of the planet (days).',
    'koi_duration': 'Duration of the planet\'s transit (hours).',
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
    'koi_period':      {min: 0.1, max: 1000,   step: 0.001, value: 365},
    'koi_duration':    {min: 0.1, max: 50,     step: 0.01,  value: 6},
    'koi_impact':      {min: 0,   max: 2,      step: 0.01,  value: 0.3},
    'koi_depth':       {min: 0,   max: 100000, step: 1,     value: 84},
    'koi_model_snr':   {min: 0,   max: 1000,   step: 1,     value: 50},
    'prad_srad_ratio': {min: 0,   max: 100,    step: 0.01,  value: 0.009},
    'teq_derived':     {min: 0,   max: 3000,   step: 1,     value: 255},
    'insol':           {min: 0,   max: 1000000,step: 0.01,  value: 1.0},
    'koi_steff':       {min: 2000,max: 10000,  step: 1,     value: 5778},
    'koi_srad':        {min: 0.1, max: 10,     step: 0.01,  value: 1.0}
  };

  // DOM
  const presetSelect = document.querySelector('.preset-select');
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

  // Planet visualization elements
  const percentageValue = document.querySelector('.percentage-value');
  const predictionStatus = document.querySelector('.prediction-status');
  const planetRing = document.querySelector('.planet-ring');
  const gradientLight = document.querySelector('.planet-gradient-light');
  const gradientMain = document.querySelector('.planet-gradient-main');
  const gradientDark = document.querySelector('.planet-gradient-dark');

  // Debounce timer for real-time prediction
  let predictionTimer = null;

  function getPlanetColors(probability) {
    if (probability >= 75) {
      return { light: '#60a5fa', main: '#3b82f6', dark: '#1e40af' };
    }
    if (probability >= 50) {
      return { light: '#fb923c', main: '#f97316', dark: '#c2410c' };
    }
    if (probability >= 25) {
      return { light: '#ef4444', main: '#dc2626', dark: '#991b1b' };
    }
    return { light: '#6b7280', main: '#4b5563', dark: '#1f2937' };
  }

  function updatePlanetVisualization(probability) {
    const colors = getPlanetColors(probability);
    
    // Update gradient colors
    gradientLight.style.stopColor = colors.light;
    gradientMain.style.stopColor = colors.main;
    gradientDark.style.stopColor = colors.dark;
    
    // Update ring progress
    const circumference = 2 * Math.PI * 140; // 879.6
    const progress = (probability / 100) * circumference;
    planetRing.setAttribute('stroke-dasharray', `${progress} ${circumference}`);
    planetRing.setAttribute('stroke', colors.main);
    
    // Update percentage text
    percentageValue.textContent = `${probability}%`;
  }

  async function predictFromSliders() {
    predictionStatus.textContent = 'Calculating...';
    predictionStatus.classList.add('loading');
    
    const row = {};
    beginnerColumns.forEach(col => {
      const slider = document.getElementById(`slider_${col}`);
      row[col] = parseFloat(slider.value);
    });
    
    try {
      const resp = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(row)
      });
      
      if (!resp.ok) throw new Error();
      const data = await resp.json();
      const probability = (data.prob * 100).toFixed(2);
      
      updatePlanetVisualization(parseFloat(probability));
      predictionStatus.textContent = 'Real-time prediction from CNN model';
      predictionStatus.classList.remove('loading');
    } catch (error) {
      predictionStatus.textContent = 'Prediction failed';
      predictionStatus.classList.remove('loading');
      console.error('Prediction error:', error);
    }
  }

  function clearSliders() { 
    Object.values(sliderContainers).forEach(c => c.innerHTML = ''); 
  }
  
  function buildSliders(initial = null) {
    clearSliders();
    beginnerColumns.forEach(col => {
      const wrap = document.createElement('div');
      wrap.className = 'slider-div';
      const label = document.createElement('label'); 
      label.textContent = simpleDescriptions[col];
      const sliderWrapper = document.createElement('div'); 
      sliderWrapper.className = 'slider-wrapper';
      const slider = document.createElement('input');
      slider.type = 'range'; 
      slider.id = `slider_${col}`;
      const r = sliderRanges[col];
      slider.min = r.min; 
      slider.max = r.max; 
      slider.step = r.step; 
      slider.value = initial?.[col] ?? r.value;
      const valueSpan = document.createElement('span'); 
      valueSpan.textContent = slider.value;
      
      slider.addEventListener('input', () => {
        valueSpan.textContent = slider.value;
        
        // Debounced prediction
        clearTimeout(predictionTimer);
        predictionTimer = setTimeout(() => {
          predictFromSliders();
        }, 500);
      });
      
      sliderWrapper.appendChild(slider); 
      sliderWrapper.appendChild(valueSpan);
      wrap.appendChild(label); 
      wrap.appendChild(sliderWrapper);
      sliderContainers[col].appendChild(wrap);
    });
    
    // Initial prediction
    setTimeout(() => predictFromSliders(), 100);
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
      opt.value = i; 
      opt.textContent = p.kepler_name || `Preset ${i+1}`;
      presetSelect.appendChild(opt);
    });
    buildSliders();
    presetSelect.onchange = () => {
      const idx = presetSelect.value;
      if (idx === '') { 
        buildSliders(); 
      } else {
        const preset = presets[idx];
        buildSliders(preset);
      }
    };
    showSection('beginner');
  });

  /* ========= SCIENTIST ========= */
  scientistBtn.addEventListener('click', (e) => { 
    e.preventDefault(); 
    showSection('scientist'); 
  });
  const scientistForm = document.getElementById('scientist-form');
  const scientistResult = document.getElementById('scientist-result');
  if (scientistForm) {
    scientistForm.addEventListener('submit', async (e) => {
      e.preventDefault(); 
      scientistResult.textContent = 'Predicting...';
      const formData = new FormData(scientistForm); 
      const payload={}; 
      formData.forEach((v,k)=>payload[k]=parseFloat(v));
      try {
        const resp = await fetch('http://127.0.0.1:5000/predict_scientist', {
          method:'POST', 
          headers:{'Content-Type':'application/json'}, 
          body: JSON.stringify(payload)
        });
        if (!resp.ok) throw new Error();
        const data = await resp.json();
        scientistResult.textContent = `Prediction: ${data.prediction}`;
      } catch { 
        scientistResult.textContent = 'Prediction failed.'; 
      }
    });
  }

  /* ========= MODAL FOR VIDEO VISUALIZATION ========= */
  const showVisualizationBtn = document.querySelector('.show-visualization');
  const videoModal = document.querySelector('#videoModal');
  const closeModal = document.querySelector('.close-modal');

  if (showVisualizationBtn && videoModal && closeModal) {
    showVisualizationBtn.addEventListener('click', () => {
      videoModal.style.display = 'flex';
    });

    closeModal.addEventListener('click', () => {
      videoModal.style.display = 'none';
      const video = videoModal.querySelector('.transit-video');
      if (video) {
        video.pause();
        video.currentTime = 0;
      }
    });

    videoModal.addEventListener('click', (e) => {
      if (e.target === videoModal) {
        videoModal.style.display = 'none';
        const video = videoModal.querySelector('.transit-video');
        if (video) {
          video.pause();
          video.currentTime = 0;
        }
      }
    });
  }

  /* ========= NASA EYES IFRAME MODAL ========= */
  const showNasaEyesBtn = document.querySelector('.show-nasa-eyes');
  const nasaEyesModal = document.querySelector('#nasaEyesModal');
  const closeNasaEyes = document.querySelector('.close-nasa-eyes');

  if (showNasaEyesBtn && nasaEyesModal && closeNasaEyes) {
    showNasaEyesBtn.addEventListener('click', () => {
      nasaEyesModal.style.display = 'flex';
      const iframe = nasaEyesModal.querySelector('.nasa-eyes-iframe');
      if (iframe && !iframe.src) {
        iframe.src = 'https://eyes.nasa.gov/apps/exo/?destinations=/alien-worlds/strange-new-worlds/#/planet/Kepler-22_b/';
      }
    });

    closeNasaEyes.addEventListener('click', () => {
      nasaEyesModal.style.display = 'none';
    });

    nasaEyesModal.addEventListener('click', (e) => {
      if (e.target === nasaEyesModal) {
        nasaEyesModal.style.display = 'none';
      }
    });
  }

  /* ========= SCROLL OBSERVER FOR SECTION BLOCKS ========= */
  const sectionBlocks = document.querySelectorAll('.section-block');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        const video = entry.target.querySelector('.section-video');
        if (video) {
          video.style.opacity = 1;
        }
      } else {
        entry.target.classList.remove('visible');
        const video = entry.target.querySelector('.section-video');
        if (video) {
          video.style.opacity = 0;
        }
      }
    });
  }, { threshold: 0.5 });

  sectionBlocks.forEach(block => observer.observe(block));
});