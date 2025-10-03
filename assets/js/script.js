var PARTICLE_NUM = 200;
var PARTICLE_BASE_RADIUS = 0.5;
var FL = 500;
var DEFAULT_SPEED = 2;
var BOOST_SPEED = 100;

var canvas;
var canvasWidth, canvasHeight;
var context;
var centerX, centerY;
var mouseX, mouseY;
var speed = DEFAULT_SPEED;
var targetSpeed = DEFAULT_SPEED;
var particles = [];

function update_position(e) {
  let _t = e.target.closest('.btn');
  if (_t) {
    let r = _t.getBoundingClientRect();
    ['x', 'y'].forEach(c => 
      _t.style.setProperty(`--${c}`, 
        `${e[`client${c.toUpperCase()}`] - r[c]}px`));
  }
}

window.addEventListener('load', async function() {
  canvas = document.getElementById('c');
  
  var resize = function() {
    canvasWidth = canvas.width = window.innerWidth;
    canvasHeight = canvas.height = window.innerHeight;
    centerX = canvasWidth * 0.5;
    centerY = canvasHeight * 0.5;
    context = canvas.getContext('2d');
    context.fillStyle = 'rgb(255, 255, 255)';
  };
  
  window.addEventListener('resize', resize);
  resize();
  
  mouseX = centerX;
  mouseY = centerY;
  
  for (var i = 0, p; i < PARTICLE_NUM; i++) {
    particles[i] = randomizeParticle(new Particle());
    particles[i].z -= 500 * Math.random();
  }
  
  document.addEventListener('mousemove', function(e) {
    mouseX = e.clientX;
    mouseY = e.clientY;
    update_position(e);
  }, false);
  
  document.addEventListener('mousedown', function(e) {
    targetSpeed = BOOST_SPEED;
  }, false);
  
  document.addEventListener('mouseup', function() {
    targetSpeed = DEFAULT_SPEED;
  }, false);
  
  document.addEventListener('mouseover', update_position);
  document.addEventListener('mouseout', update_position);

  setInterval(loop, 1000 / 120);

  const exoplanetInfo = document.querySelector('.exoplanet-info');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      } else {
        entry.target.classList.remove('visible');
      }
    });
  }, { threshold: 0.1 });
  observer.observe(exoplanetInfo);

  const allColumns = [
    'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_score',
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period',
    'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
    'koi_impact', 'koi_impact_err1', 'koi_impact_err2', 'koi_duration', 'koi_duration_err1',
    'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 'koi_depth_err2', 'koi_prad',
    'koi_prad_err1', 'koi_prad_err2', 'koi_teq', 'koi_teq_err1', 'koi_teq_err2',
    'koi_insol', 'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num',
    'koi_tce_delivname', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 'koi_slogg',
    'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2',
    'ra', 'dec', 'koi_kepmag'
  ];

  const beginnerColumns = ['koi_period', 'koi_duration', 'koi_impact', 'koi_depth', 'koi_model_snr', 
                          'prad_srad_ratio', 'teq_derived', 'insol', 'koi_steff', 'koi_srad'];

  const sliderRanges = {
    'koi_period': {min: 0.1, max: 1000, step: 0.001, value: 0.1},
    'koi_duration': {min: 0.1, max: 50, step: 0.01, value: 0.1},
    'koi_impact': {min: 0, max: 2, step: 0.01, value: 0},
    'koi_depth': {min: 0, max: 100000, step: 1, value: 0},
    'koi_model_snr': {min: 0, max: 1000, step: 1, value: 0},
    'prad_srad_ratio': {min: 0, max: 100, step: 0.01, value: 0},
    'teq_derived': {min: 0, max: 3000, step: 1, value: 0},
    'insol': {min: 0, max: 1000000, step: 0.01, value: 0},
    'koi_steff': {min: 2000, max: 10000, step: 1, value: 2000},
    'koi_srad': {min: 0.1, max: 10, step: 0.01, value: 0.1}
  };

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

  // Fetch presets
  let presets = [];
  try {
    const response = await fetch('http://127.0.0.1:5000/presets');
    if (response.ok) {
      presets = await response.json();
    } else {
      console.error('Failed to load presets');
    }
  } catch (error) {
    console.error('Error fetching presets:', error);
  }

  // Section management
  const sections = {
    getStarted: document.getElementById('get-started-section'),
    levelSelection: document.getElementById('level-selection-section'),
    beginner: document.getElementById('beginner-section'),
    scientist: document.getElementById('scientist-section')
  };

  function resetAnimations(sectionId) {
    const section = sections[sectionId];
    const animatableElements = section.querySelectorAll('h1, .btn, .level-message, .level-buttons');
    animatableElements.forEach(el => {
      el.classList.remove('absorb', 'slide-out');
      el.style.opacity = '0';
      el.style.transform = 'translateY(-50px)';
      el.style.animation = 'none';
      // Force reflow to reset animation
      el.offsetHeight;
      el.style.animation = '';
      // Re-apply floatInFromTop animation
      el.style.animation = 'floatInFromTop 1.5s ease-out forwards';
      // Adjust animation-delay based on element type
      if (el.tagName === 'H1') {
        el.style.animationDelay = '0.4s';
      } else if (el.classList.contains('btn')) {
        el.style.animationDelay = '0.6s';
      } else if (el.classList.contains('level-message')) {
        el.style.animationDelay = '0.2s';
      } else if (el.classList.contains('level-buttons')) {
        el.style.animationDelay = '0.4s';
      }
    });
  }

  function showSection(sectionId, pushState = true) {
    Object.values(sections).forEach(section => {
      section.classList.remove('active');
      section.style.display = 'none';
    });
    sections[sectionId].style.display = 'block';
    setTimeout(() => {
      sections[sectionId].classList.add('active');
      resetAnimations(sectionId);
    }, 10);
    if (pushState) {
      history.pushState({ section: sectionId }, '', `#${sectionId}`);
    }
  }

  // Handle browser back/forward
  window.addEventListener('popstate', function(event) {
    const sectionId = event.state?.section || 'getStarted';
    Object.values(sections).forEach(section => {
      section.classList.remove('active');
      section.style.display = 'none';
    });
    sections[sectionId].style.display = 'block';
    setTimeout(() => {
      sections[sectionId].classList.add('active');
      resetAnimations(sectionId);
    }, 10);
  });

  // Initialize: Show Get Started section
  showSection('getStarted', false);

  // Get Started button
  const getStartedBtn = document.querySelector('#get-started-section .btn');
  getStartedBtn.addEventListener('click', function(e) {
    e.preventDefault();
    const h1 = document.querySelector('h1');
    getStartedBtn.classList.add('slide-out');
    h1.classList.add('absorb');
    
    setTimeout(() => {
      showSection('levelSelection');
    }, 1200);
  });

  // Level selection buttons
  const beginnerBtn = document.querySelector('#level-selection-section .btn:nth-child(1)');
  const scientistBtn = document.querySelector('#level-selection-section .btn:nth-child(2)');

  beginnerBtn.addEventListener('click', function(e) {
    e.preventDefault();
    const levelMessage = document.querySelector('.level-message');
    const levelButtons = document.querySelector('.level-buttons');
    levelMessage.classList.add('absorb');
    levelButtons.classList.add('slide-out');
    
    setTimeout(() => {
      // Populate Beginner section
      const presetSelect = document.querySelector('.preset-select');
      presetSelect.innerHTML = '<option value="">Manual Input</option>';
      presets.forEach((preset, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = preset.kepler_name || `Preset ${index + 1}`;
        presetSelect.appendChild(option);
      });

      // Populate sliders
      const sliderContainers = {
        'koi_period': document.getElementById('orbital-sliders'),
        'koi_duration': document.getElementById('orbital-sliders'),
        'koi_impact': document.getElementById('orbital-sliders'),
        'koi_depth': document.getElementById('transit-sliders'),
        'koi_model_snr': document.getElementById('transit-sliders'),
        'prad_srad_ratio': document.getElementById('transit-sliders'),
        'teq_derived': document.getElementById('planet-sliders'),
        'insol': document.getElementById('planet-sliders'),
        'koi_steff': document.getElementById('stellar-sliders'),
        'koi_srad': document.getElementById('stellar-sliders')
      };

      beginnerColumns.forEach(col => {
        const sliderDiv = document.createElement('div');
        sliderDiv.className = 'slider-div';
        const label = document.createElement('label');
        label.textContent = col;
        const sliderWrapper = document.createElement('div');
        sliderWrapper.className = 'slider-wrapper';
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.id = `slider_${col}`;
        const range = sliderRanges[col];
        slider.min = range.min;
        slider.max = range.max;
        slider.step = range.step;
        slider.value = range.value;
        const valueSpan = document.createElement('span');
        valueSpan.textContent = slider.value;
        slider.addEventListener('input', () => {
          valueSpan.textContent = slider.value;
        });
        sliderWrapper.appendChild(slider);
        sliderWrapper.appendChild(valueSpan);
        sliderDiv.appendChild(label);
        sliderDiv.appendChild(sliderWrapper);
        sliderContainers[col].appendChild(sliderDiv);
      });

      // Populate parameters table
      const tbody = document.querySelector('.parameters-table tbody');
      tbody.innerHTML = '';
      beginnerColumns.forEach(col => {
        const row = document.createElement('tr');
        const td1 = document.createElement('td');
        td1.textContent = col;
        const td2 = document.createElement('td');
        td2.textContent = simpleDescriptions[col] || 'No description.';
        row.appendChild(td1);
        row.appendChild(td2);
        tbody.appendChild(row);
      });

      // Preset selection handler
      const nameInput = document.querySelector('.beginner-section input[type="text"]');
      presetSelect.addEventListener('change', () => {
        const index = presetSelect.value;
        if (index !== '') {
          const preset = presets[index];
          nameInput.value = preset.kepler_name || '';
          beginnerColumns.forEach(col => {
            const slider = document.getElementById(`slider_${col}`);
            slider.value = preset[col] || sliderRanges[col].value;
            slider.nextElementSibling.textContent = slider.value;
          });
        } else {
          nameInput.value = '';
          beginnerColumns.forEach(col => {
            const slider = document.getElementById(`slider_${col}`);
            slider.value = sliderRanges[col].value;
            slider.nextElementSibling.textContent = slider.value;
          });
        }
      });

      // Predict button handler
      const predictBtn = document.querySelector('.predict-btn');
      const predictionResult = document.querySelector('.prediction-result');
      predictBtn.addEventListener('click', async function(e) {
        e.preventDefault();
        predictionResult.textContent = 'Predicting...';
        predictionResult.style.display = 'block';
        
        const row = {};
        allColumns.forEach(col => {
          if (beginnerColumns.includes(col)) {
            row[col] = parseFloat(document.getElementById(`slider_${col}`).value);
          } else if (col === 'kepler_name') {
            row[col] = nameInput.value || '';
          } else if (col === 'koi_disposition' || col === 'koi_pdisposition') {
            row[col] = 'CANDIDATE';
          } else if (['kepoi_name', 'kepler_name', 'koi_tce_delivname'].includes(col)) {
            row[col] = '';
          } else if (col === 'koi_tce_delivname') {
            row[col] = 'q1_q17_dr25_tce';
          } else {
            row[col] = 0;
          }
        });
        
        try {
          const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(row)
          });
          
          if (!response.ok) {
            throw new Error('Prediction failed');
          }
          
          const result = await response.json();
          predictionResult.textContent = `Your planet has a ${(result.prob * 100).toFixed(2)}% chance of being an exoplanet candidate.`;
        } catch (error) {
          predictionResult.textContent = 'Prediction error. Ensure the backend server is running.';
        }
      });

      showSection('beginner');
    }, 1200);
  });

  scientistBtn.addEventListener('click', function(e) {
    e.preventDefault();
    const levelMessage = document.querySelector('.level-message');
    const levelButtons = document.querySelector('.level-buttons');
    levelMessage.classList.add('absorb');
    levelButtons.classList.add('slide-out');
    
    setTimeout(() => {
      showSection('scientist');
    }, 1200);
  });

  // Back button handlers
  document.querySelectorAll('.back-btn').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      const currentSection = btn.closest('.section');
      currentSection.classList.add('slide-out');
      setTimeout(() => {
        showSection('levelSelection');
      }, 1200);
    });
  });
}, false);

function loop() {
  context.save();
  context.fillStyle = 'rgb(0, 0, 0, 0.2)';
  context.fillRect(0, 0, canvasWidth, canvasHeight);
  context.restore();
  
  speed += (targetSpeed - speed) * 0.01;
  
  var p, cx, cy, rx, ry, f, x, y, r, pf, px, py, pr, a, a1, a2;
  var halfPi = Math.PI * 0.5;
  var atan2 = Math.atan2;
  var cos = Math.cos;
  var sin = Math.sin;
  
  context.beginPath();
  for (var i = 0; i < PARTICLE_NUM; i++) {
    p = particles[i];
    p.pastZ = p.z;
    p.z -= speed;
    
    if (p.z <= 0) {
      randomizeParticle(p);
      continue;
    }
    
    cx = centerX - (mouseX - centerX) * 1.25;
    cy = centerY - (mouseY - centerY) * 1.25;
    
    rx = p.x - cx;
    ry = p.y - cy;
    
    f = FL / p.z;
    x = cx + rx * f;
    y = cy + ry * f;
    r = PARTICLE_BASE_RADIUS * f;
    
    pf = FL / p.pastZ;
    px = cx + rx * pf;
    py = cy + ry * pf;
    pr = PARTICLE_BASE_RADIUS * pf;
    
    a = atan2(py - y, px - x);
    a1 = a + halfPi;
    a2 = a - halfPi;
    
    context.moveTo(px + pr * cos(a1), py + pr * sin(a1));
    context.arc(px, py, pr, a1, a2, true);
    context.lineTo(x + r * cos(a2), y + r * sin(a2));
    context.arc(x, y, r, a2, a1, true);
    context.closePath();
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