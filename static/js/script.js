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
  window.scrollTo(0, 0);
  
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
    const animatable = section.querySelectorAll('h1, h2, .btn, .level-message, .level-buttons, .intro-text, .section-block');
    animatable.forEach(el => {
      el.classList.remove('absorb', 'slide-out');
      el.style.opacity = '0';
      el.style.transform = 'translateY(-50px)';
      el.style.animation = 'none';
      el.offsetHeight;
      el.style.animation = '';
      el.style.animation = 'floatInFromTop 0.8s ease-out forwards';
    });
  }

  function hideAll() { 
    Object.values(sections).forEach(sec => { 
      sec.classList.remove('active'); 
      sec.style.display = 'none'; 
    }); 
  }

  function showSection(sectionId, pushState = true) {
    hideAll();
    sections[sectionId].style.display = 'block';
    sections[sectionId].classList.add('active');
    resetAnimations(sectionId);
    if (pushState) history.pushState({ section: sectionId }, '', `#${sectionId}`);
    if (sectionId !== 'beginner') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  document.querySelectorAll('nav .nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const sectionId = link.getAttribute('href').substring(1);
      showSection(sectionId);
    });
  });

  window.addEventListener('popstate', (e) => {
    const sectionId = e.state?.section || 'getStarted';
    hideAll();
    sections[sectionId].style.display = 'block';
    sections[sectionId].classList.add('active');
    resetAnimations(sectionId);
    if (sectionId !== 'beginner') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  });

  hideAll();
  sections.getStarted.style.display = 'block';
  sections.getStarted.classList.add('active');
  history.replaceState({ section: 'getStarted' }, '', '#getStarted');

  const getStartedBtn = document.querySelector('#get-started-section .btn');
  getStartedBtn.addEventListener('click', (e) => {
    e.preventDefault();
    const h1 = document.querySelector('#get-started-section h1');
    getStartedBtn.classList.add('slide-out');
    h1.classList.add('absorb');
    showSection('levelSelection');
  });

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

  const percentageValue = document.querySelector('.percentage-value');
  const predictionStatus = document.querySelector('.prediction-status');
  const planetRing = document.querySelector('.planet-ring');
  const gradientLight = document.querySelector('.planet-gradient-light');
  const gradientMain = document.querySelector('.planet-gradient-main');
  const gradientDark = document.querySelector('.planet-gradient-dark');

  let predictionTimer = null;

  // Achievement system
  const achievementNotification = document.getElementById('achievementNotification');
  const achievementSound = document.getElementById('achievementSound');
  const achievementStatus = document.querySelector('.achievement-status');
  const achievementText = document.querySelector('.achievement-text');
  const achievementCount = document.getElementById('achievementCount');
  
  // Achievement definitions
const achievements = {
    firstPlanet: {
      id: 'firstPlanet',
      name: 'The First Whisper',
      description: 'Discover your first exoplanet candidate',
      icon: 'media/achievement_icon.png'
    },
    cosmicStillness: {
      id: 'cosmicStillness',
      name: 'Cosmic Stillness',
      description: 'You did nothing for 5 minutes. The cosmos appreciates your serenity.',
      icon: 'media/achievement_icon.png'
    },
    edgeOfUniverse: {
      id: 'edgeOfUniverse',
      name: 'Edge of the Universe',
      description: 'Read to the very end. Not every researcher makes it this far.',
      icon: 'media/achievement_icon.png'
    },
    almostGod: {
      id: 'almostGod',
      name: 'Almost a God Model',
      description: '98%. Your prediction made Kepler stand up and applaud.',
      icon: 'media/achievement_icon.png'
    },
    emptySkies: {
      id: 'emptySkies',
      name: 'Empty Skies',
      description: 'Five attempts—and all misses. Don’t be sad: even the cosmos is sometimes silent.',
      icon: 'media/achievement_icon.png'
    }
  };
  
  // Load unlocked achievements from localStorage
  let unlockedAchievements = JSON.parse(localStorage.getItem('unlockedAchievements') || '[]');
  
  // Activity tracking
  let lastActivityTime = Date.now();
  let inactivityTimer = null;
  let lowPredictionCount = 0;
  
function updateAchievementStatus() {
    const remaining = Object.keys(achievements).length - unlockedAchievements.length;
    if (remaining === 0) {
      achievementStatus.classList.add('completed');
      achievementText.innerHTML = 'All achievements unlocked!';
    } else {
      achievementStatus.classList.remove('completed');
      achievementCount.textContent = remaining;
      achievementText.innerHTML = `<span id="achievementCount">${remaining}</span> hidden achievements remaining`;
    }
  }
  
  function unlockAchievement(achievementId) {
    if (unlockedAchievements.includes(achievementId)) return;
    
    const achievement = achievements[achievementId];
    if (!achievement) return;
    
    unlockedAchievements.push(achievementId);
    localStorage.setItem('unlockedAchievements', JSON.stringify(unlockedAchievements));
    
    // Update status
    updateAchievementStatus();
    
    // Update notification content
    document.querySelector('.achievement-name').textContent = achievement.name;
    document.querySelector('.achievement-description').textContent = achievement.description;
    document.querySelector('.achievement-image').src = achievement.icon;
    
    // Play sound
    achievementSound.currentTime = 0;
    achievementSound.play().catch(e => console.log('Audio play failed:', e));
    
    // Show notification
    achievementNotification.classList.add('show');
    
    // Hide after 5 seconds
    setTimeout(() => {
      achievementNotification.classList.remove('show');
    }, 5000);
  }
  
  // Initialize achievement status
  updateAchievementStatus();
  
  // Track user activity for Cosmic Stillness achievement
  function resetInactivityTimer() {
    lastActivityTime = Date.now();
    clearTimeout(inactivityTimer);
    
    inactivityTimer = setTimeout(() => {
      unlockAchievement('cosmicStillness');
    }, 5 * 60 * 1000); // 5 minutes
  }
  
  // Add activity listeners
  ['mousemove', 'keydown', 'scroll', 'click'].forEach(eventType => {
    document.addEventListener(eventType, resetInactivityTimer);
  });
  
  // Start inactivity timer
  resetInactivityTimer();

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
    
    gradientLight.style.stopColor = colors.light;
    gradientMain.style.stopColor = colors.main;
    gradientDark.style.stopColor = colors.dark;
    
    const circumference = 2 * Math.PI * 140;
    const progress = (probability / 100) * circumference;
    planetRing.setAttribute('stroke-dasharray', `${progress} ${circumference}`);
    planetRing.setAttribute('stroke', colors.main);
    
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
      
      // Check for achievements
      const prob = parseFloat(probability);
      
      // First Planet achievement (75% or higher)
      if (prob >= 75) {
        unlockAchievement('firstPlanet');
        lowPredictionCount = 0; // Reset low prediction counter
        
        // Show save planet section
        document.getElementById('savePlanetSection').style.display = 'block';
      } else {
        // Hide save planet section if probability drops below 75%
        document.getElementById('savePlanetSection').style.display = 'none';
      }
      
      // Almost a God Model achievement (98% or higher)
      if (prob >= 98) {
        unlockAchievement('almostGod');
      }
      
      // Empty Skies achievement (5 consecutive predictions below 25%)
      if (prob < 25) {
        lowPredictionCount++;
        if (lowPredictionCount >= 5) {
          unlockAchievement('emptySkies');
        }
      } else if (prob >= 25) {
        lowPredictionCount = 0; // Reset if we get a decent prediction
      }
    } catch (error) {
      predictionStatus.textContent = 'Prediction failed';
      predictionStatus.classList.remove('loading');
      console.error('Prediction error:', error);
    }
  }

  // Save Planet functionality
  const savePlanetBtn = document.getElementById('savePlanetBtn');
  const planetNameInput = document.getElementById('planetName');
  const saveMessage = document.getElementById('saveMessage');
  
  if (savePlanetBtn) {
    savePlanetBtn.addEventListener('click', async () => {
      const planetName = planetNameInput.value.trim();
      
      if (!planetName) {
        saveMessage.textContent = 'Please enter a planet name';
        saveMessage.className = 'save-message error';
        return;
      }
      
      // Get current slider values
      const planetData = {
        planet_name: planetName
      };
      
      beginnerColumns.forEach(col => {
        const slider = document.getElementById(`slider_${col}`);
        if (slider) {
          planetData[col] = parseFloat(slider.value);
        }
      });
      
      // Disable button during save
      savePlanetBtn.disabled = true;
      saveMessage.textContent = 'Generating CSV...';
      saveMessage.className = 'save-message';
      
      try {
        const resp = await fetch('http://127.0.0.1:5000/save_planet', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(planetData)
        });
        
        if (!resp.ok) throw new Error('Server error');
        
        const data = await resp.json();
        
        // Create download link
        const blob = new Blob([data.csv_data], { type: 'text/csv;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        setTimeout(() => {
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
        }, 100);
        
        saveMessage.textContent = `✓ ${data.message}`;
        saveMessage.className = 'save-message success';
        planetNameInput.value = '';
        
        // Re-enable button after 2 seconds
        setTimeout(() => {
          savePlanetBtn.disabled = false;
          saveMessage.textContent = '';
        }, 2000);
        
      } catch (error) {
        saveMessage.textContent = '✗ Failed to save planet';
        saveMessage.className = 'save-message error';
        savePlanetBtn.disabled = false;
        console.error('Save error:', error);
      }
    });
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
    
    setTimeout(() => predictFromSliders(), 100);
  }

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
  const scientistFieldIds = {
    'koi_period': 'sci_koi_period',
    'koi_duration': 'sci_koi_duration',
    'koi_depth': 'sci_koi_depth',
    'koi_prad': 'sci_koi_prad',
    'koi_teq': 'sci_koi_teq',
    'koi_insol': 'sci_koi_insol',
    'koi_steff': 'sci_koi_steff',
    'koi_srad': 'sci_koi_srad',
    'koi_kepmag': 'sci_koi_kepmag'
  };

  scientistBtn.addEventListener('click', (e) => { 
    e.preventDefault(); 
    showSection('scientist'); 
  });

  // Load Data Button
  const loadDataBtn = document.getElementById('loadDataBtn');
  const loadDataFile = document.getElementById('loadDataFile');
  
  loadDataBtn.addEventListener('click', () => {
    loadDataFile.click();
  });

  loadDataFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const lines = text.split('\n');
        
        if (lines.length < 2) {
          alert('Invalid CSV file');
          return;
        }

        // Parse CSV header and first data row
        const headers = lines[0].split(',').map(h => h.trim());
        const values = lines[1].split(',').map(v => v.trim());
        
        // Map CSV columns to form fields
        Object.keys(scientistFieldIds).forEach(key => {
          const colIndex = headers.indexOf(key);
          if (colIndex !== -1 && values[colIndex]) {
            const field = document.getElementById(scientistFieldIds[key]);
            if (field) {
              field.value = values[colIndex];
            }
          }
        });

        const resultDisplay = document.getElementById('scientistResult');
        resultDisplay.innerHTML = '<div class="result-placeholder">Data loaded! Click Predict to analyze.</div>';
        
      } catch (error) {
        alert('Error reading CSV file');
        console.error('CSV parse error:', error);
      }
    };
    
    reader.readAsText(file);
  });

  // Predict Button
  const predictScientistBtn = document.getElementById('predictScientistBtn');
  const scientistResult = document.getElementById('scientistResult');
  
  predictScientistBtn.addEventListener('click', async () => {
    const payload = {};
    let hasEmptyFields = false;

    Object.keys(scientistFieldIds).forEach(key => {
      const field = document.getElementById(scientistFieldIds[key]);
      if (field && field.value) {
        payload[key] = parseFloat(field.value);
      } else {
        hasEmptyFields = true;
      }
    });

    if (hasEmptyFields) {
      scientistResult.innerHTML = '<div class="result-placeholder" style="color: #ef4444;">Please fill all fields</div>';
      return;
    }

    scientistResult.innerHTML = '<div class="result-loading">Analyzing...</div>';

    try {
      const resp = await fetch('http://127.0.0.1:5000/predict_scientist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) throw new Error();
      
      const data = await resp.json();
      const isConfirmed = data.prediction === 'CONFIRMED';
      const resultClass = isConfirmed ? 'result-confirmed' : 'result-candidate';
      const icon = isConfirmed ? '✓' : '⚠';
      
      scientistResult.innerHTML = `<div class="${resultClass}">${icon} ${data.prediction}</div>`;
      
    } catch (error) {
      scientistResult.innerHTML = '<div class="result-placeholder" style="color: #ef4444;">Prediction failed</div>';
      console.error('Prediction error:', error);
    }
  });
  /* ========= SCROLL OBSERVER FOR SECTION BLOCKS ========= */
  const sectionBlocks = document.querySelectorAll('.section-block');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        const video = entry.target.querySelector('.section-video');
        const image = entry.target.querySelector('.section-image');
        if (video) video.style.opacity = 1;
        if (image) image.style.opacity = 1;
      } else {
        entry.target.classList.remove('visible');
        const video = entry.target.querySelector('.section-video');
        const image = entry.target.querySelector('.section-image');
        if (video) video.style.opacity = 0;
        if (image) image.style.opacity = 0;
      }
    });
  }, { threshold: 0.5 });

  sectionBlocks.forEach(block => observer.observe(block));
  
  // Edge of the Universe achievement - triggers when user scrolls to bottom of Learn More
sectionBlocks.forEach(block => observer.observe(block));
  
  // Edge of the Universe achievement - triggers when user scrolls to bottom of Learn More
  const resourcesBlock = document.querySelector('.resources-block');
  
  if (resourcesBlock) {
    const edgeObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          console.log('Resources block visible - unlocking achievement');
          unlockAchievement('edgeOfUniverse');
        }
      });
    }, { threshold: 0.3 }); // Знижено з 0.8 до 0.3 для надійності
    
    edgeObserver.observe(resourcesBlock);
  } else {
    console.log('Resources block not found');
  }
});