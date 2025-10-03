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
  
  document.addEventListener('resize', resize);
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
        observer.unobserve(entry.target);
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

  const beginnerColumns = ['koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_model_snr', 
                          'teq_derived', 'prad_srad_ratio', 'koi_steff', 'koi_srad', 'insol'];

  const sliderRanges = {
    'koi_period': {min: 0.1, max: 1000, step: 0.001, value: 0.1},
    'koi_impact': {min: 0, max: 2, step: 0.01, value: 0},
    'koi_duration': {min: 0.1, max: 50, step: 0.01, value: 0.1},
    'koi_depth': {min: 0, max: 100000, step: 1, value: 0},
    'koi_model_snr': {min: 0, max: 1000, step: 1, value: 0},
    'teq_derived': {min: 0, max: 3000, step: 1, value: 0},
    'prad_srad_ratio': {min: 0, max: 100, step: 0.01, value: 0},
    'koi_steff': {min: 2000, max: 10000, step: 1, value: 2000},
    'koi_srad': {min: 0.1, max: 10, step: 0.01, value: 0.1},
    'insol': {min: 0, max: 1000000, step: 0.01, value: 0}
  };

  const simpleDescriptions = {
    'koi_period': 'Orbital period of the planet (days).',
    'koi_impact': 'Impact parameter, related to transit trajectory.',
    'koi_duration': 'Duration of the planet’s transit (hours).',
    'koi_depth': 'Transit depth, how much the star dims (ppm).',
    'koi_model_snr': 'Signal-to-noise ratio of the transit model.',
    'teq_derived': 'Equilibrium temperature of the planet (Kelvin).',
    'prad_srad_ratio': 'Planet-to-star radius ratio.',
    'koi_steff': 'Stellar effective temperature (Kelvin).',
    'koi_srad': 'Stellar radius (solar radii).',
    'insol': 'Insolation, stellar flux received by the planet.'
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

  const btn = document.querySelector('.btn');
  const content = document.querySelector('.content');
  btn.addEventListener('click', function(e) {
    e.preventDefault();
    
    const subtitle = document.querySelector('.subtitle');
    const h1 = document.querySelector('h1');
    const exoplanetInfo = document.querySelector('.exoplanet-info');
    subtitle.classList.add('absorb');
    h1.classList.add('absorb');
    btn.classList.add('slide-out');
    exoplanetInfo.style.display = 'none';
    
    setTimeout(() => {
      subtitle.style.display = 'none';
      h1.style.display = 'none';
      btn.style.display = 'none';
      
      const levelMessage = document.createElement('div');
      levelMessage.className = 'level-message';
      levelMessage.textContent = 'New to space or a seasoned astronomer? Choose your level!';
      
      const levelButtons = document.createElement('div');
      levelButtons.className = 'level-buttons';
      
      const beginnerBtn = document.createElement('a');
      beginnerBtn.href = '#';
      beginnerBtn.className = 'btn';
      beginnerBtn.textContent = 'I’m a Beginner';
      
      const scientistBtn = document.createElement('a');
      scientistBtn.href = '#';
      scientistBtn.className = 'btn';
      scientistBtn.textContent = 'I’m a Scientist';
      
      levelButtons.appendChild(beginnerBtn);
      levelButtons.appendChild(scientistBtn);
      
      content.appendChild(levelMessage);
      content.appendChild(levelButtons);
      
      beginnerBtn.addEventListener('click', function(e) {
        e.preventDefault();
        levelMessage.classList.add('absorb');
        levelButtons.classList.add('slide-out');
        setTimeout(() => {
          levelMessage.style.display = 'none';
          levelButtons.style.display = 'none';
          
          const beginnerSection = document.createElement('div');
          beginnerSection.className = 'beginner-section';
          
          const title = document.createElement('h2');
          title.textContent = 'Create Your Exoplanet! Use sliders or choose a preset!';
          
          // Preset selection
          const presetLabel = document.createElement('label');
          presetLabel.textContent = 'Choose a preset:';
          const presetSelect = document.createElement('select');
          presetSelect.className = 'preset-select';
          presetSelect.innerHTML = '<option value="">Manual Input</option>';
          presets.forEach((preset, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${preset.kepoi_name} (${preset.koi_pdisposition})`;
            presetSelect.appendChild(option);
          });
          
          // Planet name input
          const nameLabel = document.createElement('label');
          nameLabel.textContent = 'Name your planet:';
          const nameInput = document.createElement('input');
          nameInput.type = 'text';
          nameInput.placeholder = 'e.g., MyAwesomePlanet b';
          
          // Sliders container
          const slidersContainer = document.createElement('div');
          slidersContainer.className = 'sliders-container';
          
          beginnerColumns.forEach(col => {
            const sliderDiv = document.createElement('div');
            sliderDiv.className = 'slider-div';
            
            const label = document.createElement('label');
            label.textContent = `${col}: `;
            
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
            
            sliderDiv.appendChild(label);
            sliderDiv.appendChild(slider);
            sliderDiv.appendChild(valueSpan);
            slidersContainer.appendChild(sliderDiv);
          });
          
          // Update sliders based on preset
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
          
          // Predict button
          const predictBtn = document.createElement('a');
          predictBtn.href = '#';
          predictBtn.className = 'btn';
          predictBtn.textContent = 'Predict';
          predictBtn.style.display = 'block';
          predictBtn.style.margin = '2rem auto'; /* Center button */
          
          // Prediction result div
          const predictionResult = document.createElement('div');
          predictionResult.className = 'prediction-result';
          predictionResult.style.display = 'none';
          
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
          
          const paramsTable = document.createElement('table');
          paramsTable.className = 'parameters-table';
          const thead = document.createElement('thead');
          const tbody = document.createElement('tbody');
          const headerRow = document.createElement('tr');
          const th1 = document.createElement('th');
          th1.textContent = 'Parameter';
          const th2 = document.createElement('th');
          th2.textContent = 'Description';
          headerRow.appendChild(th1);
          headerRow.appendChild(th2);
          thead.appendChild(headerRow);

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

          paramsTable.appendChild(thead);
          paramsTable.appendChild(tbody);

          beginnerSection.appendChild(title);
          beginnerSection.appendChild(presetLabel);
          beginnerSection.appendChild(presetSelect);
          beginnerSection.appendChild(nameLabel);
          beginnerSection.appendChild(nameInput);
          beginnerSection.appendChild(slidersContainer);
          beginnerSection.appendChild(predictBtn);
          beginnerSection.appendChild(predictionResult);
          beginnerSection.appendChild(paramsTable);
          
          content.appendChild(beginnerSection);
        }, 1200);
      });
      
      scientistBtn.addEventListener('click', function(e) {
        e.preventDefault();
        levelMessage.classList.add('absorb');
        levelButtons.classList.add('slide-out');
        setTimeout(() => {
          levelMessage.style.display = 'none';
          levelButtons.style.display = 'none';
          
          const fileBtn = document.createElement('a');
          fileBtn.href = '#';
          fileBtn.className = 'btn file-input';
          fileBtn.innerHTML = `
            <label for="file-upload">Select File</label>
            <input type="file" id="file-upload" accept=".csv,.txt,.json">
          `;
          
          const fileMessage = document.createElement('div');
          fileMessage.className = 'file-message';
          fileMessage.textContent = 'Upload an exoplanet dataset for analysis!';
          
          content.appendChild(fileMessage);
          content.appendChild(fileBtn);
        }, 1200);
      });
    }, 1200);
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