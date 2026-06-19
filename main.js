/**
 * HouseAI – Professional UI JavaScript
 * Handles: scroll reveals, card tilt, sliders, toggles, API, animations
 */

(function () {
  'use strict';

  /* ── Scroll progress bar ──────────────────────────────────── */
  const progressBar = document.getElementById('scroll-progress');
  function updateProgress() {
    const max = document.body.scrollHeight - window.innerHeight;
    if (progressBar && max > 0) {
      progressBar.style.width = (window.scrollY / max * 100).toFixed(2) + '%';
    }
  }
  window.addEventListener('scroll', updateProgress, { passive: true });

  /* ── Scroll-triggered reveals ─────────────────────────────── */
  const revealObs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        revealObs.unobserve(e.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll('.reveal, .reveal-3d').forEach(el => revealObs.observe(el));

  /* ── Nav smooth scroll ────────────────────────────────────── */
  document.querySelectorAll('[data-scroll]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.dataset.scroll);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // Active state
      document.querySelectorAll('.nav-link').forEach(n => n.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  /* ── Mouse-tracked card tilt ──────────────────────────────── */
  document.querySelectorAll('.card-tilt').forEach(card => {
    const shine = card.querySelector('.card-shine');

    card.addEventListener('mousemove', (e) => {
      const r  = card.getBoundingClientRect();
      const x  = (e.clientX - r.left) / r.width;   // 0–1
      const y  = (e.clientY - r.top)  / r.height;  // 0–1
      const rx = (y - 0.5) * -10;  // ±5deg
      const ry = (x - 0.5) *  10;

      card.style.transform =
        `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateZ(4px)`;

      if (shine) {
        card.style.setProperty('--shine-x', (x * 100).toFixed(1) + '%');
        card.style.setProperty('--shine-y', (y * 100).toFixed(1) + '%');
      }
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });

  /* ── Range sliders ────────────────────────────────────────── */
  function initSlider(id, displayId, fmt) {
    const el  = document.getElementById(id);
    const out = document.getElementById(displayId);
    if (!el || !out) return;
    const update = () => { out.textContent = fmt ? fmt(el.value) : el.value; };
    el.addEventListener('input', update);
    update();
  }

  initSlider('school_rating',     'school_display',   v => (+v).toFixed(1));
  initSlider('crime_rate',        'crime_display',    v => (+v).toFixed(1));
  initSlider('hoa_fee',           'hoa_display',      v => '$' + Number(+v).toLocaleString());
  initSlider('distance_downtown', 'distance_display', v => (+v).toFixed(1) + ' mi');

  /* ── Toggle buttons ───────────────────────────────────────── */
  document.querySelectorAll('[data-tog]').forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.dataset.tog;
      const hiddenInput = document.getElementById(`${group}_val`);
      document.querySelectorAll(`[data-tog="${group}"]`).forEach(b => b.classList.remove('on'));
      btn.classList.add('on');
      if (hiddenInput) hiddenInput.value = btn.dataset.val;
    });
  });

  /* ── Neighborhood → coordinates ──────────────────────────── */
  const COORDS = {
    'Luxury Hills':  { lat: 34.07, lon: -118.44 },
    'Waterfront':    { lat: 37.80, lon: -122.43 },
    'Tech District': { lat: 37.39, lon: -122.08 },
    'Downtown':      { lat: 40.71, lon: -74.01  },
    'Midtown':       { lat: 40.75, lon: -73.98  },
    'University':    { lat: 42.37, lon: -71.11  },
    'Old Town':      { lat: 38.90, lon: -77.03  },
    'Suburbs':       { lat: 41.86, lon: -87.97  },
    'Industrial':    { lat: 41.49, lon: -81.69  },
    'Rural':         { lat: 39.16, lon: -86.52  },
  };

  const nbSel = document.getElementById('neighborhood');
  if (nbSel) {
    nbSel.addEventListener('change', () => {
      const c = COORDS[nbSel.value];
      if (c) {
        const latEl = document.getElementById('lat');
        const lonEl = document.getElementById('lon');
        if (latEl) latEl.value = c.lat;
        if (lonEl) lonEl.value = c.lon;
      }
    });
  }

  /* ── UI State helpers ─────────────────────────────────────── */
  const $ = id => document.getElementById(id);

  function showError(msg) {
    const eb = $('err-banner'), et = $('err-text');
    if (eb && et) { et.textContent = msg; eb.style.display = 'flex'; }
    $('price-reveal').style.display   = 'none';
    $('breakdown').style.display      = 'none';
    $('feat-bars-wrap').style.display = 'none';
    $('empty-state').style.display    = 'flex';
    setTimeout(() => { if (eb) eb.style.display = 'none'; }, 7000);
  }

  function hideError() {
    const eb = $('err-banner');
    if (eb) eb.style.display = 'none';
  }

  /* ── Animated number counter ──────────────────────────────── */
  function countUp(el, target, ms = 900) {
    const t0 = performance.now();
    requestAnimationFrame(function tick(now) {
      const p  = Math.min((now - t0) / ms, 1);
      const ep = 1 - Math.pow(1 - p, 3);  // ease out cubic
      el.textContent = '$' + Math.round(ep * target).toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
    });
  }

  /* ── Feature bar renderer ─────────────────────────────────── */
  const FEAT_NAME_MAP = {
    tier_x_sqft:      'Tier × Sqft',
    geo_price_median: 'Geo Price Median',
    sqft_x_condition: 'Sqft × Condition',
    log_sqft:         'log(Sqft)',
    sqft:             'Square Footage',
    neighborhood_tier:'Neighborhood Tier',
    school_x_tier:    'School × Tier',
    cost_index:       'Cost Index',
    sqft_x_bath:      'Sqft × Bathrooms',
    amenity_score:    'Amenity Score',
    crime_rate:       'Crime Rate',
    hoa_fee:          'HOA Fee',
    age:              'Property Age',
  };

  function renderFeatureBars(features) {
    const container = $('feat-bars-inner');
    if (!container || !features?.length) return;
    const max = features[0].importance || 1;

    container.innerHTML = features.slice(0, 8).map(f => `
      <div class="feat-row">
        <div class="feat-meta">
          <span class="feat-name">${FEAT_NAME_MAP[f.feature] || f.feature.replace(/_/g,' ')}</span>
          <span class="feat-pct">${(f.importance * 100).toFixed(1)}%</span>
        </div>
        <div class="feat-track">
          <div class="feat-fill" data-w="${((f.importance / max) * 100).toFixed(1)}%"></div>
        </div>
      </div>`).join('');

    // Animate bars after paint
    requestAnimationFrame(() => {
      container.querySelectorAll('.feat-fill').forEach(b => {
        b.style.width = b.dataset.w;
      });
    });
  }

  /* ── Collect form data ────────────────────────────────────── */
  function collectFormData() {
    const g = id => { const e = document.getElementById(id); return e ? e.value : ''; };
    return {
      sqft:              g('sqft'),
      bedrooms:          g('bedrooms'),
      bathrooms:         g('bathrooms'),
      year_built:        g('year_built'),
      garage_spaces:     g('garage_spaces'),
      has_pool:          g('has_pool_val'),
      has_basement:      g('has_basement_val'),
      floors:            g('floors'),
      condition:         g('condition'),
      lot_size:          g('lot_size'),
      distance_downtown: g('distance_downtown'),
      school_rating:     g('school_rating'),
      crime_rate:        g('crime_rate'),
      neighborhood:      g('neighborhood'),
      renovated:         g('renovated_val'),
      hoa_fee:           g('hoa_fee'),
      lat:               g('lat'),
      lon:               g('lon'),
    };
  }

  function validate(d) {
    if (!d.sqft || +d.sqft < 200)    return 'Square footage must be at least 200 sqft.';
    if (!d.bedrooms || +d.bedrooms < 1)  return 'At least 1 bedroom is required.';
    if (!d.bathrooms || +d.bathrooms < 1) return 'At least 1 bathroom is required.';
    if (!d.year_built || +d.year_built < 1800 || +d.year_built > 2024)
      return 'Year built must be between 1800 and 2024.';
    return null;
  }

  /* ── Form submission ──────────────────────────────────────── */
  const form = document.getElementById('prediction-form');
  const btn  = document.getElementById('btn-predict');

  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const data = collectFormData();
      const err  = validate(data);
      if (err) { showError(err); return; }

      btn.classList.add('loading');
      btn.disabled = true;
      hideError();

      try {
        const res    = await fetch('/api/predict', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(data),
        });
        const result = await res.json();

        if (!res.ok || result.error) {
          showError(result.error || 'Prediction failed. Please try again.');
          return;
        }

        /* ── Show price ── */
        $('empty-state').style.display    = 'none';
        const pr = $('price-reveal');
        pr.style.display = 'block';
        pr.style.animation = 'none';
        void pr.offsetWidth;
        pr.style.animation = '';

        const priceEl = $('price-main');
        priceEl.style.animation = 'none';
        void priceEl.offsetWidth;
        priceEl.style.animation = 'price-in 0.55s cubic-bezier(0.34,1.56,0.64,1) both';
        countUp(priceEl, result.predicted_price);

        $('price-range-txt').innerHTML =
          `Confidence range: <strong>${result.formatted_range}</strong>`;

        /* ── Breakdown ── */
        const ppsqft = (result.predicted_price / +data.sqft).toFixed(0);
        $('bd-sqft').textContent = Number(+data.sqft).toLocaleString() + ' sqft';
        $('bd-beds').textContent = data.bedrooms + ' / ' + data.bathrooms;
        $('bd-nbhd').textContent = data.neighborhood;
        $('bd-ppsq').textContent = '$' + ppsqft + ' / sqft';
        $('bd-cond').textContent = data.condition;
        $('bd-year').textContent = data.year_built;
        $('breakdown').style.display = 'block';

        /* ── Feature bars ── */
        if (window._topFeatures) {
          renderFeatureBars(window._topFeatures);
          $('feat-bars-wrap').style.display = 'block';
        }

        /* ── Scroll result into view ── */
        document.getElementById('result-card')
          .scrollIntoView({ behavior: 'smooth', block: 'nearest' });

      } catch (fetchErr) {
        showError('Network error: ' + fetchErr.message);
      } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
      }
    });
  }

  /* ── Load model metadata on boot ─────────────────────────── */
  fetch('/api/metadata')
    .then(r => r.json())
    .then(meta => {
      window._topFeatures = meta.top_features;
      const r2El   = document.getElementById('live-r2');
      const rmseEl = document.getElementById('live-rmse');
      const nEl    = document.getElementById('live-n');
      if (r2El   && meta.r2_score)  r2El.textContent   = meta.r2_score;
      if (rmseEl && meta.rmse)      rmseEl.textContent  = '$' + Math.round(+meta.rmse).toLocaleString();
      if (nEl    && meta.n_samples) nEl.textContent     = Math.round(+meta.n_samples / 1000) + 'K+';
    })
    .catch(() => { /* silent – model may not be trained yet */ });

})();
