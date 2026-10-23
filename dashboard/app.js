/**
 * SolarClean AI — Complete Decision Support Logic
 * Connects to live Open-Meteo weather and local REST backend.
 * Features:
 *  1. Smart Cleaning Decision Engine (with prominent "Analyze & Recommend Cleaning" trigger)
 *  2. Rainfall Effectiveness Analyzer (historical 2022 event study & comparison)
 *  3. What-If Cleaning Simulator (Strategy A, B, C dynamic modeling & comparison)
 *  4. Historical PV Performance Chart (SRR and PI validation)
 */

let currentChartInstance = null;
let currentChartMode = 'soiling'; // 'soiling' | 'pi'
let rainfallChartInstance = null;
let historicalDataCache = null;
let rainfallDataCache = null;

let activeConfig = {
  wash_cost: 150.0,
  tariff: 0.15,
  capacity_kw: 3.0,
  fixed_interval: 45,
  clean_threshold: 7.5,
  strategy: 'B'
};

document.addEventListener('DOMContentLoaded', () => {
  initDashboard();
});

async function initDashboard() {
  let liveStatus = null;
  let historical = null;
  let rainfallAnalysis = null;
  let simulation = null;

  // 1. Fetch live status from API endpoint
  try {
    const res = await fetch('/api/live-status');
    if (res.ok) liveStatus = await res.json();
  } catch (e) {
    console.warn('Backend API not reachable over HTTP, using static fallback...');
  }

  // Fallback to static bundle if API failed or accessed via file://
  if (!liveStatus && window.SOLARCLEAN_INITIAL_DATA) {
    liveStatus = window.SOLARCLEAN_INITIAL_DATA.live_status;
    historical = window.SOLARCLEAN_INITIAL_DATA.historical;
    rainfallAnalysis = window.SOLARCLEAN_INITIAL_DATA.rainfall_analysis;
    simulation = window.SOLARCLEAN_INITIAL_DATA.simulation;
  }

  if (liveStatus) {
    renderLiveStatus(liveStatus);
    if (liveStatus.recommendation && liveStatus.recommendation.strategy_simulation) {
      simulation = liveStatus.recommendation.strategy_simulation;
    }
  } else {
    renderUnavailableState();
  }

  // 2. Load historical chart data
  if (!historical) {
    try {
      const res = await fetch('/api/historical');
      if (res.ok) historical = await res.json();
    } catch (e) {
      console.warn('Could not fetch historical API');
    }
  }

  if (historical && historical.records) {
    historicalDataCache = historical.records;
    renderHistoricalChart(historical.records, currentChartMode);
  }

  // 3. Load Rainfall Effectiveness Analyzer data
  if (!rainfallAnalysis) {
    try {
      const res = await fetch('/api/rainfall-analysis');
      if (res.ok) rainfallAnalysis = await res.json();
    } catch (e) {
      console.warn('Could not fetch rainfall-analysis API');
    }
  }

  if (rainfallAnalysis) {
    rainfallDataCache = rainfallAnalysis;
    renderRainfallAnalyzer(rainfallAnalysis);
  }

  // 4. Load What-If Simulator data
  if (!simulation) {
    try {
      const res = await fetch('/api/simulate-strategies');
      if (res.ok) simulation = await res.json();
    } catch (e) {
      console.warn('Could not fetch simulate-strategies API');
    }
  }

  if (simulation && simulation.strategies) {
    renderSimulator(simulation);
  }

  // 5. Load Weather-Aware Screening data
  let screening = null;
  try {
    const res = await fetch('/api/screening-status');
    if (res.ok) screening = await res.json();
  } catch (e) {
    console.warn('Could not fetch screening-status API');
  }

  if (!screening && window.SOLARCLEAN_INITIAL_DATA && window.SOLARCLEAN_INITIAL_DATA.screening) {
    screening = window.SOLARCLEAN_INITIAL_DATA.screening;
  }

  if (screening) {
    renderScreeningStatus(screening);
  }

  // 6. Initialize Interactive Controls
  initAnalyzeButton();
  initChartTabs();
  initConfigSliders();
}

/* ==========================================================================
   1. Smart Cleaning Decision Engine & Trigger Button
   ========================================================================== */
function initAnalyzeButton() {
  const btn = document.getElementById('btn-analyze-recommend');
  const spinner = document.getElementById('btn-spinner');
  const icon = document.getElementById('btn-icon');
  const label = document.getElementById('btn-label');

  if (!btn) return;

  btn.addEventListener('click', async () => {
    // 1. Show loading state
    btn.disabled = true;
    spinner.style.display = 'inline-block';
    icon.style.display = 'none';
    label.textContent = 'Evaluating Live Data...';

    const pill = document.getElementById('rec-status-pill');
    pill.className = 'rec-pill dim';
    pill.textContent = 'ANALYZING...';

    try {
      // Execute live meteorological & decision evaluation
      const payload = {
        operational_config: {
          urgent_clean_loss_pct: activeConfig.clean_threshold,
          electricity_tariff_usd_kwh: activeConfig.tariff,
          manual_wash_cost_usd: activeConfig.wash_cost
        },
        system_specs: {
          capacity_kw: activeConfig.capacity_kw
        }
      };

      // Query live status or evaluation
      let res = await fetch('/api/live-status');
      let liveData = null;
      if (res.ok) {
        liveData = await res.json();
      }

      // Also evaluate with current activeConfig
      const evalRes = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      let evalData = null;
      if (evalRes.ok) {
        evalData = await evalRes.json();
      }

      // Also run simulator with activeConfig
      const simPayload = {
        tariff_usd: activeConfig.tariff,
        wash_cost_usd: activeConfig.wash_cost,
        capacity_kw: activeConfig.capacity_kw,
        fixed_interval_days: activeConfig.fixed_interval,
        current_loss_pct: evalData && evalData.metrics ? evalData.metrics.estimated_soiling_loss_pct : undefined,
        forecast_rain_mm: evalData && evalData.metrics ? evalData.metrics.rain_next_72h_mm : undefined
      };

      const simRes = await fetch('/api/simulate-strategies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(simPayload)
      });
      if (simRes.ok) {
        const simData = await simRes.json();
        renderSimulator(simData);
      }

      if (evalData) {
        updateHeroWithEvaluation(evalData);
      } else if (liveData) {
        renderLiveStatus(liveData);
      }

      // Success feedback
      label.textContent = 'Analysis Complete!';
      setTimeout(() => {
        label.textContent = 'Analyze & Recommend Cleaning';
      }, 1500);

    } catch (err) {
      console.error('Error during analysis trigger:', err);
      label.textContent = 'Connection Error';
      setTimeout(() => {
        label.textContent = 'Analyze & Recommend Cleaning';
      }, 2000);
    } finally {
      btn.disabled = false;
      spinner.style.display = 'none';
      icon.style.display = 'inline-block';
    }
  });
}

function renderLiveStatus(data) {
  const rec = data.recommendation;
  const weather = data.weather;

  // Header status
  const apiText = document.getElementById('api-status-text');
  const apiPulse = document.getElementById('api-pulse');

  if (weather && weather.status === 'ONLINE') {
    apiText.innerHTML = `Open-Meteo Live &bull; Updated ${formatTime(weather.current.time)}`;
    apiPulse.className = 'pulse-indicator online';
  } else if (weather && weather.status === 'STALE_OFFLINE') {
    apiText.innerHTML = `Cached Weather &bull; Stale`;
    apiPulse.className = 'pulse-indicator stale';
  } else {
    apiText.textContent = 'Weather Unavailable';
    apiPulse.className = 'pulse-indicator offline';
  }

  updateHeroWithEvaluation(rec);

  // Live Weather Block
  if (weather && weather.current) {
    const curr = weather.current;
    document.getElementById('live-temp').textContent = `${curr.temperature_c.toFixed(1)}°C`;
    document.getElementById('live-condition').textContent = curr.weather_desc;
    document.getElementById('live-humidity').textContent = `${curr.relative_humidity_pct}%`;
    document.getElementById('live-wind').textContent = `${curr.wind_speed_kmh} km/h`;
    document.getElementById('live-rain').textContent = `${curr.precipitation_mm} mm`;

    renderForecastStrip(weather.daily_forecast);
  }
}

function updateHeroWithEvaluation(rec) {
  if (!rec) return;

  const heroCard = document.getElementById('rec-hero-card');
  const pill = document.getElementById('rec-status-pill');
  const actionTitle = document.getElementById('rec-action-title');
  const reasonText = document.getElementById('rec-reason-text');
  const timeText = document.getElementById('rec-timestamp');
  const confTag = document.getElementById('rec-confidence-tag');

  const color = rec.status_color || 'dim';
  heroCard.className = `card recommendation-hero ${color}`;
  pill.className = `rec-pill ${color}`;
  pill.textContent = rec.recommendation;
  actionTitle.textContent = rec.action_title;
  const confText = rec.data_confidence === 'high' ? 'Verified Feed' : (rec.data_confidence === 'none' ? 'No Feed' : 'Standard Feed');
  confTag.textContent = `Telemetry: ${confText}`;

  const now = new Date();
  timeText.textContent = `Updated: ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;

  // Metrics
  if (rec.metrics) {
    const m = rec.metrics;
    document.getElementById('val-soiling-ratio').textContent = `${(m.estimated_soiling_ratio * 100).toFixed(1)}%`;
    document.getElementById('val-soiling-loss').textContent = `Est. Loss: ${m.estimated_soiling_loss_pct.toFixed(1)}%`;
    document.getElementById('val-dry-streak').textContent = `${m.consecutive_dry_days} Days`;
    document.getElementById('val-last-rain').textContent = `Last Rain: ${m.last_rain_date}`;
    document.getElementById('val-rain-outlook').textContent = `${m.rain_next_72h_mm.toFixed(1)} mm`;
    document.getElementById('val-rain-prob').textContent = `Prob: ${m.rain_prob_72h_pct}%`;
  }

  // Financials
  if (rec.financials) {
    const f = rec.financials;
    document.getElementById('val-daily-revenue').textContent = `$${f.daily_loss_usd.toFixed(2)}/day`;
    document.getElementById('val-monthly-revenue').textContent = `Monthly: $${f.monthly_loss_usd.toFixed(2)}`;
  }

  // Exact Limitation Note
  if (rec.uncertainty_note) {
    const discEl = document.getElementById('rec-disclosure-text');
    if (discEl) discEl.textContent = rec.uncertainty_note;
  }
}

function renderScreeningStatus(data) {
  if (!data) return;
  const d = data.daily || {};
  const hf = data.high_frequency || {};

  const elUsable = document.getElementById('scr-usable-days');
  const elSnow = document.getElementById('scr-snow-days');
  const elLowIrr = document.getElementById('scr-low-irr-obs');
  const elCloud = document.getElementById('scr-cloud-obs');
  const elModTemp = document.getElementById('scr-mod-temp');

  if (elUsable && d.usable_clean_days !== undefined) {
    elUsable.textContent = d.usable_clean_days;
  }
  if (elSnow && d.potential_snow_inferred_days !== undefined) {
    elSnow.textContent = d.potential_snow_inferred_days;
  }
  if (elLowIrr && hf.low_irradiance_excluded !== undefined) {
    elLowIrr.textContent = hf.low_irradiance_excluded.toLocaleString();
  }
  if (elCloud && hf.cloud_unstable_excluded !== undefined) {
    elCloud.textContent = hf.cloud_unstable_excluded.toLocaleString();
  }
  if (elModTemp && hf.module_temp_sensor_count !== undefined) {
    elModTemp.textContent = `${hf.module_temp_sensor_count} Channels`;
  }
}

function renderForecastStrip(forecastDays) {
  const container = document.getElementById('forecast-strip');
  if (!container || !forecastDays) return;

  container.innerHTML = '';
  forecastDays.slice(0, 7).forEach(d => {
    const card = document.createElement('div');
    card.className = `forecast-day-card ${d.is_today ? 'today' : ''}`;

    const dateLabel = d.is_today ? 'Today' : formatDateShort(d.date);
    card.innerHTML = `
      <span class="f-date">${dateLabel}</span>
      <span class="f-temp">${Math.round(d.temp_max_c)}° / ${Math.round(d.temp_min_c)}°</span>
      <span class="f-rain">${d.precipitation_mm.toFixed(1)} mm</span>
      <span class="f-prob">${d.rain_probability_pct}% rain</span>
    `;
    container.appendChild(card);
  });
}

function renderUnavailableState() {
  const pill = document.getElementById('rec-status-pill');
  pill.className = 'rec-pill dim';
  pill.textContent = 'INSUFFICIENT DATA';
  document.getElementById('rec-action-title').textContent = 'Unable to Load Live Decision Support';
  document.getElementById('rec-reason-text').textContent = 'Live weather feeds and cached operational baselines could not be established. Direct panel sensors are not connected.';
}

/* ==========================================================================
   2. Rainfall Effectiveness Analyzer
   ========================================================================== */
function renderRainfallAnalyzer(data) {
  if (!data) return;

  const summary = data.summary || {};
  const events = data.events || [];

  // Update KPI counters
  if (summary.total_rain_events_identified !== undefined) {
    document.getElementById('ra-total-events').textContent = summary.total_rain_events_identified;
  }
  if (summary.comparable_events_count !== undefined) {
    document.getElementById('ra-comparable-events').textContent = summary.comparable_events_count;
  }
  if (summary.increased_performance_events !== undefined) {
    document.getElementById('ra-increased-events').textContent = summary.increased_performance_events;
  }
  if (summary.decreased_performance_events !== undefined) {
    document.getElementById('ra-decreased-events').textContent = summary.decreased_performance_events;
  }
  if (summary.neutral_performance_events !== undefined) {
    const neutralCount = summary.neutral_performance_events + (summary.inconclusive_events_count || 0);
    document.getElementById('ra-neutral-events').textContent = neutralCount;
  }
  if (summary.mean_pi_change !== undefined) {
    document.getElementById('ra-mean-pi').textContent = (summary.mean_pi_change >= 0 ? '+' : '') + summary.mean_pi_change.toFixed(3);
  }

  // Populate events table
  const tbody = document.getElementById('rainfall-table-body');
  if (tbody) {
    tbody.innerHTML = '';
    events.forEach(ev => {
      const tr = document.createElement('tr');

      const preStr = ev.pre_event_pi !== null ? ev.pre_event_pi.toFixed(3) : 'N/A';
      const postStr = ev.post_event_pi !== null ? ev.post_event_pi.toFixed(3) : 'N/A';
      const deltaStr = ev.delta_pi !== null ? (ev.delta_pi >= 0 ? '+' : '') + ev.delta_pi.toFixed(3) : 'Inconclusive';

      let badgeClass = 'neutral';
      let badgeLabel = 'Neutral';

      if (!ev.is_comparable) {
        badgeClass = 'inconclusive';
        badgeLabel = 'Inconclusive (Data Gap)';
      } else if (ev.classification === 'INCREASED') {
        badgeClass = 'increased';
        badgeLabel = 'Performance Increased';
      } else if (ev.classification === 'DECREASED') {
        badgeClass = 'decreased';
        badgeLabel = 'Performance Decreased';
      } else {
        badgeClass = 'neutral';
        badgeLabel = 'Minimal / Neutral Change';
      }

      tr.innerHTML = `
        <td>${ev.event_date}</td>
        <td><strong>${ev.precipitation_mm.toFixed(1)} mm</strong></td>
        <td>${preStr}</td>
        <td>${postStr}</td>
        <td><span class="${ev.delta_pi > 0 ? 'text-green' : (ev.delta_pi < 0 ? 'text-rose' : '')}">${deltaStr}</span></td>
        <td><span class="status-badge ${badgeClass}">${badgeLabel}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Render Rainfall Comparison Bar Chart (Top 14 representative rainfall events)
  renderRainfallComparisonChart(events.filter(e => e.is_comparable).slice(0, 14));
}

function renderRainfallComparisonChart(comparableEvents) {
  const canvas = document.getElementById('rainfallComparisonChart');
  if (!canvas || !window.Chart) return;

  const ctx = canvas.getContext('2d');
  if (rainfallChartInstance) {
    rainfallChartInstance.destroy();
  }

  const labels = comparableEvents.map(e => `${formatDateShort(e.event_date)} (${e.precipitation_mm}mm)`);
  const deltas = comparableEvents.map(e => e.delta_pi);
  const colors = deltas.map(d => d >= 0 ? 'rgba(16, 185, 129, 0.75)' : 'rgba(244, 63, 94, 0.75)');
  const borderColors = deltas.map(d => d >= 0 ? '#10B981' : '#F43F5E');

  rainfallChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Historical Performance Index Change (ΔPI)',
        data: deltas,
        backgroundColor: colors,
        borderColor: borderColors,
        borderWidth: 1.5,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#F8FAFC',
          bodyColor: '#94A3B8',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 8,
          callbacks: {
            label: (ctx) => `ΔPI: ${(ctx.raw >= 0 ? '+' : '')}${ctx.raw.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#64748B',
            font: { family: 'Plus Jakarta Sans', size: 9 },
            maxRotation: 45
          }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: {
            color: '#94A3B8',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (v) => `${(v >= 0 ? '+' : '')}${v.toFixed(2)}`
          }
        }
      }
    }
  });
}

/* ==========================================================================
   3. What-If Cleaning Simulator
   ========================================================================== */
function renderSimulator(sim) {
  if (!sim || !sim.strategies) return;

  const a = sim.strategies.strategy_a;
  const b = sim.strategies.strategy_b;
  const c = sim.strategies.strategy_c;

  // Strategy A
  if (a) {
    document.getElementById('strat-a-cost').textContent = `$${a.cleaning_cost_usd.toFixed(2)}`;
    document.getElementById('strat-a-loss').textContent = `${a.energy_loss_kwh.toFixed(1)} kWh`;
    document.getElementById('strat-a-recov').textContent = `${a.recoverable_energy_kwh.toFixed(1)} kWh`;
    const netEl = document.getElementById('strat-a-net');
    netEl.textContent = `${a.net_financial_benefit_usd >= 0 ? '+' : ''}$${a.net_financial_benefit_usd.toFixed(2)}`;
    netEl.className = `val ${a.net_financial_benefit_usd >= 0 ? 'text-green' : 'text-rose'}`;
    document.getElementById('strat-a-desc').textContent = a.description;
  }

  // Strategy B
  if (b) {
    document.getElementById('strat-b-cost').textContent = `$${b.cleaning_cost_usd.toFixed(2)} (Free)`;
    document.getElementById('strat-b-rain').textContent = `${b.forecast_rain_mm.toFixed(1)} mm`;
    document.getElementById('strat-b-loss').textContent = `${b.energy_loss_kwh.toFixed(1)} kWh`;
    document.getElementById('strat-b-recov').textContent = `${b.recoverable_energy_kwh.toFixed(1)} kWh`;
    const netEl = document.getElementById('strat-b-net');
    netEl.textContent = `${b.net_financial_benefit_usd >= 0 ? '+' : ''}$${b.net_financial_benefit_usd.toFixed(2)}`;
    netEl.className = `val ${b.net_financial_benefit_usd >= 0 ? 'text-green' : 'text-rose'}`;
    document.getElementById('strat-b-desc').textContent = b.description;
  }

  // Strategy C
  if (c) {
    document.getElementById('strat-c-action').textContent = `Clean every ${c.cleaning_interval_days} days`;
    document.getElementById('strat-c-cost').textContent = `$${c.cleaning_cost_usd.toFixed(2)}`;
    document.getElementById('strat-c-loss').textContent = `${c.energy_loss_kwh.toFixed(1)} kWh`;
    document.getElementById('strat-c-recov').textContent = `${c.recoverable_energy_kwh.toFixed(1)} kWh`;
    const netEl = document.getElementById('strat-c-net');
    netEl.textContent = `${c.net_financial_benefit_usd >= 0 ? '+' : ''}$${c.net_financial_benefit_usd.toFixed(2)}`;
    netEl.className = `val ${c.net_financial_benefit_usd >= 0 ? 'text-green' : 'text-rose'}`;
    document.getElementById('strat-c-desc').textContent = c.description;
  }

  // Highlight optimal strategy card (recommended) and user-selected card
  const optId = sim.optimal_strategy_id;
  const selId = sim.selected_strategy_id || optId;
  const cards = {
    'A': document.getElementById('card-strat-a'),
    'B': document.getElementById('card-strat-b'),
    'C': document.getElementById('card-strat-c')
  };
  const selTags = {
    'A': document.getElementById('strat-a-sel-tag'),
    'B': document.getElementById('strat-b-sel-tag'),
    'C': document.getElementById('strat-c-sel-tag')
  };
  const optTags = {
    'A': document.getElementById('strat-a-opt-tag'),
    'B': document.getElementById('strat-b-opt-tag'),
    'C': document.getElementById('strat-c-opt-tag')
  };

  Object.keys(cards).forEach(key => {
    if (cards[key]) {
      // Optimal card styling + badge
      if (key === optId) {
        cards[key].classList.add('recommended');
      } else {
        cards[key].classList.remove('recommended');
      }
      // User-selected card styling + badge
      if (key === selId) {
        cards[key].classList.add('selected');
      } else {
        cards[key].classList.remove('selected');
      }
      // Show/hide SELECTED badge
      if (selTags[key]) {
        selTags[key].style.display = (key === selId) ? 'inline-block' : 'none';
      }
      // Show/hide OPTIMAL badge
      if (optTags[key]) {
        optTags[key].style.display = (key === optId) ? 'inline-block' : 'none';
      }
    }
  });

  // Synthesis Box — clearly distinguish selected vs optimal
  const titleEl = document.getElementById('synthesis-optimal-title');
  const textEl = document.getElementById('synthesis-optimal-text');
  const badgeEl = document.querySelector('.synthesis-badge');
  if (titleEl) {
    titleEl.textContent = `Selected Strategy: Strategy ${selId} — ${sim.selected_strategy_name || sim.optimal_strategy_name}`;
  }
  if (badgeEl) {
    badgeEl.textContent = 'USER SELECTION';
  }
  if (textEl && sim.synthesis) {
    textEl.textContent = sim.synthesis;
  }
}

function initConfigSliders() {
  const sWash = document.getElementById('cfg-wash-cost');
  const sTariff = document.getElementById('cfg-tariff');
  const sCap = document.getElementById('cfg-capacity');
  const sInterval = document.getElementById('cfg-fixed-interval');
  const sThresh = document.getElementById('cfg-clean-threshold');
  const sStrategy = document.getElementById('cfg-strategy');

  const dWash = document.getElementById('disp-wash-cost');
  const dTariff = document.getElementById('disp-tariff');
  const dCap = document.getElementById('disp-capacity');
  const dInterval = document.getElementById('disp-fixed-interval');
  const dThresh = document.getElementById('disp-clean-thresh');
  const dStrategy = document.getElementById('disp-strategy');

  const btnReset = document.getElementById('btn-reset-simulator');

  let debounceTimer = null;

  async function handleConfigChange() {
    activeConfig.wash_cost = parseFloat(sWash.value);
    activeConfig.tariff = parseFloat(sTariff.value);
    activeConfig.capacity_kw = parseFloat(sCap.value);
    activeConfig.fixed_interval = parseInt(sInterval.value, 10);
    activeConfig.clean_threshold = parseFloat(sThresh.value);
    if (sStrategy) {
      activeConfig.strategy = sStrategy.value;
      dStrategy.textContent = `Strategy ${activeConfig.strategy}`;
    }

    dWash.textContent = `$${activeConfig.wash_cost.toFixed(0)}`;
    dTariff.textContent = `$${activeConfig.tariff.toFixed(2)}`;
    dCap.textContent = `${activeConfig.capacity_kw.toFixed(1)} kW`;
    dInterval.textContent = `${activeConfig.fixed_interval} Days`;
    dThresh.textContent = `${activeConfig.clean_threshold.toFixed(1)}%`;

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      const simErrorBanner = document.getElementById('sim-error-banner');
      const simErrorMsg = document.getElementById('sim-error-msg');

      // 1. Post to simulation endpoint
      try {
        const simPayload = {
          wash_cost_usd: activeConfig.wash_cost,
          tariff_usd: activeConfig.tariff,
          capacity_kw: activeConfig.capacity_kw,
          fixed_interval_days: activeConfig.fixed_interval,
          strategy: activeConfig.strategy
        };

        const res = await fetch('/api/simulate-strategies', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(simPayload)
        });
        if (res.ok) {
          const simData = await res.json();
          renderSimulator(simData);
          if (simErrorBanner) simErrorBanner.style.display = 'none';
        } else {
          throw new Error(`Server returned ${res.status}`);
        }
      } catch (e) {
        console.warn('Could not post to /api/simulate-strategies:', e);
        if (simErrorBanner && simErrorMsg) {
          simErrorMsg.textContent = 'Simulation request failed. Showing last cached data. Check server connection.';
          simErrorBanner.style.display = 'flex';
        }
      }

      // 2. Also update decision evaluation
      try {
        const evalPayload = {
          operational_config: {
            urgent_clean_loss_pct: activeConfig.clean_threshold,
            electricity_tariff_usd_kwh: activeConfig.tariff,
            manual_wash_cost_usd: activeConfig.wash_cost
          },
          system_specs: {
            capacity_kw: activeConfig.capacity_kw
          }
        };

        const res = await fetch('/api/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(evalPayload)
        });
        if (res.ok) {
          const rec = await res.json();
          updateHeroWithEvaluation(rec);
        }
      } catch (e) {
        console.warn('Could not post to /api/evaluate');
      }
    }, 120);
  }

  [sWash, sTariff, sCap, sInterval, sThresh, sStrategy].forEach(inp => {
    if (inp) inp.addEventListener(sStrategy === inp ? 'change' : 'input', handleConfigChange);
  });

  if (btnReset) {
    btnReset.addEventListener('click', () => {
      sWash.value = 150;
      sTariff.value = 0.15;
      sCap.value = 3.0;
      sInterval.value = 45;
      sThresh.value = 7.5;
      if (sStrategy) {
        sStrategy.value = 'B';
        dStrategy.textContent = 'Strategy B';
      }
      activeConfig.strategy = 'B';
      handleConfigChange();
    });
  }
}

/* ==========================================================================
   4. Restored Historical Baseline Chart (Chart.js)
   ========================================================================== */
function initChartTabs() {
  const tabs = document.querySelectorAll('.chart-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', (e) => {
      tabs.forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      currentChartMode = e.target.dataset.chart;
      if (historicalDataCache) {
        renderHistoricalChart(historicalDataCache, currentChartMode);
      }
    });
  });
}

function renderHistoricalChart(records, mode) {
  const canvas = document.getElementById('historicalChart');
  if (!canvas || !window.Chart) return;

  const ctx = canvas.getContext('2d');
  if (currentChartInstance) {
    currentChartInstance.destroy();
  }

  const labels = records.map(r => r.date);
  const rainData = records.map(r => (r.precipitation_mm > 0 ? r.precipitation_mm : 0));

  let primaryDataset = {};
  if (mode === 'soiling') {
    primaryDataset = {
      type: 'line',
      label: 'Estimated Soiling Ratio (SR)',
      data: records.map(r => r.soiling_ratio),
      borderColor: '#F59E0B',
      backgroundColor: 'rgba(245, 158, 11, 0.1)',
      borderWidth: 2,
      pointRadius: records.map(r => (r.is_recovery_event ? 5 : 0)),
      pointBackgroundColor: records.map(r => (r.is_recovery_event ? '#10B981' : '#F59E0B')),
      pointBorderColor: '#FFFFFF',
      pointBorderWidth: 1.5,
      pointStyle: records.map(r => (r.is_recovery_event ? 'rectRot' : 'circle')),
      spanGaps: false,
      yAxisID: 'y'
    };
  } else {
    primaryDataset = {
      type: 'line',
      label: 'Performance Index (Normalized Yield / Insolation)',
      data: records.map(r => (r.performance_index && r.performance_index > 0 && r.performance_index < 1.3 ? r.performance_index : null)),
      borderColor: '#06B6D4',
      backgroundColor: 'rgba(6, 182, 212, 0.1)',
      borderWidth: 1.5,
      pointRadius: 2,
      pointBackgroundColor: '#06B6D4',
      spanGaps: false,
      yAxisID: 'y'
    };
  }

  const precipDataset = {
    type: 'bar',
    label: 'Daily Precipitation (mm)',
    data: rainData,
    backgroundColor: 'rgba(56, 189, 248, 0.4)',
    borderWidth: 0,
    borderRadius: 2,
    yAxisID: 'y1'
  };

  currentChartInstance = new Chart(ctx, {
    data: {
      labels: labels,
      datasets: [primaryDataset, precipDataset]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            color: '#94A3B8',
            font: { family: 'Plus Jakarta Sans', size: 11 }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#F8FAFC',
          bodyColor: '#94A3B8',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          bodyFont: { family: 'JetBrains Mono', size: 12 },
          callbacks: {
            label: function(context) {
              if (context.dataset.yAxisID === 'y') {
                const val = context.raw;
                return val !== null ? `${context.dataset.label}: ${(val * 100).toFixed(1)}%` : 'Missing / Excluded';
              } else {
                return `Precipitation: ${context.raw} mm`;
              }
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#64748B',
            font: { family: 'Plus Jakarta Sans', size: 10 },
            maxTicksLimit: 12
          }
        },
        y: {
          type: 'linear',
          position: 'left',
          min: mode === 'soiling' ? 0.70 : 0.0,
          max: mode === 'soiling' ? 1.05 : 1.25,
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: {
            color: '#94A3B8',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (v) => `${(v * 100).toFixed(0)}%`
          }
        },
        y1: {
          type: 'linear',
          position: 'right',
          min: 0,
          max: 35,
          grid: { drawOnChartArea: false },
          ticks: {
            color: '#38BDF8',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: (v) => `${v} mm`
          }
        }
      }
    }
  });
}

/* ==========================================================================
   Helper Utilities
   ========================================================================== */
function formatTime(isoStr) {
  if (!isoStr) return '';
  return isoStr.replace('T', ' ');
}

function formatDateShort(dateStr) {
  if (!dateStr) return '';
  const parts = dateStr.split('-');
  if (parts.length === 3) {
    const mNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const m = parseInt(parts[1], 10) - 1;
    return `${mNames[m]} ${parseInt(parts[2], 10)}`;
  }
  return dateStr;
}

/* ==========================================================================
   Screening Status Renderer
   ========================================================================== */
function renderScreeningStatus(data) {
  if (!data) return;
  const hf = data.high_frequency || {};
  const daily = data.daily || {};

  const usableDaysEl = document.getElementById('scr-usable-days');
  const snowDaysEl = document.getElementById('scr-snow-days');
  const lowIrrEl = document.getElementById('scr-low-irr-obs');
  const cloudObsEl = document.getElementById('scr-cloud-obs');
  const modTempEl = document.getElementById('scr-mod-temp');

  if (usableDaysEl && daily.usable_clean_days !== undefined) {
    usableDaysEl.textContent = daily.usable_clean_days;
  }
  if (snowDaysEl && daily.potential_snow_inferred_days !== undefined) {
    snowDaysEl.textContent = daily.potential_snow_inferred_days;
  }
  if (lowIrrEl && hf.low_irradiance_excluded !== undefined) {
    lowIrrEl.textContent = hf.low_irradiance_excluded.toLocaleString();
  }
  if (cloudObsEl && hf.cloud_unstable_excluded !== undefined) {
    cloudObsEl.textContent = hf.cloud_unstable_excluded.toLocaleString();
  }
  if (modTempEl && hf.module_temp_sensor_count !== undefined) {
    modTempEl.textContent = `${hf.module_temp_sensor_count} Channels`;
  }
}

// Apply tooltips to screening KPI cards
const cards = document.querySelectorAll('.analyzer-kpi-card[data-tooltip]');
cards.forEach(card => {
  const tip = card.getAttribute('data-tooltip');
  if (!tip) return;
  const badge = document.createElement('span');
  badge.className = 'kpi-tooltip';
  badge.textContent = '\u24D8'; // circled question mark icon
  badge.title = tip;
  const label = card.querySelector('.kpi-label');
  if (label) label.appendChild(document.createTextNode(' '));
  if (label) label.appendChild(badge);
});
