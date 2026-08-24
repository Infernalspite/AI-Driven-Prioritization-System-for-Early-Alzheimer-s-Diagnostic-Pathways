// Global state
let currentPatients = [];
let trajectoryChartInstance = null;
let causalChartInstance = null;
let orderedPathwayTests = ["cognitive"];

// Presets
const PRESETS = {
  early_mci: {
    age: 73.5, education_years: 16.0,
    MMSE: 24.0, ADAS13: 17.5, CDR_SB: 2.0,
    abeta42_40_ratio: 0.088, ptau181: 2.1,
    hippocampal_volume_mm3: 3250, cortical_thickness_mm: 2.32,
    pet_amyloid_suvr: 1.35,
    masks: { cognitive: true, blood: true, mri: true, pet: true }
  },
  borderline_ad: {
    age: 78.0, education_years: 12.0,
    MMSE: 19.5, ADAS13: 26.0, CDR_SB: 5.5,
    abeta42_40_ratio: 0.065, ptau181: 3.8,
    hippocampal_volume_mm3: 2650, cortical_thickness_mm: 2.05,
    pet_amyloid_suvr: 1.68,
    masks: { cognitive: true, blood: true, mri: true, pet: true }
  },
  healthy_cn: {
    age: 69.0, education_years: 18.0,
    MMSE: 29.0, ADAS13: 7.0, CDR_SB: 0.0,
    abeta42_40_ratio: 0.125, ptau181: 0.9,
    hippocampal_volume_mm3: 4100, cortical_thickness_mm: 2.75,
    pet_amyloid_suvr: 1.05,
    masks: { cognitive: true, blood: true, mri: true, pet: true }
  },
  missing_pet_mri: {
    age: 72.0, education_years: 14.0,
    MMSE: 22.5, ADAS13: 19.0, CDR_SB: 2.5,
    abeta42_40_ratio: null, ptau181: null,
    hippocampal_volume_mm3: null, cortical_thickness_mm: null,
    pet_amyloid_suvr: null,
    masks: { cognitive: true, blood: false, mri: false, pet: false }
  }
};

document.addEventListener("DOMContentLoaded", async () => {
  await fetchPatients();
  applyPreset('early_mci');
  initCausalChart();
  updateCapacityAllocation();
});

// Tab Switcher
function switchTab(tabId) {
  document.querySelectorAll(".nav-tab").forEach(b => b.classList.remove("active-tab"));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

  const btn = document.getElementById(`tab-btn-${tabId}`);
  const pane = document.getElementById(`tab-content-${tabId}`);
  if (btn) btn.classList.add("active-tab");
  if (pane) pane.classList.add("active");

  if (tabId === "trajectory") {
    loadTrajectoryForPatient();
  } else if (tabId === "causal") {
    initCausalChart();
  } else if (tabId === "population") {
    updateCapacityAllocation();
  }
}

// Fetch Patients
async function fetchPatients() {
  try {
    const res = await fetch("/api/patients?limit=60");
    const patients = await res.json();
    currentPatients = patients;

    const select = document.getElementById("cohort-patient-select");
    const trajSelect = document.getElementById("traj-patient-select");
    select.innerHTML = '<option value="">-- Load Existing Cohort Patient --</option>';
    trajSelect.innerHTML = '';

    patients.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.subject_id;
      opt.textContent = `${p.subject_id} (${p.baseline_diagnosis} -> ${p.latest_diagnosis}, ${p.n_visits} visits, Split: ${p.split})`;
      select.appendChild(opt);

      const trajOpt = opt.cloneNode(true);
      trajSelect.appendChild(trajOpt);
    });
  } catch (err) {
    console.error("Failed to fetch patients", err);
  }
}

async function loadSelectedPatient() {
  const sid = document.getElementById("cohort-patient-select").value;
  if (!sid) return;
  try {
    const res = await fetch(`/api/patients/${sid}`);
    const data = await res.json();
    if (!data.visits || data.visits.length === 0) return;

    const first = data.visits[0];
    document.getElementById("input-age").value = first.age || 72.0;
    document.getElementById("input-education_years").value = first.education_years || 14.0;

    // Set modality fields
    setModalityVal("MMSE", first.cognitive.MMSE);
    setModalityVal("ADAS13", first.cognitive.ADAS13);
    setModalityVal("CDR_SB", first.cognitive.CDR_SB);
    setModalityVal("abeta42_40_ratio", first.blood.abeta42_40_ratio);
    setModalityVal("ptau181", first.blood.ptau181);
    setModalityVal("hippocampal_volume_mm3", first.mri.hippocampal_volume_mm3);
    setModalityVal("cortical_thickness_mm", first.mri.cortical_thickness_mm);
    setModalityVal("pet_amyloid_suvr", first.pet.pet_amyloid_suvr);

    setModalityActive("cognitive", first.cognitive.MMSE !== null);
    setModalityActive("blood", first.blood.ptau181 !== null);
    setModalityActive("mri", first.mri.hippocampal_volume_mm3 !== null);
    setModalityActive("pet", first.pet.pet_amyloid_suvr !== null);

    runDiagnosis();
  } catch (err) {
    console.error("Error loading patient history", err);
  }
}

function setModalityVal(id, val) {
  const el = document.getElementById(`input-${id}`);
  if (el) el.value = val !== null && val !== undefined ? val : "";
}

function setModalityActive(mod, active) {
  const mask = document.getElementById(`mask-${mod}`);
  if (mask) mask.checked = active;
  toggleModality(mod);
}

function toggleModality(mod) {
  const active = document.getElementById(`mask-${mod}`).checked;
  const group = document.getElementById(`group-${mod}`);
  if (group) {
    group.style.opacity = active ? "1" : "0.35";
    group.style.pointerEvents = active ? "auto" : "none";
  }
}

function applyPreset(presetKey) {
  const p = PRESETS[presetKey];
  if (!p) return;
  document.getElementById("input-age").value = p.age;
  document.getElementById("input-education_years").value = p.education_years;

  setModalityVal("MMSE", p.MMSE);
  setModalityVal("ADAS13", p.ADAS13);
  setModalityVal("CDR_SB", p.CDR_SB);
  setModalityVal("abeta42_40_ratio", p.abeta42_40_ratio);
  setModalityVal("ptau181", p.ptau181);
  setModalityVal("hippocampal_volume_mm3", p.hippocampal_volume_mm3);
  setModalityVal("cortical_thickness_mm", p.cortical_thickness_mm);
  setModalityVal("pet_amyloid_suvr", p.pet_amyloid_suvr);

  for (const [mod, active] of Object.entries(p.masks)) {
    setModalityActive(mod, active);
  }
  runDiagnosis();
}

function getBiomarkerPayload() {
  const payload = {
    age: parseFloat(document.getElementById("input-age").value) || 72.0,
    education_years: parseFloat(document.getElementById("input-education_years").value) || 14.0,
  };

  const cognitiveOn = document.getElementById("mask-cognitive").checked;
  const bloodOn = document.getElementById("mask-blood").checked;
  const mriOn = document.getElementById("mask-mri").checked;
  const petOn = document.getElementById("mask-pet").checked;

  payload.MMSE = cognitiveOn ? (parseFloat(document.getElementById("input-MMSE").value) || null) : null;
  payload.ADAS13 = cognitiveOn ? (parseFloat(document.getElementById("input-ADAS13").value) || null) : null;
  payload.CDR_SB = cognitiveOn ? (parseFloat(document.getElementById("input-CDR_SB").value) || null) : null;

  payload.abeta42_40_ratio = bloodOn ? (parseFloat(document.getElementById("input-abeta42_40_ratio").value) || null) : null;
  payload.ptau181 = bloodOn ? (parseFloat(document.getElementById("input-ptau181").value) || null) : null;

  payload.hippocampal_volume_mm3 = mriOn ? (parseFloat(document.getElementById("input-hippocampal_volume_mm3").value) || null) : null;
  payload.cortical_thickness_mm = mriOn ? (parseFloat(document.getElementById("input-cortical_thickness_mm").value) || null) : null;

  payload.pet_amyloid_suvr = petOn ? (parseFloat(document.getElementById("input-pet_amyloid_suvr").value) || null) : null;

  return payload;
}

// ---------------------------------------------------------------------------
// TAB 1: RUN MULTIMODAL DIAGNOSIS
// ---------------------------------------------------------------------------
async function runDiagnosis() {
  const payload = getBiomarkerPayload();
  try {
    const res = await fetch("/api/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, n_mc_passes: 30 })
    });
    const result = await res.json();
    if (result.error) return alert(result.error);

    // Update Hero Card
    const badge = document.getElementById("diag-prediction-badge");
    const pred = result.prediction;
    badge.textContent = pred === "AD" ? "AD (Alzheimer's Disease)" : (pred === "MCI" ? "MCI (Mild Cognitive Impairment)" : "CN (Cognitively Normal)");
    badge.className = `text-2xl font-black px-3.5 py-1 rounded-xl border ${
      pred === "AD" ? "bg-rose-500/20 text-rose-300 border-rose-500/30" : (
        pred === "MCI" ? "bg-amber-500/20 text-amber-300 border-amber-500/30" : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
      )
    }`;

    document.getElementById("diag-confidence-text").textContent = `${(result.confidence * 100).toFixed(1)}%`;

    // Synergy
    const synergy = result.cross_modal_attention.modal_synergy_score || 0.0;
    document.getElementById("synergy-val").textContent = synergy.toFixed(4);

    // Class probabilities
    const probs = result.class_probabilities;
    document.getElementById("prob-cn-text").textContent = `${(probs.CN * 100).toFixed(1)}%`;
    document.getElementById("prob-cn-bar").style.width = `${probs.CN * 100}%`;
    document.getElementById("prob-mci-text").textContent = `${(probs.MCI * 100).toFixed(1)}%`;
    document.getElementById("prob-mci-bar").style.width = `${probs.MCI * 100}%`;
    document.getElementById("prob-ad-text").textContent = `${(probs.AD * 100).toFixed(1)}%`;
    document.getElementById("prob-ad-bar").style.width = `${probs.AD * 100}%`;

    // Uncertainty
    document.getElementById("aleatoric-val").textContent = result.uncertainty.aleatoric_uncertainty.toFixed(3);
    document.getElementById("epistemic-val").textContent = result.uncertainty.epistemic_uncertainty.toFixed(3);
    document.getElementById("total-entropy-val").textContent = result.uncertainty.total_entropy.toFixed(3);

    // Attention Token Cards
    renderAttentionTokens(result.cross_modal_attention.token_weights, result.cross_modal_attention.present_modalities);

    // Update Pathway
    updatePathwayDisplay();
  } catch (err) {
    console.error("Diagnosis error", err);
  }
}

function renderAttentionTokens(weights, presentMods) {
  const container = document.getElementById("attention-tokens-container");
  container.innerHTML = "";
  if (!weights) return;

  for (const [token, weight] of Object.entries(weights)) {
    const isPresent = presentMods.includes(token) || token === "CLS" || token === "static";
    const card = document.createElement("div");
    card.className = "token-card";
    card.innerHTML = `
      <div class="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-0.5">${token}</div>
      <div class="text-sm font-black text-cyan-400 font-mono">${(weight * 100).toFixed(1)}%</div>
      <div class="text-[9px] mt-1 ${isPresent ? 'text-emerald-400 font-medium' : 'text-slate-400'}">
        ${isPresent ? '● Present' : '○ Masked Token'}
      </div>
    `;
    container.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// TAB 2: DIGITAL TWIN TRAJECTORY
// ---------------------------------------------------------------------------
async function loadTrajectoryForPatient() {
  const sid = document.getElementById("traj-patient-select").value || (currentPatients[0] && currentPatients[0].subject_id);
  if (!sid) return;

  try {
    const [histRes, trajRes] = await Promise.all([
      fetch(`/api/patients/${sid}`),
      fetch(`/api/trajectory/${sid}?horizon=48`)
    ]);
    const hist = await histRes.json();
    const forecast = await trajRes.json();

    document.getElementById("traj-slope-val").textContent = `${forecast.projected_slope_cdrsb_per_year > 0 ? '+' : ''}${forecast.projected_slope_cdrsb_per_year} / yr`;
    document.getElementById("traj-conversion-val").textContent = forecast.estimated_time_to_ad_conversion_months ? `~${forecast.estimated_time_to_ad_conversion_months} Months` : 'Stable (> 48 Mos)';

    renderTrajectoryChart(hist.visits || [], forecast.trajectory || []);
  } catch (err) {
    console.error("Trajectory error", err);
  }
}

function renderTrajectoryChart(historicalVisits, forecastPoints) {
  const ctx = document.getElementById("trajectoryChart").getContext("2d");
  if (trajectoryChartInstance) trajectoryChartInstance.destroy();

  const histMonths = historicalVisits.map(v => v.visit_month);
  const histCdrsb = historicalVisits.map(v => v.cognitive.CDR_SB);

  const lastHistMonth = histMonths.length > 0 ? histMonths[histMonths.length - 1] : 0;
  const futureMonths = forecastPoints.map(p => lastHistMonth + p.month);
  const futureCdrsb = forecastPoints.map(p => p.projected_cdrsb);
  const lowerCi = forecastPoints.map(p => p.cdrsb_lower_ci);
  const upperCi = forecastPoints.map(p => p.cdrsb_upper_ci);

  const labels = Array.from(new Set([...histMonths, ...futureMonths])).sort((a, b) => a - b);

  trajectoryChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels.map(m => `M${m}`),
      datasets: [
        {
          label: 'Historical Recorded CDR-SB',
          data: labels.map(m => {
            const idx = histMonths.indexOf(m);
            return idx !== -1 ? histCdrsb[idx] : null;
          }),
          borderColor: '#10b981',
          backgroundColor: '#10b981',
          pointRadius: 5,
          borderWidth: 2.5,
          tension: 0.1,
        },
        {
          label: 'Digital Twin Projected CDR-SB',
          data: labels.map(m => {
            const idx = futureMonths.indexOf(m);
            return idx !== -1 ? futureCdrsb[idx] : null;
          }),
          borderColor: '#6366f1',
          borderDash: [5, 5],
          pointRadius: 4,
          borderWidth: 2,
          tension: 0.2,
        },
        {
          label: '95% Combined Uncertainty Envelope (Upper)',
          data: labels.map(m => {
            const idx = futureMonths.indexOf(m);
            return idx !== -1 ? upperCi[idx] : null;
          }),
          borderColor: 'transparent',
          backgroundColor: 'rgba(99, 102, 241, 0.15)',
          fill: '+1',
          pointRadius: 0,
        },
        {
          label: '95% Combined Uncertainty Envelope (Lower)',
          data: labels.map(m => {
            const idx = futureMonths.indexOf(m);
            return idx !== -1 ? lowerCi[idx] : null;
          }),
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
        y: {
          grid: { color: '#1e293b' },
          ticks: { color: '#94a3b8' },
          title: { display: true, text: 'CDR-SB Score (0-18)', color: '#cbd5e1' }
        }
      },
      plugins: {
        legend: { labels: { color: '#e2e8f0', boxWidth: 12 } }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// TAB 3: ADAPTIVE RL PATHWAY
// ---------------------------------------------------------------------------
function togglePathwayTest(testName) {
  if (orderedPathwayTests.includes(testName)) {
    orderedPathwayTests = orderedPathwayTests.filter(t => t !== testName);
  } else {
    orderedPathwayTests.push(testName);
  }
  updatePathwayDisplay();
}

async function updatePathwayDisplay() {
  const payload = getBiomarkerPayload();
  try {
    const res = await fetch("/api/pathway", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: payload, ordered_tests: orderedPathwayTests })
    });
    const data = await res.json();

    document.getElementById("pathway-rec-action").textContent = data.recommended_action;
    document.getElementById("pathway-cost-val").textContent = `${data.cumulative_cost_points.toFixed(1)} pts`;
    document.getElementById("pathway-inv-val").textContent = `${data.cumulative_invasiveness_points.toFixed(1)} pts`;
    document.getElementById("pathway-voi-val").textContent = `+${data.expected_value_of_information.toFixed(3)}`;
    document.getElementById("pathway-rationale-text").textContent = data.action_rationale;

    ["blood", "mri", "pet"].forEach(t => {
      const btn = document.getElementById(`btn-toggle-${t}`);
      if (btn) {
        btn.textContent = orderedPathwayTests.includes(t) ? "Ordered (Remove)" : "Order";
        btn.className = orderedPathwayTests.includes(t) ? "text-xs text-emerald-400 font-bold" : "text-xs text-slate-400 underline";
      }
    });
  } catch (err) {
    console.error("Pathway error", err);
  }
}

// ---------------------------------------------------------------------------
// TAB 4: CAUSAL & COUNTERFACTUALS
// ---------------------------------------------------------------------------
async function initCausalChart() {
  try {
    const res = await fetch("/api/shap");
    const data = await res.json();
    if (!data.predictors) return;

    const ctx = document.getElementById("causalChart").getContext("2d");
    if (causalChartInstance) causalChartInstance.destroy();

    const keys = Object.keys(data.predictors);
    const unadjusted = keys.map(k => Math.abs(data.predictors[k].unadjusted_importance));
    const adjusted = keys.map(k => Math.abs(data.predictors[k].confounder_adjusted_importance));
    const shapData = keys.map(k => {
      if (data.shap_importances && data.shap_importances[k]) {
        return data.shap_importances[k] * 5.0; // scale for visual comparison
      }
      return 0.0;
    });

    causalChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: keys.map(k => k.replace(/_/g, ' ')),
        datasets: [
          {
            label: 'Unadjusted Association |β|',
            data: unadjusted,
            backgroundColor: 'rgba(244, 63, 94, 0.65)',
            borderColor: '#f43f5e',
            borderWidth: 1,
          },
          {
            label: 'Age/Education-Adjusted Causal |β|',
            data: adjusted,
            backgroundColor: 'rgba(6, 182, 212, 0.65)',
            borderColor: '#06b6d4',
            borderWidth: 1,
          },
          {
            label: 'Cross-Attention SHAP |φ| (Scaled)',
            data: shapData,
            backgroundColor: 'rgba(168, 85, 247, 0.65)',
            borderColor: '#a855f7',
            borderWidth: 1,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8', font: { size: 10 } } },
          y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'Comparative Importance', color: '#cbd5e1' } }
        },
        plugins: {
          legend: { labels: { color: '#e2e8f0' } }
        }
      }
    });
  } catch (err) {
    console.error("Causal chart error", err);
  }
}

async function runCounterfactual() {
  const ptauPct = parseFloat(document.getElementById("cf-ptau-slider").value) || 0;
  const eduBoost = parseFloat(document.getElementById("cf-edu-slider").value) || 0;

  document.getElementById("cf-ptau-label").textContent = `-${ptauPct}%`;
  document.getElementById("cf-edu-label").textContent = `+${eduBoost} Years`;

  const base = getBiomarkerPayload();
  const currentPtau = base.ptau181 || 2.4;
  const currentEdu = base.education_years || 14.0;

  const interventions = {
    ptau181: currentPtau * (1.0 - ptauPct / 100.0),
    education_years: currentEdu + eduBoost,
  };

  try {
    const res = await fetch("/api/counterfactual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ baseline: base, interventions })
    });
    const data = await res.json();
    const eff = data.treatment_effect;

    document.getElementById("cf-ad-red-val").textContent = `${eff.ad_risk_reduction > 0 ? '-' : '+'}${(Math.abs(eff.ad_risk_reduction) * 100).toFixed(1)}%`;
    document.getElementById("cf-slope-red-val").textContent = `-${eff.projected_slope_retardation_pct.toFixed(1)}% / yr`;
  } catch (err) {
    console.error("Counterfactual error", err);
  }
}

// ---------------------------------------------------------------------------
// TAB 5: POPULATION RESOURCE OPTIMIZER
// ---------------------------------------------------------------------------
async function updateCapacityAllocation() {
  const cap = document.getElementById("capacity-slider").value || 50;
  document.getElementById("capacity-val-label").textContent = `${cap} pts`;

  try {
    const res = await fetch(`/api/optimize?capacity=${cap}`);
    const data = await res.json();

    document.getElementById("pop-selected-count").textContent = `${data.n_selected} / ${data.total_candidates}`;
    document.getElementById("pop-total-yield").textContent = data.total_population_yield.toFixed(4);
    document.getElementById("pop-utilization").textContent = `${data.capacity_utilization_pct}%`;

    const tbody = document.getElementById("population-table-body");
    tbody.innerHTML = "";

    (data.priority_queue_top || []).forEach(row => {
      const tr = document.createElement("tr");
      tr.className = row.selected_in_current_budget ? "bg-cyan-950/20" : "";
      tr.innerHTML = `
        <td class="p-3 font-mono font-bold text-slate-400">#${row.rank}</td>
        <td class="p-3 font-semibold text-slate-200">${row.visit_id}</td>
        <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${row.current_diagnosis === 'AD' ? 'bg-rose-500/20 text-rose-300' : (row.current_diagnosis === 'MCI' ? 'bg-amber-500/20 text-amber-300' : 'bg-emerald-500/20 text-emerald-300')}">${row.current_diagnosis}</span></td>
        <td class="p-3 text-slate-400">${row.missing_modalities.join(", ")}</td>
        <td class="p-3 font-mono">${row.cost} pts</td>
        <td class="p-3 font-mono text-cyan-400 font-bold">${row.diagnostic_yield.toFixed(4)}</td>
        <td class="p-3">
          <span class="px-2 py-0.5 rounded-full text-[10px] font-semibold ${row.selected_in_current_budget ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-slate-800 text-slate-400'}">
            ${row.selected_in_current_budget ? '✓ Allocated' : 'Queued'}
          </span>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Population optimization error", err);
  }
}
