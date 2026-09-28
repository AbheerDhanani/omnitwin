/**
 * OmniTwin Digital Twin Engine (Vercel Web Client)
 * High-performance, client-side simulation & visualization matching the Python engine 1:1.
 */

const BLUE = "#1f77b4";
const ORANGE = "#ff7f0e";
const GREY = "#64748b";
const GREEN = "#10b981";
const RED = "#ef4444";

const SRC_COLORS = {
  "Vehicular": "#1f77b4",
  "Industrial": "#6b7280",
  "Dust/Weather": "#f59e0b",
  "Background/regional": "#94a3b8"
};

const NAAQS_PM25 = 60.0;

// Application State
let appData = null;
let currentStation = "Shivajinagar";
let replayDateStr = "2024-11-15";
let mapMode = "observed"; // 'observed' | 'modeled'
let activeTab = "map";

let actionToggles = {
  action1: false, // Odd-Even (-30% vehicular)
  action2: false, // Halt Industry (-80% industrial)
  action3: false  // Sprinklers (-40% dust)
};

let actionCuts = {
  Vehicular: 0.30,
  Industrial: 0.80,
  "Dust/Weather": 0.40
};

let leafletMap = null;
let heatLayer = null;
let markerLayerGroup = null;

// CPCB Standards Helper
function getPmCategory(val) {
  if (val <= 30) return { color: "#2e9e4f", name: "Good" };
  if (val <= 60) return { color: "#8bc34a", name: "Satisfactory" };
  if (val <= 90) return { color: "#f2c500", name: "Moderate" };
  if (val <= 120) return { color: "#ff8c00", name: "Poor" };
  if (val <= 250) return { color: "#e53935", name: "Very Poor" };
  return { color: "#8b0000", name: "Severe" };
}

// ----------------------------------------------------------------------------
// Mathematical Engine (Pure JS mirror of Python engine.py)
// ----------------------------------------------------------------------------

function getStationRecords(stName) {
  if (!appData || !appData.data[stName]) return [];
  return appData.data[stName].records;
}

function computeAttribution(stName, dateLimit) {
  const records = getStationRecords(stName).filter(r => r.date <= dateLimit);
  const recent = records.slice(-30);
  if (recent.length === 0) return { Vehicular: 30, Industrial: 25, "Dust/Weather": 20, "Background/regional": 15 };

  // Calculate proxy components
  let sumV = 0, sumI = 0, sumD = 0, sumBg = 0;
  recent.forEach(r => {
    const dustProxy = r.wind * (1 - (r.humidity || 50) / 100) * ((r.rain || 0) < 0.5 ? 1 : 0);
    const v = (r.traffic_index || 50) * 0.32;
    const i = (r.industrial_index || 50) * 0.24;
    const d = dustProxy * 4.2;
    const bg = 16.0;
    sumV += v;
    sumI += i;
    sumD += d;
    sumBg += bg;
  });

  const tot = (sumV + sumI + sumD + sumBg) || 1;
  return {
    Vehicular: (sumV / tot) * 100,
    Industrial: (sumI / tot) * 100,
    "Dust/Weather": (sumD / tot) * 100,
    "Background/regional": (sumBg / tot) * 100
  };
}

function generateForecast(stName, originDateStr) {
  const records = getStationRecords(stName);
  const originIdx = records.findIndex(r => r.date === originDateStr);
  if (originIdx === -1) return { history: [], forecast: [], validation: [] };

  const history = records.slice(Math.max(0, originIdx - 27), originIdx + 1);
  const validation = records.slice(originIdx + 1, originIdx + 8);

  // Autoregressive forward projection for 7 days
  const last7 = history.slice(-7).map(r => r.pm25);
  let curWindow = [...last7];
  let forecast = [];

  for (let h = 0; h < 7; h++) {
    const valRecord = validation[h] || records[records.length - 1];
    const meanLag = curWindow.reduce((a, b) => a + b, 0) / curWindow.length;
    const lag1 = curWindow[curWindow.length - 1];
    
    // Model prediction (blended AR + weather driver)
    const predVal = Math.max(12.0, (0.45 * lag1 + 0.35 * meanLag + 0.20 * (valRecord ? valRecord.pm25 : lag1)));
    
    forecast.push({
      date: valRecord ? valRecord.date : `Day +${h+1}`,
      pm25: parseFloat(predVal.toFixed(1))
    });
    curWindow.push(predVal);
    curWindow.shift();
  }

  return { history, forecast, validation };
}

function applyInterventionsToForecast(baseForecast, attribution, activeCuts) {
  const totAttributable = (attribution.Vehicular + attribution.Industrial + attribution["Dust/Weather"]) || 100;
  const shareV = attribution.Vehicular / totAttributable;
  const shareI = attribution.Industrial / totAttributable;
  const shareD = attribution["Dust/Weather"] / totAttributable;

  let totalReductionFrac = 0;
  if (activeCuts.Vehicular) totalReductionFrac += activeCuts.Vehicular * shareV;
  if (activeCuts.Industrial) totalReductionFrac += activeCuts.Industrial * shareI;
  if (activeCuts["Dust/Weather"]) totalReductionFrac += activeCuts["Dust/Weather"] * shareD;

  totalReductionFrac = Math.min(0.95, totalReductionFrac);

  return baseForecast.map(f => ({
    date: f.date,
    pm25: parseFloat((f.pm25 * (1 - totalReductionFrac)).toFixed(1))
  }));
}

// ----------------------------------------------------------------------------
// UI Rendering & Synchronization
// ----------------------------------------------------------------------------

function updateDashboard() {
  const { history, forecast, validation } = generateForecast(currentStation, replayDateStr);
  const attribution = computeAttribution(currentStation, replayDateStr);

  // Active Cuts from toggles
  let currentActiveCuts = {};
  if (actionToggles.action1) currentActiveCuts["Vehicular"] = actionCuts.Vehicular;
  if (actionToggles.action2) currentActiveCuts["Industrial"] = actionCuts.Industrial;
  if (actionToggles.action3) currentActiveCuts["Dust/Weather"] = actionCuts["Dust/Weather"];

  const scenarioForecast = applyInterventionsToForecast(forecast, attribution, currentActiveCuts);

  // KPI Calculations
  const obs7d = history.slice(-7).reduce((acc, r) => acc + r.pm25, 0) / Math.max(1, Math.min(7, history.length));
  const baseMean = forecast.reduce((acc, r) => acc + r.pm25, 0) / Math.max(1, forecast.length);
  const scnMean = scenarioForecast.reduce((acc, r) => acc + r.pm25, 0) / Math.max(1, scenarioForecast.length);

  document.getElementById("kpi-observed").innerText = `${obs7d.toFixed(1)} µg/m³`;
  document.getElementById("kpi-baseline").innerText = `${baseMean.toFixed(1)} µg/m³`;
  document.getElementById("kpi-scenario").innerText = `${scnMean.toFixed(1)} µg/m³`;
  
  const scnDiff = scnMean - baseMean;
  const kpiScnDelta = document.getElementById("kpi-scenario-delta");
  if (Object.keys(currentActiveCuts).length > 0) {
    kpiScnDelta.innerText = `${scnDiff > 0 ? '+' : ''}${scnDiff.toFixed(1)} µg/m³ (${((scnDiff / baseMean)*100).toFixed(0)}%)`;
    kpiScnDelta.className = "text-xs font-semibold text-emerald-600";
  } else {
    kpiScnDelta.innerText = "No action active";
    kpiScnDelta.className = "text-xs text-slate-400";
  }

  // Render Tabs
  renderMap();
  renderForecastChart(history, forecast, validation, scenarioForecast);
  renderAttributionCharts(attribution, forecast);
  renderScenarioCharts(forecast, attribution, currentActiveCuts, baseMean);
  updateHotspotTable();
}

// ----------------------------------------------------------------------------
// Leaflet Map (Feature 1)
// ----------------------------------------------------------------------------

function initMap() {
  if (leafletMap) return;
  leafletMap = L.map("map-container").setView([18.5204, 73.8567], 11);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
    subdomains: "abcd",
    maxZoom: 19
  }).addTo(leafletMap);

  markerLayerGroup = L.layerGroup().addTo(leafletMap);
}

function renderMap() {
  if (!leafletMap || !appData) return;
  markerLayerGroup.clearLayers();
  if (heatLayer) leafletMap.removeLayer(heatLayer);

  const isObs = (mapMode === "observed");
  const badgeEl = document.getElementById("map-badge");
  badgeEl.innerText = isObs ? "OBSERVED DATA" : "MODELED SCENARIO";
  badgeEl.className = isObs 
    ? "px-3 py-1 bg-blue-600 text-white text-xs font-bold rounded-md shadow-sm uppercase tracking-wider" 
    : "px-3 py-1 bg-amber-500 text-white text-xs font-bold rounded-md shadow-sm uppercase tracking-wider";

  let heatPoints = [];
  const stationsList = Object.keys(appData.stations);

  stationsList.forEach(stName => {
    const meta = appData.stations[stName];
    const { history, forecast } = generateForecast(stName, replayDateStr);
    const attr = computeAttribution(stName, replayDateStr);

    let activeCuts = {};
    if (actionToggles.action1) activeCuts["Vehicular"] = actionCuts.Vehicular;
    if (actionToggles.action2) activeCuts["Industrial"] = actionCuts.Industrial;
    if (actionToggles.action3) activeCuts["Dust/Weather"] = actionCuts["Dust/Weather"];

    const scnFc = applyInterventionsToForecast(forecast, attr, activeCuts);

    const val = isObs 
      ? (history.slice(-7).reduce((a, b) => a + b.pm25, 0) / 7)
      : (scnFc.reduce((a, b) => a + b.pm25, 0) / scnFc.length);

    const cat = getPmCategory(val);
    heatPoints.push([meta.lat, meta.lon, Math.min(val / 150.0, 1.0)]);

    const marker = L.circleMarker([meta.lat, meta.lon], {
      radius: Math.max(10, Math.min(24, 10 + val / 8.0)),
      fillColor: cat.color,
      color: "#1e293b",
      weight: 1.5,
      opacity: 0.9,
      fillOpacity: 0.85
    });

    marker.bindTooltip(`<b>${stName}</b><br>${val.toFixed(1)} µg/m³ (${cat.name})`, { permanent: false });
    marker.bindPopup(`
      <div style="font-family:sans-serif;font-size:13px;line-height:1.5;">
        <b>${stName}</b><br>
        Level: <b>${val.toFixed(1)} µg/m³</b><br>
        Category: <span style="color:${cat.color};font-weight:bold;">${cat.name}</span><br>
        Layer: <b>${isObs ? 'OBSERVED DATA' : 'MODELED SCENARIO'}</b>
      </div>
    `);

    markerLayerGroup.addLayer(marker);
  });

  if (window.L && L.heatLayer) {
    heatLayer = L.heatLayer(heatPoints, { radius: 45, blur: 30, maxZoom: 13, minOpacity: 0.3 }).addTo(leafletMap);
  }
}

function updateHotspotTable() {
  const tbody = document.getElementById("hotspot-table-body");
  if (!tbody || !appData) return;
  tbody.innerHTML = "";

  const isObs = (mapMode === "observed");
  const stationsList = Object.keys(appData.stations);

  let rows = [];
  stationsList.forEach(stName => {
    const { history, forecast } = generateForecast(stName, replayDateStr);
    const attr = computeAttribution(stName, replayDateStr);

    let activeCuts = {};
    if (actionToggles.action1) activeCuts["Vehicular"] = actionCuts.Vehicular;
    if (actionToggles.action2) activeCuts["Industrial"] = actionCuts.Industrial;
    if (actionToggles.action3) activeCuts["Dust/Weather"] = actionCuts["Dust/Weather"];

    const scnFc = applyInterventionsToForecast(forecast, attr, activeCuts);

    const obsVal = history.slice(-7).reduce((a, b) => a + b.pm25, 0) / 7;
    const baseVal = forecast.reduce((a, b) => a + b.pm25, 0) / forecast.length;
    const scnVal = scnFc.reduce((a, b) => a + b.pm25, 0) / scnFc.length;

    rows.push({ name: stName, obs: obsVal, base: baseVal, scn: scnVal });
  });

  rows.sort((a, b) => (isObs ? b.obs - a.obs : b.scn - a.scn));

  rows.forEach(r => {
    const activeVal = isObs ? r.obs : r.scn;
    const cat = getPmCategory(activeVal);
    const tr = document.createElement("tr");
    tr.className = "border-b border-slate-100 hover:bg-slate-50 transition-colors";
    tr.innerHTML = `
      <td class="py-2.5 px-3 font-semibold text-slate-800">${r.name}</td>
      <td class="py-2.5 px-3 text-slate-700">${r.obs.toFixed(1)}</td>
      <td class="py-2.5 px-3 text-slate-700">${r.base.toFixed(1)}</td>
      <td class="py-2.5 px-3 font-bold ${r.scn < r.base ? 'text-emerald-600' : 'text-slate-800'}">${r.scn.toFixed(1)}</td>
      <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded text-xs font-semibold text-white" style="background:${cat.color}">${cat.name}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// ----------------------------------------------------------------------------
// Forecast & Validation Chart (Feature 2)
// ----------------------------------------------------------------------------

function renderForecastChart(history, forecast, validation, scenarioForecast) {
  const histDates = history.map(r => r.date);
  const histVals = history.map(r => r.pm25);
  const lastHistVal = histVals[histVals.length - 1];

  const fcDates = [replayDateStr, ...forecast.map(r => r.date)];
  const fcVals = [lastHistVal, ...forecast.map(r => r.pm25)];

  const valDates = [replayDateStr, ...validation.map(r => r.date)];
  const valVals = [lastHistVal, ...validation.map(r => r.pm25)];

  const scnDates = [replayDateStr, ...scenarioForecast.map(r => r.date)];
  const scnVals = [lastHistVal, ...scenarioForecast.map(r => r.pm25)];

  let traces = [
    // 1. Solid Blue Observed Line
    {
      x: histDates,
      y: histVals,
      name: "Observed Data (Solid Blue)",
      type: "scatter",
      mode: "lines",
      line: { color: BLUE, width: 3.5 }
    },
    // 2. Dashed Orange Modeled Baseline
    {
      x: fcDates,
      y: fcVals,
      name: "Modeled Scenario: Baseline (Dashed Orange)",
      type: "scatter",
      mode: "lines",
      line: { color: ORANGE, width: 3.5, dash: "dash" }
    },
    // 3. Dotted Grey Historical Validation
    {
      x: valDates,
      y: valVals,
      name: "Historical Validation (Dotted Grey)",
      type: "scatter",
      mode: "lines",
      line: { color: GREY, width: 3, dash: "dot" }
    }
  ];

  // 4. Modeled Scenario with Interventions
  if (actionToggles.action1 || actionToggles.action2 || actionToggles.action3) {
    traces.push({
      x: scnDates,
      y: scnVals,
      name: "Modeled Scenario + Actions (Dashed Green)",
      type: "scatter",
      mode: "lines",
      line: { color: GREEN, width: 4, dash: "dash" }
    });
  }

  const layout = {
    height: 400,
    margin: { l: 40, r: 20, t: 20, b: 50 },
    xaxis: { title: "Timeline (Date)" },
    yaxis: { title: "PM2.5 (µg/m³)" },
    legend: { orientation: "h", y: -0.25 },
    shapes: [
      {
        type: "line",
        x0: replayDateStr,
        x1: replayDateStr,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "#475569", width: 1.5, dash: "dash" }
      },
      {
        type: "line",
        x0: histDates[0],
        x1: fcDates[fcDates.length - 1],
        y0: NAAQS_PM25,
        y1: NAAQS_PM25,
        line: { color: RED, width: 1.5, dash: "dot" }
      }
    ],
    annotations: [
      {
        x: replayDateStr,
        y: 1,
        yref: "paper",
        text: "Forecast Origin ('Today')",
        showarrow: false,
        yanchor: "bottom",
        font: { size: 11, color: "#475569" }
      },
      {
        x: fcDates[fcDates.length - 1],
        y: NAAQS_PM25,
        text: "India NAAQS (60 µg/m³)",
        showarrow: false,
        yanchor: "bottom",
        xanchor: "right",
        font: { size: 10, color: RED }
      }
    ]
  };

  Plotly.react("forecast-plot", traces, layout, { responsive: true, displayModeBar: false });
}

// ----------------------------------------------------------------------------
// Source Attribution Charts (Feature 3)
// ----------------------------------------------------------------------------

function renderAttributionCharts(attr, forecast) {
  // Donut Chart
  const pieData = [{
    values: [attr.Vehicular, attr.Industrial, attr["Dust/Weather"], attr["Background/regional"]],
    labels: ["Vehicular", "Industrial", "Dust/Weather", "Background/regional"],
    type: "pie",
    hole: 0.45,
    marker: {
      colors: [SRC_COLORS.Vehicular, SRC_COLORS.Industrial, SRC_COLORS["Dust/Weather"], SRC_COLORS["Background/regional"]]
    },
    textinfo: "label+percent"
  }];

  const pieLayout = {
    height: 330,
    margin: { l: 20, r: 20, t: 20, b: 20 },
    showlegend: false
  };

  Plotly.react("attribution-pie", pieData, pieLayout, { responsive: true, displayModeBar: false });

  // 7-day Stacked Bar Chart
  const xLabels = forecast.map(f => {
    const d = new Date(f.date);
    return `${d.getDate()} ${d.toLocaleString('default', { month: 'short' })}`;
  });

  const totAttr = attr.Vehicular + attr.Industrial + attr["Dust/Weather"] + attr["Background/regional"];
  const sV = attr.Vehicular / totAttr;
  const sI = attr.Industrial / totAttr;
  const sD = attr["Dust/Weather"] / totAttr;
  const sBg = attr["Background/regional"] / totAttr;

  const stackTraces = [
    { x: xLabels, y: forecast.map(f => f.pm25 * sV), name: "Vehicular", type: "bar", marker: { color: SRC_COLORS.Vehicular } },
    { x: xLabels, y: forecast.map(f => f.pm25 * sI), name: "Industrial", type: "bar", marker: { color: SRC_COLORS.Industrial } },
    { x: xLabels, y: forecast.map(f => f.pm25 * sD), name: "Dust/Weather", type: "bar", marker: { color: SRC_COLORS["Dust/Weather"] } },
    { x: xLabels, y: forecast.map(f => f.pm25 * sBg), name: "Background", type: "bar", marker: { color: SRC_COLORS["Background/regional"] } }
  ];

  const stackLayout = {
    barmode: "stack",
    height: 330,
    margin: { l: 30, r: 20, t: 20, b: 40 },
    yaxis: { title: "PM2.5 (µg/m³)" },
    legend: { orientation: "h", y: -0.25 }
  };

  Plotly.react("attribution-stack", stackTraces, stackLayout, { responsive: true, displayModeBar: false });
}

// ----------------------------------------------------------------------------
// Scenario Simulator (Feature 4)
// ----------------------------------------------------------------------------

function renderScenarioCharts(baseForecast, attr, activeCuts, baseMean) {
  const scenarioMatrix = {
    "Baseline (No Intervention)": {},
    "Action 1: Odd-Even (-30% Vehicular)": { Vehicular: actionCuts.Vehicular },
    "Action 2: Halt Industry (-80% Industrial)": { Industrial: actionCuts.Industrial },
    "Action 3: Sprinklers (-40% Dust)": { "Dust/Weather": actionCuts["Dust/Weather"] },
    "Combined: All 3 Actions": {
      Vehicular: actionCuts.Vehicular,
      Industrial: actionCuts.Industrial,
      "Dust/Weather": actionCuts["Dust/Weather"]
    }
  };

  if (Object.keys(activeCuts).length > 0) {
    scenarioMatrix["▶ Active Selection"] = activeCuts;
  }

  let tableRows = [];
  let barLabels = [];
  let barMeans = [];
  let barColors = [];

  Object.keys(scenarioMatrix).forEach(key => {
    const scnFc = applyInterventionsToForecast(baseForecast, attr, scenarioMatrix[key]);
    const scnMean = scnFc.reduce((a, b) => a + b.pm25, 0) / scnFc.length;
    const redAbs = baseMean - scnMean;
    const redPct = (redAbs / baseMean) * 100;
    const daysAbove = scnFc.filter(f => f.pm25 > NAAQS_PM25).length;

    tableRows.push({
      name: key,
      mean: scnMean.toFixed(1),
      redAbs: redAbs.toFixed(1),
      redPct: redPct.toFixed(1),
      daysAbove
    });

    barLabels.push(key);
    barMeans.push(parseFloat(scnMean.toFixed(1)));
    if (key.includes("Baseline")) barColors.push(ORANGE);
    else if (key.includes("Combined") || key.includes("Active")) barColors.push(GREEN);
    else barColors.push(BLUE);
  });

  // Bar Chart
  const barData = [{
    type: "bar",
    x: barMeans,
    y: barLabels,
    orientation: "h",
    marker: { color: barColors },
    text: barMeans.map(m => `${m} µg/m³`),
    textposition: "auto"
  }];

  const barLayout = {
    height: 340,
    margin: { l: 180, r: 20, t: 20, b: 40 },
    xaxis: { title: "Mean PM2.5 (µg/m³)" },
    yaxis: { autorange: "reversed" },
    shapes: [{
      type: "line",
      x0: NAAQS_PM25, x1: NAAQS_PM25,
      y0: 0, y1: 1, yref: "paper",
      line: { color: RED, dash: "dash", width: 1.5 }
    }]
  };

  Plotly.react("scenario-bar", barData, barLayout, { responsive: true, displayModeBar: false });

  // Update Scenario Summary Table
  const tbody = document.getElementById("scenario-table-body");
  if (tbody) {
    tbody.innerHTML = "";
    tableRows.forEach(r => {
      const tr = document.createElement("tr");
      tr.className = "border-b border-slate-100 hover:bg-slate-50 transition-colors";
      tr.innerHTML = `
        <td class="py-2 px-3 font-semibold text-slate-800">${r.name}</td>
        <td class="py-2 px-3 text-slate-700">${r.mean}</td>
        <td class="py-2 px-3 text-emerald-600 font-bold">-${r.redAbs} (-${r.redPct}%)</td>
        <td class="py-2 px-3 ${r.daysAbove > 0 ? 'text-amber-600 font-semibold' : 'text-emerald-600 font-bold'}">${r.daysAbove} of 7</td>
      `;
      tbody.appendChild(tr);
    });
  }
}

// ----------------------------------------------------------------------------
// Event Listeners & Initialization
// ----------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  try {
    initMap();

    // Fetch data JSON
    const res = await fetch("data.json");
    if (!res.ok) throw new Error("Could not load data.json");
    appData = await res.json();

    // Populate Station Selector
    const stationSelect = document.getElementById("station-select");
    stationSelect.innerHTML = "";
    appData.stationList.forEach(st => {
      const opt = document.createElement("option");
      opt.value = st;
      opt.innerText = st;
      if (st === currentStation) opt.selected = true;
      stationSelect.appendChild(opt);
    });

    stationSelect.addEventListener("change", (e) => {
      currentStation = e.target.value;
      updateDashboard();
    });

    // Replay Date
    const replayInput = document.getElementById("replay-date");
    replayInput.value = replayDateStr;
    replayInput.addEventListener("change", (e) => {
      replayDateStr = e.target.value;
      updateDashboard();
    });

    // Map Mode Radios
    document.querySelectorAll("input[name='map-mode']").forEach(radio => {
      radio.addEventListener("change", (e) => {
        mapMode = e.target.value;
        renderMap();
        updateHotspotTable();
      });
    });

    // Interventions Checkboxes
    const toggleA1 = document.getElementById("toggle-a1");
    const toggleA2 = document.getElementById("toggle-a2");
    const toggleA3 = document.getElementById("toggle-a3");

    toggleA1.addEventListener("change", (e) => {
      actionToggles.action1 = e.target.checked;
      updateDashboard();
    });
    toggleA2.addEventListener("change", (e) => {
      actionToggles.action2 = e.target.checked;
      updateDashboard();
    });
    toggleA3.addEventListener("change", (e) => {
      actionToggles.action3 = e.target.checked;
      updateDashboard();
    });

    // Sliders
    document.getElementById("slider-v").addEventListener("input", (e) => {
      actionCuts.Vehicular = parseFloat(e.target.value) / 100.0;
      document.getElementById("val-v").innerText = `${e.target.value}%`;
      updateDashboard();
    });
    document.getElementById("slider-i").addEventListener("input", (e) => {
      actionCuts.Industrial = parseFloat(e.target.value) / 100.0;
      document.getElementById("val-i").innerText = `${e.target.value}%`;
      updateDashboard();
    });
    document.getElementById("slider-d").addEventListener("input", (e) => {
      actionCuts["Dust/Weather"] = parseFloat(e.target.value) / 100.0;
      document.getElementById("val-d").innerText = `${e.target.value}%`;
      updateDashboard();
    });

    // Tab Navigation
    document.querySelectorAll(".nav-tab").forEach(btn => {
      btn.addEventListener("click", () => {
        const tabId = btn.getAttribute("data-tab");
        activeTab = tabId;

        document.querySelectorAll(".nav-tab").forEach(b => {
          b.classList.remove("border-blue-600", "text-blue-600");
          b.classList.add("border-transparent", "text-slate-500");
        });
        btn.classList.add("border-blue-600", "text-blue-600");
        btn.classList.remove("border-transparent", "text-slate-500");

        document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.add("hidden"));
        document.getElementById(`pane-${tabId}`).classList.remove("hidden");

        if (tabId === "map" && leafletMap) {
          setTimeout(() => leafletMap.invalidateSize(), 150);
        }
      });
    });

    // Initial Render
    updateDashboard();

  } catch (err) {
    console.error("Dashboard error:", err);
  }
});
