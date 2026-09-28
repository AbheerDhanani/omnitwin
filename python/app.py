"""
OmniTwin: Urban Environmental Digital Twin (HackMatrix Prototype)
Strictly fulfilling all HackMatrix problem statement outcomes and PRD specifications.
"""

import os
import io
import datetime as dt
import folium
from folium.plugins import HeatMap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import engine as E

# ----------------------------------------------------------------------------
# Theme & Color Palette (Standardized across visualizations)
# ----------------------------------------------------------------------------
BLUE = "#1f77b4"     # OBSERVED DATA
ORANGE = "#ff7f0e"   # MODELED SCENARIO
GREY = "#64748b"     # HISTORICAL VALIDATION
GREEN = "#10b981"    # INTERVENTIONS / ACTIONS
RED = "#ef4444"      # AIR QUALITY STANDARD (NAAQS)

SRC_COLORS = {
    "Vehicular": "#1f77b4",
    "Industrial": "#6b7280",
    "Dust/Weather": "#f59e0b",
    "Background/regional": "#94a3b8",
}

STATED_ASSUMPTION = (
    "Assumption: Industrial baseline is constant; traffic scales with rush hour; "
    "dust correlates with wind speed."
)

st.set_page_config(
    page_title="OmniTwin · Urban Environmental Digital Twin",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Visual Status Badges (Strict PRD Compliance)
# ----------------------------------------------------------------------------
def render_badge(badge_type: str, custom_text: str = None):
    badge_map = {
        "obs": ("OBSERVED DATA", BLUE, "#ffffff"),
        "mod": ("MODELED SCENARIO", ORANGE, "#ffffff"),
        "val": ("HISTORICAL VALIDATION", GREY, "#ffffff"),
        "act": ("ACTIVE INTERVENTIONS", GREEN, "#ffffff"),
    }
    label, bg_col, text_col = badge_map.get(badge_type, ("DATA", "#333", "#fff"))
    if custom_text:
        label = custom_text

    st.markdown(
        f"""
        <div style="
            background: {bg_col};
            color: {text_col};
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 0.92rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            margin: 4px 0 10px 0;">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Data Loading & Caching
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading Pune urban environmental data...")
def load_dataset(uploaded_bytes):
    if uploaded_bytes is not None:
        return pd.read_csv(io.BytesIO(uploaded_bytes), parse_dates=["date"])
    
    # Check default pre-generated CSV
    default_csv = os.path.join(os.path.dirname(__file__), "data", "omnitwin_pune_historical.csv")
    if os.path.exists(default_csv):
        return pd.read_csv(default_csv, parse_dates=["date"])
    
    # Fallback to generating data
    from data.generate_pune_data import generate_dataset
    return generate_dataset()


@st.cache_data(show_spinner=False)
def get_station_series(df, station_name):
    return E.get_series(df, station_name)


@st.cache_resource(show_spinner="Training predictive model...")
def get_forecast_model(_series, cache_key, model_type):
    return E.fit_forecaster(_series, model_type)


@st.cache_data(show_spinner="Running rolling-origin back-test...")
def get_backtest(_series, _model, cache_key):
    return E.backtest(_model, _series)


# ----------------------------------------------------------------------------
# Sidebar Controls & Twin Configuration
# ----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 4px 0 10px 0;">
        <h2 style="margin: 0; color: #0f172a;">🌫️ OmniTwin</h2>
        <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">
            Urban Environmental Digital Twin · Pune
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload custom CSV (Optional)",
    type=["csv"],
    help="Upload your own CPCB/Kaggle CSV dataset. Defaults to Pune Urban Region historical records.",
)

file_bytes = uploaded_file.getvalue() if uploaded_file else None
is_custom_data = file_bytes is not None

with st.sidebar.expander("📍 Study Area Coordinates"):
    area_name = st.text_input("Area Name", "Pune Urban Region")
    area_lat = st.number_input("Latitude", value=18.5204, format="%.4f")
    area_lon = st.number_input("Longitude", value=73.8567, format="%.4f")

raw_df = load_dataset(file_bytes)

try:
    df, prep_notes = E.prepare_df(raw_df, area_lat, area_lon, area_name)
except ValueError as err:
    st.sidebar.error(str(err))
    st.stop()

if prep_notes:
    with st.sidebar.expander("ℹ️ Data Ingestion Notes"):
        for note in prep_notes:
            st.caption(f"• {note}")

stations = list(df.groupby("station").size().index)
twin_stations = (["City average"] + stations) if len(stations) > 1 else stations

selected_station = st.sidebar.selectbox("Select Twin Neighborhood", twin_stations)
selected_model_type = st.sidebar.selectbox("Forecast Algorithm", ["Ridge regression", "Gradient boosting"])

s_current = get_station_series(df, selected_station)
min_required = E.TEST_DAYS + 30
if len(s_current) < min_required:
    st.error(f"Dataset too small. Need at least {min_required} daily records.")
    st.stop()

test_start = E.test_start_date(s_current)
min_replay = (test_start + pd.Timedelta(days=7)).date()
max_replay = (s_current.index.max() - pd.Timedelta(days=E.HORIZON)).date()

replay_date = st.sidebar.date_input(
    "Historical Replay Origin ('Today')",
    value=min_replay + dt.timedelta(days=25),
    min_value=min_replay,
    max_value=max_replay,
    help="Select 'Today' inside the held-out historical test period to validate the 7-day forecast."
)
origin_ts = pd.Timestamp(replay_date)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Digital Twin Scenarios")
st.sidebar.caption("Toggle interventions to observe modeled PM2.5 reductions in real-time.")

active_interventions = {}
action_sliders = {}

with st.sidebar.expander("⚙️ Edit Intervention Assumptions"):
    for action_name, category, default_val in E.ACTIONS:
        action_sliders[category] = st.slider(
            f"{action_name} — {category} cut (%)",
            0, 100, int(default_val * 100)
        ) / 100.0

for action_name, category, default_val in E.ACTIONS:
    if st.sidebar.checkbox(action_name, value=False):
        active_interventions[category] = action_sliders[category]

if active_interventions:
    st.sidebar.success(f"{len(active_interventions)} intervention(s) active")


# ----------------------------------------------------------------------------
# Model Training & Inference Execution
# ----------------------------------------------------------------------------
attribution_model = E.fit_attribution(df, origin_ts)

def compute_bundle(st_name):
    series = get_station_series(df, st_name)
    cache_key = f"{st_name}_{selected_model_type}"
    model = get_forecast_model(series, cache_key, selected_model_type)
    fc_base = E.forecast(model, series, origin_ts)
    fc_scene = E.apply_actions(fc_base, attribution_model, series, active_interventions)
    return series, model, fc_base, fc_scene

series, model, forecast_baseline, forecast_scenario = compute_bundle(selected_station)
backtest_df = get_backtest(series, model, f"{selected_station}_{selected_model_type}")
metrics_table = E.score(backtest_df)

obs_7d_mean = series.loc[origin_ts - pd.Timedelta(days=6):origin_ts, "pm25"].mean()
base_7d_mean = forecast_baseline.mean()
scene_7d_mean = forecast_scenario.mean()

model_mae = metrics_table.loc["OmniTwin model", "MAE (µg/m³)"]
baseline_mae = metrics_table.loc["Persistence baseline", "MAE (µg/m³)"]
improvement_pct = max(0.0, (1.0 - model_mae / max(baseline_mae, 1e-6)) * 100.0)


# ----------------------------------------------------------------------------
# Executive Top Header & KPI Bar
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 8px;">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #0f172a;">
            🌍 OmniTwin: Urban Environmental Digital Twin
        </h1>
        <p style="margin: 4px 0 16px 0; font-size: 1.05rem; color: #475569;">
            Prototype for <b>Pune Urban Region</b> · Fulfilling HackMatrix Expected Outcomes
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# High-Visibility Legend Bar
st.markdown(
    f"""
    <div style="background: #f1f5f9; padding: 10px 16px; border-radius: 8px; margin-bottom: 18px; border: 1px solid #e2e8f0; display: flex; flex-wrap: wrap; gap: 16px; align-items: center;">
        <span style="font-weight: 700; color: #334155; font-size: 0.9rem;">LABEL GUIDE:</span>
        <span style="display: inline-flex; align-items: center; gap: 6px; font-weight: 600; color: {BLUE}; font-size: 0.88rem;">
            <span style="display:inline-block; width:12px; height:12px; background:{BLUE}; border-radius:3px;"></span> OBSERVED DATA (Historical Readings)
        </span>
        <span style="display: inline-flex; align-items: center; gap: 6px; font-weight: 600; color: {ORANGE}; font-size: 0.88rem;">
            <span style="display:inline-block; width:12px; height:12px; background:{ORANGE}; border-radius:3px;"></span> MODELED SCENARIO (7-Day Forecast & Interventions)
        </span>
        <span style="display: inline-flex; align-items: center; gap: 6px; font-weight: 600; color: {GREY}; font-size: 0.88rem;">
            <span style="display:inline-block; width:12px; height:12px; background:{GREY}; border-radius:3px;"></span> HISTORICAL VALIDATION (Held-Out Ground Truth)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(
        "OBSERVED · Past 7 Days",
        f"{obs_7d_mean:.1f} µg/m³",
        help="Mean measured PM2.5 ending on selected replay date.",
    )
with kpi2:
    st.metric(
        "MODELED · Baseline (7d)",
        f"{base_7d_mean:.1f} µg/m³",
        help="7-day forward prediction with zero intervention.",
    )
with kpi3:
    delta_str = f"{scene_7d_mean - base_7d_mean:+.1f} µg/m³" if active_interventions else "No action active"
    st.metric(
        "MODELED · With Actions",
        f"{scene_7d_mean:.1f} µg/m³",
        delta=delta_str if active_interventions else None,
        delta_color="inverse",
        help="Forecast after applying active scenario cuts.",
    )
with kpi4:
    st.metric(
        "VALIDATION · Backtest MAE",
        f"{model_mae:.1f} µg/m³",
        delta=f"{improvement_pct:.0f}% better than persistence",
        delta_color="normal",
        help="Tested on 90-day held-out historical period.",
    )

st.markdown("---")

# ----------------------------------------------------------------------------
# Core Tabs (Features 1 - 4 + Methodology)
# ----------------------------------------------------------------------------
tab_map, tab_forecast, tab_attribution, tab_scenarios, tab_method = st.tabs([
    "🗺️ Hotspot Map",
    "📈 Forecast & Validation",
    "🧪 Source Attribution",
    "🎛️ Scenario Simulator",
    "📋 Methodology & PRD Rubric",
])


# ============================================================================
# TAB 1: MAP-BASED HOTSPOT VISUALIZATION (Feature 1)
# ============================================================================
with tab_map:
    st.markdown("### Feature 1: Map-Based Pollution Hotspots")
    st.caption("Inspect pollution concentration across Pune neighborhoods with observed vs modeled layers.")

    map_mode = st.radio(
        "Map Display Layer",
        [
            "OBSERVED DATA · Historical 7-Day Average",
            "MODELED SCENARIO · Next 7-Day Forecast (With Interventions)"
        ],
        horizontal=True,
    )
    is_observed_view = map_mode.startswith("OBSERVED")

    # Build Hotspot Table
    coords_dict = df.groupby("station")[["lat", "lon"]].first()
    hotspot_records = []

    for st_name in stations:
        st_s, _, st_fc_base, st_fc_scene = compute_bundle(st_name)
        obs_val = st_s.loc[origin_ts - pd.Timedelta(days=6):origin_ts, "pm25"].mean()
        base_val = st_fc_base.mean()
        scene_val = st_fc_scene.mean()

        hotspot_records.append({
            "Neighborhood": st_name,
            "lat": coords_dict.loc[st_name, "lat"],
            "lon": coords_dict.loc[st_name, "lon"],
            "Observed PM2.5 (µg/m³)": round(obs_val, 1),
            "Modeled Baseline (µg/m³)": round(base_val, 1),
            "Modeled + Interventions (µg/m³)": round(scene_val, 1),
        })

    hotspot_df = pd.DataFrame(hotspot_records)
    active_col = "Observed PM2.5 (µg/m³)" if is_observed_view else "Modeled + Interventions (µg/m³)"

    # Folium Map
    center_lat = hotspot_df["lat"].mean()
    center_lon = hotspot_df["lon"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    # HeatMap Layer
    heat_points = [
        [r["lat"], r["lon"], min(float(r[active_col]) / 160.0, 1.0)]
        for _, r in hotspot_df.iterrows()
    ]
    HeatMap(heat_points, radius=55, blur=38, min_opacity=0.3).add_to(fmap)

    # Circle Markers with CPCB Color Standards
    for _, r in hotspot_df.iterrows():
        cat_color, cat_name = E.pm_category(r[active_col])
        radius_size = max(10, int(10 + r[active_col] / 8.0))
        
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 13px; line-height: 1.5;">
            <b>{r['Neighborhood']}</b><br>
            Level: <b>{r[active_col]:.1f} µg/m³</b><br>
            Status: <span style="color: {cat_color}; font-weight: bold;">{cat_name}</span><br>
            Layer: {'OBSERVED' if is_observed_view else 'MODELED'}
        </div>
        """
        
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=radius_size,
            color="#1e293b",
            weight=1.5,
            fill=True,
            fill_color=cat_color,
            fill_opacity=0.85,
            tooltip=f"{r['Neighborhood']}: {r[active_col]:.1f} µg/m³ ({cat_name})",
            popup=folium.Popup(popup_html, max_width=220)
        ).add_to(fmap)

    # Map Badging & Legend
    layer_tag = "OBSERVED DATA" if is_observed_view else "MODELED SCENARIO"
    layer_col = BLUE if is_observed_view else ORANGE

    custom_html = f"""
    <div style="
        position: fixed; top: 14px; left: 60px; z-index: 9999;
        background: {layer_col}; color: white; padding: 8px 16px;
        border-radius: 8px; font-weight: 800; font-size: 16px; font-family: sans-serif;">
        {layer_tag}
    </div>
    <div style="
        position: fixed; bottom: 20px; left: 20px; z-index: 9999;
        background: rgba(255,255,255,0.95); padding: 10px 14px;
        border-radius: 6px; border: 1px solid #cbd5e1; font-family: sans-serif; font-size: 12px;">
        <b>CPCB PM2.5 Standards</b><br>
        <span style="color: #2e9e4f;">●</span> 0–30 Good<br>
        <span style="color: #8bc34a;">●</span> 31–60 Satisfactory<br>
        <span style="color: #f2c500;">●</span> 61–90 Moderate<br>
        <span style="color: #ff8c00;">●</span> 91–120 Poor<br>
        <span style="color: #e53935;">●</span> 121–250 Very Poor<br>
        <span style="color: #8b0000;">●</span> &gt;250 Severe
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(custom_html))

    render_badge("obs" if is_observed_view else "mod")
    st_folium(fmap, height=500, use_container_width=True, returned_objects=[])

    st.markdown("#### Neighborhood Pollution Hotspot Ranking")
    table_display = hotspot_df.drop(columns=["lat", "lon"]).set_index("Neighborhood").sort_values(
        active_col, ascending=False
    )
    st.dataframe(table_display, use_container_width=True)


# ============================================================================
# TAB 2: FORECASTING & HISTORICAL VALIDATION (Feature 2)
# ============================================================================
with tab_forecast:
    st.markdown(f"### Feature 2: Air Quality Forecasting & Historical Validation · {selected_station}")
    st.caption("Recursive 7-day forward predictions evaluated against strictly held-out historical observations.")

    c_b1, c_b2, c_b3 = st.columns(3)
    with c_b1:
        render_badge("obs", "OBSERVED (Up to 'Today')")
    with c_b2:
        render_badge("mod", "MODELED SCENARIO (Next 7 Days)")
    with c_b3:
        render_badge("val", "HISTORICAL VALIDATION (Ground Truth)")

    history_series = series.loc[origin_ts - pd.Timedelta(days=28):origin_ts, "pm25"]
    actual_future = series.loc[forecast_baseline.index, "pm25"]
    last_val = history_series.iloc[-1]

    fig_fc = go.Figure()

    # 1. Solid Blue Observed Line up to 'Today'
    fig_fc.add_trace(go.Scatter(
        x=history_series.index,
        y=history_series.values,
        name="Observed Data (Solid Blue)",
        line=dict(color=BLUE, width=3.5),
        mode="lines"
    ))

    # 2. Dashed Orange Modeled Scenario for Next 7 Days
    fig_fc.add_trace(go.Scatter(
        x=[origin_ts] + list(forecast_baseline.index),
        y=[last_val] + list(forecast_baseline.values),
        name="Modeled Scenario: Baseline Forecast (Dashed Orange)",
        line=dict(color=ORANGE, width=3.5, dash="dash"),
        mode="lines"
    ))

    # 3. Dotted Grey Actual Historical Validation Line
    fig_fc.add_trace(go.Scatter(
        x=[origin_ts] + list(actual_future.index),
        y=[last_val] + list(actual_future.values),
        name="Historical Validation: Actual Recorded (Dotted Grey)",
        line=dict(color=GREY, width=3, dash="dot"),
        mode="lines"
    ))

    # 4. Modeled Scenario with Active Interventions
    if active_interventions:
        fig_fc.add_trace(go.Scatter(
            x=[origin_ts] + list(forecast_scenario.index),
            y=[last_val] + list(forecast_scenario.values),
            name="Modeled Scenario: With Interventions (Dashed Green)",
            line=dict(color=GREEN, width=4, dash="dash"),
            mode="lines"
        ))

    # Replay origin indicator
    fig_fc.add_vline(x=origin_ts, line_width=1.5, line_dash="dash", line_color="#475569")
    fig_fc.add_annotation(
        x=origin_ts, y=1, yref="paper",
        text="Forecast Origin ('Today')",
        showarrow=False, yanchor="bottom", font=dict(color="#475569", size=12)
    )

    # NAAQS 60 ug/m3 line
    fig_fc.add_hline(
        y=E.NAAQS_PM25, line_width=1.5, line_dash="dot", line_color=RED,
        annotation_text="India 24-hr Standard: 60 µg/m³",
        annotation_position="bottom right"
    )

    fig_fc.update_layout(
        height=450,
        xaxis_title="Timeline",
        yaxis_title="PM2.5 (µg/m³)",
        legend=dict(orientation="h", y=-0.22, x=0),
        margin=dict(l=40, r=40, t=30, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("#### Historical Back-Test Metrics (90-Day Held-Out Test Period)")
    col_metrics, col_backtest_chart = st.columns([1, 1])

    with col_metrics:
        st.dataframe(metrics_table.round(2), use_container_width=True)
        st.markdown(
            f"""
            > **Evaluation Summary**:
            > - OmniTwin model achieves **{model_mae:.2f} µg/m³ MAE**, outperforming the persistence baseline by **{improvement_pct:.1f}%**.
            > - No historical test observations were used during model parameter estimation.
            """
        )

    with col_backtest_chart:
        bt_sorted = backtest_df.sort_values("date")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt_sorted["date"], y=bt_sorted["actual"],
            name="Actual Ground Truth",
            line=dict(color=GREY, width=2, dash="dot")
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt_sorted["date"], y=bt_sorted["pred"],
            name="Model Rolling Forecast",
            line=dict(color=ORANGE, width=2, dash="dash")
        ))
        fig_bt.update_layout(
            height=320,
            title="90-Day Rolling Forecast vs Ground Truth",
            yaxis_title="PM2.5 (µg/m³)",
            margin=dict(l=30, r=30, t=40, b=30),
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_bt, use_container_width=True)


# ============================================================================
# TAB 3: SOURCE ATTRIBUTION (Feature 3)
# ============================================================================
with tab_attribution:
    st.markdown(f"### Feature 3: PM2.5 Source Attribution · {selected_station}")
    st.caption("Constrained statistical attribution identifying relative sectoral shares.")

    render_badge("mod")

    # Hardcoded Stated Assumption (Strict PRD Requirement)
    st.markdown(
        f"""
        <div style="background: #fef3c7; border-left: 5px solid #f59e0b; padding: 12px 16px; border-radius: 6px; margin: 12px 0;">
            <span style="font-weight: 800; color: #92400e;">STATED ASSUMPTION:</span>
            <p style="margin: 4px 0 0 0; color: #78350f; font-weight: 500;">
                "{STATED_ASSUMPTION}"
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    recent_30d = series.loc[origin_ts - pd.Timedelta(days=29):origin_ts]
    recent_contrib = E.contributions(attribution_model, recent_30d).mean()

    c_attr1, c_attr2 = st.columns([1, 1])

    with c_attr1:
        # Donut Chart for Past 30 Days Attribution
        fig_pie = go.Figure(go.Pie(
            labels=list(recent_contrib.index),
            values=recent_contrib.values,
            hole=0.45,
            marker=dict(colors=[SRC_COLORS.get(k, "#cbd5e1") for k in recent_contrib.index]),
            textinfo="label+percent"
        ))
        fig_pie.update_layout(
            height=360,
            title="Estimated Source Contribution (Last 30 Days)",
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_attr2:
        # Stacked Bar Chart for the Next 7 Forecast Days
        fc_contrib = E.contributions(attribution_model, series.loc[forecast_baseline.index])
        scale_factors = forecast_baseline / fc_contrib.sum(axis=1).replace(0, np.nan)
        fc_contrib_scaled = fc_contrib.mul(scale_factors, axis=0).fillna(0)

        fig_stack = go.Figure()
        for source in fc_contrib_scaled.columns:
            fig_stack.add_trace(go.Bar(
                x=[d.strftime("%a %d %b") for d in fc_contrib_scaled.index],
                y=fc_contrib_scaled[source],
                name=source,
                marker_color=SRC_COLORS.get(source, "#cbd5e1"),
            ))

        fig_stack.update_layout(
            barmode="stack",
            height=360,
            title="Modeled 7-Day Forward Forecast Split by Source",
            yaxis_title="PM2.5 (µg/m³)",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("#### Neighborhood Source Profile Comparison")
    neigh_contribs = []
    for st_name in stations:
        st_s = get_station_series(df, st_name).loc[origin_ts - pd.Timedelta(days=29):origin_ts]
        c_mean = E.contributions(attribution_model, st_s).mean()
        shares = (c_mean / max(c_mean.sum(), 1e-9) * 100.0).rename(st_name)
        neigh_contribs.append(shares)

    neigh_df = pd.DataFrame(neigh_contribs)
    fig_neigh = go.Figure()
    for col_name in neigh_df.columns:
        fig_neigh.add_trace(go.Bar(
            y=neigh_df.index,
            x=neigh_df[col_name],
            name=col_name,
            orientation="h",
            marker_color=SRC_COLORS.get(col_name, "#cbd5e1"),
        ))
    fig_neigh.update_layout(
        barmode="stack",
        height=320,
        title="Cross-Neighborhood Source Variations in Pune",
        xaxis_title="Estimated Percentage Contribution (%)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=-0.3)
    )
    st.plotly_chart(fig_neigh, use_container_width=True)


# ============================================================================
# TAB 4: SCENARIO SIMULATOR (Feature 4)
# ============================================================================
with tab_scenarios:
    st.markdown(f"### Feature 4: Digital Twin Scenario Simulator · {selected_station}")
    st.caption("Compare the modeled impact of distinct policy interventions over the next 7 days.")

    render_badge("mod")

    # Define standard scenario runs
    scenario_matrix = {
        "Baseline (No Intervention)": {},
        "Action 1: Odd-Even Traffic Rule (-30% Vehicular)": {"Vehicular": action_sliders["Vehicular"]},
        "Action 2: Halt Heavy Industry (-80% Industrial)": {"Industrial": action_sliders["Industrial"]},
        "Action 3: Mandate Construction Sprinklers (-40% Dust)": {"Dust/Weather": action_sliders["Dust/Weather"]},
        "Combined: All 3 Actions Active": {
            "Vehicular": action_sliders["Vehicular"],
            "Industrial": action_sliders["Industrial"],
            "Dust/Weather": action_sliders["Dust/Weather"],
        }
    }

    if active_interventions:
        scenario_matrix["▶ Current Sidebar Selection"] = active_interventions

    summary_records = []
    for scn_label, scn_cuts in scenario_matrix.items():
        simulated_series = E.apply_actions(forecast_baseline, attribution_model, series, scn_cuts)
        mean_level = simulated_series.mean()
        reduction_abs = base_7d_mean - mean_level
        reduction_pct = (reduction_abs / max(base_7d_mean, 1e-6)) * 100.0
        days_above_naaqs = int((simulated_series > E.NAAQS_PM25).sum())

        summary_records.append({
            "Policy Scenario": scn_label,
            "Mean PM2.5 (µg/m³)": round(mean_level, 1),
            "Reduction (µg/m³)": round(reduction_abs, 1),
            "Reduction (%)": round(reduction_pct, 1),
            "Days > 60 µg/m³ (of 7)": days_above_naaqs,
        })

    summary_table = pd.DataFrame(summary_records).set_index("Policy Scenario")

    c_sc1, c_sc2 = st.columns([1, 1])

    with c_sc1:
        # Horizontal comparison bar
        bar_colors = []
        for name in summary_table.index:
            if "Baseline" in name:
                bar_colors.append(ORANGE)
            elif "All 3" in name or "Selection" in name:
                bar_colors.append(GREEN)
            else:
                bar_colors.append(BLUE)

        fig_sc_bar = go.Figure(go.Bar(
            x=summary_table["Mean PM2.5 (µg/m³)"],
            y=summary_table.index,
            orientation="h",
            marker_color=bar_colors,
            text=summary_table["Mean PM2.5 (µg/m³)"].astype(str) + " µg/m³",
            textposition="auto"
        ))
        fig_sc_bar.add_vline(x=E.NAAQS_PM25, line_dash="dash", line_color=RED, annotation_text="60 µg/m³ (NAAQS)")
        fig_sc_bar.update_layout(
            height=380,
            title="Comparison of Modeled PM2.5 Across Scenarios",
            xaxis_title="Mean Projected PM2.5 (µg/m³)",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sc_bar, use_container_width=True)

    with c_sc2:
        # Multi-line time series projection
        fig_sc_lines = go.Figure()
        fig_sc_lines.add_trace(go.Scatter(
            x=forecast_baseline.index, y=forecast_baseline.values,
            name="Baseline", line=dict(color=ORANGE, width=3, dash="dash")
        ))
        
        for scn_label, scn_cuts in scenario_matrix.items():
            if scn_label.startswith("Action") or "All 3" in scn_label or "Selection" in scn_label:
                sim_vals = E.apply_actions(forecast_baseline, attribution_model, series, scn_cuts)
                fig_sc_lines.add_trace(go.Scatter(
                    x=sim_vals.index, y=sim_vals.values,
                    name=scn_label.split(":")[0],
                    line=dict(width=2.2, dash="solid" if "All" in scn_label else "dash")
                ))

        fig_sc_lines.add_hline(y=E.NAAQS_PM25, line_dash="dot", line_color=RED, annotation_text="60 µg/m³")
        fig_sc_lines.update_layout(
            height=380,
            title="7-Day Forward Pollution Trajectories",
            yaxis_title="PM2.5 (µg/m³)",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_sc_lines, use_container_width=True)

    st.markdown("#### Scenario Impact Summary")
    st.dataframe(summary_table, use_container_width=True)

    best_action = summary_table.drop(index=[
        "Baseline (No Intervention)",
        "Combined: All 3 Actions Active",
    ] + (["▶ Current Sidebar Selection"] if active_interventions else []))["Reduction (µg/m³)"].idxmax()

    st.success(
        f"💡 **Key Insight**: Highest individual intervention impact in {selected_station} is **{best_action}**. "
        f"Intervention effectiveness is localized based on neighborhood emission shares."
    )


# ============================================================================
# TAB 5: METHODOLOGY & RUBRIC VERIFICATION
# ============================================================================
with tab_method:
    st.markdown("### Feature Matrix & Methodology")
    st.caption("Direct mapping to HackMatrix Problem Statement Requirements & Rubrics.")

    st.markdown(
        """
        | HackMatrix Expected Outcome | OmniTwin Implementation | UI Labeling & Evidence |
        |---|---|---|
        | **1. Forecasts at least one air-quality indicator** | Recursive 7-day daily PM2.5 forecasting using Ridge/Gradient Boosting with lag-1, lag-2, lag-3, lag-7, weather & cyclical harmonics. | Labeled as **MODELED SCENARIO** in orange dash line. |
        | **2. Attributes likely contributions across >=3 sources** | Non-Negative statistical regression estimating: 1) Vehicular, 2) Industrial, 3) Dust/Weather. | Stated assumption prominently displayed. Labeled as **MODELED SCENARIO**. |
        | **3. Compares >=3 possible actions to reduce pollution** | Action 1: Odd-Even (-30% vehicular), Action 2: Halt Heavy Industry (-80% industrial), Action 3: Construction Sprinklers (-40% dust). | Interactive toggles in sidebar + scenario matrix table & bar chart. |
        | **4. Displays hotspots on a map & validates against historical periods** | Folium interactive map with CPCB heat layer & circle markers. 90-day rolling back-test benchmarked against persistence. | Toggle between **OBSERVED DATA** and **MODELED SCENARIO**. Dotted grey validation line. |
        | **5. Clearly labels 'observed' separately from 'modeled'** | High-contrast visual badges: Solid Blue = OBSERVED DATA, Orange = MODELED SCENARIO, Grey = HISTORICAL VALIDATION. | Explicit badges and color codes applied on every single tab and chart. |
        """
    )

    st.markdown("---")
    st.markdown("#### Stated Assumptions & Scientific Limitations")
    st.info(
        f"• **Attribution Assumption**: {STATED_ASSUMPTION}\n\n"
        f"• **Meteorological Assumption**: Future 7-day weather inputs during historical replay are drawn from recorded meteorology (assumes a 7-day weather forecast is available).\n\n"
        f"• **Intervention Independence**: Actions are modeled as proportional cuts to specific sectoral shares without secondary chemical transformation or spatial leakage.\n\n"
        f"• **Statistical Nature**: Source attribution is derived from empirical statistical correlations, not a 3D Eulerian chemical transport model (e.g. WRF-Chem)."
    )

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption("OmniTwin · HackMatrix Prototype · Urban Environmental Digital Twin")
