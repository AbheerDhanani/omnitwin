"""
OmniTwin Analytical Engine: Data Processing, Forecasting, Backtesting,
Source Attribution, and Scenario Simulation.

Designed for the HackMatrix Urban Environmental Digital Twin Hackathon.
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# Scikit-learn imports with NumPy/SciPy fallback
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

HORIZON = 7          # 7-day forecast horizon
TEST_DAYS = 90       # 90-day held-out historical validation period
NAAQS_PM25 = 60.0    # India National Ambient Air Quality Standard (24-h average: 60 ug/m3)

NUM_COLS = [
    "pm25", "temp", "humidity", "wind", "rain",
    "traffic_index", "industrial_index"
]
REQUIRED = ["date", "station", "lat", "lon"] + NUM_COLS
SOURCES = ["Vehicular", "Industrial", "Dust/Weather"]

# Core Interventions matching the PRD
ACTIONS = [
    ("Action 1: Odd-Even Traffic Rule", "Vehicular", 0.30),
    ("Action 2: Halt Heavy Industry", "Industrial", 0.80),
    ("Action 3: Mandate Construction Sprinklers", "Dust/Weather", 0.40),
]

STATIONS = {
    "Shivajinagar": (18.5308, 73.8475, 1.25, 0.45, 0.85),
    "Hadapsar": (18.5089, 73.9260, 1.10, 1.35, 1.05),
    "Pimpri-Chinchwad": (18.6298, 73.7997, 1.00, 1.95, 0.90),
    "Katraj": (18.4575, 73.8677, 0.90, 0.40, 1.55),
    "Kothrud": (18.5074, 73.8077, 0.85, 0.30, 0.70),
}

ALIASES = {
    "pm2.5": "pm25", "pm_2.5": "pm25", "pm_2_5": "pm25", "pm25": "pm25",
    "wind_speed": "wind", "windspeed": "wind", "wind_spd": "wind",
    "traffic": "traffic_index", "traffic_index": "traffic_index",
    "industrial": "industrial_index", "temperature": "temp",
    "rainfall": "rain", "precipitation": "rain", "humid": "humidity",
    "latitude": "lat", "longitude": "lon", "city": "station", "location": "station",
    "neighbourhood": "station", "neighborhood": "station", "area": "station"
}

MINIMUM_COLS = ["date", "pm25", "traffic_index", "wind"]
DEFAULTS = {"temp": 25.0, "humidity": 50.0, "rain": 0.0, "industrial_index": 50.0}


# ----------------------------------------------------------------------------
# Fallback Estimators when scikit-learn is not installed
# ----------------------------------------------------------------------------
class FallbackRidgeRegressor:
    """NumPy-based Ridge Regression with StandardScaler."""
    def __init__(self, alpha=5.0):
        self.alpha = alpha
        self.mean_x = None
        self.std_x = None
        self.weights = None
        self.intercept = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_x = np.mean(X, axis=0)
        self.std_x = np.std(X, axis=0)
        self.std_x[self.std_x < 1e-8] = 1.0

        X_scaled = (X - self.mean_x) / self.std_x
        # Add intercept column
        n, p = X_scaled.shape
        X_design = np.column_stack([np.ones(n), X_scaled])
        eye = np.eye(p + 1)
        eye[0, 0] = 0.0  # Do not regularize intercept

        # Solve (X^T X + alpha*I) beta = X^T y
        A = X_design.T @ X_design + self.alpha * eye
        b = X_design.T @ y
        beta = np.linalg.solve(A, b)
        self.intercept = beta[0]
        self.weights = beta[1:]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = (X - self.mean_x) / self.std_x
        return X_scaled @ self.weights + self.intercept


def calc_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / max(ss_tot, 1e-9)))


# ----------------------------------------------------------------------------
# Data Cleaning & Normalization
# ----------------------------------------------------------------------------
def prepare_df(df: pd.DataFrame, lat=18.5204, lon=73.8567, area="Pune Urban Region"):
    """
    Accepts any uploaded CSV (CPCB/Kaggle or synthetic), validates minimum schema,
    resolves synonyms, imputes missing defaults, and returns a clean standardized DataFrame.
    """
    df = df.copy()
    norm = lambda c: str(c).strip().lower().replace(" ", "_")
    df.columns = [ALIASES.get(norm(c), norm(c)) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in MINIMUM_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV must contain at least: Date, PM2.5, Traffic_Index, Wind_Speed. "
            f"Missing: {missing}. Present: {list(df.columns)}"
        )

    notes = []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "pm25"])

    if "station" not in df:
        df["station"] = area
        notes.append("No station/neighbourhood column: treated as single area.")

    if "lat" not in df or "lon" not in df:
        df["lat"], df["lon"] = lat, lon
        if "station" in df and df["station"].nunique() > 1:
            notes.append("No coordinates: using default city coordinates for all stations.")

    for k, v in DEFAULTS.items():
        if k not in df:
            df[k] = v
            notes.append(f"No '{k}' column: filled with baseline {v}.")

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["pm25", "traffic_index", "wind"])
    agg = {c: "mean" for c in NUM_COLS}
    agg.update(lat="first", lon="first")
    df = df.groupby(["station", "date"], as_index=False).agg(agg).sort_values(["station", "date"])
    return df, notes


def get_series(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """Extracts continuous daily time-series with dust proxy feature."""
    if station == "City average":
        s = df.groupby("date")[NUM_COLS].mean()
    else:
        s = df[df.station == station].set_index("date")[NUM_COLS].sort_index()

    s = s.asfreq("D").interpolate(limit_direction="both")
    # Physical Dust Proxy: Wind speed * dry atmospheric fraction * dry ground condition
    s["dust"] = s["wind"] * (1.0 - np.clip(s["humidity"], 0, 100) / 100.0) * (s["rain"] < 0.5).astype(float)
    return s


def test_start_date(s: pd.DataFrame) -> pd.Timestamp:
    """Historical test period begins exactly TEST_DAYS before the end of the series."""
    return s.index.max() - pd.Timedelta(days=TEST_DAYS - 1)


# ----------------------------------------------------------------------------
# Feature Engineering & Model Training
# ----------------------------------------------------------------------------
def row_features(window, ex, date):
    """
    Builds lag features (t-1, t-2, t-3, t-7, 7d-mean), exogenous weather/activity,
    and annual cyclical harmonics.
    """
    doy = date.dayofyear
    return [
        window[-1], window[-2], window[-3], window[-7],
        float(np.mean(window)),
        ex["temp"], ex["humidity"], ex["wind"], ex["rain"],
        ex["traffic_index"], ex["industrial_index"],
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ]


def make_xy(s: pd.DataFrame):
    pm = s["pm25"].values
    X, y = [], []
    for t in range(7, len(s)):
        X.append(row_features(pm[t - 7:t], s.iloc[t], s.index[t]))
        y.append(pm[t])
    return np.array(X), np.array(y)


def new_model(kind: str):
    if not HAS_SKLEARN:
        return FallbackRidgeRegressor(alpha=5.0)

    if kind == "Ridge regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=5.0))

    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )


def fit_forecaster(s: pd.DataFrame, kind: str):
    """
    Trains model strictly BEFORE the held-out historical test period.
    Guarantees ZERO data leakage for genuine historical validation.
    """
    train = s[s.index < test_start_date(s)]
    if len(train) < 30:
        raise ValueError(f"Need at least 30 training days (found {len(train)}).")
    X, y = make_xy(train)
    return new_model(kind).fit(X, y)


def forecast(model, s: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """
    Recursive 7-day forward prediction from replay origin ('Today').
    """
    if origin not in s.index:
        raise ValueError("Replay date is not in the time series index.")

    window = list(s.loc[:origin, "pm25"].values[-7:])
    if len(window) < 7:
        raise ValueError("Need at least 7 historical observations preceding replay date.")

    days = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON)
    out = []
    for d in days:
        if d not in s.index:
            raise ValueError(f"Driver data missing for forecast date {d.date()}")
        x = row_features(window[-7:], s.loc[d], d)
        p = float(max(model.predict([x])[0], 2.0))
        out.append(p)
        window.append(p)

    return pd.Series(out, index=days, name="forecast")


def backtest(model, s: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling-origin backtest across the held-out test period at 7-day intervals.
    """
    ts = test_start_date(s)
    origin = ts - pd.Timedelta(days=1)
    rows = []
    while origin + pd.Timedelta(days=HORIZON) <= s.index.max():
        fc = forecast(model, s, origin)
        persist = s.loc[origin, "pm25"]
        for h, (d, p) in enumerate(fc.items(), start=1):
            rows.append((origin, d, h, p, s.loc[d, "pm25"], persist))
        origin += pd.Timedelta(days=HORIZON)

    return pd.DataFrame(rows, columns=["origin", "date", "horizon", "pred", "actual", "persistence"])


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """Computes benchmark scoring table against persistence baseline."""
    if HAS_SKLEARN:
        def m(col):
            return {
                "MAE (µg/m³)": mean_absolute_error(bt.actual, bt[col]),
                "RMSE (µg/m³)": float(np.sqrt(mean_squared_error(bt.actual, bt[col]))),
                "R²": r2_score(bt.actual, bt[col]),
            }
    else:
        def m(col):
            return {
                "MAE (µg/m³)": calc_mae(bt.actual.values, bt[col].values),
                "RMSE (µg/m³)": calc_rmse(bt.actual.values, bt[col].values),
                "R²": calc_r2(bt.actual.values, bt[col].values),
            }

    return pd.DataFrame({
        "OmniTwin model": m("pred"),
        "Persistence baseline": m("persistence")
    }).T


# ----------------------------------------------------------------------------
# Source Attribution & Digital Twin Scenarios
# ----------------------------------------------------------------------------
class Attribution:
    def __init__(self, coef, intercept, industrial_assumed):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(intercept)
        self.industrial_assumed = industrial_assumed


def fit_attribution(df: pd.DataFrame, origin: pd.Timestamp, industrial_share: float = 0.15) -> Attribution:
    """
    Interpretable statistical regression decomposing PM2.5 into:
      PM2.5 = beta_traffic * Traffic + beta_ind * Industrial + beta_dust * Dust + Intercept
    Using Non-Negative Least Squares (NNLS) to guarantee physical feasibility.
    """
    window = df[(df.date <= origin) & (df.date > origin - pd.Timedelta(days=365))].copy()
    window["dust"] = window["wind"] * (1.0 - np.clip(window["humidity"], 0, 100) / 100.0) * (window["rain"] < 0.5).astype(float)

    ind = window["industrial_index"]
    # Check if industrial variation is statistically identifiable
    if ind.std() < 0.05 * max(ind.mean(), 1e-9) or window["station"].nunique() == 1:
        # Heuristic prior based on stated assumption
        k = industrial_share * window["pm25"].mean() / max(ind.mean(), 1e-9)
        residual = np.clip(window["pm25"].values - k * ind.values, 0, None)

        X = np.column_stack([
            window["traffic_index"].values,
            window["dust"].values,
            np.ones(len(window))
        ])
        coefs, _ = nnls(X, residual)
        return Attribution([coefs[0], k, coefs[1]], coefs[2], True)

    X = np.column_stack([
        window["traffic_index"].values,
        window["industrial_index"].values,
        window["dust"].values,
        np.ones(len(window))
    ])
    coefs, _ = nnls(X, window["pm25"].values)
    return Attribution([coefs[0], coefs[1], coefs[2]], coefs[3], False)


def contributions(attribution_model: Attribution, rows: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame({
        "Vehicular": attribution_model.coef_[0] * rows["traffic_index"],
        "Industrial": attribution_model.coef_[1] * rows["industrial_index"],
        "Dust/Weather": attribution_model.coef_[2] * rows["dust"],
    }, index=rows.index)
    c["Background/regional"] = max(attribution_model.intercept_, 0.0)
    return c


def apply_actions(forecast_values: pd.Series, attribution_model: Attribution,
                  series: pd.DataFrame, cuts: dict) -> pd.Series:
    """
    Applies intervention percentage reductions specifically to their corresponding
    modeled emission source shares.
    """
    c = contributions(attribution_model, series.loc[forecast_values.index])
    source_total = c[SOURCES].sum(axis=1).replace(0, np.nan)
    shares = c[SOURCES].div(source_total, axis=0).fillna(0)

    total_cut = sum(cuts.get(source, 0.0) * shares[source] for source in SOURCES)
    return forecast_values * (1.0 - total_cut.clip(0.0, 0.95))


def pm_category(value: float):
    """CPCB National Air Quality Index (NAQI) PM2.5 categories and official color codes."""
    if value <= 30.0:
        return "#2e9e4f", "Good"
    if value <= 60.0:
        return "#8bc34a", "Satisfactory"
    if value <= 90.0:
        return "#f2c500", "Moderate"
    if value <= 120.0:
        return "#ff8c00", "Poor"
    if value <= 250.0:
        return "#e53935", "Very Poor"
    return "#8b0000", "Severe"
