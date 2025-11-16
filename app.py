import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------
@st.cache_data
def synth_generate(N_days=120, seed=42):
    np.random.seed(seed)
    hours = N_days * 24
    t = np.arange(hours)

    # temperature
    temp = 20 + 6 * np.sin(2 * np.pi * (t % 24) / 24) + 0.01 * (t / 24) 
    temp += np.random.normal(scale=0.8, size=hours)

    # humidity
    humidity = 65 - 0.8 * (temp - 20) + np.random.normal(scale=3.0, size=hours)

    # soil moisture
    soil = 40 + 5 * np.sin(2 * np.pi * (t / (24*7))) 
    soil += np.random.normal(scale=2.5, size=hours)

    # light
    hour_of_day = t % 24
    light = np.where(
        (hour_of_day >= 6) & (hour_of_day <= 18),
        1000 * np.sin(np.pi*(hour_of_day-6)/12) + np.random.normal(scale=50, size=hours),
        np.random.normal(scale=10, size=hours)
    )

    # CO2
    co2 = 400 + 30 * np.sin(2 * np.pi * (t / 24)) + np.random.normal(scale=6.0, size=hours)

    # AI target variables
    water_need = np.clip((45 - soil) / 30 + 0.1*(temp - 22)/10 + np.random.normal(0, 0.05, hours), 0, 1)
    pest_risk = np.clip(0.05 + 0.2*(humidity - 60)/40 + 0.1*(temp - 25)/10 + np.random.normal(0, 0.03, hours), 0, 1)
    yield_est = np.clip(0.6 + 0.1*(light/1000) + 0.05*(humidity-60)/40 - 0.03*pest_risk + np.random.normal(0, 0.03, hours), 0, 1)

    df = pd.DataFrame({
        'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=hours, freq='H'),
        'temp': temp,
        'humidity': humidity,
        'soil_moisture': soil,
        'light': light,
        'co2': co2,
        'water_need': water_need,
        'pest_risk': pest_risk,
        'yield_est': yield_est
    })

    return df


# ---------------------------------------------------------
# Train AI models
# ---------------------------------------------------------
@st.cache_resource
def train_models(df):
    features = ['temp', 'humidity', 'soil_moisture', 'light', 'co2']
    X = df[features]

    y_water = df['water_need']
    y_pest = df['pest_risk']
    y_yield = df['yield_est']

    X_train, X_test, yw_train, yw_test = train_test_split(X, y_water, test_size=0.2, random_state=42)
    _, _, yp_train, yp_test = train_test_split(X, y_pest, test_size=0.2, random_state=42)
    _, _, yy_train, yy_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)

    m_water = RandomForestRegressor(n_estimators=40, random_state=42)
    m_pest = RandomForestRegressor(n_estimators=40, random_state=42)
    m_yield = RandomForestRegressor(n_estimators=40, random_state=42)

    m_water.fit(X_train, yw_train)
    m_pest.fit(X_train, yp_train)
    m_yield.fit(X_train, yy_train)

    preds = {
        'water_rmse': mean_squared_error(yw_test, m_water.predict(X_test), squared=False),
        'pest_rmse': mean_squared_error(yp_test, m_pest.predict(X_test), squared=False),
        'yield_rmse': mean_squared_error(yy_test, m_yield.predict(X_test), squared=False),
    }

    return m_water, m_pest, m_yield, preds


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Greenhouse Simulation", layout="wide")
st.title("🌱 AI-driven IoT Greenhouse — Interactive Simulation")


# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Simulation Controls")

    days = st.slider("Number of days of synthetic training data:", 30, 365, 120)
    df = synth_generate(N_days=days)
    st.write("Dataset size:", len(df), "rows")

    timestep = st.slider("Select hour index", 0, len(df)-1, len(df)-24)

    st.subheader("Manual Override")
    manual_irrigation = st.checkbox("Force irrigation ON")
    manual_vent = st.checkbox("Force ventilation ON")
    manual_shade = st.checkbox("Force shade ON")

with col2:
    st.header("Dashboard")

    current = df.iloc[timestep]

    # Sensor readings
    st.subheader("Current Sensor Values")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Temp (°C)", f"{current['temp']:.2f}")
    s2.metric("Humidity (%)", f"{current['humidity']:.2f}")
    s3.metric("Soil Moisture (%)", f"{current['soil_moisture']:.2f}")
    s4.metric("Light (lux)", f"{current['light']:.0f}")
    s5.metric("CO₂ (ppm)", f"{current['co2']:.0f}")

    # AI models
    m_water, m_pest, m_yield, preds = train_models(df)
    features = np.array(current[['temp','humidity','soil_moisture','light','co2']]).reshape(1, -1)

    pred_water = float(m_water.predict(features)[0])
    pred_pest = float(m_pest.predict(features)[0])
    pred_yield = float(m_yield.predict(features)[0])

    st.subheader("AI Predictions")
    p1, p2, p3 = st.columns(3)
    p1.metric("Water Need (0–1)", f"{pred_water:.2f}")
    p2.metric("Pest Risk (0–1)", f"{pred_pest:.2f}")
    p3.metric("Yield Estimation (0–1)", f"{pred_yield:.2f}")

    # Actuator decisions
    irrigation = manual_irrigation or (pred_water > 0.45 and current['soil_moisture'] < 45)
    ventilation = manual_vent or (current['temp'] > 28 or pred_pest > 0.4)
    shade = manual_shade or (current['light'] > 800 and current['temp'] > 25)

    st.subheader("Recommended Actuator Actions")
    a1, a2, a3 = st.columns(3)
    a1.metric("Irrigation", "ON" if irrigation else "OFF")
    a2.metric("Ventilation", "ON" if ventilation else "OFF")
    a3.metric("Shade", "ON" if shade else "OFF")

    # Past 72-hour chart
    st.subheader("Last 72 Hours Trend")
    window = df.iloc[max(0, timestep-72): timestep+1]
    fig = px.line(window, x='timestamp', y=['temp','humidity','soil_moisture'])
    st.plotly_chart(fig, use_container_width=True)

    # Future yield prediction
    st.subheader("Predicted Yield for Next 24 Hours")
    future_rows = []
    last_row = current.copy()

    for i in range(24):
        last_row['temp'] += np.random.normal(scale=0.5)
        last_row['humidity'] += np.random.normal(scale=1.0)
        last_row['soil_moisture'] += -0.2 + np.random.normal(scale=0.5)
        last_row['light'] = max(0, last_row['light'] + np.random.normal(scale=30))
        last_row['co2'] += np.random.normal(scale=2.0)

        feat = np.array(last_row[['temp','humidity','soil_moisture','light','co2']]).reshape(1, -1)
        ypred = float(m_yield.predict(feat)[0])

        future_rows.append({"hour": i+1, "pred_yield": ypred})

    fut_df = pd.DataFrame(future_rows)
    fig2 = px.line(fut_df, x='hour', y='pred_yield')
    st.plotly_chart(fig2, use_container_width=True)


st.sidebar.header("Model Performance (RMSE)")
st.sidebar.json(preds)


