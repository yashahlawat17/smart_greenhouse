# synth_data.py
np.random.seed(RANDOM_SEED)


# generate N days of hourly data
def generate(N_days=120):
hours = N_days * 24
t = np.arange(hours)


# base diurnal temperature pattern + seasonal drift
temp = 20 + 6 * np.sin(2 * np.pi * (t % 24) / 24) + 0.01 * (t / 24)
temp += np.random.normal(scale=0.8, size=hours)


# humidity inversely related to temp with noise
humidity = 65 - 0.8 * (temp - 20) + np.random.normal(scale=3.0, size=hours)


# soil moisture decreases slowly and is influenced by irrigation events
soil = 40 + 5 * np.sin(2 * np.pi * (t / (24*7))) + np.random.normal(scale=2.5, size=hours)


# light: strong daytime/weak night
hour_of_day = t % 24
light = np.where((hour_of_day>=6) & (hour_of_day<=18),
1000 * np.sin(np.pi*(hour_of_day-6)/12) + np.random.normal(scale=50, size=hours),
np.random.normal(scale=10, size=hours))


co2 = 400 + 30 * np.sin(2 * np.pi * (t / 24)) + np.random.normal(scale=6.0, size=hours)


# target variables (what our AI will predict)
# water_need (0-1), pest_risk (0-1), yield_est (normalized)
water_need = np.clip((45 - soil) / 30 + 0.1*(temp - 22)/10 + np.random.normal(scale=0.05, size=hours), 0, 1)
pest_risk = np.clip(0.05 + 0.2 * (humidity - 60)/40 + 0.1 * (temp - 25)/10 + np.random.normal(scale=0.03, size=hours), 0, 1)
yield_est = np.clip(0.6 + 0.1 * (light/1000) + 0.05*(humidity-60)/40 - 0.03*(pest_risk) + np.random.normal(scale=0.03, size=hours), 0, 1)


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


if __name__ == '__main__':
df = generate(90)
print(df.head())
df.to_csv('synthetic_greenhouse.csv', index=False)
