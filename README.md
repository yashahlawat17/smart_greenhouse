# smart_greenhouse
AI driven IoT based smart greenhouse for societal well-being.
Interactive AI-driven IoT Greenhouse — Simulation (Streamlit)


What this provides:
- A software-only, presentation-ready simulation of an AI+IoT greenhouse.
- A dashboard showing simulated sensors, AI predictions, and actions.
- A tiny ML pipeline that trains on synthetic data when the app starts.


Files included in this doc:
- requirements.txt
- synth_data.py -> script that generates synthetic historical data
- app.py -> the Streamlit app you run


How to run (local):
1. Create a Python 3.8+ virtual environment and activate it.
2. Install dependencies:
pip install -r requirements.txt
3. Run the app:
streamlit run app.py


Where to host:
- Streamlit Community Cloud (recommended for easy sharing)
- Heroku / Render / any server that supports Python web apps
- Put the files into a GitHub repo and connect to Streamlit Cloud for one-click deploy


Notes:
- The app trains a small model at start using synthetic data (takes ~5–15s depending on your machine).
- No hardware required. Replace synthetic data generator with your real data source when ready.


Customization suggestions:
- Swap RandomForest for LSTM if you want time-series forecasting.
- Plug in image-based pest detection (YOLO) as a separate microservice.
- Add MQTT simulator to mimic real IoT device streams.
