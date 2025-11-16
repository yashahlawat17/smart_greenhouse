# app.py
st.subheader('Recommended actuator actions')
act_cols = st.columns(3)
act_cols[0].write('Irrigation')
act_cols[0].markdown('**ON**' if irrigation else 'OFF')
act_cols[1].write('Ventilation')
act_cols[1].markdown('**ON**' if ventilation else 'OFF')
act_cols[2].write('Shade')
act_cols[2].markdown('**ON**' if shade else 'OFF')


# show small time-series plots (last 72 hours)
window = df.iloc[max(0, timestep-72):timestep+1].copy()
st.subheader('Recent trends (last 72 hours)')
fig = px.line(window, x='timestamp', y=['temp','humidity','soil_moisture'], labels={'value':'Value','timestamp':'Time'})
st.plotly_chart(fig, use_container_width=True)


st.subheader('Yield estimate over next hours (simulated)')
# naive: show predicted yield for next 24 hours using model + synthetic forward sim
future_rows = []
last_row = current.copy()
future_rows = []
last_row = current.copy()

for i in range(24):
    # small random walk for sensors
    last_row['temp'] = last_row['temp'] + np.random.normal(scale=0.5)
    last_row['humidity'] = last_row['humidity'] + np.random.normal(scale=1.0)
    last_row['soil_moisture'] = last_row['soil_moisture'] - 0.2 + np.random.normal(scale=0.5)
    last_row['light'] = max(0, last_row['light'] + np.random.normal(scale=30))
    last_row['co2'] = last_row['co2'] + np.random.normal(scale=2.0)

    feat = np.array(last_row[['temp','humidity','soil_moisture','light','co2']]).reshape(1,-1)
    ypred = float(m_yield.predict(feat)[0])

    future_rows.append({'hour': i+1, 'pred_yield': ypred})
fut_df = pd.DataFrame(future_rows)
fig2 = px.line(fut_df, x='hour', y='pred_yield', labels={'pred_yield':'Predicted yield (norm)'})
st.plotly_chart(fig2, use_container_width=True)


st.sidebar.markdown('---')
st.sidebar.write('Model RMSEs (trained on synthetic data)')
st.sidebar.write(preds)

