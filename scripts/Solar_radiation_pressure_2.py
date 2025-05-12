
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the dataset
file_path = 'Space.weather.txt'
columns = ['Year', 'Month', 'Day', 'BSRN', 'ND', 'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8', 'Sum_Kp', 'Ap1', 'Ap2', 'Ap3', 'Ap4', 'Ap5', 'Ap6', 'Ap7', 'Ap8', 'Avg_Ap', 'Cp', 'C9', 'ISN', 'F10.7_1', 'Q1', 'Ctr81_1', 'Lst81_1', 'F10.7_2', 'Ctr81_2', 'Lst81_2']

# Identify the correct starting row
with open(file_path, 'r') as file:
    lines = file.readlines()

start_idx = next(i for i, line in enumerate(lines) if line.strip() and line[0].isdigit())

# Read the data, skipping metadata rows
data = pd.read_csv(file_path, sep='\s+', names=columns, skiprows=start_idx, engine='python', on_bad_lines='skip')

# Create a proper date column
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors='coerce')
data = data.dropna(subset=['Date'])  # Remove rows with invalid dates
data = data.sort_values(by='Date')

# Selecting relevant columns for Prophet
prophet_df = data[['Date', 'F10.7_1']].rename(columns={'Date': 'ds', 'F10.7_1': 'y'})

# Handle missing values with interpolation
prophet_df['y'] = prophet_df['y'].interpolate()

# Initialize and configure Prophet model
model = Prophet()
model.add_seasonality(name='solar_cycle', period=11*365.25, fourier_order=10)  # 11-year solar cycle
model.fit(prophet_df)

# Generate future dates
future = model.make_future_dataframe(periods=15*365, freq='D')  # Predicting 15 years ahead
forecast = model.predict(future)
'''
# Plot results
plt.figure(figsize=(12, 6))
plt.plot(prophet_df['ds'], prophet_df['y'], label='Historical F10.7_1', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted F10.7_1', color='red', linestyle='dashed')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('F10.7 Value')
plt.title('F10.7 Historical and Predicted Values (Prophet Model)')
plt.legend()
plt.show()
'''
# The 'trend' column contains all the values of F10.7

f10_7_arr = forecast['yhat'].to_numpy()

average_10_7 = np.average(f10_7_arr)

Solar_flux = f10_7_arr/average_10_7*1367

forecast = forecast['ds'].to_numpy()

Solar_flux_values = pd.DataFrame({'Date':forecast, 'Solar Flux':Solar_flux})

f10_7_values = pd.DataFrame({'Date':forecast, 'F10.7':f10_7_arr})


