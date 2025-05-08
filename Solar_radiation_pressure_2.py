#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the dataset
file_path = 'Space.weather.txt'
columns = ['Year', 'Month', 'Day', 'BSRN', 'ND', 'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8', 
           'Sum_Kp', 'Ap1', 'Ap2', 'Ap3', 'Ap4', 'Ap5', 'Ap6', 'Ap7', 'Ap8', 'Avg_Ap', 'Cp', 'C9', 'ISN', 
           'F10.7_1', 'Q1', 'Ctr81_1', 'Lst81_1', 'F10.7_2', 'Ctr81_2', 'Lst81_2']

# Identify the correct starting row
try:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    start_idx = next(i for i, line in enumerate(lines) if line.strip() and line[0].isdigit())
except FileNotFoundError:
    print("Error: 'Space.weather.txt' not found.")
    exit(1)
except StopIteration:
    print("Error: No valid data found in 'Space.weather.txt'.")
    exit(1)

# Read the data, skipping metadata rows
data = pd.read_csv(file_path, sep='\s+', names=columns, skiprows=start_idx, engine='python', on_bad_lines='skip')

# Create a proper date column
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors='coerce')
data = data.dropna(subset=['Date'])  # Remove rows with invalid dates
data = data.sort_values(by='Date')

# Aggregate to monthly averages for consistency
data['YearMonth'] = data['Date'].dt.to_period('M')
monthly_data = data.groupby('YearMonth').agg({
    'Date': 'first',  # Use the first date of the month
    'F10.7_1': 'mean'  # Average F10.7_1 for the month
}).reset_index()

# Prepare data for Prophet
prophet_df = monthly_data[['Date', 'F10.7_1']].rename(columns={'Date': 'ds', 'F10.7_1': 'y'})

# Handle missing values with interpolation
prophet_df['y'] = prophet_df['y'].interpolate()

# Initialize and configure Prophet model
model = Prophet(
    changepoint_prior_scale=0.5,  # Allow more flexibility in trend changes
    seasonality_prior_scale=10.0  # Allow more flexibility in seasonality
)
model.add_seasonality(name='solar_cycle', period=11*365.25, fourier_order=10)  # 11-year solar cycle
model.fit(prophet_df)

# Determine the last historical date
last_historical_date = monthly_data['Date'].max()
print(f"Last historical date: {last_historical_date}")

# Generate future dates for prediction (from the day after the last historical date to Dec 31, 2030)
start_date = last_historical_date + pd.Timedelta(days=1)
end_date = pd.to_datetime('2030-12-31')
future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # MS = Month Start
future_df = pd.DataFrame({'ds': future_dates})

# Forecast only for future dates
forecast = model.predict(future_df)

# Extract F10.7 values for 2000–2030
# Historical data (monthly averages)
historical_f10_7 = monthly_data[['Date', 'F10.7_1']].copy()
historical_f10_7 = historical_f10_7[
    (historical_f10_7['Date'] >= pd.to_datetime('2000-01-01')) &
    (historical_f10_7['Date'] <= pd.to_datetime('2030-12-31'))
].rename(columns={'Date': 'ds', 'F10.7_1': 'F10.7'})

# Predicted data (only for unknown dates)
predicted_f10_7 = forecast[['ds', 'yhat']].copy()
predicted_f10_7 = predicted_f10_7[
    (predicted_f10_7['ds'] >= pd.to_datetime('2000-01-01')) &
    (predicted_f10_7['ds'] <= pd.to_datetime('2030-12-31'))
].rename(columns={'yhat': 'F10.7'})

# Combine historical and predicted data
# Historical data takes precedence; predictions only for unknown dates
combined_f10_7 = pd.concat([historical_f10_7, predicted_f10_7])
combined_f10_7 = combined_f10_7.sort_values('ds').drop_duplicates(subset='ds', keep='first')

# Save to CSV
output_file = 'f10_7_2000_2030.csv'
combined_f10_7[['ds', 'F10.7']].to_csv(output_file, index=False)
print(f"F10.7 values from 2000 to 2030 saved to '{output_file}'")

# Plot results (focus on 2000–2030)
plt.figure(figsize=(12, 6))
plt.plot(historical_f10_7['ds'], historical_f10_7['F10.7'], label='Historical F10.7 (Monthly Avg)', color='blue')
plt.plot(predicted_f10_7['ds'], predicted_f10_7['F10.7'], label='Predicted F10.7', color='red', linestyle='dashed')
plt.fill_between(predicted_f10_7['ds'], 
                 forecast.loc[forecast['ds'].isin(predicted_f10_7['ds']), 'yhat_lower'], 
                 forecast.loc[forecast['ds'].isin(predicted_f10_7['ds']), 'yhat_upper'], 
                 color='red', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('F10.7 Value')
plt.title('F10.7 Historical and Predicted Values (2000–2030)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and save solar flux (as in original script)
f10_7_arr = forecast['yhat'].to_numpy()
average_10_7 = np.average(f10_7_arr)
solar_flux = f10_7_arr / average_10_7 * 1367
solar_flux_values = pd.DataFrame({'Date': forecast['ds'], 'Solar Flux': solar_flux})
solar_flux_values.to_csv('solar_flux_values.csv', index=False)
print("Solar flux values saved to 'solar_flux_values.csv'")