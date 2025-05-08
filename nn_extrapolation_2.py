#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the dataset
file_path = 'snfuture.csv'
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: 'snfuture.csv' not found.")
    exit(1)

# Verify expected columns
expected_columns = ['days', 'dB', 'SN', 'Ap']
if not all(col in data.columns for col in expected_columns):
    print(f"Error: Dataset must contain columns {expected_columns}")
    exit(1)

# Create a date column starting from Jan 1, 1932
data['Date'] = pd.to_datetime('1932-01-01') + pd.to_timedelta(data['days'] - 1, unit='D')

# Aggregate historical data to monthly averages to reduce noise
data['YearMonth'] = data['Date'].dt.to_period('M')
monthly_data = data.groupby('YearMonth').agg({
    'Date': 'first',  # Use the first date of the month
    'Ap': 'mean',     # Average Ap for the month
    'days': 'first'   # Keep the first day of the month
}).reset_index()

# Filter historical data (up to day 34046)
historical_data = monthly_data[monthly_data['days'] <= 34046].copy()

# Prepare data for Prophet
prophet_df = historical_data[['Date', 'Ap']].rename(columns={'Date': 'ds', 'Ap': 'y'})

# Initialize and configure Prophet model
model = Prophet(
    changepoint_prior_scale=0.5,  # Allow more flexibility in trend changes
    seasonality_prior_scale=10.0  # Allow more flexibility in seasonality
)
model.add_seasonality(name='solar_cycle', period=11 * 365.25, fourier_order=10)  # 11-year solar cycle
model.add_seasonality(name='yearly', period=365.25, fourier_order=5)  # Yearly seasonality
model.fit(prophet_df)

# Determine the last historical date
last_historical_date = historical_data['Date'].max()
print(f"Last historical date: {last_historical_date}")

# Generate future dates for prediction (from the day after the last historical date to Dec 1, 2030)
start_date = last_historical_date + pd.Timedelta(days=1)
end_date = pd.to_datetime('2030-12-01')
future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # MS = Month Start
future_df = pd.DataFrame({'ds': future_dates})

# Forecast only for future dates
forecast = model.predict(future_df)

# Extract Ap values for 2000–2030
# Historical data (monthly averages)
historical_ap = monthly_data[['Date', 'Ap']].copy()
historical_ap = historical_ap[
    (historical_ap['Date'] >= pd.to_datetime('2000-01-01')) &
    (historical_ap['Date'] <= pd.to_datetime('2030-12-01'))
].rename(columns={'Date': 'ds', 'Ap': 'Ap'})

# Predicted data (only for unknown dates)
predicted_ap = forecast[['ds', 'yhat']].copy()
predicted_ap = predicted_ap[
    (predicted_ap['ds'] >= pd.to_datetime('2000-01-01')) &
    (predicted_ap['ds'] <= pd.to_datetime('2030-12-01'))
].rename(columns={'yhat': 'Ap'})

# Combine historical and predicted data
# Historical data takes precedence; predictions only for unknown dates
combined_ap = pd.concat([historical_ap, predicted_ap])
combined_ap = combined_ap.sort_values('ds').drop_duplicates(subset='ds', keep='first')

# Filter to keep only first-of-the-month dates
combined_ap = combined_ap[combined_ap['ds'].dt.day == 1]

# Remove any rows with missing Ap values
combined_ap = combined_ap.dropna(subset=['Ap'])

# Save to CSV
output_file = 'ap_2000_2030.csv'
combined_ap[['ds', 'Ap']].to_csv(output_file, index=False)
print(f"Ap values from 2000 to 2030 saved to '{output_file}'")

# Plot results (focus on 2000–2030)
plt.figure(figsize=(12, 6))
plt.plot(historical_ap['ds'], historical_ap['Ap'], label='Historical Ap (Monthly Avg)', color='blue')
plt.plot(predicted_ap['ds'], predicted_ap['Ap'], label='Predicted Ap', color='red', linestyle='dashed')
plt.fill_between(predicted_ap['ds'], 
                 forecast.loc[forecast['ds'].isin(predicted_ap['ds']), 'yhat_lower'], 
                 forecast.loc[forecast['ds'].isin(predicted_ap['ds']), 'yhat_upper'], 
                 color='red', alpha=0.2)
plt.xlabel('Date')
plt.ylabel('Ap Value')
plt.title('Ap Historical and Predicted Values (2000–2030)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE for monthly predictions (if ground truth exists beyond day 34046)
monthly_data_validation = data[(data['days'] > 34046) & (data['days'] <= 39926)].copy()
if not monthly_data_validation.empty:
    monthly_data_validation['Date'] = pd.to_datetime('1932-01-01') + pd.to_timedelta(monthly_data_validation['days'] - 1, unit='D')
    monthly_data_validation['YearMonth'] = monthly_data_validation['Date'].dt.to_period('M')
    monthly_avg = monthly_data_validation.groupby('YearMonth').agg({
        'Date': 'first',
        'Ap': 'mean'
    }).reset_index()
    
    forecast['YearMonth'] = forecast['ds'].dt.to_period('M')
    merged = pd.merge(
        monthly_avg,
        forecast[['YearMonth', 'yhat']],
        on='YearMonth',
        how='inner'
    )
    
    if not merged.empty:
        rmse = np.sqrt(np.mean((merged['Ap'] - merged['yhat']) ** 2))
        print(f"Root Mean Squared Error for monthly predictions: {rmse:.4f}")
    else:
        print("Warning: No overlapping dates for RMSE calculation.")
else:
    print("No ground truth monthly data available for RMSE calculation.")