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

# Create a date column starting from Jan 1, 1932
data['Date'] = pd.to_datetime('1932-01-01') + pd.to_timedelta(data['days'] - 1, unit='D')

# Handle outliers in Ap (cap extreme values)
# data['Ap'] = data['Ap'].clip(upper=300)  # Cap Ap at 300 to handle extreme outliers

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

# Generate future dates for prediction (days 34076 to 39926, monthly)
start_day = 34076
end_day = 39926
start_date = pd.to_datetime('1932-01-01') + pd.to_timedelta(start_day - 1, unit='D')
end_date = pd.to_datetime('1932-01-01') + pd.to_timedelta(end_day - 1, unit='D')

# Create monthly future dates
future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # MS = Month Start
future_df = pd.DataFrame({'ds': future_dates})

# Forecast
forecast = model.predict(future_df)

# Plot results (focus on the last 20 years of history plus predictions)
plt.figure(figsize=(12, 6))
# Plot historical data (last 20 years)
plot_start_date = pd.to_datetime('1932-01-01')
historical_plot = prophet_df[prophet_df['ds'] >= plot_start_date]
plt.plot(historical_plot['ds'], historical_plot['y'], label='Historical Ap (Monthly Avg)', color='blue')

# Plot predictions
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Ap', color='red', linestyle='dashed')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)

plt.xlabel('Date')
plt.ylabel('Ap Value')
plt.title('Ap Historical and Predicted Values (Prophet Model)')
plt.legend()
plt.grid(True)
plt.show()

# Save predictions
predictions = pd.DataFrame({
    'Date': forecast['ds'],
    'Predicted_Ap': forecast['yhat'],
    'Lower_Bound': forecast['yhat_lower'],
    'Upper_Bound': forecast['yhat_upper']
})
predictions.to_csv('ap_predictions.csv', index=False)
print("Predictions saved to 'ap_predictions.csv'")

# Calculate RMSE for monthly predictions (if ground truth exists)
monthly_data_validation = data[(data['days'] >= 34076) & (data['days'] <= 39926)].copy()
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