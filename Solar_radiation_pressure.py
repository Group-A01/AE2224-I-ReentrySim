import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load fluxtable data
flux_file_path = "fluxtable.txt"
flux_data = pd.read_csv(flux_file_path, sep='\s+', comment='#', engine='python')

# Remove potential header/footer rows and ensure numerical conversion
flux_data = flux_data[pd.to_numeric(flux_data['fluxdate'], errors='coerce').notna()]
flux_data['fluxdate'] = pd.to_datetime(flux_data['fluxdate'].astype(int).astype(str), format='%Y%m%d')
flux_data['fluxobsflux'] = pd.to_numeric(flux_data['fluxobsflux'], errors='coerce')

# Set fixed date for reference
yesterday = datetime(2025, 3, 12)

# Filter data until yesterday
flux_data = flux_data[flux_data['fluxdate'] <= yesterday]

# Compute the average of fluxobsflux up to yesterday
average_flux = flux_data['fluxobsflux'].mean()

# Compute the scaled flux
flux_data['scaled_flux'] = (flux_data['fluxobsflux'] / average_flux) * 1360

# Load SW-Last5Years data
sw_file_path = "SW-Last5Years.txt"
sw_data = pd.read_csv(sw_file_path, sep='\s+', comment='#', skiprows=18, header=None, engine='python')

# Define column names based on known format
sw_columns = ['year', 'month', 'day', 'other', 'other2', 'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8', 'other3', 'Ap1', 'Ap2', 'Ap3', 'Ap4', 'Ap5', 'Ap6', 'Ap7', 'Ap8', 'other4', 'other5', 'other6', 'F10.7']
sw_data = sw_data.iloc[:, :len(sw_columns)]  # Trim to expected columns
sw_data.columns = sw_columns

# Remove potential footer rows
sw_data = sw_data[pd.to_numeric(sw_data['year'], errors='coerce').notna()]

# Convert date columns to datetime
sw_data['date'] = pd.to_datetime(sw_data[['year', 'month', 'day']])
sw_data['F10.7'] = pd.to_numeric(sw_data['F10.7'], errors='coerce')

# Set fixed date for today and last predicted date
today = datetime(2025, 3, 13)
last_predicted_date = datetime(2041, 10, 1)

# Filter data from today until last predicted date
sw_data = sw_data[(sw_data['date'] >= today) & (sw_data['date'] <= last_predicted_date)]

# Replace future missing values with the last known prediction (69.8)
sw_data['F10.7'].fillna(69.8, inplace=True)

# Compute the average of obs from today onwards
average_obs = sw_data['F10.7'].mean()

# Compute the scaled obs
sw_data['scaled_obs'] = (sw_data['F10.7'] / average_obs) * 1360

# Combine data for plotting
all_dates = pd.concat([flux_data['fluxdate'], sw_data['date']])
all_scaled = pd.concat([flux_data['scaled_flux'], sw_data['scaled_obs']])

# Remove outliers using a threshold (e.g., 3 times the standard deviation)
outlier_threshold = all_scaled.mean() + 3 * all_scaled.std()
all_scaled = all_scaled.where(all_scaled < outlier_threshold, np.nan).interpolate()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(all_dates, all_scaled, label='Scaled Solar Flux', color='orange')
plt.xlabel('Year')
plt.ylabel('Scaled Solar Flux (W/m^2)')
plt.title('Solar Minima and Maxima Over Time')
plt.xlim(flux_data['fluxdate'].min(), last_predicted_date)
plt.legend()
plt.grid()
plt.show()
