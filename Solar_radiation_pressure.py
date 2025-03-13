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

# Load SW-Last5Years data (assuming you already have this loaded)
sw_file_path = "SW-Last5Years.txt"
sw_data = pd.read_csv(sw_file_path, sep='\s+', comment='#', skiprows=18, header=None, engine='python')

# Trim excess columns if necessary (based on the known format)
sw_data = sw_data.iloc[:, :34]  # Keep the first 30 columns

# Remove potential footer rows by checking if the 'year' column (index 0) is numeric
sw_data = sw_data[pd.to_numeric(sw_data.iloc[:, 0], errors='coerce').notna()]

# Clean the 'day' column (index 2) by converting to integers and ensuring proper formatting
sw_data.iloc[:, 2] = sw_data.iloc[:, 2].astype(str).str.split('.').str[0]  # Remove any decimals
sw_data.iloc[:, 2] = sw_data.iloc[:, 2].astype(int)  # Explicitly cast the day column to integers

# Ensure proper formatting with two digits (e.g., "02" instead of "2")
sw_data.iloc[:, 2] = sw_data.iloc[:, 2].astype(str).str.zfill(2)  # Now ensure two digits

# Convert the date columns (year, month, day) to datetime format
sw_data['date'] = pd.to_datetime(
    sw_data.iloc[:, 0].astype(str) + '-' +  # year column (index 0)
    sw_data.iloc[:, 1].astype(str).str.zfill(2) + '-' +  # month column (index 1)
    sw_data.iloc[:, 2]  # day column (index 2), now properly formatted
)

# Set fixed date for today and last predicted date
today = datetime(2025, 3, 13)
last_predicted_date = datetime(2041, 10, 1)

# Filter data from today until last predicted date
sw_data = sw_data[(sw_data['date'] >= today) & (sw_data['date'] <= last_predicted_date)]

# Convert column at index 30 (index 29) to numeric and fill NaN values with 69.8
sw_data.iloc[:, 30] = pd.to_numeric(sw_data.iloc[:, 30], errors='coerce')
sw_data.iloc[:, 30].fillna(69.8, inplace=True)

# Compute the scaled observations using a predefined average_flux

sw_data['scaled_obs'] = (sw_data.iloc[:, 30] / average_flux) * 1360

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
