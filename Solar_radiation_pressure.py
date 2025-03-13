import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# Input file name
input_file = 'fulldata.txt'

# Initialize lists to store data
dates = []
flux_values = []

# Read the file
with open(input_file, 'r') as file:
    next(file)  # Skip the header
    for line in file:
        parts = line.strip().split(',')
        if len(parts) == 2:
            date, flux = parts
            dates.append(date)
            flux_values.append(float(flux))

# Convert lists to NumPy arrays
dates = np.array(dates, dtype=str)  # Store dates as strings
flux_values = np.array(flux_values, dtype=float)  # Convert flux values to float

# Compute the average of the flux values
average_flux = np.mean(flux_values)

# Normalize and scale the flux values
scaled_flux = (flux_values / average_flux) * 1360

# Remove outliers using Z-score method
z_scores = stats.zscore(scaled_flux)
filtered_flux = scaled_flux[np.abs(z_scores) < 3]  # Remove values with z-score > 3
filtered_dates = dates[np.abs(z_scores) < 3]  # Corresponding dates for the filtered data

# Convert date strings to datetime objects for proper plotting
date_objects = pd.to_datetime(filtered_dates, format='%Y%m%d')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(date_objects, filtered_flux, label="Scaled Flux (W/m²)", color='b', linestyle='-', marker='o', markersize=3)

# Format the x-axis to show dates properly
plt.xlabel("Date")
plt.ylabel("Solar Flux (W/m²)")
plt.title("Normalized Solar Flux Over Time")
plt.xticks(rotation=45, fontsize=10)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates on x-axis
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Show major ticks for each year
plt.grid(True)
plt.legend()

# Adjust layout to fit labels
plt.tight_layout()

# Show the plot
plt.show()
