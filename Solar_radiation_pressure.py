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



filtered_flux_past = filtered_flux[:-244]
filtered_flux_near_future_one_a_day = filtered_flux[-243:-198]
filtered_flux_far_future_monthly = filtered_flux[-197:]
arr = 12*[2411.67566288]
arr2 = np.array(arr)
filtered_flux_near_future_without_april = np.repeat(filtered_flux_near_future_one_a_day,3)
filtered_flux_near_future  = np.concatenate((filtered_flux_near_future_without_april,arr2))

import numpy as np

# Function to calculate the days in each month considering leap years
def days_in_month(year, month):
    # Leap year check
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        # February has 29 days in leap years
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        # February has 28 days in non-leap years
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    return days_in_month[month - 1]

# Generate the array for each day's solar flux value
daily_flux = []
start_year = 2025
start_month = 5

# Loop through each month in the solar flux data
for i in range(len(filtered_flux_far_future_monthly)):
    current_year = start_year + (start_month + i - 1) // 12
    current_month = (start_month + i - 1) % 12 + 1

    # Get the number of days in the current month
    days_current_month = days_in_month(current_year, current_month)
    
    # Add daily flux for the current month
    daily_flux.extend([filtered_flux_far_future_monthly[i]] * days_current_month)

# Convert the result into a numpy array
daily_flux = np.array(daily_flux)
filtered_flux_far_future = np.repeat(daily_flux,3)
# Check the length of the resulting daily flux array

flux = np.concatenate((filtered_flux_past,filtered_flux_near_future,filtered_flux_far_future))
print(len(flux))  # The total number of days should now match the correct number of days
accelarations = []
for value in flux:
    accelaration = 1.3 * value / (2.99*10**8) * 0.22 / 2.8
    accelarations.append(accelaration)
array_accelarations = np.array(accelarations)
