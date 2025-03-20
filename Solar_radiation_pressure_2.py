import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Load the dataset
file_path = 'Space.weather.txt'
columns = ['Year', 'Month', 'Day', 'BSRN', 'ND', 'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8', 'Sum_Kp', 'Ap1', 'Ap2', 'Ap3', 'Ap4', 'Ap5', 'Ap6', 'Ap7', 'Ap8', 'Avg_Ap', 'Cp', 'C9', 'ISN', 'F10.7']

# Identify the correct starting row
with open(file_path, 'r') as file:
    lines = file.readlines()

start_idx = next(i for i, line in enumerate(lines) if line.strip() and line[0].isdigit())

# Read the data, skipping metadata rows
data = pd.read_csv(file_path, sep='\s+', names=columns, skiprows=start_idx, engine='python', on_bad_lines='skip')

# Create a date column
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors='coerce')
data = data.dropna(subset=['Date'])  # Remove rows with invalid dates
data = data.sort_values(by='Date')

# Select relevant features and target variable
features = ['Year', 'Month', 'Day', 'Sum_Kp', 'Avg_Ap', 'Cp', 'ISN', 'F10.7']
target = 'F10.7'

# Handle missing values
data = data.dropna()

# Check if data is empty
if data.empty:
    raise ValueError("Error: Processed dataset is empty. Check data loading and preprocessing steps.")

# Split into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Determine the last valid date for prediction start
date_start = data['Date'].max()
if pd.isna(date_start):
    print("Warning: No valid dates found in dataset. Using today's date as fallback.")
    date_start = datetime.today()
print("First few rows of data:")
print(data.head())

print("Max date in dataset:", data['Date'].max())

# Generate future dates
date_range = pd.date_range(start=date_start + timedelta(days=1), end='2040-12-31')
future_dates = pd.DataFrame({'Date': date_range})
future_dates['Year'] = future_dates['Date'].dt.year
future_dates['Month'] = future_dates['Date'].dt.month
future_dates['Day'] = future_dates['Date'].dt.day

# Ensure future_dates has required columns
missing_cols = [col for col in features if col not in future_dates.columns]
for col in missing_cols:
    future_dates[col] = data[col].mean() if col in data.columns else 0  # Default to 0 if missing

# Check if future dates data is valid
if future_dates.empty:
    raise ValueError("Error: No valid future dates were generated.")

# Scale and predict
X_future = scaler.transform(future_dates[features])
future_dates['Predicted_F10.7'] = model.predict(X_future)

# Save predictions
future_dates.to_csv('/mnt/data/predicted_space_weather.csv', index=False)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['F10.7'], label='Historical F10.7', color='blue')
plt.plot(future_dates['Date'], future_dates['Predicted_F10.7'], label='Predicted F10.7', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('F10.7 Value')
plt.title('F10.7 Historical and Predicted Values')
plt.legend()
plt.show()