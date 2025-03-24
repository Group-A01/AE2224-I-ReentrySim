""" import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Load the dataset
file_path = 'Space.weather.txt'
columns = ['Year', 'Month', 'Day', 'BSRN', 'ND', 'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8', 'Sum_Kp', 'Ap1', 'Ap2', 'Ap3', 'Ap4', 'Ap5', 'Ap6', 'Ap7', 'Ap8', 'Avg_Ap', 'Cp', 'C9', 'ISN', 'F10.7_1', 'Q1', 'Ctr81_1', 'Lst81_1', 'F10.7_2', 'Ctr81_2', 'Lst81_2']

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

# Feature Engineering
data['Day_of_Year'] = data['Date'].dt.dayofyear  # Capture seasonal effects
data['Sin_Year'] = np.sin(2 * np.pi * data['Day_of_Year'] / 365)
data['Cos_Year'] = np.cos(2 * np.pi * data['Day_of_Year'] / 365)

# Select relevant features and target variables
features = ['Year', 'Month', 'Day', 'Day_of_Year', 'Sin_Year', 'Cos_Year', 'Sum_Kp', 'Avg_Ap', 'Cp', 'ISN']
targets = ['F10.7_1', 'F10.7_2']

# Handle missing values with interpolation
data = data.interpolate()

# Check if data is empty
if data.empty:
    raise ValueError("Error: Processed dataset is empty. Check data loading and preprocessing steps.")

# Split into training and testing sets
X = data[features]
y = data[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train polynomial regression model
poly_degree = 3 # Reduced degree to avoid overfitting
poly_model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
poly_model.fit(X_train_scaled, y_train)

# Generate future dates
date_start = data['Date'].max()
if pd.isna(date_start):
    print("Warning: No valid dates found in dataset. Using today's date as fallback.")
    date_start = datetime.today()

date_range = pd.date_range(start=date_start + timedelta(days=1), end='2040-12-31')
future_dates = pd.DataFrame({'Date': date_range})
future_dates['Year'] = future_dates['Date'].dt.year
future_dates['Month'] = future_dates['Date'].dt.month
future_dates['Day'] = future_dates['Date'].dt.day
future_dates['Day_of_Year'] = future_dates['Date'].dt.dayofyear
future_dates['Sin_Year'] = np.sin(2 * np.pi * future_dates['Day_of_Year'] / 365)
future_dates['Cos_Year'] = np.cos(2 * np.pi * future_dates['Day_of_Year'] / 365)

# Ensure future_dates has required columns
missing_cols = [col for col in features if col not in future_dates.columns]
for col in missing_cols:
    future_dates[col] = data[col].mean() if col in data.columns else 0  # Default to 0 if missing

# Check if future dates data is valid
if future_dates.empty:
    raise ValueError("Error: No valid future dates were generated.")

# Scale and predict
X_future = scaler.transform(future_dates[features])
future_predictions = poly_model.predict(X_future)

# Store predictions in dataframe
future_dates['Predicted_F10.7_1'] = future_predictions[:, 0]
future_dates['Predicted_F10.7_2'] = future_predictions[:, 1]

# Save predictions
future_dates.to_csv('predicted_F10.7_values.csv', index=False)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['F10.7_1'], label='Historical F10.7_1', color='blue')
plt.plot(future_dates['Date'], future_dates['Predicted_F10.7_1'], label='Predicted F10.7_1', color='red', linestyle='dashed')
plt.plot(data['Date'], data['F10.7_2'], label='Historical F10.7_2', color='green')
plt.plot(future_dates['Date'], future_dates['Predicted_F10.7_2'], label='Predicted F10.7_2', color='orange', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('F10.7 Value')
plt.title('F10.7 Historical and Predicted Values (Polynomial Regression)')
plt.legend()
plt.show()
 """
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
