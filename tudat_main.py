import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime
import time
import pymsis
import urllib.request
from alive_progress import alive_bar
import os

# Create output directories if they don't exist
os.makedirs('results/C3', exist_ok=True)
os.makedirs('results/n3xt', exist_ok=True)

# Load spice kernels
spice.load_standard_kernels()

# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Earth", "Moon"]

satname = "Delfi-C3"

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

const_temp = 1000  # Realistic thermospheric temperature in K

def density_f(h, lon, lat, time):
    # Time is seconds since simulation start (2021-11-13)
    start_date = np.datetime64("2000-01-01T00:00")
    timedate = start_date + np.timedelta64(int(time), 's')
    # Use h in kilometers (pymsis expects km)
    data = pymsis.calculate(timedate, lon, lat, h, geomagnetic_activity=-1, version=2.1)
    density = data[0, pymsis.Variable.MASS_DENSITY]
    # Optional: Log for debugging (uncomment to verify density)
    # print(f"Altitude: {h} km, Lon: {lon} deg, Lat: {lat} deg, Time: {timedate}, Density: {density} kg/m^3")
    return density

body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.custom_four_dimensional_constant_temperature(
    density_f,
    const_temp,
    8.314 / 0.016,  # Scale height in km (typical for thermosphere)
    1.667)  # R/M for atomic oxygen (~519 J/(kg·K))
# Create empty body settings for the satellite
body_settings.add_empty_settings(satname)

# Create aerodynamic coefficient interface settings
reference_area_drag = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area_drag, [drag_coefficient, 0.0, 0.0]
)

# Add the aerodynamic interface to the body settings
body_settings.get(satname).aerodynamic_coefficient_settings = aero_coefficient_settings

# Create radiation pressure settings
reference_area_radiation = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
radiation_pressure_coefficient = 1.2
occulting_bodies_dict = dict()
occulting_bodies_dict["Sun"] = ["Earth"]
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict)

# Add the radiation pressure interface to the body settings
body_settings.get(satname).radiation_pressure_target_settings = vehicle_target_settings

bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get(satname).mass = 2.2  # kg

# Define bodies that are propagated
bodies_to_propagate = [satname]

# Define central bodies of propagation
central_bodies = ["Earth"]

# Define accelerations acting on satellite by Sun and Earth.
accelerations_settings_delfi_c3 = dict(
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Earth=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
        propagation_setup.acceleration.aerodynamic()
    ],
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)

# Create global accelerations settings dictionary.
acceleration_settings = {satname: accelerations_settings_delfi_c3}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

# User choice for termination condition
print("Simulation Termination Options:")
print("1. Simulate until altitude reaches 200 km, from 13 November 2021")
print("2. Specify a custom end date (YYYY-MM-DD), from 13 November 2021")
choice = input("Enter your choice (1 or 2): ")

# Set simulation start epoch
year, month, day = 2021, 11, 13
simulation_start_epoch = DateTime(year, month, day).epoch()

tle_data = ("1 32789U 08021G   21317.57983842  .00002962  00000-0  18660-3 0  9993",
            "2 32789  97.3635 347.1658 0011139 357.7651   2.3527 15.09855247739326")

delfi_tle = environment.Tle(tle_data[0], tle_data[1])
delfi_ephemeris = environment.TleEphemeris("Earth", "J2000", delfi_tle, False)
initial_state = delfi_ephemeris.cartesian_state(simulation_start_epoch)

# Define list of dependent variables to save
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration(satname),
    propagation_setup.dependent_variable.keplerian_state(satname, "Earth"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, satname, "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, satname, "Moon"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, satname, "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.aerodynamic_type, satname, "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.radiation_pressure_type, satname, "Sun"
    ),
    propagation_setup.dependent_variable.altitude(satname, "Earth"),
    propagation_setup.dependent_variable.periapsis_altitude(satname, "Earth"),
    propagation_setup.dependent_variable.apoapsis_altitude(satname, "Earth")
]

# Create termination settings based on altitude (terminate when altitude <= 120 km)
altitude_variable = propagation_setup.dependent_variable.altitude(satname, "Earth")
altitude_termination = propagation_setup.propagator.dependent_variable_termination(
    dependent_variable_settings=altitude_variable,
    limit_value=200.0e3,  # 200 km in meters
    use_as_lower_limit=True,
    terminate_exactly_on_final_condition=False
)

if choice == "1":
    termination_condition = altitude_termination
    print("Simulating until altitude reaches 200 km...")
elif choice == "2":
    while True:
        try:
            end_date_str = input("Enter end date (YYYY-MM-DD): ")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            simulation_end_epoch = DateTime(end_date.year, end_date.month, end_date.day).epoch()
            if simulation_end_epoch <= simulation_start_epoch:
                print("End date must be after start date (2021-11-13). Try again.")
                continue
            time_termination = propagation_setup.propagator.time_termination(simulation_end_epoch)
            termination_condition = propagation_setup.propagator.hybrid_termination(
                [altitude_termination, time_termination], fulfill_single_condition=True)
            print(f"Simulating until {end_date_str} or 200 km altitude, whichever comes first...")
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD (e.g., 2024-12-31).")
else:
    print("Invalid choice. Terminating script.")
    exit()

# Create numerical integrator settings
control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-10, 1.0E-10)
validation_settings = propagation_setup.integrator.step_size_validation(0.001, 1000.0)
fixed_step_size = 60.0
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
    initial_time_step=fixed_step_size,
    coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
    step_size_control_settings=control_settings,
    step_size_validation_settings=validation_settings
)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_condition,
    output_variables=dependent_variables_to_save
)

print("OKKK off we go")
with alive_bar(title="Numerical integration:") as bar:
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )
    bar()

# Extract the resulting state and dependent variable history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.propagation_results.dependent_variable_history
dep_vars_array = result2array(dep_vars)

# Extract data
time_hours = (dep_vars_array[:, 0] - dep_vars_array[0, 0]) / 3600
periapsis = dep_vars_array[:, 16] / 1000  # Convert to km
apoapsis = dep_vars_array[:, 17] / 1000   # Convert to km
acceleration_norm_pm_sun = dep_vars_array[:, 10]
acceleration_norm_pm_moon = dep_vars_array[:, 11]
acceleration_norm_sh_earth = dep_vars_array[:, 12]
acceleration_norm_aero_earth = dep_vars_array[:, 13]
acceleration_norm_rp_sun = dep_vars_array[:, 14]

# Smooth the data using a moving average (excluding aerodynamic acceleration)
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 10  # Increased for smoother lines
periapsis_smooth = moving_average(periapsis, window_size)
apoapsis_smooth = moving_average(apoapsis, window_size)
time_smooth = moving_average(time_hours, window_size)

# Calculate the corresponding time indices for the smoothed data
smoothed_length = len(periapsis_smooth)  # Length of smoothed arrays
start_idx = (window_size - 1) // 2  # Center the smoothing window
smooth_indices = np.arange(start_idx, start_idx + smoothed_length)

# Extract the corresponding time seconds for the smoothed data
time_seconds = dep_vars_array[:, 0] - dep_vars_array[0, 0]  # Time in seconds from start
time_seconds_smooth = time_seconds[smooth_indices]

# Convert smoothed time to dates
start_date = datetime(year, month, day)  # Simulation start date
dates_smooth = np.array([timedelta(seconds=time) + start_date for time in time_seconds_smooth])

# Convert smoothed time to years
time_years_smooth = time_seconds_smooth / (365.25 * 24 * 3600) + year 

# Slice original data to match smoothed data length for plotting
acceleration_norm_aero_earth_sliced = acceleration_norm_aero_earth[smooth_indices]  # Use unsmoothed data

# Calculate final simulation date
final_time_seconds = time_seconds[-1]
final_date = start_date + timedelta(seconds=final_time_seconds)

# Create figure with improved styling for altitude plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot smoothed periapsis and apoapsis with thin, smooth lines
ax1.plot(dates_smooth, periapsis, 'b-', label='Periapsis Altitude', linewidth=1)
ax1.plot(dates_smooth, apoapsis, 'r-', label='Apoapsis Altitude', linewidth=1)

ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Altitude [km]', fontsize=12)
ax1.set_title(f'Apoapsis and Periapsis Altitudes of {satname} Over Time', fontsize=14, pad=20)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xlim([min(dates_smooth), max(dates_smooth)])

# Adjust y-axis limits for better visibility
ax1.set_ylim([min(min(periapsis_smooth), min(apoapsis_smooth)) * 0.95, max(max(periapsis_smooth), max(apoapsis_smooth)) * 1.05])

# Add some styling
plt.tight_layout()
plt.savefig('results/C3/C3_altitude.png')

# Create figure for aerodynamic acceleration (unsmoothed) with thin, smooth line
plt.figure(figsize=(9, 5))
plt.plot(dates_smooth, acceleration_norm_aero_earth_sliced, 'g-', label='Aerodynamic Earth Acceleration', linewidth=1)
plt.xlim([min(dates_smooth), max(dates_smooth)])
plt.xlabel('Date', fontsize=12)
plt.ylabel('Acceleration Norm [m/s$^2$]', fontsize=12)
plt.title(f'Aerodynamic Acceleration of {satname} Over Time', fontsize=14, pad=20)
plt.legend(loc='upper right', fontsize=10)
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/C3/C3_drag_acceleration.png')

plt.show()

# Store all extracted variables in an np array (using original full-length data)
data = np.vstack([dates_smooth, apoapsis, periapsis, acceleration_norm_aero_earth])
data = np.transpose(data)
headr = "Time (Hours), Apoapsis, Periapsis, Acceleration Norm Aero Earth"

# Store the data array in a csv with header
print("Writing to file: {0}.csv...".format(satname))
np.savetxt("results/C3/"+satname + ".csv", data, header=headr, delimiter=',')
print("Done!")
print(f"Final simulation time: {time_hours[-1]:.2f} hours")
print(f"Final simulation date: {final_date.strftime('%Y-%m-%d')}")