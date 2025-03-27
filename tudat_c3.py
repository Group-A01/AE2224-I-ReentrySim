# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array

from tudatpy.astro.time_conversion import DateTime
from datetime import datetime
import time

import pymsis #wrapper for the nrlmsis fortran code

import urllib.request #for fetching a txt file with live TLE for n3Xt

from alive_progress import alive_bar

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

# def density_f(h, lon, lat, time): #long and lat in deg, h in km, time in datetime64, versions: 0, 2.0, 2.1
#     timedate = np.datetime64("2000-01-01T00:00") + np.timedelta64(int(time), 's')
#     # print(timedate)
#     data = pymsis.calculate(timedate, lon, lat, h/1000, geomagnetic_activity=-1, version=2.1)
#     return data[0,pymsis.Variable.MASS_DENSITY]

# def const_temp(h, lon, lat, time):
#     timedate = np.datetime64("2000-01-01T00:00") + np.timedelta64(time, 's')
#     # print(timedate)
#     data = pymsis.calculate(timedate, lon, lat, h/1000, geomagnetic_activity=-1, version=2.1)
#     return data[0,pymsis.Variable.TEMPERATURE]

# body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.custom_four_dimensional_constant_temperature(
#     density_f,
#     991.96893,
#     300.0, 
#     1.4) 
body_settings.get( "Earth" ).atmosphere_settings = environment_setup.atmosphere.nrlmsise00()
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
print("1. Simulate until altitude reaches 120 km")
print("2. Specify a custom end date (YYYY-MM-DD)")
choice = input("Enter your choice (1 or 2): ")

# Set simulation start epoch
simulation_start_epoch = DateTime(2008, 4, 28).epoch()

tle_data = ("1 32789U 07021G   08119.60740078 -.00000054  00000-0  00000+0 0  9999",
                "2 32789 098.0082 179.6267 0015321 307.2977 051.0656 14.81417433    68")

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
    limit_value= 200.0e3,  #in meters
    use_as_lower_limit=True,  # Terminate when altitude drops below this value
    terminate_exactly_on_final_condition=False
)

if choice == "1":
    termination_condition = altitude_termination
    print("Simulating until altitude reaches 120 km...")
elif choice == "2":
    while True:
        try:
            end_date_str = input("Enter end date (YYYY-MM-DD): ")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            simulation_end_epoch = DateTime(end_date.year, end_date.month, end_date.day).epoch()
            if simulation_end_epoch <= simulation_start_epoch:
                print("End date must be after start date (2024-03-08). Try again.")
                continue
            time_termination = propagation_setup.propagator.time_termination(simulation_end_epoch)
            termination_condition = propagation_setup.propagator.hybrid_termination(
                [altitude_termination, time_termination], fulfill_single_condition=True)
            print(f"Simulating until {end_date_str} or 120 km altitude, whichever comes first...")
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD (e.g., 2024-12-31).")
else:
    print("Invalid choice. Terminating script.")
    exit()


# Create numerical integrator settings
fixed_step_size = 60.0
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
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
    # Create simulation object and propagate the dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )
    bar()

console_print_settings = propagator_settings.print_settings
console_print_settings.print_state_indices = True
console_print_settings.print_dependent_variable_indices = True
console_print_settings.print_propagation_clock_time = True
console_print_settings.print_termination_reason = True
console_print_settings.print_number_of_function_evaluations = True

# Extract the resulting state and dependent variable history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.propagation_results.dependent_variable_history
dep_vars_array = result2array(dep_vars)

# Extract data
time_hours = (dep_vars_array[:, 0] - dep_vars_array[0, 0]) / 3600
periapsis = dep_vars_array[:, 16] / 1000  # Convert to km
apoapsis = dep_vars_array[:, 17] / 1000   # Convert to km

# Smooth the data using a moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 50  # Adjust this value based on how much smoothing you want
periapsis_smooth = moving_average(periapsis, window_size)
apoapsis_smooth = moving_average(apoapsis, window_size)
time_smooth = moving_average(time_hours, window_size)

# Create figure with improved styling
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot smoothed periapsis and apoapsis
ax1.plot(time_smooth, periapsis_smooth, 'b-', label='Periapsis Altitude (Smoothed)', linewidth=2)
ax1.plot(time_smooth, apoapsis_smooth, 'r-', label='Apoapsis Altitude (Smoothed)', linewidth=2)
# Plot original data faintly for comparison
ax1.plot(time_hours, periapsis, 'b-', alpha=0.2, label='Periapsis (Raw)', linewidth=1)
ax1.plot(time_hours, apoapsis, 'r-', alpha=0.2, label='Apoapsis (Raw)', linewidth=1)

ax1.set_xlabel('Time [hours]', fontsize=12)
ax1.set_ylabel('Altitude [km]', fontsize=12)
ax1.set_title(f'Apoapsis and Periapsis Altitudes of {satname} Over Time', fontsize=14, pad=20)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xlim([min(time_hours), max(time_hours)])

# Adjust y-axis limits for better visibility
ax1.set_ylim([min(min(periapsis), min(apoapsis)) * 0.95, max(max(periapsis), max(apoapsis)) * 1.05])

# Add some styling
plt.tight_layout()

plt.figure(figsize=(9, 5))
# Point Mass Gravity Acceleration Sun
acceleration_norm_pm_sun = dep_vars_array[:, 10]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Sun')

# Point Mass Gravity Acceleration Moon
acceleration_norm_pm_moon = dep_vars_array[:, 11]
plt.plot(time_hours, acceleration_norm_pm_moon, label='PM Moon')

# Spherical Harmonic Gravity Acceleration Earth
acceleration_norm_sh_earth = dep_vars_array[:, 12]
plt.plot(time_hours, acceleration_norm_sh_earth, label='SH Earth')

# Aerodynamic Acceleration Earth
acceleration_norm_aero_earth = dep_vars_array[:, 13]
plt.plot(time_hours, acceleration_norm_aero_earth, label='Aerodynamic Earth')

# Cannonball Radiation Pressure Acceleration Sun
acceleration_norm_rp_sun = dep_vars_array[:, 14]
plt.plot(time_hours, acceleration_norm_rp_sun, label='Radiation Pressure Sun')

plt.xlim([min(time_hours), max(time_hours)])
plt.xlabel('Time [hr]')
plt.ylabel('Acceleration Norm [m/s$^2$]')
plt.legend(bbox_to_anchor=(1.005, 1))
plt.yscale('log')
plt.grid()
plt.tight_layout()

plt.show()

# Cannonball Radiation Pressure Acceleration Sun
acceleration_norm_rp_sun = dep_vars_array[:, 14]

# Store all extracted variables in an np array
data = np.vstack([time_hours, periapsis, apoapsis, acceleration_norm_pm_sun, acceleration_norm_pm_moon, 
                  acceleration_norm_sh_earth, acceleration_norm_aero_earth, acceleration_norm_rp_sun])
data = np.transpose(data)
headr = "Time (Hours), Periapsis, Apoapsis Altitude, Eccentricity, Inclination" \
        "Acceleration Norm PM Sun, Acceleration Norm PM Moon " \
        "Acceleration Norm SH Earth, Acceleration Norm Aero Earth, Acceleration Norm RP Sun"

# Store the data array in a csv with header
print("Writing to file: {0}.csv...".format(satname))
np.savetxt(satname + ".csv", data, header=headr, delimiter=',')
print("Done!")
print(f"Final simulation time: {time_hours[-1]:.2f} hours")

# 3D Dynamic Visualization with Full Orbit
# Extract Cartesian coordinates from the state history
time = states_array[:, 0]  # Time in seconds
x = states_array[:, 1] / 1e3  # Convert to km
y = states_array[:, 2] / 1e3
z = states_array[:, 3] / 1e3

# Subsample for animation (e.g., 1000 frames for a lightweight video)
step = max(1, len(time) // 1000)  # Aim for ~1000 frames
frame_indices = np.arange(0, len(time), step)
x_sub = x[frame_indices]
y_sub = y[frame_indices]
z_sub = z[frame_indices]
time_sub = time[frame_indices]

print(f"Original frames: {len(time)}, Subsampled frames for animation: {len(frame_indices)}")

# Create a 3D figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth's surface (approximated as a sphere)
u = np.linspace(0, 2 * np.pi, 50)  # Reduced resolution for faster rendering
v = np.linspace(0, np.pi, 50)
earth_radius = 6371  # Earth's radius in km
x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3)

# Plot the full orbit trace statically (using all points for detail)
ax.plot(x, y, z, 'g-', linewidth=1, label='Full Orbit Trace')

# Initialize the satellite's position (animated part)
sat_point, = ax.plot([x_sub[0]], [y_sub[0]], [z_sub[0]], 'ro', label=satname, markersize=5)

# Set plot limits
ax.set_xlim([-10000, 10000])
ax.set_ylim([-10000, 10000])
ax.set_zlim([-10000, 10000])
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.set_title('Dynamic 3D Orbit of {0}'.format(satname))
ax.legend()

plt.show()
while True:
    try:
        inp = input("Save animation? (y/n): ")
        if(inp=="y"):
            # Animation initialization function
            def init():
                sat_point.set_data_3d([x_sub[0]], [y_sub[0]], [z_sub[0]])
                return sat_point,

            # Animation update function (only moves the satellite)
            def update(frame):
                sat_point.set_data_3d([x_sub[frame]], [y_sub[frame]], [z_sub[frame]])
                return sat_point,

            # Create animation
            ani = FuncAnimation(fig, update, frames=len(frame_indices), init_func=init, blit=False, interval=50)

            # Save the animation as a lightweight MP4 file
            print("Saving animation to 'delfi_n3xt_orbit.mp4'...")
            ani.save('delfi_n3xt_orbit.mp4', writer='ffmpeg', fps=30, dpi=80, bitrate=2000)  # Lower DPI and set bitrate
            print("Animation saved!")
        elif(inp=='n'):
            break
        else:
            raise Exception
    except:
        print("Inlvaid input")