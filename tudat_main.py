# Load standard modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta  # CHANGED: Added timedelta for date calculations

# Load tudatpy modules
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

from C3_TLE_data_extraction import TLE_extract

# Load spice kernels
spice.load_standard_kernels()

# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus"]

# satname = "Delfi-n3xt"
satname = "Delfi-C3"

#actual data import
actual_periapsis, actual_apoapsis, actual_dates = TLE_extract()

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

const_temp=999

def density_f(h, lon, lat, time): #long and lat in deg, h in km, time in datetime64, versions: 0, 2.0, 2.1
    timedate = np.datetime64("2000-01-01T00:00") + np.timedelta64(int(time), 's')
    # print(timedate)
    data = pymsis.calculate(timedate, lon, lat, h/1000, geomagnetic_activity=-1, version=2.1)
    global const_temp
    const_temp = data[0, pymsis.Variable.MASS_DENSITY]
    return data[0,pymsis.Variable.MASS_DENSITY]

# def const_temp(h, lon, lat, time):
#     timedate = np.datetime64("2000-01-01T00:00") + np.timedelta64(time, 's')
#     print(timedate)
#     data = pymsis.calculate(timedate, lon, lat, h, geomagnetic_activity=-1, version=2.1)
#     return data[0,pymsis.Variable.TEMPERATURE]

body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.custom_four_dimensional_constant_temperature(
    density_f,
    # 991.96893,
    const_temp,
    300.0, 
    1.4) 
#  const. temp, sp. gas const, ratio of sp. heats

# Create empty body settings for the satellite
body_settings.add_empty_settings(satname)

# # Create aerodynamic coefficient interface settings
# reference_area_drag = (4*0.3404*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
# drag_coefficient = 1.2
# aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
#     reference_area_drag, [drag_coefficient, 0.0, 0.0]
# )

# # Add the aerodynamic interface to the body settings
# body_settings.get(satname).aerodynamic_coefficient_settings = aero_coefficient_settings

# # Create radiation pressure settings
# reference_area_radiation = (4*0.3404*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
# radiation_pressure_coefficient = 1.2
# occulting_bodies_dict = dict()
# occulting_bodies_dict["Sun"] = ["Earth"]
# vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
#     reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict)

# Create aerodynamic coefficient interface settings
reference_area_drag = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
drag_coefficient = 2 #coeffs closer to 2 might be more representative
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area_drag, [drag_coefficient, 0.0, 0.0]
)
# def aerodynamic_force_coefficients(time):
#     reference_area = 0.1*np.sin(2*np.pi/4 *time)
#     drag_coefficient = 2 * reference_area  # Example calculation
#     return [drag_coefficient, 0.0, 0.0] # [CD, CY, CL]

# aero_coefficient_settings = environment_setup.aerodynamic_coefficients.custom_aerodynamic_force_coefficients(
#     force_coefficient_function = aerodynamic_force_coefficients,
#     independent_variable_names=["time"]
# )

# Add the aerodynamic interface to the body settings
body_settings.get(satname).aerodynamic_coefficient_settings = aero_coefficient_settings

# Create radiation pressure settings
# reference_area_radiation = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
reference_area_radiation = 0.085 # [m] Average projection area of a 3U CubeSat
radiation_pressure_coefficient = 1.2
occulting_bodies_dict = dict()
occulting_bodies_dict["Sun"] = ["Earth"]
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict)

# Add the radiation pressure interface to the body settings
body_settings.get(satname).radiation_pressure_target_settings = vehicle_target_settings

bodies = environment_setup.create_system_of_bodies(body_settings)
# bodies.get(satname).mass = 3.5  # kg
bodies.get(satname).mass = 2.2  # kg

# Define bodies that are propagated
bodies_to_propagate = [satname]

# Define central bodies of propagation
central_bodies = ["Earth"]

# Define accelerations acting on satellite by Sun and Earth.
accelerations_settings_delfi_n3xt = dict(
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Earth=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5), #J terms here
        propagation_setup.acceleration.aerodynamic()
    ],
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Mars=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Venus=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)

# Create global accelerations settings dictionary.
acceleration_settings = {satname: accelerations_settings_delfi_n3xt}

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

# Create termination settings based on altitude (terminate when altitude <= 120 km)
altitude_variable = propagation_setup.dependent_variable.altitude(satname, "Earth")
altitude_termination = propagation_setup.propagator.dependent_variable_termination(
    dependent_variable_settings=altitude_variable,
    limit_value= 120.0e3,  #in meters
    use_as_lower_limit=True,  # Terminate when altitude drops below this value
    terminate_exactly_on_final_condition=False
)

# Create numerical integrator settings
control_settings = propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance( 1.0E-10, 1.0E-10 )
validation_settings = propagation_setup.integrator.step_size_validation( 0.001, 1000.0 )
fixed_step_size = 60.0
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
    initial_time_step=fixed_step_size, 
    coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
    step_size_control_settings = control_settings,
    step_size_validation_settings = validation_settings 
)

# print("Start date: {0}-{1}-{2}".format(year, month, day))
start_date_str = input("Enter start date (YYYY-MM-DD): ")
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
simulation_start_epoch = DateTime(start_date.year, start_date.month, start_date.day).epoch()

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
                print("End date must be after start date. Try again.")
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

# # Set simulation start epoch
# year, month, day = 2024, 4, 21
# # year, month, day = 2018, 3, 16
# start_date = DateTime(year, month, day)  # Simulation start date
# Retrieve the initial state of satellite using Two-Line-Elements (TLEs) (n3xt = 39428U)
targeturl = "https://celestrak.org/NORAD/elements/gp.php?GROUP=cubesat&FORMAT=tle"
tle_data = ""
inp = input("Find current data or use prebaked? (1,2): ")
if inp == "1":
    with urllib.request.urlopen(targeturl) as response:
        data = response.read().decode('utf-8')
        lines = data.splitlines()
        for i in range(len(lines)):
            slice = lines[i][0:8]
            if slice == "1 39428U":
                tle_data = (lines[i], lines[i+1])
                break
    print("Data as of {0}: {1}".format(datetime.today(), tle_data))
elif inp == "2":
    tle_data = (
        "1 32789U 08021G   21317.57983842  .00002962  00000-0  18660-3 0  9993",
        "2 32789  97.3635 347.1658 0011139 357.7651   2.3527 15.09855247739326"
    )

    "1 39428U 13066N   22270.40190000  .00000000  00000-0  00000-0 0  9999",
    "2 39428  97.8162 312.5019 0115520   0.0000 200.4051 14.8500000 000017" 

    # tle_data = (
    # "1 39428U 13066N   18075.32153000  .00000240  00000-0  21562-4 0  1232",
    # "2 39428  97.6502   4.9842 0120600 207.1048 170.7736 14.67040000 00019"
    # )   
    #from https://in-the-sky.org/spacecraft_elements.php?id=39428&startday=24&startmonth=2&startyear=2016&endday=24&endmonth=3&endyear=2018
    print("Data as of {0}: {1}".format(np.datetime64('today'), tle_data))

delfi_tle = environment.Tle(tle_data[0], tle_data[1])
delfi_ephemeris = environment.TleEphemeris("Earth", "J2000", delfi_tle, False)
initial_state = delfi_ephemeris.cartesian_state(simulation_start_epoch)


# Define list of dependent variables to save
dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration(satname),
    propagation_setup.dependent_variable.keplerian_state(satname, "Earth"),
    propagation_setup.dependent_variable.latitude(satname, "Earth"),
    propagation_setup.dependent_variable.longitude(satname, "Earth"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, satname, "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, satname, "Moon"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, satname, "Mars"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, satname, "Venus"
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

# Extract the resulting state and dependent variable history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.propagation_results.dependent_variable_history
dep_vars_array = result2array(dep_vars)
print(dep_vars_array.shape)
# Convert time to years and calculate final date
start_date = datetime(start_date.year, start_date.month, start_date.day)  # Simulation start date
time_seconds = dep_vars_array[:, 0] - dep_vars_array[0, 0]  # Time in seconds from start
dates = np.array([timedelta(seconds=time) + start_date for time in time_seconds])
# print("deltas {0}".format(dates))
time_years = time_seconds / (365.25 * 24 * 3600) + start_date.year  # Convert to years, starting at 2008
altitude = dep_vars_array[:, 19] / 1000  # Altitude in km
periapsis = dep_vars_array[:,20] /1000 #Periapsis in km
apoapsis = dep_vars_array[:,21] /1000 #Periapsis in km
average_alt = (periapsis + apoapsis) * 0.5

# Calculate final simulation date
final_time_seconds = time_seconds[-1]
final_date = start_date + timedelta(seconds=final_time_seconds)

#import data to compare with
actual_data = pd.read_csv("actual_orb_data/n3xt_actualdata.csv")
act_dates = pd.to_datetime(actual_data.iloc[:, 0])
act_vals = actual_data.iloc[:, 1:]

# # Plot altitude vs. time in years
# plt.figure(figsize=(9, 5))
# plt.title("Altitude of {0} over time".format(satname))
# plt.plot(dates, periapsis, label="Periapsis")
# plt.plot(dates, apoapsis, label="Apoapsis")
# plt.plot(dates, average_alt, label="Av.Altitude")
# plt.plot(act_dates, act_vals.iloc[:,3], label="Historical Data")
# plt.xlabel("Time [years]")
# plt.ylabel("Altitude [km]")
# plt.xlim([min(dates), max(dates)])
# plt.ylim([0, 800])
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

time_hours = time_seconds / 3600  # Still in hours for other plots
total_acceleration_norm = np.linalg.norm(dep_vars_array[:, 1:4], axis=1)

latitude = dep_vars_array[:, 10]
longitude = dep_vars_array[:, 11]
hours = 3
subset = int(len(dates) / 24 * hours)
latitude = np.rad2deg(latitude[0:subset])
longitude = np.rad2deg(longitude[0:subset])
colors = np.linspace(0, 100, len(latitude))

# Smooth the data using a moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_size = 50  # Adjust this value based on how much smoothing you want
periapsis_smooth = moving_average(periapsis, window_size)
apoapsis_smooth = moving_average(apoapsis, window_size)
time_smooth = moving_average(time_hours, window_size)

kepler_elements = dep_vars_array[:, 4:10]

semi_major_axis = kepler_elements[:, 0] / 1e3
eccentricity = kepler_elements[:, 1]
inclination = np.rad2deg(kepler_elements[:, 2])
argument_of_periapsis = np.rad2deg(kepler_elements[:, 3])
raan = np.rad2deg(kepler_elements[:, 4])
true_anomaly = np.rad2deg(kepler_elements[:, 5])

# Point Mass Gravity Acceleration Sun
acceleration_norm_pm_sun = dep_vars_array[:, 12]
# Point Mass Gravity Acceleration Moon
acceleration_norm_pm_moon = dep_vars_array[:, 13]
# Point Mass Gravity Acceleration Mars
acceleration_norm_pm_mars = dep_vars_array[:, 14]
# Point Mass Gravity Acceleration Venus
acceleration_norm_pm_venus = dep_vars_array[:, 15]
# Spherical Harmonic Gravity Acceleration Earth
acceleration_norm_sh_earth = dep_vars_array[:, 16]
# Aerodynamic Acceleration Earth
acceleration_norm_aero_earth = dep_vars_array[:, 17]
# Cannonball Radiation Pressure Acceleration Sun
acceleration_norm_rp_sun = dep_vars_array[:, 18]

# Store all extracted variables in an np array
data = np.vstack([time_hours, altitude, semi_major_axis, eccentricity, inclination, argument_of_periapsis, raan, true_anomaly,
                  acceleration_norm_pm_sun, acceleration_norm_pm_moon, acceleration_norm_pm_mars, acceleration_norm_pm_venus, 
                  acceleration_norm_sh_earth, acceleration_norm_aero_earth, acceleration_norm_rp_sun])
data = np.transpose(data)
headr = "Time (Hours), Altitude, Semi Major Axis, Eccentricity, Inclination, Argument Of Periapsis, RAAN, True Anomaly, " \
        "Acceleration Norm PM Sun, Acceleration Norm PM Moon, Acceleration Norm PM Mars, Acceleration Norm PM Venus, " \
        "Acceleration Norm SH Earth, Acceleration Norm Aero Earth, Acceleration Norm RP Sun"

# Store the data array in a csv with header
print("Writing to file: {0}.csv...".format(satname))
np.savetxt("results/n3xt/"+satname + ".csv", data, header=headr, delimiter=',')
print("Done!")
print(f"Final simulation time: {time_hours[-1]:.2f} hours")

# plt.show()

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
            with alive_bar(title="Saving... ") as bar:
                ani.save('delfi_n3xt_orbit.mp4', writer='ffmpeg', fps=30, dpi=80, bitrate=2000)  # Lower DPI and set bitrate
                print("Animation saved!")
            break
        elif(inp=='n'):
            break
        else:
            raise Exception
    except:
        print("Inlvaid input")


# Plot total acceleration (keeping hours for consistency with other plots)
plt.figure(figsize=(9, 5))
plt.title("Total acceleration norm on {0} over the course of propagation.".format(satname))
plt.plot(dates, total_acceleration_norm)
plt.xlabel('Date [yyyy-mm-dd]')
plt.ylabel('Total Acceleration [m/s$^2$]')
plt.xlim([min(dates), max(dates)])
plt.ylim(0,10)
plt.grid()
plt.tight_layout()
plt.savefig("results/n3xt/"+satname+"_tot_acceleration")

# Plot ground track for a period of 3 hours
plt.figure(figsize=(9, 5))
plt.title("3 hour ground track of {0}".format(satname))
plt.scatter(longitude, latitude, s=1, c=colors, cmap='viridis')
plt.colorbar()
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xlim([min(longitude), max(longitude)])
plt.yticks(np.arange(-90, 91, step=45))
plt.grid()
plt.tight_layout()
plt.savefig("results/n3xt/"+satname+"_groundtrack")

# Plot Kepler elements as a function of time
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Evolution of Kepler elements over the course of the propagation.')
# Semi-major Axis
ax1.plot(dates, semi_major_axis)
ax1.set_ylabel('Semi-major axis [km]')
ax1.set_ylim(0,10000)
# Eccentricity
ax2.plot(dates, eccentricity)
ax2.set_ylabel('Eccentricity [-]')
# Inclination
ax3.plot(dates, inclination)
ax3.set_ylabel('Inclination [deg]')
# Argument of Periapsis
ax4.plot(dates, argument_of_periapsis)
ax4.set_ylabel('Argument of Periapsis [deg]')
# Right Ascension of the Ascending Node
ax5.plot(dates, raan)
ax5.set_ylabel('RAAN [deg]')
# True Anomaly
ax6.scatter(dates, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))
for ax in fig.get_axes():
    ax.set_xlabel('Date [yyyy-mm-dd]')
    ax.set_xlim([min(dates), max(dates)])
    ax.grid()
plt.tight_layout()
plt.savefig("results/n3xt/"+satname+"_elements")

#plot accelerations
plt.figure(figsize=(9, 5))
plt.plot(dates, acceleration_norm_pm_sun, label='PM Sun')
plt.plot(dates, acceleration_norm_pm_moon, label='PM Moon')
plt.plot(dates, acceleration_norm_pm_mars, label='PM Mars')
plt.plot(dates, acceleration_norm_pm_venus, label='PM Venus')
plt.plot(dates, acceleration_norm_sh_earth, label='SH Earth')
plt.plot(dates, acceleration_norm_aero_earth, label='Aerodynamic Earth')
plt.plot(dates, acceleration_norm_rp_sun, label='Radiation Pressure Sun')
plt.xlim([min(dates), max(dates)])
plt.xlabel('Date [yyyy-mm-dd]')
plt.ylabel('Acceleration Norm [m/s$^2$]')
plt.legend(bbox_to_anchor=(1.005, 1))
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.savefig("results/n3xt/"+satname+"_accelerations")

plt.close("all")

# Create figure with improved styling
fig, axs = plt.subplots(figsize=(12, 6))

# # Plot smoothed periapsis and apoapsis
# axs.plot(time_smooth, periapsis_smooth, 'b-', label='Periapsis Altitude (Smoothed)', linewidth=2)
# axs.plot(time_smooth, apoapsis_smooth, 'r-', label='Apoapsis Altitude (Smoothed)', linewidth=2)
# Plot original data faintly for comparison
axs.plot(dates[::int(86400/fixed_step_size)], periapsis[::int(86400/fixed_step_size)], 'b-', alpha=1, label='Periapsis (Raw)', linewidth=1)
axs.plot(dates[::int(86400/fixed_step_size)], apoapsis[::int(86400/fixed_step_size)], 'r-', alpha=1, label='Apoapsis (Raw)', linewidth=1)
#actual
axs.plot(actual_dates, actual_periapsis, 'b-', alpha=1, label='Periapsis (Raw)', linewidth=1)
axs.plot(actual_dates, actual_apoapsis, 'r-', alpha=1, label='Apoapsis (Raw)', linewidth=1)

axs.set_xlabel('Time [hours]', fontsize=12)
axs.set_ylabel('Altitude [km]', fontsize=12)
axs.set_title(f'Apoapsis and Periapsis Altitudes of {satname} Over Time', fontsize=14, pad=20)
axs.grid(True, linestyle='--', alpha=0.7)
axs.legend(loc='upper left', fontsize=10)
axs.set_xlim([min(dates), max(dates)])

# Adjust y-axis limits for better visibility
axs.set_ylim([min(min(periapsis), min(apoapsis)) * 0.95, max(max(periapsis), max(apoapsis)) * 1.05])

# Add some styling
plt.tight_layout()
plt.savefig("results/n3xt/"+satname+"_altitude_over_time")
plt.show()