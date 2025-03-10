# Load standard modules
import numpy as np
import matplotlib
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

# Load spice kernels
spice.load_standard_kernels()

# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus"]

satname = "Delfi-n3xt"

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Earth"
global_frame_orientation = "J2000"

# Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)


def density_f(h, lon, lat, time): #long and lat in deg, h in km, time in datetime64, versions: 0, 2.0, 2.1
    timedate = np.datetime64("2000-01-01T00:00") + np.timedelta64(int(time), 's')
    # print(timedate)
    data = pymsis.calculate(timedate, lon, lat, h, geomagnetic_activity=-1, version=2.1)
    return data[0,pymsis.Variable.MASS_DENSITY]

# def const_temp(h, lon, lat, time):
#     timedate = np.datetime64("2000-01-01T00:00") + np.timedelta64(time, 's')
#     # print(timedate)
#     data = pymsis.calculate(timedate, lon, lat, h, geomagnetic_activity=-1, version=2.1)
#     return data[0,pymsis.Variable.TEMPERATURE]

body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.custom_four_dimensional_constant_temperature(
    density_f,
    991.96893,
    300.0, 
    1.4) 
 #const. temp, sp. gas const, ratio of sp. heats

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
bodies.get(satname).mass = 2.8  # kg

# Define bodies that are propagated
bodies_to_propagate = [satname]

# Define central bodies of propagation
central_bodies = ["Earth"]

# Define accelerations acting on Delfi-n3xt by Sun and Earth.
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
    ],
    Mars=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Venus=[
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

# Set simulation start epoch
simulation_start_epoch = DateTime(2024, 3, 8).epoch()

# Retrieve the initial state of Delfi-n3xt using Two-Line-Elements (TLEs)
targeturl = "https://celestrak.org/NORAD/elements/gp.php?GROUP=cubesat&FORMAT=tle"
tle_data = ""
with urllib.request.urlopen(targeturl) as response:
    data = response.read().decode('utf-8')
    lines = data.splitlines()
    for i in range(len(lines)):
        slice = lines[i][0:8]
        if slice == "1 39428U":
            tle_data = (lines[i], lines[i+1])
            break
    print("Data as of {0}: {1}".format(datetime.today(), tle_data))

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
    propagation_setup.dependent_variable.altitude(satname, "Earth")
]

# Create termination settings based on altitude (terminate when altitude <= 120 km)
altitude_variable = propagation_setup.dependent_variable.altitude(satname, "Earth")
termination_condition = propagation_setup.propagator.dependent_variable_termination(
    dependent_variable_settings=altitude_variable,
    limit_value= 80.0e3,  #in meters
    use_as_lower_limit=True,  # Terminate when altitude drops below this value
    terminate_exactly_on_final_condition=False
)

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
tik = time.time()
# Create simulation object and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)
tok = time.time()

print("Elapsed time: {0} seconds".format(tok - tik))

# Extract the resulting state and dependent variable history and convert it to an ndarray
states = dynamics_simulator.propagation_results.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.propagation_results.dependent_variable_history
dep_vars_array = result2array(dep_vars)

# Plot total acceleration as function of time
time_hours = (dep_vars_array[:, 0] - dep_vars_array[0, 0]) / 3600
total_acceleration_norm = np.linalg.norm(dep_vars_array[:, 1:4], axis=1)
altitude = dep_vars_array[:, 19] / 1000
plt.figure(figsize=(9, 5))
plt.title("Altitude of Delfi-n3xt over time")
plt.plot(time_hours, altitude)
plt.xlabel("Time [hr]")
plt.ylabel("Altitude [km]")
plt.xlim([min(time_hours), max(time_hours)])
plt.grid()
plt.tight_layout()

plt.figure(figsize=(9, 5))
plt.title("Total acceleration norm on Delfi-n3xt over the course of propagation.")
plt.plot(time_hours, total_acceleration_norm)
plt.xlabel('Time [hr]')
plt.ylabel('Total Acceleration [m/s$^2$]')
plt.xlim([min(time_hours), max(time_hours)])
plt.grid()
plt.tight_layout()

# Plot ground track for a period of 3 hours
latitude = dep_vars_array[:, 10]
longitude = dep_vars_array[:, 11]
hours = 3
subset = int(len(time_hours) / 24 * hours)
latitude = np.rad2deg(latitude[0:subset])
longitude = np.rad2deg(longitude[0:subset])
colors = np.linspace(0, 100, len(latitude))
plt.figure(figsize=(9, 5))
plt.title("3 hour ground track of Delfi-n3xt")
plt.scatter(longitude, latitude, s=1, c=colors, cmap='viridis')
plt.colorbar()
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xlim([min(longitude), max(longitude)])
plt.yticks(np.arange(-90, 91, step=45))
plt.grid()
plt.tight_layout()

# Plot Kepler elements as a function of time
kepler_elements = dep_vars_array[:, 4:10]
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

# Semi-major Axis
semi_major_axis = kepler_elements[:, 0] / 1e3
ax1.plot(time_hours, semi_major_axis)
ax1.set_ylabel('Semi-major axis [km]')

# Eccentricity
eccentricity = kepler_elements[:, 1]
ax2.plot(time_hours, eccentricity)
ax2.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_elements[:, 2])
ax3.plot(time_hours, inclination)
ax3.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(kepler_elements[:, 3])
ax4.plot(time_hours, argument_of_periapsis)
ax4.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_elements[:, 4])
ax5.plot(time_hours, raan)
ax5.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_elements[:, 5])
ax6.scatter(time_hours, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(time_hours), max(time_hours)])
    ax.grid()
plt.tight_layout()

plt.figure(figsize=(9, 5))

# Point Mass Gravity Acceleration Sun
acceleration_norm_pm_sun = dep_vars_array[:, 12]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Sun')

# Point Mass Gravity Acceleration Moon
acceleration_norm_pm_moon = dep_vars_array[:, 13]
plt.plot(time_hours, acceleration_norm_pm_moon, label='PM Moon')

# Point Mass Gravity Acceleration Mars
acceleration_norm_pm_mars = dep_vars_array[:, 14]
plt.plot(time_hours, acceleration_norm_pm_mars, label='PM Mars')

# Point Mass Gravity Acceleration Venus
acceleration_norm_pm_venus = dep_vars_array[:, 15]
plt.plot(time_hours, acceleration_norm_pm_venus, label='PM Venus')

# Spherical Harmonic Gravity Acceleration Earth
acceleration_norm_sh_earth = dep_vars_array[:, 16]
plt.plot(time_hours, acceleration_norm_sh_earth, label='SH Earth')

# Aerodynamic Acceleration Earth
acceleration_norm_aero_earth = dep_vars_array[:, 17]
plt.plot(time_hours, acceleration_norm_aero_earth, label='Aerodynamic Earth')

# Cannonball Radiation Pressure Acceleration Sun
acceleration_norm_rp_sun = dep_vars_array[:, 18]

# Store all extracted variables in an np array
data = np.vstack([time_hours, altitude, semi_major_axis, eccentricity, inclination, argument_of_periapsis, raan, true_anomaly,
                  acceleration_norm_pm_sun, acceleration_norm_pm_moon, acceleration_norm_pm_mars, acceleration_norm_pm_venus, acceleration_norm_sh_earth, acceleration_norm_aero_earth, acceleration_norm_rp_sun])
data = np.transpose(data)
headr = "Time (Hours), Altitude, Semi Major Axis, Eccentricity, Inclination, Argument Of Periapsis, RAAN, True Anomaly, Acceleration Norm PM Sun, Acceleration Norm PM Moon, Acceleration Norm PM Mars, Acceleration Norm PM Venus, Acceleration Norm SH Earth, Acceleration Norm Aero Earth, Acceleration Norm RP Sun"

# Store the data array in a csv with header
print("Writing to file: {0}.csv...".format(satname))
np.savetxt(satname + ".csv", data, header=headr, delimiter=',')
print("Done!")
print(time_hours[-1])

plt.plot(time_hours, acceleration_norm_rp_sun, label='Radiation Pressure Sun')

plt.xlim([min(time_hours), max(time_hours)])
plt.xlabel('Time [hr]')
plt.ylabel('Acceleration Norm [m/s$^2$]')
plt.legend(bbox_to_anchor=(1.005, 1))
plt.suptitle("Accelerations norms on Delfi-n3xt, distinguished by type and origin, over the course of propagation.")
plt.yscale('log')
plt.grid()
plt.tight_layout()
	
# 3D Dynamic Visualization
# Extract Cartesian coordinates from the state history
time = states_array[:, 0]  # Time in seconds
x = states_array[:, 1] / 1e3  # Convert to km
y = states_array[:, 2] / 1e3
z = states_array[:, 3] / 1e3

# Debugging: Check the number of frames
print(f"Number of frames to animate: {len(time)}")
if len(time) == 0:
    raise ValueError("No frames available for animation. Check the state history.")

# Create a 3D figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth's surface (approximated as a sphere)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
earth_radius = 6371  # Earth's radius in km
x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3)

# Initialize the satellite's position and orbit trace
sat_point, = ax.plot([], [], [], 'ro', label='Delfi-n3xt', markersize=5)  # Red dot for satellite
orbit_line, = ax.plot([], [], [], 'g-', linewidth=1)  # Green line for orbit trace

# Set plot limits (adjust based on your orbit)
ax.set_xlim([-10000, 10000])
ax.set_ylim([-10000, 10000])
ax.set_zlim([-10000, 10000])
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')
ax.set_title('Dynamic 3D Orbit of Delfi-n3xt')
ax.legend()

# Animation initialization function
def init():
    sat_point.set_data_3d([], [], [])
    orbit_line.set_data_3d([], [], [])
    return sat_point, orbit_line

# Animation update function
def update(frame):
    sat_point.set_data_3d([x[frame]], [y[frame]], [z[frame]])
    orbit_line.set_data_3d(x[:frame+1], y[:frame+1], z[:frame+1])
    return sat_point, orbit_line

# Create animation (disable blit for saving compatibility)
ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=False, interval=50)

# Save the animation as an MP4 file
print("Saving animation to 'delfi_n3xt_orbit.mp4'...")
ani.save('delfi_n3xt_orbit.mp4', writer='ffmpeg', fps=30, dpi=100)
print("Animation saved!")

# Display all plots (optional, comment out if only saving is needed)
plt.show()