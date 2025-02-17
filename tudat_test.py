import tudatpy
from tudatpy.kernel import numerical_simulation, constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
import numpy as np
import matplotlib.pyplot as plt

# Load spice kernels (for celestial bodies' ephemerides)
spice.load_standard_kernels()

# Define bodies in simulation
bodies = environment_setup.create_system_of_bodies(
    environment_setup.get_default_body_settings(["Earth"], "Earth")
)

# Set up Earth's atmosphere model (e.g., NRLMSISE-00)
earth = bodies.get_body("Earth")
earth.set_atmosphere_model(
    environment_setup.create_atmosphere_model(
        environment_setup.atmosphere.nrlmsise00(), "Earth"
    )
)

# Create a satellite object
satellite_name = "ReentrySatellite"
bodies.create_empty_body(satellite_name)
bodies.get_body(satellite_name).set_constant_mass(500)  # 500 kg

# Define aerodynamic properties (drag coefficient, area)
reference_area = 2.5  # m²
drag_coefficient = 2.2  # Assumed constant
aero_coefficient_settings = environment_setup.aerodynamics.constant_aerodynamic_coefficients(
    reference_area, [drag_coefficient, 0, 0], independent_variable_settings=[]
)
bodies.get_body(satellite_name).set_aerodynamic_coefficient_interface(
    environment_setup.create_aerodynamic_coefficient_interface(
        aero_coefficient_settings, satellite_name, bodies
    )
)

# Define acceleration models
accelerations_on_satellite = {
    satellite_name: [
        propagation_setup.acceleration.point_mass_gravity("Earth"),
        propagation_setup.acceleration.aerodynamic("Earth"),
    ]
}

# Propagation settings
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, accelerations_on_satellite, [satellite_name], ["Earth"]
)

# Define initial conditions (altitude ~120 km, reentry trajectory)
initial_altitude = 120000  # meters
initial_velocity = 7800  # m/s (approximate low-Earth orbit velocity)
initial_state = np.array([
    earth.shape_model.radius + initial_altitude,  # x (altitude above Earth's surface)
    0.0,  # y
    0.0,  # z
    0.0,  # vx
    initial_velocity,  # vy (tangential velocity)
    0.0,  # vz
])

# Define termination conditions (e.g., stop when below 20 km altitude)
termination_altitude = 20000  # meters
termination_settings = propagation_setup.propagator.hybrid_termination(
    [
        propagation_setup.propagator.time_termination(1000),  # Max simulation time
        propagation_setup.propagator.dependent_variable_termination(
            propagation_setup.dependent_variable.altitude(satellite_name, "Earth"),
            termination_altitude,  # Stop at 20 km
            True
        ),
    ]
)

# Define integrator settings (Runge-Kutta 4(5))
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    initial_time=0,
    initial_step_size=1.0,
    coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
    minimum_step_size=0.1,
    maximum_step_size=10.0,
    relative_error_tolerance=1e-8,
    absolute_error_tolerance=1e-8
)

# Set up propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies=["Earth"],
    bodies_to_integrate=[satellite_name],
    acceleration_models=acceleration_models,
    initial_states=initial_state,
    termination_settings=termination_settings
)

# Run the simulation
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings, integrator_settings
)

# Extract results
states = dynamics_simulator.state_history

# Extract altitude and time
time_values = np.array(list(states.keys()))
altitudes = np.array([np.linalg.norm(state[:3]) - earth.shape_model.radius for state in states.values()])

# Plot altitude over time
plt.figure()
plt.plot(time_values, altitudes / 1000)  # Convert to km
plt.xlabel("Time (s)")
plt.ylabel("Altitude (km)")
plt.title("Satellite Reentry Altitude Over Time")
plt.grid()
plt.show()
