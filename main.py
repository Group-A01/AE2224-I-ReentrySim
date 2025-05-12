import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment, environment_setup, propagation_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy.util import result2array
import aaapymsis
from alive_progress import alive_bar
from Data_extract import TLE_extract, convert_to_date
import os, time
import urllib.request

def fetch_tle_data():
    """Fetch TLE data for Delfi-n3Xt from Celestrak."""
    target_url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=39428"
    try:
        with urllib.request.urlopen(target_url) as response:
            data = response.read().decode('utf-8')
            lines = data.splitlines()
            for i in range(len(lines) - 1):
                if lines[i].startswith("1 39428U"):
                    return (lines[i], lines[i + 1])
            
    except Exception as e:
        raise RuntimeError(f"Failed to fetch TLE data: {e}")

def setup_body_settings(satname, reference_area, drag_coefficient, radiation_pressure_coefficient, atm_model):
    """Set up celestial and satellite body settings with environment configurations."""
    bodies_to_create = ["Sun", "Earth", "Moon", "Jupiter"]
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation
    )

    # Configure Earth's atmosphere
    if atm_model == "MSIS":
        const_temp = 1000  # Realistic thermospheric temperature in K

        def density_f(h, lon, lat, time):
            # Time is seconds since simulation start
            start_date = np.datetime64("2000-01-01T00:00")
            timedate = start_date + np.timedelta64(int(time), 's')
            data = aaapymsis.calculate(timedate, lon, lat, h / 1000, geomagnetic_activity=-1, version=2.0)
            density = data[0, aaapymsis.Variable.MASS_DENSITY]
            return density

        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.custom_four_dimensional_constant_temperature(
            density_f,
            const_temp,
            8.314 / 0.016,  # Scale height in km
            1.667  # R/M for atomic oxygen
        )
    elif atm_model == "NRLMSISE-00":
        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00()
    elif atm_model == "Exponential":
        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.exponential_predefined("Earth")
    elif atm_model == "u76":
        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.us76()

    # Configure satellite settings
    body_settings.add_empty_settings(satname)
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0.0, 0.0]
    )
    body_settings.get(satname).aerodynamic_coefficient_settings = aero_coefficient_settings

    occulting_bodies_dict = {"Sun": ["Earth"]}
    vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area, radiation_pressure_coefficient, occulting_bodies_dict
    )
    body_settings.get(satname).radiation_pressure_target_settings = vehicle_target_settings

    return body_settings

def setup_accelerations(satname):
    """Define acceleration models for the satellite."""
    accelerations_settings = {
        satname: {
            "Sun": [
                propagation_setup.acceleration.radiation_pressure(),
                propagation_setup.acceleration.point_mass_gravity()
            ],
            "Earth": [
                propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
                propagation_setup.acceleration.aerodynamic()
            ],
            "Moon": [propagation_setup.acceleration.point_mass_gravity()],
            "Jupiter": [propagation_setup.acceleration.point_mass_gravity()]
        }
    }
    return accelerations_settings

def main(override = False, sat_choice='', atm_choice='', duration_choice='', term_choice='', end_date_str=''):
    """Main function to run the satellite orbit simulation."""
    # Create output directories
    satellites = ["Delfi-C3", "Delfi-PQ", "Delfi-n3Xt"]
    for sat in satellites:
        os.makedirs(f'results/{sat}', exist_ok=True)

    # Load SPICE kernels
    spice.load_standard_kernels()
    
    # User input for satellite
    while True:
        if not override:
            print("Available Satellites:")
            print("1. Delfi-C3\n2. Delfi-PQ\n3. Delfi-n3Xt")
            sat_choice = input("Enter your choice (1-3, or 'q' to quit): ").strip().lower()
        if sat_choice == 'q':
            print("Exiting program.")
            return
        if sat_choice in ['1', '2', '3']:
            tle_data_n3Xt=('1 39428U 13066N   25127.74836985  .00006212  00000-0  63855-3 0  9990',
                        '2 39428  97.8359  45.7788 0087348 108.1758 252.8993 14.87866514614197')
            if sat_choice == '3':
                # tle_data_n3Xt = fetch_tle_data() #fetch from online
                print(f'The TLE of n3Xt is {tle_data_n3Xt}')
            break
        print("Invalid choice. Please enter 1, 2, 3, or 'q'.")

    # User input for atmospheric model
    while True:
        if not override:
            print("\nAvailable Atmospheric Models:")
            print("1. MSIS (NRLMSISE-2.0 via pymsis)\n2. NRLMSISE-00\n3. Exponential\n4. US76")
            atm_choice = input("Enter your choice (1-4, or 'q' to quit): ").strip().lower()
        if atm_choice == 'q':
            print("Exiting program.")
            return
        if atm_choice in ['1', '2', '3', '4']:
            break
        print("Invalid choice. Please enter 1, 2, 3, 4, or 'q'.")

    # User input for simulation duration
    while True:
        if not override:
            print("\nSimulation Duration Options:")
            print("1. Full simulation\n2. Last 2 years of the satellite's operation")
            duration_choice = input("Enter your choice (1-2, or 'q' to quit): ").strip().lower()
        if duration_choice == 'q':
            print("Exiting program.")
            return
        if duration_choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1, 2, or 'q'.")

    # User input for termination condition
    while True:
        if not override:
            print(f"\nSimulation Termination Options:")
            print("1. Until altitude reaches 200 km\n2. Specify an end date")
            term_choice = input("Enter your choice (1-2, or 'q' to quit): ").strip().lower()
        if term_choice == 'q':
            print("Exiting program.")
            return
        if term_choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1, 2, or 'q'.")
    
    satname = satellites[int(sat_choice) - 1]
    atm_model = ["MSIS", "NRLMSISE-00", "Exponential", "u76"][int(atm_choice) - 1]
    
    # Satellite parameters
    satellite_params = {
        "Delfi-C3": {
            "mass": 2.2,
            "reference_area": 0.06335596,
            "drag_coefficient": 3.884171,
            "tle_initial": (
                "1 32789U 07021G   08119.60740078 -.00000054  00000-0  00000+0 0  9999",
                "2 32789 098.0082 179.6267 0015321 307.2977 051.0656 14.81417433    68"
            ),
            "start_initial": "2008-04-28",
            "tle_last2": (
                "1 32789U 08021G   21316.65200823  .00002906  00000-0  18317-3 0  9999",
                "2 32789  97.3635 346.2745 0011231   1.1170 359.0083 15.09850051739183"
            ),
            "start_last2": "2021-11-12"
        },
        "Delfi-PQ": {
            "mass": 0.6,
            "reference_area": 0.02888152,
            "drag_coefficient": 3.295226,
            "tle_initial": (
                "1 51074U 22002CU  22018.63976129  .00005793  00000-0  31877-3 0  9992",
                "2 51074  97.5269  88.2628 0013258 250.6199 109.3600 15.14370988   760"
            ),
            "start_initial": "2022-01-18",
            "tle_last2": (
                "1 51074U 22002CU  22018.63976129  .00005793  00000-0  31877-3 0  9992",
                "2 51074  97.5269  88.2628 0013258 250.6199 109.3600 15.14370988   760"
            ),
            "start_last2": "2022-01-18"
        },
        "Delfi-n3Xt": {
            "mass": 2.8,
            "reference_area": 0.10421905,
            "drag_coefficient": 2.459680e+00,
            "tle_initial": (
                "1 39428U 13066N   13326.98735140  .00000434  00000-0  85570-4 0  9994",
                "2 39428 097.7885 039.5438 0131608 184.9556 175.0377 14.61934043   196"
            ),
            "start_initial": "2013-11-22",
            # "start_initial": "2025-05-09",
            "tle_last2": tle_data_n3Xt,
            "start_last2": str(convert_to_date(tle_data_n3Xt[0][17:32]))[:10]
        }
    }

    params = satellite_params[satname]
    mass = params["mass"]
    reference_area = params["reference_area"]
    drag_coefficient = params["drag_coefficient"]

    # Set simulation start date and TLE
    try:
        if duration_choice == "1":
            period = 'full'
            start_date = datetime.strptime(params["start_initial"], "%Y-%m-%d")
            tle_data = params["tle_initial"]
            print(f"Selected: Full simulation for {satname} starting from {params['start_initial']}")
        else:
            period = 'last2years'
            start_date = datetime.strptime(params["start_last2"], "%Y-%m-%d")
            tle_data = params["tle_last2"]
            print(f"Selected: Last 2 years for {satname} starting from {params['start_last2']}")

        simulation_start_epoch = DateTime(start_date.year, start_date.month, start_date.day).epoch()
        delfi_tle = environment.Tle(*tle_data)
        delfi_ephemeris = environment.TleEphemeris("Earth", "J2000", delfi_tle, False)
        initial_state = delfi_ephemeris.cartesian_state(simulation_start_epoch)
    except Exception as e:
        print(f"Error setting up TLE or start date: {e}")
        return

    # Setup bodies
    try:
        body_settings = setup_body_settings(satname, reference_area, drag_coefficient, 1.2, atm_model)
        bodies = environment_setup.create_system_of_bodies(body_settings)
        bodies.get(satname).mass = mass
    except Exception as e:
        print(f"Error setting up bodies: {e}")
        return

    # Setup termination condition
    altitude_variable = propagation_setup.dependent_variable.altitude(satname, "Earth")
    altitude_termination = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=altitude_variable, limit_value=200.0e3, use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )

    if duration_choice == "2" and term_choice == "1":
        termination_condition = altitude_termination
        print("Simulating until altitude reaches 200 km...")
    elif duration_choice == "2" and term_choice == "2":
        while True:
            try:
                if not override:
                    end_date_str = input("Enter end date for last 2 years simulation (YYYY-MM-DD): ").strip()
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                simulation_end_epoch = DateTime(end_date.year, end_date.month, end_date.day).epoch()
                if simulation_end_epoch <= simulation_start_epoch:
                    print(f"End date must be after {start_date.strftime('%Y-%m-%d')}. Try again.")
                    continue
                time_termination = propagation_setup.propagator.time_termination(simulation_end_epoch)
                termination_condition = propagation_setup.propagator.hybrid_termination(
                    [altitude_termination, time_termination], fulfill_single_condition=True
                )
                print(f"Simulating from {start_date.strftime('%Y-%m-%d')} to {end_date_str} or 200 km altitude...")
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD (e.g., 2024-12-31).")
    elif term_choice == "1":
        termination_condition = altitude_termination
        print("Simulating until altitude reaches 200 km...")
    else:
        while True:
            try:
                if not override:
                    end_date_str = input("Enter end date (YYYY-MM-DD): ").strip()
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                simulation_end_epoch = DateTime(end_date.year, end_date.month, end_date.day).epoch()
                if simulation_end_epoch <= simulation_start_epoch:
                    print(f"End date must be after {start_date.strftime('%Y-%m-%d')}. Try again.")
                    continue
                time_termination = propagation_setup.propagator.time_termination(simulation_end_epoch)
                termination_condition = propagation_setup.propagator.hybrid_termination(
                    [altitude_termination, time_termination], fulfill_single_condition=True
                )
                print(f"Simulating until {end_date_str} or 200 km altitude...")
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD (e.g., 2024-12-31).")

    # Setup propagation
    bodies_to_propagate = [satname]
    central_bodies = ["Earth"]
    acceleration_settings = setup_accelerations(satname)
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    dependent_variables_to_save = [
        #propagation_setup.dependent_variable.total_acceleration(satname),
        #propagation_setup.dependent_variable.keplerian_state(satname, "Earth"),
        #propagation_setup.dependent_variable.single_acceleration_norm(
            #propagation_setup.acceleration.point_mass_gravity_type, satname, "Sun"
        #),
        #propagation_setup.dependent_variable.single_acceleration_norm(
        #     propagation_setup.acceleration.point_mass_gravity_type, satname, "Moon"
        # ),
        # propagation_setup.dependent_variable.single_acceleration_norm(
        #     propagation_setup.acceleration.spherical_harmonic_gravity_type, satname, "Earth"
        # ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.aerodynamic_type, satname, "Earth"
        ),
        # propagation_setup.dependent_variable.single_acceleration_norm(
        #     propagation_setup.acceleration.radiation_pressure_type, satname, "Sun"
        # ),
        #propagation_setup.dependent_variable.altitude(satname, "Earth"),
        propagation_setup.dependent_variable.periapsis_altitude(satname, "Earth"),
        propagation_setup.dependent_variable.apoapsis_altitude(satname, "Earth"),
        # propagation_setup.dependent_variable.single_acceleration_norm(
        #     propagation_setup.acceleration.point_mass_gravity_type, satname, "Jupiter"
        # )
    ]

    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step=60.0,
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_45,
        step_size_control_settings=propagation_setup.integrator.step_size_control_elementwise_scalar_tolerance(1.0E-10, 1.0E-10),
        step_size_validation_settings=propagation_setup.integrator.step_size_validation(1, 1000.0)
    )

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies, acceleration_models, bodies_to_propagate, initial_state,
        simulation_start_epoch, integrator_settings, termination_condition,
        output_variables=dependent_variables_to_save
    )

    # Run simulation
    try:
        print("Starting numerical integration...")
        tik = time.time()
        with alive_bar(title="Numerical integration:") as bar:
            dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
            bar()
    except Exception as e:
        print(f"Simulation failed: {e}")
        return
    tok = time.time()
    print("Elapsed time: {0}s".format(timedelta(seconds=int(tok-tik))))
    # Process results
    states = dynamics_simulator.propagation_results.state_history
    states_array = result2array(states)
    dep_vars = dynamics_simulator.propagation_results.dependent_variable_history
    dep_vars_array = result2array(dep_vars)

    time_seconds = dep_vars_array[:, 0]
    time_hours = (time_seconds - time_seconds[0]) / 3600
    periapsis = dep_vars_array[:, 2] / 1000
    apoapsis = dep_vars_array[:, 3] / 1000
    acceleration_norm_aero_earth = dep_vars_array[:, 1]

    data = pd.DataFrame({
        'periapsis': periapsis,
        'apoapsis': apoapsis,
        'acc_aero_earth': acceleration_norm_aero_earth
    })

    time_datetime = start_date + pd.to_timedelta(time_seconds - time_seconds[0], unit='s')
    data.index = time_datetime

    # Resample data
    first_midnight = pd.Timestamp(start_date).ceil('D')
    try:
        initial_data = data.loc[:first_midnight].resample('h').mean()
        post_midnight_data = data.loc[first_midnight:].resample('1h', origin=first_midnight).mean()
        resampled_data = pd.concat([initial_data, post_midnight_data])
        resampled_data = resampled_data.loc[~resampled_data.index.duplicated(keep='first')]
        resampled_dates = resampled_data.index
    except Exception as e:
        print(f"Error resampling data: {e}")
        return

    resampled_time_seconds = (resampled_dates - start_date).total_seconds()
    resampled_time_hours = resampled_time_seconds / 3600
    final_date = start_date + timedelta(seconds=time_seconds[-1])

    # Plotting
    # Note: Using 'seaborn-v0_8' or 'ggplot' as a fallback. Install seaborn (`pip install seaborn`) for full Seaborn styles.
    # try:
    #     plt.style.use('seaborn-v0_8')
    # except:
    #     plt.style.use('ggplot')
    
    # Call TLE_extract and capture tle_list
    actual_periapsis, actual_apoapsis, _, actual_dates = TLE_extract("TLEs_Satellites/"+satname+"_TLE")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(resampled_dates, resampled_data['periapsis'], label='Periapsis Altitude', linewidth=1)
    ax1.plot(resampled_dates, resampled_data['apoapsis'], label='Apoapsis Altitude', linewidth=1)
    # ax1.plot(actual_dates, actual_periapsis, label='Actual Periapsis Altitude', linewidth=1)
    # ax1.plot(actual_dates, actual_apoapsis, label='Actual Apoapsis Altitude', linewidth=1)
    
    ax1.plot()
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Altitude [km]', fontsize=12)
    ax1.set_title(f'Apoapsis and Periapsis Altitudes of {satname} (Atm: {atm_model})', fontsize=14, pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=10)
    # ax1.set_xlim([min(min(actual_dates), min(resampled_dates)), max(max(resampled_dates), max(actual_dates))])
    ax1.set_ylim([min(resampled_data['periapsis'].min(), resampled_data['apoapsis'].min()) * 0.95,
                  max(resampled_data['periapsis'].max(), resampled_data['apoapsis'].max()) * 1.05])
    plt.tight_layout()
    output_path = f"results/{satname}/{satname}_altitude_{atm_model.replace(' ', '_')}_{period}.png"
    plt.savefig(output_path)
    plt.close()

    fig, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(resampled_dates, resampled_data['acc_aero_earth'], 'g-', label='Aerodynamic Earth Acceleration', linewidth=1)
    ax2.set_xlim([min(resampled_dates), max(resampled_dates)])
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Acceleration Norm [m/s²]', fontsize=12)
    ax2.set_title(f'Aerodynamic Acceleration of {satname} (Atm: {atm_model})', fontsize=14, pad=20)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = f"results/{satname}/{satname}_drag_acceleration_{atm_model.replace(' ', '_')}_{period}.png"
    plt.savefig(output_path)
    plt.close()

    # Save data to CSV
    data_resampled = np.vstack([resampled_time_hours, resampled_data['apoapsis'], resampled_data['periapsis'],
                                resampled_data['acc_aero_earth']]).T
    header="Time (Hours), Apoapsis, Periapsis, Acceleration Norm Aero Earth"
    output_path = f"results/{satname}/{satname}_{atm_model.replace(' ', '_')}_{period}.csv"
    try:
        print(f"Writing to file: {output_path}...")
        np.savetxt(output_path, data_resampled, header=header, delimiter=',', fmt='%.5e')
        print("Done!")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    print(f"Final simulation time: {time_hours[-1]:.2f} hours")
    print(f"Final simulation date: {resampled_dates[-1]}\n-----------------------\n")

if __name__ == "__main__":
    main(False, '3', '1', '2', '2', '2025-07-15')