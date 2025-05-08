import numpy as np
import pandas as pd
from pathlib import Path

def TLE_extract(path):
    mu_earth = 398600.4418  # km^3/s^2
    seconds_per_day = 86400

    # Load TLE file
    tle_file_path = Path(path)
    tle_lines = tle_file_path.read_text().splitlines()

    # Extract TLEs
    tle_list = []
    i = 0
    while i < len(tle_lines) - 1:
        line1 = tle_lines[i].strip()
        line2 = tle_lines[i + 1].strip()
        if line1.startswith('1') and line2.startswith('2'):
            tle_list.append((line1, line2))
            i += 2
        else:
            i += 1

    # Arrays to store periapsis and apoapsis
    periapsis_list = []
    apoapsis_list = []

    # Process each TLE
    for line1, line2 in tle_list:
        mean_motion_rev_per_day = float(line2[52:63])
        eccentricity = float(f"0.{line2[26:33]}")
        
        mean_motion_rad_per_sec = mean_motion_rev_per_day * 2 * np.pi / seconds_per_day
        semi_major_axis_km = (mu_earth / mean_motion_rad_per_sec**2)**(1/3)
        
        periapsis = semi_major_axis_km * (1 - eccentricity)
        apoapsis = semi_major_axis_km * (1 + eccentricity)
        
        periapsis_list.append(periapsis)
        apoapsis_list.append(apoapsis)

    # Convert to numpy arrays and subtract Earth's radius
    periapsis_array = np.subtract(np.array(periapsis_list), 6378)
    apoapsis_array = np.subtract(np.array(apoapsis_list), 6378)

    # Generate datetime array
    start_datetime = pd.to_datetime("2021-11-12 00:00:00")
    end_datetime = pd.to_datetime("2023-11-13 00:00:00")
    n_tles = len(periapsis_array)

    datetime_array = pd.date_range(start=start_datetime, end=end_datetime, periods=n_tles)
    
    # Convert datetime to hours since start
    hours_array = (datetime_array - start_datetime).total_seconds() / 3600.0
    hours_array_np = np.array(hours_array)

    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'Hours_since_start': hours_array_np,
        'Periapsis_altitude_km': periapsis_array,
        'Apoapsis_altitude_km': apoapsis_array
    })
    # df.to_csv('tle_data.csv', index=False)

    return periapsis_array, apoapsis_array, hours_array_np

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    per, ap, time = TLE_extract('TLEs Satellites\Delfi-n3Xt_TLE')
    plt.plot(time, per, label='Periapsis')
    plt.plot(time, ap, label='Apoapsis')
    plt.xlabel('Hours since 2021-11-12')
    plt.ylabel('Altitude (km)')
    plt.legend()
    plt.savefig('tle_plot.png')