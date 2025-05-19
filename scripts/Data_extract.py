import numpy as np
import pandas as pd
from pathlib import Path

def convert_to_date(input):
    year = "20" + input[1:3] + "-01-01"
    num = float(input[3:]) - 1
    date = np.datetime64(year) + np.timedelta64(int(num * 24 * 60 * 60), "s")
    return date

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

    # Lists to store results
    periapsis_list = []
    apoapsis_list = []
    epoch_list = []

    # Process each TLE
    for line1, line2 in tle_list:
        mean_motion_rev_per_day = float(line2[52:63])
        eccentricity = float(f"0.{line2[26:33]}")

        mean_motion_rad_per_sec = mean_motion_rev_per_day * 2 * np.pi / seconds_per_day
        semi_major_axis_km = (mu_earth / mean_motion_rad_per_sec**2)**(1/3)

        periapsis = semi_major_axis_km * (1 - eccentricity)
        apoapsis = semi_major_axis_km * (1 + eccentricity)

        periapsis_list.append(periapsis - 6378)
        apoapsis_list.append(apoapsis - 6378)

        epoch = convert_to_date(line1[17:32])
        epoch_list.append(epoch)

    datetime_array = pd.to_datetime(epoch_list)
    start_datetime = datetime_array[0]
    hours_array = (datetime_array - start_datetime).total_seconds() / 3600.0

    # Create DataFrame
    df = pd.DataFrame({
        'Hours_since_start': hours_array,
        'Datetime': datetime_array,
        'Periapsis_altitude_km': periapsis_list,
        'Apoapsis_altitude_km': apoapsis_list
    })

    # Save to CSV
    output_csv_path = 'tle_data_extracted.csv'
    df.to_csv(output_csv_path, index=False)

    return df

if __name__ == "__main__":
    import matplotlib.pyplot as plt

<<<<<<< HEAD:Data_extract.py
    per, ap, time, datetime = TLE_extract('TLEs_Satellites/Delfi-n3Xt_TLE')
    plt.plot(datetime, per, label='Periapsis')
    plt.plot(datetime, ap, label='Apoapsis')
    plt.xlabel('Hours since 2021-11-12')
=======
    df = TLE_extract('TLEs_Satellites/Delfi-n3Xt_TLE')

    plt.plot(df['Datetime'], df['Periapsis_altitude_km'], label='Periapsis')
    plt.plot(df['Datetime'], df['Apoapsis_altitude_km'], label='Apoapsis')
    plt.xlabel('Date')
>>>>>>> 72a5089bfee05f860abbeba367b6509a687270c6:scripts/Data_extract.py
    plt.ylabel('Altitude (km)')
    plt.legend()
    plt.grid(True)
    plt.title('Apoapsis and Periapsis vs. Time')
    plt.savefig('tle_plot.png')
    plt.show()
