import matplotlib.pyplot as plt
import numpy as np

import pymsis


lon = 0
lat = 70
alts = np.linspace(0, 1000, 1000)
f107 = 150
f107a = 150
ap = 7
aps = [[ap] * 7]

date = np.datetime64("2025-04-26T00:00")
output_midnight2 = pymsis.calculate(date, lon, lat, alts, f107, f107a, aps)
output_midnight0 = pymsis.calculate(date, lon, lat, alts, f107, f107a, aps, version=0)
diff_midnight = (output_midnight2 - output_midnight0) / output_midnight0 * 100

date = np.datetime64("2025-04-26T12:00")
output_noon2 = pymsis.calculate(date, lon, lat, alts, f107, f107a, aps)
output_noon0 = pymsis.calculate(date, lon, lat, alts, f107, f107a, aps, version=0)
diff_noon = (output_noon2 - output_noon0) / output_noon0 * 100


#  output is now of the shape (1, 1, 1, 1000, 11)
# Get rid of the single dimensions
diff_midnight = np.squeeze(diff_midnight)
diff_noon = np.squeeze(diff_noon)

_, ax = plt.subplots()
(line,) = ax.plot(diff_midnight[:,  pymsis.Variable.MASS_DENSITY], alts, linestyle="--", label = 'Mass Density at midnight')
ax.plot(diff_noon[:,  pymsis.Variable.MASS_DENSITY], alts, c=line.get_color(), label= 'Mass Density at noon')

ax.legend()

ax.set_xlim(-50, 50)
ax.set_ylim(0, 1000)
ax.set_xlabel("Change from MSIS-00 to MSIS2 (%)", fontsize = 18)
ax.set_ylabel("Altitude (km)", fontsize = 18)

plt.savefig('Altitude_density_difference.png')