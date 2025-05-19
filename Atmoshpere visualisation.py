import matplotlib.pyplot as plt
import numpy as np
import pymsis

# Setup common parameters
lons = np.arange(-180, 185, 5)
lats = np.arange(-90, 95, 5)
alt = 400
f107 = 150
f107a = 150
ap = 7
date = np.datetime64("2025-04-26T12:00")
aps = [[ap] * 7]

# Calculate MSIS for version 0
output_v0 = pymsis.calculate(date, lons, lats, alt, geomagnetic_activity=-1, version=0)
output_v0 = np.squeeze(output_v0)
density_v0 = output_v0[:, :, pymsis.Variable.MASS_DENSITY]

# Calculate MSIS for version 2.1
output_v21 = pymsis.calculate(date, lons, lats, alt, geomagnetic_activity=-1, version=2.1)
output_v21 = np.squeeze(output_v21)
density_v21 = output_v21[:, :, pymsis.Variable.MASS_DENSITY]

# Calculate percentage difference: ((v2.1 - v0) / v0) * 100
percentage_diff = ((density_v21 - density_v0) / density_v0) * 100

# Create meshgrid for plotting
xx, yy = np.meshgrid(lons, lats)

# Plot percentage difference
fig, ax = plt.subplots()
mesh_err = ax.pcolormesh(xx, yy, percentage_diff.T, shading="auto", cmap="RdBu")
plt.colorbar(mesh_err, label="Percentage difference (%)")
ax.set_title(f"Percentage difference in air density at {alt} km (MSIS v2.1 vs v0)")
ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
plt.savefig("msis_percentage_diff.png")
plt.close()

# Calculate summary statistics
mean_percentage_diff = np.mean(percentage_diff)
max_percentage_diff = np.max(percentage_diff)
min_percentage_diff = np.min(percentage_diff)
print(f"Mean percentage difference: {mean_percentage_diff:.2f}%")
print(f"Max percentage difference: {max_percentage_diff:.2f}%")
print(f"Min percentage difference: {min_percentage_diff:.2f}%")