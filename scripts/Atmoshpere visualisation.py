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

# Calculate MSIS for version 2.1
output_v21 = pymsis.calculate(date, lons, lats, alt, geomagnetic_activity=-1, version=2.1)
output_v21 = np.squeeze(output_v21)

# Create meshgrid for plotting
xx, yy = np.meshgrid(lons, lats)

# Plot for version 0
fig, ax = plt.subplots()
mesh_v0 = ax.pcolormesh(xx, yy, output_v0[:, :, pymsis.Variable.MASS_DENSITY].T, shading="auto")
cbar_v0 = plt.colorbar(mesh_v0)
cbar_v0.set_label(label="Mass density (kg/m$^3$)", fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel("Longitude (deg)", fontsize = 18)
ax.set_ylabel("Latitude (deg)", fontsize = 18)
plt.savefig("scripts/script_results/msis_v0.png")
plt.close()

# Plot for version 2.1
fig, ax = plt.subplots()
mesh_v21 = ax.pcolormesh(xx, yy, output_v21[:, :, pymsis.Variable.MASS_DENSITY].T, shading="auto")
cbar_v21 = plt.colorbar(mesh_v21)
cbar_v21.set_label(label="Mass density (kg/m$^3$)", fontsize = 18)
plt.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel("Longitude (deg)", fontsize = 18)
ax.set_ylabel("Latitude (deg)", fontsize = 18)
plt.savefig("scripts/script_results/msis_v21.png")
plt.close()

# Calculate and plot the error (difference) between versions
error = output_v21[:, :, pymsis.Variable.MASS_DENSITY] - output_v0[:, :, pymsis.Variable.MASS_DENSITY]
fig, ax = plt.subplots()
mesh_err = ax.pcolormesh(xx, yy, error.T, shading="auto", cmap="RdBu")
cbar_err = plt.colorbar(mesh_err)
cbar_err.set_label(label="Mass density difference (kg/m$^3$)", fontsize =18)
plt.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel("Longitude (deg)", fontsize =18)
ax.set_ylabel("Latitude (deg)", fontsize = 18)
plt.savefig("scripts/script_results/msis_error.png")
plt.close()