import numpy as np
import pymsis

# dates = np.arange(np.datetime64("2020-10-28T00:00"), np.datetime64("2025-03-04T00:00"), np.timedelta64(30, "m"))
longs = np.arange(-180,180,1)
lats = np.arange(-90,90,1)
# geomagnetic_activity=-1 is a storm-time run
data1 = pymsis.calculate(np.datetime64("2025-03-04T00:00"), longs, 0, 600, geomagnetic_activity=-1, version=0)
data2 = pymsis.calculate(np.datetime64("2025-03-04T00:00"), longs, 0, 600, geomagnetic_activity=-1, version=2.0)
data3 = pymsis.calculate(np.datetime64("2025-03-04T00:00"), longs, 0, 600, geomagnetic_activity=-1, version=2.1)

# Plot the data
import matplotlib.pyplot as plt
# Total mass density over time
plt.plot(longs, data1[0, :, 0, 0, 0], label="v0")
plt.plot(longs, data2[0, :, 0, 0, 0], label="v2.0")
plt.plot(longs, data3[0, :, 0, 0, 0], label="v2.1")
plt.legend()
plt.show()