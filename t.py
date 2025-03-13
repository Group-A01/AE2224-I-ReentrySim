import numpy as np
import pymsis

time = 70000000000000.0
timedate = np.datetime64("2000-01-01T00:00") + np.timedelta64(int(time), 's')
print(timedate)