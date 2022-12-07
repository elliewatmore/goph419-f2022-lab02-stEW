import numpy as np
import matplotlib.pyplot as plt
from linalg_interp import spline_function

#import air and water density data
air_data = np.loadtxt('air_density_vs_temp_eng_toolbox.txt')
water_data = np.loadtxt('water_density_vs_temp_usgs.txt')

#solving spline interpolation functions
#air density data
x1 = air_data[:,0]
y1 = air_data[:,1]
spline1 = spline_function(x1, y1, order = 1)
spline2 = spline_function(x1, y1, order = 2)
spline3 = spline_function(x1, y1, order = 3)

#water density data
x2 = water_data[:,0]
y2 = water_data[:,1]
spline4 = spline_function(x2, y2, order = 1)
spline5 = spline_function(x2, y2, order = 2)
spline6 = spline_function(x2, y2, order = 3)

#calculating interpolated spline data over 100 equally spaced temperature values

#obtaining 100 equally spaced temperature values between min and max values
air_temps = np.linspace(np.min(x1),np.max(x1), 100)
water_temps = np.linspace(np.min(x2), np.max(x2), 100)


fig, ax =plt.subplots(2,3, figsize = (15,10))

#order = 1
y_air_1 = spline1(air_temps)
ax[0][0].plot(air_temps,y_air_1,'m')
ax[0][0].plot(x1,y1, 'ro')
ax[0][0].set_xlabel('Temperature (°C)')
ax[0][0].set_ylabel('Water density (g/cm^3)')
ax[0][0].set_title('Water density vs temp (linear spline function)')

#order =2 
y_air_2 = spline2(air_temps)
ax[0][1].plot(air_temps,y_air_2,'c')
ax[0][1].plot(x1,y1, 'bs')
ax[0][1].set_xlabel('Temperature (°C)')
ax[0][1].set_ylabel('Water density (g/cm^3)')
ax[0][1].set_title('Water density vs temp (quadratic spline function)')

#order = 3
y_air_3 = spline3(air_temps)
ax[0][2].plot(air_temps,y_air_3,'tab:green')
ax[0][2].plot(x1,y1, 'g^')
ax[0][2].set_xlabel('Temperature (°C)')
ax[0][2].set_ylabel('Water density (g/cm^3)')
ax[0][2].set_title('Water density vs temp (cubic spline function)')

#order 1
y_water_1 = spline4(water_temps)
ax[1][0].plot(water_temps,y_water_1,'m')
ax[1][0].plot(x2,y2, 'ro')
ax[1][0].set_xlabel('Temperature (°C)')
ax[1][0].set_ylabel('Air density (kg/m^3)')
ax[1][0].set_title('Air density vs temp (linear spline function)')
#order =2 
y_water_2 = spline5(water_temps)
ax[1][1].plot(water_temps,y_water_2,'c')
ax[1][1].plot(x2,y2, 'bs')
ax[1][1].set_xlabel('Temperature (°C)')
ax[1][1].set_ylabel('Air density (kg/m^3)')
ax[1][1].set_title('Air density vs temp (quadratic spline function)')

#order = 3
y_water_3 = spline6(water_temps)
ax[1][2].plot(water_temps,y_water_3,'tab:green')
ax[1][2].plot(x2,y2, 'g^')
ax[1][2].set_xlabel('Temperature (°C)')
ax[1][2].set_ylabel('Air density (kg/m^3)')
ax[1][2].set_title('Air density vs temp (cubic spline function)')

plt.savefig("Air/water density vs temp")


