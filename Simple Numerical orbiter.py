import numpy as np
from alive_progress import alive_bar

#Convenient functions
def mag(v):
    return np.linalg.norm(v)

def setMag(v, newMag):
    return v*newMag/mag(v)

#Earth params
mu_earth = 3.986004418*pow(10, 14)
T_earth = 86164.0905
r_earth = 6378136

l_cube = 0.1 #Satellite side length [m]
mass_cube = 1.5 #Satellite mass [kg]

t = 0
t_res = 10 #Numerical time resolution [s]
t_speed = 100000 #Simulation speed [min/s/t_res]

#Initial position and velocity vectors
position = np.array([600000+r_earth, 0, 0]) #[m]
velocity = np.array([0, np.sqrt(mu_earth/mag(position)), 0]) #[m/s]

highAccuracy = False

def drag(velocity, h): #Atmospheric drag
    temperature = 2000-(2000-186.946)*np.exp((90000-h*1000)/50000)
    print(mag(velocity), temperature)
    if(h <= 150):
        density = 1.225*pow(0.863615,h)
    else:
        density = 0.14081*pow(10,-8)*pow(0.985172,h)

    M = mag(velocity)/np.sqrt(1.4*287*temperature)
    # cd = 1.67
    cd = 2.1*np.exp(-1.16*(M+0.35))-6.5*np.exp(-2.23*(M+0.35))+1.67
    # print(cd)
    return setMag(-velocity, 0.5*cd*density*pow(mag(velocity),2)*3/2*pow(l_cube,2))

with alive_bar(1000, title="Orbit") as bar:
    while(mag(position) > r_earth): #Numerical iteration
        for i in range(t_speed):
            if(mag(position)-r_earth < 250 and highAccuracy == False):
                t_speed = 1
                t_res = 1
                highAccuracy = True
                break
                
            g_force = setMag(-position, mu_earth/pow(mag(position), 2))
            drag_force = drag(velocity, (mag(position)-r_earth)/1000)
            
            resultant_force = g_force + drag_force/mass_cube
            
            velocity = velocity + resultant_force*t_res
            position = position + velocity*t_res
            
            t += t_res

        print(t, (mag(position)-r_earth)/1000, mag(velocity)) #Prints satellite height, speed, and time
        bar()