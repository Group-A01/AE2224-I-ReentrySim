from matplotlib import pyplot as plt
import numpy as np
import math

h0 = 150000
inclination = 97

l_sat = 0.1
mass_sat = 2.2

r_earth = 6378136
T_earth = 86154

mu_earth = 3.986004418e14
mu_moon = 4.9048695e12
mu_sun = 1.32712440018e20

dt = 0.1
t_speed = 10000

posMag = []
velMag = []
accMag = []
time = []

class Body:
    def __init__(self, name, mu, position, velocity):
        self.name = name
        self.mu = mu
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

        self.x = []
        self.y = []

    def gravitational_acceleration(self, other):
        r = other.position - self.position
        r_mag = np.linalg.norm(r)
        return (other.mu/r_mag**3)*r

    def update_position_velocity(self, acceleration, dt):
        self.velocity += acceleration*dt
        self.position += self.velocity*dt

        self.x.append(self.position[0])
        self.y.append(self.position[1])

class Satellite(Body):
    def __init__(self, name, mass, position, velocity, drag_area, drag_coefficient):
        super().__init__(name, None, position, velocity)
        self.mass = mass
        self.drag_area = drag_area
        self.drag_coefficient = drag_coefficient

    def atmospheric_drag(self, velocity_rel, altitude):
        if altitude <= 150:
            density = 1.225 * (0.863615 ** altitude)
        else:
            density = 0.14081e-8 * (0.985172 ** altitude)
            
        return -0.5*density*self.drag_coefficient*self.drag_area*np.linalg.norm(velocity_rel)*velocity_rel

    def j2_accelerations(self, position, t):
        latitude = math.atan(position[2]/math.sqrt(position[0]**2+position[1]**2))
        longitude = math.atan2(position[1], position[0])

        if position[1] < 0:
            longitude += 2*math.pi

        longitude -= t*2*math.pi/T_earth

        j2a = 0#3*mu_earth*1082.6267e-6*r_earth**2*np.linalg.norm(position)**-4*math.sin(latitude)*math.cos(latitude)*np.array([0,0,1])
        j22a = 0#-12*mu_earth*1.76e-6*r_earth**2*np.linalg.norm(position)**-4*math.cos(latitude)*math.sin(2*longitude)*np.array([-math.sin(longitude), math.cos(longitude),0])
        
        return j2a+j22a

class Simulation:
    def __init__(self, bodies, satellite, dt, t_speed):
        self.bodies = bodies
        self.satellite = satellite
        self.dt = dt
        self.t_speed = t_speed
        self.t = 0
        self.high_accuracy = False

    def run(self):
        while np.linalg.norm(self.satellite.position - self.bodies['Earth'].position) > r_earth:
            for i in range(self.t_speed):
                altitude = np.linalg.norm(self.satellite.position-self.bodies['Earth'].position)-r_earth
                if altitude < 1000 and not self.high_accuracy:
                    self.t_speed = 1
                    self.dt = 1
                    self.high_accuracy = True
                    break
                
                total_acceleration = np.zeros(3)
                for body in self.bodies.values():
                    total_acceleration += self.satellite.gravitational_acceleration(body)
                    
                total_acceleration += self.satellite.j2_accelerations(satellite.position-earth.position, self.t)
                
                drag_force = self.satellite.atmospheric_drag(self.satellite.velocity-self.bodies['Earth'].velocity, altitude/1000)
                SRP_force = 0.9*4.5e-6*3/2*l_sat**2*self.satellite.position/np.linalg.norm(self.satellite.position)
                
                total_acceleration += (drag_force+SRP_force)/self.satellite.mass
                self.satellite.update_position_velocity(total_acceleration, self.dt)
                
                for body in self.bodies.values():
                    if body.name != 'Sun':
                        acceleration = sum(body.gravitational_acceleration(other) for other in self.bodies.values() if other != body)
                        body.update_position_velocity(acceleration, self.dt)
                
                self.t += self.dt

                posMag.append((np.linalg.norm(satellite.position-earth.position)-r_earth)/1000)
                velMag.append(np.linalg.norm(satellite.velocity-earth.velocity))
                accMag.append(np.linalg.norm(total_acceleration/9.81))

                time.append(self.t)

            print(self.t, altitude/1000, np.linalg.norm(self.satellite.velocity-self.bodies['Earth'].velocity))


sun = Body("Sun", mu_sun, [0,0,0], [0,0,0])
earth = Body("Earth", mu_earth, [148.16e9, 0, 0], [0, math.cos(23.44*math.pi/180)*math.sqrt(mu_sun/148.16e9), math.sin(23.44*math.pi/180)*math.sqrt(mu_sun/148.16e9)])
moon = Body("Moon", mu_moon, earth.position - [385e6, 0, 0], earth.velocity - math.sqrt(mu_earth/385e6)*np.array([0, math.cos(5.14*math.pi/180), math.sin(5.14*math.pi/180)]))
satellite = Satellite("Satellite", mass_sat, earth.position + [h0+r_earth, 0, 0], earth.velocity + math.sqrt(mu_earth/(h0+r_earth))*np.array([0, math.cos(inclination*math.pi/180), math.sin(inclination*math.pi/180)]),3/2*l_sat**2, 1.67)

sim = Simulation({'Earth': earth, 'Moon': moon, 'Sun' : sun}, satellite, dt, t_speed)
sim.run()

plt.plot(time, posMag)
plt.title("Height - Time")
plt.xlabel("Time [s]")
plt.ylabel("Height [km]")
plt.show()

plt.plot(time, velMag)
plt.title("Speed - Time")
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.show()


plt.plot(time, accMag)
plt.title("Acceleration - Time")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [g]")
plt.show()

orbit_x = []
orbit_y = []
circle = plt.Circle((0, 0), r_earth, color='blue')

for i in range(len(earth.x)):
    orbit_x.append(satellite.x[i]-earth.x[i])
    orbit_y.append(satellite.y[i]-earth.y[i])
             
plt.axes().set_aspect('equal')
plt.gca().set_facecolor((0,0,0))
plt.plot(orbit_x, orbit_y, color='red')
plt.gca().add_patch(circle)
plt.title("Satellite Trajectory")
plt.show()
