from matplotlib import pyplot as plt
import numpy as np
import math

h0 = 150000

l_sat = 0.1
mass_sat = 2.2

r_earth = 6378136
mu_earth = 3.986004418e14
mu_moon = 4.9048695e12
mu_sun = 1.32712440018e20

dt = 1
t_speed = 10000

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
            
        return -0.5 * density * self.drag_coefficient * self.drag_area * np.linalg.norm(velocity_rel) * velocity_rel

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
                
                drag_force = self.satellite.atmospheric_drag(self.satellite.velocity-self.bodies['Earth'].velocity, altitude/1000)
                total_acceleration += drag_force/self.satellite.mass
                
                self.satellite.update_position_velocity(total_acceleration, self.dt)
                
                for body in self.bodies.values():
                    if body.name != 'Sun':
                        acceleration = sum(body.gravitational_acceleration(other) for other in self.bodies.values() if other != body)
                        body.update_position_velocity(acceleration, self.dt)
                
                self.t += self.dt
                
            print(self.t, altitude/1000, np.linalg.norm(self.satellite.velocity-self.bodies['Earth'].velocity))


sun = Body("Sun", mu_sun, [0,0,0], [0,0,0])
earth = Body("Earth", mu_earth, [148.16e9, 0, 0], [0, math.sqrt(mu_sun/148.16e9), 0])
moon = Body("Moon", mu_moon, earth.position - np.array([385e6, 0, 0]), earth.velocity - np.array([0, math.sqrt(mu_earth/385e6), 0]))
satellite = Satellite("Satellite", mass_sat, earth.position + np.array([h0+r_earth, 0, 0]), earth.velocity + np.array([0, math.sqrt(mu_earth/(h0+r_earth)), 0]),3/2*l_sat**2, 1.67)

sim = Simulation({'Earth': earth, 'Moon': moon, 'Sun' : sun}, satellite, dt, t_speed)
sim.run()

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
plt.show()
