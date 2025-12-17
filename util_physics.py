import numpy as np
import arcade
from util_functions import color_from_charge
import math
from constants import meter, g, energy_loss_edge, H, W

class Force(np.ndarray):
    def __new__(cls, input_array, pos=None):
        obj = np.asarray(input_array).view(cls)
        obj.pos = np.asarray([0.0, 0.0], dtype=float) if pos is None else np.asarray(pos, dtype=float)
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.pos = getattr(obj, 'pos', None)

class Ball:
    def __init__(self, x:float , y:float, r:float, mass:float=1, airresistance:float=0.01, fixed:bool=False, charge:float=0, gravity:bool=True):
        """ initialise ball object
        x, y : position in meters
        r : radius in pixels
        mass : mass in kg
        airresistance : coefficient of air resistance (0-1)
        fixed : if True, ball does not move
        charge : electric charge in Coulombs
        gravity : if True, ball is affected by gravity
        """
        self.pos = np.array([x, y], dtype=float) 
        self.r = r 
        self.drag = False
        self.mass = mass
        self.v = np.array([0.0, 0.0])
        self.acc = np.array([0.0, 0.0])
        self.velcoeff = 1 - airresistance
        self.fixed = fixed
        self.charge = charge
        self.gravity=gravity

    def draw(self):
        arcade.draw_circle_filled(self.pos[0]*meter, self.pos[1]*meter, self.r, color_from_charge(self.charge))

    def update(self, dt, forces=[Force([0.0, 0.0])]):
        if not self.drag and not self.fixed:
            if self.gravity:
                self.acc = g + np.sum(forces, axis=0) / self.mass
            else:
                self.acc = np.sum(forces, axis=0) / self.mass
            self.pos += self.v*dt + 0.5 * self.acc * dt**2
            self.v += self.acc * dt * self.velcoeff
            self.do_collisions(energy_loss_edge)
    
    def do_collisions(self, s):
        loss = math.sqrt(s)
        p_pos = self.pos * meter

        # X-axis collisions
        if p_pos[0] < self.r:
            self.pos[0] = (self.r)/meter
            self.v[0] *= -loss
        elif p_pos[0] > W - self.r:
            self.pos[0] = (W - self.r)/meter
            self.v[0] *= -loss

        # Y-axis collisions
        if p_pos[1] < self.r:
            self.pos[1] = self.r/meter
            self.v[1] *= -loss
        elif p_pos[1] > H - self.r:
            self.pos[1] = (H - self.r)/meter
            self.v[1] *= -loss

    def check_hit(self, x, y):
        return (x - self.pos[0])**2 + (y - self.pos[1])**2   <= (self.r/meter)**2