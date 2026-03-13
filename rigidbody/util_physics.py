import numpy as np
import arcade
from util_drawing import color_from_charge
import math
from constants import meter, g, energy_loss_ball_edge, H, W, trail_length

ball_edge_coeff_restitution = math.sqrt(energy_loss_ball_edge)

class Force(np.ndarray):
    def __new__(cls, input_array, pos=None):
        obj = np.asarray(input_array).view(cls)
        obj.pos = np.asarray([0.0, 0.0]) if pos is None else np.asarray(pos, dtype=float)
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.pos = getattr(obj, 'pos', None)

class Ball:
    def __init__(self, x:float , y:float, r:float, mass:float=1, airresistance:float=0.01, fixed:bool=False, charge:float=0, gravity:bool=True, leaves_trail:bool=False):
        """ initialise ball object
        x, y : position in meters
        r : radius in pixels
        mass : mass in kg
        airresistance : coefficient of air resistance (0-1)
        fixed : if True, ball does not move
        charge : electric charge in Coulombs
        gravity : if True, ball is affected by gravity
        trail : if True, ball leaves a trail behind
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
        self.leaves_trail = leaves_trail
        self.trail = []

    def draw(self):
        if self.leaves_trail:
            n = 1
            for point in self.trail:
                alpha = np.interp(n, [1, len(self.trail)], [50, 255])
                color = color_from_charge(self.charge)
                arcade.draw_circle_filled(point[0]*meter, point[1]*meter, 2, (color[0], color[1], color[2], round(alpha)))
                n += 1

        arcade.draw_circle_filled(self.pos[0]*meter, self.pos[1]*meter, self.r, color_from_charge(self.charge))

    def update(self, dt, forces=[Force([0.0, 0.0])]):
        # update ball position and velocity using kinematic equations
        if not self.drag and not self.fixed:
            self.acc = np.sum(forces, axis=0) / self.mass
            if self.gravity:
                self.acc += g
            self.pos += self.v*dt + 0.5 * self.acc * dt**2
            self.v += self.acc * dt * self.velcoeff
            self.do_edge_collisions(ball_edge_coeff_restitution)

        if self.leaves_trail:
            if len(self.trail) < trail_length:
                self.trail.append(self.pos.copy())
            else:
                self.trail.pop(0)
                self.trail.append(self.pos.copy())
    
    def collide_normal(self, normal, rcoeff, pos_correction=0.0):
        normal = np.array(normal, dtype=float)
        mag = np.linalg.norm(normal)
        if mag < 1e-8:
            return

        n = normal / mag

        v_rel = np.dot(self.v, n)
        if v_rel < 0:
            self.v = self.v - (1 + rcoeff) * v_rel * n

        if pos_correction != 0:
            self.pos += (n * pos_correction) / meter

    def do_edge_collisions(self, s):
        p_pos = self.pos * meter

        if p_pos[0] < self.r:
            self.pos[0] = self.r / meter
            self.collide_normal([1, 0], s)

        elif p_pos[0] > W - self.r:
            self.pos[0] = (W - self.r) / meter
            self.collide_normal([-1, 0], s)

        if p_pos[1] < self.r:
            self.pos[1] = self.r / meter
            self.collide_normal([0, 1], s)

        elif p_pos[1] > H - self.r:
            self.pos[1] = (H - self.r) / meter
            self.collide_normal([0, -1], s)

    def check_hit(self, x, y):
        return (x - self.pos[0])**2 + (y - self.pos[1])**2   <= (self.r/meter)**2
    
    def occupied_pixels(self):
        pixels = []

        cx = int(self.pos[0] * meter)
        cy = int(self.pos[1] * meter)
        r = int(self.r)

        r2 = r * r

        for x in range(cx - r, cx + r + 1):
            dx2 = (x - cx) * (x - cx)
            for y in range(cy - r, cy + r + 1):
                dy = y - cy
                if dx2 + dy*dy <= r2:
                    pixels.append((x, y))

        return pixels