import numpy as np
import arcade
from util_functions import color_from_charge
import math
from constants import meter, g, energy_loss_edge, H, W

class Force(np.ndarray):
    def __new__(cls, input_array, pos=None):
        obj = np.asarray(input_array).view(cls)
        obj.pos = np.asarray([0.0, 0.0]) if pos is None else np.asarray(pos, dtype=float)
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
        # update ball position and velocity using kinematic equations
        if not self.drag and not self.fixed:
            self.acc = np.sum(forces, axis=0) / self.mass
            if self.gravity:
                self.acc += g
            self.pos += self.v*dt + 0.5 * self.acc * dt**2
            self.v += self.acc * dt * self.velcoeff
            self.do_edge_collisions(energy_loss_edge)
    
    def do_edge_collisions(self, s):
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
    
class Rod:
    def __init__(self, x:float, y:float, r:float, l:float=10, angle:float=0, mass:float=1, airresistance:float=0.01, fixed:bool=False, charge:float=0, gravity:bool=True, angle_damping=0.9):
        """ initialise ball object
        x, y : position in meters
        r : radius in pixels
        l : length in meters
        mass : mass in kg
        airresistance : coefficient of air resistance (0-1)
        fixed : if True, ball does not move
        charge : electric charge in Coulombs
        gravity : if True, ball is affected by gravity
        angle : angle in radians
        angle_damping : damping factor for angular velocity (0-1)
        """
        self.pos = np.array([x, y], dtype=float)
        self.r = r 
        self.l = l
        self.angle = angle
        self.angle_vel = 0
        self.angle_acc = 0
        self.angle_damping = angle_damping
        self.drag = False
        self.mass = mass
        self.v = np.array([0.0, 0.0])
        self.acc = np.array([0.0, 0.0])
        self.velcoeff = 1 - airresistance
        self.fixed = fixed
        self.charge = charge
        self.gravity=gravity

    def I(self):
        return (self.mass * self.l**2)/12

    def vec(self):
        """returns the half length vector from the center to one end of the rod"""
        return (self.l / 2)*np.array([np.cos(self.angle), np.sin(self.angle)])

    def draw(self):
        """draw the rod as a capsule"""
        vec = self.vec()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return

        perp = np.array([-vec[1], vec[0]]) / norm

        #draw center rectangle
        a = (self.pos + vec) * meter + perp * self.r
        b = (self.pos + vec) * meter - perp * self.r
        c = (self.pos - vec) * meter - perp * self.r
        d = (self.pos - vec) * meter + perp * self.r
        arcade.draw_polygon_filled([tuple(a), tuple(b), tuple(c), tuple(d)], color=color_from_charge(self.charge))

        #draw side circles
        p = (self.pos + vec) * meter
        q = (self.pos - vec) * meter
        arcade.draw_circle_filled(p[0], p[1], self.r, color=color_from_charge(self.charge))
        arcade.draw_circle_filled(q[0], q[1], self.r, color=color_from_charge(self.charge))

    def update(self, dt, forces=[Force([0.0, 0.0])]):
        # update rod position, velocity and angle based on kinematic equations
        if not self.drag and not self.fixed:
            net_force = np.sum(forces, axis=0)
            net_torque = 0.0
            for force in forces:
                net_torque += np.cross((force.pos-self.pos), force)

            self.acc = g + net_force/self.mass
            self.pos += self.v * dt + (self.acc*dt**2)/2
            self.v += self.acc * dt * self.velcoeff

            self.angle_acc = net_torque/ self.I()
            self.angle += self.angle_vel * dt + (self.angle_acc*dt**2)/2
            self.angle_vel += self.angle_acc*dt*self.angle_damping*self.velcoeff

            self.do_edge_collissions(energy_loss_edge)

    def to_local(self, point:np.ndarray):
        """
        Convert a point from global coordinates to rod-local coordinates.
        Local frame: origin at rod center, x-axis along rod, y-axis perpendicular.
        """
        # shift to rod center
        rel = point - self.pos

        # rotation matrix
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[ c,  s],
                    [-s,  c]])

        # multiply
        local = R @ rel
        return local
    
    def to_global(self, local_point: np.ndarray):
        """
        Convert a point from rod-local coordinates to global coordinates.
        Local frame: origin at rod center, x-axis along rod, y-axis perpendicular.
        """
        # rotation matrix for +angle
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s],
                    [s,  c]])

        # rotate and shift back to global origin
        global_point = R @ local_point + self.pos
        return global_point
    
    def do_edge_collissions(self, s):
        loss = math.sqrt(s)
        vec = self.vec()
        e1 = self.pos + vec
        e2 = self.pos - vec

        tol = 4

        contact_vel = None
        r = None
        col_normal = None

        # --- Left wall ---
        if e1[0] * meter < self.r or e2[0] * meter < self.r:
            penetration = self.r / meter - min(e1[0], e2[0])
            if penetration > 0:
                self.pos[0] += penetration

            dx = e1[0] * meter - e2[0] * meter
            if abs(dx) <= tol:
                r = np.array([0.0, 0.0])
                contact_vel = self.v
            elif dx < 0:
                r = vec
                contact_vel = self.v + self.angle_vel * np.array([-r[1], r[0]])
            else:
                r = -vec
                contact_vel = self.v + self.angle_vel * np.array([-r[1], r[0]])
            col_normal = np.array([1.0, 0.0])

        # --- Right wall ---
        elif e1[0] * meter > W - self.r or e2[0] * meter > W - self.r:
            penetration = max(e1[0], e2[0]) - (W - self.r)/meter
            if penetration > 0:
                self.pos[0] -= penetration

            dx = e1[0] * meter - e2[0] * meter
            if abs(dx) <= tol:
                r = np.array([0.0, 0.0])
                contact_vel = self.v
            elif dx < 0:
                r = vec
                contact_vel = self.v + self.angle_vel * np.array([-r[1], r[0]])
            else:
                r = -vec
                contact_vel = self.v + self.angle_vel * np.array([-r[1], r[0]])
            col_normal = np.array([-1.0, 0.0])

        # --- Floor ---
        elif e1[1] * meter < self.r or e2[1] * meter < self.r:
            penetration = self.r / meter - min(e1[1], e2[1])
            if penetration > 0:
                self.pos[1] += penetration

            dy = e1[1] * meter - e2[1] * meter
            if abs(dy) <= tol:
                r = np.array([0.0, 0.0])
                contact_vel = self.v
            elif dy < 0:
                r = vec
                contact_vel = self.v + self.angle_vel * np.array([-r[1], r[0]])
            else:
                r = -vec
                contact_vel = self.v + self.angle_vel * np.array([-r[1], r[0]])
            col_normal = np.array([0.0, 1.0])

        # --- Ceiling ---
        elif e1[1] * meter > H - self.r or e2[1] * meter > H - self.r:
            penetration = max(e1[1], e2[1]) - (H - self.r)/meter
            if penetration > 0:
                self.pos[1] -= penetration

            dy = e1[1] * meter - e2[1] * meter
            if abs(dy) <= tol:
                r = np.array([0.0, 0.0])
                contact_vel = self.v
            elif dy < 0:
                r = vec
                contact_vel = self.v + self.angle_vel * np.array([-r[1], r[0]])
            else:
                r = -vec
                contact_vel = self.v + self.angle_vel * np.array([-r[1], r[0]])
            col_normal = np.array([0.0, -1.0])

        if contact_vel is not None:
            reflect_vel = (contact_vel - 2 * np.dot(contact_vel, col_normal) * col_normal) * loss
            deltav = reflect_vel - contact_vel

            effective_mass = 1 / (1/self.mass + (np.cross(r, col_normal)**2)/self.I())

            J = deltav * effective_mass
            self.v += J / self.mass
            self.angle_vel += np.cross(r, J) / self.I()

