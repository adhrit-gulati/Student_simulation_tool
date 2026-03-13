import arcade
import numpy as np
import math
import time
import arcade.key
import arcade.gui
from util_drawing import sigmoid_color, create_arrow_texture
from constants import field_grid_spacing, W, H, meter, g, k, energy_loss_ball_ball, barrier_draw_radius, barrier_remove_radius, energy_loss_ball_barrier, number_substeps
from util_physics import Ball, Force
from util_ui import Ball_edit_ui, Simulation_edit_ui

#create textures
arrow_tex = create_arrow_texture() 

ball_ball_coeff_restitution = math.sqrt(energy_loss_ball_ball)
ball_barrier_coeff_restitution = math.sqrt(energy_loss_ball_barrier)

#class CollissionBox
class Game(arcade.Window):
    def __init__(self):
        super().__init__(W, H, "Drag", antialiasing=True)
        arcade.set_background_color(arcade.color.BLACK)
        self.balls = [
            Ball((W//4)/meter, (H//2)/meter, 10, charge= -10e-6, leaves_trail=True),
            Ball((3*W//4)/meter, (H//2)/meter, 10, charge= 10e-6, leaves_trail=True)
        ]
        self.dragged_ball = None
        self.pause = False
        self.ui = arcade.gui.UIManager()
        self.ui.enable()
        self.show_box = False
        self.UI = None
        self.gravity_enabled = True
        self.coulomb_enabled = True
        self.visualize_electric_field = False
        self.arrow_list = arcade.SpriteList(use_spatial_hash=False)
        self.barrier_pixels = set()
        self.making_barrier = False
        self.removing_barrier = False
        for ygrid in range(round(H/field_grid_spacing)+1):
            for xgrid in range(round(W/field_grid_spacing)+1):
                sprite = arcade.Sprite(center_x=xgrid*field_grid_spacing, center_y=ygrid*field_grid_spacing)
                sprite.texture = arrow_tex
                sprite.height = sprite.texture.height
                self.arrow_list.append(sprite)

    def on_draw(self):
        arcade.Window.clear(self)
        if self.visualize_electric_field:
            self.draw_electric_field()
        arcade.draw_points(self.barrier_pixels, (150,150,150), 1)
        for ball in self.balls:
            ball.draw()
        if self.UI:
            rect = self.UI.rect
            padding = 12
            l = rect.x-rect.width/2
            b = rect.y-rect.height/2
            arcade.draw_lbwh_rectangle_filled(l-padding, b-padding, rect.width+2*padding, rect.height+2*padding, (10, 40, 100, 200))
            self.ui.draw()

    def on_update(self, dt):
        dt = dt/number_substeps
        for i in range(number_substeps):
            if not self.pause:
                force_array = []
                self.charges = []

                #loop though all balls
                for ball in self.balls:
                    ball.gravity = self.gravity_enabled
                    forces = [Force([0.0, 0.0])]
                    if ball.charge != 0 and self.visualize_electric_field:
                        self.charges.append([ball.pos[0], ball.pos[1], ball.charge])

                    # ball- ball interactions
                    for other in self.balls:
                        if other != ball:
                            r = ball.pos - other.pos
                            if not (abs(ball.pos[0] - other.pos[0]) > (ball.r + other.r)/meter or abs(ball.pos[1] - other.pos[1]) > (ball.r + other.r)/meter):
                                m_r2 = r[0]**2 + r[1]**2

                                if m_r2 == 0:
                                    continue

                                # collision check
                                if m_r2 < ((ball.r + other.r)/meter)**2:
                                    m_r = math.sqrt(m_r2)
                                    overlap = (ball.r + other.r) - (m_r * meter)
                                    n = r / m_r
                                    v_rel = ball.v - other.v
                                    v_rel_n = np.dot(v_rel, n)

                                    if v_rel_n < 0:
                                        j = -(1 + ball_ball_coeff_restitution) * v_rel_n / (1/ball.mass + 1/other.mass)

                                        ball.v += j * n / ball.mass
                                        other.v -= j * n / other.mass

                                    overlap = ((ball.r + other.r) / meter) - m_r
                                    if overlap > 0:
                                        ball.pos += n * (overlap * 0.5)
                                        other.pos -= n * (overlap * 0.5)
                            # apply coulomb force
                            if self.coulomb_enabled:
                                if not (ball.charge == 0 or other.charge == 0):
                                    m_r = np.linalg.norm(r)
                                    forces.append((Force(k * (ball.charge * other.charge) * r) / (m_r)**3) if m_r > 1e-5 else Force([0.0, 0.0]))
                        
                    # ball - barrier interactions
                    nearest_pixel = None
                    min_d2 = float("inf")

                    for p in ball.occupied_pixels():
                        if p in self.barrier_pixels:
                            dx = ball.pos[0] - p[0]/meter
                            dy = ball.pos[1] - p[1]/meter
                            d2 = dx*dx + dy*dy

                            if d2 < min_d2:
                                min_d2 = d2
                                nearest_pixel = p

                    if nearest_pixel:
                        r = ball.pos - np.array(nearest_pixel)/meter
                        dist = math.sqrt(min_d2) * meter
                        overlap = ball.r - dist

                        if overlap > 0:
                            ball.collide_normal(r, ball_barrier_coeff_restitution, overlap)
                    
                    
                    force_array.append(forces)

                # update all balls
                for i, ball in enumerate(self.balls):
                    ball.update(dt, force_array[i])

                #initialise charges for electric field
                if self.visualize_electric_field:
                    self.charges = np.array(self.charges)

    def on_mouse_press(self, x, y, b, m):
        shift = m & arcade.key.MOD_SHIFT
        ctrl = m & arcade.key.MOD_CTRL

        if b == arcade.MOUSE_BUTTON_LEFT:
            for ball in self.balls:
                #check if ball is clicked
                if ball.check_hit(x/meter, y/meter):
                    if ctrl:
                        #ctrl + click deletes a ball
                        self.delete_ball(ball)
                        return
                    if shift:
                        # shift+click opens properties of ball
                        self.pause = True
                        self.ball_edit(ball)
                        return
                    else:
                        #drag the ball
                        ball.drag = True
                        ball.v = np.array([0.0, 0.0])
                        self.dragged_ball = ball
                        return  
            #if no ball is clicked
            if ctrl:
                #ctrl + click adds a new ball
                self.balls.append(Ball(x/meter, y/meter, 10))
            elif shift:
                #opens simulation properties
                self.pause = True
                self.simulation_edit()
        elif b == arcade.MOUSE_BUTTON_RIGHT:
            if shift:
                self.remove_barrier(x, y, barrier_draw_radius)
                self.removing_barrier = True
            else:
                self.make_barrier(x, y, barrier_draw_radius)
                self.making_barrier = True

    def on_mouse_release(self, x, y, b, m):
        if self.dragged_ball:
            self.dragged_ball.drag = False
            self.dragged_ball = None
        self.making_barrier = False
        self.removing_barrier = False

    def on_mouse_motion(self, x, y, dx, dy):
        if self.making_barrier or self.removing_barrier:
            r = barrier_draw_radius if self.making_barrier else barrier_remove_radius

            step_measure = max(abs(dx), abs(dy))

            if step_measure <= r/2:
                if self.making_barrier:
                    self.make_barrier(x, y, r)
                else:
                    self.remove_barrier(x, y, r)
                return

            steps = int(step_measure // (r/2)) + 1
            x0 = x - dx
            y0 = y - dy
            for i in range(steps + 1):
                t = i / steps
                xi = x0 + dx * t
                yi = y0 + dy * t

                if self.making_barrier:
                    self.make_barrier(xi, yi, r)
                else:
                    self.remove_barrier(xi, yi, r)


        tnow = time.perf_counter()
        tlast = getattr(self, 'last_time', tnow)
        dt = tnow - tlast
        if self.dragged_ball:
            # ball must follow mouse
            self.dragged_ball.pos[0] = x/meter
            self.dragged_ball.pos[1] = y/meter
            #update velocity of the ball
            if dt > 0:
                self.dragged_ball.v = np.array([(dx/dt)/meter, (dy/dt)/meter])

        #update time
        self.last_time = tnow

    def on_key_press(self, key, modifiers):
        #toggle pause/resume of the simulation, and hide the UI box
        if key == arcade.key.SPACE:
            self.pause = not self.pause
            self.UI = None

    def draw_electric_field(self):
        for i, sprite in enumerate(self.arrow_list):
                pos = np.array([sprite.center_x, sprite.center_y]) / meter #position of one point (unit-meters)

                #electric field at point
                E=np.array([0.0, 0.0])
                for charge in self.charges:
                    if np.linalg.norm(pos-charge[:2]) > 1e-5:
                        E += (k*charge[2]*(pos-charge[:2]))/(np.linalg.norm(pos-charge[:2])**3)

                #Creating field vector (unit- pixels)
                start = pos * meter
                end = (pos+(E*1e-3))*meter
                mag = np.linalg.norm(end-start)
                maxmag = field_grid_spacing # arrows longer than this get scaled logarithmically
                s = sigmoid_color(np.linalg.norm(E)-10000, 0.0005)#tuned constants for colour (choosen empirically after trial and error)
                if mag>maxmag:
                    vec = ((end-start)/mag)*(maxmag+np.log(mag/maxmag)) / 1.6
                else:
                    vec = (end-start)/1.6

                #drawing vector
                sprite.width = np.linalg.norm(vec)*2
                sprite.angle=np.degrees(np.arctan2(vec[0], vec[1]))-90
                sprite.color = (s,180,255-s, 100)
        self.arrow_list.draw()

    def delete_ball(self, ball):
        self.balls.remove(ball)

    def ball_edit(self, ball):
        #create the UI box for editing ball properties
        self.ui.clear()

        self.show_box = True
        anchor = arcade.gui.UIAnchorLayout()
        box = Ball_edit_ui(ball)
        anchor.add(box, anchor_x="left", anchor_y="top", align_x=5, align_y=-15)
        self.UI = box
        self.ui.add(anchor)

    def simulation_edit(self):
        #create the UI box for editing simulation properties
        self.ui.clear()
        
        self.show_box = True
        anchor = arcade.gui.UIAnchorLayout()
        box = Simulation_edit_ui(self)
        anchor.add(box, anchor_x="left", anchor_y="top", align_x=5, align_y=-10)
        self.UI = box
        self.ui.add(anchor)

    def make_barrier(self, X, Y, r):
        X, Y, r = int(X), int(Y), int(r)
        for x in range(X-r, X+r+1):
            for y in range(Y-r, Y+r+1):
                ar = (x, y)
                if (np.sqrt((X-x)**2 + (Y-y)**2) <= r):
                    self.barrier_pixels.add(ar)

    def remove_barrier(self, X, Y, r):
        X, Y, r = int(X), int(Y), int(r)
        for x in range(X - r, X + r + 1):
            for y in range(Y - r, Y + r + 1):
                ar = (x, y)
                if np.sqrt((X - x)**2 + (Y - y)**2) <= r:
                    self.barrier_pixels.discard(ar)

window = Game()
arcade.run()