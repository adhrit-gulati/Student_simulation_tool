import arcade
import numpy as np
import math
import time
import arcade.key
import arcade.gui
from util_functions import sigmoid_color, create_arrow_texture, color_from_charge
from PIL import Image, ImageDraw
from constants import field_grid_spacing, W, H, meter, g, k, energy_loss_edge, energy_loss_ball
from util_physics import Ball, Rod, Force

#create textures
arrow_tex = create_arrow_texture()
off_tex = arcade.load_texture("off.png")
on_tex = arcade.load_texture("on.png")

class Simulation_edit_ui(arcade.gui.UIBoxLayout):
    def __init__(self, game):
        super().__init__(align="right")

        #gravity checkbox
        b1 = arcade.gui.UIBoxLayout(vertical=False)
        glabel = arcade.gui.UILabel(text="Gravity:", font_size=20, text_color=arcade.color.WHITE)
        gravity_checkbox = arcade.gui.UITextureToggle(value=game.gravity_enabled, width=28, height=32, on_texture=on_tex, off_texture=off_tex)
        @gravity_checkbox.event("on_change")
        def on_change(event):
            game.gravity_enabled = gravity_checkbox.value
        b1.add(glabel)
        b1.add(gravity_checkbox)
        self.add(b1)

        # Electrostatic force checkbox
        b2 = arcade.gui.UIBoxLayout(vertical=False)
        clabel = arcade.gui.UILabel(text="Electrostatic force:", font_size=20, text_color=arcade.color.WHITE)
        coulomb_checkbox = arcade.gui.UITextureToggle(value=game.coulomb_enabled, on_texture=on_tex, off_texture=off_tex, width=28, height=32)
        @coulomb_checkbox.event("on_change")
        def on_change(event):
            game.coulomb_enabled = coulomb_checkbox.value
        b2.add(clabel)
        b2.add(coulomb_checkbox)
        self.add(b2)

        # electric field visualisation checkbox
        b3 = arcade.gui.UIBoxLayout(vertical=False)
        eflabel = arcade.gui.UILabel(text="See Electric field:", font_size=20, text_color=arcade.color.WHITE)
        field_checkbox = arcade.gui.UITextureToggle(value=game.visualize_electric_field, on_texture=on_tex, off_texture=off_tex, width=28, height=32)
        @field_checkbox.event("on_change")
        def on_change(event):
            game.visualize_electric_field = field_checkbox.value
        b3.add(eflabel)
        b3.add(field_checkbox)
        self.add(b3)
class Ball_edit_ui(arcade.gui.UIBoxLayout):
    def __init__(self, ball):
        super().__init__(align="left")

        # charge label
        b1 = arcade.gui.UIBoxLayout(vertical=False)
        b1.add(arcade.gui.UILabel(
            text = f"Charge:",
            width=300, text_color=arcade.color.WHITE, bold=True, font_name="roboto"
        ))
        # charge input box
        self.charge_input = arcade.gui.UIInputText(text=str(ball.charge*1e6),width=40, text_color=arcade.color.CYAN, border_width=0)
        self.charge_input.with_padding(top=5)
        b1.add(self.charge_input)
        # charge unit label
        b1.add(arcade.gui.UILabel(
            text = " μC", text_color=arcade.color.WHITE, bold=True, font_name="roboto"))
        
        # charge change function
        @self.charge_input.event("on_change")
        def on_charge_change(event):
            try:
                val = float(self.charge_input.text)
                ball.charge = val * 1e-6
            except ValueError:
                pass
        self.add(b1)


        # mass label
        b2 = arcade.gui.UIBoxLayout(vertical=False)
        b2.add(arcade.gui.UILabel(
            text="Mass:",
            width=300,
            text_color=arcade.color.WHITE,
            bold=True,
            font_name="roboto"
        ))

        # mass input box
        self.mass_input = arcade.gui.UIInputText(
            text=str(ball.mass),
            width=40,
            text_color=arcade.color.CYAN,
            border_width=0
        )
        self.mass_input.with_padding(top=5)
        b2.add(self.mass_input)

        # mass unit label
        b2.add(arcade.gui.UILabel(
            text=" kg",
            text_color=arcade.color.WHITE,
            bold=True,
            font_name="roboto"
        ))

        # mass change function
        @self.mass_input.event("on_change")
        def on_mass_change(event):
            try:
                assert float(self.mass_input.text) != 0.0
                val = float(self.mass_input.text)
                ball.mass = val
                #linearly scale radius by mass
                ball.r = 20*val/9 + 70/9 
            except ValueError or AssertionError:
                pass
        self.add(b2)

        # trail checkbox
        b3 = arcade.gui.UIBoxLayout(vertical=False)
        tlabel = arcade.gui.UILabel(text="Trail:", bold=True, font_name="roboto", text_color=arcade.color.WHITE)
        tcheckbox = arcade.gui.UITextureToggle(value=ball.leaves_trail, on_texture=on_tex, off_texture=off_tex, width=28, height=32)
        @tcheckbox.event("on_change")
        def on_change(event):
            ball.leaves_trail = tcheckbox.value
        b3.add(tlabel)
        b3.add(tcheckbox)
        self.add(b3)

        # Display position, velocity, acceleration
        self.add(arcade.gui.UILabel(
            text=f"Position: ({ball.pos[0]:.2f}, {ball.pos[1]:.2f}) m",
            width=300, text_color=arcade.color.WHITE, bold=True, font_name="roboto"
        ))
        self.add(arcade.gui.UILabel(
            text=f"Velocity: ({ball.v[0]:.2f}, {ball.v[1]:.2f}) m/s",
            width=300, text_color=arcade.color.WHITE, bold=True, font_name="helvetica"
        ))
        self.add(arcade.gui.UILabel(
            text=f"Acceleration: ({ball.acc[0]:.2f}, {ball.acc[1]:.2f}) m/s²",
            width=300, text_color=arcade.color.WHITE, bold=True, font_name="helvetica"
        ))


class Game(arcade.Window):
    def __init__(self):
        super().__init__(W, H, "Drag", antialiasing=True)
        arcade.set_background_color(arcade.color.BLACK)
        self.balls = [
            Ball((W//4)/meter, (H//2)/meter, 10, charge= -10e-6, leaves_trail=True),
            Ball((3*W//4)/meter, (H//2)/meter, 10, charge= 10e-6, leaves_trail=True)
        ]
        self.rods = []
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
        for ball in self.balls:
            ball.draw()
        for rod in self.rods:
            rod.draw()
        if self.UI:
            rect = self.UI.rect
            padding = 12
            l = rect.x-rect.width/2
            b = rect.y-rect.height/2
            arcade.draw_lbwh_rectangle_filled(l-padding, b-padding, rect.width+2*padding, rect.height+2*padding, (10, 40, 100, 200))
            self.ui.draw()

    def on_update(self, dt):
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
                        # vector from other ball to this ball (meters)
                        r = ball.pos - other.pos
                        # magnitude of vector
                        m_r = np.linalg.norm(r)

                        #normal vector from other ball to this ball
                        if m_r == 0:
                            n = np.array([0.0, 0.0])
                        else:
                            n = r / m_r

                        # if collission of balls
                        if m_r < (ball.r + other.r)/meter and m_r > 0:
                            #relative velocity along normal
                            v_rel = ball.v - other.v
                            v_rel_n = np.dot(v_rel, n)
                            
                            #if balls are moving towards each other
                            if v_rel_n < 0:
                                #apply collision
                                j = -(1 + math.sqrt(energy_loss_ball)) * v_rel_n / (1/ball.mass + 1/other.mass)
                                
                                ball.v += j * n / ball.mass
                                other.v -= j * n / ball.mass
                                
                                #fix positional overlap of balls
                                overlap = (ball.r + other.r)/meter - m_r
                                ball.pos += n * (overlap * 0.5)
                                other.pos -= n * (overlap * 0.5)

                        # apply coulomb force
                        if self.coulomb_enabled:
                            forces.append((Force(k * (ball.charge * other.charge) * r) / m_r**3) if m_r > 1e-5 else Force([0.0, 0.0]))
                
                force_array.append(forces)

            # update all balls
            for i, ball in enumerate(self.balls):
                ball.update(dt, force_array[i])

            #update all rods
            for rod in self.rods:
                rod.update(dt)

            #initialise charges for electric field
            if self.visualize_electric_field:
                self.charges = np.array(self.charges)

    def on_mouse_press(self, x, y, b, m):
        shift = m & arcade.key.MOD_SHIFT
        ctrl = m & arcade.key.MOD_CTRL

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

    def on_mouse_release(self, x, y, b, m):
        # undrag the ball
        if self.dragged_ball:
            self.dragged_ball.drag = False
            self.dragged_ball = None

    def on_mouse_motion(self, x, y, dx, dy):
        #todrag the ball

        #get change in time
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
window = Game()
arcade.run()