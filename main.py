import arcade
import numpy as np
import math
import time
import arcade.key
import arcade.gui

from PIL import Image, ImageDraw

def create_arrow_texture(w=40, h=5):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    cx = w // 2      # center x (base of arrow)
    cy = h // 2      # center y

    shaft_len = w * 0.4    # length to the right of center
    head_len  = w * 0.1

    # Shaft: from center to near the right
    d.rectangle(
        [ (cx, cy - h*0.15), (cx + shaft_len, cy + h*0.15) ],
        fill="white"
    )

    # Arrow head
    d.polygon(
        [
            (cx + shaft_len, cy - h*0.4),
            (cx + shaft_len + head_len, cy),
            (cx + shaft_len, cy + h*0.4)
        ],
        fill="white"
    )

    return arcade.Texture(img)


arrow_tex = create_arrow_texture()

off_tex = arcade.load_texture("off.png")
on_tex = arcade.load_texture("on.png")

field_grid_spacing = 50

def sigmoid_color(q, kc=0.5):
    s = 1 / (1 + np.exp(-kc * q))
    return int(s * 255)

W, H = 800, 600

meter = 50  # 50 pixels = 1 meter
g = np.array([0.0, -9.81])  # m/s^2
k = 8.98e9  # N m^2 C^-2

energy_loss_edge = 0.8  # energy loss on edge collisions
energy_loss_ball = 0.5 # energy loss on ball collisions
energy_loss_edge_rod = 0.4

def color_from_charge(q, kc=0.5):
    # Map charge to red and blue
    R = int(255 / (1 + np.exp(-kc * q * 1e6)))   # positive charges → more red
    B = int(255 / (1 + np.exp(-kc * -q * 1e6)))  # negative charges → more blue
    
    # Green depends on closeness to 128
    G = int(128 * (1 - (abs(R-128) + abs(B-128)) / 255))
    
    # Clamp values to 0-255
    G = max(0, min(255, G))
    
    return (R, G, B)

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

class Game(arcade.Window):
    def __init__(self):
        super().__init__(W, H, "Drag", antialiasing=True)
        arcade.set_background_color(arcade.color.BLACK)
        self.balls = [
            Ball((W//4)/meter, (H//2)/meter, 10, charge= -10e-6),
            Ball((W//2)/meter, (H//2)/meter, 10),
            Ball((3*W//4)/meter, (H//2)/meter, 10, charge= 10e-6)
        ]
        self.dragged_ball = None
        self.pause = False
        self.ui = arcade.gui.UIManager()
        self.ui.enable()
        self.show_box = False
        self.gravity_enabled = True
        self.coulomb_enabled = True
        self.visualize_electric_field = False
        self.arrow_list = arcade.SpriteList(use_spatial_hash=False)

    def on_draw(self):
        arcade.Window.clear(self)
        self.arrow_list.clear()
        if self.visualize_electric_field:
            for ygrid in range(round(H/field_grid_spacing)+1):
                for xgrid in range(round(W/field_grid_spacing)+1):
                    pos = np.array([xgrid*field_grid_spacing, ygrid*field_grid_spacing])/meter
                    E=0
                    for charge in self.charges:
                        E += (k*charge[2]*(pos-charge[:2]))/(np.linalg.norm(pos-charge[:2])**3)
                    start = pos * meter
                    end = (pos+(E*1e-3))*meter
                    mag = np.linalg.norm(end-start)
                    maxmag = field_grid_spacing
                    s = sigmoid_color(np.linalg.norm(E)-10000, 0.0005)
                    if mag>maxmag:
                        vec = ((end-start)/mag)*(maxmag+np.log(mag/maxmag)) / 1.6
                        sprite = arcade.Sprite(center_x=start[0], center_y=start[1])
                        sprite.texture = arrow_tex
                        sprite.width = np.linalg.norm(vec)*2
                        sprite.height = sprite.texture.height
                        sprite.angle=np.degrees(np.arctan2(vec[0], vec[1]))-90
                        sprite.color = (s,180,255-s, 100)
                        self.arrow_list.append(sprite)
                    else:
                        vec = (end-start)/1.6
                        sprite = arcade.Sprite(center_x=start[0], center_y=start[1])
                        sprite.texture = arrow_tex
                        sprite.width = np.linalg.norm(vec)*2
                        sprite.height = sprite.texture.height
                        sprite.angle=np.degrees(np.arctan2(vec[0], vec[1]))-90
                        sprite.color = (s,180,255-s, 100)
                        self.arrow_list.append(sprite)
            self.arrow_list.draw()
        for ball in self.balls:
            ball.draw()
        if self.show_box:
            arcade.draw_lbwh_rectangle_filled(0, H - 120, 260, 280, (10, 67, 108, 160))
            self.ui.draw()

    def on_update(self, dt):
        if self.visualize_electric_field:
            self.charges = []
        if not self.pause:
            force_array = []
            for ball in self.balls:
                ball.gravity = self.gravity_enabled
                forces = [Force([0.0, 0.0])]
                if ball.charge != 0 and self.visualize_electric_field:
                    self.charges.append([ball.pos[0], ball.pos[1], ball.charge])
                for other in self.balls:
                    if other != ball:
                        e = 0.8  # coefficient of restitution
                        r = ball.pos - other.pos
                        m_r = np.linalg.norm(r)
                        if m_r == 0:  # avoid divide by zero
                            n = np.array([0.0, 0.0])
                        else:
                            n = r / m_r

                        if m_r < (ball.r + other.r)/meter and m_r > 0:
                            v_rel = ball.v - other.v
                            v_rel_n = np.dot(v_rel, n)
                            
                            if v_rel_n < 0: 
                                j = -(1 + e) * v_rel_n / (1/ball.mass + 1/other.mass)
                                
                                ball.v += j * n / ball.mass
                                other.v -= j * n / other.mass
                                
                                overlap = (ball.r + other.r)/meter - m_r
                                ball.pos += n * (overlap * 0.5)
                                other.pos -= n * (overlap * 0.5)
                        if self.coulomb_enabled:
                            forces.append((Force(k * (ball.charge * other.charge) * r) / m_r**3) if m_r > 1e-5 else Force([0.0, 0.0]))
                force_array.append(forces)

            for i, ball in enumerate(self.balls):
                ball.update(dt, force_array[i])
            if self.visualize_electric_field:
                self.charges = np.array(self.charges)

    def on_mouse_press(self, x, y, b, m):
        shift = m & arcade.key.MOD_SHIFT
        ctrl = m & arcade.key.MOD_CTRL

        for ball in self.balls:
            if ball.check_hit(x/meter, y/meter):
                if ctrl:
                    self.balls.remove(ball)
                    return
                if shift:
                    self.pause = True
                    self.ui.clear()
                    self.show_box = True

                    box = arcade.gui.UIBoxLayout(x=5, y=H-115, align="left")

                    b1 = arcade.gui.UIBoxLayout(vertical=False)
                    b1.add(arcade.gui.UILabel(
                        text = f"Charge:",
                        width=300, text_color=arcade.color.WHITE, bold=True, font_name="roboto"
                    ))
                    self.charge_input = arcade.gui.UIInputText(text=str(ball.charge*1e6),width=40, text_color=arcade.color.CYAN, border_width=0)
                    self.charge_input.with_padding(top=5)
                    b1.add(self.charge_input)
                    b1.add(arcade.gui.UILabel(
                        text = " μC", text_color=arcade.color.WHITE, bold=True, font_name="roboto"))
                    
                    @self.charge_input.event("on_change")
                    def on_charge_change(event):
                        try:
                            val = float(self.charge_input.text)
                            ball.charge = val * 1e-6
                        except ValueError:
                            pass  # ignore invalid input
                    box.add(b1)

                    b2 = arcade.gui.UIBoxLayout(vertical=False)
                    b2.add(arcade.gui.UILabel(
                        text="Mass:",
                        width=300,
                        text_color=arcade.color.WHITE,
                        bold=True,
                        font_name="roboto"
                    ))

                    self.mass_input = arcade.gui.UIInputText(
                        text=str(ball.mass),
                        width=40,
                        text_color=arcade.color.CYAN,
                        border_width=0
                    )
                    self.mass_input.with_padding(top=5)
                    b2.add(self.mass_input)

                    b2.add(arcade.gui.UILabel(
                        text=" kg",
                        text_color=arcade.color.WHITE,
                        bold=True,
                        font_name="roboto"
                    ))

                    @self.mass_input.event("on_change")
                    def on_mass_change(event):
                        try:
                            assert float(self.mass_input.text) != 0.0
                            val = float(self.mass_input.text)
                            ball.mass = val
                            ball.r = 20*val/9 + 70/9 #linearly scale radius by mass
                        except ValueError or AssertionError:
                            pass  # ignore invalid input

                    box.add(b2)
                    box.add(arcade.gui.UILabel(
                        text=f"Position: ({ball.pos[0]:.2f}, {ball.pos[1]:.2f}) m",
                        width=300, text_color=arcade.color.WHITE, bold=True, font_name="roboto"
                    ))
                    box.add(arcade.gui.UILabel(
                        text=f"Velocity: ({ball.v[0]:.2f}, {ball.v[1]:.2f}) m/s",
                        width=300, text_color=arcade.color.WHITE, bold=True, font_name="helvetica"
                    ))
                    box.add(arcade.gui.UILabel(
                        text=f"Acceleration: ({ball.acc[0]:.2f}, {ball.acc[1]:.2f}) m/s²",
                        width=300, text_color=arcade.color.WHITE, bold=True, font_name="helvetica"
                    ))

                    # anchor the info box near the top-left
                    self.ui.add(box)

                    return

                else:
                    ball.drag = True
                    ball.v = np.array([0.0, 0.0])
                    self.dragged_ball = ball
                    return    
        if ctrl:
            self.balls.append(Ball(x/meter, y/meter, 10))
        elif shift:
            self.pause = True
            self.ui.clear()
            self.show_box = True

            box = arcade.gui.UIBoxLayout(x=5, y=H - 115, align="left")

            b1 = arcade.gui.UIBoxLayout(vertical=False)
            glabel = arcade.gui.UILabel(text="Gravity:", font_size=20, text_color=arcade.color.WHITE)
            gravity_checkbox = arcade.gui.UITextureToggle(value=self.gravity_enabled, width=28, height=32, on_texture=on_tex, off_texture=off_tex)
            @gravity_checkbox.event("on_change")
            def on_change(event):
                self.gravity_enabled = gravity_checkbox.value
            b1.add(glabel)
            b1.add(gravity_checkbox)
            box.add(b1)

            b2 = arcade.gui.UIBoxLayout(vertical=False)
            clabel = arcade.gui.UILabel(text="Electrostatic force:", font_size=20, text_color=arcade.color.WHITE)
            coulomb_checkbox = arcade.gui.UITextureToggle(value=self.coulomb_enabled, on_texture=on_tex, off_texture=off_tex, width=28, height=32)
            @coulomb_checkbox.event("on_change")
            def on_change(event):
                self.coulomb_enabled = coulomb_checkbox.value
            b2.add(clabel)
            b2.add(coulomb_checkbox)
            box.add(b2)

            b3 = arcade.gui.UIBoxLayout(vertical=False)
            eflabel = arcade.gui.UILabel(text="See Electric field:", font_size=20, text_color=arcade.color.WHITE)
            field_checkbox = arcade.gui.UITextureToggle(value=self.visualize_electric_field, on_texture=on_tex, off_texture=off_tex, width=28, height=32)
            @field_checkbox.event("on_change")
            def on_change(event):
                self.visualize_electric_field = field_checkbox.value
            b3.add(eflabel)
            b3.add(field_checkbox)
            box.add(b3)

            self.ui.add(box)

    def on_mouse_release(self, x, y, b, m):
        if self.dragged_ball:
            self.dragged_ball.drag = False
            self.dragged_ball = None

    def on_mouse_motion(self, x, y, dx, dy):
        tnow = time.perf_counter()
        tlast = getattr(self, 'last_time', tnow)
        if self.dragged_ball:
            self.dragged_ball.pos[0] = x/meter
            self.dragged_ball.pos[1] = y/meter
            dt = tnow - tlast
            if dt > 0:
                self.dragged_ball.v = np.array([(dx/dt)/meter, (dy/dt)/meter])
        self.last_time = tnow

    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE:
            self.pause = not self.pause
            self.show_box = False

window = Game()

arcade.run()
