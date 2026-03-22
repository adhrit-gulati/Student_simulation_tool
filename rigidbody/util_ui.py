import arcade
import arcade.gui

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

class Navbar_ui(arcade.gui.UIBoxLayout):
    def __init__(self, height, size_hint, game):
        super().__init__(
            vertical=False,
            size_hint=size_hint,
            height=height
        )
        self.game = game

        text_style = {
            "font_size": 14,
            "font_color": arcade.color.WHITE,
            "bg": None,
            "border": None,
        }

        hover_style = {
            "font_size": 14,
            "font_color": arcade.color.LIGHT_GRAY,
            "bg": None,
            "border": None,
        }

        press_style = {
            "font_size": 14,
            "font_color": arcade.color.GRAY,
            "bg": None,
            "border": None,
        }

        style = {
            "normal": text_style,
            "hover": hover_style,
            "press": press_style,
        }

        self.space_between = 10
        self.padding = (10, 5, 10, 5)
        btn_height = height

        # --- Home button ---
        home_btn = arcade.gui.UIFlatButton(text="Home", width=100, height=btn_height)
        home_btn.style = style
        @home_btn.event("on_click")
        def on_home_click(event):
            self.on_home()

        self.add(home_btn)

        # --- Pause / Play toggle ---
        self.pause_btn = arcade.gui.UIFlatButton(text="Pause", width=120, height=btn_height, )
        self.pause_btn.style = style
        @self.pause_btn.event("on_click")
        def on_pause_click(event):
            self.on_pause_toggle()

        self.add(self.pause_btn)

        # --- Clear button ---
        clear_btn = arcade.gui.UIFlatButton(text="Clear", width=100, height=btn_height)
        clear_btn.style = style
        @clear_btn.event("on_click")
        def on_clear_click(event):
            self.on_clear()

        self.add(clear_btn)

        # --- Simulation settings ---
        settings_btn = arcade.gui.UIFlatButton(text="Settings", width=120, height=btn_height)
        settings_btn.style = style
        @settings_btn.event("on_click")
        def on_settings_click(event):
            self.on_settings()

        self.add(settings_btn)

    # --------handlers --------

    def on_home(self):
        pass
        #when homescreen is added this will get something to do

    def on_pause_toggle(self):
        self.game.pause_sim_toggle()
        if self.game.pause:
            self.pause_btn.text = "Play"
        else:
            self.pause_btn.text = "Pause"

    def on_clear(self):
        self.game.clear()

    def on_settings(self):
        self.game.simulation_edit()


class Sidebar_ui(arcade.gui.UIGridLayout):
    def __init__(self, width):
        super().__init__(
            column_count=1,
            row_count=5,
            size_hint=(None, 1.0),
            width=width
        )

        self.padding = (5, 5, 5, 5)
        self.vertical_spacing = 5

        for i in range(5):
            self.add(
                arcade.gui.UILabel(
                    text=f"Tool {i+1}",
                    text_color=arcade.color.WHITE
                ),
                column=0,
                row=i
            )