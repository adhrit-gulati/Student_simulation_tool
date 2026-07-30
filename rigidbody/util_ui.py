import arcade
import arcade.gui
from util_physics import Ball
import numpy as np

off_tex = arcade.load_texture("assets\off.png")
on_tex = arcade.load_texture("assets\on.png")

class ToolCard(arcade.gui.UIBoxLayout):
    def __init__(self, image_path, text, on_click=None):
        super().__init__(vertical=True, width=100, height=120)

        self.style = {
            "normal": {
                "font_size": 14,
                "font_color": arcade.color.WHITE,
                "bg": None,
                "border": None,
            },
            "hover": {
                "font_size": 14,
                "font_color": arcade.color.LIGHT_GRAY,
                "bg": None,
                "border": None,
            },
            "press": {
                "font_size": 14,
                "font_color": arcade.color.GRAY,
                "bg": None,
                "border": None,
            },
        }

        texture = arcade.load_texture(image_path)

        self.image = arcade.gui.UIImage(texture=texture, width=100, height=100)

        # Button replaces label AND handles click
        self.button = arcade.gui.UIFlatButton(
            text=text,
            width=100,
            height=30, 
            style=self.style
        )

        if on_click:
            self.button.on_click = lambda event: on_click()

        self.add(self.image)
        self.add(self.button)

class LabeledCheckbox(arcade.gui.UIBoxLayout):
    def __init__(self, text, value, callback):
        super().__init__(vertical=False)
        self.space_between = 10

        self.label = arcade.gui.UILabel(
            text=text,
            font_size=20,
            text_color=arcade.color.WHITE
        )

        self.toggle = arcade.gui.UITextureToggle(value=value, on_texture=on_tex, off_texture=off_tex, width=28, height=20)

        self.toggle.event("on_change")(lambda e: callback(self.toggle.value))

        self.add(self.label)
        self.add(self.toggle)

class LabeledInput(arcade.gui.UIBoxLayout):
    def __init__(self, text, value, unit, callback, input_width=50):
        super().__init__(vertical=False)
        self.space_between = 5

        self.add(arcade.gui.UILabel(
            text=text,
            width=300,
            text_color=arcade.color.WHITE,
            bold=True,
            font_name="roboto"
        ))

        self.input = arcade.gui.UIInputText(
            text=str(value),
            width=input_width,
            text_color=arcade.color.CYAN,
            border_width=0
        )
        self.input.with_padding(top=5)

        self.input.event("on_change")(lambda e: self._safe_update(callback))

        self.add(self.input)

        self.add(arcade.gui.UILabel(
            text=unit,
            text_color=arcade.color.WHITE,
            bold=True,
            font_name="roboto"
        ))

    def _safe_update(self, callback):
        try:
            callback(float(self.input.text))
        except ValueError:
            pass

class VectorInput(arcade.gui.UIBoxLayout):
    def __init__(self, text, value: np.ndarray, unit, callback=None):
        super().__init__(vertical=False) # Changed to False for one line
        self.space_between = 5

        self.value = value
        self.callback = callback

        self.add(arcade.gui.UILabel(
            text=text,
            width=100, # Reduced width
            text_color=arcade.color.WHITE,
            bold=True,
            font_name="roboto"
        ))

        x_y_layout = arcade.gui.UIBoxLayout(vertical=False, space_between=5)

        self.x_input = LabeledInput(
            text="(",
            value=f"{self.value[0]:.2f}",
            unit=",",
            callback=self._update_x,
            input_width=50
        )
        self.y_input = LabeledInput(
            text="",
            value=f"{self.value[1]:.2f}",
            unit=")",
            callback=self._update_y,
            input_width=50
        )
        x_y_layout.add(self.x_input)
        x_y_layout.add(self.y_input)
        x_y_layout.add(arcade.gui.UILabel(
            text=unit,
            text_color=arcade.color.WHITE,
            bold=True,
            font_name="roboto"
        ))

        self.add(x_y_layout)

    def _update_x(self, x_val):
        self.value[0] = x_val
        if self.callback:
            self.callback(self.value)

    def _update_y(self, y_val):
        self.value[1] = y_val
        if self.callback:
            self.callback(self.value)

class Simulation_edit_ui(arcade.gui.UIBoxLayout):
    def __init__(self, game):
        super().__init__(align="right")

        self.add(LabeledCheckbox(
            "Gravity:",
            game.gravity_enabled,
            lambda v: setattr(game, "gravity_enabled", v)
        ))

        self.add(LabeledCheckbox(
            "Electrostatic force:",
            game.coulomb_enabled,
            lambda v: setattr(game, "coulomb_enabled", v)
        ))

        self.add(LabeledCheckbox(
            "See Electric field:",
            game.visualize_electric_field,
            lambda v: setattr(game, "visualize_electric_field", v)
        ))

class Ball_edit_ui(arcade.gui.UIBoxLayout):
    def __init__(self, ball):
        super().__init__(align="left")

        # Info labels (position, velocity, acceleration)
        self.add(self._info_label("Position", ball.pos, "m"))
        self.add(self._info_label("Velocity", ball.v, "m/s"))
        
        # Acceleration as a regular label
        self.add(arcade.gui.UILabel(
            text=f"Acceleration: ({ball.acc[0]:.2f}, {ball.acc[1]:.2f}) m/s²",
            width=300,
            text_color=arcade.color.WHITE,
            bold=True,
            font_name="roboto"
        ))

        # Charge
        self.add(LabeledInput(
            "Charge:",
            ball.charge * 1e6,
            " μC",
            lambda v: setattr(ball, "charge", v * 1e-6)
        ))

        # Mass
        def update_mass(v):
            if v != 0:
                ball.mass = v
                ball.r = 20 * v / 9 + 70 / 9

        self.add(LabeledInput(
            "Mass:",
            ball.mass,
            " kg",
            update_mass
        ))

        # Trail
        self.add(LabeledCheckbox(
            "Trail:",
            ball.leaves_trail,
            lambda v: setattr(ball, "leaves_trail", v)
        ))

    def _info_label(self, name, vec, unit):
        return VectorInput(text=name, value=vec, unit=unit)

class Navbar_ui(arcade.gui.UIBoxLayout):
    def __init__(self, height, size_hint, game):
        super().__init__(vertical=False, size_hint=size_hint, height=height)
        self.game = game
        self.style = {
            "normal": {
                "font_size": 14,
                "font_color": arcade.color.WHITE,
                "bg": None,
                "border": None,
            },
            "hover": {
                "font_size": 14,
                "font_color": arcade.color.LIGHT_GRAY,
                "bg": None,
                "border": None,
            },
            "press": {
                "font_size": 14,
                "font_color": arcade.color.GRAY,
                "bg": None,
                "border": None,
            },
        }
        self.space_between = 10
        self.padding = (10, 5, 10, 5)

        self.add(self._btn("Home", 100, self.on_home))
        self.pause_btn = self._btn("Pause", 120, self.on_pause_toggle)
        self.add(self.pause_btn)
        self.add(self._btn("Clear", 100, self.on_clear))
        self.add(self._btn("Settings", 120, self.on_settings))

    def _btn(self, text, width, handler):
        btn = arcade.gui.UIFlatButton(text=text, width=width, height=self.height, style=self.style)
        btn.event("on_click")(lambda e: handler())
        return btn

    def on_home(self):
        pass

    def on_pause_toggle(self):
        self.game.pause_sim_toggle()
        self.pause_btn.text = "Play" if self.game.pause else "Pause"

    def on_clear(self):
        self.game.clear()

    def on_settings(self):
        self.game.simulation_edit()

class Sidebar_ui(arcade.gui.UIGridLayout):
    def __init__(self, width, game):
        super().__init__(column_count=2, row_count=5, size_hint=(None, 1.0), width=width)
        self.game = game

        def add_ball():
            b = Ball(0, 0, 10)
            b.drag = True
            b.v = np.array([0.0, 0.0])
            self.game.balls.append(b)
            self.game.dragged_ball = b
        self.add(ToolCard("assets\\ballicon.png", "Add Ball", add_ball), column=0, row=0)