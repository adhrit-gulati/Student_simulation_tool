import arcade

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