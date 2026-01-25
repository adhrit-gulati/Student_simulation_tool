from circuit import Circuit
import arcade, arcade.gui
from drawcomponents import draw_resistor, draw_wire
import numpy as np

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Circuit Simulation"

GRID_SPACING = 30
POINT_RADIUS = 1

def dist_to_line_endpoints(line, x, y):
    return abs(np.linalg.det([line[1]-line[0], line[0]-np.array([x, y])]))/np.linalg.norm(line[1]-line[0])

class ComponentEditUI(arcade.gui.UIBoxLayout):
    def __init__(self, component):
        super().__init__(align="left", vertical=True)

        self.component = component

        self.add(arcade.gui.UILabel(text=f"{component['type'].upper()}", font_size=22, bold=True, text_color=arcade.color.WHITE))

        # Resistor-specific controls
        if component["type"] == "resistor":
            r_box = arcade.gui.UIBoxLayout(vertical=False)
            r_label = arcade.gui.UILabel(text="Resistance (Ω):", text_color=arcade.color.WHITE)
            r_input = arcade.gui.UIInputText(
                text=str(component["resistance"]),
                text_color=arcade.color.CYAN,
                border_width=0
            )

            @r_input.event("on_change")
            def on_r_change(event):
                try:
                    component["resistance"] = float(r_input.text)
                except ValueError:
                    pass

            r_box.add(r_label)
            r_box.add(r_input)
            self.add(r_box)

class GridWindow(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.BLACK)
        self.draw_mode = False
        self.drawn_component = {}
        self.circuit_components = []
        self.ui = arcade.gui.UIManager()
        self.ui.enable()
        self.selected_component = None
        self.component_ui = None
        self.UI = None

    def draw_component(self, component, color):
        if component["type"] == "wire":
            draw_wire(component["endpoints"], color=color, V=component["V"], G=component["G"])
        if component["type"] == "resistor":
            draw_resistor(component["endpoints"], 2, color, V=component["V"], G=component["G"])

    def on_draw(self):
        self.clear()
        if self.draw_mode:
            for x in range(0, SCREEN_WIDTH + 1, GRID_SPACING):
                for y in range(0, SCREEN_HEIGHT + 1, GRID_SPACING):
                    arcade.draw_circle_filled(
                        x, y, POINT_RADIUS, arcade.color.GRAY
                    )
        if self.drawn_component:
            self.draw_component(self.drawn_component, color=(100, 100, 100))
        if self.UI:
            rect = self.UI.rect
            padding = 12
            l = rect.x-rect.width/2
            b = rect.y-rect.height/2
            arcade.draw_lbwh_rectangle_filled(l-padding, b-padding, rect.width+2*padding, rect.height+2*padding, (10, 40, 100, 200))
            self.ui.draw()
        for component in self.circuit_components:
            self.draw_component(component, color=(230, 230, 230))

    def on_mouse_press(self, x, y, button, modifiers):
        if self.draw_mode:
            # grid points closest to clicked point
            grid_x = round(x / GRID_SPACING) * GRID_SPACING
            grid_y = round(y / GRID_SPACING) * GRID_SPACING
            
            #for creating/deleting a component
            if not self.drawn_component:

                if arcade.MOUSE_BUTTON_LEFT == button:
                    leftbutton = False
                if arcade.MOUSE_BUTTON_RIGHT == button:
                    leftbutton = True

                #CTRL + click deletes a component
                if modifiers & arcade.key.MOD_CTRL:
                    for i, component in enumerate(self.circuit_components):
                        if dist_to_line_endpoints(component["endpoints"], x, y) < 10:
                            del self.circuit_components[i]
                            return

                # right click creates a voltage source at start of component
                # SHIFT + click creates a resistor
                if modifiers & arcade.key.MOD_SHIFT:
                    self.drawn_component = {"endpoints": [np.array([grid_x, grid_y]), np.array([grid_x, grid_y])], "type":"resistor", "V":leftbutton, "G": False, "resistance":8}
                # simple click creates a wire
                else:
                    self.drawn_component = {"endpoints": [np.array([grid_x, grid_y]), np.array([grid_x, grid_y])], "type":"wire", "V":leftbutton, "G": False}

            # finish creating a component
            elif self.drawn_component:
                ep = self.drawn_component["endpoints"]
                if not np.array_equal(ep[0], ep[1]):
                    copy = self.drawn_component.copy()
                    #right click creates ground at end of component
                    if button == arcade.MOUSE_BUTTON_LEFT:
                        copy["G"] = False
                    if button == arcade.MOUSE_BUTTON_RIGHT:
                        copy["G"] = True

                    self.circuit_components.append(copy)
                self.drawn_component = {}
        else:
            if self.UI and (x, y) in self.UI.rect:
                return
            for component in self.circuit_components:
                if dist_to_line_endpoints(component["endpoints"], x, y) < 10:
                    self.make_component_ui(component)
                    return
            self.ui.clear()
            self.UI = None

    def on_mouse_motion(self, x, y, dx, dy):
        if self.drawn_component:
            self.drawn_component["endpoints"][1] = (round(x / GRID_SPACING) * GRID_SPACING, round(y / GRID_SPACING) * GRID_SPACING)

    def make_component_ui(self, component):
        self.ui.clear()
        anchor = arcade.gui.UIAnchorLayout()
        box = ComponentEditUI(component)
        anchor.add(box, anchor_x="left", anchor_y="top", align_x=5, align_y=-10)
        self.ui.add(anchor)
        self.UI = box

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.SPACE:
            self.draw_mode = not self.draw_mode
            self.drawn_component = {}

def main():
    window = GridWindow()
    arcade.run()


if __name__ == "__main__":
    main()
