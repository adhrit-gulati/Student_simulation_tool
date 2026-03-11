from circuit import Circuit
import arcade
import arcade.gui
import numpy as np
from drawcomponents import draw_resistor, draw_wire

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Circuit Simulation"

GRID_SPACING = 30
POINT_RADIUS = 1
HIT_RADIUS = 10

def snap(x, y):
    return (
        round(x / GRID_SPACING) * GRID_SPACING,
        round(y / GRID_SPACING) * GRID_SPACING
    )

def dist_to_line(line, x, y):
    p = np.array([x, y], dtype=float)
    p0, p1 = map(np.array, line)

    v = p1 - p0
    w = p - p0

    seg_len = np.linalg.norm(v)
    if seg_len == 0:
        return np.linalg.norm(p - p0)

    t = np.dot(w, v) / (seg_len ** 2)
    t = np.clip(t, 0.0, 1.0)

    projection = p0 + t * v
    return np.linalg.norm(p - projection)

class Component:
    def __init__(self, start, left_voltage=False):
        self.type = "component"
        self.endpoints = [np.array(start), np.array(start)]
        self.V = left_voltage
        self.G = False
        self.name = ""

    def set_endpoint(self, index, position):
        self.endpoints[index] = np.array(position)

    def draw(self, color):
        pass

class Wire(Component):
    def __init__(self, start, left_voltage=False):
        super().__init__(start, left_voltage)
        self.type = "wire"

    def draw(self):
        draw_wire(self.endpoints, (12, 203, 118), self.V, self.G)

class Resistor(Component):
    def __init__(self, start, left_voltage=False, resistance=8.0):
        super().__init__(start, left_voltage)
        self.type = "resistor"
        self.resistance = resistance

    def draw(self):
        draw_resistor(self.endpoints, 2, (44, 222, 145), self.V, self.G)

class ComponentEditUI(arcade.gui.UIBoxLayout):

    def __init__(self, component, window):
        super().__init__(align="left", vertical=True)

        self.component = component
        self.window = window
        circuit = window.circuit

        self.add(arcade.gui.UILabel(
            text=component.type.upper(),
            font_size=22,
            bold=True,
            text_color=arcade.color.WHITE
        ))

        self.add(arcade.gui.UILabel(
            text=f"name: {component.name}",
            text_color=arcade.color.WHITE
        ))

        a, b = component.endpoints

        node_a = window.node_from_point(a)
        node_b = window.node_from_point(b)

        self.node_a = node_a
        self.node_b = node_b

        self.add(arcade.gui.UILabel(
            text=f"nodes: {node_a}-{node_b}",
            text_color=arcade.color.LIGHT_GRAY
        ))

        try:
            current = circuit.get_current(component.name)
        except RuntimeError:
            current = "?"

        self.add(arcade.gui.UILabel(
            text=f"current: {current}",
            text_color=arcade.color.LIGHT_GRAY
        ))

        # resistor editor
        if isinstance(component, Resistor):
            self.add_resistance_editor()

        # voltage node editor
        self.add_voltage_editor()

    def add_resistance_editor(self):

        row = arcade.gui.UIBoxLayout(vertical=False)

        label = arcade.gui.UILabel(
            text="Resistance (Ω):",
            text_color=arcade.color.WHITE
        )

        field = arcade.gui.UIInputText(
            text=str(self.component.resistance),
            text_color=arcade.color.CYAN
        )

        @field.event("on_change")
        def change(event):
            try:
                value = float(field.text)
                self.component.resistance = value

                for u, v, key, data in self.window.circuit.graph.edges(data=True, keys=True):
                    if key == self.component.name:
                        data["resistance"] = value
            except ValueError:
                pass

        row.add(label)
        row.add(field)
        self.add(row)

    def add_voltage_editor(self):

        circuit = self.window.circuit
        voltage_node = None

        # check if either node is a voltage node
        for node in [self.node_a, self.node_b]:
            if node in circuit.graph.nodes:
                data = circuit.graph.nodes[node]

                if data.get("type") == "voltage_source":
                    voltage_node = node
                    break

        if voltage_node is None:
            return

        node_data = circuit.graph.nodes[voltage_node]
        voltage = node_data.get("voltage", 0)

        row = arcade.gui.UIBoxLayout(vertical=False)

        label = arcade.gui.UILabel(
            text="Voltage (V):",
            text_color=arcade.color.WHITE
        )

        field = arcade.gui.UIInputText(
            text=str(voltage),
            text_color=arcade.color.ORANGE
        )

        @field.event("on_change")
        def change(event):
            try:
                value = float(field.text)

                circuit.graph.nodes[voltage_node]["voltage"] = value

                # also update component node if needed
                if hasattr(self.component, "voltage"):
                    self.component.voltage = value

            except ValueError:
                pass

        row.add(label)
        row.add(field)

        self.add(row)

class GridWindow(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color((12, 12, 17))

        self.draw_mode = False
        self.preview = None
        self.components = []
        self.node_map = {}
        self.junction_counter = 1
        self.voltage_counter = 1
        self.resistance_counter = 1
        self.wire_counter = 1
        self.circuit = Circuit()
        self.ui = arcade.gui.UIManager()
        self.ui.enable()
        self.active_ui = None
        self.make_solve_button()

    def on_draw(self):
        self.clear()
        if self.draw_mode:
            self.draw_grid()
        if self.preview:
            self.preview.draw()
        
        for c in self.components:
            c.draw()

        self.draw_ui_panel()

    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH + 1, GRID_SPACING):
            for y in range(0, SCREEN_HEIGHT + 1, GRID_SPACING):
                arcade.draw_circle_filled(x, y, POINT_RADIUS, (12, 203, 118))

    def make_solve_button(self):

        self.solve_button = arcade.gui.UIFlatButton(text="Solve", width=100)

        @self.solve_button.event("on_click")
        def click(event):
            self.circuit.solve()

        anchor = arcade.gui.UIAnchorLayout()
        anchor.add(self.solve_button,
                anchor_x="left",
                anchor_y="bottom",
                align_x=5,
                align_y=5)

        self.ui.add(anchor)

    def node_from_point(self, pos):

        pos = snap(pos[0], pos[1])
        return self.node_map.get(pos)

    def add_node(self, pos, node_type):

        pos = snap(pos[0], pos[1])

        if pos in self.node_map:
            return self.node_map[pos]

        name = self.create_name(node_type)
        self.node_map[pos] = name

        if node_type == "voltage":
            self.circuit.add_voltage_source(name, 5)

        elif node_type == "ground":
            self.circuit.add_ground()

        else:
            self.circuit.add_junction(name)

        return name

    def create_name(self, type): 

        if type == "junction":
            name = f"J{self.junction_counter}"
            self.junction_counter += 1
            return name

        if type == "voltage":
            name = f"V{self.voltage_counter}"
            self.voltage_counter += 1
            return name

        if type == "ground":
            return "GND"

        if type == "resistor":
            name = f"R{self.resistance_counter}"
            self.resistance_counter += 1
            return name

        if type == "wire":
            name = f"W{self.wire_counter}"
            self.wire_counter += 1
            return name

    def draw_ui_panel(self):

        if self.active_ui:

            rect = self.active_ui.rect
            xoff = 5
            yoff = 5
            arcade.draw_lbwh_rectangle_filled(
                rect.x - rect.width / 2 - xoff,
                rect.y - rect.height / 2 - yoff,
                rect.width + 2*xoff,
                rect.height + 2*yoff,
                (31, 31, 122)
            )

        self.ui.draw()

    def on_mouse_press(self, x, y, button, modifiers):

        if self.draw_mode:
            self.handle_draw_click(x, y, button, modifiers)
        else:
            self.handle_select_click(x, y)

    def handle_draw_click(self, x, y, button, modifiers):

        pos = snap(x, y)

        if modifiers & arcade.key.MOD_CTRL:
            self.delete_component(x, y)
            return

        left_voltage = button == arcade.MOUSE_BUTTON_RIGHT

        if not self.preview:

            if modifiers & arcade.key.MOD_SHIFT:
                self.preview = Resistor(pos, left_voltage)
            else:
                self.preview = Wire(pos, left_voltage)

            if left_voltage:
                self.add_node(pos, "voltage")
            else:
                self.add_node(pos, "junction")

            return

        self.finish_component(button)

    def finish_component(self, button):

        a, b = self.preview.endpoints
        self.circuit.voltages = {}

        if not np.array_equal(a, b):

            a = snap(a[0], a[1])
            b = snap(b[0], b[1])

            if button == arcade.MOUSE_BUTTON_RIGHT:
                self.preview.G = True
                node_b = self.add_node(b, "ground")
            else:
                self.preview.G = False
                node_b = self.add_node(b, "junction")

            node_a = self.node_from_point(a)

            if isinstance(self.preview, Wire):

                name = self.create_name("wire")
                self.circuit.add_wire(node_a, node_b, name)
                self.preview.name = name

            elif isinstance(self.preview, Resistor):

                name = self.create_name("resistor")
                self.circuit.add_resistor(node_a, node_b, name, self.preview.resistance)
                self.preview.name = name

            self.components.append(self.preview)

        self.preview = None

    def delete_component(self, x, y):

        for i, c in enumerate(self.components):

            if dist_to_line(c.endpoints, x, y) < HIT_RADIUS:

                u = snap(c.endpoints[0][0], c.endpoints[0][1])
                v = snap(c.endpoints[1][0], c.endpoints[1][1])

                node_u = self.node_from_point(u)
                node_v = self.node_from_point(v)

                if node_u and node_v:

                    try:
                        self.circuit.graph.remove_edge(node_u, node_v, key=c.name)
                    except:
                        pass

                    for pos, node in [(u, node_u), (v, node_v)]:

                        if node and self.circuit.graph.degree(node) == 0:

                            self.circuit.graph.remove_node(node)

                            if pos in self.node_map:
                                del self.node_map[pos]

                del self.components[i]
                return

    def handle_select_click(self, x, y):

        if self.active_ui and (x, y) in self.active_ui.rect:
            return

        for component in self.components:

            if dist_to_line(component.endpoints, x, y) < HIT_RADIUS:

                self.open_component_ui(component)
                return

        self.clear_ui()

    def open_component_ui(self, component):

        self.ui.clear()

        anchor = arcade.gui.UIAnchorLayout()

        panel = ComponentEditUI(component, self)

        anchor.add(panel,
                anchor_x="left",
                anchor_y="top",
                align_x=5,
                align_y=-10)

        self.ui.add(anchor)
        self.active_ui = panel

    def clear_ui(self):
        self.ui.clear()
        self.active_ui = None

        if not self.draw_mode:
            self.make_solve_button()

    def on_mouse_motion(self, x, y, dx, dy):
        if self.preview:
            self.preview.set_endpoint(1, snap(x, y))

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.SPACE:

            self.draw_mode = not self.draw_mode
            self.preview = None

            self.solve_button.visible = not self.draw_mode

            self.clear_ui()


def main():
    GridWindow()
    arcade.run()


if __name__ == "__main__":
    main()