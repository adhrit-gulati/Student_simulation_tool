import networkx as nx
import numpy as np

class Circuit:
    def __init__(self):
        self.graph = nx.MultiGraph()
        self.voltages = {}

    def add_junction(self, name: str):
        self.graph.add_node(
            name,
            type="junction"
        )

    def add_ground(self, name: str = "GND"):
        self.graph.add_node(
            name,
            type="ground",
            voltage = 0
        )

    def add_resistor(self, n1: str, n2: str, name, resistance: float):
        self.graph.add_edge(
            n1,
            n2,
            key=name,
            type="resistor",
            resistance=resistance
        )

    def add_voltage_source(self, name:str, voltage):
        self.graph.add_node(
            name,
            type="voltage_source",
            voltage = voltage
        )

    def add_wire(self, n1: str, n2: str, name):
        self.graph.add_edge(
            n1,
            n2,
            key = name,
            type="wire",
            resistance=1e-6
        )

    def solve(self):
        n = sum(1 for _, data in self.graph.nodes(data=True) if data.get('type') == 'junction')
        Matrix = np.zeros((n, n))
        B = np.zeros(n)

        node_index = {}
        i = 0
        for node in self.graph.nodes:
            if self.graph.nodes[node]['type'] not in ['ground', 'voltage_source']:
                node_index[node] = i
                i += 1

        row = 0
        for node in self.graph.nodes:
            if self.graph.nodes[node]['type'] in ['ground', 'voltage_source']:
                continue
            diagonalcoefficient = 0
            for neighbor in self.graph.neighbors(node):
                for _, edge_data in self.graph.get_edge_data(node, neighbor).items():
                    resistance = edge_data['resistance']

                    neighbor_type = self.graph.nodes[neighbor]['type']

                    if neighbor_type in ['voltage_source', 'ground']:
                        B[row] += self.graph.nodes[neighbor]['voltage'] / resistance

                    elif neighbor_type == 'junction':
                        Matrix[row, node_index[neighbor]] += -1 / resistance
                    
                    diagonalcoefficient += 1 / resistance
            Matrix[row, row]= diagonalcoefficient

            row += 1
        
        solution = np.linalg.solve(Matrix, B)
        self.voltages = {node: round(solution[idx], 3) for node, idx in node_index.items()}
        for node, data in self.graph.nodes(data=True):
            if data['type'] == 'voltage_source':
                self.voltages[node] = data['voltage']
            if data["type"] == "ground":
                self.voltages[node] = 0
        return self.voltages

    def get_current(self, resistor_name):
        for u, v, key, data in self.graph.edges(data=True, keys=True):
            if key == resistor_name:
                return round( abs(self.voltages[u]-self.voltages[v]) / data["resistance"] , 3)

    def __repr__(self):
        return (
            f"Circuit(nodes={self.graph.number_of_nodes()}, "
            f"components={self.graph.number_of_edges()})"
        )
