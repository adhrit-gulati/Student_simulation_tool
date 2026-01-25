import arcade
import math 
from typing import List, Tuple
import numpy as np

WIDTH = 10
SPACING = 5

def draw_zigzag(endpoints: List[np.ndarray], line_width, color):
    length = np.linalg.norm(endpoints[1] - endpoints[0])
    if length == 0:
        return

    unit_vec = (endpoints[1] - endpoints[0]) / length
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])

    n = round(length / SPACING)
    point_list = np.linspace(endpoints[0], endpoints[1], n + n % 4 + 1)

    sign = 1j
    for i in range(len(point_list)):
        point_list[i] += sign.real * WIDTH * perp_vec
        sign *= 1j

    points = [tuple(p) for p in point_list]
    arcade.draw_line_strip(points, color=color, line_width=line_width)

def draw_voltage_tick(endpoints: List[np.ndarray], color):
    length = np.linalg.norm(endpoints[1] - endpoints[0])
    if length == 0:
        return

    unit_vec = (endpoints[1] - endpoints[0]) / length
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])

    p1 = endpoints[0] + perp_vec * WIDTH * 1.5
    p2 = endpoints[0] - perp_vec * WIDTH * 1.5

    arcade.draw_line(p1[0], p1[1], p2[0], p2[1], color=color)

def draw_ground_tick(endpoints: List[np.ndarray], color):
    length = np.linalg.norm(endpoints[1] - endpoints[0])
    if length == 0:
        return

    unit_vec = (endpoints[1] - endpoints[0]) / length
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])

    p1 = endpoints[1] + perp_vec * WIDTH
    p2 = endpoints[1] - perp_vec * WIDTH

    arcade.draw_line(p1[0], p1[1], p2[0], p2[1], color=color)

def draw_resistor(endpoints: List[np.ndarray], line_width, color, V, G):
    draw_zigzag(endpoints, line_width, color)
    if V:
        draw_voltage_tick(endpoints, color)
    if G:
        draw_ground_tick(endpoints, color)

def draw_wire(endpoints: List[np.ndarray], color, V, G):
    arcade.draw_line(endpoints[0][0], endpoints[0][1], endpoints[1][0], endpoints[1][1], color=color)
    if V:
        draw_voltage_tick(endpoints, color)
    if G:
        draw_ground_tick(endpoints, color)