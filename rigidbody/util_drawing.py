import numpy as np
from PIL import Image, ImageDraw
import arcade
import math
from constants import meter

def sigmoid_color(q, kc=0.5):
    s = 1 / (1 + np.exp(-kc * q))
    return int(s * 255)

def create_arrow_texture(w=40, h=5):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    cx = w // 2
    cy = h // 2

    shaft_len = w * 0.4
    head_len  = w * 0.1

    d.rectangle([(cx, cy - h*0.15), (cx + shaft_len, cy + h*0.15)], fill="white")

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

def draw_arrow(x, y, vector: np.ndarray, color=arcade.color.WHITE, scale=4.0, width=2, head_length=10, head_angle=30):
    end_x = x + (vector[0] * scale)
    end_y = y + (vector[1] * scale)

    arcade.draw_line(x, y, end_x, end_y, color, width)

    angle_rad = np.atan2(vector[1], vector[0])
    head_rad = np.radians(head_angle)

    left_x = end_x - head_length * np.cos(angle_rad - head_rad)
    left_y = end_y - head_length * np.sin(angle_rad - head_rad)

    right_x = end_x - head_length * np.cos(angle_rad + head_rad)
    right_y = end_y - head_length * np.sin(angle_rad + head_rad)

    arcade.draw_line(end_x, end_y, left_x, left_y, color, width)
    arcade.draw_line(end_x, end_y, right_x, right_y, color, width)
    

def color_from_charge(q, kc=0.5):
    # Map charge to red and blue
    R = int(255 / (1 + np.exp(-kc * q * 1e6)))   # positive charges → more red
    B = int(255 / (1 + np.exp(-kc * -q * 1e6)))  # negative charges → more blue
    
    # Green depends on closeness to 128
    G = int(128 * (1 - (abs(R-128) + abs(B-128)) / 255))
    
    # Clamp values to 0-255
    G = max(0, min(255, G))
    
    return (R, G, B)