import numpy as np
from PIL import Image, ImageDraw
import arcade

def sigmoid_color(q, kc=0.5):
    s = 1 / (1 + np.exp(-kc * q))
    return int(s * 255)

def create_arrow_texture(w=40, h=5):

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

def color_from_charge(q, kc=0.5):
    # Map charge to red and blue
    R = int(255 / (1 + np.exp(-kc * q * 1e6)))   # positive charges → more red
    B = int(255 / (1 + np.exp(-kc * -q * 1e6)))  # negative charges → more blue
    
    # Green depends on closeness to 128
    G = int(128 * (1 - (abs(R-128) + abs(B-128)) / 255))
    
    # Clamp values to 0-255
    G = max(0, min(255, G))
    
    return (R, G, B)