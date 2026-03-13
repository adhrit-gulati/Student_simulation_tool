import numpy as np

# Constants
field_grid_spacing = 50  # pixels
W, H = 800, 600

meter = 50                  # 50 pixels = 1 meter
g = np.array([0.0, -9.81])  # m/s^2
k = 8.98e9                  # N m^2 C^-2

energy_loss_ball_edge = 0.8
energy_loss_ball_ball = 0.8
energy_loss_ball_barrier = 0.3

trail_length = 50            # number of points a trail
barrier_draw_radius = 5
barrier_remove_radius = 15

number_substeps = 10