import numpy as np


# grid
spacing = 0.01
length = 1.5
x = np.arange(0, length, spacing)

# velocity
v = 1.

# time
start = 0.
end = 1.
step = 0.01

# initial gauss profile
loc = 0.3
scale = 0.1
u = np.exp(-1 / scale**2 * (x - loc)**2)
u0 = u.copy()

# time loop - Lax method
factor = (v * step) / (2 * spacing)

for t in np.arange(start, end, step):
    u_left = np.roll(u, 1)
    u_right = np.roll(u, -1)
    u1 = 0.5 * (u_right + u_left) - factor * (u_right - u_left)
    u = u1.copy()
