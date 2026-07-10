#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "./data/05_velocity.csv"

df = pd.read_csv(CSV_FILE)
timesteps = sorted(df['timestep'].unique())
t, tp = timesteps[-2], timesteps[-1]
print(f"Last two timesteps: {t}, {tp}")

x_vals = sorted(df['x'].unique())
y_vals = sorted(df['y'].unique())
nx, ny = len(x_vals), len(y_vals)
L = nx

X_grid, Y_grid = np.meshgrid(
    (np.array(x_vals) - x_vals[0]) / L,
    (np.array(y_vals) - y_vals[0]) / L
)

def get_velocities(ts):
    chunk = df[df['timestep'] == ts].sort_values(['x', 'y'])
    ux = chunk['ux'].values.reshape(nx, ny).T
    uy = chunk['uy'].values.reshape(nx, ny).T
    return ux, uy

ux_t, uy_t = get_velocities(t)
ux_tp, uy_tp = get_velocities(tp)

du_x = ux_t - ux_tp
du_y = uy_t - uy_tp
du_mag = np.sqrt(du_x**2 + du_y**2)
speed_t = np.sqrt(ux_t**2 + uy_t**2)
speed_tp = np.sqrt(ux_tp**2 + uy_tp**2)
dspeed = speed_t - speed_tp

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
strm = ax.streamplot(X_grid, Y_grid, ux_t, uy_t, color=speed_t, cmap='viridis',
                     density=1.5, linewidth=1)
ax.set_title(f't = {t}')
ax.set_xlabel('x / L')
ax.set_ylabel('y / L')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.colorbar(strm.lines, ax=ax, label='|u|')

ax = axes[0, 1]
strm = ax.streamplot(X_grid, Y_grid, ux_tp, uy_tp, color=speed_tp, cmap='viridis',
                     density=1.5, linewidth=1)
ax.set_title(f"t' = {tp}")
ax.set_xlabel('x / L')
ax.set_ylabel('y / L')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.colorbar(strm.lines, ax=ax, label='|u|')

ax = axes[1, 0]
contour = ax.contourf(X_grid, Y_grid, dspeed, levels=50, cmap='RdBu_r')
ax.set_title('|u|(t) - |u|(t\')')
ax.set_xlabel('x / L')
ax.set_ylabel('y / L')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.colorbar(contour, ax=ax, label='Δ|u|')

ax = axes[1, 1]
contour = ax.contourf(X_grid, Y_grid, du_mag, levels=50, cmap='hot')
ax.set_title('|Δu| – Magnitude of velocity difference')
ax.set_xlabel('x / L')
ax.set_ylabel('y / L')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.colorbar(contour, ax=ax, label='|Δu|')

fig.tight_layout()
fig.savefig('../archive/diff_frames.png', dpi=150)
print("Saved ../archive/diff_frames.png")
