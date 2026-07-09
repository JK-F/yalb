#!/usr/bin/env python
# coding: utf-8

CSV_FILE = "./04_distrib.csv"
LID_VELOCITY = 0.1
OUTPUT_FILE = "moving_lid.mp4"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

DIR = {
    0: ( 0,  0),
    1: ( 1,  0),
    2: ( 0,  1),
    3: (-1,  0),
    4: ( 0, -1),
    5: ( 1,  1),
    6: (-1,  1),
    7: (-1, -1),
    8: ( 1, -1),
}

def get_grid_dims(csv_file):
    x_set, y_set = set(), set()
    for chunk in pd.read_csv(csv_file, chunksize=100000, usecols=['x', 'y']):
        x_set.update(chunk['x'].unique())
        y_set.update(chunk['y'].unique())
    return sorted(x_set), sorted(y_set)

def compute_velocity(dist, nx, ny):
    rho = dist.sum(axis=2)
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    for d, (cx, cy) in DIR.items():
        ux += cx * dist[:, :, d]
        uy += cy * dist[:, :, d]
    ux /= rho
    uy /= rho
    return ux, uy

def frame_generator(csv_file, nx, ny):
    val_cols = [f'd{i}' for i in range(9)]
    buffer = []
    buffer_ts = None
    for chunk in pd.read_csv(csv_file, chunksize=100000):
        for ts, group in chunk.groupby('timestep', sort=False):
            if buffer_ts is None:
                buffer_ts = ts
                buffer = [group]
            elif ts == buffer_ts:
                buffer.append(group)
            else:
                df_ts = pd.concat(buffer, ignore_index=True)
                piv = df_ts.pivot_table(index=['x', 'y'], columns='dir', values='dist_value')
                piv.columns = val_cols
                piv.reset_index(inplace=True)
                dist = piv[val_cols].values.reshape(nx, ny, 9)
                yield buffer_ts, *compute_velocity(dist, nx, ny)
                buffer_ts = ts
                buffer = [group]
    if buffer:
        df_ts = pd.concat(buffer, ignore_index=True)
        piv = df_ts.pivot_table(index=['x', 'y'], columns='dir', values='dist_value')
        piv.columns = val_cols
        piv.reset_index(inplace=True)
        dist = piv[val_cols].values.reshape(nx, ny, 9)
        yield buffer_ts, *compute_velocity(dist, nx, ny)

x_vals, y_vals = get_grid_dims(CSV_FILE)
nx, ny = len(x_vals), len(y_vals)
L = nx
print(f"Grid: {nx}x{ny}, L={L}")

X = (np.array(y_vals) - y_vals[0]) / L
Y = (np.array(x_vals) - x_vals[0]) / L

fig, ax = plt.subplots(figsize=(8, 8))
fig.tight_layout(pad=3)

def animate(frame_data):
    ts, ux, uy = frame_data
    if ts % 500:
        print(f"animating {ts}")
    ax.cla()
    speed = np.sqrt(ux**2 + uy**2)
    strm = ax.streamplot(X, Y, ux, uy, color=speed, cmap='viridis',
                         density=1.5, linewidth=1)
    ax.set_xlabel('x / L')
    ax.set_ylabel('y / L')
    ax.set_title(f'Lid-Driven Cavity Flow (timestep {ts})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.annotate('', xy=(0.85, 1.06), xytext=(0.15, 1.06),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                annotation_clip=False)
    ax.text(0.5, 1.09, f'Lid velocity = {LID_VELOCITY}',
            ha='center', va='bottom', fontsize=10)
    return strm.lines,

anim = FuncAnimation(fig, animate, frames=frame_generator(CSV_FILE, nx, ny), repeat=False)

if OUTPUT_FILE:
    print(f"Saving to {OUTPUT_FILE}...")
    anim.save(OUTPUT_FILE, writer='ffmpeg', dpi=100)
    print("Done")
