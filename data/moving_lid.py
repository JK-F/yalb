#!/usr/bin/env python
# coding: utf-8

import sys
from datetime import datetime

format_code = "%Y_%m_%d_%Hh%Mm"
now = datetime.now()
date_prefix = now.strftime(format_code)

CSV_FILE = "./data/05_velocity.csv"

args = sys.argv[1:]
if len(args) >= 1:
    LID_VELOCITY = float(args[0])
else:
    LID_VELOCITY = 0.3

if len(args) >= 2:
    OMEGA = float(args[1])
else:
    OMEGA = 1.0

OUTPUT_FILE = f"../archive/{date_prefix}_{LID_VELOCITY}_moving_lid.mp4"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_grid_dims(csv_file):
    df = pd.read_csv(csv_file, usecols=['x', 'y'])
    return sorted(df['x'].unique()), sorted(df['y'].unique())

def frame_generator(csv_file, nx, ny):
    df = pd.read_csv(csv_file)
    for ts, group in df.groupby('timestep', sort=False):
        sorted_group = group.sort_values(['x', 'y'])
        ux = sorted_group['ux'].values.reshape(nx, ny)
        uy = sorted_group['uy'].values.reshape(nx, ny)
        yield ts, ux, uy

x_vals, y_vals = get_grid_dims(CSV_FILE)
nx, ny = len(x_vals), len(y_vals)
L = nx
print(f"Grid: {nx}x{ny}, L={L}")

df_all = pd.read_csv(CSV_FILE)
global_max_speed = np.sqrt(df_all['ux']**2 + df_all['uy']**2).max()

X = (np.array(y_vals) - y_vals[0]) / L
Y = (np.array(x_vals) - x_vals[0]) / L

fig, ax = plt.subplots(figsize=(8, 8))
fig.tight_layout(pad=1)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=global_max_speed))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='|u| - Velocity Magnitude')

def animate(frame_data):
    ts, ux, uy = frame_data
    if ts % 500 == 0:
        print(f"\r animating {ts}")
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
    ax.text(0.5, 1.09, f'Lid velocity = {LID_VELOCITY}, Ω = {OMEGA}',
            ha='center', va='bottom', fontsize=10)
    return strm.lines,

anim = FuncAnimation(fig, animate, frames=frame_generator(CSV_FILE, nx, ny), repeat=False)

if OUTPUT_FILE:
    print(f"Saving to {OUTPUT_FILE}...")
    anim.save(OUTPUT_FILE, writer='ffmpeg', dpi=100)
    print("Done")
