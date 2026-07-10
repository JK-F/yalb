#!/usr/bin/env python
# coding: utf-8

import sys
from datetime import datetime

format_code = "%Y_%m_%d_%Hh%Mm"
now = datetime.now()
date_prefix = now.strftime(format_code)

CSV_FILE = "./data/05_velocity.csv"

args = sys.argv[1:]
LID_VELOCITY = float(args[0]) if len(args) >= 1 else 0.3
OMEGA = float(args[1]) if len(args) >= 2 else 1.0

OUTPUT_FILE = f"../archive/{date_prefix}_{LID_VELOCITY}_moving_lid"

import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

step_rows = int(subprocess.run(
    ["awk", "-F,", 'NR==2{ts=$1; c=1} NR>2 && $1==ts{c++} NR>2 && $1!=ts{print c; printed=1; exit} END{if(!printed && c) print c}', CSV_FILE],
    capture_output=True, text=True, check=True
).stdout.strip())

reader = pd.read_csv(CSV_FILE, chunksize=step_rows)

first_chunk = next(reader)
x_vals = sorted(first_chunk['x'].unique())
y_vals = sorted(first_chunk['y'].unique())
nx, ny = len(x_vals), len(y_vals)
L = nx
print(f"Grid: {nx}x{ny}, L={L}, step_rows={step_rows}")

total_lines = int(subprocess.run(
    ["wc", "-l", CSV_FILE], capture_output=True, text=True, check=True
).stdout.split()[0])
n_timesteps = (total_lines - 1) // step_rows

X_grid, Y_grid = np.meshgrid(
    (np.array(x_vals) - x_vals[0]) / L,
    (np.array(y_vals) - y_vals[0]) / L
)

fig, ax = plt.subplots(figsize=(8, 8))
fig.tight_layout(pad=2)

def get_frame(chunk):
    sorted_chunk = chunk.sort_values(['x', 'y'])
    ts = chunk['timestep'].iloc[0]
    ux = sorted_chunk['ux'].values.reshape(nx, ny).T
    uy = sorted_chunk['uy'].values.reshape(nx, ny).T
    speed = np.sqrt(chunk['ux']**2 + chunk['uy']**2).max()
    return ts, ux, uy, speed

first_ts, first_ux, first_uy, global_max_speed = get_frame(first_chunk)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=global_max_speed))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='|u| - Velocity Magnitude')

frame_counter = [0]

def animate(frame_data):
    ts, ux, uy, _ = frame_data
    frame_counter[0] += 1
    if ts % 100 == 0:
        print(f"\ranimating timestep {ts}, frame {frame_counter[0]}      ")
    ax.cla()
    speed = np.sqrt(ux**2 + uy**2)
    strm = ax.streamplot(X_grid, Y_grid, ux, uy, color=speed, cmap='viridis',
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

def frame_generator():
    yield first_ts, first_ux, first_uy, global_max_speed
    for chunk in reader:
        ts, ux, uy, ms = get_frame(chunk)
        yield ts, ux, uy, ms  

if OUTPUT_FILE:
    if n_timesteps == 1:
        OUTPUT_FILE = OUTPUT_FILE + ".png"
        print(f"Saving static image to {OUTPUT_FILE}...")
        animate((first_ts, first_ux, first_uy, global_max_speed))
        fig.savefig(OUTPUT_FILE, dpi=100)
        print("Done")
    else:
        anim = FuncAnimation(fig, animate, frames=frame_generator(), repeat=False, save_count=None)
        OUTPUT_FILE = OUTPUT_FILE + ".mp4"
        print(f"Saving to {OUTPUT_FILE}...")
        anim.save(f"{OUTPUT_FILE}", writer='ffmpeg', dpi=100)
        print(f"\nDone — {frame_counter[0]} frames rendered")
