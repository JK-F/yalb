#!/usr/bin/env python
# coding: utf-8

"""Combine the per-rank velocity CSVs written by an MPI milestone06 run
(data/06_p<px>_<py>_velocity.csv) into a single CSV with global grid
coordinates, so it can be used with moving_lid.py / diff_frames.py.

Each rank's file is written one timestep at a time and in increasing
timestep order (one contiguous block per print_velocity() call), so this
merges the files timestep-by-timestep and streams straight to the output
file. Peak memory is bounded by a single timestep's worth of data across
all ranks (i.e. the global grid), not by the total number of timesteps -
this is what lets it handle arbitrarily long runs.
"""

import argparse
import csv
import glob
import itertools
import re
import sys

FILENAME_PATTERN = re.compile(r"_p(\d+)_(\d+)_velocity\.csv$")


def iter_timestep_groups(path):
    """Yield (timestep, rows) for one rank's CSV, in file order.

    Relies on each rank's file already being grouped by timestep - true for
    milestone06's output - so only one timestep's rows are ever buffered.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for timestep, rows in itertools.groupby(reader, key=lambda row: row["timestep"]):
            yield int(timestep), list(rows)


def shift_to_global(rows, px, py, nx_local, ny_local):
    x_min = min(int(row["x"]) for row in rows)
    y_min = min(int(row["y"]) for row in rows)
    shifted = [
        (
            int(row["x"]) - x_min + px * nx_local,
            int(row["y"]) - y_min + py * ny_local,
            row["ux"],
            row["uy"],
        )
        for row in rows
    ]
    shifted.sort(key=lambda t: (t[0], t[1]))
    return shifted


def combine(input_glob, output_path):
    files = sorted(glob.glob(input_glob))
    if not files:
        sys.exit(f"No files matched: {input_glob}")

    ranks = []
    for path in files:
        match = FILENAME_PATTERN.search(path)
        if not match:
            sys.exit(f"Filename doesn't look like '..._p<px>_<py>_velocity.csv': {path}")
        px, py = int(match.group(1)), int(match.group(2))
        ranks.append({"px": px, "py": py, "path": path, "iter": iter_timestep_groups(path), "nx": None, "ny": None})

    total_rows = 0
    last_timestep = None
    with open(output_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["timestep", "x", "y", "ux", "uy"])

        while True:
            items = [next(r["iter"], None) for r in ranks]
            finished = [item is None for item in items]

            if all(finished):
                break
            if any(finished):
                stuck = [r["path"] for r, done in zip(ranks, finished) if not done]
                sys.exit(f"Rank files have different numbers of timesteps; still had data in: {stuck}")

            timesteps = {ts for ts, _ in items}
            if len(timesteps) != 1:
                sys.exit(f"Rank files are out of sync - saw multiple timesteps at once: {sorted(timesteps)}")
            timestep = timesteps.pop()
            if last_timestep is not None and timestep <= last_timestep:
                sys.exit(f"Timesteps are not increasing: {last_timestep} then {timestep}")
            last_timestep = timestep

            combined_rows = []
            for r, (_, rows) in zip(ranks, items):
                if r["nx"] is None:
                    r["nx"] = len({row["x"] for row in rows})
                    r["ny"] = len({row["y"] for row in rows})
                combined_rows.extend(shift_to_global(rows, r["px"], r["py"], r["nx"], r["ny"]))

            combined_rows.sort(key=lambda t: (t[0], t[1]))
            for gx, gy, ux, uy in combined_rows:
                writer.writerow([timestep, gx, gy, ux, uy])
                total_rows += 1

    print(f"Combined {len(files)} files ({total_rows} rows across {len(ranks)} ranks) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="data/06_p*_velocity.csv",
        help="Glob pattern matching per-rank velocity CSVs",
    )
    parser.add_argument(
        "--output",
        default="data/06_velocity.csv",
        help="Path to write the combined CSV",
    )
    args = parser.parse_args()
    combine(args.input_glob, args.output)
