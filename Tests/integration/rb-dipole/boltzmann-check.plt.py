#! /usr/bin/env python3
"""Plot the simulated dipole polar-angle density against the Boltzmann estimate.

Reads an rb-traj trajectory directly: each frame's 3x3 orientation matrix
rotates the body-frame dipole axis into the lab frame, and the polar angle
against the lab z axis is histogrammed to give the simulated density.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as mcolors

KT_KCAL_MOL = 0.58622592  # simulation temperature, kcal/mol
DR = 0.05
RMAX = 3.14159
DIPOLE_BODY = np.array([1.0, 0.0, 0.0])  # body-frame dipole axis
LAB_AXIS = np.array([0.0, 0.0, 1.0])     # lab reference axis

INERTIA = 1000.0          # Da AA^2, isotropic rigid-body moment of inertia (BrownDyn.bd)
KT_UNIT_FACTOR = 2.3900574e-09  # AA**2 Da/ns**2 -> kcal/mol (see CLAUDE.md unit notes)


def info(*obj):
    print('INFO: ', *obj, file=sys.stderr)


def load_rotations(trajectory: Path) -> np.ndarray:
    """Return an (N, 3, 3) stack of orientation matrices from the rb-traj file."""
    rows = []
    with trajectory.open() as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 14 or not fields[1].startswith("point"):
                continue
            # fields 5..13 = rotXX rotXY rotXZ rotYX rotYY rotYZ rotZX rotZY rotZZ
            rows.append([float(v) for v in fields[5:14]])
    if not rows:
        raise ValueError(f"No point records found in {trajectory}")
    return np.asarray(rows).reshape(-1, 3, 3)


def orientation_density(trajectory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_center, density) rows for the dipole polar-angle distribution."""
    rotations = load_rotations(trajectory)

    dipole_lab = rotations @ DIPOLE_BODY          # (N, 3)
    cos_phi = dipole_lab @ LAB_AXIS               # (N,) == rotZX per frame
    angles = np.arccos(np.clip(cos_phi, -1.0, 1.0))

    edges = np.arange(int(np.ceil(RMAX / DR)) + 1) * DR
    counts, _ = np.histogram(angles, bins=edges)
    centers = edges[:-1] + 0.5 * DR
    density = counts / (angles.size * DR)
    return centers, density


def load_angular_velocities(trajectory: Path) -> np.ndarray:
    """Return (N, 3) angVelX/Y/Z rows from the rb-traj file."""
    rows = []
    with trajectory.open() as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 20 or not fields[1].startswith("point"):
                continue
            rows.append([float(v) for v in fields[-3:]])
    if not rows:
        raise ValueError(f"No point records found in {trajectory}")
    return np.asarray(rows)


def equipartition_kT(trajectory: Path) -> np.ndarray:
    """Return the per-axis kT (kcal/mol) implied by angular-velocity equipartition.

    Port of equipartition.tcl: I <w^2> == kT for each axis at equilibrium.
    """
    w = load_angular_velocities(trajectory)
    return INERTIA * KT_UNIT_FACTOR * np.mean(w ** 2, axis=0)


def boltzmann_estimate(xmin, xmax):
    x = np.linspace(xmin, xmax, 100)
    y = np.exp(-2 * np.cos(x) / KT_KCAL_MOL) * np.sin(x)
    y = y / (np.sum(y) * np.mean(np.diff(x)))
    return x, y


def hsv2rgb(h, s, v):
    hsv = np.reshape([h, s, v], [1, 1, 3])
    return mcolors.hsv_to_rgb(hsv)[0, 0, :]


def hsv2rgb256(h, s, v):
    return hsv2rgb(h / 360, s / 255, v / 255)


# Paul Tol colorblind-safe picks
COLOR_SIMULATED = hsv2rgb256(280, 226, 94)
COLOR_EXPECTED = hsv2rgb256(211, 150, 180)


def get_tick_divisions(xmin, xmax):
    dx = np.abs(xmax - xmin)
    minticksize = dx / 2
    magnitude = 10 ** np.floor(np.log10(minticksize))
    residual = minticksize / magnitude

    if np.abs(np.round(residual * 2) - residual * 2) < 1e-6 and magnitude > 0.01:
        return residual * magnitude
    if residual > 5:
        return 10 * magnitude
    if residual > 2:
        return 5 * magnitude
    if residual > 1:
        return 2 * magnitude
    return magnitude


def set_ticks(axes, dx, dy, num_x_minor=1, num_y_minor=1):
    axes.xaxis.set_major_locator(plticker.MultipleLocator(base=dx))
    axes.xaxis.set_minor_locator(plticker.MultipleLocator(base=dx / (num_x_minor + 1)))
    axes.yaxis.set_major_locator(plticker.MultipleLocator(base=dy))
    axes.yaxis.set_minor_locator(plticker.MultipleLocator(base=dy / (num_y_minor + 1)))


def set_range(axes, minmax):
    xmin, xmax, ymin, ymax = minmax
    axes.axis(minmax)
    set_ticks(axes, get_tick_divisions(xmin, xmax), get_tick_divisions(ymin, ymax))


def make_title(ax, text):
    ax.annotate(text, xy=(0.05, 0.92),
                fontsize=9, xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top')


def plot(ax, x, y, label, color):
    return ax.bar(x, y, label=label, lw=0, width=np.mean(x[1:] - x[:-1]), alpha=0.5, color=color)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="rb-traj file to read orientations from")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output plot path")
    args = parser.parse_args()

    x, y = orientation_density(args.trajectory)
    kT_est = equipartition_kT(args.trajectory)
    info(f"equipartition kT estimate (x,y,z) [kcal/mol]: {kT_est}")

    _, ax1 = plt.subplots(1)
    make_title(ax1, 'Dipole in uniform field')
    ax1.set_ylabel('likelihood')
    ax1.get_yaxis().set_label_coords(-0.085, 0.5)
    ax1.set_xlabel('phi')

    xmin, xmax = 0, np.pi
    set_range(ax1, [xmin, xmax, 0, 1.5])

    ex, ey = boltzmann_estimate(xmin, xmax)

    plot(ax1, x, y, 'simulated', COLOR_SIMULATED)
    plot(ax1, ex, ey, 'expected', COLOR_EXPECTED)

    ax1.annotate(f"equipartition kT $\\approx$ {np.mean(kT_est):.3f} kcal/mol",
                 xy=(0.05, 0.05), fontsize=8, xycoords='axes fraction',
                 horizontalalignment='left', verticalalignment='bottom')

    plt.legend(loc='center left')
    plt.tight_layout()
    plt.savefig(args.output)
    info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
