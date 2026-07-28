"""
Visualization helpers for scan trajectories and reconstructed ODS.
"""

import matplotlib.pyplot as plt
import numpy as np

from .demodulate import evaluate_ods

__all__ = ["plot_trajectory", "plot_ods"]


def plot_trajectory(x, y, ax=None, **kwargs):
    """
    Plot the Lissajous scan trajectory in normalized coordinates.

    :param x: normalized x trajectory
    :param y: normalized y trajectory
    :param ax: optional matplotlib axes
    :param kwargs: forwarded to ``ax.plot``
    :return: the matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots()
    kwargs.setdefault("linewidth", 0.5)
    ax.plot(x, y, **kwargs)
    ax.set_xlabel("x [-]")
    ax.set_ylabel("y [-]")
    ax.set_aspect("equal")
    return ax


def plot_ods(coefficients, resolution=200, part="real", ax=None, **kwargs):
    """
    Plot a reconstructed operating deflection shape.

    :param coefficients: complex Chebyshev coefficient matrix
        (see :func:`pycsldv.demodulate.demodulate_ods`)
    :param resolution: grid resolution per direction
    :param part: ``'real'``, ``'imag'`` or ``'abs'`` part of the shape
    :param ax: optional matplotlib axes
    :param kwargs: forwarded to ``ax.pcolormesh``
    :return: the matplotlib axes
    """
    x_grid, y_grid, z = evaluate_ods(coefficients, resolution)
    z = {"real": np.real, "imag": np.imag, "abs": np.abs}[part](z)
    if ax is None:
        _, ax = plt.subplots()
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("shading", "gouraud")
    mesh = ax.pcolormesh(x_grid, y_grid, z, **kwargs)
    ax.figure.colorbar(mesh, ax=ax, label="deflection [-]")
    ax.set_xlabel("x [-]")
    ax.set_ylabel("y [-]")
    ax.set_aspect("equal")
    return ax
