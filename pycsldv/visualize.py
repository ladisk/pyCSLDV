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

    Supports both 1D and 2D Chebyshev coefficient arrays.

    A multimodal or multi-sensor reconstruction -- a list of coefficient
    arrays, or an array stacked along a leading axis -- is drawn as a row of
    panels, one per shape, and an array of axes is returned. Pass ``ax`` only
    for a single shape; several shapes need several axes.

    :param coefficients: complex Chebyshev coefficients. For a 1D scan,
        shape ``(P + 1,)``. For a 2D scan, shape
        ``(Px + 1, Py + 1)``. A list or a stack of either draws one panel
        per shape.
    :param resolution: number of points per direction used for reconstruction
    :param part: ``'real'``, ``'imag'`` or ``'abs'`` part of the shape
    :param ax: optional matplotlib axes, for a single shape
    :param kwargs: forwarded to ``ax.plot`` for 1D or
        ``ax.pcolormesh`` for 2D
    :return: the matplotlib axes, or an array of them for several shapes
    """
    if part not in {"real", "imag", "abs"}:
        raise ValueError("`part` must be 'real', 'imag' or 'abs'.")

    if isinstance(coefficients, (list, tuple)) or np.asarray(coefficients).ndim > 2:
        shapes = list(coefficients)
        if ax is not None:
            raise ValueError(
                f"`ax` takes a single shape, but {len(shapes)} were given; "
                "leave it out and one panel per shape is created.")
        _, axes = plt.subplots(1, len(shapes),
                               figsize=(3.2 * len(shapes), 3.0),
                               squeeze=False)
        for one, shape_axes in zip(shapes, axes[0]):
            plot_ods(one, resolution=resolution, part=part, ax=shape_axes,
                     **kwargs)
        return axes[0]

    coefficients = np.asarray(coefficients)

    if ax is None:
        _, ax = plt.subplots()
    x_grid = np.linspace(-1, 1, resolution)

    if coefficients.ndim == 1:
        x_grid, z = evaluate_ods(coefficients, resolution)
        z = {"real": np.real, "imag": np.imag, "abs": np.abs}[part](z)

        ax.plot(x_grid, z, **kwargs)
        ax.set_xlabel("x [-]")
        ax.set_ylabel("deflection [-]")

    elif coefficients.ndim == 2:
        x_grid, y_grid, z = evaluate_ods(coefficients, resolution)
        z = {"real": np.real, "imag": np.imag, "abs": np.abs}[part](z)

        kwargs.setdefault("cmap", "viridis")
        kwargs.setdefault("shading", "gouraud")
        mesh = ax.pcolormesh(x_grid, y_grid, z, **kwargs)
        ax.figure.colorbar(mesh, ax=ax, label="deflection [-]")
        ax.set_xlabel("x [-]")
        ax.set_ylabel("y [-]")
        ax.set_aspect("equal")

    else:
        raise ValueError(
            "`coefficients` must be a 1D or 2D array."
        )

    return ax
