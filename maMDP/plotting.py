import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import colors
from typing import Tuple, List
from matplotlib.patches import RegularPolygon


def normalise(x):
    """Normalises data to 0-1"""
    return (x - np.nanmin(x)) / np.nanmax(x - np.nanmin(x))


# SQUARE GRID PLOTTING FUNCTIONS


def plot_grids(
    grids: np.ndarray,
    colours: List[str] = None,
    alphas: List[float] = None,
    ax: plt.axes = None,
    *args,
    **kwargs
) -> plt.axes:
    """
    Plots multiple grids

    Args:
        grids (np.ndarray): 3D array of grids. The first dimension represents the grid number, the
        second and third dimensions are the X and Y dimensions of each grid
        ax (plt.axes): Axes to use for plotting.
        colours (list, optional): Colour to use for each grid. Defaults to None.

    """

    if not grids.ndim == 3:
        raise AttributeError(
            "Grids must be supplied as a 3D numpy array, (n_grids, Xdim, Ydim)"
        )

    if ax is None:
        _, ax = plt.subplots(*args, **kwargs)

    # Plot each grid
    for grid in range(grids.shape[0]):
        if colours is None:
            colour = mpl.cm.tab10(grid)
        else:
            colour = colours[grid]

        # Get colours - alpha depends on intensity of feature
        cmap = colors.LinearSegmentedColormap.from_list(
            "singleColour", [colour, colour], N=2
        )  # Single colour colormap
        colour_array = colors.Normalize(0, 1, clip=True)(grids[grid].T)
        colour_array = cmap(colour_array)
        colour_array[..., -1] = grids[grid].T  # Set alpha dependent on intensity

        if alphas is None:
            alpha = 0.5
        else:
            alpha = alphas[grid]

        ax.imshow(colour_array, alpha=alpha)

    ax.grid(which="major", axis="both", linestyle="-", color="#333333", linewidth=1)
    ax.set_xticks(np.arange(-0.5, grids.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, grids.shape[2], 1))
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax


def plot_grid_values(
    grid: np.ndarray, cmap: str = "viridis", ax: plt.axes = None, *args, **kwargs
) -> plt.axes:
    """Plots continuous values on a grid.

    Args:
        grid (np.ndarray): 2D array of values to plot
        cmap (str, optional): Colormap. Defaults to 'viridis'.
        ax (plt.axes): Axes to use for plotting.

    Returns:
        plt.axes: Axes
    """

    if ax is None:
        _, ax = plt.subplots(*args, **kwargs)

    # Plot each grid
    ax.imshow(grid, cmap=cmap)

    ax.grid(which="major", axis="both", linestyle="-", color="#333333", linewidth=1)
    ax.set_xticks(np.arange(-0.5, grid.shape[0], 1))
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax


def plot_agent(ax: plt.axes, position: Tuple, marker: str = "X", *args, **kwargs):

    ax.scatter(*position, marker=marker, *args, **kwargs)


def plot_trajectory(
    ax,
    trajectory: List[Tuple],
    colour: str = "black",
    head_width=0.3,
    head_length=0.3,
    *args,
    **kwargs
):

    ## Iteratively plot arrows.
    for i in range(len(trajectory) - 1):

        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]

        ## Define arrow coordinates.
        x, y = x1, y1
        dx, dy = x2 - x1, y2 - y1

        ## Plot.
        ax.arrow(
            x,
            y,
            dx,
            dy,
            color=colour,
            head_width=head_width,
            head_length=head_length,
            *args,
            **kwargs
        )

    return ax


def plot_sequence(
    ax, trajectory: List[Tuple], colour: str = "black", size: int = 10, *args, **kwargs
):
    """
    Plots a trajectory as a sequence of numbers

    Args:
        ax (plt.axes): Axes on which to plot
        trajectory (List[Tuple]): Trajectory of state indices
        colour (str, optional): Colour of the numbers. Defaults to 'black'.
        size (int, optional): Size of the numbers.

    """

    ## Iteratively plot numbers.
    for i in range(len(trajectory)):

        x, y = trajectory[i]

        ## Plot.
        ax.text(
            x - 0.25, y - 0.25, str(i + 1), color=colour, size=size, *args, **kwargs
        )

    return ax


# HEX GRID PLOTTING FUNCTIONS
def draw_hexagons(
    X,
    outer_radius=1,
    edgecolor="#787878",
    ax=None,
    alpha=None,
    facecolor=None,
    return_coords=False,
    labels=False,
    hide_idx: List[Tuple[int]] = None,
    **kwargs
):

    n_cols = X.shape[0]
    n_rows = X.shape[1]

    if hide_idx is None:
        hide_idx = []

    inner_radius = 0.86602540 * outer_radius

    y_coord = []
    x_coord = []

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(n_cols, n_rows * 0.33), dpi=200)

    ax.set_aspect("equal")

    coords = np.zeros((2, X.shape[0], X.shape[1]))

    # Get coordinates and draw hexagons
    for x in range(n_cols):
        for z in range(n_rows):
            if (x, z) not in hide_idx:
                if x % 2:
                    y_coord.append(z * (inner_radius * 2))
                else:
                    y_coord.append(
                        z * (inner_radius * 2) + ((inner_radius * 2) / 2)
                    )
                x_coord.append(x * outer_radius * 1.5)
                if facecolor is None:
                    cell_colour = X[x, z]
                else:
                    cell_colour = facecolor
                hex = RegularPolygon(
                    (x_coord[-1], y_coord[-1]),
                    numVertices=6,
                    radius=outer_radius,
                    orientation=np.radians(30),
                    alpha=alpha,
                    facecolor=cell_colour,
                    edgecolor=edgecolor,
                    **kwargs
                )
                ax.add_patch(hex)
                if labels:
                    ax.text(
                        x_coord[-1] - inner_radius / 2,
                        y_coord[-1],
                        "{0}, {1}".format(x, z),
                        fontsize=5,
                    )
                coords[0, x, z] = x_coord[-1]
                coords[1, x, z] = y_coord[-1]

    ax.set_xlim(0 - outer_radius, np.max(x_coord) + outer_radius)
    ax.set_ylim(0 - outer_radius, np.max(y_coord) + outer_radius)
    ax.axis("off")

    if return_coords:
        return coords


def plot_hex_grids(
    grids: np.ndarray,
    colours: List[str] = None,
    alphas: List[float] = None,
    ax: plt.axes = None,
    *args,
    **kwargs
):
    """
    Plots multiple grids

    Args:
        grids (np.ndarray): 3D array of grids. The first dimension represents the grid number, the
        second and third dimensions are the X and Y dimensions of each grid
        ax (plt.axes): Axes to use for plotting.
        colours (list, optional): Colour to use for each grid. Defaults to None.

    """

    if not grids.ndim == 3:
        raise AttributeError(
            "Grids must be supplied as a 3D numpy array, (n_grids, Xdim, Ydim)"
        )

    if ax is None:
        f, ax = plt.subplots(*args, **kwargs)

    # Plot each grid
    for grid in range(grids.shape[0]):
        if colours is None:
            colour = mpl.cm.tab10(grid)
        else:
            colour = colours[grid]

        # Get colours - alpha depends on intensity of feature
        cmap = colors.LinearSegmentedColormap.from_list(
            "singleColour", [colour, colour], N=2
        )  # Single colour colormap
        colour_array = colors.Normalize(0, 1, clip=True)(grids[grid])
        colour_array = cmap(colour_array)

        if alphas is None:
            alpha = 0.5
        else:
            alpha = alphas[grid]
        colour_array[..., -1] = (
            colors.Normalize(0, 1, clip=True)(grids[grid]) * alpha
        )  # Set alpha dependent on intensity
        draw_hexagons(colour_array, linewidth=0, ax=ax, return_coords=False)

    coords = draw_hexagons(
        np.ones((grids.shape[1], grids.shape[2])),
        ax=ax,
        facecolor=(0, 0, 0, 0),
        return_coords=True,
    )
    ax.set_xticks(np.arange(-0.5, grids.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, grids.shape[2], 1))
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax, coords


def plot_hex_grid_values(
    grid: np.ndarray,
    cmap: str = "viridis",
    ax=None,
    hide_idx: List[Tuple[int]] = None,
    edgecolor: str = "#787878",
    edgewidth: float = 0.5,
    hex_args=[],
    hex_kwargs={},
    *args,
    **kwargs
) -> plt.axes:
    """Plots continuous values on a hex grid.

    Args:
        grid (np.ndarray): 2D array of values to plot
        cmap (str, optional): Colormap. Defaults to 'viridis'.
        ax (plt.axes, optional): Axes to use for plotting. Defaults to None.
        hide_idx (list, optional): List of tuples of indices to hide. Defaults to None.

    Returns:
        plt.axes: Axes
    """

    if ax is None:
        f, ax = plt.subplots(*args, **kwargs)

    # Get colours
    cmap = plt.get_cmap(cmap)
    colour_array = cmap(normalise(grid))

    # Plot  grid
    draw_hexagons(
        colour_array, linewidth=0, ax=ax, return_coords=False, hide_idx=hide_idx,
        *hex_args, **hex_kwargs
    )

    coords = draw_hexagons(
        np.ones((grid.shape[0], grid.shape[1])),
        ax=ax,
        facecolor=(0, 0, 0, 0),
        return_coords=True,
        hide_idx=hide_idx,
        edgecolor=edgecolor,
        linewidth=edgewidth,
        *hex_args, **hex_kwargs
    )
    ax.set_xticks(np.arange(-0.5, grid.shape[0], 1))
    ax.set_yticks(np.arange(-0.5, grid.shape[1], 1))
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax, coords
