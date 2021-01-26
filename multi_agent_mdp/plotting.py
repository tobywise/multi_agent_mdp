import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import colors
from typing import Tuple, List

def plot_grids(grids:np.ndarray, colours:List[str]=None, alphas:List[float]=None, *args, **kwargs):
    """
    Plots multiple grids

    Args:
        grids (np.ndarray): 3D array of grids. The first dimension represents the grid number, the
        second and third dimensions are the X and Y dimensions of each grid
        colours (list, optional): Colour to use for each grid. Defaults to None.

    """

    if not grids.ndim == 3:
        raise AttributeError("Grids must be supplied as a 3D numpy array, (n_grids, Xdim, Ydim)")

    f, ax = plt.subplots(*args, **kwargs)

    # Plot each grid
    for grid in range(grids.shape[0]):
        if colours is None:
            colour = mpl.cm.tab10(grid)
        else:
            colour = colours[grid]

        # Get colours - alpha depends on intensity of feature
        cmap = colors.LinearSegmentedColormap.from_list('singleColour', [colour, colour], N=2)  # Single colour colormap
        colour_array = colors.Normalize(0, 1, clip=True)(grids[grid])
        colour_array = cmap(colour_array)
        colour_array[..., -1] = grids[grid]  # Set alpha dependent on intensity

        if alphas is None:
            alpha = 0.5
        else:
            alpha = alphas[grid]

        ax.imshow(colour_array, alpha=alpha)

    ax.grid(which='major', axis='both', linestyle='-', color='#333333', linewidth=1)
    ax.set_xticks(np.arange(-.5, grids.shape[1], 1));
    ax.set_yticks(np.arange(-.5, grids.shape[2], 1));
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax

def plot_grid_values(grid:np.ndarray, cmap:str='viridis', *args, **kwargs) -> plt.axes:
    """Plots continuous values on a grid.

    Args:
        grid (np.ndarray): 2D array of values to plot
        cmap (str, optional): Colormap. Defaults to 'viridis'.

    Returns:
        plt.axes: Axes
    """

    f, ax = plt.subplots(*args, **kwargs)

    # Plot each grid
    ax.imshow(grid, cmap=cmap)

    ax.grid(which='major', axis='both', linestyle='-', color='#333333', linewidth=1)
    ax.set_xticks(np.arange(-.5, grid.shape[0], 1));
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1));
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax


def plot_agent(ax:plt.axes, position:Tuple, marker:str="X", *args, **kwargs):

    ax.scatter(*position, marker=marker, *args, **kwargs)


def plot_trajectory(ax, trajectory:List[Tuple], colour:str='black', 
                    head_width=0.3, head_length=0.3, *args, **kwargs):
                    
    ## Iteratively plot arrows.
    for i in range(len(trajectory)-1):

        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i+1]

        ## Define arrow coordinates.
        x, y = x1, y1
        dx, dy = x2-x1, y2-y1
        
        ## Plot.
        ax.arrow(x, y, dx, dy, color=colour, head_width=head_width, head_length=head_length, *args, **kwargs)

    return ax


# class SquareGridPlottingMixin():

#     def plot(self, colours:list=None, alphas:list=None, *args, **kwargs):

#         ax = plot_grids(self.features_as_grid(), colours=colours, alphas=alphas, *args, **kwargs)

#     def state_to_position(self, state):

#         idx = self.state_to_idx(state)
        
#         return idx
