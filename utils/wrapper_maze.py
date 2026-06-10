"""Maze generation (mazelib) and mapping to a THRML-ready graph.

The maze grid is a (2H+1, 2W+1) numpy array of 0/1 (1 = wall) whose outer
ring is wall except for the start/end openings. Every cell, walls included,
becomes one 3-state CategoricalNode.
"""

from dataclasses import dataclass

import numpy as np
from mazelib import Maze
from mazelib.generate.Prims import Prims
from thrml import Block, CategoricalNode

from utils import config


def generate_maze(width=config.MAZE_WIDTH, height=config.MAZE_HEIGHT, seed=config.SEED):
    """Generate a maze with mazelib. Returns (grid, start, end).

    grid: (2*height+1, 2*width+1) int array, 1 = wall, 0 = corridor.
    start, end: (row, col) cells on the outer ring, opened in the grid.
    """
    m = Maze(seed)
    m.generator = Prims(height, width)
    m.generate()
    m.generate_entrances()
    grid = np.array(m.grid, dtype=int)
    grid[m.start] = 0
    grid[m.end] = 0
    return grid, m.start, m.end


@dataclass
class MazeGraph:
    """The maze as static THRML structure plus the model's parameter arrays."""

    grid: np.ndarray                  # (H, W) of 0/1, 1 = wall
    start: tuple
    end: tuple
    coords: list                      # (row, col) per node, row-major scan order
    nodes: list                       # CategoricalNode per cell, parallel to coords
    coord_to_node: dict
    blocks: list                      # Gibbs update Blocks
    biases: np.ndarray                # (n_nodes, 3)

    def state_to_grid(self, state):
        """Reshape a flat per-node state vector (parallel to nodes) to grid shape."""
        return np.asarray(state).reshape(self.grid.shape)


def _make_blocks(coords, coord_to_node, block_scheme):
    if block_scheme == "checkerboard":
        n_blocks = 2
        color = lambda r, c: (r + c) % 2
    elif block_scheme == "two_hop":
        # A radius-1 degree factor touches N/S/E/W neighbors, which are up to
        # two grid hops apart. This 5-coloring separates every node in that
        # local cross into a distinct Gibbs block.
        n_blocks = 5
        color = lambda r, c: (r + 2 * c) % 5
    else:
        raise ValueError("block_scheme must be 'checkerboard' or 'two_hop'")

    return [
        Block([coord_to_node[(r, c)] for (r, c) in coords if color(r, c) == block])
        for block in range(n_blocks)
    ]


def maze_to_graph(grid, start, end,
                  bias_wall=config.BIAS_WALL,
                  bias_goal=config.BIAS_GOAL,
                  bias_other=config.BIAS_OTHER,
                  block_scheme="two_hop"):
    """Map a maze grid to a MazeGraph ready for THRML sampling."""
    n_rows, n_cols = grid.shape
    coords = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    nodes = [CategoricalNode() for _ in coords]
    coord_to_node = dict(zip(coords, nodes))

    blocks = _make_blocks(coords, coord_to_node, block_scheme)

    # Per-node bias by cell type.
    biases = np.empty((len(coords), config.N_STATES))
    for i, coord in enumerate(coords):
        if coord == start or coord == end:
            biases[i] = bias_goal
        elif grid[coord] == 1:
            biases[i] = bias_wall
        else:
            biases[i] = bias_other

    return MazeGraph(grid=grid, start=start, end=end, coords=coords, nodes=nodes,
                     coord_to_node=coord_to_node, blocks=blocks, biases=biases)
