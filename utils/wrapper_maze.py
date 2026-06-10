"""Maze generation (mazelib) and mapping to a THRML-ready graph.

The maze grid is a (2H+1, 2W+1) numpy array of 0/1 (1 = wall) whose outer
ring is wall except for the start/end openings. Every cell, walls included,
becomes one 3-state CategoricalNode. The 4-neighbor grid graph is bipartite
(checkerboard), giving the two update blocks.
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
    blocks: list                      # the two bipartite Blocks [even, odd]
    edges: np.ndarray                 # (n_edges, 2) node indices, u before v in scan order
    biases: np.ndarray                # (n_nodes, 3), border padding folded in
    edge_weights: np.ndarray          # (3, 3) matrix applied per edge as W[s_u, s_v]

    def state_to_grid(self, state):
        """Reshape a flat per-node state vector (parallel to nodes) to grid shape."""
        return np.asarray(state).reshape(self.grid.shape)


def maze_to_graph(grid, start, end,
                  bias_wall=config.BIAS_WALL,
                  bias_goal=config.BIAS_GOAL,
                  bias_other=config.BIAS_OTHER,
                  edge_weights=config.EDGE_WEIGHTS,
                  block_scheme='checkerboard'):
    """Map a maze grid to a MazeGraph ready for THRML sampling."""
    n_rows, n_cols = grid.shape
    coords = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    nodes = [CategoricalNode() for _ in coords]
    coord_to_node = dict(zip(coords, nodes))
    index = {coord: i for i, coord in enumerate(coords)}

    if block_scheme == 'two_hop':
        # 5-coloring: no two cells within Manhattan distance 2 share a block.
        blocks = [
            Block([coord_to_node[(r, c)] for (r, c) in coords if (r + 2 * c) % 5 == color])
            for color in range(5)
        ]
    else:
        # Checkerboard 2-coloring: no two adjacent cells share a block.
        blocks = [
            Block([coord_to_node[(r, c)] for (r, c) in coords if (r + c) % 2 == parity])
            for parity in (0, 1)
        ]

    # Edges oriented u -> v with u earlier in scan order (right and down neighbors).
    edges = []
    for (r, c) in coords:
        if c + 1 < n_cols:
            edges.append((index[(r, c)], index[(r, c + 1)]))
        if r + 1 < n_rows:
            edges.append((index[(r, c)], index[(r + 1, c)]))
    edges = np.array(edges, dtype=int)

    # Per-node bias by cell type.
    biases = np.empty((len(coords), config.N_STATES))
    for i, coord in enumerate(coords):
        if coord == start or coord == end:
            biases[i] = bias_goal
        elif grid[coord] == 1:
            biases[i] = bias_wall
        else:
            biases[i] = bias_other

    # Border padding: every off-grid neighbor counts as a Wall, folded into the
    # bias with the same scan-order orientation as real edges. A missing right
    # or down neighbor contributes W[s, WALL]; a missing left or up neighbor
    # contributes W[WALL, s].
    edge_weights = np.asarray(edge_weights)
    for i, (r, c) in enumerate(coords):
        n_after = (c + 1 >= n_cols) + (r + 1 >= n_rows)    # node is u, pad is v
        n_before = (c - 1 < 0) + (r - 1 < 0)               # pad is u, node is v
        biases[i] += n_after * edge_weights[:, config.WALL]
        biases[i] += n_before * edge_weights[config.WALL, :]

    return MazeGraph(grid=grid, start=start, end=end, coords=coords, nodes=nodes,
                     coord_to_node=coord_to_node, blocks=blocks, edges=edges,
                     biases=biases, edge_weights=edge_weights)
